"""
Analyze Insider Purchase Performance

Usage:
    python manage.py analyze_insider_performance [--date YYYY-MM-DD]
"""
import yfinance as yf
import re
from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import date, timedelta, datetime
from decimal import Decimal

from core.models import Discovery, Trade, Stock, Advisor, SmartAnalysis, Consensus
from django.db.models import Q


class Command(BaseCommand):
    help = 'Analyze performance of insider purchases from a specific date'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--date',
            type=str,
            help='Date to analyze (YYYY-MM-DD). Defaults to yesterday.',
        )
        parser.add_argument(
            '--days',
            type=int,
            default=1,
            help='Number of days to look back (default: 1)',
        )
    
    def handle(self, *args, **options):
        # Get Insider advisor
        try:
            insider_advisor = Advisor.objects.get(name='Insider')
        except Advisor.DoesNotExist:
            self.stdout.write(self.style.ERROR('Insider advisor not found'))
            return
        
        # Determine date range
        if options['date']:
            try:
                target_date = datetime.strptime(options['date'], '%Y-%m-%d').date()
            except ValueError:
                self.stdout.write(self.style.ERROR('Invalid date format. Use YYYY-MM-DD'))
                return
        else:
            target_date = date.today() - timedelta(days=1)
        
        start_date = target_date - timedelta(days=options['days'] - 1)
        
        self.stdout.write(f'\nAnalyzing Insider purchases from {start_date} to {target_date}')
        self.stdout.write('=' * 80)
        
        # Find SmartAnalysis sessions from the date range
        sa_sessions = SmartAnalysis.objects.filter(
            started__date__gte=start_date,
            started__date__lte=target_date
        ).order_by('started')
        
        if not sa_sessions.exists():
            self.stdout.write(self.style.WARNING(f'No SmartAnalysis sessions found for {start_date} to {target_date}'))
            return
        
        # Get all discoveries from Insider advisor in these sessions
        discoveries = Discovery.objects.filter(
            advisor=insider_advisor,
            sa__in=sa_sessions
        ).select_related('stock', 'sa').order_by('sa__started', 'stock__symbol')
        
        if not discoveries.exists():
            self.stdout.write(self.style.WARNING('No Insider discoveries found for this period'))
            return
        
        self.stdout.write(f'\nFound {discoveries.count()} Insider discoveries')
        
        # Get all BUY trades for these stocks in these sessions
        discovered_stocks = discoveries.values_list('stock', flat=True).distinct()
        trades = Trade.objects.filter(
            stock__in=discovered_stocks,
            sa__in=sa_sessions,
            action='BUY'
        ).select_related('stock', 'user', 'sa').order_by('sa__started', 'stock__symbol')
        
        self.stdout.write(f'Found {trades.count()} BUY trades\n')
        
        # Group by stock
        stock_performance = {}
        
        # Get consensus scores for discovered stocks
        discovered_stock_ids = discoveries.values_list('stock', flat=True).distinct()
        consensus_scores = {}
        for sa in sa_sessions:
            consensus_records = Consensus.objects.filter(
                sa=sa,
                stock_id__in=discovered_stock_ids
            ).select_related('stock')
            for consensus in consensus_records:
                symbol = consensus.stock.symbol
                if symbol not in consensus_scores:
                    consensus_scores[symbol] = []
                consensus_scores[symbol].append(float(consensus.avg_confidence))
        
        for discovery in discoveries:
            symbol = discovery.stock.symbol
            if symbol not in stock_performance:
                stock_performance[symbol] = {
                    'discoveries': [],
                    'trades': [],
                    'current_price': None,
                    'discovery_price': None,
                    'purchase_price': None,
                    'explanation': None,
                    'consensus_score': None,
                    'discovery_score': None,
                    'peak_price': None,
                    'peak_date': None,
                    'discovery_date': None,
                }
            
            stock_performance[symbol]['discoveries'].append(discovery)
            if not stock_performance[symbol]['discovery_price']:
                # Try to get price from stock model
                stock_performance[symbol]['discovery_price'] = discovery.stock.price
            if not stock_performance[symbol]['discovery_date']:
                # Store the earliest discovery date
                stock_performance[symbol]['discovery_date'] = discovery.created.date()
            stock_performance[symbol]['explanation'] = discovery.explanation
            
            # Parse Insider discovery score from explanation
            # Format: "Insider purchase: X purchase(s), total score Y.YY | ..."
            explanation = discovery.explanation or ""
            discovery_score_match = re.search(r'total score\s+([\d.]+)', explanation, re.IGNORECASE)
            if discovery_score_match:
                discovery_score = float(discovery_score_match.group(1))
                if stock_performance[symbol].get('discovery_score') is None:
                    stock_performance[symbol]['discovery_score'] = discovery_score
                else:
                    # Use max if multiple discoveries
                    stock_performance[symbol]['discovery_score'] = max(
                        stock_performance[symbol]['discovery_score'], 
                        discovery_score
                    )
            
            # Get consensus score (use max if multiple)
            if symbol in consensus_scores:
                stock_performance[symbol]['consensus_score'] = max(consensus_scores[symbol])
        
        for trade in trades:
            symbol = trade.stock.symbol
            if symbol in stock_performance:
                stock_performance[symbol]['trades'].append(trade)
                if not stock_performance[symbol]['purchase_price']:
                    stock_performance[symbol]['purchase_price'] = trade.price
        
        # Fetch current prices
        symbols = list(stock_performance.keys())
        self.stdout.write(f'Fetching current prices for {len(symbols)} stocks...')
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Try to get current/latest price (prefer fast_info for speed, fallback to info)
                try:
                    fast_info = ticker.fast_info
                    current_price = fast_info.get('lastPrice') or fast_info.get('regularMarketPrice')
                except:
                    current_price = None
                
                # Fallback to info if fast_info didn't work
                if not current_price:
                    try:
                        info = ticker.info
                        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
                    except:
                        current_price = None
                
                # Last resort: use history (yesterday's close)
                if not current_price:
                    try:
                        hist = ticker.history(period='1d')
                        if not hist.empty:
                            current_price = float(hist['Close'].iloc[-1])
                    except:
                        current_price = None
                
                if current_price:
                    stock_performance[symbol]['current_price'] = float(current_price)
                else:
                    # Try to get current price from stock model
                    stock = Stock.objects.get(symbol=symbol)
                    if stock.price:
                        stock_performance[symbol]['current_price'] = float(stock.price)
                
                # Fetch historical prices to find peak
                discovery_date = stock_performance[symbol].get('discovery_date')
                if discovery_date:
                    try:
                        # Get history from discovery date to now
                        hist = ticker.history(start=discovery_date, end=date.today() + timedelta(days=1))
                        if not hist.empty and 'Close' in hist.columns:
                            # Find peak price and date
                            peak_idx = hist['Close'].idxmax()
                            peak_price = float(hist.loc[peak_idx, 'Close'])
                            # Handle pandas Timestamp
                            if hasattr(peak_idx, 'date'):
                                peak_date = peak_idx.date()
                            elif hasattr(peak_idx, 'to_pydatetime'):
                                peak_date = peak_idx.to_pydatetime().date()
                            else:
                                peak_date = peak_idx
                            stock_performance[symbol]['peak_price'] = peak_price
                            stock_performance[symbol]['peak_date'] = peak_date
                    except Exception as e:
                        # Silently fail for historical data - not critical
                        pass
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'Could not fetch price for {symbol}: {e}'))
        
        # Display results
        self.stdout.write('\n' + '=' * 80)
        self.stdout.write('PERFORMANCE ANALYSIS')
        self.stdout.write('=' * 80 + '\n')
        
        total_gain_loss = Decimal('0.00')
        total_invested = Decimal('0.00')
        
        for symbol in sorted(stock_performance.keys()):
            data = stock_performance[symbol]
            current = data['current_price']
            purchase = data['purchase_price']
            discovery = data['discovery_price']
            
            if not current or not purchase:
                continue
            
            # Convert to float for calculations
            current_f = float(current)
            purchase_f = float(purchase)
            discovery_f = float(discovery) if discovery else None
            
            # Calculate gain/loss
            gain_pct = ((current_f - purchase_f) / purchase_f) * 100
            gain_amt = current_f - purchase_f
            
            # Calculate total invested and gain/loss
            total_shares = sum(t.shares for t in data['trades'])
            total_invested += Decimal(str(purchase_f * total_shares))
            total_gain_loss += Decimal(str(gain_amt * total_shares))
            
            # Color coding
            if gain_pct > 0:
                style = self.style.SUCCESS
                gain_str = f'+{gain_pct:.2f}%'
            else:
                style = self.style.ERROR
                gain_str = f'{gain_pct:.2f}%'
            
            consensus = data.get('consensus_score')
            discovery_score = data.get('discovery_score')
            consensus_str = f'CS: {consensus:.2f}' if consensus else 'CS: N/A'
            discovery_str = f'DS: {discovery_score:.2f}' if discovery_score is not None else 'DS: N/A'
            
            self.stdout.write(style(f'\n{symbol} ({consensus_str}, {discovery_str})'))
            self.stdout.write(f'  Discovery Price: ${discovery_f:.2f}' if discovery_f else '  Discovery Price: N/A')
            self.stdout.write(f'  Purchase Price:  ${purchase_f:.2f}')
            self.stdout.write(f'  Current Price:   ${current_f:.2f}')
            
            # Show peak price if available
            peak_price = data.get('peak_price')
            peak_date = data.get('peak_date')
            if peak_price and discovery_f:
                peak_gain_pct = ((peak_price - discovery_f) / discovery_f) * 100
                peak_gain_amt = peak_price - discovery_f
                if peak_price > current_f:
                    peak_style = self.style.SUCCESS
                    peak_str = f'  Peak Price:     ${peak_price:.2f} on {peak_date} (+{peak_gain_pct:.2f}% from discovery, +${peak_gain_amt:.2f})'
                    self.stdout.write(peak_style(peak_str))
                    # Show opportunity cost if current is below peak
                    opportunity_loss_pct = ((peak_price - current_f) / current_f) * 100
                    opportunity_loss_amt = peak_price - current_f
                    self.stdout.write(f'  Opportunity:    Could have sold at peak for {opportunity_loss_pct:.2f}% more (${opportunity_loss_amt:.2f} per share)')
            
            self.stdout.write(f'  Gain/Loss:       {gain_str} (${gain_amt:.2f} per share)')
            self.stdout.write(f'  Total Shares:    {total_shares}')
            self.stdout.write(f'  Total Value:     ${current_f * total_shares:.2f}')
            
            # Show discovery details
            if data['explanation']:
                self.stdout.write(f'  Discovery Info:  {data["explanation"][:100]}...')
            
            # Show trade details
            if data['trades']:
                self.stdout.write(f'  Trades:')
                for trade in data['trades']:
                    self.stdout.write(f'    - {trade.user.username}: {trade.shares} shares @ ${trade.price:.2f} (SA #{trade.sa.id})')
        
        # Summary
        self.stdout.write('\n' + '=' * 80)
        self.stdout.write('SUMMARY')
        self.stdout.write('=' * 80)
        self.stdout.write(f'Total Stocks Analyzed: {len([s for s in stock_performance.values() if s["current_price"] and s["purchase_price"]])}')
        self.stdout.write(f'Total Invested: ${total_invested:.2f}')
        
        if total_gain_loss > 0:
            self.stdout.write(self.style.SUCCESS(f'Total Gain/Loss: +${total_gain_loss:.2f}'))
            self.stdout.write(self.style.SUCCESS(f'Total Return: +{(total_gain_loss / total_invested * 100):.2f}%'))
        else:
            self.stdout.write(self.style.ERROR(f'Total Gain/Loss: ${total_gain_loss:.2f}'))
            self.stdout.write(self.style.ERROR(f'Total Return: {(total_gain_loss / total_invested * 100):.2f}%'))
        
        # Calculate performance if we only discovered stocks with CS >= 0.55
        cs_threshold = Decimal('0.55')
        filtered_stocks_cs = {
            s: d for s, d in stock_performance.items()
            if d.get('consensus_score') is not None 
            and Decimal(str(d['consensus_score'])) >= cs_threshold
            and d['current_price'] and d['purchase_price']
        }
        
        if filtered_stocks_cs:
            filtered_invested = Decimal('0.00')
            filtered_gain_loss = Decimal('0.00')
            
            for symbol, data in filtered_stocks_cs.items():
                current_f = float(data['current_price'])
                purchase_f = float(data['purchase_price'])
                total_shares = sum(t.shares for t in data['trades'])
                filtered_invested += Decimal(str(purchase_f * total_shares))
                gain_amt = current_f - purchase_f
                filtered_gain_loss += Decimal(str(gain_amt * total_shares))
            
            self.stdout.write('\n' + '=' * 80)
            self.stdout.write(f'PERFORMANCE IF CS >= {cs_threshold} ONLY')
            self.stdout.write('=' * 80)
            self.stdout.write(f'Stocks with CS >= {cs_threshold}: {len(filtered_stocks_cs)}')
            self.stdout.write(f'Total Invested: ${filtered_invested:.2f}')
            
            if filtered_gain_loss > 0:
                self.stdout.write(self.style.SUCCESS(f'Total Gain/Loss: +${filtered_gain_loss:.2f}'))
                if filtered_invested > 0:
                    self.stdout.write(self.style.SUCCESS(f'Total Return: +{(filtered_gain_loss / filtered_invested * 100):.2f}%'))
            else:
                self.stdout.write(self.style.ERROR(f'Total Gain/Loss: ${filtered_gain_loss:.2f}'))
                if filtered_invested > 0:
                    self.stdout.write(self.style.ERROR(f'Total Return: {(filtered_gain_loss / filtered_invested * 100):.2f}%'))
            
            # Show which stocks would be included
            self.stdout.write(f'\nIncluded stocks: {", ".join(sorted(filtered_stocks_cs.keys()))}')
        
        # Calculate performance if we only discovered stocks with Discovery Score >= 0.55
        ds_threshold = Decimal('0.55')
        filtered_stocks_ds = {
            s: d for s, d in stock_performance.items()
            if d.get('discovery_score') is not None 
            and Decimal(str(d['discovery_score'])) >= ds_threshold
            and d['current_price'] and d['purchase_price']
        }
        
        if filtered_stocks_ds:
            filtered_invested = Decimal('0.00')
            filtered_gain_loss = Decimal('0.00')
            
            for symbol, data in filtered_stocks_ds.items():
                current_f = float(data['current_price'])
                purchase_f = float(data['purchase_price'])
                total_shares = sum(t.shares for t in data['trades'])
                filtered_invested += Decimal(str(purchase_f * total_shares))
                gain_amt = current_f - purchase_f
                filtered_gain_loss += Decimal(str(gain_amt * total_shares))
            
            self.stdout.write('\n' + '=' * 80)
            self.stdout.write(f'PERFORMANCE IF DISCOVERY SCORE >= {ds_threshold} ONLY')
            self.stdout.write('=' * 80)
            self.stdout.write(f'Stocks with Discovery Score >= {ds_threshold}: {len(filtered_stocks_ds)}')
            self.stdout.write(f'Total Invested: ${filtered_invested:.2f}')
            
            if filtered_gain_loss > 0:
                self.stdout.write(self.style.SUCCESS(f'Total Gain/Loss: +${filtered_gain_loss:.2f}'))
                if filtered_invested > 0:
                    self.stdout.write(self.style.SUCCESS(f'Total Return: +{(filtered_gain_loss / filtered_invested * 100):.2f}%'))
            else:
                self.stdout.write(self.style.ERROR(f'Total Gain/Loss: ${filtered_gain_loss:.2f}'))
                if filtered_invested > 0:
                    self.stdout.write(self.style.ERROR(f'Total Return: {(filtered_gain_loss / filtered_invested * 100):.2f}%'))
            
            # Show which stocks would be included
            self.stdout.write(f'\nIncluded stocks: {", ".join(sorted(filtered_stocks_ds.keys()))}')
        
        # Analyze what went wrong
        losers = [s for s, d in stock_performance.items() 
                 if d['current_price'] and d['purchase_price'] 
                 and d['current_price'] < d['purchase_price']]
        
        if losers:
            self.stdout.write('\n' + '=' * 80)
            self.stdout.write('UNDERPERFORMING STOCKS ANALYSIS')
            self.stdout.write('=' * 80)
            
            for symbol in losers:
                data = stock_performance[symbol]
                current_f = float(data['current_price'])
                purchase_f = float(data['purchase_price'])
                loss_pct = ((current_f - purchase_f) / purchase_f) * 100
                consensus = data.get('consensus_score')
                discovery_score = data.get('discovery_score')
                consensus_str = f' (CS: {consensus:.2f})' if consensus else ''
                discovery_str = f' (DS: {discovery_score:.2f})' if discovery_score is not None else ''
                self.stdout.write(f'\n{symbol}{consensus_str}{discovery_str}: {loss_pct:.2f}% loss')
                if data['explanation']:
                    self.stdout.write(f'  Discovery: {data["explanation"]}')

