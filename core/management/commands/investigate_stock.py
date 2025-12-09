"""
Investigate Stock Success Story

Usage:
    python manage.py investigate_stock <symbol> <username>
    python manage.py investigate_stock ANVS user10
"""

from django.core.management.base import BaseCommand, CommandError
from core.models import (
    User, Stock, Holding, Discovery, Recommendation, Consensus, 
    Trade, SmartAnalysis, SellInstruction
)
from decimal import Decimal
from django.utils import timezone


class Command(BaseCommand):
    help = 'Investigate a successful stock pick - shows discovery, analysis, and performance'

    def add_arguments(self, parser):
        parser.add_argument('symbol', type=str, help='Stock symbol to investigate')
        parser.add_argument('username', type=str, help='Username to investigate for')

    def handle(self, *args, **options):
        symbol = options['symbol'].upper()
        username = options['username']

        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            raise CommandError(f'User "{username}" does not exist')

        try:
            stock = Stock.objects.get(symbol=symbol)
        except Stock.DoesNotExist:
            raise CommandError(f'Stock "{symbol}" does not exist')

        # Get current holding
        holding = Holding.objects.filter(user=user, stock=stock).first()

        if not holding:
            self.stdout.write(self.style.WARNING(f'User {username} does not currently hold {symbol}'))
            return

        # Calculate current P&L
        current_price = stock.price or Decimal('0')
        avg_price = holding.average_price or Decimal('0')
        shares = holding.shares or 0
        pl_amount = (current_price - avg_price) * shares
        pl_percent = ((current_price / avg_price) * 100 - 100) if avg_price > 0 else Decimal('0')

        self.stdout.write(self.style.SUCCESS('\n' + '='*80))
        self.stdout.write(self.style.SUCCESS(f'INVESTIGATING: {symbol} for {username}'))
        self.stdout.write(self.style.SUCCESS('='*80 + '\n'))

        # Current Holding Info
        self.stdout.write(self.style.SUCCESS('📊 CURRENT HOLDING'))
        self.stdout.write('-' * 80)
        self.stdout.write(f'Shares: {shares}')
        self.stdout.write(f'Average Price: ${avg_price:.2f}')
        self.stdout.write(f'Current Price: ${current_price:.2f}')
        self.stdout.write(f'Total Value: ${current_price * shares:.2f}')
        self.stdout.write(f'P&L Amount: ${pl_amount:+.2f}')
        self.stdout.write(f'P&L Percentage: {pl_percent:+.2f}%')
        self.stdout.write(f'Current Consensus: {holding.consensus:.2f}')
        self.stdout.write('')

        # Get all trades for this stock/user
        trades = Trade.objects.filter(user=user, stock=stock).order_by('sa__started', 'id')
        
        if trades.exists():
            self.stdout.write(self.style.SUCCESS('💰 TRADE HISTORY'))
            self.stdout.write('-' * 80)
            for trade in trades:
                sa = trade.sa
                self.stdout.write(f'{trade.action}: {trade.shares} @ ${trade.price:.2f} (SA #{sa.id}, {sa.started.strftime("%Y-%m-%d %H:%M")})')
                if trade.explanation:
                    self.stdout.write(f'  Explanation: {trade.explanation}')
            self.stdout.write('')

        # Get discovery information
        discoveries = Discovery.objects.filter(stock=stock).order_by('sa__started')
        
        if discoveries.exists():
            # Get the discovery that led to the first buy
            first_trade = trades.filter(action='BUY').first()
            discovery = None
            
            if first_trade:
                # Try to find discovery in the same SA as first buy
                discovery = discoveries.filter(sa=first_trade.sa).first()
            
            # If not found, get the earliest discovery
            if not discovery:
                discovery = discoveries.first()

            if discovery:
                self.stdout.write(self.style.SUCCESS('🔍 DISCOVERY'))
                self.stdout.write('-' * 80)
                self.stdout.write(f'Advisor: {discovery.advisor.name}')
                self.stdout.write(f'SA Session: #{discovery.sa.id}')
                self.stdout.write(f'Discovered: {discovery.created.strftime("%Y-%m-%d %H:%M:%S")}')
                self.stdout.write(f'Explanation: {discovery.explanation}')
                self.stdout.write('')

                # Get sell instructions
                instructions = SellInstruction.objects.filter(discovery=discovery).order_by('id')
                if instructions.exists():
                    self.stdout.write(self.style.SUCCESS('📋 SELL INSTRUCTIONS'))
                    self.stdout.write('-' * 80)
                    for inst in instructions:
                        value_str = f'${inst.value:.2f}' if inst.value else 'N/A'
                        self.stdout.write(f'{inst.instruction}: {value_str}')
                    self.stdout.write('')

        # Get consensus and recommendations from the discovery SA
        if discovery:
            consensus = Consensus.objects.filter(sa=discovery.sa, stock=stock).first()
            
            if consensus:
                self.stdout.write(self.style.SUCCESS('🎯 CONSENSUS (at discovery)'))
                self.stdout.write('-' * 80)
                self.stdout.write(f'Recommendations: {consensus.recommendations}')
                self.stdout.write(f'Average Confidence: {consensus.avg_confidence:.2f}')
                self.stdout.write(f'Total Confidence: {consensus.tot_confidence:.2f}')
                self.stdout.write('')

                # Get individual recommendations
                recommendations = Recommendation.objects.filter(
                    sa=discovery.sa, 
                    stock=stock
                ).select_related('advisor').order_by('-confidence')

                if recommendations.exists():
                    self.stdout.write(self.style.SUCCESS('📈 ADVISOR RECOMMENDATIONS'))
                    self.stdout.write('-' * 80)
                    for rec in recommendations:
                        self.stdout.write(f'{rec.advisor.name}: {rec.confidence:.2f} confidence')
                        if rec.explanation:
                            self.stdout.write(f'  {rec.explanation[:100]}...' if len(rec.explanation) > 100 else f'  {rec.explanation}')
                    self.stdout.write('')

        # Price movement analysis
        if discovery and first_trade:
            buy_price = first_trade.price
            days_held = (timezone.now() - first_trade.sa.started).days
            
            self.stdout.write(self.style.SUCCESS('📅 TIMELINE'))
            self.stdout.write('-' * 80)
            self.stdout.write(f'Discovery: {discovery.created.strftime("%Y-%m-%d %H:%M")}')
            self.stdout.write(f'First Buy: {first_trade.sa.started.strftime("%Y-%m-%d %H:%M")} @ ${buy_price:.2f}')
            self.stdout.write(f'Days Held: {days_held} days')
            self.stdout.write(f'Current Price: ${current_price:.2f}')
            if buy_price > 0:
                price_change = ((current_price / buy_price) * 100 - 100)
                self.stdout.write(f'Price Change from Buy: {price_change:+.2f}%')
            self.stdout.write('')

        # Summary
        self.stdout.write(self.style.SUCCESS('✅ SUMMARY'))
        self.stdout.write('-' * 80)
        self.stdout.write(f'Stock: {symbol} ({stock.company or "N/A"})')
        self.stdout.write(f'User: {username}')
        self.stdout.write(f'Current P&L: {pl_percent:+.2f}% (${pl_amount:+.2f})')
        if discovery:
            self.stdout.write(f'Discovered by: {discovery.advisor.name}')
        if consensus:
            self.stdout.write(f'Initial Consensus: {consensus.avg_confidence:.2f} avg, {consensus.tot_confidence:.2f} total')
        self.stdout.write('')


































