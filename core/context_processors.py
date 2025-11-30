from django.db.models import Sum
from decimal import Decimal
from django.utils import timezone
from datetime import timedelta


def get_ticker_messages(user):
    """Generate ticker messages for display"""
    from core.models import SmartAnalysis, Holding, Trade

    # Space before content
    messages = ['','','','','','','','','','','','','','','']
    
    # Welcome message with account age
    account_age = (timezone.now() - user.date_joined).days
    messages.append(f"Welcome {user.username} ({account_age} days old)")
    
    # Get latest SmartAnalysis for this user, or latest overall if none exists
    latest_sa = SmartAnalysis.objects.order_by('-started').first()
    
    if latest_sa:
        # Format date
        now = timezone.now()
        sa_time = latest_sa.started
        time_diff = now - sa_time
        
        if sa_time.date() == now.date():
            time_str = f"today {sa_time.strftime('%H:%M')}"
        elif sa_time.date() == (now - timedelta(days=1)).date():
            time_str = f"yesterday {sa_time.strftime('%H:%M')}"
        else:
            time_str = f"{sa_time.strftime('%A')}* {sa_time.strftime('%H:%M')}"
        
        messages.append(f"Last SmartAnalysis #{latest_sa.id} {time_str}")
        
        # Get BUY recommendations from latest SA
        buy_trades = Trade.objects.filter(
            sa=latest_sa,
            user=user,
            action='BUY'
        ).select_related('stock').order_by('-created')[:13]
        
        if buy_trades.exists():
            messages.append(f"BUYs {buy_trades.count()} stocks")
            for trade in buy_trades:
                # Refresh stock to get current price
                trade.stock.refresh()
                
                # Calculate percentage change from buy price to current price
                if trade.price > 0:
                    pct_change = ((trade.stock.price - trade.price) / trade.price) * 100
                else:
                    pct_change = 0
                
                if pct_change > 0:
                    pct_str = f"+{pct_change:.2f}%"
                    messages.append({'text': f"{trade.stock.symbol}: {trade.shares} shares @ {trade.stock.price:.2f} {pct_str}", 'class': 'positive'})
                elif pct_change < 0:
                    pct_str = f"{pct_change:.2f}%"
                    messages.append({'text': f"{trade.stock.symbol}: {trade.shares} shares @ {trade.stock.price:.2f} {pct_str}", 'class': 'negative'})
                else:
                    messages.append({'text': f"{trade.stock.symbol}: {trade.shares} shares @ {trade.stock.price:.2f} 0.00", 'class': ''})
        
        # Get SELL trades from latest SA
        sell_trades = Trade.objects.filter(
            sa=latest_sa,
            user=user,
            action='SELL'
        ).select_related('stock').order_by('-created')[:10]
        
        if sell_trades.exists():
            messages.append(f"SELLs {sell_trades.count()} stocks")
            for trade in sell_trades:
                # Refresh stock to get current price
                trade.stock.refresh()
                
                # Find the original BUY trade to get the buy price
                buy_trade = Trade.objects.filter(
                    user=user,
                    stock=trade.stock,
                    action='BUY'
                ).order_by('created').first()  # Get the first buy (original purchase)
                
                if buy_trade and buy_trade.price > 0:
                    # Calculate percentage change from buy price to current price
                    pct_change = ((trade.stock.price - buy_trade.price) / buy_trade.price) * 100
                else:
                    # Fallback: compare current price to sell price (if no buy trade found)
                    pct_change = ((trade.stock.price - trade.price) / trade.price) * 100
                
                if pct_change > 0:
                    pct_str = f"+{pct_change:.2f}%"
                    messages.append({'text': f"{trade.stock.symbol}: {trade.shares} shares @ {trade.stock.price:.2f} {pct_str}", 'class': 'positive'})
                elif pct_change < 0:
                    pct_str = f"{pct_change:.2f}%"
                    messages.append({'text': f"{trade.stock.symbol}: {trade.shares} shares @ {trade.stock.price:.2f} {pct_str}", 'class': 'negative'})
                else:
                    messages.append({'text': f"{trade.stock.symbol}: {trade.shares} shares @ {trade.stock.price:.2f} 0.00", 'class': ''})
    
    return messages


def get_portfolio_widget_data(user):
    """Calculate real portfolio widget data"""
    user_profile = user.profile_set.first()
    if not user_profile:
        return {
            'total_value': 0,
            'cash': 0,
            'return_percent': 0,
            'invested': 0,
            'holdings_count': 0,
            'trades_count': 0,
            'shares_count': 0,
        }
    
    # Get holdings data
    holdings = user.holding_set.all()
    holdings_count = holdings.count()
    
    # Calculate total shares
    shares_count = holdings.aggregate(total=Sum('shares'))['total'] or 0
    
    # Calculate current holdings value (live market worth)
    holdings_value = sum(Decimal(str(h.shares)) * h.stock.price for h in holdings)
    invested = holdings_value
    total_value = holdings_value + user_profile.cash
    
    # Calculate return percentage
    if user_profile.investment > 0:
        return_amount = total_value - user_profile.investment
        return_percent = ((total_value - user_profile.investment) / user_profile.investment) * 100
    else:
        return_amount = Decimal('0.0')
        return_percent = Decimal('0.0')
    
    # Get trade count
    trades_count = user.trade_set.count()
    
    return {
        'total_value': float(total_value),
        'cash': float(user_profile.cash),
        'return_amount': float(return_amount),
        'invested': float(invested),
        'holdings_count': holdings_count,
        'trades_count': trades_count,
        'shares_count': shares_count,
        'return_percent': float(return_percent),
    }


def portfolio_widget(request):
    """Add portfolio widget data to all templates"""
    if request.user.is_authenticated:
        widget_data = get_portfolio_widget_data(request.user)
        widget_data['ticker_messages'] = get_ticker_messages(request.user)
        return {
            'portfolio_widget': widget_data
        }
    return {'portfolio_widget': {}}

