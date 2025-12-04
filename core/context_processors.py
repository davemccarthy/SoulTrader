from django.db.models import Sum
from decimal import Decimal
from django.utils import timezone
from datetime import timedelta


def get_ticker_messages(user):
    """Generate ticker messages for display"""
    from core.models import Trade

    # Space before content
    messages = ['','','','','','','','','','','','','','','']
    
    # Welcome message with account age and risk level
    account_age = (timezone.now() - user.date_joined).days
    
    # Get user's risk level
    user_profile = user.profile_set.first()
    risk_level = user_profile.risk if user_profile else 'MODERATE'
    risk_display = risk_level.capitalize()  # Convert "MODERATE" to "Moderate"
    
    messages.append(f"Welcome {user.username} ({risk_display} - {account_age} days old)")
    
    # Get current day trades instead of last SmartAnalysis
    now = timezone.now()
    today = now.date()
    
    # Get BUY trades from today
    buy_trades = Trade.objects.filter(
        user=user,
        action='BUY',
        created__date=today
    ).select_related('stock').order_by('-created')[:13]
    
    # Get SELL trades from today
    sell_trades = Trade.objects.filter(
        user=user,
        action='SELL',
        created__date=today
    ).select_related('stock').order_by('-created')[:10]
    
    # Show today's trading activity message
    total_trades_today = buy_trades.count() + sell_trades.count()
    if total_trades_today > 0:
        messages.append(f"Today's Trading Activity ({total_trades_today} trades)")
    else:
        messages.append("Today's Trading Activity (no trades yet)")
    
    if buy_trades.exists():
        messages.append(f"BUYs {buy_trades.count()} stocks")
        for trade in buy_trades:
            
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
    
    if sell_trades.exists():
        messages.append(f"SELLs {sell_trades.count()} stocks")
        for trade in sell_trades:
            
            # Use stored cost basis if available (more accurate than finding first buy)
            if trade.cost and trade.cost > 0:
                # Calculate percentage change from cost basis to sell price (realized P&L)
                pct_change = ((trade.price - trade.cost) / trade.cost) * 100
            else:
                # Fallback: Find the original BUY trade to get the buy price
                buy_trade = Trade.objects.filter(
                    user=user,
                    stock=trade.stock,
                    action='BUY'
                ).order_by('created').first()  # Get the first buy (original purchase)
                
                if buy_trade and buy_trade.price > 0:
                    # Calculate percentage change from buy price to sell price
                    pct_change = ((trade.price - buy_trade.price) / buy_trade.price) * 100
                else:
                    # Fallback: compare sell price to itself (no buy trade found)
                    pct_change = 0
            
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

    # Calculate Trade P&L (realized P&L from SELL trades with cost basis)
    sell_trades = user.trade_set.filter(action='SELL', cost__isnull=False)
    trade_pnl = sum(
        (Decimal(str(trade.price)) - Decimal(str(trade.cost))) * Decimal(trade.shares)
        for trade in sell_trades
        if trade.cost
    )

    # Calculate total cost basis for SELL trades (for percentage calculation)
    trade_cost_basis = sum(
        Decimal(str(trade.cost)) * Decimal(trade.shares)
        for trade in sell_trades
        if trade.cost
    )

    # Calculate Trade P&L as percentage
    if trade_cost_basis > 0:
        trade_pnl_percent = (trade_pnl / trade_cost_basis) * 100
    else:
        trade_pnl_percent = Decimal('0.0')

    # Total cost basis using average_price
    holdings_cost_basis = sum(
        (h.average_price or Decimal('0')) * Decimal(str(h.shares))
        for h in holdings
    )

    # Portfolio P&L
    holdings_pnl = holdings_value - holdings_cost_basis

    # Portfolio P&L percentage
    if holdings_cost_basis > 0:
        holdings_pnl = (holdings_pnl / holdings_cost_basis) * 100
    else:
        holdings_pnl = Decimal('0.0')
    
    return {
        'total_value': float(total_value),
        'cash': float(user_profile.cash),
        'return_amount': float(return_amount),
        'invested': float(invested),
        'holdings_count': holdings_count,
        'trades_count': trades_count,
        'trade_pnl': float(trade_pnl_percent),  # Trade P&L as percentage
        'holdings_pnl': holdings_pnl,
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

