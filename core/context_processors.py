from django.db.models import Sum
from decimal import Decimal


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
        return {
            'portfolio_widget': get_portfolio_widget_data(request.user)
        }
    return {'portfolio_widget': {}}

