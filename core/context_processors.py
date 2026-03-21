from django.db.models import Sum
from decimal import Decimal
from django.utils import timezone
from datetime import timedelta

from core.fund_session import get_current_fund
from core.models import Holding, Trade


def get_ticker_messages(request):
    """Generate ticker messages for display (scoped to session fund)."""
    user = request.user
    fund = get_current_fund(request)

    # Space before content
    messages = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

    # Welcome message with account age and risk level
    account_age = (timezone.now() - user.date_joined).days
    risk_level = fund.risk if fund else 'MODERATE'
    risk_display = risk_level.capitalize()

    messages.append(f"Welcome {user.username} ({risk_display} - {account_age} days old)")

    now = timezone.now()
    today = now.date()

    if fund is None:
        buy_trades = Trade.objects.none()
        sell_trades = Trade.objects.none()
    else:
        buy_trades = (
            Trade.objects.filter(
                fund=fund,
                action='BUY',
                created__date=today,
            )
            .select_related('stock')
            .order_by('-created')[:13]
        )
        sell_trades = (
            Trade.objects.filter(
                fund=fund,
                action='SELL',
                created__date=today,
            )
            .select_related('stock')
            .order_by('-created')[:10]
        )

    total_trades_today = buy_trades.count() + sell_trades.count()
    if total_trades_today > 0:
        messages.append(f"Today's Trading Activity ({total_trades_today} trades)")
    else:
        messages.append("Today's Trading Activity (no trades yet)")

    if buy_trades.exists():
        messages.append(f"BUYs {buy_trades.count()} stocks")
        for trade in buy_trades:
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
            if trade.cost and trade.cost > 0:
                pct_change = ((trade.price - trade.cost) / trade.cost) * 100
            else:
                buy_trade = None
                if fund is not None:
                    buy_trade = (
                        Trade.objects.filter(
                            fund=fund,
                            stock=trade.stock,
                            action='BUY',
                        )
                        .order_by('created')
                        .first()
                    )

                if buy_trade and buy_trade.price > 0:
                    pct_change = ((trade.price - buy_trade.price) / buy_trade.price) * 100
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

    return messages


def get_portfolio_widget_data(request):
    """Calculate portfolio widget data for the session-scoped fund."""
    fund = get_current_fund(request)
    if not fund:
        return {
            'total_value': 0,
            'cash': 0,
            'return_percent': 0,
            'invested': 0,
            'holdings_count': 0,
            'trades_count': 0,
            'shares_count': 0,
            'trade_pnl': 0,
            'holdings_pnl': 0,
            'return_amount': 0,
        }

    holdings = Holding.objects.filter(fund=fund)
    holdings_count = holdings.count()
    shares_count = holdings.aggregate(total=Sum('shares'))['total'] or 0

    for h in holdings:
        pass
        # h.stock.refresh()

    holdings_value = sum(Decimal(str(h.shares)) * h.stock.price for h in holdings)
    invested = holdings_value
    total_value = holdings_value + fund.cash

    if fund.investment > 0:
        return_amount = total_value - fund.investment
        return_percent = ((total_value - fund.investment) / fund.investment) * 100
    else:
        return_amount = Decimal('0.0')
        return_percent = Decimal('0.0')

    trades_count = Trade.objects.filter(fund=fund).count()

    sell_trades = Trade.objects.filter(fund=fund, action='SELL', cost__isnull=False)
    trade_pnl = sum(
        (Decimal(str(trade.price)) - Decimal(str(trade.cost))) * Decimal(trade.shares)
        for trade in sell_trades
        if trade.cost
    )

    trade_cost_basis = sum(
        Decimal(str(trade.cost)) * Decimal(trade.shares)
        for trade in sell_trades
        if trade.cost
    )

    if trade_cost_basis > 0:
        trade_pnl_percent = (trade_pnl / trade_cost_basis) * 100
    else:
        trade_pnl_percent = Decimal('0.0')

    holdings_cost_basis = sum(
        (h.average_price or Decimal('0')) * Decimal(str(h.shares))
        for h in holdings
    )

    holdings_pnl_raw = holdings_value - holdings_cost_basis

    if holdings_cost_basis > 0:
        holdings_pnl_pct = (holdings_pnl_raw / holdings_cost_basis) * 100
    else:
        holdings_pnl_pct = Decimal('0.0')

    return {
        'total_value': float(total_value),
        'cash': float(fund.cash),
        'return_amount': float(return_amount),
        'invested': float(invested),
        'holdings_count': holdings_count,
        'trades_count': trades_count,
        'trade_pnl': float(trade_pnl_percent),
        'holdings_pnl': holdings_pnl_pct,
        'return_percent': float(return_percent),
    }


def portfolio_widget(request):
    """Add portfolio widget data and session-scoped fund to all templates."""
    if request.user.is_authenticated:
        current_fund = get_current_fund(request)
        widget_data = get_portfolio_widget_data(request)
        widget_data['ticker_messages'] = get_ticker_messages(request)
        return {
            'portfolio_widget': widget_data,
            'current_fund': current_fund,
        }
    return {'portfolio_widget': {}, 'current_fund': None}
