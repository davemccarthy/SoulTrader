from decimal import Decimal
from django.utils import timezone

from core.fund_session import get_current_fund
from core.models import Trade
from core.portfolio_metrics import get_portfolio_dashboard_data


def get_ticker_messages(request):
    """Generate ticker messages for display (scoped to session fund)."""
    user = request.user
    fund = get_current_fund(request)

    # Space before content
    messages = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

    # Welcome/header message based on whether a fund is selected
    if fund is None:
        display_name = user.get_full_name().strip() or user.username
        messages.append(f"Welcome {display_name}")
    else:
        messages.append(f"Today's {fund.name} Trading Activity")

    now = timezone.now()
    today = now.date()

    if fund is None:
        buy_trades = (
            Trade.objects.filter(
                fund__enabled=True,
                action='BUY',
                created__date=today,
            )
            .select_related('stock')
            .order_by('-created')[:13]
        )
        sell_trades = (
            Trade.objects.filter(
                fund__enabled=True,
                action='SELL',
                created__date=today,
            )
            .select_related('stock')
            .order_by('-created')[:10]
        )
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
    if fund is not None:
        if total_trades_today > 0:
            messages.append(f"{total_trades_today} trades today")
        else:
            messages.append("No trades yet today")
    else:
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
                else:
                    buy_trade = (
                        Trade.objects.filter(
                            fund__enabled=True,
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
    """Session fund if selected; otherwise aggregate all enabled funds (``0``)."""
    fund = get_current_fund(request)
    if fund:
        return get_portfolio_dashboard_data(fund)
    return get_portfolio_dashboard_data(0)


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
