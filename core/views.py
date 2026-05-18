from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import F, Q, Case, When, Value, CharField, Max, Count
from django.http import JsonResponse, Http404
from django.templatetags.static import static
from django.utils import timezone
from decimal import Decimal
from collections import defaultdict
import datetime
import json
import logging
import re

import yfinance as yf

from .models import Profile, Holding, Discovery, Trade, Recommendation, SellInstruction, Health, Snapshot, Advisor
from .fund_session import (
    get_current_fund,
    init_fund_session_after_login,
    set_current_fund,
    clear_fund_session,
)
from .health_display import format_health_score, health_record_template_context
from .portfolio_metrics import get_portfolio_dashboard_data


logger = logging.getLogger(__name__)


def _scoreboard_cutoff_lookback(days: int):
    now = timezone.now()
    period_date = now.date() - datetime.timedelta(days=days)
    midnight = datetime.datetime.combine(period_date, datetime.datetime.min.time())
    if timezone.get_current_timezone():
        return timezone.make_aware(midnight, timezone.get_current_timezone())
    return midnight


def _scoreboard_discovery_advisor_map(sa_ids: list[int], stock_ids: list[int]) -> dict[tuple[int, int], str]:
    discoveries = (
        Discovery.objects.filter(sa_id__in=sa_ids, stock_id__in=stock_ids)
        .select_related("advisor")
        .order_by("sa_id", "stock_id", "created")
    )
    out: dict[tuple[int, int], str] = {}
    for d in discoveries:
        key = (d.sa_id, d.stock_id)
        if key not in out:
            out[key] = d.advisor.name
    return out


def _scoreboard_fifo_outcomes_for_user_stock(
    user_id: int,
    stock_id: int,
    cutoff_dt,
    advisor_for_buy: dict[tuple[int, int], str],
) -> list[dict]:
    trades = (
        Trade.objects.filter(user_id=user_id, stock_id=stock_id, action__in=("BUY", "SELL"))
        .select_related("sa", "stock")
        .order_by("created")
    )
    queue = []
    sold_from = defaultdict(lambda: {"shares": 0, "proceeds": Decimal("0")})
    for t in trades:
        if t.action == "BUY":
            queue.append((t.id, t.shares, t.price))
        else:
            remaining = t.shares
            sell_price = t.price
            while remaining > 0 and queue:
                buy_id, lot_shares, cost_per_share = queue[0]
                take = min(remaining, lot_shares)
                sold_from[buy_id]["shares"] += take
                sold_from[buy_id]["proceeds"] += Decimal(take) * sell_price
                remaining -= take
                if lot_shares == take:
                    queue.pop(0)
                else:
                    queue[0] = (buy_id, lot_shares - take, cost_per_share)

    remaining_by_buy_id = {buy_id: sh for (buy_id, sh, _) in queue}
    outcomes = []
    buy_trades = (
        Trade.objects.filter(user_id=user_id, stock_id=stock_id, action="BUY")
        .select_related("sa", "stock", "stock__advisor")
        .order_by("created")
    )
    for t in buy_trades:
        if t.sa.started < cutoff_dt:
            continue
        advisor = advisor_for_buy.get((t.sa_id, t.stock_id))
        if not advisor and t.stock and getattr(t.stock, "advisor_id", None):
            advisor = t.stock.advisor.name
        if not advisor:
            continue
        cost = Decimal(t.shares) * t.price
        sold = sold_from.get(t.id, {"shares": 0, "proceeds": Decimal("0")})
        proceeds = sold["proceeds"]
        shares_remaining = remaining_by_buy_id.get(t.id, 0)
        current_price = t.stock.price if t.stock else Decimal("0")
        exit_value = proceeds + (Decimal(shares_remaining) * current_price)
        outcomes.append(
            {
                "advisor": advisor,
                "cost": cost,
                "exit_value": exit_value,
                "winner": exit_value >= cost,
            }
        )
    return outcomes


def _fund_advisor_scoreboard_rows(fund_id: int, days: int) -> list[dict]:
    cutoff = _scoreboard_cutoff_lookback(days)
    buy_list = list(
        Trade.objects.filter(fund_id=fund_id, action="BUY", sa__started__gte=cutoff)
        .select_related("sa", "user", "stock")
        .order_by("created")
    )
    if not buy_list:
        return []

    sa_ids = [t.sa_id for t in buy_list]
    stock_ids = [t.stock_id for t in buy_list]
    advisor_for_buy = _scoreboard_discovery_advisor_map(sa_ids, stock_ids)

    by_user_stock = defaultdict(list)
    for t in buy_list:
        by_user_stock[(t.user_id, t.stock_id)].append(t)

    outcomes = []
    for (user_id, stock_id), _ in by_user_stock.items():
        outcomes.extend(_scoreboard_fifo_outcomes_for_user_stock(user_id, stock_id, cutoff, advisor_for_buy))
    if not outcomes:
        return []

    by_name = defaultdict(list)
    for o in outcomes:
        by_name[o["advisor"]].append(o)

    id_by_name = {
        a.name: a.id
        for a in Advisor.objects.filter(name__in=list(by_name.keys())).only("id", "name")
    }

    rows = []
    for name in sorted(by_name.keys()):
        advisor_id = id_by_name.get(name)
        if advisor_id is None:
            continue
        advisor_outcomes = by_name[name]
        n = len(advisor_outcomes)
        winners = sum(1 for o in advisor_outcomes if o["winner"])
        losers = n - winners
        win_rate = (winners / n * 100) if n else 0.0
        total_cost = sum(o["cost"] for o in advisor_outcomes)
        total_exit = sum(o["exit_value"] for o in advisor_outcomes)
        gain_loss_pct = float((total_exit - total_cost) / total_cost * 100) if total_cost else 0.0
        rows.append(
            {
                "advisor_id": advisor_id,
                "trades": n,
                "winners": winners,
                "losers": losers,
                "win_rate": round(win_rate, 1),
                "gain_loss_pct": round(gain_loss_pct, 2),
            }
        )
    rows.sort(key=lambda r: r["advisor_id"])
    return rows


def home(request):
    """Root URL: no marketing home — send users to login or Funds."""
    if request.user.is_authenticated:
        return redirect('core:funds')
    return redirect('core:login')


def _notify_superusers_web_login(user, request) -> None:
    """Fire-and-forget superuser push on successful session login (must not break login)."""
    try:
        from core.services.push import push_super

        when = timezone.localtime(timezone.now()).strftime('%Y-%m-%d %H:%M %Z')
        ip = (request.META.get('REMOTE_ADDR') or '').strip() or 'unknown'
        name = user.get_username()
        push_super(f'Web login: {name} at {when} (IP {ip})')
    except Exception:
        logger.exception('push_super after web login failed')


def login_view(request):
    """Login page"""
    if request.user.is_authenticated:
        return redirect('core:funds')
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            _notify_superusers_web_login(user, request)
            init_fund_session_after_login(request)
            return redirect('core:funds')
        else:
            messages.error(request, 'Invalid credentials')
    return render(request, 'core/login.html', {'current_page': 'login'})


def logout_view(request):
    """Logout"""
    logout(request)
    return redirect('core:login')


@login_required
def funds(request):
    """List enabled funds and switch session-scoped active fund."""
    if request.method == 'POST':
        fund_id = request.POST.get('fund_id')
        if set_current_fund(request, fund_id):
            return redirect('core:holdings')
        messages.error(request, 'Could not switch to that fund.')
        return redirect('core:funds')

    # GET: always clear active fund — user must choose (or re-choose) a fund
    clear_fund_session(request)

    fund_list = Profile.objects.filter(enabled=True).order_by('id')
    current = get_current_fund(request)
    fund_rows = [
        {
            'fund': f,
            'dashboard': get_portfolio_dashboard_data(f),
            'is_current': current is not None and current.pk == f.pk,
        }
        for f in fund_list
    ]
    context = {
        'current_page': 'funds',
        'fund_rows': fund_rows,
        'current_fund_pk': current.pk if current else None,
    }
    return render(request, 'core/funds.html', context)


@login_required
def holdings(request):
    """Holdings page - displays user's stock holdings"""
    fund = get_current_fund(request)
    if fund is None:
        return redirect('core:funds')

    holdings_list = (
        Holding.objects
        .filter(fund=fund)
        .order_by('-id')
        .select_related('stock', 'stock__advisor', 'discovery__advisor')
    )
    
    # Annotate with calculated fields
    holdings_data = []
    for holding in holdings_list:
        stock = holding.stock
        current_price = stock.price or Decimal('0')
        avg_price = holding.average_price or Decimal('0')
        shares = holding.shares or Decimal('0')
        
        # Calculate change percentage
        if avg_price and avg_price > 0:
            change_percent = ((current_price / avg_price) * 100 - 100)
        else:
            change_percent = Decimal('0')
        
        # Calculate P&L
        pl = (current_price * shares) - (avg_price * shares)
        total_value = current_price * shares
        if change_percent > 0:
            price_class = 'positive'
        elif change_percent < 0:
            price_class = 'negative'
        else:
            price_class = 'neutral'
        
        # Get advisor name for discovery - use holding provenance, fallback to stock.advisor
        discovery_obj = holding.discovery
        discovery_comment = None
        if discovery_obj:
            discovery = discovery_obj.advisor.name if discovery_obj.advisor else ""
            discovery_advisor = discovery_obj.advisor

            # Derive a short comment from the discovery explanation (first non-empty clause)
            explanation = (discovery_obj.explanation or "").strip()
            if explanation:
                normalized = " ".join(explanation.split())
                if normalized:
                    segments = [segment.strip() for segment in normalized.split("|") if segment.strip()]
                    for segment in segments:
                        lower = segment.lower()
                        # Skip bare URLs
                        if segment.startswith("http://") or segment.startswith("https://"):
                            continue
                        # Handle "Article: Title" style segments
                        if lower.startswith("article:"):
                            title = segment.split(":", 1)[1].strip()
                            if title:
                                discovery_comment = title
                                break
                        else:
                            discovery_comment = segment
                            break
        else:
            # Fallback to stock.advisor if no discovery found
            discovery = stock.advisor.name if stock.advisor else ""
            discovery_advisor = stock.advisor

        holdings_data.append({
            'symbol': stock.symbol,
            'company': stock.company,
            'price': current_price,
            'change_percent': change_percent,
            'pl': pl,
            'discovery': discovery,
            'discovery_logo': _advisor_logo_url(discovery_advisor),
            'stock_id': stock.id,
            'shares': shares,
            'average_price': avg_price,
            'total_value': total_value,
            'price_class': price_class,
            'website': stock.website,
            'buy_date': holding.created,
            'discovery_comment': discovery_comment,
        })
    
    # Build snapshot data for holdings stacked bar chart (last 120 days)
    # Aggregate by week when > 14 points to avoid bunched bars
    today = timezone.now().date()
    start_date = today - datetime.timedelta(days=120)
    snapshots_qs = (
        Snapshot.objects
        .filter(fund=fund, date__gte=start_date)
        .order_by('date')
    )
    snapshots_list = list(snapshots_qs)

    holdings_chart_data = []
    if len(snapshots_list) > 14:
        # Group by week (Monday as week start), use last snapshot of each week
        by_week = defaultdict(list)
        for snap in snapshots_list:
            week_monday = snap.date - datetime.timedelta(days=snap.date.weekday())
            by_week[week_monday].append(snap)
        for week_monday in sorted(by_week.keys()):
            snaps_in_week = by_week[week_monday]
            last_snap = max(snaps_in_week, key=lambda s: s.date)
            holdings_chart_data.append({
                'date': last_snap.date.isoformat(),
                'label': f"{last_snap.date.strftime('%b')} {last_snap.date.day}",
                'cash': float(last_snap.cash_value or Decimal('0')),
                'holdings': float(last_snap.holdings_value or Decimal('0')),
            })
    else:
        for snap in snapshots_list:
            holdings_chart_data.append({
                'date': snap.date.isoformat(),
                'cash': float(snap.cash_value or Decimal('0')),
                'holdings': float(snap.holdings_value or Decimal('0')),
            })

    context = {
        'current_page': 'holdings',
        'holdings': holdings_data,
        'holdings_chart_data_json': json.dumps(holdings_chart_data),
    }
    return render(request, 'core/holdings.html', context)


def _format_trade_timestamp(dt):
    """Friendly timestamp for trades: Today HH:MM/Yesterday/Weekday/Date."""
    if not dt:
        return None

    dt_local = timezone.localtime(dt)
    now_local = timezone.localtime(timezone.now())

    trade_date = dt_local.date()
    today = now_local.date()

    if trade_date == today:
        # Show only time (24-hour format) for today's entries
        return dt_local.strftime('%H:%M')

    if trade_date == today - datetime.timedelta(days=1):
        return "Yesterday"

    # Within the last 5 days (but not today/yesterday): show weekday
    if today - datetime.timedelta(days=5) < trade_date < today - datetime.timedelta(days=1):
        weekday = dt_local.strftime('%A')
        return weekday

    # Fallback: plain date
    return dt_local.strftime('%Y-%m-%d')


def _advisor_logo_url(advisor):
    if not advisor:
        return None

    python_class = getattr(advisor, 'python_class', None)
    if not python_class:
        return None

    filename = f"core/advisors/{python_class.lower()}.png"
    return static(filename)


@login_required
def holding_detail(request, stock_id):
    from django.utils import timezone

    fund = get_current_fund(request)
    if fund is None:
        return JsonResponse({'error': 'Select a fund first.'}, status=403)

    try:
        holding = (
            Holding.objects
            .select_related('stock')
            .get(fund=fund, stock_id=stock_id)
        )
    except Holding.DoesNotExist:
        raise Http404("Holding not found")

    stock = holding.stock
    shares = holding.shares or 0
    avg_price = holding.average_price or Decimal('0')
    current_price = stock.price or Decimal('0')

    worth = current_price * shares
    invested = avg_price * shares
    return_amount = worth - invested

    if avg_price and avg_price > 0:
        pl_percent = ((current_price / avg_price) * Decimal('100')) - Decimal('100')
    else:
        pl_percent = Decimal('0')

    trades_qs = list(
        Trade.objects
        .filter(fund=fund, stock_id=stock_id)
        .order_by('-id')
    )

    sa_ids = {trade.sa_id for trade in trades_qs if trade.sa_id}

    discoveries_map = {}
    if sa_ids:
        for discovery in Discovery.objects.select_related('advisor', 'sa').filter(sa_id__in=sa_ids, stock_id=stock_id):
            discoveries_map[discovery.sa_id] = discovery

    recommendations_map = defaultdict(list)
    if sa_ids:
        for recommendation in Recommendation.objects.select_related('advisor').filter(sa_id__in=sa_ids, stock_id=stock_id):
            recommendations_map[recommendation.sa_id].append(recommendation)

    trades_payload = []
    for trade in trades_qs:
        trade_value = None
        if trade.price is not None:
            trade_value = float((trade.price or Decimal('0')) * trade.shares)

        trade_payload = {
            'id': trade.id,
            'action': trade.action,
            'price': float(trade.price) if trade.price is not None else None,
            'shares': trade.shares,
            'sa_id': trade.sa_id,
            'sa_started': trade.sa.started.isoformat() if trade.sa and trade.sa.started else None,
            'created': trade.created.isoformat() if trade.created else None,
            'value': trade_value,
        }

        discovery = discoveries_map.get(trade.sa_id)
        if discovery:
            trade_payload['discovery'] = {
                'id': discovery.id,
                'advisor': discovery.advisor.name if discovery.advisor else '',
                'advisor_logo': _advisor_logo_url(discovery.advisor),
                'explanation': discovery.explanation,
                'sa_id': discovery.sa_id,
                'sa_started': discovery.sa.started.isoformat() if discovery.sa and discovery.sa.started else None,
            }
            
            # Get sell instructions for this discovery
            instructions = SellInstruction.objects.filter(discovery=discovery).order_by('id')
            instruction_choices = dict(SellInstruction.choices)
            sell_instructions = []
            for instruction in instructions:
                instruction_data = {
                    'instruction': instruction.instruction,
                    'description': instruction_choices.get(instruction.instruction, instruction.instruction),
                    'value': float(instruction.value1) if instruction.value1 is not None else None,
                    'value2': float(instruction.value2) if instruction.value2 is not None else None,
                }
                
                # Add status/context for each instruction type
                if instruction.instruction in ['STOP_LOSS', 'STOP_PRICE'] and instruction.value1:
                    instruction_data['status'] = 'active' if current_price <= instruction.value1 else 'pending'
                    instruction_data['current_price'] = float(current_price)
                elif instruction.instruction == 'STOP_PERCENTAGE' and instruction.value1:
                    basis = avg_price or discovery.price
                    mult = float(instruction.value1)
                    stop_px = (mult * float(basis)) if basis and mult <= 3 else float(instruction.value1)
                    instruction_data['stop_price'] = stop_px
                    instruction_data['multiplier'] = mult if mult <= 3 else None
                    instruction_data['status'] = (
                        'active' if float(current_price) <= stop_px else 'pending'
                    )
                    instruction_data['current_price'] = float(current_price)
                elif instruction.instruction == 'TARGET_PRICE' and instruction.value1:
                    instruction_data['status'] = 'active' if current_price >= instruction.value1 else 'pending'
                    instruction_data['current_price'] = float(current_price)
                elif instruction.instruction == 'TARGET_PERCENTAGE' and instruction.value1:
                    basis = avg_price or discovery.price
                    target_px = float(basis) * float(instruction.value1) if basis else None
                    instruction_data['target_price'] = target_px
                    instruction_data['multiplier'] = float(instruction.value1)
                    instruction_data['status'] = (
                        'active' if target_px is not None and float(current_price) >= target_px else 'pending'
                    )
                    instruction_data['current_price'] = float(current_price)
                elif instruction.instruction == 'AFTER_DAYS' and instruction.value1:
                    days_held = (timezone.now() - discovery.created).days
                    instruction_data['days_held'] = days_held
                    instruction_data['status'] = 'active' if days_held >= int(instruction.value1) else 'pending'
                elif instruction.instruction == 'DESCENDING_TREND' and instruction.value1 is not None:
                    # DESCENDING_TREND value is a threshold (e.g., -0.20 means sell if trend < -0.20)
                    instruction_data['value'] = float(instruction.value1)
                    instruction_data['status'] = 'pending'  # Status determined during analysis
                    threshold = float(instruction.value1)
                    buy_price = float(avg_price) if avg_price else float(discovery.price) if discovery.price else None
                    in_loss = buy_price is not None and float(current_price) < buy_price

                    trend = holding.stock.calc_trend(hours=2)

                    if trend is not None:
                        instruction_data['current_trend'] = float(trend)
                        instruction_data['status'] = 'active' if (not in_loss and float(trend) < threshold) else 'pending'
                elif instruction.instruction in ['TARGET_DIMINISHING', 'PERCENTAGE_DIMINISHING'] and instruction.value1:
                    # value1 = original target price, value2 = max_days
                    max_days = int(instruction.value2) if instruction.value2 is not None else 14
                    instruction_data['max_days'] = max_days
                    # Calculate current diminished target
                    days_held = (timezone.now() - discovery.created).days if discovery.created else 0
                    buy_price = float(avg_price) if avg_price else float(discovery.price) if discovery.price else None
                    original_target = float(instruction.value1)
                    if buy_price is not None:
                        if days_held <= max_days:
                            progress = float(days_held) / float(max_days) if max_days > 0 else 1.0
                            current_target = original_target - progress * (original_target - buy_price)
                        else:
                            current_target = buy_price  # After max_days, target = buy_price (break-even)
                        instruction_data['current_target'] = current_target
                        instruction_data['days_held'] = days_held
                        instruction_data['status'] = 'active' if current_price >= current_target else 'pending'
                    else:
                        instruction_data['status'] = 'pending'  # Status determined during analysis
                elif instruction.instruction in ['STOP_AUGMENTING', 'PERCENTAGE_AUGMENTING'] and instruction.value1:
                    # value1 = original stop price, value2 = max_days
                    instruction_data['max_days'] = int(instruction.value2) if instruction.value2 is not None else 28
                    instruction_data['status'] = 'pending'  # Status determined during analysis
                elif instruction.instruction == 'PEAKED' and instruction.value1 and discovery and discovery.created:
                    # value1 = percentage threshold of peak gain giveback (e.g., 33 = give back 33% of peak gain)
                    giveback_pct = float(instruction.value1)
                    min_peak_gain_pct = float(instruction.value2) if instruction.value2 is not None else 5.0
                    buy_price = float(avg_price) if avg_price else float(discovery.price) if discovery.price else None
                    instruction_data['status'] = 'active' if buy_price is not None and holding.stock.downturned(
                        discovery.created,
                        buy_price=buy_price,
                        giveback_pct=giveback_pct,
                        min_peak_gain_pct=min_peak_gain_pct,
                    ) else 'pending'
                    instruction_data['current_price'] = float(current_price)
                    peak_price = holding.stock.peak_since(discovery.created)
                    if peak_price and peak_price > 0:
                        cp = float(current_price)
                        instruction_data['peak_price'] = peak_price
                        instruction_data['pct_from_peak'] = ((cp - peak_price) / peak_price) * 100
                
                sell_instructions.append(instruction_data)
            trade_payload['sell_instructions'] = sell_instructions
        else:
            trade_payload['discovery'] = None
            trade_payload['sell_instructions'] = []

        recommendations = recommendations_map.get(trade.sa_id, [])
        trade_payload['recommendations'] = [
            {
                'id': recommendation.id,
                'advisor': recommendation.advisor.name if recommendation.advisor else '',
                'advisor_logo': _advisor_logo_url(recommendation.advisor),
                'confidence': float(recommendation.confidence) if recommendation.confidence is not None else None,
                'explanation': recommendation.explanation,
            }
            for recommendation in recommendations
        ]

        trades_payload.append(trade_payload)

    # Determine trend class for detail view
    trend_value = stock.calc_trend()
    if trend_value is not None:
        if trend_value > 0:
            trend_class = 'positive'
        elif trend_value < 0:
            trend_class = 'negative'
        else:
            trend_class = 'neutral'
    else:
        trend_class = 'neutral'
    
    payload = {
        'stock': {
            'id': stock.id,
            'symbol': stock.symbol,
            'company': stock.company,
            'trend': float(trend_value) if trend_value is not None else None,
            'trend_class': trend_class,
            'price': float(current_price),
        },
        'holding': {
            'shares': shares,
            'average_price': float(avg_price),
            'worth': float(worth),
            'invested': float(invested),
            'return_amount': float(return_amount),
            'pl_percent': float(pl_percent),
        },
        'trades': trades_payload,
    }

    return JsonResponse(payload)


@login_required
def holding_history(request, stock_id):
    """Holdings history view - unified view with heading, discovery, health history, and trade history"""
    from django.utils import timezone

    fund = get_current_fund(request)
    if fund is None:
        return JsonResponse({'error': 'Select a fund first.'}, status=403)

    holding = (
        Holding.objects
        .select_related('stock')
        .filter(fund=fund, stock_id=stock_id)
        .first()
    )

    # Trade history is the canonical source for this detail view; this lets
    # closed positions (no active Holding row) still render from historical trades.
    trades_for_position = list(
        Trade.objects
        .select_related('stock')
        .filter(fund=fund, stock_id=stock_id)
        .order_by('created', 'id')
    )

    if holding is None and not trades_for_position:
        raise Http404("No holding or trade history found")

    stock = holding.stock if holding is not None else trades_for_position[0].stock

    if holding is not None:
        shares = holding.shares or 0
        avg_price = holding.average_price or Decimal('0')
    else:
        # Reconstruct current position from trade ledger for closed positions.
        position_shares = Decimal('0')
        position_avg_price = Decimal('0')
        for trade in trades_for_position:
            trade_shares = Decimal(trade.shares or 0)
            trade_price = Decimal(trade.price or 0)

            if trade.action == 'BUY':
                total_cost = (position_avg_price * position_shares) + (trade_price * trade_shares)
                position_shares += trade_shares
                position_avg_price = (total_cost / position_shares) if position_shares > 0 else Decimal('0')
            else:
                position_shares = max(position_shares - trade_shares, Decimal('0'))
                if position_shares == 0:
                    position_avg_price = Decimal('0')

        shares = position_shares
        avg_price = position_avg_price

    current_price = stock.price or Decimal('0')
    worth = current_price * shares

    # Recent price history for chart (e.g., last 60 days, daily closes)
    price_history = []
    try:
        hist = yf.Ticker(stock.symbol).history(period="2mo", interval="1d")
        if not hist.empty and "Close" in hist.columns:
            for ts, row in hist.iterrows():
                close = row.get("Close")
                if close is None:
                    continue
                try:
                    close_value = float(close)
                except (TypeError, ValueError):
                    continue
                price_history.append(
                    {
                        "date": ts.date().isoformat(),
                        "close": close_value,
                    }
                )
    except Exception as e:
        logger.warning(f"Could not fetch price history for {stock.symbol}: {e}")

    # Calculate change (from average price, like holdings view)
    stock.refresh()
    change_percent = 0.0
    if avg_price and avg_price > 0:
        change_percent = float(((current_price / avg_price) * 100 - 100))

    # Heading data
    trend_value = stock.calc_trend()
    if trend_value is not None:
        if trend_value > 0:
            trend_class = 'positive'
        elif trend_value < 0:
            trend_class = 'negative'
        else:
            trend_class = 'neutral'
    else:
        trend_class = 'neutral'

    heading = {
        'symbol': stock.symbol,
        'company': stock.company,
        'image': f"https://images.financialmodelingprep.com/symbol/{stock.symbol}.png",
        'average_price': float(avg_price),
        'worth': float(worth),
        'shares': shares,
        'price': float(current_price),
        'change_percent': change_percent,
        'price_class': 'positive' if change_percent >= 0 else 'negative',
        'change_class': 'positive' if change_percent >= 0 else 'negative',
        'trend': float(trend_value) if trend_value is not None else None,
        'trend_class': trend_class,
        'sector': stock.sector or None,
        'industry': stock.industry or None,
        'exchange': stock.exchange or None,
        'beta': float(stock.beta) if stock.beta is not None else None,
    }

    # Discovery data (most recent)
    discovery_payload = None
    sell_instructions = []
    discovery_obj = (
        Discovery.objects
        .select_related('advisor', 'sa')
        .filter(stock=stock)
        .order_by('-created')
        .first()
    )
    
    if discovery_obj:
        discovery_payload = {
            'id': discovery_obj.id,
            'advisor': discovery_obj.advisor.name if discovery_obj.advisor else '',
            'advisor_logo': _advisor_logo_url(discovery_obj.advisor),
            'explanation': discovery_obj.explanation,
            'created': discovery_obj.created.isoformat() if discovery_obj.created else None,
            'sa_id': discovery_obj.sa_id,
            'url': None,  # Extract URL from explanation if present
        }
        # Extract URL from explanation (format: "Article: [title] | [url] | ...")
        # The URL is the segment immediately after "Article: [title]"
        if 'Article:' in discovery_obj.explanation:
            parts = [p.strip() for p in discovery_obj.explanation.split('|')]
            for i, part in enumerate(parts):
                if part.startswith('Article:'):
                    # Found "Article: [title]", URL should be next part if it looks like a URL
                    if i + 1 < len(parts):
                        potential_url = parts[i + 1].strip()
                        if potential_url.startswith('http://') or potential_url.startswith('https://'):
                            discovery_payload['url'] = potential_url
                            break
        
        # Get sell instructions for this discovery
        instructions = SellInstruction.objects.filter(discovery=discovery_obj).order_by('id')
        # Create mapping from instruction code to description
        instruction_choices = dict(SellInstruction.choices)
        
        for instruction in instructions:
            instruction_data = {
                'instruction': instruction.instruction,
                'description': instruction_choices.get(instruction.instruction, instruction.instruction),
                'value': float(instruction.value1) if instruction.value1 is not None else None,
                'value2': float(instruction.value2) if instruction.value2 is not None else None,
            }
            
            # Add status/context for each instruction type
            if instruction.instruction == 'STOP_PRICE' and instruction.value1:
                instruction_data['status'] = 'active' if current_price <= instruction.value1 else 'pending'
                instruction_data['current_price'] = float(current_price)
            elif instruction.instruction == 'STOP_PERCENTAGE' and instruction.value1:
                basis = avg_price or discovery_obj.price
                mult = float(instruction.value1)
                stop_px = (mult * float(basis)) if basis and mult <= 3 else float(instruction.value1)
                instruction_data['stop_price'] = stop_px
                instruction_data['multiplier'] = mult if mult <= 3 else None
                instruction_data['status'] = (
                    'active' if float(current_price) <= stop_px else 'pending'
                )
                instruction_data['current_price'] = float(current_price)
            elif instruction.instruction == 'TARGET_PRICE' and instruction.value1:
                instruction_data['status'] = 'active' if current_price >= instruction.value1 else 'pending'
                instruction_data['current_price'] = float(current_price)
            elif instruction.instruction == 'TARGET_PERCENTAGE' and instruction.value1:
                basis = avg_price or discovery_obj.price
                target_px = float(basis) * float(instruction.value1) if basis else None
                instruction_data['target_price'] = target_px
                instruction_data['multiplier'] = float(instruction.value1)
                instruction_data['status'] = (
                    'active' if target_px is not None and float(current_price) >= target_px else 'pending'
                )
                instruction_data['current_price'] = float(current_price)
            elif instruction.instruction == 'AFTER_DAYS' and instruction.value1:
                days_held = (timezone.now() - discovery_obj.created).days
                instruction_data['days_held'] = days_held
                instruction_data['status'] = 'active' if days_held >= int(instruction.value1) else 'pending'
            elif instruction.instruction == 'DESCENDING_TREND' and instruction.value1 is not None:
                instruction_data['value'] = float(instruction.value1)
                threshold = float(instruction.value1)
                buy_price = float(avg_price) if avg_price else float(discovery_obj.price) if discovery_obj.price else None
                in_loss = buy_price is not None and float(current_price) < buy_price

                trend = stock.calc_trend(hours=2)
                if trend is not None:
                    instruction_data['current_trend'] = float(trend)
                    instruction_data['status'] = 'active' if (not in_loss and float(trend) < threshold) else 'pending'
                else:
                    instruction_data['status'] = 'pending'
            elif instruction.instruction in ['TARGET_DIMINISHING', 'PERCENTAGE_DIMINISHING'] and instruction.value1:
                # value1 = original target price, value2 = max_days
                max_days = int(instruction.value2) if instruction.value2 is not None else 14
                instruction_data['max_days'] = max_days
                # Calculate current diminished target
                days_held = (timezone.now() - discovery_obj.created).days if discovery_obj.created else 0
                buy_price = float(avg_price) if avg_price else float(discovery_obj.price) if discovery_obj.price else None
                original_target = float(instruction.value1)
                if buy_price is not None:
                    if days_held <= max_days:
                        progress = float(days_held) / float(max_days) if max_days > 0 else 1.0
                        current_target = original_target - progress * (original_target - buy_price)
                    else:
                        current_target = buy_price  # After max_days, target = buy_price (break-even)
                    instruction_data['current_target'] = current_target
                    instruction_data['days_held'] = days_held
                    instruction_data['status'] = 'active' if current_price >= current_target else 'pending'
                else:
                    instruction_data['status'] = 'pending'  # Status determined during analysis
            elif instruction.instruction in ['STOP_AUGMENTING', 'PERCENTAGE_AUGMENTING'] and instruction.value1:
                # value1 = original stop price, value2 = max_days
                instruction_data['max_days'] = int(instruction.value2) if instruction.value2 is not None else 28
                instruction_data['status'] = 'pending'  # Status determined during analysis
            elif instruction.instruction == 'PEAKED' and instruction.value1 and discovery_obj and discovery_obj.created:
                # value1 = percentage threshold of peak gain giveback (e.g., 33 = give back 33% of peak gain)
                giveback_pct = float(instruction.value1)
                min_peak_gain_pct = float(instruction.value2) if instruction.value2 is not None else 5.0
                buy_price = float(avg_price) if avg_price else float(discovery_obj.price) if discovery_obj.price else None
                instruction_data['status'] = 'active' if buy_price is not None and stock.downturned(
                    discovery_obj.created,
                    buy_price=buy_price,
                    giveback_pct=giveback_pct,
                    min_peak_gain_pct=min_peak_gain_pct,
                ) else 'pending'
                instruction_data['current_price'] = float(current_price)
                peak_price = stock.peak_since(discovery_obj.created)
                if peak_price and peak_price > 0:
                    cp = float(current_price)
                    instruction_data['peak_price'] = peak_price
                    instruction_data['pct_from_peak'] = ((cp - peak_price) / peak_price) * 100
            
            sell_instructions.append(instruction_data)

    # Health history (derive from discovery.health so optional health checks work)
    # Note: multiple discoveries can point to the same Health row; dedupe by Health.id.
    health_history = []
    seen_health_ids = set()
    discovery_health_checks = (
        Discovery.objects
        .select_related('health', 'health__sa')
        .filter(stock=stock, health__isnull=False)
        .order_by('-health__created')
    )

    for disc_health in discovery_health_checks:
        health = getattr(disc_health, 'health', None)
        if not health or health.id in seen_health_ids:
            continue
        seen_health_ids.add(health.id)

        meta = health.meta or {}
        health_data = {
            'id': health.id,
            'score': float(health.score),
            'created': health.created.isoformat() if health.created else None,
            'meta': meta,
            'confidence_score': meta.get('confidence_score'),
            'health_score': meta.get('health_score'),
            'valuation_score': meta.get('valuation_score'),
            'piotroski': meta.get('piotroski'),
            'altman_z': meta.get('altman_z'),
            'gemini_weight': meta.get('gemini_weight'),
            'gemini_rec': meta.get('gemini_recommendation'),
            'gemini_explanation': meta.get('gemini_explanation'),
        }
        health_history.append(health_data)

    # Trade history
    trades_data = []
    trades_qs = (
        Trade.objects
        .select_related('sa')
        .filter(fund=fund, stock_id=stock_id)
        .order_by('-created', '-id')
    )
    
    for trade in trades_qs:
        # Calculate P&L
        pl_amount = None
        pl_percent = None
        pl_class = 'neutral'
        
        if trade.action == 'SELL' and trade.cost:
            # Realized P&L for SELL trades
            cost_basis = Decimal(str(trade.cost))
            pl_amount = float((Decimal(str(trade.price)) - cost_basis) * trade.shares)
            if cost_basis > 0:
                pl_percent = float(((Decimal(str(trade.price)) - cost_basis) / cost_basis) * 100)
        elif trade.action == 'BUY':
            # Unrealized P&L for BUY trades
            buy_price = Decimal(str(trade.price))
            pl_amount = float((current_price - buy_price) * trade.shares)
            if buy_price > 0:
                pl_percent = float(((current_price - buy_price) / buy_price) * 100)
        
        if pl_amount is not None:
            pl_class = 'positive' if pl_amount >= 0 else 'negative'
        
        # Calculate current price class for colorization
        current_price_class = 'neutral'
        if current_price and trade.price:
            if current_price > trade.price:
                current_price_class = 'positive'
            elif current_price < trade.price:
                current_price_class = 'negative'
        
        trade_data = {
            'id': trade.id,
            'action': trade.action,
            'price': float(trade.price) if trade.price is not None else None,
            'current_price': float(current_price) if current_price else None,
            'current_price_class': current_price_class,
            'shares': trade.shares,
            'cost': float(trade.cost) if trade.cost is not None else None,
            'pl_amount': pl_amount,
            'pl_percent': pl_percent,
            'pl_class': pl_class,
            'created': trade.created.isoformat() if trade.created else None,
            'sa_id': trade.sa_id,
        }
        trades_data.append(trade_data)

    payload = {
        'heading': heading,
        'discovery': discovery_payload,
        'health_history': health_history,
        'trades': trades_data,
        'sell_instructions': sell_instructions,
        'price_history': price_history,
    }

    return JsonResponse(payload)


@login_required
def trades(request):
    """Trades page - summarize executed trades using holdings styling."""
    fund = get_current_fund(request)
    if fund is None:
        return redirect('core:funds')

    trade_qs = (
        Trade.objects
        .filter(fund=fund)
        .select_related('stock', 'stock__advisor', 'sa')
        .order_by('sa__started', 'id')
    )

    # Prefetch discoveries for all trades by (stock_id, sa_id)
    sa_ids = {t.sa_id for t in trade_qs if t.sa_id}
    stock_ids = {t.stock_id for t in trade_qs}

    discoveries_map = {}
    if sa_ids:
        discoveries = (
            Discovery.objects
            .select_related('advisor')
            .filter(sa_id__in=sa_ids, stock_id__in=stock_ids)
        )
        for discovery in discoveries:
            discoveries_map[(discovery.stock_id, discovery.sa_id)] = discovery

    inventory = defaultdict(lambda: {'shares': Decimal('0'), 'avg_cost': Decimal('0')})
    metrics = {}

    for trade in trade_qs:
        state = inventory[trade.stock_id]
        shares = Decimal(trade.shares or 0)
        price = Decimal(trade.price or 0)
        amount = shares * price
        current_price = Decimal(trade.stock.price or 0)

        if trade.action == 'BUY':
            total_cost = (state['avg_cost'] * state['shares']) + amount
            total_shares = state['shares'] + shares
            avg_cost = total_cost / total_shares if total_shares else Decimal('0')
            state['shares'] = total_shares
            state['avg_cost'] = avg_cost

            pl_amount = (current_price - price) * shares if shares else Decimal('0')
            pl_percent = ((current_price / price) * Decimal('100') - Decimal('100')) if price > 0 else None

            metrics[trade.id] = {
                'cost': amount,
                'pl_amount': pl_amount,
                'pl_percent': pl_percent,
                'realized': False,
            }
        else:
            # Use stored cost if available (for SELL trades), otherwise use inventory tracking
            if trade.cost:
                avg_cost = trade.cost
            else:
                avg_cost = state['avg_cost'] if state['shares'] > 0 else price
            pl_amount = (price - avg_cost) * shares if shares else Decimal('0')
            pl_percent = ((price / avg_cost) * Decimal('100') - Decimal('100')) if avg_cost > 0 else None
            state['shares'] = max(state['shares'] - shares, Decimal('0'))

            metrics[trade.id] = {
                'cost': amount,
                'pl_amount': pl_amount,
                'pl_percent': pl_percent,
                'realized': True,
            }

    trades_payload = []
    for trade in reversed(trade_qs):
        data = metrics.get(trade.id, {})
        pl_amount = data.get('pl_amount')
        realized = data.get('realized', False)

        if pl_amount is not None and pl_amount > 0:
            pl_class = 'positive'
        elif pl_amount is not None and pl_amount < 0:
            pl_class = 'negative'
        else:
            pl_class = 'neutral'
        
        if realized:
            price_class = pl_class
        else:
            price_class = 'neutral'

        # For BUY trades, prefer the specific discovery tied to this SA.
        # For SELL or trades without a matching discovery, fall back to stock.advisor.
        discovery_obj = discoveries_map.get((trade.stock_id, trade.sa_id)) if trade.action == 'BUY' else None

        if discovery_obj and discovery_obj.advisor:
            advisor_obj = discovery_obj.advisor
            discovery_payload = {
                'advisor': advisor_obj.name,
                'advisor_logo': _advisor_logo_url(advisor_obj),
                'sa_id': discovery_obj.sa_id,
            }
        else:
            advisor_obj = getattr(trade.stock, 'advisor', None)
            if advisor_obj:
                discovery_payload = {
                    'advisor': advisor_obj.name,
                    'advisor_logo': _advisor_logo_url(advisor_obj),
                    'sa_id': None,
                }
            else:
                discovery_payload = None

        trades_payload.append({
            'id': trade.id,
            'stock_id': trade.stock_id,
            'symbol': trade.stock.symbol,
            'company': trade.stock.company,
            'shares': trade.shares,
            'price': trade.price,
            'sa_id': trade.sa_id,
            'timestamp': trade.sa.started if trade.sa else None,
            'timestamp_display': _format_trade_timestamp(trade.sa.started) if trade.sa else None,
            'action': trade.action,
            'cost': data.get('cost', Decimal('0')),
            'pl_amount': pl_amount,
            'pl_percent': data.get('pl_percent'),
            'pl_class': pl_class,
            'price_class': price_class,
            'realized': realized,
            'explanation': trade.explanation,
            'discovery': discovery_payload,
        })

    # Build snapshot data for trades stacked bar chart (last 30 days)
    today = timezone.now().date()
    start_date = today - datetime.timedelta(days=30)
    snapshots_qs = (
        Snapshot.objects
        .filter(fund=fund, date__gte=start_date)
        .order_by('date')
    )

    trades_chart_data = []
    for snap in snapshots_qs:
        trades_chart_data.append({
            'date': snap.date.isoformat(),
            'trade_daily': float(snap.trade_daily or Decimal('0')),
            'trade_cumulative': float(snap.trade_cumulative or Decimal('0')),
        })

    context = {
        'current_page': 'trades',
        'trades': trades_payload,
        'trades_chart_data_json': json.dumps(trades_chart_data),
    }
    return render(request, 'core/trades.html', context)


@login_required
def advisory(request):
    """Advisory page - advisor list with simple lookback stats for the selected fund."""
    fund = get_current_fund(request)
    if fund is None:
        return redirect('core:funds')

    lookback_options = (7, 30, 90)
    try:
        lookback_days = int(request.GET.get('days', 30))
    except (TypeError, ValueError):
        lookback_days = 30
    if lookback_days not in lookback_options:
        lookback_days = 30

    cutoff = timezone.now() - datetime.timedelta(days=lookback_days)
    python_classes = list(fund.advisors or [])

    advisors = list(Advisor.objects.filter(python_class__in=python_classes))
    advisor_by_class = {advisor.python_class: advisor for advisor in advisors}
    ordered_advisors = [advisor_by_class[pc] for pc in python_classes if pc in advisor_by_class]
    advisor_ids = [advisor.id for advisor in ordered_advisors]

    discovery_counts = {}
    if advisor_ids:
        discovery_counts = {
            row['advisor_id']: row['n']
            for row in (
                Discovery.objects
                .filter(advisor_id__in=advisor_ids, created__gte=cutoff)
                .values('advisor_id')
                .annotate(n=Count('id'))
            )
        }

    scoreboard_rows = _fund_advisor_scoreboard_rows(fund.id, lookback_days)
    scoreboard_by_advisor_id = {row['advisor_id']: row for row in scoreboard_rows}

    advisory_rows = []
    for advisor in ordered_advisors:
        score = scoreboard_by_advisor_id.get(advisor.id, {})
        advisory_rows.append({
            'id': advisor.id,
            'name': advisor.name,
            'description': advisor.description or '',
            'discovery_count': discovery_counts.get(advisor.id, 0),
            'winners': score.get('winners', 0),
            'losers': score.get('losers', 0),
            'trades': score.get('trades', 0),
            'win_rate': score.get('win_rate', 0.0),
            'gain_loss_pct': score.get('gain_loss_pct', 0.0),
            'logo_url': _advisor_logo_url(advisor),
        })

    context = {
        'current_page': 'advisory',
        'lookback_days': lookback_days,
        'lookback_options': lookback_options,
        'advisory_rows': advisory_rows,
    }
    return render(request, 'core/advisory.html', context)


@login_required
def advisory_discoveries(request, advisor_id: int):
    """Advisor discoveries page - list recent discoveries for one advisor."""
    fund = get_current_fund(request)
    if fund is None:
        return redirect('core:funds')

    lookback_options = (7, 30, 90)
    try:
        lookback_days = int(request.GET.get('days', 30))
    except (TypeError, ValueError):
        lookback_days = 30
    if lookback_days not in lookback_options:
        lookback_days = 30

    cutoff = timezone.now() - datetime.timedelta(days=lookback_days)
    advisor = get_object_or_404(Advisor, pk=advisor_id)

    scoreboard_rows = _fund_advisor_scoreboard_rows(fund.id, lookback_days)
    advisor_stats = next((row for row in scoreboard_rows if row.get('advisor_id') == advisor.id), None)

    discoveries_qs = (
        Discovery.objects
        .filter(advisor_id=advisor.id, created__gte=cutoff)
        .select_related('stock', 'health')
        .order_by('-id')
    )

    discoveries = list(discoveries_qs)

    discovery_rows = []
    for discovery in discoveries:
        company = (discovery.stock.company or '').strip()
        score = float(discovery.health.score) if discovery.health else None
        outcome = _discovery_outcome(discovery, score)
        discovery_price = discovery.price
        current_price = discovery.stock.price
        pnl_pct = None
        if discovery_price and discovery_price > 0 and current_price is not None:
            pnl_pct = float((current_price - discovery_price) / discovery_price * 100)
        score_display = format_health_score(score)

        discovery_rows.append({
            'id': discovery.id,
            'symbol': discovery.stock.symbol,
            'company': company or discovery.stock.symbol,
            'price': current_price,
            'industry': discovery.stock.industry or '',
            'sector': discovery.stock.sector or '',
            'exchange': discovery.stock.exchange or '',
            'discovery_price': discovery_price,
            'current_price': current_price,
            'pnl_pct': pnl_pct,
            'created': discovery.created,
            'explanation': discovery.explanation or '',
            'explanation_paragraphs': _discovery_paragraphs(discovery.explanation),
            'health_score_display': score_display,
            'health': health_record_template_context(discovery.health) if discovery.health else None,
            'outcome': outcome,
        })

    context = {
        'current_page': 'advisory',
        'lookback_days': lookback_days,
        'lookback_options': lookback_options,
        'header_row': {
            'name': advisor.name,
            'description': advisor.description or '',
            'logo_url': _advisor_logo_url(advisor),
            'discovery_count': len(discovery_rows),
            'trades': advisor_stats.get('trades', 0) if advisor_stats else 0,
            'winners': advisor_stats.get('winners', 0) if advisor_stats else 0,
            'losers': advisor_stats.get('losers', 0) if advisor_stats else 0,
            'win_rate': advisor_stats.get('win_rate', 0.0) if advisor_stats else 0.0,
            'gain_loss_pct': advisor_stats.get('gain_loss_pct', 0.0) if advisor_stats else 0.0,
        },
        'advisor': {
            'id': advisor.id,
            'name': advisor.name,
            'description': advisor.description or '',
            'logo_url': _advisor_logo_url(advisor),
        },
        'advisor_stats': advisor_stats,
        'discovery_rows': discovery_rows,
    }
    return render(request, 'core/advisory_discoveries.html', context)


def _discovery_excerpt(explanation: str | None) -> str:
    """Short discovery copy from first meaningful explanation segment."""
    if not explanation:
        return ""
    normalized = " ".join(explanation.split()).strip()
    if not normalized:
        return ""
    segments = [segment.strip() for segment in normalized.split("|") if segment.strip()]
    for segment in segments:
        if segment.startswith("http://") or segment.startswith("https://"):
            continue
        lower = segment.lower()
        if lower.startswith("article:"):
            title = segment.split(":", 1)[1].strip()
            if title:
                return title
            continue
        return segment
    return ""


def _discovery_paragraphs(explanation: str | None) -> list[dict[str, str]]:
    """Render-ready explanation blocks with article title URL linking."""
    if not explanation:
        return []
    segments = [
        segment.strip()
        for segment in re.split(r"\s*\|\s*|\n+", explanation)
        if segment and segment.strip()
    ]
    if not segments:
        return []

    blocks: list[dict[str, str]] = []
    i = 0
    while i < len(segments):
        segment = segments[i]
        lower = segment.lower()
        if lower.startswith("article:") and i + 1 < len(segments):
            title = segment.split(":", 1)[1].strip()
            next_segment = segments[i + 1].strip()
            if title and (next_segment.startswith("http://") or next_segment.startswith("https://")):
                blocks.append({
                    "kind": "link",
                    "label": title,
                    "url": next_segment,
                })
                i += 2
                continue

        if segment.startswith("http://") or segment.startswith("https://"):
            blocks.append({
                "kind": "link",
                "label": segment,
                "url": segment,
            })
        else:
            display_text = segment
            if lower.startswith("article:"):
                display_text = segment.split(":", 1)[1].strip()
            blocks.append({
                "kind": "text",
                "text": display_text,
            })
        i += 1

    return blocks


def _discovery_outcome(discovery: Discovery, score: float | None) -> dict[str, object]:
    """Classify discovery quality from score, age, and mark-to-market return."""
    now = timezone.now()
    age = now - discovery.created if discovery.created else datetime.timedelta.max
    age_hours = age.total_seconds() / 3600
    discovery_price = discovery.price
    current_price = discovery.stock.price if discovery.stock else None

    # TOO EARLY means "no fresh price signal yet", not simply elapsed time.
    if age < datetime.timedelta(days=1):
        has_fresh_price_signal = (
            discovery_price is not None
            and current_price is not None
            and discovery_price > 0
            and current_price > 0
            and abs(float(current_price - discovery_price)) > 1e-9
        )
        if not has_fresh_price_signal:
            return {
                "label": "TOO EARLY",
                "state": "TOO_EARLY",
                "reason_code": "too_early_no_fresh_price",
                "reason_text": "Waiting for updated market prices before judging outcome.",
                "css_class": "outcome-too-early",
                "return_pct": None,
                "age_hours": round(age_hours, 2),
            }

    if score is None:
        return {
            "label": "UNKNOWN",
            "state": "UNKNOWN",
            "reason_code": "missing_health_score",
            "reason_text": "Health score is unavailable.",
            "css_class": "outcome-unknown",
            "return_pct": None,
            "age_hours": round(age_hours, 2),
        }
    if discovery_price is None or discovery_price <= 0:
        return {
            "label": "UNKNOWN",
            "state": "UNKNOWN",
            "reason_code": "missing_discovery_price",
            "reason_text": "Discovery price is missing.",
            "css_class": "outcome-unknown",
            "return_pct": None,
            "age_hours": round(age_hours, 2),
        }
    if current_price is None or current_price <= 0:
        return {
            "label": "UNKNOWN",
            "state": "UNKNOWN",
            "reason_code": "missing_current_price",
            "reason_text": "Current price is missing.",
            "css_class": "outcome-unknown",
            "return_pct": None,
            "age_hours": round(age_hours, 2),
        }

    return_pct = float((current_price - discovery_price) / discovery_price * 100)
    score_midpoint = 25.0

    # Missed opportunity: strong gain despite below-average score.
    if score < score_midpoint and return_pct >= 8.0:
        label = "DISASTER"
        reason_code = "low_score_high_return"
        css_class = "outcome-disaster"
        reason_text = "Below-average score missed a strong upside move."
    # Rule: otherwise flat or positive return is at least ADEQUATE.
    elif return_pct >= 0 and score >= score_midpoint and return_pct >= 8.0:
        label = "PERFECT"
        reason_code = "high_score_high_return"
        css_class = "outcome-perfect"
        reason_text = "High conviction and strong return aligned well."
    elif return_pct >= 0:
        label = "ADEQUATE"
        reason_code = "non_negative_return"
        css_class = "outcome-adequate"
        reason_text = "Return is flat or positive without a major score/return mismatch."
    elif score >= score_midpoint and return_pct <= -4.0:
        label = "DISASTER"
        reason_code = "high_score_low_return"
        css_class = "outcome-disaster"
        reason_text = "High conviction but the return moved sharply against the call."
    elif score >= score_midpoint:
        label = "DISAPPOINTING"
        reason_code = "high_score_subpar_return"
        css_class = "outcome-disappointing"
        reason_text = "High conviction has not produced the expected upside yet."
    elif return_pct <= -8.0:
        label = "DISAPPOINTING"
        reason_code = "deep_negative_return"
        css_class = "outcome-disappointing"
        reason_text = "Return is materially negative versus discovery price."
    else:
        label = "ADEQUATE"
        reason_code = "middle_band_alignment"
        css_class = "outcome-adequate"
        reason_text = "Score and return are broadly in line."

    return {
        "label": label,
        "state": "FINAL",
        "reason_code": reason_code,
        "reason_text": reason_text,
        "css_class": css_class,
        "return_pct": round(return_pct, 2),
        "age_hours": round(age_hours, 2),
    }
