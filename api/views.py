import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime as dt, timedelta
from decimal import Decimal
from typing import Any, Dict

from django.conf import settings
from django.db.models import Count, F, Max, Sum
from django.templatetags.static import static
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .serializers import HoldingSerializer, TradeSerializer, ProfileSerializer, UserSerializer
from django.http import Http404
from django.shortcuts import get_object_or_404

from core.models import Advisor, Discovery, Holding, Profile, Snapshot, Trade
from core.portfolio_metrics import get_portfolio_dashboard_data

logger = logging.getLogger(__name__)

# Mobile/API trades list: newest-first slice (avoid unbounded responses).
_TRADES_LIST_LIMIT = 250


def _compose_fund_description_for_api(fund: Profile, dashboard: Dict[str, Any]) -> str:
    """
    Automated intro (risk, spread, sentiment, age, equity share) + optional admin free text
    from Profile.description.
    """
    name = (fund.name or "").strip() or "This fund"

    raw_advisors = [a for a in (fund.advisors or []) if a and str(a).strip()]
    advisor_kind = "multi-advisor" if len(raw_advisors) >= 2 else "single-advisor"

    risk_label = (fund.risk or "MODERATE").replace("_", " ").lower()
    spread_key = fund.spread or "MEDIUM"
    spread_label = str(spread_key).replace("_", " ").lower()

    sentiment_labels = {
        "STRONG_BULL": "strong bull",
        "BULL": "bull",
        "STAG": "neutral",
        "AUTO": "auto",
        "BEAR": "bear",
        "STRONG_BEAR": "strong bear",
    }
    sentiment_label = sentiment_labels.get(
        fund.sentiment,
        (fund.sentiment or "AUTO").replace("_", " ").lower(),
    )

    age_days = max(int(dashboard.get("estab_days") or 1), 1)

    total_value = Decimal(str(dashboard.get("total_value") or "0"))
    cash_dec = Decimal(str(dashboard.get("cash") or "0"))
    if total_value > 0:
        holdings_mv = total_value - cash_dec
        if holdings_mv < 0:
            holdings_mv = Decimal("0")
        equity_pct = float((holdings_mv / total_value) * Decimal("100"))
    else:
        equity_pct = 0.0

    intro = (
        f"{name} is a {advisor_kind} fund with {risk_label} risk, {spread_label} spread "
        f"and bull–bear sentiment of {sentiment_label}. The fund is {age_days} days old "
        f"with equity representing {equity_pct:.1f}% of total portfolio."
    )

    body = (fund.description or "").strip()
    if body:
        return f"{intro}\n\n{body}"
    return intro

_ALLOWED_YF_PERIODS = frozenset({
    '1d', '5d', '1mo', '2mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max',
})
_ALLOWED_YF_INTERVALS = frozenset({
    '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo',
})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_current_user(request):
    """Get current user info"""
    serializer = UserSerializer(request.user)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_holdings(request):
    """Get user holdings"""
    fund_id = request.query_params.get('fund_id')
    if fund_id:
        holdings_qs = Holding.objects.filter(fund_id=fund_id)
    else:
        holdings_qs = request.user.holding_set.all()

    # Newest holdings first: created (when set), then id. nulls_last so legacy rows
    # without created don't float above recent ones on PostgreSQL.
    holdings_qs = holdings_qs.order_by(
        F('created').desc(nulls_last=True),
        '-id',
    ).select_related('stock', 'stock__advisor', 'discovery__advisor')
    holdings = list(holdings_qs)
    serializer = HoldingSerializer(holdings, many=True)
    payload = serializer.data

    for item, holding in zip(payload, holdings):
        discovery = holding.discovery
        if discovery:
            item['discovery_name'] = discovery.advisor.name if discovery.advisor else ""
            item['discovery_logo'] = _advisor_logo_absolute_url(request, discovery.advisor)
            item['discovery_comment'] = _discovery_comment(discovery.explanation)
            item['discovery_explanation'] = discovery.explanation or ""
        else:
            fallback_advisor = getattr(holding.stock, 'advisor', None)
            item['discovery_name'] = fallback_advisor.name if fallback_advisor else ""
            item['discovery_logo'] = _advisor_logo_absolute_url(request, fallback_advisor)
            item['discovery_comment'] = None
            item['discovery_explanation'] = ""

    return Response(payload)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_holding_health_history(request, stock_id: int):
    """
    Health checks for a single holding's stock, scoped by fund (matches web holding_history).
    Query: fund_id (required when using fund-scoped holdings).
    """
    fund_id = request.query_params.get('fund_id')
    if not fund_id:
        return Response({'error': 'fund_id is required.'}, status=400)

    # Match get_holdings when fund_id is set: scope by fund only, not user.
    # (Many deployments use fund-scoped rows where user may differ from JWT user.)
    if not Holding.objects.filter(fund_id=fund_id, stock_id=stock_id).exists():
        raise Http404('Holding not found')

    health_history = []
    seen_health_ids = set()
    discovery_health_checks = (
        Discovery.objects
        .select_related('health', 'health__sa')
        .filter(stock_id=stock_id, health__isnull=False)
        .order_by('-health__created')
    )

    for disc_health in discovery_health_checks:
        health = getattr(disc_health, 'health', None)
        if not health or health.id in seen_health_ids:
            continue
        seen_health_ids.add(health.id)

        health_history.append(_health_record_payload(health))

    return Response({'health_history': health_history})


def _health_record_payload(health):
    """JSON shape aligned with `HealthHistoryRecord` on iOS (holding health_history)."""
    meta = health.meta or {}
    overlay = meta.get('overlay') if isinstance(meta.get('overlay'), dict) else {}
    return {
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
        'overlay_points': overlay.get('points'),
        'overlay_reasons': overlay.get('reasons') if isinstance(overlay.get('reasons'), list) else [],
    }


def _advisor_logo_url(advisor):
    if not advisor:
        return None
    python_class = getattr(advisor, 'python_class', None)
    if not python_class:
        return None
    return static(f"core/advisors/{python_class.lower()}.png")


def _advisor_logo_absolute_url(request, advisor):
    """Public URL for PNG under STATIC_URL; safe behind reverse proxies when PUBLIC_BASE_URL or proxy headers are set."""
    path = _advisor_logo_url(advisor)
    if not path:
        return None
    base = getattr(settings, 'PUBLIC_BASE_URL', '') or ''
    base = base.strip().rstrip('/')
    if base:
        return f'{base}{path}'
    return request.build_absolute_uri(path)


def _discovery_comment(explanation: str | None):
    if not explanation:
        return None
    normalized = " ".join(explanation.split()).strip()
    if not normalized:
        return None
    segments = [segment.strip() for segment in normalized.split("|") if segment.strip()]
    for segment in segments:
        lower = segment.lower()
        if segment.startswith("http://") or segment.startswith("https://"):
            continue
        if lower.startswith("article:"):
            title = segment.split(":", 1)[1].strip()
            if title:
                return title
            continue
        return segment
    return None


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_trades(request):
    """Get user trades (read-only - system is 100% automated)"""
    fund_id = request.query_params.get('fund_id')
    if fund_id:
        # Match web behavior: trades are scoped by selected fund.
        trades = Trade.objects.filter(fund_id=fund_id)
    else:
        trades = request.user.trade_set.all()
    trades = trades.order_by('-id')[:_TRADES_LIST_LIMIT]
    serializer = TradeSerializer(trades, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_funds(request):
    """Get enabled funds with dashboard metrics (matches web Funds page intent)."""
    funds = Profile.objects.filter(enabled=True).order_by('id')
    payload = []

    for fund in funds:
        dashboard = get_portfolio_dashboard_data(fund)
        payload.append({
            'id': fund.id,
            'name': fund.name,
            'description': _compose_fund_description_for_api(fund, dashboard),
            'spread': fund.spread,
            'risk': fund.risk,
            'advisors': fund.advisors,
            'dashboard': dashboard,
        })

    response = Response(payload)
    # Discourage any intermediary from caching fund JSON (schema evolves; iOS must see fresh fields).
    response['Cache-Control'] = 'private, no-store'
    return response


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_fund_advisors(request):
    """
    Advisors listed on a fund's Profile.advisors (ordered).
    Omits python_class entries with no Advisor row.
    Discovery count uses lookback filter (default 30d) for consistency with advisory scoreboard/discoveries.
    GET /api/funds/advisors/?fund_id=26&days=30
    """
    fund_id = request.query_params.get('fund_id')
    if not fund_id:
        return Response({'error': 'fund_id is required.'}, status=400)
    try:
        days = int(request.query_params.get('days', '30'))
    except ValueError:
        days = 30
    days = max(7, min(days, 365))
    cutoff = timezone.now() - timedelta(days=days)
    fund = get_object_or_404(Profile, pk=int(fund_id), enabled=True)

    pcs_ordered = list(fund.advisors or [])
    by_pc = {
        a.python_class: a
        for a in Advisor.objects.filter(python_class__in=pcs_ordered)
    }
    ordered_advisors = [by_pc[pc] for pc in pcs_ordered if pc in by_pc]
    advisor_ids = [a.id for a in ordered_advisors]

    counts = {}
    if advisor_ids:
        counts = {
            row['advisor_id']: row['n']
            for row in Discovery.objects.filter(advisor_id__in=advisor_ids, created__gte=cutoff)
            .values('advisor_id')
            .annotate(n=Count('id'))
        }

    advisors_out = []
    for adv in ordered_advisors:
        advisors_out.append({
            'id': adv.id,
            'python_class': adv.python_class,
            'name': adv.name,
            'description': adv.description or '',
            'discovery_count': counts.get(adv.id, 0),
            'image_url': _advisor_logo_absolute_url(request, adv),
        })

    return Response({'fund_id': fund.id, 'days': days, 'advisors': advisors_out})


@dataclass
class _ApiLotOutcome:
    advisor: str
    cost: Decimal
    exit_value: Decimal
    winner: bool


def _scoreboard_cutoff_lookback(days: int):
    now = timezone.now()
    period_date = now.date() - timedelta(days=days)
    midnight = dt.combine(period_date, dt.min.time())
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
) -> list[_ApiLotOutcome]:
    trades = (
        Trade.objects.filter(user_id=user_id, stock_id=stock_id, action__in=("BUY", "SELL"))
        .select_related("sa", "stock")
        .order_by("created")
    )
    queue: list[tuple[int, int, Decimal]] = []
    sold_from: dict[int, dict[str, Any]] = defaultdict(lambda: {"shares": 0, "proceeds": Decimal("0")})
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
    outcomes: list[_ApiLotOutcome] = []
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
            _ApiLotOutcome(
                advisor=advisor,
                cost=cost,
                exit_value=exit_value,
                winner=(exit_value >= cost),
            )
        )
    return outcomes


def _fund_advisor_scoreboard_rows(fund_id: int, days: int) -> list[dict[str, Any]]:
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

    by_user_stock: dict[tuple[int, int], list[Trade]] = defaultdict(list)
    for t in buy_list:
        by_user_stock[(t.user_id, t.stock_id)].append(t)

    outcomes: list[_ApiLotOutcome] = []
    for (user_id, stock_id), _ in by_user_stock.items():
        outcomes.extend(_scoreboard_fifo_outcomes_for_user_stock(user_id, stock_id, cutoff, advisor_for_buy))
    if not outcomes:
        return []

    by_name: dict[str, list[_ApiLotOutcome]] = defaultdict(list)
    for o in outcomes:
        by_name[o.advisor].append(o)

    id_by_name = {
        a.name: a.id
        for a in Advisor.objects.filter(name__in=list(by_name.keys())).only("id", "name")
    }
    rows: list[dict[str, Any]] = []
    for name in sorted(by_name.keys()):
        advisor_id = id_by_name.get(name)
        if advisor_id is None:
            continue
        advisor_outcomes = by_name[name]
        n = len(advisor_outcomes)
        winners = sum(1 for o in advisor_outcomes if o.winner)
        losers = n - winners
        win_rate = (winners / n * 100) if n else 0.0
        total_cost = sum(o.cost for o in advisor_outcomes)
        total_exit = sum(o.exit_value for o in advisor_outcomes)
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


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_fund_advisor_scoreboard(request):
    """
    FIFO-attributed BUY lot stats per advisor for one fund (fund-scoped trades).
    Uses DB stock prices only (no live refresh) to keep the endpoint responsive.

    GET /api/funds/advisors/scoreboard/?fund_id=26&days=30
    """
    fund_id = request.query_params.get('fund_id')
    if not fund_id:
        return Response({'error': 'fund_id is required.'}, status=400)
    try:
        days = int(request.query_params.get('days', '30'))
    except ValueError:
        days = 30
    days = max(7, min(days, 365))
    fund = get_object_or_404(Profile, pk=int(fund_id), enabled=True)
    rows = _fund_advisor_scoreboard_rows(fund.id, days)
    return Response({'fund_id': fund.id, 'days': days, 'advisors': rows})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_advisor_discoveries(request, advisor_id: int):
    """
    Discoveries for an advisor (newest first), lookback-filtered by created date.
    Not fund-scoped.
    GET /api/advisors/<id>/discoveries/?days=30
    """
    get_object_or_404(Advisor, pk=advisor_id)
    try:
        days = int(request.query_params.get('days', '30'))
    except ValueError:
        days = 30
    days = max(7, min(days, 365))
    cutoff = timezone.now() - timedelta(days=days)
    discoveries = (
        Discovery.objects.filter(advisor_id=advisor_id, created__gte=cutoff)
        .select_related('stock', 'health')
        .order_by('-id')[:250]
    )
    out = []
    for d in discoveries:
        stock = d.stock
        out.append({
            'id': d.id,
            'stock': {
                'symbol': stock.symbol,
                'company': (stock.company or '').strip(),
                'price': str(stock.price),
            },
            'explanation_line': _discovery_comment(d.explanation) or '',
            'health_score': float(d.health.score) if d.health else None,
        })
    return Response({'advisor_id': advisor_id, 'days': days, 'discoveries': out})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_discovery_detail(request, discovery_id: int):
    """
    Single discovery: stock snapshot, full explanation, advisor, linked health (if any).
    GET /api/discoveries/<id>/
    """
    d = get_object_or_404(
        Discovery.objects.select_related('stock', 'advisor', 'health'),
        pk=discovery_id,
    )
    stock = d.stock
    health_payload = _health_record_payload(d.health) if d.health else None

    return Response({
        'id': d.id,
        'explanation': d.explanation or '',
        'discovery_price': str(d.price) if d.price is not None else None,
        'created': d.created.isoformat() if d.created else None,
        'stock': {
            'symbol': stock.symbol,
            'company': (stock.company or '').strip(),
            'industry': (stock.industry or '').strip(),
            'sector': (stock.sector or '').strip(),
            'exchange': (stock.exchange or '').strip(),
            'price': str(stock.price),
        },
        'advisor': {
            'id': d.advisor_id,
            'name': d.advisor.name if d.advisor else '',
            'logo_url': _advisor_logo_absolute_url(request, d.advisor) if d.advisor else None,
        },
        'health': health_payload,
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_dashboard(request):
    """Get aggregate dashboard metrics across all enabled funds."""
    return Response(get_portfolio_dashboard_data(0))


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_dashboard_history(request):
    """
    Snapshot wealth series for charting.
    - global scope: /api/dashboard/history/?days=90
    - fund scope:   /api/dashboard/history/?fund_id=26&days=90
    """
    days = int(request.query_params.get('days', 90) or 90)
    days = max(7, min(days, 3650))
    start_date = timezone.localdate() - timedelta(days=days - 1)
    fund_id = request.query_params.get('fund_id')

    if fund_id:
        snapshots = Snapshot.objects.filter(
            fund_id=fund_id,
            date__gte=start_date,
        ).order_by('date')
        points = [
            {
                'date': snap.date.isoformat(),
                'wealth': float((snap.cash_value or 0) + (snap.holdings_value or 0)),
                'cash': float(snap.cash_value or 0),
                'holdings': float(snap.holdings_value or 0),
            }
            for snap in snapshots
        ]
    else:
        date_rows = (
            Snapshot.objects
            .filter(fund__enabled=True, date__gte=start_date)
            .values('date')
            .annotate(
                cash=Sum('cash_value'),
                holdings=Sum('holdings_value'),
            )
            .order_by('date')
        )
        points = [
            {
                'date': row['date'].isoformat(),
                'wealth': float((row['cash'] or 0) + (row['holdings'] or 0)),
                'cash': float(row['cash'] or 0),
                'holdings': float(row['holdings'] or 0),
            }
            for row in date_rows
        ]

    change_percent = 0.0
    if len(points) >= 2:
        first = points[0]['wealth']
        last = points[-1]['wealth']
        if first > 0:
            change_percent = ((last - first) / first) * 100.0

    return Response({
        'points': points,
        'change_percent': change_percent,
        'days': days,
        'scope': 'fund' if fund_id else 'global',
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_stock_price_history(request):
    """
    Daily close history for a symbol via yfinance (same approach as web holding_history).
    GET /api/stocks/price_history/?symbol=AAPL&period=2mo&interval=1d
    """
    symbol = (request.query_params.get('symbol') or '').strip().upper()
    if not symbol or len(symbol) > 20:
        return Response({'error': 'valid symbol is required'}, status=400)

    period = (request.query_params.get('period') or '2mo').strip().lower()
    interval = (request.query_params.get('interval') or '1d').strip().lower()
    if period not in _ALLOWED_YF_PERIODS:
        period = '2mo'
    if interval not in _ALLOWED_YF_INTERVALS:
        interval = '1d'

    points = []
    try:
        import yfinance as yf

        hist = yf.Ticker(symbol).history(period=period, interval=interval)
        if not hist.empty and 'Close' in hist.columns:
            for ts, row in hist.iterrows():
                close = row.get('Close')
                if close is None:
                    continue
                try:
                    close_value = float(close)
                except (TypeError, ValueError):
                    continue
                try:
                    day = ts.date().isoformat()
                except AttributeError:
                    continue
                points.append({'date': day, 'close': close_value})
    except Exception as e:
        logger.warning('stock price history failed for %s: %s', symbol, e)

    return Response({
        'symbol': symbol,
        'period': period,
        'interval': interval,
        'points': points,
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_profile(request):
    """Get user profile"""
    from core.models import Profile
    user_profile = request.user.profile_set.first()
    if not user_profile:
        from decimal import Decimal
        user_profile = Profile.objects.create(
            user=request.user,
            risk='MODERATE',
            cash=Decimal('100000.00'),
            investment=Decimal('100000.00')
        )
    serializer = ProfileSerializer(user_profile)
    return Response(serializer.data)

