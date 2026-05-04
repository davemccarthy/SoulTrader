import logging
from datetime import timedelta

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
    ).select_related('stock', 'stock__advisor')
    holdings = list(holdings_qs)
    serializer = HoldingSerializer(holdings, many=True)
    payload = serializer.data

    stock_ids = [holding.stock_id for holding in holdings]
    latest_sa_by_stock = {}
    if stock_ids:
        trade_qs = Trade.objects.filter(stock_id__in=stock_ids)
        if fund_id:
            trade_qs = trade_qs.filter(fund_id=fund_id)
        else:
            trade_qs = trade_qs.filter(user=request.user)
        latest_sa_rows = trade_qs.values('stock_id').annotate(latest_sa=Max('sa_id'))
        latest_sa_by_stock = {row['stock_id']: row['latest_sa'] for row in latest_sa_rows if row['latest_sa']}

    discovery_map = {}
    if latest_sa_by_stock:
        sa_ids = list(set(latest_sa_by_stock.values()))
        discoveries = (
            Discovery.objects
            .select_related('advisor')
            .filter(stock_id__in=stock_ids, sa_id__in=sa_ids)
        )
        for discovery in discoveries:
            key = (discovery.stock_id, discovery.sa_id)
            if key not in discovery_map:
                discovery_map[key] = discovery

    for item, holding in zip(payload, holdings):
        latest_sa = latest_sa_by_stock.get(holding.stock_id)
        discovery = discovery_map.get((holding.stock_id, latest_sa)) if latest_sa else None
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
    trades = trades.order_by('-id')[:50]
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
            'spread': fund.spread,
            'risk': fund.risk,
            'advisors': fund.advisors,
            'dashboard': dashboard,
        })

    return Response(payload)


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


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_fund_advisor_scoreboard(request):
    """
    FIFO-attributed BUY lot stats per advisor for one fund (fund-scoped trades).
    Uses DB stock prices only (no live refresh) to keep the endpoint responsive.

    GET /api/funds/advisors/scoreboard/?fund_id=26&days=30
    """
    from core.services.discovery_scoreboard import fund_advisor_scoreboard_rows

    fund_id = request.query_params.get('fund_id')
    if not fund_id:
        return Response({'error': 'fund_id is required.'}, status=400)
    try:
        days = int(request.query_params.get('days', '30'))
    except ValueError:
        days = 30
    days = max(7, min(days, 365))
    fund = get_object_or_404(Profile, pk=int(fund_id), enabled=True)
    rows = fund_advisor_scoreboard_rows(fund.id, days, refresh_prices=False)
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

