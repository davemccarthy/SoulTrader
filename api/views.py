from datetime import timedelta

from django.db.models import Sum
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .serializers import HoldingSerializer, TradeSerializer, ProfileSerializer, UserSerializer
from core.models import Profile, Holding, Trade, Snapshot
from core.portfolio_metrics import get_portfolio_dashboard_data


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
        # Match web behavior: holdings are scoped by selected fund.
        holdings = Holding.objects.filter(fund_id=fund_id)
    else:
        holdings = request.user.holding_set.all()
    serializer = HoldingSerializer(holdings, many=True)
    return Response(serializer.data)


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

