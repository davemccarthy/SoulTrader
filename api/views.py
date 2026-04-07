from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .serializers import HoldingSerializer, TradeSerializer, ProfileSerializer, UserSerializer
from core.models import Profile
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
    holdings = request.user.holding_set.all()
    serializer = HoldingSerializer(holdings, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_trades(request):
    """Get user trades (read-only - system is 100% automated)"""
    trades = request.user.trade_set.all().order_by('-id')[:50]
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

