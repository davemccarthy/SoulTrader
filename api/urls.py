from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from . import views

app_name = 'api'

urlpatterns = [
    # Authentication
    path('auth/login/', TokenObtainPairView.as_view(), name='token_obtain'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/user/', views.get_current_user, name='current_user'),

    # Funds
    path('funds/', views.get_funds, name='funds'),
    path('dashboard/', views.get_dashboard, name='dashboard'),
    path('dashboard/history/', views.get_dashboard_history, name='dashboard_history'),

    # Stock chart (yfinance; matches web holding_history price series)
    path('stocks/price_history/', views.get_stock_price_history, name='stock_price_history'),

    # Holdings
    path('holdings/', views.get_holdings, name='holdings'),
    path('holdings/<int:stock_id>/health_history/', views.get_holding_health_history, name='holding_health_history'),
    
    # Trades (read-only - system is 100% automated)
    path('trades/', views.get_trades, name='trades'),
    
    # Profile
    path('profile/', views.get_profile, name='profile'),
]

