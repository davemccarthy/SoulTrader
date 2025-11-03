from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from . import views

app_name = 'api'

urlpatterns = [
    # Authentication
    path('auth/login/', TokenObtainPairView.as_view(), name='token_obtain'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/user/', views.get_current_user, name='current_user'),
    
    # Holdings
    path('holdings/', views.get_holdings, name='holdings'),
    
    # Trades (read-only - system is 100% automated)
    path('trades/', views.get_trades, name='trades'),
    
    # Profile
    path('profile/', views.get_profile, name='profile'),
]

