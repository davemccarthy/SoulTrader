from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register, name='register'),
    path('holdings/', views.holdings, name='holdings'),
    path('trades/', views.trades, name='trades'),
    path('profile/', views.profile, name='profile'),
]

