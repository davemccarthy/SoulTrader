from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.db.models import F, Q, Case, When, Value, CharField
from decimal import Decimal
from .models import Profile, Holding, Stock


def home(request):
    """Landing page"""
    context = {'current_page': 'home'}
    return render(request, 'core/home.html', context)


def login_view(request):
    """Login page"""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('core:holdings')
        else:
            messages.error(request, 'Invalid credentials')
    return render(request, 'core/login.html', {'current_page': 'login'})


def logout_view(request):
    """Logout"""
    logout(request)
    return redirect('core:home')


def register(request):
    """Registration page"""
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        risk = request.POST.get('risk', 'MODERATE')
        
        if password1 == password2:
            if not User.objects.filter(username=username).exists():
                user = User.objects.create_user(username=username, email=email, password=password1)
                # Profile auto-created via signal
                messages.success(request, 'Account created successfully!')
                login(request, user)
                return redirect('core:holdings')
            else:
                messages.error(request, 'Username already exists')
        else:
            messages.error(request, 'Passwords do not match')
    
    return render(request, 'core/register.html', {'current_page': 'register'})


@login_required
def holdings(request):
    """Holdings page - displays user's stock holdings"""
    # Query holdings with related stock and advisor data
    holdings_list = Holding.objects.filter(user=request.user).select_related('stock', 'stock__advisor')
    
    # Annotate with calculated fields
    holdings_data = []
    for holding in holdings_list:
        stock = holding.stock
        current_price = stock.price
        avg_price = holding.average_price
        
        # Calculate change percentage
        if avg_price and avg_price > 0:
            change_percent = ((current_price / avg_price) * 100 - 100)
        else:
            change_percent = Decimal('0')
        
        # Calculate P&L
        pl = (current_price * holding.shares) - (avg_price * holding.shares)
        
        # Get advisor name for discovery
        discovery = stock.advisor.name if stock.advisor else ""
        
        holdings_data.append({
            'image': stock.image,
            'symbol': stock.symbol,
            'company': stock.company,
            'price': current_price,
            'change_percent': change_percent,
            'pl': pl,
            'discovery': discovery,
            'holding': holding,  # Keep reference for potential future use
        })
    
    context = {
        'current_page': 'holdings',
        'holdings': holdings_data,
    }
    return render(request, 'core/holdings.html', context)


@login_required
def trades(request):
    """Trades page - content to be designed later"""
    context = {
        'current_page': 'trades',
    }
    return render(request, 'core/trades.html', context)


@login_required
def profile(request):
    """Profile page"""
    user_profile = request.user.profile_set.first()
    if not user_profile:
        user_profile = Profile.objects.create(
            user=request.user,
            risk='MODERATE',
            cash=100000.00,
            investment=100000.00
        )
    context = {
        'current_page': 'profile',
        'profile': user_profile,
    }
    return render(request, 'core/profile.html', context)
