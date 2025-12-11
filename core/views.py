from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.db.models import F, Q, Case, When, Value, CharField, Max
from django.http import JsonResponse, Http404
from django.templatetags.static import static
from decimal import Decimal
from collections import defaultdict
from .models import Profile, Holding, Discovery, Trade, Recommendation, SellInstruction, Health


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
    holdings_list = (
        Holding.objects
        .filter(user=request.user)
        .annotate(
            latest_trade_sa=Max(
                'stock__trade__sa_id',
                filter=Q(stock__trade__user=request.user)
            )
        )
        .order_by('-latest_trade_sa', '-id')
        .select_related('stock', 'stock__advisor')
    )
    
    # Get all SA IDs for discoveries lookup
    sa_ids = {h.latest_trade_sa for h in holdings_list if h.latest_trade_sa}
    
    # Prefetch discoveries for all holdings
    discoveries_map = {}
    if sa_ids:
        discoveries = Discovery.objects.select_related('advisor').filter(
            sa_id__in=sa_ids,
            stock_id__in=[h.stock_id for h in holdings_list]
        )
        for discovery in discoveries:
            key = (discovery.stock_id, discovery.sa_id)
            if key not in discoveries_map:
                discoveries_map[key] = discovery
    
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
        
        # Get advisor name for discovery - use Discovery if available, fallback to stock.advisor
        discovery_obj = None
        if holding.latest_trade_sa:
            discovery_obj = discoveries_map.get((stock.id, holding.latest_trade_sa))
        
        if discovery_obj:
            discovery = discovery_obj.advisor.name if discovery_obj.advisor else ""
            discovery_advisor = discovery_obj.advisor
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
        })
    
    context = {
        'current_page': 'holdings',
        'holdings': holdings_data,
    }
    return render(request, 'core/holdings.html', context)


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
    
    try:
        holding = (
            Holding.objects
            .select_related('stock')
            .get(user=request.user, stock_id=stock_id)
        )
    except Holding.DoesNotExist:
        raise Http404("Holding not found")

    # Get user's profile for risk-based values
    profile = Profile.objects.get(user=request.user)
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
        .filter(user=request.user, stock_id=stock_id)
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
            sell_instructions = []
            for instruction in instructions:
                instruction_data = {
                    'instruction': instruction.instruction,
                    'value': float(instruction.value) if instruction.value is not None else None,
                }
                
                # Add status/context for each instruction type
                if instruction.instruction in ['STOP_LOSS', 'STOP_PERCENTAGE', 'STOP_PRICE'] and instruction.value:
                    instruction_data['status'] = 'active' if current_price <= instruction.value else 'pending'
                    instruction_data['current_price'] = float(current_price)
                elif instruction.instruction in ['TARGET_PRICE', 'TARGET_PERCENTAGE'] and instruction.value:
                    instruction_data['status'] = 'active' if current_price >= instruction.value else 'pending'
                    instruction_data['current_price'] = float(current_price)
                elif instruction.instruction == 'AFTER_DAYS' and instruction.value:
                    days_held = (timezone.now() - discovery.created).days
                    instruction_data['days_held'] = days_held
                    instruction_data['status'] = 'active' if days_held >= instruction.value else 'pending'
                elif instruction.instruction == 'DESCENDING_TREND' and instruction.value is not None:
                    # DESCENDING_TREND value is a threshold (e.g., -0.20 means sell if trend < -0.20)
                    instruction_data['value'] = float(instruction.value)
                    instruction_data['status'] = 'pending'  # Status determined during analysis

                    trend = holding.stock.calc_trend()

                    if trend is not None:
                        instruction_data['current_trend'] = float(trend)
                        instruction_data['status'] = 'active' if trend < instruction.value else 'pending'
                
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
    
    try:
        holding = (
            Holding.objects
            .select_related('stock')
            .get(user=request.user, stock_id=stock_id)
        )
    except Holding.DoesNotExist:
        raise Http404("Holding not found")

    stock = holding.stock
    shares = holding.shares or 0
    avg_price = holding.average_price or Decimal('0')
    current_price = stock.price or Decimal('0')
    worth = current_price * shares

    # Calculate change (from average price, like holdings view)
    stock.refresh()
    change_percent = 0.0
    if avg_price and avg_price > 0:
        change_percent = float(((current_price / avg_price) * 100 - 100))

    # Heading data
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
    }

    # Discovery data (most recent)
    discovery = None
    sell_instructions = []
    discovery_obj = (
        Discovery.objects
        .select_related('advisor', 'sa')
        .filter(stock=stock)
        .order_by('-created')
        .first()
    )
    
    if discovery_obj:
        discovery = {
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
                            discovery['url'] = potential_url
                            break
        
        # Get sell instructions for this discovery
        profile = Profile.objects.get(user=request.user)
        instructions = SellInstruction.objects.filter(discovery=discovery_obj).order_by('id')
        # Create mapping from instruction code to description
        instruction_choices = dict(SellInstruction.choices)
        
        for instruction in instructions:
            instruction_data = {
                'instruction': instruction.instruction,
                'description': instruction_choices.get(instruction.instruction, instruction.instruction),
                'value': float(instruction.value) if instruction.value is not None else None,
            }
            
            # Add status/context for each instruction type
            if instruction.instruction in ['STOP_PRICE', 'STOP_PERCENTAGE'] and instruction.value:
                instruction_data['status'] = 'active' if current_price <= instruction.value else 'pending'
                instruction_data['current_price'] = float(current_price)
            elif instruction.instruction in ['TARGET_PRICE', 'TARGET_PERCENTAGE'] and instruction.value:
                instruction_data['status'] = 'active' if current_price >= instruction.value else 'pending'
                instruction_data['current_price'] = float(current_price)
            elif instruction.instruction == 'AFTER_DAYS' and instruction.value:
                days_held = (timezone.now() - discovery_obj.created).days
                instruction_data['days_held'] = days_held
                instruction_data['status'] = 'active' if days_held >= instruction.value else 'pending'
            elif instruction.instruction == 'DESCENDING_TREND' and instruction.value is not None:
                instruction_data['value'] = float(instruction.value)
                trend = stock.calc_trend()
                if trend is not None:
                    instruction_data['current_trend'] = float(trend)
                    instruction_data['status'] = 'active' if trend < instruction.value else 'pending'
                else:
                    instruction_data['status'] = 'pending'
            
            sell_instructions.append(instruction_data)

    # Health history (all health checks for this stock)
    health_history = []
    health_checks = (
        Health.objects
        .select_related('sa')
        .filter(stock=stock)
        .order_by('-created')
    )
    
    for health in health_checks:
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
        .filter(user=request.user, stock_id=stock_id)
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
        
        trade_data = {
            'id': trade.id,
            'action': trade.action,
            'price': float(trade.price) if trade.price is not None else None,
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
        'discovery': discovery,
        'health_history': health_history,
        'trades': trades_data,
        'sell_instructions': sell_instructions,
    }

    return JsonResponse(payload)


@login_required
def trades(request):
    """Trades page - summarize executed trades using holdings styling."""
    trade_qs = (
        Trade.objects
        .filter(user=request.user)
        .select_related('stock', 'sa')
        .order_by('sa__started', 'id')
    )

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

        if realized:
            if pl_amount is not None and pl_amount > 0:
                pl_class = 'positive'
            elif pl_amount is not None and pl_amount < 0:
                pl_class = 'negative'
            else:
                pl_class = 'neutral'
            price_class = pl_class
        else:
            pl_class = 'muted'
            price_class = 'neutral'

        trades_payload.append({
            'id': trade.id,
            'stock_id': trade.stock_id,
            'symbol': trade.stock.symbol,
            'company': trade.stock.company,
            'shares': trade.shares,
            'price': trade.price,
            'sa_id': trade.sa_id,
            'timestamp': trade.sa.started if trade.sa else None,
            'action': trade.action,
            'cost': data.get('cost', Decimal('0')),
            'pl_amount': pl_amount,
            'pl_percent': data.get('pl_percent'),
            'pl_class': pl_class,
            'price_class': price_class,
            'realized': realized,
            'explanation': trade.explanation,
        })

    context = {
        'current_page': 'trades',
        'trades': trades_payload,
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
