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
from .models import Profile, Holding, Stock, Discovery, Trade, Consensus, Recommendation, SellInstruction


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
    confidence_low = Decimal(str(Profile.RISK[profile.risk]["confidence_low"]))

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
        .select_related('sa', 'consensus', 'consensus__stock')
        .filter(user=request.user, stock_id=stock_id)
        .order_by('-id')
    )

    sa_ids = {trade.sa_id for trade in trades_qs if trade.sa_id}

    discoveries_map = {}
    if sa_ids:
        for discovery in Discovery.objects.select_related('advisor', 'sa').filter(sa_id__in=sa_ids, stock_id=stock_id):
            discoveries_map[discovery.sa_id] = discovery

    consensus_map = {}
    if sa_ids:
        for consensus in Consensus.objects.filter(sa_id__in=sa_ids, stock_id=stock_id):
            consensus_map[consensus.sa_id] = consensus

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
                if instruction.instruction == 'STOP_LOSS' and instruction.value:
                    instruction_data['status'] = 'active' if current_price <= instruction.value else 'pending'
                    instruction_data['current_price'] = float(current_price)
                elif instruction.instruction == 'TARGET_PRICE' and instruction.value:
                    instruction_data['status'] = 'active' if current_price >= instruction.value else 'pending'
                    instruction_data['current_price'] = float(current_price)
                elif instruction.instruction == 'AFTER_DAYS' and instruction.value:
                    days_held = (timezone.now() - discovery.created).days
                    instruction_data['days_held'] = days_held
                    instruction_data['status'] = 'active' if days_held >= instruction.value else 'pending'
                elif instruction.instruction == 'CS_FLOOR':
                    # CS_FLOOR value is always None in DB, populate with user's risk profile confidence_low
                    instruction_data['value'] = float(confidence_low)
                    instruction_data['status'] = 'pending'  # CS_FLOOR is checked during analysis
                    instruction_data['current_consensus'] = float(holding.consensus) if holding.consensus else None
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

        consensus = trade.consensus or consensus_map.get(trade.sa_id)
        if consensus:
            trade_payload['consensus'] = {
                'id': consensus.id,
                'recommendations': consensus.recommendations,
                'avg_confidence': float(consensus.avg_confidence) if consensus.avg_confidence is not None else None,
                'tot_confidence': float(consensus.tot_confidence) if consensus.tot_confidence is not None else None,
                'sa_id': consensus.sa_id,
            }
        else:
            trade_payload['consensus'] = None

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
        })

    context = {
        'current_page': 'trades',
        'trades': trades_payload,
    }
    return render(request, 'core/trades.html', context)


@login_required
def trade_detail(request, trade_id):
    """Trade detail view - returns JSON for a single trade"""
    try:
        trade = (
            Trade.objects
            .select_related('stock', 'sa', 'consensus')
            .get(id=trade_id, user=request.user)
        )
    except Trade.DoesNotExist:
        raise Http404("Trade not found")

    stock = trade.stock
    shares = trade.shares or 0
    price = trade.price or Decimal('0')
    value = price * shares

    # Get discovery for this stock
    # For SELL trades, discovery might be from an earlier SA, so find the earliest discovery
    # Look for discovery in the trade's SA first, then fall back to earliest discovery for this stock
    discovery = None
    discovery_obj = None
    
    # First try to find discovery in the same SA as the trade
    if trade.sa_id:
        discovery_obj = Discovery.objects.select_related('advisor', 'sa').filter(
            sa_id=trade.sa_id,
            stock_id=stock.id
        ).first()
    
    # If not found, get the earliest discovery for this stock (for SELL trades from earlier SA)
    if not discovery_obj:
        discovery_obj = Discovery.objects.select_related('advisor', 'sa').filter(
            stock_id=stock.id
        ).order_by('sa__started', 'id').first()
    
    if discovery_obj:
        discovery = {
            'id': discovery_obj.id,
            'advisor': discovery_obj.advisor.name if discovery_obj.advisor else '',
            'advisor_logo': _advisor_logo_url(discovery_obj.advisor),
            'explanation': discovery_obj.explanation,
            'sa_id': discovery_obj.sa_id,
            'sa_started': discovery_obj.sa.started.isoformat() if discovery_obj.sa and discovery_obj.sa.started else None,
        }

    # Get consensus
    consensus = None
    if trade.consensus:
        consensus = {
            'id': trade.consensus.id,
            'recommendations': trade.consensus.recommendations,
            'avg_confidence': float(trade.consensus.avg_confidence) if trade.consensus.avg_confidence is not None else None,
            'tot_confidence': float(trade.consensus.tot_confidence) if trade.consensus.tot_confidence is not None else None,
            'sa_id': trade.consensus.sa_id,
        }
    elif trade.sa_id:
        # Try to find consensus for this SA
        consensus_obj = Consensus.objects.filter(sa_id=trade.sa_id, stock_id=stock.id).first()
        if consensus_obj:
            consensus = {
                'id': consensus_obj.id,
                'recommendations': consensus_obj.recommendations,
                'avg_confidence': float(consensus_obj.avg_confidence) if consensus_obj.avg_confidence is not None else None,
                'tot_confidence': float(consensus_obj.tot_confidence) if consensus_obj.tot_confidence is not None else None,
                'sa_id': consensus_obj.sa_id,
            }

    # Get recommendations
    recommendations = []
    if trade.sa_id:
        for recommendation in Recommendation.objects.select_related('advisor').filter(
            sa_id=trade.sa_id,
            stock_id=stock.id
        ):
            recommendations.append({
                'id': recommendation.id,
                'advisor': recommendation.advisor.name if recommendation.advisor else '',
                'advisor_logo': _advisor_logo_url(recommendation.advisor),
                'confidence': float(recommendation.confidence) if recommendation.confidence is not None else None,
                'explanation': recommendation.explanation,
            })

    payload = {
        'trade': {
            'id': trade.id,
            'action': trade.action,
            'shares': shares,
            'price': float(price),
            'value': float(value),
            'sa_id': trade.sa_id,
            'sa_started': trade.sa.started.isoformat() if trade.sa and trade.sa.started else None,
            'created': trade.created.isoformat() if trade.created else None,
            'explanation': trade.explanation,
        },
        'stock': {
            'id': stock.id,
            'symbol': stock.symbol,
            'company': stock.company,
            'price': float(stock.price) if stock.price else 0.0,
        },
        'discovery': discovery,
        'consensus': consensus,
        'recommendations': recommendations,
    }

    return JsonResponse(payload)


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
