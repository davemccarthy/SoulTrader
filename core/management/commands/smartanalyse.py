"""
Smart Analysis Management Command

Usage:
    python manage.py smartanalyse <username>
    python manage.py smartanalyse --all
"""
from operator import attrgetter
from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Sum, F, DecimalField

from core.services import analysis
from core.services.advisors.advisor import AdvisorBase
from core.models import SmartAnalysis, Advisor, Profile, Snapshot, Holding, Trade
from django.contrib.auth.models import User
from django.utils import timezone
from decimal import Decimal

# Register advisors from here
from core.services.advisors import alpha
from core.services import advisors as advisor_modules

import logging



logger = logging.getLogger(__name__)


def calculate_trade_pnl_percentages(user, snapshot_date):
    """
    Calculate Trade P&L percentages for a user.
    
    Args:
        user: User instance
        snapshot_date: Date for the snapshot (used to determine yesterday for daily calculation)
    
    Returns:
        tuple: (trade_cumulative, trade_daily) as Decimal percentages
    """
    # Cumulative: 100 - (sum(cost*shares) * 100) / sum(price*shares)
    # For all SELL trades (all-time)
    cumulative_trades = Trade.objects.filter(
        user=user,
        action='SELL',
        cost__isnull=False
    ).aggregate(
        total_cost=Sum(F('cost') * F('shares'), output_field=DecimalField()),
        total_proceeds=Sum(F('price') * F('shares'), output_field=DecimalField())
    )
    
    total_cost = cumulative_trades['total_cost'] or Decimal('0')
    total_proceeds = cumulative_trades['total_proceeds'] or Decimal('0')
    
    if total_proceeds > 0:
        trade_cumulative = Decimal('100') - ((total_cost * Decimal('100')) / total_proceeds)
    else:
        trade_cumulative = Decimal('0.0')
    
    # Daily: 100 - (sum(cost*shares) * 100) / sum(price*shares)
    # For SELL trades from yesterday (previous day's trading)
    yesterday = snapshot_date - timedelta(days=1)
    daily_trades = Trade.objects.filter(
        user=user,
        action='SELL',
        cost__isnull=False,
        created__date=yesterday
    ).aggregate(
        daily_cost=Sum(F('cost') * F('shares'), output_field=DecimalField()),
        daily_proceeds=Sum(F('price') * F('shares'), output_field=DecimalField())
    )
    
    daily_cost = daily_trades['daily_cost'] or Decimal('0')
    daily_proceeds = daily_trades['daily_proceeds'] or Decimal('0')
    
    if daily_proceeds > 0:
        trade_daily = Decimal('100') - ((daily_cost * Decimal('100')) / daily_proceeds)
    else:
        trade_daily = Decimal('0.0')
    
    return trade_cumulative, trade_daily


class Command(BaseCommand):
    help = 'Run Smart Analysis for automated trading'
    
    def add_arguments(self, parser):
        parser.add_argument(
            'username',
            nargs='?',
            type=str,
            help='Username to analyze (optional)'
        )
        parser.add_argument(
            '--holdings-only',
            action='store_true',
            help='Only analyze holdings (skip discovery)'
        )
        parser.add_argument(
            '--discovery-only',
            action='store_true',
            help='Only discover new stocks (skip holdings)'
        )
        parser.add_argument(
            '--reuse',
            action='store_true',
            help='Reuse SA record to save on API calls'
        )
        parser.add_argument(
            '--discover',
            type=str,
            help='Force discovery for given stock symbol'
        )
        parser.add_argument(
            '--explanation',
            type=str,
            help='Explanation for discovered stock (used with --discover)'
        )
    
    def handle(self, *args, **options):

        # Params
        param_holdings_only = options['holdings_only']
        param_discovery_only = options['discovery_only']

        param_resuse = options['reuse']
        
        # TODO: Implement based on your analysis.py smart_analyse() function
        # 1. Create SmartAnalysis session
        # 2. Get users to analyze
        # 3. Run analyse_holdings() if not discovery-only
        # 4. Run analyse_discovery() if not holdings-only
        # 5. Display results

        # Get users
        users = []

        if options['username']:
            try:
                user = User.objects.get(username=options['username'])
                users.append(user)
            except User.DoesNotExist:
                raise CommandError(f'User "{options["username"]}" does not exist')
        else:
            users = list(User.objects.filter(is_active=True, is_superuser=False))

        # Create snapshots for all users at the start of SA
        today = timezone.now().date()
        for user in users:
            try:
                profile = Profile.objects.get(user=user)
                cash_value = profile.cash or Decimal('0')
                
                # Calculate holdings value
                holdings_value = Decimal('0')
                for holding in Holding.objects.filter(user=user).select_related('stock'):
                    if holding.stock and holding.stock.price and holding.shares:
                        holdings_value += holding.stock.price * Decimal(holding.shares)
                
                # Calculate Trade P&L percentages
                trade_cumulative, trade_daily = calculate_trade_pnl_percentages(user, today)
                
                # Create snapshot for today (only if it doesn't exist)
                # get_or_create ensures one snapshot per day - created on first SA run, never updated
                Snapshot.objects.get_or_create(
                    user=user,
                    date=today,
                    defaults={
                        'cash_value': cash_value,
                        'holdings_value': holdings_value,
                        'trade_cumulative': trade_cumulative,
                        'trade_daily': trade_daily,
                    }
                )
            except Profile.DoesNotExist:
                # Skip users without profiles
                continue

        # Session smart analysis
        if not param_resuse:
            sa = SmartAnalysis()
            sa.save()
        else:
            sa = SmartAnalysis.objects.last()
            print(sa)

        self.stdout.write(f"Starting Smart Analysis ({sa.id}) ...")

        # Tmp: create missing profiles
        for user in users:
            Profile.objects.get_or_create(user=user)

        # Get advisor classes
        advisors = []

        for a in Advisor.objects.filter(enabled=True):
            module_name = a.python_class.lower()
            module = getattr(advisor_modules, module_name)
            pythonClass = getattr(module, a.python_class)

            adv = pythonClass(a)
            advisors.append(adv)

            # Lookout for user discovery
            if options['discover'] and a.python_class == "User":

                # Split on comma and strip whitespace
                symbols = [s.strip().upper() for s in options['discover'].split(',')]

                # Get explanation if provided
                explanation = options.get('explanation')

                for symbol in symbols:
                    adv.discovered( sa=sa, symbol=symbol, explanation=explanation)


        # Lets go
        if not param_discovery_only:
            analysis.analyze_holdings(sa, users, advisors)

        if not param_holdings_only:
            analysis.analyze_discovery(sa, users, advisors)

        sa.duration = timezone.now() - sa.started
        # save we all stats from session : users, trades, buys, sells, spend
        sa.save()
        
        self.stdout.write(
            self.style.SUCCESS('Smart Analysis complete!')
        )


