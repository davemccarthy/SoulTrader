"""
Smart Analysis Management Command

Usage:
    python manage.py smartanalyse <username>
    python manage.py smartanalyse --all
"""
from operator import attrgetter

from django.core.management.base import BaseCommand, CommandError

from core.services import analysis
from core.services.advisors.advisor import AdvisorBase
from core.models import SmartAnalysis, Advisor, Profile
from django.contrib.auth.models import User
from django.utils import timezone
from decimal import Decimal

# Register advisors from here
from core.services.advisors import alpha
from core.services import advisors as advisor_modules

import logging



logger = logging.getLogger(__name__)


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
    
    def handle(self, *args, **options):

        self.stdout.write('Starting Smart Analysis...')

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

        # Session smart analysis
        if not param_resuse:
            sa = SmartAnalysis()
            sa.save()
        else:
            sa = SmartAnalysis.objects.last()
            print(sa)

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

                for symbol in symbols:
                    adv.discovered( sa=sa, symbol=symbol)


        # Lets go
        if not param_holdings_only:
            analysis.analyze_discovery(sa, users, advisors)

        if not param_discovery_only:
            analysis.analyze_holdings(sa, users, advisors)

        sa.duration = timezone.now() - sa.started
        # save we all stats from session : users, trades, buys, sells, spend
        sa.save()
        
        self.stdout.write(
            self.style.SUCCESS('Smart Analysis complete!')
        )


