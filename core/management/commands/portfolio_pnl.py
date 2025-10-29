"""
Portfolio P&L Management Command

Usage:
    python manage.py portfolio_pnl [username]
    python manage.py portfolio_pnl --all
    python manage.py portfolio_pnl user1 user2
"""

from django.core.management.base import BaseCommand, CommandError
from core.models import User, Profile, Holding
from decimal import Decimal


class Command(BaseCommand):
    help = 'Calculate profit and loss for user portfolios'

    def add_arguments(self, parser):
        parser.add_argument(
            'usernames',
            nargs='*',
            type=str,
            help='Username(s) to calculate P&L for (optional)'
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Calculate P&L for all active users'
        )
        parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed holdings breakdown'
        )

    def handle(self, *args, **options):
        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS('PORTFOLIO PROFIT & LOSS CALCULATION'))
        self.stdout.write('='*60 + '\n')

        # Get users to process
        if options['all']:
            users = User.objects.filter(is_superuser=False, is_active=True)
        elif options['usernames']:
            users = []
            for username in options['usernames']:
                try:
                    user = User.objects.get(username=username)
                    users.append(user)
                except User.DoesNotExist:
                    self.stdout.write(self.style.ERROR(f'User "{username}" does not exist'))
        else:
            # Default: show all active non-superuser users
            users = User.objects.filter(is_superuser=False, is_active=True)

        if not users:
            self.stdout.write(self.style.WARNING('No users found'))
            return

        detailed = options['detailed']
        total_portfolio_value = Decimal('0.00')
        total_initial_investment = Decimal('0.00')

        for user in users:
            try:
                profile = Profile.objects.get(user=user)
                holdings = Holding.objects.filter(user=user, shares__gt=0)

                self.stdout.write(f'\n--- {self.style.SUCCESS(user.username.upper())} ---')
                self.stdout.write(f'Initial Investment: ${profile.investment:,.2f}')
                self.stdout.write(f'Current Cash: ${profile.cash:,.2f}')

                # Calculate holdings value
                total_cost_basis = Decimal('0.00')
                total_current_value = Decimal('0.00')
                holdings_list = []

                for holding in holdings:
                    cost_basis = Decimal(str(holding.shares)) * holding.average_price
                    current_value = Decimal(str(holding.shares)) * holding.stock.price
                    unrealized_pnl = current_value - cost_basis
                    pnl_pct = ((holding.stock.price - holding.average_price) / holding.average_price * 100) if holding.average_price > 0 else 0

                    total_cost_basis += cost_basis
                    total_current_value += current_value

                    holdings_list.append({
                        'symbol': holding.stock.symbol,
                        'shares': holding.shares,
                        'avg_price': holding.average_price,
                        'current_price': holding.stock.price,
                        'cost_basis': cost_basis,
                        'current_value': current_value,
                        'pnl': unrealized_pnl,
                        'pnl_pct': pnl_pct
                    })

                if detailed and holdings_list:
                    self.stdout.write(f'\nHoldings ({len(holdings_list)} positions):')
                    self.stdout.write('-' * 60)
                    for h in sorted(holdings_list, key=lambda x: x['current_value'], reverse=True):
                        pnl_sign = "+" if h['pnl'] >= 0 else ""
                        pnl_color = self.style.SUCCESS if h['pnl'] >= 0 else self.style.ERROR
                        self.stdout.write(
                            f"{h['symbol']:8s} | {h['shares']:4d} shares | "
                            f"Avg: ${h['avg_price']:7.2f} | Curr: ${h['current_price']:7.2f} | "
                            f"Value: ${h['current_value']:10,.2f} | "
                            f"P&L: {pnl_color(f'{pnl_sign}${h['pnl']:8,.2f} ({pnl_sign}{h['pnl_pct']:.2f}%)')}"
                        )
                    self.stdout.write('-' * 60)

                # Calculate total portfolio value
                total_portfolio_value_user = profile.cash + total_current_value
                total_portfolio_value += total_portfolio_value_user
                total_initial_investment += profile.investment

                # Unrealized P&L (from holdings)
                unrealized_pnl = total_current_value - total_cost_basis

                # Total P&L (portfolio value - initial investment)
                total_pnl = total_portfolio_value_user - profile.investment
                total_pnl_pct = (total_pnl / profile.investment * 100) if profile.investment > 0 else 0

                self.stdout.write(f'\nPortfolio Summary:')
                self.stdout.write(f'  Cash:              ${profile.cash:,.2f}')
                self.stdout.write(f'  Holdings Value:    ${total_current_value:,.2f}')
                self.stdout.write(f'  Total Portfolio:   ${total_portfolio_value_user:,.2f}')
                self.stdout.write(f'  Unrealized P&L:    ${unrealized_pnl:+,.2f}')
                
                # Color code total P&L
                if total_pnl >= 0:
                    self.stdout.write(self.style.SUCCESS(f'  Total P&L:         ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)'))
                else:
                    self.stdout.write(self.style.ERROR(f'  Total P&L:         ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)'))

            except Profile.DoesNotExist:
                self.stdout.write(self.style.WARNING(f'\n--- {user.username.upper()} ---'))
                self.stdout.write(self.style.WARNING('No profile found'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'\n--- {user.username.upper()} ---'))
                self.stdout.write(self.style.ERROR(f'Error: {e}'))

        # Overall summary if multiple users
        if len(users) > 1:
            total_pnl_overall = total_portfolio_value - total_initial_investment
            total_pnl_pct_overall = (total_pnl_overall / total_initial_investment * 100) if total_initial_investment > 0 else 0
            
            self.stdout.write('\n' + '='*60)
            self.stdout.write(self.style.SUCCESS('OVERALL SUMMARY'))
            self.stdout.write('='*60)
            self.stdout.write(f'Total Initial Investment: ${total_initial_investment:,.2f}')
            self.stdout.write(f'Total Portfolio Value:    ${total_portfolio_value:,.2f}')
            if total_pnl_overall >= 0:
                self.stdout.write(self.style.SUCCESS(f'Total P&L:                ${total_pnl_overall:+,.2f} ({total_pnl_pct_overall:+.2f}%)'))
            else:
                self.stdout.write(self.style.ERROR(f'Total P&L:                ${total_pnl_overall:+,.2f} ({total_pnl_pct_overall:+.2f}%)'))

        self.stdout.write('\n' + '='*60 + '\n')

