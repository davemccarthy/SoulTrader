"""
Force Sell Management Command

Force sell specific stocks for all users who hold them, or only for named funds (profiles).

Usage:
    python manage.py force_sell NVS ABBV AZN
    python manage.py force_sell --symbols NVS ABBV AZN
    python manage.py force_sell EX2 --funds "Experimental" "Main"
"""

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core.models import Holding, Stock, Profile, SmartAnalysis
from core.services.execution import execute_sell


class Command(BaseCommand):
    help = 'Force sell specific stocks; optionally limit to named funds (Profile.name)'

    def add_arguments(self, parser):
        parser.add_argument(
            'symbols',
            nargs='*',
            type=str,
            help='Stock symbols to force sell (e.g., NVS ABBV AZN)'
        )
        parser.add_argument(
            '--symbols',
            nargs='+',
            type=str,
            help='Stock symbols to force sell (alternative format)'
        )
        parser.add_argument(
            '--funds',
            nargs='+',
            type=str,
            help='Only sell holdings in these funds (Profile.name, exact match)'
        )
        parser.add_argument(
            '--explanation',
            type=str,
            default='Force sold by user',
            help='Explanation for the force sell trades'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be sold without actually selling'
        )

    def _resolve_fund(self, holding, fund_filter_ids):
        """
        Return the Profile (fund) to use for this holding and whether to skip.
        When fund_filter_ids is set, holding.fund must be in that set (queryset already filters).
        """
        fund = holding.fund
        if fund_filter_ids is not None:
            if fund is None or fund.id not in fund_filter_ids:
                return None, 'no fund or fund not in --funds list'
            return fund, None

        if fund is not None:
            return fund, None

        qs = Profile.objects.filter(user=holding.user)
        n = qs.count()
        if n == 0:
            return None, 'no profile for user and holding.fund is unset'
        if n > 1:
            return (
                qs.order_by('id').first(),
                'holding.fund unset; multiple profiles for user — using oldest profile',
            )
        return qs.get(), None

    def handle(self, *args, **options):
        symbols = options.get('symbols') or []
        if not symbols and args:
            symbols = list(args)

        if not symbols:
            raise CommandError('You must provide at least one stock symbol to sell')

        fund_names = options.get('funds')
        fund_filter_ids = None
        if fund_names:
            profiles = []
            for raw in fund_names:
                name = raw.strip()
                try:
                    profiles.append(Profile.objects.get(name=name))
                except Profile.DoesNotExist:
                    raise CommandError(f'Fund (profile) not found: {name!r}')
            fund_filter_ids = {p.id for p in profiles}
            self.stdout.write(f'Limiting to fund(s): {", ".join(p.name for p in profiles)}')

        explanation = options.get('explanation')
        dry_run = options.get('dry_run', False)

        if not dry_run:
            sa = SmartAnalysis.objects.create(
                username='force_sell',
                started=timezone.now()
            )
            self.stdout.write(f'Created SmartAnalysis session {sa.id} for force sell')
        else:
            sa = None
            self.stdout.write(self.style.WARNING('DRY RUN - No trades will be executed'))

        total_sold = 0
        total_value = 0

        for symbol in symbols:
            symbol = symbol.upper().strip()

            try:
                stock = Stock.objects.get(symbol=symbol)
            except Stock.DoesNotExist:
                self.stdout.write(self.style.WARNING(f'Stock {symbol} not found in database'))
                continue

            holdings = Holding.objects.filter(
                stock=stock,
                shares__gt=0,
                user__is_active=True,
            ).select_related('user', 'stock', 'fund')

            if fund_filter_ids is not None:
                holdings = holdings.filter(fund_id__in=fund_filter_ids)

            if not holdings.exists():
                self.stdout.write(f'No holdings found for {symbol}')
                continue

            self.stdout.write(f'\n{symbol}: Found {holdings.count()} holding(s)')

            for holding in holdings:
                fund, warn = self._resolve_fund(holding, fund_filter_ids)
                if fund is None:
                    self.stdout.write(
                        self.style.WARNING(
                            f'  Skipping {holding.user.username}: {warn}'
                        )
                    )
                    continue
                if warn:
                    self.stdout.write(
                        self.style.WARNING(f'  {holding.user.username}: {warn}')
                    )

                holding.stock.refresh()
                sell_value = holding.shares * holding.stock.price
                label = f'{holding.user.username} / {fund.name}'

                if dry_run:
                    self.stdout.write(
                        f'  Would sell {holding.shares} shares of {symbol} for {label} '
                        f'at ${holding.stock.price:.2f} (value: ${sell_value:.2f})'
                    )
                else:
                    self.stdout.write(
                        f'  Selling {holding.shares} shares of {symbol} for {label} '
                        f'at ${holding.stock.price:.2f} (value: ${sell_value:.2f})'
                    )
                    execute_sell(
                        sa=sa,
                        fund=fund,
                        holding=holding,
                        explanation=f'{explanation} ({symbol})',
                    )

                total_sold += holding.shares
                total_value += sell_value

        if dry_run:
            self.stdout.write(
                self.style.WARNING(
                    f'\nDRY RUN: Would sell {total_sold} total shares worth ${total_value:.2f}'
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    f'\nForce sell complete: Sold {total_sold} total shares worth ${total_value:.2f}'
                )
            )
