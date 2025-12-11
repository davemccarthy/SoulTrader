"""
Force Sell Management Command

Force sell specific stocks for all users who hold them.

Usage:
    python manage.py force_sell NVS ABBV AZN
    python manage.py force_sell --symbols NVS ABBV AZN
    python manage.py force_sell --all-users --symbols NVS
"""

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core.models import Holding, Stock, Profile, SmartAnalysis
from core.services.execution import execute_sell


class Command(BaseCommand):
    help = 'Force sell specific stocks for all users who hold them'

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
            '--explanation',
            type=str,
            default='Force sell: stock was force discovered and gives false results',
            help='Explanation for the force sell trades'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be sold without actually selling'
        )

    def handle(self, *args, **options):
        # Get symbols from either positional args or --symbols flag
        symbols = options.get('symbols') or []
        if not symbols and args:
            symbols = list(args)
        
        if not symbols:
            raise CommandError('You must provide at least one stock symbol to sell')

        explanation = options.get('explanation', 'Force sell: stock was force discovered and gives false results')
        dry_run = options.get('dry_run', False)

        # Create a SmartAnalysis session for tracking
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

            # Find all holdings for this stock
            holdings = Holding.objects.filter(stock=stock, shares__gt=0).select_related('user', 'stock')
            
            if not holdings.exists():
                self.stdout.write(f'No holdings found for {symbol}')
                continue

            self.stdout.write(f'\n{symbol}: Found {holdings.count()} holding(s)')

            for holding in holdings:
                user = holding.user
                try:
                    profile = Profile.objects.get(user=user)
                except Profile.DoesNotExist:
                    self.stdout.write(self.style.WARNING(f'  Skipping {user.username}: No profile found'))
                    continue

                # Refresh stock price
                holding.stock.refresh()
                sell_value = holding.shares * holding.stock.price

                if dry_run:
                    self.stdout.write(
                        f'  Would sell {holding.shares} shares of {symbol} for {user.username} '
                        f'at ${holding.stock.price:.2f} (value: ${sell_value:.2f})'
                    )
                else:
                    self.stdout.write(
                        f'  Selling {holding.shares} shares of {symbol} for {user.username} '
                        f'at ${holding.stock.price:.2f} (value: ${sell_value:.2f})'
                    )
                    
                    # Force sell (consensus can be None)
                    execute_sell(
                        sa=sa,
                        user=user,
                        profile=profile,
                        holding=holding,
                        explanation=f'{explanation} ({symbol})'
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































