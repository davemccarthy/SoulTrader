"""
Track Vunder Advisor Stocks for a User

Usage:
    python manage.py track_vunder User23
    python manage.py track_vunder User23 --detailed
"""
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.db.models import Q, F
from django.utils import timezone
from decimal import Decimal
from datetime import timedelta

from core.models import Holding, Discovery, Advisor, Stock, SellInstruction


class Command(BaseCommand):
    help = 'Track Vunder advisor stocks for a user'
    
    def add_arguments(self, parser):
        parser.add_argument(
            'username',
            type=str,
            help='Username to track Vunder stocks for'
        )
        parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed information including sell instructions'
        )
    
    def handle(self, *args, **options):
        username = options['username']
        detailed = options['detailed']
        
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'User "{username}" not found'))
            return
        
        # Get Vunder advisor
        try:
            vunder_advisor = Advisor.objects.get(name='Vunder')
        except Advisor.DoesNotExist:
            self.stdout.write(self.style.ERROR('Vunder advisor not found'))
            return
        
        # Get all holdings for this user
        holdings = Holding.objects.filter(user=user)
        
        if not holdings.exists():
            self.stdout.write(self.style.WARNING(f'No holdings found for {username}'))
            return
        
        # Find which holdings came from Vunder discoveries
        # Get all Vunder discoveries for this user's stocks
        vunder_discoveries = Discovery.objects.filter(
            advisor=vunder_advisor,
            stock__in=[h.stock for h in holdings]
        ).select_related('stock').order_by('-created')
        
        # Match holdings with discoveries
        vunder_holdings = []
        for holding in holdings:
            # Find the most recent Vunder discovery for this stock
            discovery = vunder_discoveries.filter(stock=holding.stock).first()
            if discovery:
                vunder_holdings.append((holding, discovery))
        
        if not vunder_holdings:
            self.stdout.write(self.style.WARNING(f'No Vunder stocks found in {username}\'s holdings'))
            return
        
        self.stdout.write(self.style.SUCCESS(f'\n{"="*80}'))
        self.stdout.write(self.style.SUCCESS(f'Vunder Stocks for {username} ({len(vunder_holdings)} stocks)'))
        self.stdout.write(self.style.SUCCESS(f'{"="*80}\n'))
        
        # Refresh stock prices
        for holding, discovery in vunder_holdings:
            holding.stock.refresh()
        
        # Calculate totals
        total_cost = Decimal('0')
        total_value = Decimal('0')
        
        # Display holdings
        self.stdout.write(f'{"Symbol":<8} {"Shares":>8} {"Buy Price":>12} {"Current":>12} {"P&L":>12} {"P&L %":>10} {"Days":>6} {"Discovery Price":>15}')
        self.stdout.write('-' * 80)
        
        for holding, discovery in sorted(vunder_holdings, key=lambda x: x[0].stock.symbol):
            stock = holding.stock
            current_price = stock.price
            buy_price = holding.average_price
            shares = holding.shares
            
            cost_basis = buy_price * shares
            current_value = current_price * shares
            pnl = current_value - cost_basis
            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else Decimal('0')
            
            # Calculate days held
            days_held = (timezone.now() - discovery.created).days if discovery.created else 0
            
            total_cost += cost_basis
            total_value += current_value
            
            # Color coding for P&L
            pnl_str = f"${pnl:>11,.2f}"
            pnl_pct_str = f"{pnl_pct:>9.2f}%"
            
            if pnl > 0:
                pnl_str = self.style.SUCCESS(pnl_str)
                pnl_pct_str = self.style.SUCCESS(pnl_pct_str)
            elif pnl < 0:
                pnl_str = self.style.ERROR(pnl_str)
                pnl_pct_str = self.style.ERROR(pnl_pct_str)
            
            discovery_price = discovery.price if discovery.price else Decimal('0')
            
            self.stdout.write(
                f"{stock.symbol:<8} {shares:>8} ${buy_price:>11.2f} ${current_price:>11.2f} "
                f"{pnl_str} {pnl_pct_str} {days_held:>6} ${discovery_price:>14.2f}"
            )
            
            if detailed:
                # Show discovery explanation
                self.stdout.write(f"  Discovery: {discovery.explanation[:100]}")
                
                # Show sell instructions
                instructions = SellInstruction.objects.filter(discovery=discovery).order_by('id')
                if instructions.exists():
                    self.stdout.write(f"  Sell Instructions:")
                    for inst in instructions:
                        if inst.instruction == 'PERCENTAGE_DIMINISHING':
                            # Calculate current target
                            # value1 is already an absolute price (was converted from percentage when created)
                            if inst.value1 and buy_price:
                                max_days = int(inst.value2) if inst.value2 is not None else 180
                                if days_held <= max_days:
                                    progress = float(days_held) / float(max_days) if max_days > 0 else 1.0
                                    original_target = float(inst.value1)  # Already absolute price
                                    current_target = original_target - progress * (original_target - float(buy_price))
                                else:
                                    current_target = float(buy_price)
                                # Calculate percentage for display
                                target_pct = (original_target / float(buy_price) * 100) if buy_price > 0 else 0
                                self.stdout.write(
                                    f"    - PERCENTAGE_DIMINISHING: {target_pct:.0f}% target "
                                    f"(current: ${current_target:.2f}, day {days_held}/{max_days})"
                                )
                        elif inst.instruction == 'PERCENTAGE_AUGMENTING':
                            # Calculate current stop
                            # value1 is already an absolute price (was converted from percentage when created)
                            if inst.value1 and buy_price:
                                max_days = int(inst.value2) if inst.value2 is not None else 180
                                if days_held <= max_days:
                                    progress = float(days_held) / float(max_days) if max_days > 0 else 1.0
                                    original_stop = float(inst.value1)  # Already absolute price
                                    current_stop = original_stop + progress * (float(buy_price) - original_stop)
                                else:
                                    current_stop = float(buy_price)
                                # Calculate percentage for display
                                stop_pct = (original_stop / float(buy_price) * 100) if buy_price > 0 else 0
                                self.stdout.write(
                                    f"    - PERCENTAGE_AUGMENTING: {stop_pct:.0f}% stop "
                                    f"(current: ${current_stop:.2f}, day {days_held}/{max_days})"
                                )
                        else:
                            self.stdout.write(f"    - {inst.instruction}: {inst.value1}")
                self.stdout.write("")
        
        # Summary
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else Decimal('0')
        
        self.stdout.write('-' * 80)
        self.stdout.write(f'{"TOTAL":<8} {"":>8} {"":>12} {"":>12} ', ending='')
        
        if total_pnl > 0:
            self.stdout.write(self.style.SUCCESS(f"${total_pnl:>11,.2f} {total_pnl_pct:>9.2f}%"))
        elif total_pnl < 0:
            self.stdout.write(self.style.ERROR(f"${total_pnl:>11,.2f} {total_pnl_pct:>9.2f}%"))
        else:
            self.stdout.write(f"${total_pnl:>11,.2f} {total_pnl_pct:>9.2f}%")
        
        self.stdout.write(f'\nTotal Cost: ${total_cost:,.2f}')
        self.stdout.write(f'Total Value: ${total_value:,.2f}')
        self.stdout.write(f'Total P&L: ${total_pnl:,.2f} ({total_pnl_pct:.2f}%)\n')

