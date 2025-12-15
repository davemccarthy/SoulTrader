"""
Analyse Holdings Management Command

Analyzes current holdings grouped by the advisor that discovered them.
Shows return statistics for holdings discovered by each advisor.

Usage:
    python manage.py analyse_holdings
    python manage.py analyse_holdings --user user19
    python manage.py analyse_holdings --advisor Yahoo
    python manage.py analyse_holdings --user user19 --advisor StockStory
    python manage.py analyse_holdings --verbose
    python manage.py analyse_holdings --user user19 --verbose
"""

from collections import defaultdict
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User

from core.models import Holding, Discovery, Advisor, Stock


class Command(BaseCommand):
    help = 'Analyze current holdings grouped by advisor with return statistics'

    def add_arguments(self, parser):
        parser.add_argument(
            '--user',
            type=str,
            help='Filter by username (e.g., user19)'
        )
        parser.add_argument(
            '--advisor',
            type=str,
            help='Filter by advisor name (e.g., Yahoo, StockStory)'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show individual stock details for each advisor'
        )

    def handle(self, *args, **options):
        username = options.get('user')
        advisor_name = options.get('advisor')
        verbose = options.get('verbose', False)

        # Build holdings queryset
        # Filter by active users only (exclude superusers)
        holdings_qs = Holding.objects.select_related('stock', 'user').filter(
            shares__gt=0,
            user__is_active=True,
            user__is_superuser=False
        )
        
        if username:
            try:
                user = User.objects.get(username=username, is_active=True, is_superuser=False)
                holdings_qs = holdings_qs.filter(user=user)
            except User.DoesNotExist:
                self.stdout.write(self.style.ERROR(f'User "{username}" not found or is not active'))
                return

        holdings = list(holdings_qs)
        if not holdings:
            self.stdout.write(self.style.WARNING('No holdings found for provided criteria.'))
            return

        # Group holdings by advisor
        # For each holding, find the most recent discovery for that stock
        advisor_holdings = defaultdict(list)
        
        for holding in holdings:
            stock = holding.stock
            
            # Find most recent discovery for this stock
            # Optionally filter by advisor if --advisor specified
            discovery_qs = Discovery.objects.filter(stock=stock).order_by('-created')
            
            if advisor_name:
                try:
                    advisor = Advisor.objects.get(name=advisor_name)
                    discovery_qs = discovery_qs.filter(advisor=advisor)
                except Advisor.DoesNotExist:
                    self.stdout.write(self.style.ERROR(f'Advisor "{advisor_name}" not found'))
                    return
            
            discovery = discovery_qs.first()
            
            if discovery:
                advisor_holdings[discovery.advisor.name].append({
                    'holding': holding,
                    'discovery': discovery,
                    'stock': stock
                })
            else:
                # No discovery found for this stock
                # Only add to [No Discovery] if not filtering by advisor
                if not advisor_name:
                    advisor_holdings['[No Discovery]'].append({
                        'holding': holding,
                        'discovery': None,
                        'stock': stock
                    })
                # If filtering by advisor and no discovery found, skip this holding

        if not advisor_holdings:
            self.stdout.write(self.style.WARNING('No holdings found with matching discoveries.'))
            return

        # Calculate statistics per advisor
        advisor_stats = {}
        
        for advisor_name, holdings_list in sorted(advisor_holdings.items()):
            total_holdings = len(holdings_list)
            total_cost = Decimal('0.0')
            total_current_value = Decimal('0.0')
            returns = []
            
            for item in holdings_list:
                holding = item['holding']
                stock = item['stock']
                
                # Calculate cost basis
                cost = Decimal(str(holding.average_price)) * Decimal(str(holding.shares))
                total_cost += cost
                
                # Current value
                current_price = Decimal(str(stock.price))
                current_value = current_price * Decimal(str(holding.shares))
                total_current_value += current_value
                
                # Individual return
                if holding.average_price > 0:
                    return_pct = ((current_price - Decimal(str(holding.average_price))) / Decimal(str(holding.average_price))) * 100
                    returns.append(float(return_pct))
            
            # Aggregate return
            if total_cost > 0:
                aggregate_return = ((total_current_value - total_cost) / total_cost) * 100
            else:
                aggregate_return = Decimal('0.0')
            
            # Average return
            avg_return = sum(returns) / len(returns) if returns else 0.0
            
            advisor_stats[advisor_name] = {
                'count': total_holdings,
                'total_cost': total_cost,
                'total_value': total_current_value,
                'aggregate_return': float(aggregate_return),
                'avg_return': avg_return,
                'holdings': holdings_list
            }

        # Display results
        self.stdout.write("\nHoldings Analysis by Advisor\n")
        self.stdout.write("=" * 80)
        
        for advisor_name in sorted(advisor_stats.keys()):
            stats = advisor_stats[advisor_name]
            
            # Format return with sign
            return_str = f"{stats['aggregate_return']:+.1f}%"
            avg_return_str = f"{stats['avg_return']:+.1f}%"
            
            # Format currency values
            cost_str = f"${stats['total_cost']:,.2f}"
            value_str = f"${stats['total_value']:,.2f}"
            
            self.stdout.write(
                f"{advisor_name:<20} "
                f"Holdings: {stats['count']:>3} | "
                f"Return: {return_str:>8} | "
                f"Avg Return: {avg_return_str:>8} | "
                f"Cost: {cost_str:>12} | "
                f"Value: {value_str:>12}"
            )
            
            # Show individual holdings if --verbose flag is set
            if verbose:
                for item in stats['holdings']:
                    holding = item['holding']
                    stock = item['stock']
                    user = holding.user.username
                    
                    if holding.average_price > 0:
                        return_pct = ((Decimal(str(stock.price)) - Decimal(str(holding.average_price))) / Decimal(str(holding.average_price))) * 100
                        return_pct_str = f"{float(return_pct):+.1f}%"
                    else:
                        return_pct_str = "N/A"
                    
                    self.stdout.write(
                        f"  {stock.symbol:>6} ({user:>10}) "
                        f"Shares: {holding.shares:>4} @ ${holding.average_price:>7.2f} → ${stock.price:>7.2f} "
                        f"Return: {return_pct_str:>8}"
                    )

        # Summary
        total_holdings = sum(s['count'] for s in advisor_stats.values())
        total_cost_all = sum(s['total_cost'] for s in advisor_stats.values())
        total_value_all = sum(s['total_value'] for s in advisor_stats.values())
        
        if total_cost_all > 0:
            total_return = ((total_value_all - total_cost_all) / total_cost_all) * 100
        else:
            total_return = Decimal('0.0')
        
        self.stdout.write("=" * 80)
        self.stdout.write(
            f"{'TOTAL':<20} "
            f"Holdings: {total_holdings:>3} | "
            f"Return: {total_return:+.1f}% | "
            f"Cost: ${total_cost_all:,.2f} | "
            f"Value: ${total_value_all:,.2f}"
        )







