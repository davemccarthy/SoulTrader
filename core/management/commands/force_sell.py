"""
Force Sell Management Command

Force sell specific stocks across funds (profiles), or only named funds.

Usage:
    python manage.py force_sell NVS ABBV AZN
    python manage.py force_sell --symbols NVS ABBV AZN
    python manage.py force_sell DAL --funds ZOMB
    python manage.py force_sell NVS ABBV --dry-run
"""

from decimal import Decimal

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core.models import Holding, Stock, Profile, SmartAnalysis
from core.services.execution import execute_sell


def _position_pnl(shares, price, average_price):
    """P&L if sold at price. Cost basis is average_price, same as execute_sell."""
    price = price or Decimal("0")
    avg = average_price or Decimal("0")
    shares_d = Decimal(shares)
    proceeds = shares_d * price
    cost = shares_d * avg
    pnl = proceeds - cost
    pnl_pct = (pnl / cost * Decimal("100")) if cost else None
    return proceeds, cost, pnl, pnl_pct


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
            help='Show what would be sold, with P&L by fund, without selling'
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

    def _fmt_pnl(self, pnl, pnl_pct):
        pct = "n/a" if pnl_pct is None else f"{pnl_pct:+.2f}%"
        text = f"${pnl:+,.2f} ({pct})"
        if pnl > 0:
            return self.style.SUCCESS(text)
        if pnl < 0:
            return self.style.ERROR(text)
        return text

    def _write_fund_pnl(self, by_fund):
        if not by_fund:
            return
        self.stdout.write("\nP&L by fund:")
        names = list(by_fund)
        if len(names) > 1:
            names.append("TOTAL")
        name_w = max(len(name) for name in names)
        totals = {"proceeds": Decimal("0"), "cost": Decimal("0"), "pnl": Decimal("0")}
        for name, row in by_fund.items():
            self._write_pnl_row(name, row, name_w)
            for key in totals:
                totals[key] += row[key]
        if len(by_fund) > 1:
            self._write_pnl_row("TOTAL", totals, name_w)

    def _write_pnl_row(self, name, row, name_w):
        cost = row["cost"]
        pnl = row["pnl"]
        pnl_pct = (pnl / cost * Decimal("100")) if cost else None
        self.stdout.write(
            f"  {name:<{name_w}}  value ${row['proceeds']:,.2f}  "
            f"cost ${cost:,.2f}  P&L {self._fmt_pnl(pnl, pnl_pct)}"
        )

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
        total_value = Decimal("0")
        by_fund = {}

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
                fund_id__isnull=False,
            ).select_related('stock', 'fund')

            if fund_filter_ids is not None:
                holdings = holdings.filter(fund_id__in=fund_filter_ids)
            else:
                holdings = holdings.filter(fund__enabled=True)

            if not holdings.exists():
                self.stdout.write(f'No holdings found for {symbol}')
                continue

            self.stdout.write(f'\n{symbol}: Found {holdings.count()} holding(s)')

            for holding in holdings:
                fund, warn = self._resolve_fund(holding, fund_filter_ids)
                if fund is None:
                    self.stdout.write(
                        self.style.WARNING(
                            f'  Skipping holding {holding.id}: {warn}'
                        )
                    )
                    continue
                if warn:
                    self.stdout.write(
                        self.style.WARNING(f'  {fund.name}: {warn}')
                    )

                holding.stock.refresh()
                proceeds, cost, pnl, pnl_pct = _position_pnl(
                    holding.shares, holding.stock.price, holding.average_price
                )
                label = fund.name
                pnl_text = self._fmt_pnl(pnl, pnl_pct)
                verb = "Would sell" if dry_run else "Selling"
                self.stdout.write(
                    f'  {verb} {holding.shares} shares of {symbol} for {label} '
                    f'at ${holding.stock.price:.2f} '
                    f'(value: ${proceeds:,.2f}, cost: ${cost:,.2f}, P&L: {pnl_text})'
                )
                if not dry_run:
                    execute_sell(
                        sa=sa,
                        fund=fund,
                        holding=holding,
                        explanation=f'{explanation} ({symbol})',
                    )

                total_sold += holding.shares
                total_value += proceeds
                bucket = by_fund.setdefault(
                    label,
                    {"proceeds": Decimal("0"), "cost": Decimal("0"), "pnl": Decimal("0")},
                )
                bucket["proceeds"] += proceeds
                bucket["cost"] += cost
                bucket["pnl"] += pnl

        if dry_run:
            self.stdout.write(
                self.style.WARNING(
                    f'\nDRY RUN: Would sell {total_sold} total shares worth ${total_value:,.2f}'
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    f'\nForce sell complete: Sold {total_sold} total shares worth ${total_value:,.2f}'
                )
            )
        self._write_fund_pnl(by_fund)
