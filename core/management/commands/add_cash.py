"""
Adjust Profile.cash in thousands (e.g. 100 = +$100,000, -50 = −$50,000).

Does not change investment or spread; tranche size (average_spend) is unchanged.

Usage:
    python manage.py add_cash 100 --fund "Flux Test"
    python manage.py add_cash -25 --fund "Flux Test" --dry-run
    python manage.py add_cash --list-funds
"""

from decimal import Decimal

from django.core.management.base import BaseCommand, CommandError

from core.models import Holding, Profile


class Command(BaseCommand):
    help = "Add or remove fund cash in thousands (100 = $100,000). Requires --fund."

    def add_arguments(self, parser):
        parser.add_argument(
            "amount",
            nargs="?",
            type=int,
            help="Delta in thousands (100 = +$100,000, -100 = −$100,000)",
        )
        parser.add_argument(
            "--fund",
            type=str,
            help="Fund name (Profile.name, exact match)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show the change without saving",
        )
        parser.add_argument(
            "--allow-negative",
            action="store_true",
            help="Allow cash to go below zero after the adjustment",
        )
        parser.add_argument(
            "--list-funds",
            action="store_true",
            help="List profiles and cash balances, then exit",
        )

    def handle(self, *args, **options):
        if options.get("list_funds"):
            self._list_funds()
            return

        amount_k = options.get("amount")
        if amount_k is None:
            raise CommandError("amount is required (thousands), unless using --list-funds")

        fund_name = options.get("fund")
        if not fund_name or not str(fund_name).strip():
            raise CommandError("--fund is required (Profile.name)")

        fund_name = str(fund_name).strip()
        try:
            fund = Profile.objects.get(name=fund_name)
        except Profile.DoesNotExist:
            raise CommandError(f"Fund (profile) not found: {fund_name!r}")

        delta = Decimal(amount_k) * Decimal("1000")
        cash_before = fund.cash or Decimal("0")
        cash_after = cash_before + delta

        if cash_after < 0 and not options.get("allow_negative"):
            raise CommandError(
                f"Cash would become ${cash_after:,.2f}. "
                f"Use --allow-negative to permit, or use a smaller withdrawal."
            )

        holdings_value = self._holdings_market_value(fund)
        wealth_before = cash_before + holdings_value
        wealth_after = cash_after + holdings_value

        sign = "+" if delta >= 0 else ""
        self.stdout.write(f"Fund: {fund.name} (id={fund.id})")
        self.stdout.write(f"Delta: {sign}${delta:,.2f} ({amount_k:+d}k)")
        self.stdout.write(f"Cash:  ${cash_before:,.2f} → ${cash_after:,.2f}")
        self.stdout.write(
            f"Wealth (cash + holdings @ last price): "
            f"${wealth_before:,.2f} → ${wealth_after:,.2f}"
        )
        self.stdout.write(
            f"Investment (unchanged): ${fund.investment:,.2f} | "
            f"spread={fund.spread or '—'} | sentiment={fund.sentiment}"
        )
        if fund.spread:
            self.stdout.write(f"average_spend(): ${fund.average_spend():,.2f}")

        if options.get("dry_run"):
            self.stdout.write(self.style.WARNING("DRY RUN — cash not updated"))
            return

        fund.cash = cash_after
        fund.save(update_fields=["cash"])
        self.stdout.write(self.style.SUCCESS("Cash updated."))

    def _list_funds(self):
        profiles = Profile.objects.order_by("name")
        if not profiles.exists():
            self.stdout.write("No profiles found.")
            return
        self.stdout.write(f"{'Name':<30} {'Enabled':<8} {'Cash':>14} {'Spread':<8}")
        for p in profiles:
            cash = p.cash or Decimal("0")
            self.stdout.write(
                f"{p.name:<30} {str(p.enabled):<8} ${cash:>13,.2f} {p.spread or '—':<8}"
            )

    def _holdings_market_value(self, fund: Profile) -> Decimal:
        total = Decimal("0")
        for holding in Holding.objects.filter(fund=fund).select_related("stock"):
            if holding.stock and holding.stock.price and holding.shares:
                total += Decimal(str(holding.stock.price)) * Decimal(str(holding.shares))
        return total
