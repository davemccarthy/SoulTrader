"""
Transfer a holding between funds at the current mark.

Creates a User discovery (default SIs) on the destination; leaves the source
discovery untouched for any other funds still linked to it.

Usage:
    python manage.py transfer_holding --from PLSE-S --to ADHOC QCOM
    python manage.py transfer_holding --from EXP1 --to ADHOC AAPL --dry-run
"""

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core.models import Holding, Profile, SmartAnalysis, Stock
from core.services.execution import TransferError, transfer_holding


class Command(BaseCommand):
    help = "Transfer an entire holding from one fund to another at current mark"

    def add_arguments(self, parser):
        parser.add_argument(
            "symbol",
            type=str,
            help="Stock symbol to transfer (e.g. QCOM)",
        )
        parser.add_argument(
            "--from",
            dest="from_fund",
            required=True,
            help="Source fund Profile.name",
        )
        parser.add_argument(
            "--to",
            dest="to_fund",
            required=True,
            help="Destination fund Profile.name",
        )
        parser.add_argument(
            "--explanation",
            type=str,
            default="",
            help="Optional trade explanation (applied to both legs if set)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show transfer details without executing",
        )

    def handle(self, *args, **options):
        symbol = str(options["symbol"]).strip().upper()
        from_name = str(options["from_fund"]).strip()
        to_name = str(options["to_fund"]).strip()
        explanation = (options.get("explanation") or "").strip() or None
        dry_run = bool(options.get("dry_run"))

        try:
            from_fund = Profile.objects.get(name=from_name)
        except Profile.DoesNotExist as exc:
            raise CommandError(f"Source fund not found: {from_name!r}") from exc

        try:
            to_fund = Profile.objects.get(name=to_name)
        except Profile.DoesNotExist as exc:
            raise CommandError(f"Destination fund not found: {to_name!r}") from exc

        try:
            stock = Stock.objects.get(symbol=symbol)
        except Stock.DoesNotExist as exc:
            raise CommandError(f"Stock not found: {symbol}") from exc

        holding = (
            Holding.objects.filter(fund=from_fund, stock=stock, shares__gt=0)
            .select_related("stock")
            .first()
        )
        if holding is None:
            raise CommandError(f"{from_name} has no {symbol} holding")

        stock.refresh()
        shares = int(holding.shares)
        price = stock.price
        notional = (price or 0) * shares
        avg = holding.average_price or 0

        self.stdout.write(f"From: {from_name}  cash=${from_fund.cash}")
        self.stdout.write(f"To:   {to_name}  cash=${to_fund.cash}")
        self.stdout.write(
            f"{symbol}: {shares} shares @ mark ${price} "
            f"(avg ${avg}, notional ${notional:.2f})"
        )

        if dry_run:
            self.stdout.write(
                self.style.WARNING(
                    f"DRY RUN: would transfer {shares} {symbol} "
                    f"{from_name} → {to_name} at ${price}"
                )
            )
            if to_fund.cash < notional:
                self.stdout.write(
                    self.style.ERROR(
                        f"Would fail: {to_name} needs ${notional:.2f}, "
                        f"has ${to_fund.cash}"
                    )
                )
            return

        sa = SmartAnalysis.objects.create(
            username="transfer_holding",
            started=timezone.now(),
        )
        self.stdout.write(f"SmartAnalysis session {sa.id}")

        try:
            result = transfer_holding(
                sa,
                from_fund,
                to_fund,
                stock,
                explanation=explanation,
            )
        except TransferError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                f"Transferred {result['shares']} {result['symbol']} "
                f"{result['from_fund']} → {result['to_fund']} "
                f"@ ${result['price']} (discovery {result['discovery_id']})"
            )
        )
