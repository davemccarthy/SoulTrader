"""
Force Buy Management Command

Buy one symbol into a named fund. Spends one spread tranche (Profile.average_spend)
from that fund's cash. Writes a User Advisor discovery with the given explanation
and no sell instructions, so holdings analysis can group the sleeve without the
trading engine selling it.

ETFs are created directly. Stock.create() rejects NYSE Arca, which is where the
sector ETFs trade.

Usage:
    python manage.py force_buy XLK --funds FOFS --explanation "Technology"
    python manage.py force_buy XLK --funds FOFS --explanation "Technology" --dry-run
"""

import logging
from decimal import Decimal, InvalidOperation

import yfinance as yf
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core.models import Advisor, Discovery, Holding, Profile, SmartAnalysis, Stock
from core.services.execution import execute_buy

logger = logging.getLogger(__name__)

TRADE_EXPLANATION_MAX = 256


def _yahoo_profile(symbol):
    """Company, exchange, and last price. Exchange is stored, not used as a gate."""
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    company = info.get("longName") or info.get("shortName") or ""
    exchange = info.get("fullExchangeName") or info.get("exchange") or ""
    fast = ticker.fast_info
    price = None
    if fast is not None:
        price = fast.get("lastPrice") or fast.get("regularMarketPrice")
    return (company or "")[:200], (exchange or "")[:32], _as_price(price)


def _as_price(value):
    if not value:
        return None
    try:
        price = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if price <= 0:
        return None
    return price


def _planned_buy_shares(allowance, cash, price, held_shares):
    """
    Share count execute_buy will add when force=False.

    Target size is int(spend / price) with spend capped by cash. Existing
    shares count toward that target, so a full tranche is not bought twice.
    """
    allowance = allowance or Decimal("0")
    cash = cash or Decimal("0")
    held = int(held_shares or 0)
    if price is None or price <= 0 or cash <= 0 or allowance <= 0:
        return 0
    spend = allowance if allowance <= cash else cash
    target = int(spend / price)
    if target <= 0:
        return 0
    buy = target - held
    if buy <= 0:
        return 0
    if cash < buy * price:
        return 0
    return buy


class Command(BaseCommand):
    help = "Force buy one symbol into a named fund at one spread tranche"

    def add_arguments(self, parser):
        parser.add_argument(
            "symbol",
            type=str,
            help="Symbol to buy (one per invocation, so the explanation stays with that sleeve)",
        )
        parser.add_argument(
            "--funds",
            type=str,
            required=True,
            help="Fund to buy into (Profile.name, exact match). Required.",
        )
        parser.add_argument(
            "--explanation",
            type=str,
            required=True,
            help='Sleeve text stored on the discovery (e.g. "Technology")',
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show the tranche, share count, and cash impact without buying",
        )

    def handle(self, *args, **options):
        symbol = (options["symbol"] or "").upper().strip()
        fund_name = (options["funds"] or "").strip()
        explanation = " ".join((options["explanation"] or "").split()).strip()
        dry_run = options["dry_run"]

        if not symbol:
            raise CommandError("You must provide a symbol to buy")
        if not fund_name:
            raise CommandError("You must provide --funds (Profile.name)")
        if not explanation:
            raise CommandError("You must provide --explanation (the sleeve text)")

        try:
            fund = Profile.objects.get(name=fund_name)
        except Profile.DoesNotExist:
            raise CommandError(f"Fund (profile) not found: {fund_name!r}")
        except Profile.MultipleObjectsReturned:
            raise CommandError(f"More than one fund named {fund_name!r}")

        if not fund.spread or fund.spread not in Profile.SPREAD:
            raise CommandError(
                f"{fund.name} has no spread. Set spread (MICRO, SMALL, ...) before force_buy."
            )

        self._warn_fund_setup(fund)

        try:
            advisor = Advisor.objects.get(python_class="User")
        except Advisor.DoesNotExist:
            raise CommandError("User Advisor not found")
        except Advisor.MultipleObjectsReturned:
            raise CommandError("More than one advisor with python_class User")

        allowance = fund.average_spend()
        cash = fund.cash or Decimal("0")
        holding = (
            Holding.objects.filter(fund=fund, stock__symbol=symbol)
            .select_related("stock", "discovery")
            .first()
        )
        held = holding.shares if holding else 0
        _company, _exchange, price = _yahoo_profile(symbol)

        if dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN - No trade will be executed"))

        self._report_plan(symbol, fund, explanation, allowance, cash, price, held)
        buy_shares = _planned_buy_shares(allowance, cash, price, held)
        if price is None:
            raise CommandError(f"No price for {symbol}")
        if buy_shares <= 0:
            self._report_no_buy(fund, symbol, allowance, cash, price, held)
            return
        if dry_run:
            cost = buy_shares * price
            self.stdout.write(
                self.style.WARNING(
                    f"Would buy {buy_shares} {symbol} for {fund.name} "
                    f"at ${price:.2f} (cost ${cost:,.2f}, cash left ${cash - cost:,.2f})"
                )
            )
            return

        stock, created_stock = self._ensure_stock(symbol, advisor, explanation)
        stock.refresh()
        price = stock.price if stock.price and stock.price > 0 else None
        buy_shares = _planned_buy_shares(allowance, cash, price, held)
        if price is None or buy_shares <= 0:
            self._drop_unused_stock(stock, created_stock)
            if price is None:
                raise CommandError(f"No price for {symbol}")
            self._report_no_buy(fund, symbol, allowance, cash, price, held)
            return

        sa = self._latest_sa()
        discovery = holding.discovery if holding and holding.discovery_id else None
        created_discovery = False
        if discovery is None:
            discovery = Discovery.objects.create(
                sa=sa,
                stock=stock,
                price=price,
                advisor=advisor,
                explanation=explanation,
                weight=Decimal("1.0"),
            )
            created_discovery = True
            self.stdout.write(
                f"Discovery {discovery.id} on session {sa.id} via {advisor.name}"
            )
        else:
            self.stdout.write(
                f"Keeping discovery {discovery.id}: {discovery.explanation}"
            )

        execute_buy(
            sa,
            fund,
            stock,
            allowance,
            explanation[:TRADE_EXPLANATION_MAX],
            discovery=discovery,
        )

        holding = (
            Holding.objects.filter(fund=fund, stock=stock, shares__gt=0)
            .select_related("discovery")
            .first()
        )
        if holding is None:
            if created_discovery:
                discovery.delete()
            self._drop_unused_stock(stock, created_stock)
            raise CommandError(f"Buy did not open a {symbol} holding in {fund.name}")

        fund.refresh_from_db(fields=["cash"])
        self.stdout.write(
            self.style.SUCCESS(
                f"Bought {symbol} for {fund.name}: {holding.shares} shares "
                f"@ ${holding.average_price:.2f}, cash ${fund.cash:,.2f}"
            )
        )

    def _report_no_buy(self, fund, symbol, allowance, cash, price, held):
        spend = allowance if allowance <= cash else cash
        target = int(spend / price) if price and spend > 0 else 0
        if held > 0 and target <= held:
            self.stdout.write(
                self.style.WARNING(
                    f"{fund.name} already holds {held} {symbol} at this tranche. Nothing to buy."
                )
            )
            return
        self.stdout.write(
            self.style.WARNING(
                f"Cannot buy {symbol} for {fund.name}: "
                f"cash ${cash:,.2f}, tranche ${allowance:,.2f}, price ${price:.2f}."
            )
        )

    def _report_plan(self, symbol, fund, explanation, allowance, cash, price, held):
        price_text = f"${price:.2f}" if price else "no price"
        buy_shares = _planned_buy_shares(allowance, cash, price, held)
        self.stdout.write(
            f"{symbol} {fund.name}: tranche ${allowance:,.2f}  cash ${cash:,.2f}  "
            f"price {price_text}  held {held}  buy {buy_shares}"
        )
        self.stdout.write(f"  explanation: {explanation}")

    def _drop_unused_stock(self, stock, created_stock):
        if not created_stock:
            return
        if Holding.objects.filter(stock=stock).exists():
            return
        if Discovery.objects.filter(stock=stock).exists():
            return
        symbol = stock.symbol
        stock.delete()
        self.stdout.write(f"Removed unused stock {symbol}")

    def _warn_fund_setup(self, fund):
        if not fund.enabled:
            self.stdout.write(
                self.style.WARNING(
                    f"{fund.name} is disabled. smartanalyse will not refresh it "
                    "unless you pass the fund name."
                )
            )
        if fund.sentiment_enabled():
            self.stdout.write(
                self.style.WARNING(
                    f"{fund.name} sentiment is {fund.sentiment}. "
                    "Equity purge can sell these holdings."
                )
            )
        if list(fund.advisors or []):
            self.stdout.write(
                self.style.WARNING(
                    f"{fund.name} advisors={list(fund.advisors)}. "
                    "Discovery buys can enter this fund."
                )
            )

    def _latest_sa(self):
        sa = SmartAnalysis.objects.order_by("-id").first()
        if sa is not None:
            self.stdout.write(f"Using SmartAnalysis session {sa.id}")
            return sa
        sa = SmartAnalysis.objects.create(
            username="force_buy",
            started=timezone.now(),
        )
        self.stdout.write(f"Created SmartAnalysis session {sa.id} for force buy")
        return sa

    def _ensure_stock(self, symbol, advisor, sector_label):
        """
        Return the Stock row, creating it when missing.

        Does not use Stock.create(), which drops symbols outside the national
        exchange list. Existing rows keep their company, exchange, and sector.
        """
        stock = Stock.objects.filter(symbol=symbol).first()
        if stock is not None:
            return stock, False

        company, exchange, _price = _yahoo_profile(symbol)
        stock = Stock(
            symbol=symbol,
            advisor=advisor,
            company=company,
            exchange=exchange,
            sector=(sector_label or "")[:100],
        )
        stock.save()
        logger.info("force_buy created stock %s (%s)", symbol, sector_label)
        self.stdout.write(f"Created stock {symbol} ({exchange or 'exchange unknown'})")
        return stock, True
