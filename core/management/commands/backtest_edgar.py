from datetime import timedelta
from decimal import Decimal

import yfinance as yf
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from core.models import (
    Advisor,
    Consensus,
    Discovery,
    Holding,
    Profile,
    SellInstruction,
    SmartAnalysis,
    Stock,
    Trade,
)
from core.services.advisors.edgar import (
    Edgar,
    _filing_date_or_none,
    cik_to_ticker,
)
from edgar import find


class Command(BaseCommand):
    help = (
        "Backtest a single 8-K earnings filing with the Edgar advisor.\n"
        "Creates a Discovery and BUY Trade using the stock price on the filing date,\n"
        "without modifying the live execution or discovery paths."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "accession",
            type=str,
            help="EDGAR accession number (e.g. 0001185185-25-002022)",
        )
        parser.add_argument(
            "--username",
            type=str,
            help="Username to backtest for (defaults to the first user)",
        )

    def handle(self, *args, **options):
        accession = options["accession"]
        username = options.get("username")

        User = get_user_model()

        # Pick user
        if username:
            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                raise CommandError(f"User '{username}' does not exist")
        else:
            user = User.objects.order_by("id").first()
            if not user:
                raise CommandError("No users found. Create a user or specify --username.")

        # Get advisor row and implementation for Edgar
        try:
            advisor_row = Advisor.objects.get(python_class="Edgar")
        except Advisor.DoesNotExist:
            raise CommandError("Edgar advisor (python_class='Edgar') not found in Advisor table")

        # Resolve Python class the same way as run_edgar_standalone
        from core.services import advisors as advisor_modules

        module_name = advisor_row.python_class.lower()
        module = getattr(advisor_modules, module_name)
        PythonClass = getattr(module, advisor_row.python_class)

        impl: Edgar = PythonClass(advisor_row)

        # Minimal SmartAnalysis stub for this backtest run
        sa = SmartAnalysis.objects.create(username=user.username)

        self.stdout.write(self.style.NOTICE(f"Backtesting {accession} for user '{user.username}'"))

        # Locate filing
        try:
            filing = find(accession)
        except Exception as e:
            raise CommandError(f"Could not find EDGAR filing {accession}: {e}")

        # Determine ticker and filing date
        cik = str(getattr(filing, "cik", "") or "")
        ticker = getattr(filing, "ticker", None) or cik_to_ticker(cik)
        if not ticker:
            raise CommandError("Filing has no tradable ticker (CIK to ticker resolution failed)")

        filing_date = _filing_date_or_none(filing)
        if not filing_date:
            raise CommandError("Filing date could not be determined for this 8-K")

        self.stdout.write(f"Ticker: {ticker}, filing_date: {filing_date.isoformat()}")

        # Run Edgar's basic + advanced analysis to ensure the filing passes filters
        if not impl.analyze_8k_basic(filing):
            raise CommandError("Edgar basic 8-K filters rejected this filing (no backtest created).")

        advanced = impl.analyze_8k_advanced(filing)
        if advanced is None:
            raise CommandError("Edgar advanced 8-K filters rejected this filing (no backtest created).")

        # Build explanation and score details (for logging only)
        explanation = impl.build_explanation(filing, advanced)
        score, bonuses, penalties = impl.score_results(
            filing,
            advanced["ex99"],
            advanced["media"],
        )

        self.stdout.write(self.style.SUCCESS(f"Edgar score: {score:.2f}"))
        if bonuses:
            self.stdout.write("Bonuses: " + " | ".join(bonuses))
        if penalties:
            self.stdout.write("Penalties: " + " | ".join(penalties))

        # Fetch historical price for the filing date
        # For typical 8-Ks released before the market opens, we use the opening price
        # of the filing day. If that is unavailable, we fall back to the previous
        # day's close.
        price_on_filing = self._get_historical_price(ticker, filing_date)
        if price_on_filing is None:
            raise CommandError(f"Could not fetch historical price for {ticker} on {filing_date}")

        self.stdout.write(
            self.style.NOTICE(
                f"Using historical opening/previous-close price {price_on_filing} for backtest BUY on {filing_date}"
            )
        )

        # Get or create Stock (do NOT override live Stock.price)
        try:
            stock = Stock.objects.get(symbol=ticker)
        except Stock.DoesNotExist:
            stock = Stock.create(ticker, advisor_row)
            if stock is None:
                raise CommandError(f"Could not create Stock for ticker {ticker}")

        # Create Discovery with historical price and Edgar weight
        discovery_weight = Decimal(str(advanced["weight"]))

        discovery = Discovery.objects.create(
            sa=sa,
            stock=stock,
            price=price_on_filing,
            advisor=advisor_row,
            explanation=explanation,
            weight=discovery_weight,
        )

        # Mirror Edgar's default sell instructions (but priced off historical price)
        sell_instructions = [
            ("PERCENTAGE_DIMINISHING", Decimal("1.20"), 14),
            ("PERCENTAGE_AUGMENTING", Decimal("0.90"), 14),
            ("PEAKED", Decimal("7.0"), None),
            ("PROFIT_FLAT", Decimal("0.5"), 4),
            ("DESCENDING_TREND", Decimal("-0.20"), None),
        ]
        self._create_sell_instructions(discovery, stock, price_on_filing, sell_instructions)

        # Compute allowance using the same weighting approach as analyze_discovery()
        profile = Profile.objects.get(user=user)
        risk_settings = Profile.RISK[profile.risk]
        base_allowance = profile.average_spend()
        risk_weight = Decimal(str(risk_settings["weight"]))

        advisor_weight = Decimal(str(advisor_row.weight))
        combined_weight = advisor_weight * discovery_weight

        allowance = base_allowance
        if combined_weight > Decimal("1.0"):
            allowance = allowance * (combined_weight * risk_weight)
        else:
            allowance = allowance * (combined_weight / risk_weight)

        self.stdout.write(
            self.style.NOTICE(
                f"Base allowance {base_allowance:.2f}, combined_weight {combined_weight:.3f}, "
                f"risk_weight {risk_weight:.3f} -> backtest allowance {allowance:.2f}"
            )
        )

        # Execute a synthetic BUY trade at the historical price (without touching live execute_buy)
        trade = self._execute_buy_backtest(sa, user, stock, allowance, price_on_filing, explanation)

        self.stdout.write(
            self.style.SUCCESS(
                f"Backtest BUY created: {trade.shares} shares of {stock.symbol} at {trade.price} "
                f"(Trade id={trade.id}, Discovery id={discovery.id})"
            )
        )

    def _get_historical_price(self, symbol, trade_date):
        """
        Get a realistic execution price for an 8-K backtest.

        Primary: opening price on trade_date (typical for pre-open 8-Ks).
        Fallback: previous day's close if opening price is unavailable.

        Returns:
            Decimal or None
        """
        try:
            ticker = yf.Ticker(symbol)
            # Fetch [trade_date, trade_date + 1) so we get exactly one row
            hist = ticker.history(
                start=trade_date,
                end=trade_date + timedelta(days=1),
                interval="1d",
            )
            if hist is None or hist.empty:
                return None

            # Prefer the opening price for the filing day
            if "Open" in hist.columns and not hist["Open"].isna().all():
                open_val = float(hist["Open"].iloc[0])
                if open_val > 0:
                    return Decimal(str(open_val))

            # Fallback: previous day's close
            if "Close" in hist.columns and not hist["Close"].isna().all():
                close_val = float(hist["Close"].iloc[0])
                if close_val > 0:
                    return Decimal(str(close_val))

            return None
        except Exception:
            return None

    def _create_sell_instructions(self, discovery, stock, price_on_filing, instructions):
        """
        Create SellInstruction rows mirroring AdvisorBase.discovered(), but priced off
        the historical filing-day price instead of the live stock price.
        """
        for instruction_type, instruction_value, value2 in instructions:
            inst = SellInstruction()
            inst.discovery = discovery
            inst.instruction = instruction_type

            if instruction_type in [
                "STOP_PERCENTAGE",
                "TARGET_PERCENTAGE",
                "END_DAY",
                "PERCENTAGE_DIMINISHING",
                "PERCENTAGE_AUGMENTING",
            ]:
                inst.value1 = (
                    price_on_filing * Decimal(str(instruction_value))
                    if instruction_value is not None
                    else None
                )
            elif instruction_type in ["TARGET_DIMINISHING", "STOP_AUGMENTING"]:
                inst.value1 = (
                    Decimal(str(instruction_value)) if instruction_value is not None else None
                )
            elif instruction_type == "NOT_TRENDING":
                inst.value1 = None
            else:
                inst.value1 = (
                    Decimal(str(instruction_value)) if instruction_value is not None else None
                )

            inst.value2 = Decimal(str(value2)) if value2 is not None else None
            inst.save()

    def _execute_buy_backtest(self, sa, user, stock, allowance, trade_price, explanation=""):
        """
        Minimal clone of execute_buy that uses a provided trade_price and does NOT
        refresh or overwrite Stock.price (to avoid touching live paths).
        """
        profile = Profile.objects.get(user=user)
        holding = Holding.objects.filter(user=user, stock=stock).first()

        if holding is None:
            holding = Holding()

        # Cap allowance to available cash
        if allowance > profile.cash:
            allowance = profile.cash

        if trade_price <= 0:
            raise CommandError(f"Invalid trade_price {trade_price} for {stock.symbol}")

        # Number of shares to buy at the historical price
        shares = int(allowance / trade_price)
        if shares <= 0:
            raise CommandError(
                f"Backtest: cannot afford any shares of {stock.symbol} with allowance {allowance}"
            )

        cost = shares * trade_price
        if profile.cash < cost:
            raise CommandError(
                f"Backtest: not enough cash to buy {shares} shares of {stock.symbol} "
                f"(needed {cost}, have {profile.cash})"
            )

        profile.cash -= cost

        # Create Trade (BUY) at historical price
        trade = Trade.objects.create(
            sa=sa,
            user=user,
            stock=stock,
            action="BUY",
            price=trade_price,
            shares=shares,
            consensus=None,
            cost=None,
            explanation=explanation[:256] if explanation else "",
        )

        # Update / create Holding with weighted-average cost based on historical price
        holding.user = user
        holding.stock = stock

        old_shares = holding.shares or 0
        old_avg = holding.average_price or Decimal("0")

        holding.shares = old_shares + shares
        if holding.shares > 0:
            total_cost = (old_avg * Decimal(old_shares)) + (trade_price * shares)
            holding.average_price = total_cost / Decimal(holding.shares)
        else:
            holding.average_price = trade_price

        holding.save()
        profile.save()

        return trade

