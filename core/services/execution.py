"""
Trade Execution Service

Executes buy/sell trades and updates holdings
"""

import logging
from decimal import Decimal, InvalidOperation
from typing import Optional

from django.db import transaction

from core.models import Advisor, Holding, Trade

logger = logging.getLogger(__name__)


def _fmt_usd(value) -> str:
    """Format dollars for logs (2 dp; avoids float / long-Decimal string noise)."""
    if value is None:
        return "n/a"
    try:
        d = value if isinstance(value, Decimal) else Decimal(str(value))
        d = d.quantize(Decimal("0.01"))
    except (InvalidOperation, TypeError, ValueError):
        return str(value)
    return format(d, ".2f")


# Sell all for now
def execute_sell(sa, fund, holding, explanation):

    # Latest price
    holding.stock.refresh()
    logger.info(
        f"Trade: {fund.name} selling {holding.shares} shares of {holding.stock.symbol} "
        f"at ${_fmt_usd(holding.stock.price)}. {explanation}"
    )

    # Capture cost basis and references BEFORE deleting holding
    cost_basis = holding.average_price or Decimal('0')
    stock_ref = holding.stock  # Save reference before deleting
    shares_ref = holding.shares  # Save reference before deleting
    sell_price = stock_ref.price  # Save price before deleting
    discovery_ref = holding.discovery  # Preserve position provenance for SELL trade

    # Transfer funds
    fund.cash += holding.shares * holding.stock.price

    # Delete holding
    holding.delete()

    # Create trade record
    trade = Trade()

    trade.sa = sa
    trade.user = fund.user  # TODO: Remove. Required for backward compatibility temp
    trade.fund = fund
    trade.action = "SELL"
    trade.stock = stock_ref
    trade.discovery = discovery_ref
    trade.price = sell_price
    trade.shares = shares_ref
    trade.cost = cost_basis  # Store cost basis for P&L calculation
    trade.explanation = explanation
    trade.save()

    fund.save()

def execute_buy(sa, fund, stock, allowance, explanation="", force = False, discovery=None):

    # Check for existing stock
    #profile = Profile.objects.get(user=user)
    holding = Holding.objects.filter(fund=fund, stock=stock).first()

    if holding is None:
        holding = Holding()

    # Check cash
    if allowance > fund.cash:
        logger.warning(f"{fund.name} low on cash ${_fmt_usd(fund.cash)}")
        allowance = fund.cash

    # Latest price
    stock.refresh()

    # Verify stock price
    if stock.price == 0.0:
        logger.warning(f"Trade: no price for {stock.symbol}")
        return

    # Calculate no. shares to buy
    shares = int(allowance / stock.price)

    if shares ==  0:
        logger.info(f"Trade: {fund.name} NOT buying shares of {stock.symbol}. Can't afford any")
        return

    # No buy if have shares (surrender allowance for subsequent purchases)
    if shares - holding.shares <= 0 and not force:
        logger.info(f"{fund.name} already has {holding.shares} {stock.symbol} shares")
        return

    if not force:
        shares -= holding.shares

    cost = shares * stock.price

    if fund.cash < cost:
        logger.info(f"Trade: {fund.name} NOT buying shares of {stock.symbol}. Not enough cash")
        return

    logger.info(
        f"Trade: {fund.name} buying {shares} shares of {stock.symbol} "
        f"at ${_fmt_usd(stock.price)}. Holding {holding.shares}"
    )

    fund.cash -= cost

    # Create trade record
    trade = Trade()

    trade.sa = sa
    trade.user = fund.user  # TODO: Remove. Required for backward compatibility temp
    trade.fund = fund
    trade.action = "BUY"
    trade.stock = stock
    trade.discovery = discovery
    trade.price = stock.price
    trade.shares = shares
    trade.explanation = explanation
    trade.save()

    # Update holdings
    holding.fund = fund
    holding.user = fund.user  # TODO: Remove. Required for backward compatibility temp
    holding.stock = stock
    # Forward-only population: persist origin discovery for new/legacy-unlinked holdings.
    if discovery is not None and holding.discovery_id is None:
        holding.discovery = discovery
    old_shares = holding.shares
    old_avg = holding.average_price or Decimal('0')
    holding.shares += shares

    if holding.shares > 0:
        total_cost = (old_avg * Decimal(old_shares)) + (stock.price * shares)
        holding.average_price = total_cost / Decimal(holding.shares)
    else:
        holding.average_price = stock.price

    holding.save()
    fund.save()


def execute_buy_exact_shares(
    sa,
    fund,
    stock,
    shares: int,
    explanation: str = "",
    discovery=None,
    *,
    replace_discovery: bool = False,
) -> Optional[Holding]:
    """
    Buy an exact share count at the current quote (cash must cover notional).

    Unlike execute_buy (allowance-based), used for cross-fund transfers.
    When replace_discovery is True, holding.discovery is set to discovery even if
    already set (so the receiving fund uses the new User discovery / default SIs).
    """
    shares = int(shares)
    if shares <= 0:
        logger.info("Trade: %s NOT buying %s — shares must be positive", fund.name, stock.symbol)
        return None

    stock.refresh()
    if not stock.price or stock.price <= 0:
        logger.warning("Trade: no price for %s", stock.symbol)
        return None

    price = stock.price
    cost = price * Decimal(shares)
    if fund.cash < cost:
        logger.info(
            "Trade: %s NOT buying %s shares of %s — need $%s have $%s",
            fund.name,
            shares,
            stock.symbol,
            _fmt_usd(cost),
            _fmt_usd(fund.cash),
        )
        return None

    holding = Holding.objects.filter(fund=fund, stock=stock).first()
    if holding is None:
        holding = Holding()
        holding.shares = 0
        holding.average_price = Decimal("0")

    logger.info(
        "Trade: %s buying %s shares of %s at $%s (exact). Holding %s",
        fund.name,
        shares,
        stock.symbol,
        _fmt_usd(price),
        holding.shares,
    )

    fund.cash -= cost

    trade = Trade()
    trade.sa = sa
    trade.user = fund.user
    trade.fund = fund
    trade.action = "BUY"
    trade.stock = stock
    trade.discovery = discovery
    trade.price = price
    trade.shares = shares
    trade.explanation = (explanation or "")[:256]
    trade.save()

    holding.fund = fund
    holding.user = fund.user
    holding.stock = stock
    if discovery is not None and (holding.discovery_id is None or replace_discovery):
        holding.discovery = discovery

    old_shares = int(holding.shares or 0)
    old_avg = holding.average_price or Decimal("0")
    holding.shares = old_shares + shares
    if holding.shares > 0:
        total_cost = (old_avg * Decimal(old_shares)) + (price * Decimal(shares))
        holding.average_price = total_cost / Decimal(holding.shares)
    else:
        holding.average_price = price

    holding.save()
    fund.save()
    return holding


class TransferError(Exception):
    """Raised when a cross-fund holding transfer cannot proceed."""


def transfer_holding(
    sa,
    from_fund,
    to_fund,
    stock,
    *,
    explanation: Optional[str] = None,
) -> dict:
    """
    Move an entire holding from one fund to another at the current mark.

    - Seller realizes P&L vs average cost (honest sell at mark).
    - Buyer opens/adds at mark with a new User discovery (default SIs).
    - Source discovery / SIs are left untouched for other funds still on that discovery.
    - Risk gate is not applied (ops / sleeve transfer).
    """
    if from_fund.id == to_fund.id:
        raise TransferError("from_fund and to_fund must differ")

    holding = (
        Holding.objects.filter(fund=from_fund, stock=stock, shares__gt=0)
        .select_related("stock", "discovery")
        .first()
    )
    if holding is None:
        raise TransferError(f"{from_fund.name} has no {stock.symbol} holding")

    stock.refresh()
    if not stock.price or stock.price <= 0:
        raise TransferError(f"No usable price for {stock.symbol}")

    shares = int(holding.shares)
    price = stock.price
    notional = price * Decimal(shares)
    avg = holding.average_price or Decimal("0")

    if to_fund.cash < notional:
        raise TransferError(
            f"{to_fund.name} needs ${_fmt_usd(notional)} cash "
            f"(has ${_fmt_usd(to_fund.cash)})"
        )

    from_name = from_fund.name or f"fund#{from_fund.id}"
    to_name = to_fund.name or f"fund#{to_fund.id}"
    sell_explanation = (explanation or f"Transfer to {to_name} at mark")[:256]
    buy_explanation = (explanation or f"Transfer from {from_name} at mark")[:256]
    discovery_explanation = (
        f"Transfer from {from_name} to {to_name} at mark "
        f"({shares} @ ${_fmt_usd(price)})"
    )

    # Import here to avoid circular imports at module load.
    from core.models import Discovery
    from core.services.advisors.user import User

    try:
        advisor_row = Advisor.objects.get(python_class="User")
    except Advisor.DoesNotExist as exc:
        raise TransferError("User advisor row missing") from exc

    with transaction.atomic():
        from_fund = type(from_fund).objects.select_for_update().get(pk=from_fund.pk)
        to_fund = type(to_fund).objects.select_for_update().get(pk=to_fund.pk)

        holding = (
            Holding.objects.select_for_update()
            .filter(fund=from_fund, stock=stock, shares__gt=0)
            .first()
        )
        if holding is None or int(holding.shares) != shares:
            raise TransferError(
                f"{from_name} {stock.symbol} holding changed during transfer"
            )

        stock.refresh()
        if not stock.price or stock.price <= 0:
            raise TransferError(f"No usable price for {stock.symbol}")
        price = stock.price
        notional = price * Decimal(shares)
        if to_fund.cash < notional:
            raise TransferError(
                f"{to_name} needs ${_fmt_usd(notional)} cash "
                f"(has ${_fmt_usd(to_fund.cash)})"
            )

        user_adv = User(advisor_row)
        created = user_adv.discovered(sa, stock.symbol, discovery_explanation)
        if created is None:
            raise TransferError(f"User.discovered failed for {stock.symbol}")

        discovery = (
            Discovery.objects.filter(sa=sa, stock=stock, advisor=advisor_row)
            .order_by("-created", "-id")
            .first()
        )
        if discovery is None:
            raise TransferError(f"No User discovery created for {stock.symbol}")

        execute_sell(sa, from_fund, holding, sell_explanation)
        bought = execute_buy_exact_shares(
            sa,
            to_fund,
            stock,
            shares,
            explanation=buy_explanation,
            discovery=discovery,
            replace_discovery=True,
        )
        if bought is None:
            raise TransferError(f"Buy failed on {to_name} after sell — check cash/price")

    logger.info(
        "Transfer: %s → %s %s x%s @ $%s (avg was $%s)",
        from_name,
        to_name,
        stock.symbol,
        shares,
        _fmt_usd(price),
        _fmt_usd(avg),
    )
    return {
        "symbol": stock.symbol,
        "shares": shares,
        "price": price,
        "notional": notional,
        "from_avg": avg,
        "from_fund": from_name,
        "to_fund": to_name,
        "discovery_id": discovery.id,
        "sell_explanation": sell_explanation,
        "buy_explanation": buy_explanation,
    }
