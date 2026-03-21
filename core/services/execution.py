"""
Trade Execution Service

Executes buy/sell trades and updates holdings
"""

import logging
from decimal import Decimal
from core.models import Holding, Trade

logger = logging.getLogger(__name__)


# Sell all for now
def execute_sell(sa, fund, holding, explanation):

    # Latest price
    holding.stock.refresh()
    logger.info(f"Trade: {fund.name} selling {holding.shares} shares of {holding.stock.symbol} at ${holding.stock.price}. {explanation}")

    # Capture cost basis and references BEFORE deleting holding
    cost_basis = holding.average_price or Decimal('0')
    stock_ref = holding.stock  # Save reference before deleting
    shares_ref = holding.shares  # Save reference before deleting
    sell_price = stock_ref.price  # Save price before deleting

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
    trade.price = sell_price
    trade.shares = shares_ref
    trade.cost = cost_basis  # Store cost basis for P&L calculation
    trade.explanation = explanation
    trade.save()

    fund.save()

def execute_buy(sa, fund, stock, allowance, explanation="", force = False):

    # Check for existing stock
    #profile = Profile.objects.get(user=user)
    holding = Holding.objects.filter(fund=fund, stock=stock).first()

    if holding is None:
        holding = Holding()

    # Check cash
    if allowance > fund.cash:
        logger.warning(f"{fund.name} low on cash ${fund.cash}")
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

    logger.info(f"Trade: {fund.name} buying {shares} shares of {stock.symbol} at ${stock.price}. Holding {holding.shares}")

    fund.cash -= cost

    # Create trade record
    trade = Trade()

    trade.sa = sa
    trade.user = fund.user  # TODO: Remove. Required for backward compatibility temp
    trade.fund = fund
    trade.action = "BUY"
    trade.stock = stock
    trade.price = stock.price
    trade.shares = shares
    trade.explanation = explanation
    trade.save()

    # Update holdings
    holding.fund = fund
    holding.user = fund.user  # TODO: Remove. Required for backward compatibility temp
    holding.stock = stock
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
