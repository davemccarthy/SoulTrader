"""
Trade Execution Service

Executes buy/sell trades and updates holdings
"""

import logging
from decimal import Decimal
from core.models import Holding, Trade, Profile

logger = logging.getLogger(__name__)


# Sell all for now
def execute_sell(sa, user, profile, holding, explanation):

    # Latest price
    holding.stock.refresh()
    logger.info(f"Trade: {user.username} selling {holding.shares} shares of {holding.stock.symbol} at ${holding.stock.price}. {explanation}")

    # Capture cost basis and references BEFORE deleting holding
    cost_basis = holding.average_price or Decimal('0')
    stock_ref = holding.stock  # Save reference before deleting
    shares_ref = holding.shares  # Save reference before deleting
    sell_price = stock_ref.price  # Save price before deleting

    # Transfer funds
    profile.cash += holding.shares * holding.stock.price

    # Delete holding
    holding.delete()

    # Create trade record
    trade = Trade()

    trade.sa = sa
    trade.user = user
    trade.action = "SELL"
    trade.stock = stock_ref
    trade.price = sell_price
    trade.shares = shares_ref
    trade.cost = cost_basis  # Store cost basis for P&L calculation
    trade.explanation = explanation
    trade.save()

    profile.save()

def execute_buy(sa, user, stock, allowance, explanation=""):

    # Check for existing stock
    profile = Profile.objects.get(user=user)
    holding = Holding.objects.filter(user=user, stock=stock).first()

    if holding is None:
        holding = Holding()

    # Check cash
    if allowance > profile.cash:
        logger.warning(f"{user.username} low on cash ${profile.cash}")
        allowance = profile.cash

    # Latest price
    stock.refresh()

    # Verify stock price
    if stock.price == 0.0:
        logger.warning(f"Trade: no price for {stock.symbol}")
        return

    # Calculate no. shares to buy
    shares = int(allowance / stock.price)

    if shares ==  0:
        logger.info(f"Trade: {user.username} NOT buying shares of {stock.symbol}. Can't afford any")
        return

    # No buy if have shares (surrender allowance for subsequent purchases)
    if shares - holding.shares <= 0:
        logger.info(f"{user.username} already has {holding.shares} {stock.symbol} shares")
        return

    shares -= holding.shares
    cost = shares * stock.price

    if profile.cash < cost:
        logger.info(f"Trade: {user.username} NOT buying shares of {stock.symbol}. Not enough cash")
        return

    logger.info(f"Trade: {user.username} buying {shares} shares of {stock.symbol} at ${stock.price}. Holding {holding.shares}")

    profile.cash -= cost

    # Create trade record
    trade = Trade()

    trade.sa = sa
    trade.user = user
    trade.action = "BUY"
    trade.stock = stock
    trade.price = stock.price
    trade.shares = shares
    trade.explanation = explanation
    trade.save()

    # Update holdings
    holding.user = user
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
    profile.save()
