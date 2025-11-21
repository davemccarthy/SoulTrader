"""
Trade Execution Service

Executes buy/sell trades and updates holdings
"""

import logging
from decimal import Decimal
from core.models import Stock, Holding, Discovery, Recommendation, Consensus, Trade, Profile

logger = logging.getLogger(__name__)


# Sell all for now
def execute_sell(sa, user, profile, consensus, holding, explanation):

    # Latest price
    holding.stock.refresh()
    logger.info(f"Trade: {user.username} selling {holding.shares} shares of {holding.stock.symbol} at ${holding.stock.price}.")

    # Transfer funds
    profile.cash += holding.shares * holding.stock.price

    # Delete holding
    holding.delete()

    # Create trade record
    trade = Trade()

    trade.sa = sa
    trade.user = user
    trade.consensus = consensus
    trade.action = "SELL"
    trade.stock = holding.stock
    trade.price = holding.stock.price
    trade.shares = holding.shares
    trade.explanation = explanation
    trade.save()

    profile.save()

def execute_buy(sa, user, consensus, allowance, tot_consensus, stk_consensus):

    # Check for existing stock
    profile = Profile.objects.get(user=user)
    holding = Holding.objects.filter(user=user, stock=consensus.stock).first()

    if holding is None:
        holding = Holding()

    # Calculate potential spend for this stock based on confidence (no more than half allowance for single stock)
    allowance = (stk_consensus / tot_consensus) * allowance

    # Latest price
    consensus.stock.refresh()

    # Verify stock price
    if consensus.stock.price == 0.0:
        logger.warning(f"Trade: no price for {consensus.stock.symbol}")
        return

    # Calculate no. shares to buy
    shares = int(allowance / consensus.stock.price)

    if shares ==  0:
        logger.info(f"Trade: {user.username} NOT buying shares of {consensus.stock.symbol}. Can't afford any")
        return

    # No buy if have shares (surrender allowance for subsequent purchases)
    if shares - holding.shares < 0:
        logger.info(f"Trade: {user.username} NOT buying shares of {consensus.stock.symbol}. Holding {holding.shares} shares already")
        return

    # Return allowance for existing shares TODO
    #profile.allowance += (shares - holding.shares) * consensus.stock.price
    shares -= holding.shares
    cost = shares * consensus.stock.price
    # TODO plus commission

    # Sacrifce stock for better stock
    while profile.cash < cost:
        sacrifice = (
            Holding.objects.filter(user=user, consensus__lt=stk_consensus, consensus__gt=0, shares__gt=0)
            .order_by("consensus")
            .first()
        )

        if sacrifice:
            logger.warning(
                "Sacrificing %s (CS %.2f) for %s (CS %.2f)",
                sacrifice.stock.symbol,
                float(sacrifice.consensus or 0),
                consensus.stock.symbol,
                stk_consensus,
            )
            execute_sell(sa, user, profile, None, sacrifice, f"Sacrificed for better stock {consensus.stock.symbol}")
            profile.refresh_from_db(fields=["cash"])
        else:
            logger.warning("No cash to buy %s", consensus.stock.symbol)
            return

    logger.info(f"Trade: {user.username} buying {shares} shares of {consensus.stock.symbol} at ${consensus.stock.price}. Holding {holding.shares}")

    profile.cash -= cost

    # Create trade record
    trade = Trade()

    trade.sa = sa
    trade.user = user
    trade.consensus = consensus
    trade.action = "BUY"
    trade.stock = consensus.stock
    trade.price = consensus.stock.price
    trade.shares = shares
    trade.save()

    # Update holdings
    holding.user = user
    holding.stock = consensus.stock
    old_shares = holding.shares
    old_avg = holding.average_price or Decimal('0')
    holding.shares += shares
    holding.consensus = stk_consensus

    if holding.shares > 0:
        total_cost = (old_avg * Decimal(old_shares)) + (consensus.stock.price * shares)
        holding.average_price = total_cost / Decimal(holding.shares)
    else:
        holding.average_price = consensus.stock.price
    holding.save()

    profile.save()
