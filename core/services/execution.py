"""
Trade Execution Service

Executes buy/sell trades and updates holdings
"""

import logging
from core.models import Stock, Holding, Discovery, Recommendation, Consensus, Trade, Profile

logger = logging.getLogger(__name__)


# Sell all for now
def execute_sell(sa, user, consensus, holding):
    # TODO bank
    profile = Profile.objects.get(user=user)
    logger.info(f"Trade: {user.username} selling {holding.shares} shares of {consensus.stock.symbol} at ${consensus.stock.price}.")

    # Transfer funds
    profile.cash += holding.shares * consensus.stock.price

    # Delete holding
    holding.delete()

    # Create trade record
    trade = Trade()

    trade.sa = sa
    trade.user = user
    trade.consensus = consensus
    trade.action = "SELL"
    trade.stock = holding.stock
    trade.price = consensus.stock.price
    trade.shares = holding.shares
    trade.save()

    profile.save()

def execute_buy(sa, user, consensus, allowance, tot_consensus, stk_consensus):

    # Check for existing stock
    profile = Profile.objects.get(user=user)
    holding = Holding.objects.filter(user=user, stock=consensus.stock).first()

    if holding is None:
        holding = Holding()

    # Calculate potential spend for this stock based on confidence
    allowance = (stk_consensus / tot_consensus) * allowance
    #print(f"ALLOWANCE {allowance}")
    #print(f"CONSENSUS {stk_consensus}")
    #print(f"TOTAL {tot_consensus}")
    #print(f"PERCENTAGE: {(stk_consensus / tot_consensus) * 100}")
    # TODO MAX PER STOCK BUY

    # Calculate no. shares to buy
    shares = int(allowance / consensus.stock.price)

    # No buy if have shares (surrender allowance for subsequent purchases)
    if shares - holding.shares < 0:
        #profile.allowance += allowance
        logger.info(f"Trade: {user.username} NOT buying shares of {consensus.stock.symbol}. Holding {holding.shares} shares already")
        return

    # Return allowance for existing shares TODO
    #profile.allowance += (shares - holding.shares) * consensus.stock.price
    shares -= holding.shares
    cost = shares * consensus.stock.price
    # TODO plus commission

    logger.info(f"Trade: {user.username} buying {shares} shares of {consensus.stock.symbol} at ${consensus.stock.price}. Holding {holding.shares}")

    profile.cash -= cost
    #profile.allowance -= cost

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
    holding.shares += shares
    holding.average_price = consensus.stock.price
    # TODO calculate average price correctly
    holding.save()

    profile.save()
