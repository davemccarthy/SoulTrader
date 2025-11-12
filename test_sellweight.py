from decimal import *


def test_sellweight2(investment, cash):

    cash_ratio = cash / (investment+cash)
    print(f"cash_ratio={cash_ratio}")

    print(f"cash_ratio={min((cash / (investment + cash)) * 10, 1.0)}")
    return min(cash_ratio * 10, 1.0)

    #thresold_ratio = 0.66
    cash_ratio = cash / investment
    print(f"cash_ratio={cash_ratio}")

    #if cash_ratio < thresold_ratio:
    #    return 1.0

    weight = 1.0 - (cash_ratio / 5)

    return weight
    total_wealth = investment + cash
    weight_zone = total_wealth - (total_wealth * 0.5)

    print(f"weight_zone = {weight_zone}")

    if investment < weight_zone:
        return 1.0


def test_sellweight(investment, cash):
    thresold_per = 50
    cash_per = (cash * 100) / investment
    print(f"cash_per={cash_per}")
    if cash_per < thresold_per:
        return 1.0

    weight = 100 - (cash_per - thresold_per)

    return weight
    total_wealth = investment + cash
    weight_zone = total_wealth - (total_wealth * 0.5)

    print(f"weight_zone = {weight_zone}")

    if investment < weight_zone:
        return 1.0



    cash_ratio = cash / weight_zone
    print(f'cash_p = {cash_ratio}')

    weight = cash_ratio / 100
    print(f"weight={weight}")