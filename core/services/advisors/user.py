from decimal import Decimal

from core.services.advisors.advisor import AdvisorBase, register
from core.models import Stock


def _default_user_sell_instructions(stock):
    """Build default sell instructions for user discovery (--discover). Uses stock.price for TARGET_DIMINISHING."""
    if not stock or not stock.price:
        return [
            ("STOP_PERCENTAGE", Decimal("0.95"), None),
            ("PEAKED", Decimal("5.0"), None),
        ]
    target_price = (Decimal(str(stock.price)) * Decimal("1.20")).quantize(Decimal("0.01"))
    return [
        ("STOP_PERCENTAGE", Decimal("0.90"), None),   # Stop loss at 90% of discovery price
        ("TARGET_DIMINISHING", target_price, 10),     # Take profit at 120% diminishing over 14 days
        ("PEAKED", Decimal("7.0"), None),             # Sell when down 5% from peak since purchase
        ("PROFIT_FLAT", Decimal("0.5"), 4),           # Profit hovering around 5% for 5 days
    ]


class User(AdvisorBase):

    def discovered(self, sa, symbol, explanation=None, sell_instructions=None):
        # Use provided explanation or default
        if explanation is None:
            explanation = "Stock discovered by user"
        # Use provided sell instructions or default for user discovery
        if sell_instructions is not None:
            instructions = sell_instructions
        else:
            try:
                stock = Stock.objects.get(symbol=symbol)
            except Stock.DoesNotExist:
                stock = Stock.create(symbol, self.advisor)
            stock.refresh()
            instructions = _default_user_sell_instructions(stock)
        super().discovered(sa=sa, symbol=symbol, explanation=explanation, sell_instructions=instructions)


register(name="User Advisor", python_class="User")
