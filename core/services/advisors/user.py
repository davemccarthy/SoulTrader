from decimal import Decimal

from core.services.advisors.advisor import AdvisorBase, register
from core.models import Stock


def _default_user_sell_instructions(stock):
    if not stock or not stock.price:
        return [
            ("STOP_PERCENTAGE", Decimal("0.95"), None),
            ("PEAKED", Decimal("5.0"), None),
        ]

    return [
        ("STOP_PERCENTAGE", Decimal("0.90"), None),   # Stop loss at 90% of discovery price
        ("PEAKED", Decimal("20.0"), 5.0),             # Sell when down 20% from peak since purchase
        ("PROFIT_FLAT", Decimal("0.5"), 14),           # Profit hovering around 5% for 14 days
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
                if stock is None:
                    return
                stock.refresh()
            else:
                stock.refresh()
            instructions = _default_user_sell_instructions(stock)
        super().discovered(sa=sa, symbol=symbol, explanation=explanation, sell_instructions=instructions)


register(name="User Advisor", python_class="User")
