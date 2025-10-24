
import random
from core.services.advisors.advisor import AdvisorBase, register

class Yahoo(AdvisorBase):

    def discover(self, sa):
        super().discovered(sa, "MSFT", "Microsoft", 0.5,"Windows 95")

    def analyze(self, sa, stock):
        super().recommend(sa,stock, "SELL", random.random())


register(name="Yahoo Finances", python_class="Yahoo")