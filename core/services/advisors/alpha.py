
import random
from core.services.advisors.advisor import AdvisorBase, register

class Alpha(AdvisorBase):

    def discover(self, sa):
        super().discovered(sa, "GOOGL", "Google", 0.25, "Android")
        return

    def analyze(self, sa, stock):
        super().recommend(sa, stock, "BUY", random.random())


register(name="Alpha Vantage", python_class="Alpha")