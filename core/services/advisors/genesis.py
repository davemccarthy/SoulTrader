
import random
from core.services.advisors.advisor import AdvisorBase, register

class Genesis(AdvisorBase):

    def discover(self, sa):
        super().discovered(sa, "APPL", "Apple", 0.5,"Windows 95")

    def analyze(self, sa, stock):
        super().recommend(sa,stock, "BUY", random.random())


register(name="Google Genesis", python_class="Genesis")