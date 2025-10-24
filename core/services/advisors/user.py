import random
from core.services.advisors.advisor import AdvisorBase, register

class User(AdvisorBase):

    def discovered(self, sa, symbol):
        # TODO get company from sysmbol or collect value from command line --discover
        super().discovered(sa=sa, symbol=symbol, company="Unknown", confidence=10.0, explanation="Stock discovered by user")


register(name="User Advisor", python_class="User")