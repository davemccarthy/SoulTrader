
from core.services.advisors.advisor import AdvisorBase, register

class User(AdvisorBase):

    def discovered(self, sa, symbol, explanation=None):
        # Use provided explanation or default
        if explanation is None:
            explanation = "Stock discovered by user"
        super().discovered(sa=sa, symbol=symbol, company="Unknown", explanation=explanation)


register(name="User Advisor", python_class="User")