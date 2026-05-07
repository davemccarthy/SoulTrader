from core.services.advisors.advisor import AdvisorBase, register


class User(AdvisorBase):
    pass


register(name="User Advisor", python_class="User")
