from core.services.advisors.advisor import AdvisorBase, register

class Gemini(AdvisorBase):

    def discovered(self, sa, symbol):
        pass


register(name="Google Gemini", python_class="Gemini")