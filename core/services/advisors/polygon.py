from core.services.advisors.advisor import AdvisorBase, register

class Polygon(AdvisorBase):

    def discovered(self, sa, symbol):
        pass


register(name="Polygon.io", python_class="Polygon")