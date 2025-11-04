
import logging
from core.models import Stock, Discovery, Recommendation, Advisor

logger = logging.getLogger(__name__)

class AdvisorBase:

    def __init__(self, advisor):
        self.advisor = advisor

    # Are these two needed?
    def discover(self, sa):
        return

    def analyze(self, sa, stock):
        return

    def discovered(self, sa, symbol, company, explanation):
        #-- find stock or create stock
        try:
            stock = Stock.objects.get(symbol=symbol)

        except Stock.DoesNotExist:
            stock = Stock()
            stock.symbol = symbol
            # Let yahoo fix this later
            # stock.company = company
            stock.advisor = self.advisor
            stock.save()

            logger.info(f"{self.advisor.name} created stock {stock.symbol}")

        # Check for repeating discovery
        if Discovery.objects.filter(advisor=self.advisor,stock=stock): # TODO sa=sa-1
            logger.info(f"{self.advisor.name} already discovered stock {stock.symbol}")
            return stock

        # Create new Discovery record
        discovery = Discovery()
        discovery.sa = sa
        discovery.stock = stock
        discovery.advisor = self.advisor
        discovery.explanation = explanation[:500]
        discovery.save()

        logger.info(f"{self.advisor.name} discovers {stock.symbol}")
        return stock

    def recommend(self, sa, stock, confidence, explanation=""):

        recommendation = Recommendation()
        recommendation.sa = sa
        recommendation.stock = stock
        recommendation.advisor = self.advisor
        recommendation.confidence = confidence
        recommendation.explanation = explanation
        recommendation.save()

        logger.info(f"{self.advisor.name} scores {stock.symbol} a confidences of {confidence:.2f}")


def register(name, python_class):
    try:
        Advisor.objects.get(python_class=python_class)

    except Advisor.DoesNotExist:
        logger.info(f"Created new advisor: {name}")

        advisor = Advisor(name=name, python_class=python_class)
        advisor.save()
