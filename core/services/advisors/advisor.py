
import logging
from core.models import Stock, Discovery, Recommendation, Advisor

logger = logging.getLogger(__name__)

class AdvisorBase:

    def __init__(self, name):
        self.name = name

    # Are these two needed?
    def discover(self, sa):
        return

    def analyze(self, sa, stock):
        return

    def discovered(self, sa, symbol, company, confidence, explanation):
        #-- find stock or create stock
        try:
            stock = Stock.objects.get(symbol=symbol)

        except Stock.DoesNotExist:
            stock = Stock()
            stock.symbol = symbol
            stock.company = company
            stock.save()
            # TODO GOOD TIME TO GET STOCK IMAGE

            logger.info(f"{self.name} created stock {stock.symbol}")

        discovery = Discovery()
        discovery.sa = sa
        discovery.stock = stock
        discovery.advisor = Advisor.objects.get(name=self.name)
        discovery.explanation = explanation
        discovery.save()

        logger.info(f"{self.name} discovers {stock.symbol}")

    def recommend(self, sa, stock, confidence, explanation=""):
        # TODO BUY, SELL .etc
        #create recommendation record
        recommendation = Recommendation()
        recommendation.sa = sa
        recommendation.stock = stock
        recommendation.advisor = Advisor.objects.get(name=self.name)
        recommendation.confidence = confidence
        recommendation.explanation = explanation
        recommendation.save()

        logger.info(f"{self.name} scores {stock.symbol} a confidences of {confidence}")


def register(name, python_class):
    try:
        Advisor.objects.get(python_class=python_class)
        logger.info(f"Advisor {name} ({python_class}) already exists, skipping registration")

    except Advisor.DoesNotExist:
        logger.info(f"Created new advisor: {name}")

        advisor = Advisor(name=name, python_class=python_class)
        advisor.save()
