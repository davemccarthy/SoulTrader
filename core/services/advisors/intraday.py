"""Legacy Intraday advisor (stub)."""

import logging

from core.services.advisors.advisor import AdvisorBase, register

logger = logging.getLogger(__name__)


class Intraday(AdvisorBase):
    """No-op placeholder; strategy moved to Vulture advisor."""

    def discover(self, sa):
        logger.info("Intraday advisor is stubbed; use Vulture for active logic.")
        return

    def analyze(self, sa, stock):
        logger.debug("Intraday analyze noop for %s", getattr(stock, "symbol", "?"))
        return


register(name="Intraday", python_class="Intraday")

