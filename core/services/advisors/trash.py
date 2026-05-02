"""
Trash Advisor (stub)

Starting scaffold for a bad-sell recovery advisor:
- collect recent "bad" SELL outcomes
- pre-vet quality into watchlist
- trigger rediscovery on upturn confirmation

Logic intentionally minimal for first pass.
"""

import logging

from core.services.advisors.advisor import AdvisorBase, register

logger = logging.getLogger(__name__)


class Trash(AdvisorBase):
    """
    Starter advisor shell.

    Future implementation will:
    1) scan recent SELL losses,
    2) pre-vet quality and add watchlist entries,
    3) rediscover on confirmed upturn.
    """

    def discover(self, sa):
        logger.info("Trash: stub discover() invoked for SA %s", getattr(sa, "id", None))
        return


register(name="Trash", python_class="Trash")
