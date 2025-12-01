"""
Advisor modules - auto-imports and exposes advisor classes
"""
import core.services.advisors.alpha as alpha
import core.services.advisors.yahoo as yahoo
import core.services.advisors.finnhub as finnhub
import core.services.advisors.fmp as fmp
import core.services.advisors.polygon as polygon
import core.services.advisors.story as stockstory
import core.services.advisors.fda as fda
import core.services.advisors.insider as insider
import core.services.advisors.user as user
import core.services.advisors.intraday as intraday

# Create advisors namespace
advisors = type('Advisors', (), {
    'alpha': alpha,
    'yahoo': yahoo,
    'user': user,
    'finnhub': finnhub,
    'fmp': fmp,
    'polygon': polygon,
    'stockstory' : stockstory,
    'insider': insider,
    'fda': fda,
    'intraday': intraday,
})()
