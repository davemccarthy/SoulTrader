"""
Advisor modules - auto-imports and exposes advisor classes
"""
import core.services.advisors.polygon as polygon
import core.services.advisors.story as stockstory
import core.services.advisors.fda as fda
import core.services.advisors.insider as insider
import core.services.advisors.user as user
import core.services.advisors.intraday as intraday
import core.services.advisors.flux as flux

# Create advisors namespace
advisors = type('Advisors', (), {
    'user': user,
    'polygon': polygon,
    'stockstory' : stockstory,
    'insider': insider,
    'fda': fda,
    'intraday': intraday,
    'flux': flux,
})()
