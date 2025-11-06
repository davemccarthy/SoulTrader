"""
Advisor modules - auto-imports and exposes advisor classes
"""
import core.services.advisors.alpha as alpha
import core.services.advisors.yahoo as yahoo
import core.services.advisors.finnhub as finnhub
import core.services.advisors.fmp as fmp
import core.services.advisors.gemini as gemini
import core.services.advisors.polygon as polygon
import core.services.advisors.story as stockstory
import core.services.advisors.user as user

# Create advisors namespace
advisors = type('Advisors', (), {
    'alpha': alpha,
    'yahoo': yahoo,
    'user': user,
    'finnhub': finnhub,
    'fmp': fmp,
    'gemini': gemini,
    'polygon': polygon,
    'stockstory' : stockstory,
})()
