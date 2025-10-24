"""
Advisor modules - auto-imports and exposes advisor classes
"""
import core.services.advisors.alpha as alpha
import core.services.advisors.yahoo as yahoo
import core.services.advisors.genesis as genesis
import core.services.advisors.user as user

# Create advisors namespace
advisors = type('Advisors', (), {
    'alpha': alpha,
    'yahoo': yahoo,
    'user': user,
    'genesis': genesis,
})()
