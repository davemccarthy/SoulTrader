"""Display helpers for discovery/trade explanation strings (stored format unchanged)."""

import re

from django import template

register = template.Library()

_LEAD_ARTICLE = re.compile(r"^\s*Article:\s*", re.IGNORECASE)


@register.filter(name="strip_article_lead")
def strip_article_lead(value):
    """
    Remove a leading 'Article:' prefix from the start of the explanation for display.
    Parsing still uses the raw explanation elsewhere; this is presentation-only.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return _LEAD_ARTICLE.sub("", value.strip(), count=1).strip()
