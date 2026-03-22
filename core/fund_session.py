"""
Session-backed current fund (Profile) for UI scope.

Login sets ``request.session[FUND_SESSION_KEY]`` to the first enabled profile's pk.
Other views read it via ``get_current_fund(request)``.
"""
from __future__ import annotations

from typing import Optional

from django.http import HttpRequest

from core.models import Profile

FUND_SESSION_KEY = "fund_id"


def get_default_fund() -> Optional[Profile]:
    """First enabled profile by primary key (stable default)."""
    return Profile.objects.filter(enabled=True).order_by("id").first()


def init_fund_session_after_login(request: HttpRequest) -> None:
    """Set session fund to the default (first enabled profile). Call after login()."""
    fund = get_default_fund()
    if fund is not None:
        request.session[FUND_SESSION_KEY] = fund.pk
    elif FUND_SESSION_KEY in request.session:
        del request.session[FUND_SESSION_KEY]


def get_current_fund(request: HttpRequest) -> Optional[Profile]:
    """
    Profile for the current UI scope.

    Uses session fund id when valid; otherwise falls back to ``get_default_fund()``
    and stores it in session so navigation stays consistent (e.g. pre-login sessions).
    """
    if not getattr(request, "user", None) or not request.user.is_authenticated:
        return None

    fund_id = request.session.get(FUND_SESSION_KEY)
    fund = None
    if fund_id is not None:
        try:
            fund = Profile.objects.filter(pk=int(fund_id), enabled=True).first()
        except (TypeError, ValueError):
            fund = None

    if fund is None:
        fund = get_default_fund()
        if fund is not None:
            request.session[FUND_SESSION_KEY] = fund.pk

    return fund


def set_current_fund(request: HttpRequest, fund_id) -> bool:
    """
    Validate fund_id refers to an enabled Profile and set session.

    Returns True if session was updated, False if invalid.
    """
    try:
        pk = int(fund_id)
    except (TypeError, ValueError):
        return False
    fund = Profile.objects.filter(pk=pk, enabled=True).first()
    if fund is None:
        return False
    request.session[FUND_SESSION_KEY] = fund.pk
    return True
