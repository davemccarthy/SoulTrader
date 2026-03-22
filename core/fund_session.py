"""
Session-backed current fund (Profile) for UI scope.

``fund_id`` in session:
  - ``0`` means no fund selected (explicit).
  - Positive integer: enabled Profile pk.

Login sets ``0``. Visiting the Funds page (GET) resets to ``0``.
"""
from __future__ import annotations

from typing import Optional

from django.http import HttpRequest

from core.models import Profile

FUND_SESSION_KEY = "fund_id"
NO_FUND_SESSION_VALUE = 0


def get_default_fund() -> Optional[Profile]:
    """First enabled profile by primary key (for management commands / CLI)."""
    return Profile.objects.filter(enabled=True).order_by("id").first()


def init_fund_session_after_login(request: HttpRequest) -> None:
    """After login: no fund selected until user picks one on Funds page."""
    request.session[FUND_SESSION_KEY] = NO_FUND_SESSION_VALUE


def get_current_fund(request: HttpRequest) -> Optional[Profile]:
    """
    Active fund for the UI, or None when ``fund_id`` is 0 / missing / invalid.

    Does not auto-select a default fund.
    """
    if not getattr(request, "user", None) or not request.user.is_authenticated:
        return None

    raw = request.session.get(FUND_SESSION_KEY)
    if raw is None:
        return None
    try:
        fid = int(raw)
    except (TypeError, ValueError):
        return None
    if fid == NO_FUND_SESSION_VALUE:
        return None

    return Profile.objects.filter(pk=fid, enabled=True).first()


def set_current_fund(request: HttpRequest, fund_id) -> bool:
    """
    Validate fund_id refers to an enabled Profile and set session.

    Returns True if session was updated, False if invalid.
    """
    try:
        pk = int(fund_id)
    except (TypeError, ValueError):
        return False
    if pk <= NO_FUND_SESSION_VALUE:
        return False
    fund = Profile.objects.filter(pk=pk, enabled=True).first()
    if fund is None:
        return False
    request.session[FUND_SESSION_KEY] = fund.pk
    return True


def clear_fund_session(request: HttpRequest) -> None:
    """Set session to no active fund (e.g. Funds page GET)."""
    request.session[FUND_SESSION_KEY] = NO_FUND_SESSION_VALUE
