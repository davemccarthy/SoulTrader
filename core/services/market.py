from datetime import datetime, timedelta

import pytz


def _observed_market_holiday(day):
    """NYSE-style observation for fixed-date full-day holidays."""
    if day.weekday() == 5:  # Saturday observed on Friday
        return day - timedelta(days=1)
    if day.weekday() == 6:  # Sunday observed on Monday
        return day + timedelta(days=1)
    return day


def _nth_weekday(year, month, weekday, n):
    day = datetime(year, month, 1).date()
    days_until_weekday = (weekday - day.weekday()) % 7
    return day + timedelta(days=days_until_weekday + (n - 1) * 7)


def _last_weekday(year, month, weekday):
    if month == 12:
        day = datetime(year + 1, 1, 1).date() - timedelta(days=1)
    else:
        day = datetime(year, month + 1, 1).date() - timedelta(days=1)
    return day - timedelta(days=(day.weekday() - weekday) % 7)


def _good_friday(year):
    """Return Good Friday using the Gregorian Easter calculation."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime(year, month, day).date() - timedelta(days=2)


def is_full_day_market_holiday(day):
    """Full-day US market holidays; does not model early closes."""
    year = day.year
    holidays = {
        _observed_market_holiday(datetime(year, 1, 1).date()),  # New Year's Day
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
        _good_friday(year),
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed_market_holiday(datetime(year, 6, 19).date()),  # Juneteenth
        _observed_market_holiday(datetime(year, 7, 4).date()),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed_market_holiday(datetime(year, 12, 25).date()),  # Christmas
    }
    return day in holidays


def market_open():
    """
    Check market open status.

    Returns:
        int: Minutes until market opens (negative = not open yet, positive = already open)
             Examples:
             - -30 = market opens in 30 minutes
             - 0 = market just opened
             - +30 = market has been open for 30 minutes
             - None = market is closed (weekend, holiday, or after hours)
    """
    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(et)

    # Check if it's a weekday (Monday=0, Sunday=6)
    if now_et.weekday() >= 5:  # Saturday or Sunday
        return None
    if is_full_day_market_holiday(now_et.date()):
        return None

    # Market hours: 9:30 AM - 4:00 PM ET
    market_open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    # If after market close, return None (market closed)
    if now_et.time() >= market_close_time.time():
        return None

    # Calculate minutes until/from market open
    minutes_diff = (now_et - market_open_time).total_seconds() / 60

    return int(minutes_diff)


def in_opening_noise_window(minutes: int = 60) -> bool:
    """True during the first `minutes` after the 9:30 ET open (regular session)."""
    status = market_open()
    return status is not None and 0 <= status < minutes
