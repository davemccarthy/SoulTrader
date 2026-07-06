"""
Damaged-quality recovery diagnostic for a single stock.

Scores price damage, business quality, recovery tape, market belief, and news
into a structured result with stage (PASS / WATCH / WARM / RECOVERY) and
decision (PASS / WAIT / BUY READY). Used by the Vulture advisor and lab scripts;
not the same model as health v2 composite assessment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yfinance as yf

from core.services.financial.yahoo import latest_headlines
from core.services.health.consensus import score_consensus_health

MIN_MARKET_CAP = 2_000_000_000
STRONG_MARKET_CAP = 10_000_000_000
MIN_AVG_VOLUME = 750_000
MIN_DOLLAR_VOLUME = 10_000_000
MIN_TURNOVER_RATIO = 0.003

WATCH_THRESHOLD = 9
WARM_THRESHOLD = 13
RECOVERY_THRESHOLD = 16
BUY_READY_MIN_CANDIDATE_QUALITY_PCT = 75.0
BUY_READY_MIN_RECOVERY_CONFIDENCE_PCT = 70.0
SEVERE_DEBT_TO_EQUITY_MAX = 300.0
SEVERE_CURRENT_RATIO_MIN = 0.75
CAP_MARKET_BELIEF_DISPERSION_MAX = 75.0
SEVERE_TARGET_DISPERSION_MAX = 100.0

NEGATIVE_HEADLINE_TERMS = (
    "bankruptcy",
    "chapter 11",
    "going concern",
    "delist",
    "sec probe",
    "sec investigation",
    "fraud",
    "criminal",
    "lawsuit",
    "class action",
    "downgrade",
    "cuts guidance",
    "cut guidance",
    "misses estimates",
    "plunges",
    "crashes",
    "tumbles",
    "liquidity",
)

POSITIVE_HEADLINE_TERMS = (
    "raises guidance",
    "raised guidance",
    "beats estimates",
    "upgrade",
    "upgraded",
    "buy rating",
    "outperform",
    "partnership",
    "approval",
    "record revenue",
    "profit rises",
    "rebound",
    "recovery",
)


@dataclass
class Factor:
    section: str
    label: str
    points: int
    passed: bool
    detail: str = ""
    max_points: int = 0


@dataclass
class DiagnosticResult:
    symbol: str
    status: str
    decision: str
    decision_reason: str
    score: int
    max_score: int
    candidate_quality_pct: Optional[float] = None
    recovery_confidence_pct: Optional[float] = None
    market_belief_pct: Optional[float] = None
    factors: List[Factor] = field(default_factory=list)
    top_for: List[str] = field(default_factory=list)
    top_against: List[str] = field(default_factory=list)
    why_not_buy: List[str] = field(default_factory=list)
    buy_ready_caveats: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    snapshot: Dict[str, Any] = field(default_factory=dict)


# Backward-compatible alias for lab scripts and advisor work-in-progress.
VultureResult = DiagnosticResult


def diagnostic_to_dict(result: DiagnosticResult) -> Dict[str, Any]:
    """JSON-serializable payload for Watchlist.meta or snapshot files."""
    return asdict(result)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if out != out or out in (float("inf"), float("-inf")):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _pct_change(current: Optional[float], prior: Optional[float]) -> Optional[float]:
    if current is None or prior is None or prior <= 0:
        return None
    return (current - prior) / prior * 100.0


def _fmt_money(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:.2f}"


def _fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+.1f}%"


def _normalize_history(hist: pd.DataFrame) -> pd.DataFrame:
    if hist is None or hist.empty:
        return pd.DataFrame()
    out = hist.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.droplevel(-1)
    out.columns = [str(c).lower() for c in out.columns]
    return out.dropna(how="all")


def _history(symbol: str) -> pd.DataFrame:
    try:
        return _normalize_history(
            yf.Ticker(symbol).history(period="1y", interval="1d", raise_errors=False)
        )
    except Exception:
        return pd.DataFrame()


def _info(symbol: str) -> Dict[str, Any]:
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}


def _price_from_info_or_history(info: Dict[str, Any], hist: pd.DataFrame) -> Optional[float]:
    price = _safe_float(
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )
    if price is not None:
        return price
    if not hist.empty and "close" in hist:
        return _safe_float(hist["close"].iloc[-1])
    return None


def _down_from_high(price: Optional[float], high: Optional[float]) -> Optional[float]:
    if price is None or high is None or high <= 0:
        return None
    return (price - high) / high * 100.0


def _has_higher_lows(hist: pd.DataFrame, days: int = 3) -> bool:
    if hist.empty or "low" not in hist or len(hist) < days + 1:
        return False
    lows = hist["low"].tail(days + 1).astype(float).tolist()
    return all(lows[i] > lows[i - 1] for i in range(1, len(lows)))


def _recent_lows_display(hist: pd.DataFrame, days: int = 3) -> str:
    if hist.empty or "low" not in hist or len(hist) < days + 1:
        return "insufficient low history"
    lows = hist["low"].tail(days + 1).astype(float).tolist()
    direction = "rising" if all(lows[i] > lows[i - 1] for i in range(1, len(lows))) else "not rising"
    return f"last {days + 1} lows {direction}: " + ", ".join(f"{low:.2f}" for low in lows)


def _no_new_low(hist: pd.DataFrame, lookback: int = 20, recent_days: int = 5) -> bool:
    if hist.empty or "low" not in hist or len(hist) < lookback:
        return False
    lows = hist["low"].tail(lookback).astype(float)
    recent = lows.tail(recent_days)
    prior = lows.iloc[: -recent_days]
    if prior.empty:
        return False
    return float(recent.min()) > float(prior.min())


def _no_new_low_display(hist: pd.DataFrame, lookback: int = 20, recent_days: int = 5) -> str:
    if hist.empty or "low" not in hist or len(hist) < lookback:
        return "insufficient low history"
    lows = hist["low"].tail(lookback).astype(float)
    recent = lows.tail(recent_days)
    prior = lows.iloc[: -recent_days]
    if prior.empty:
        return "insufficient prior-low history"
    recent_min = float(recent.min())
    prior_min = float(prior.min())
    if recent_min > prior_min:
        return f"recent low {recent_min:.2f} above prior {lookback}-day low {prior_min:.2f}"
    return f"recent low {recent_min:.2f} retested/broke prior {lookback}-day low {prior_min:.2f}"


def _volume_cooling(hist: pd.DataFrame) -> bool:
    if hist.empty or "volume" not in hist or len(hist) < 25:
        return False
    vol = hist["volume"].astype(float)
    recent = float(vol.tail(5).mean())
    prior = float(vol.tail(20).head(15).mean())
    return prior > 0 and recent < prior * 0.85


def _volume_cooling_display(hist: pd.DataFrame) -> str:
    if hist.empty or "volume" not in hist or len(hist) < 25:
        return "insufficient volume history"
    vol = hist["volume"].astype(float)
    recent = float(vol.tail(5).mean())
    prior = float(vol.tail(20).head(15).mean())
    ratio = recent / prior if prior > 0 else None
    if ratio is None:
        return "prior volume unavailable"
    state = "cooling" if ratio < 0.85 else "not cooling"
    return f"5-day avg {state}: {ratio:.2f}x prior 15-day avg"


def _realized_volatility(hist: pd.DataFrame, days: int = 20) -> Optional[float]:
    if hist.empty or "close" not in hist or len(hist) < days + 1:
        return None
    returns = hist["close"].astype(float).pct_change().dropna().tail(days)
    if returns.empty:
        return None
    return float(returns.std() * (252 ** 0.5) * 100.0)


def _target_dispersion_pct(info: Dict[str, Any], price: Optional[float]) -> Optional[float]:
    high = _safe_float(info.get("targetHighPrice"))
    low = _safe_float(info.get("targetLowPrice"))
    if high is None or low is None or price is None or price <= 0 or high < low:
        return None
    return (high - low) / price * 100.0


def _turnover_ratio(avg_volume: Optional[float], shares_outstanding: Optional[float]) -> Optional[float]:
    if avg_volume is None or shares_outstanding is None or shares_outstanding <= 0:
        return None
    return avg_volume / shares_outstanding


def _liquidity_detail(
    avg_volume: Optional[float],
    dollar_volume: Optional[float],
    turnover: Optional[float],
) -> str:
    volume_part = f"avg volume {avg_volume:,.0f}" if avg_volume is not None else "avg volume n/a"
    dollar_part = f"dollar volume {_fmt_money(dollar_volume)}" if dollar_volume is not None else "dollar volume n/a"
    turnover_part = f"turnover {turnover * 100:.2f}%" if turnover is not None else "turnover n/a"
    return f"{volume_part}, {dollar_part}, {turnover_part}"


def _headline_score(symbol: str, limit: int) -> Tuple[int, str, List[str]]:
    headlines = latest_headlines(symbol, limit=limit, max_age_days=7)
    joined = " ".join(headlines).lower()
    negative_hits = [term for term in NEGATIVE_HEADLINE_TERMS if term in joined]
    positive_hits = [term for term in POSITIVE_HEADLINE_TERMS if term in joined]

    if negative_hits:
        return -2, f"negative flags: {', '.join(negative_hits[:3])}", headlines
    if positive_hits:
        return 2, f"positive flags: {', '.join(positive_hits[:3])}", headlines
    if headlines and not headlines[0].lower().startswith(("no ", "error ")):
        return 1, "recent headlines found; no obvious red flags", headlines
    return 0, "no useful recent headline signal", headlines


def _factor(
    factors: List[Factor],
    section: str,
    label: str,
    points: int,
    passed: bool,
    detail: str = "",
) -> None:
    factors.append(
        Factor(
            section=section,
            label=label,
            points=points if passed else 0,
            passed=passed,
            detail=detail,
            max_points=max(points, 0),
        )
    )


def _section_pct(factors: List[Factor], sections: Sequence[str]) -> Optional[float]:
    section_set = set(sections)
    selected = [f for f in factors if f.section in section_set]
    max_score = sum(f.max_points for f in selected)
    if max_score <= 0:
        return None
    score = sum(f.points for f in selected)
    return round(max(0.0, min(100.0, score / max_score * 100.0)), 1)


def _score_band(pct: Optional[float]) -> str:
    if pct is None:
        return "n/a"
    if pct >= 85:
        return "Excellent"
    if pct >= 70:
        return "Strong"
    if pct >= 50:
        return "Mixed"
    return "Weak"


def analyze_symbol(symbol: str, *, headline_limit: int = 3) -> DiagnosticResult:
    sym = symbol.strip().upper()
    hist = _history(sym)
    info = _info(sym)
    price = _price_from_info_or_history(info, hist)
    factors: List[Factor] = []

    high_3m = None
    high_52w = _safe_float(info.get("fiftyTwoWeekHigh"))
    if not hist.empty and "high" in hist:
        high_3m = _safe_float(hist["high"].tail(63).max())
        high_52w = high_52w or _safe_float(hist["high"].tail(252).max())

    damage_3m = _down_from_high(price, high_3m)
    damage_52w = _down_from_high(price, high_52w)

    _factor(
        factors,
        "Price Damage",
        "Down at least 20% from 3-month high",
        3,
        damage_3m is not None and damage_3m <= -20.0,
        f"{_fmt_pct(damage_3m)} from {_fmt_money(high_3m)} high",
    )
    _factor(
        factors,
        "Price Damage",
        "Down at least 30% from 52-week high",
        3,
        damage_52w is not None and damage_52w <= -30.0,
        f"{_fmt_pct(damage_52w)} from {_fmt_money(high_52w)} high",
    )
    extreme_damage = damage_52w is not None and damage_52w <= -70.0
    factors.append(
        Factor(
            "Price Damage",
            "Not in extreme collapse territory",
            -3 if extreme_damage else 1,
            not extreme_damage,
            f"{_fmt_pct(damage_52w)} from 52-week high",
            max_points=1,
        )
    )

    market_cap = _safe_float(info.get("marketCap"))
    avg_volume = _safe_float(info.get("averageVolume") or info.get("averageDailyVolume10Day"))
    shares_outstanding = _safe_float(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))
    dollar_volume = avg_volume * price if avg_volume is not None and price is not None else None
    turnover = _turnover_ratio(avg_volume, shares_outstanding)
    eps = _safe_float(info.get("trailingEps"))
    op_cashflow = _safe_float(info.get("operatingCashflow"))
    free_cashflow = _safe_float(info.get("freeCashflow"))
    debt_to_equity = _safe_float(info.get("debtToEquity"))
    current_ratio = _safe_float(info.get("currentRatio"))

    _factor(
        factors,
        "Quality",
        f"Market cap above {_fmt_money(MIN_MARKET_CAP)}",
        2,
        market_cap is not None and market_cap >= MIN_MARKET_CAP,
        _fmt_money(market_cap),
    )
    _factor(
        factors,
        "Quality",
        f"Market cap above {_fmt_money(STRONG_MARKET_CAP)}",
        1,
        market_cap is not None and market_cap >= STRONG_MARKET_CAP,
        _fmt_money(market_cap),
    )
    _factor(
        factors,
        "Quality",
        "Good trading liquidity",
        1,
        (
            dollar_volume is not None
            and dollar_volume >= MIN_DOLLAR_VOLUME
            and (
                turnover is None
                or turnover >= MIN_TURNOVER_RATIO
                or avg_volume >= MIN_AVG_VOLUME
            )
        ),
        _liquidity_detail(avg_volume, dollar_volume, turnover),
    )
    _factor(
        factors,
        "Quality",
        "Profitable or cash-flow positive",
        3,
        (eps is not None and eps > 0)
        or (op_cashflow is not None and op_cashflow > 0)
        or (free_cashflow is not None and free_cashflow > 0),
        f"EPS {eps if eps is not None else 'n/a'}, OCF {_fmt_money(op_cashflow)}, FCF {_fmt_money(free_cashflow)}",
    )
    balance_stress = (
        (debt_to_equity is not None and debt_to_equity > 250.0)
        or (current_ratio is not None and current_ratio < 0.8)
    )
    severe_balance_stress = (
        (debt_to_equity is not None and debt_to_equity > SEVERE_DEBT_TO_EQUITY_MAX)
        or (current_ratio is not None and current_ratio < SEVERE_CURRENT_RATIO_MIN)
    )
    factors.append(
        Factor(
            "Quality",
            "No obvious balance-sheet stress",
            -2 if balance_stress else 2,
            not balance_stress,
            f"D/E {debt_to_equity if debt_to_equity is not None else 'n/a'}, current ratio {current_ratio if current_ratio is not None else 'n/a'}",
            max_points=2,
        )
    )

    close_5d_ago = None
    latest_close = price
    if not hist.empty and "close" in hist and len(hist) >= 6:
        close_5d_ago = _safe_float(hist["close"].iloc[-6])
        latest_close = _safe_float(hist["close"].iloc[-1]) or price
    gain_5d = _pct_change(latest_close, close_5d_ago)
    vol_20d = _realized_volatility(hist)
    high_vol = vol_20d is not None and vol_20d > 90.0
    no_new_low = _no_new_low(hist)
    raw_higher_lows = _has_higher_lows(hist)
    higher_lows = no_new_low and raw_higher_lows
    volume_cooling = _volume_cooling(hist)

    _factor(
        factors,
        "Recovery",
        "No new 20-day low in recent sessions",
        2,
        no_new_low,
        _no_new_low_display(hist),
    )
    _factor(
        factors,
        "Recovery",
        "5-day price gain above 3%",
        2,
        gain_5d is not None and gain_5d >= 3.0,
        _fmt_pct(gain_5d),
    )
    _factor(
        factors,
        "Recovery",
        "Higher lows forming",
        2,
        higher_lows,
        (
            _recent_lows_display(hist)
            if no_new_low
            else f"blocked by 20-day low failure; {_recent_lows_display(hist)}"
        ),
    )
    _factor(
        factors,
        "Recovery",
        "Volume cooling after selloff",
        1,
        volume_cooling,
        _volume_cooling_display(hist),
    )
    factors.append(
        Factor(
            "Recovery",
            "Volatility not excessive",
            -2 if high_vol else 1,
            not high_vol,
            f"20d annualized volatility {vol_20d:.1f}%" if vol_20d is not None else "n/a",
            max_points=1,
        )
    )

    consensus = score_consensus_health(sym)
    analyst_count = consensus.analyst_count
    upside = consensus.upside_to_mean_pct
    dispersion = _target_dispersion_pct(info, price)
    rec_key = consensus.recommendation_key or "n/a"
    cap_market_belief = dispersion is not None and dispersion > CAP_MARKET_BELIEF_DISPERSION_MAX
    dispersion_note = (
        f"suppressed by {dispersion:.1f}% analyst target dispersion"
        if cap_market_belief
        else ""
    )

    _factor(
        factors,
        "Market Belief",
        "Analyst consensus is Buy or better",
        1,
        not cap_market_belief and rec_key in ("buy", "strong_buy", "outperform", "overweight"),
        dispersion_note or f"{rec_key}, {analyst_count if analyst_count is not None else 'n/a'} analysts",
    )
    _factor(
        factors,
        "Market Belief",
        "Consensus target upside above 20%",
        2,
        not cap_market_belief and upside is not None and upside >= 20.0,
        dispersion_note or _fmt_pct(upside),
    )
    _factor(
        factors,
        "Market Belief",
        "Bearish target still leaves upside",
        1,
        not cap_market_belief
        and consensus.upside_to_low_pct is not None
        and consensus.upside_to_low_pct >= 0.0,
        dispersion_note or _fmt_pct(consensus.upside_to_low_pct),
    )
    wide_dispersion = dispersion is not None and dispersion > 60.0
    severe_dispersion = dispersion is not None and dispersion > SEVERE_TARGET_DISPERSION_MAX
    factors.append(
        Factor(
            "Market Belief",
            "Analyst target dispersion is not too wide",
            -2 if wide_dispersion else 1,
            not wide_dispersion,
            f"{dispersion:.1f}% high-low spread vs price" if dispersion is not None else "n/a",
            max_points=1,
        )
    )

    headline_points, headline_detail, headlines = _headline_score(sym, headline_limit)
    factors.append(
        Factor(
            "News",
            "Recent headlines do not show obvious red flags",
            headline_points,
            headline_points >= 0,
            headline_detail,
            max_points=2,
        )
    )

    score = sum(f.points for f in factors)
    max_score = sum(f.max_points for f in factors)
    recovery_score = sum(f.points for f in factors if f.section == "Recovery")
    negative_news = any(f.section == "News" and f.points < 0 for f in factors)
    candidate_quality_pct = _section_pct(factors, ("Price Damage", "Quality"))
    recovery_confidence_pct = _section_pct(factors, ("Recovery",))
    market_belief_pct = _section_pct(factors, ("Market Belief", "News"))
    risk_flags: List[str] = []
    if severe_balance_stress:
        risk_flags.append(
            f"severe balance-sheet stress (D/E {debt_to_equity if debt_to_equity is not None else 'n/a'}, "
            f"current ratio {current_ratio if current_ratio is not None else 'n/a'})"
        )
    if severe_dispersion:
        risk_flags.append(f"extreme analyst target dispersion ({dispersion:.1f}% high-low spread vs price)")
    if negative_news:
        risk_flags.append("recent negative headline red flag")

    if negative_news:
        status = "PASS"
    elif score >= RECOVERY_THRESHOLD and recovery_score >= 5:
        status = "RECOVERY"
    elif score >= WARM_THRESHOLD and recovery_score >= 3:
        status = "WARM"
    elif score >= WATCH_THRESHOLD:
        status = "WATCH"
    else:
        status = "PASS"

    buy_ready = (
        status == "RECOVERY"
        and candidate_quality_pct is not None
        and candidate_quality_pct >= BUY_READY_MIN_CANDIDATE_QUALITY_PCT
        and recovery_confidence_pct is not None
        and recovery_confidence_pct >= BUY_READY_MIN_RECOVERY_CONFIDENCE_PCT
        and not risk_flags
    )

    if buy_ready:
        decision = "BUY READY"
        decision_reason = "Candidate quality and recovery confirmation are both strong enough for a discovery candidate."
    elif status == "RECOVERY":
        decision = "WAIT"
        if risk_flags:
            decision_reason = f"Recovery structure is present, but blocked by: {'; '.join(risk_flags)}."
        else:
            decision_reason = (
                f"Recovery structure is present, but BUY READY requires candidate quality >= "
                f"{BUY_READY_MIN_CANDIDATE_QUALITY_PCT:.0f}% and recovery confidence >= "
                f"{BUY_READY_MIN_RECOVERY_CONFIDENCE_PCT:.0f}%."
            )
    elif status in ("WATCH", "WARM"):
        decision = "WAIT"
        if risk_flags:
            decision_reason = (
                f"{_score_band(candidate_quality_pct)} long-term candidate, "
                f"but blocked by: {'; '.join(risk_flags)}."
            )
        else:
            decision_reason = (
                f"{_score_band(candidate_quality_pct)} long-term candidate; "
                f"recovery confidence is {_score_band(recovery_confidence_pct).lower()}."
            )
    else:
        decision = "PASS"
        decision_reason = "Candidate does not clear the recovery watch threshold."

    positive = sorted((f for f in factors if f.points > 0), key=lambda f: f.points, reverse=True)
    negative = sorted((f for f in factors if f.points < 0 or not f.passed), key=lambda f: f.points)
    why_not_buy = [
        f"{f.label}: {f.detail}"
        for f in factors
        if f.section == "Recovery" and not f.passed and f.points >= 0
    ]
    if wide_dispersion:
        why_not_buy.append("Analyst targets are too dispersed for high confidence.")
    for flag in risk_flags:
        why_not_buy.append(flag)
    buy_ready_caveats = [
        f"{f.label}: {f.detail}"
        for f in negative
        if f.points >= 0 and f.section in ("Quality", "Market Belief", "Recovery", "News")
    ][:5]

    return DiagnosticResult(
        symbol=sym,
        status=status,
        decision=decision,
        decision_reason=decision_reason,
        score=score,
        max_score=max_score,
        candidate_quality_pct=candidate_quality_pct,
        recovery_confidence_pct=recovery_confidence_pct,
        market_belief_pct=market_belief_pct,
        factors=factors,
        top_for=[f"{f.label} ({f.detail})".strip() for f in positive[:3]],
        top_against=[f"{f.label} ({f.detail})".strip() for f in negative[:3]],
        why_not_buy=why_not_buy[:5],
        buy_ready_caveats=buy_ready_caveats,
        risk_flags=risk_flags,
        snapshot={
            "price": price,
            "market_cap": market_cap,
            "damage_3m_pct": damage_3m,
            "damage_52w_pct": damage_52w,
            "gain_5d_pct": gain_5d,
            "volatility_20d_annualized_pct": vol_20d,
            "recommendation_key": rec_key,
            "upside_to_mean_pct": upside,
            "target_dispersion_pct": dispersion,
            "turnover_ratio": turnover,
            "dollar_volume": dollar_volume,
            "shares_outstanding": shares_outstanding,
            "risk_flags": risk_flags,
            "headlines": headlines,
        },
    )


def _print_factor(factor: Factor) -> None:
    if factor.points > 0:
        marker = "+"
        points = f"+{factor.points}"
    elif factor.points < 0:
        marker = "-"
        points = str(factor.points)
    else:
        marker = "x" if not factor.passed else "."
        points = " 0"
    detail = f" ({factor.detail})" if factor.detail else ""
    print(f"{marker} {factor.label:<48} {points:>3}{detail}")


def _fmt_dimension(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f}% ({_score_band(value)})"


def print_report(result: DiagnosticResult) -> None:
    print("=" * 72)
    print(f"RECOVERY DIAGNOSTIC: {result.symbol}")
    print("=" * 72)
    print(f"Score              : {result.score} / {result.max_score}")
    print(f"Candidate Quality  : {_fmt_dimension(result.candidate_quality_pct)}")
    print(f"Recovery Confidence: {_fmt_dimension(result.recovery_confidence_pct)}")
    print(f"Market Belief      : {_fmt_dimension(result.market_belief_pct)}")
    print(f"Recovery Stage     : {result.status}")
    print(f"Decision           : {result.decision}")
    print(f"Reason             : {result.decision_reason}")
    if result.risk_flags:
        print(f"Risk Flags         : {'; '.join(result.risk_flags)}")

    by_section: Dict[str, List[Factor]] = {}
    for factor in result.factors:
        by_section.setdefault(factor.section, []).append(factor)

    for section, section_factors in by_section.items():
        print(f"\n{section}")
        print("-" * len(section))
        for factor in section_factors:
            _print_factor(factor)

    print("\nTop Reasons For")
    print("---------------")
    for reason in result.top_for or ["None"]:
        print(f"+ {reason}")

    print("\nTop Reasons Against")
    print("-------------------")
    for reason in result.top_against or ["None"]:
        print(f"- {reason}")

    if result.decision != "BUY READY":
        print("\nWhy not BUY today?")
        print("------------------")
        for reason in result.why_not_buy or ["No specific recovery blocker identified."]:
            print(f"- {reason}")
    else:
        print("\nBuy Ready Caveats")
        print("-----------------")
        for reason in result.buy_ready_caveats or ["No material caveats from current scoring rules."]:
            print(f"- {reason}")

    headlines = result.snapshot.get("headlines") or []
    if headlines:
        print("\nRecent Headlines")
        print("----------------")
        for headline in headlines:
            print(f"- {headline}")


def filter_buy_ready(results: Sequence[DiagnosticResult]) -> List[DiagnosticResult]:
    return [r for r in results if r.decision == "BUY READY"]


def print_summary_table(results: Sequence[DiagnosticResult], *, title: str = "Summary") -> None:
    if not results:
        print(f"\n{title}")
        print("-" * len(title))
        print("No symbols.")
        return
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"{'symbol':<8} {'score':>8} {'quality':>8} {'recovery':>9} "
        f"{'belief':>8} {'stage':<8} {'decision'}"
    )
    for result in sorted(results, key=lambda r: r.score, reverse=True):
        q = f"{result.candidate_quality_pct:.0f}%" if result.candidate_quality_pct is not None else "n/a"
        r = f"{result.recovery_confidence_pct:.0f}%" if result.recovery_confidence_pct is not None else "n/a"
        b = f"{result.market_belief_pct:.0f}%" if result.market_belief_pct is not None else "n/a"
        print(
            f"{result.symbol:<8} {result.score:>3}/{result.max_score:<4} "
            f"{q:>8} {r:>9} {b:>8} {result.status:<8} {result.decision}"
        )
