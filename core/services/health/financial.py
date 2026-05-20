"""
Financial health component (20% of final buy score in v2 model).

Measures business quality independent of share price.
Data: yfinance statements + info (no LLM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yfinance as yf

from core.services.health._util import clip_score, linear_map, pct_change, safe_div

# Sub-metric weights (sum = 1.0) — aligned with v2 spec
METRIC_WEIGHTS = {
    "revenue_growth": 0.20,
    "eps_growth": 0.20,
    "operating_margin": 0.15,
    "fcf_margin": 0.20,
    "debt_to_equity": 0.15,
    "return_on_equity": 0.10,
}

COMPONENT_WEIGHT = 0.20  # share of final buy score when all components exist


@dataclass
class MetricResult:
    key: str
    label: str
    weight: float
    raw: Any = None
    raw_display: str = "N/A"
    score: Optional[float] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "weight": self.weight,
            "raw": self.raw,
            "raw_display": self.raw_display,
            "score": self.score,
            "note": self.note,
        }


@dataclass
class FinancialHealthResult:
    symbol: str
    score: Optional[float]
    metrics: List[MetricResult] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "component": "financial",
            "component_weight": COMPONENT_WEIGHT,
            "score": self.score,
            "metrics": [m.to_dict() for m in self.metrics],
            "missing": self.missing,
            "error": self.error,
        }


def _score_growth_yoy(fraction: Optional[float]) -> Optional[float]:
    """YoY growth as fraction (+0.15 = +15%)."""
    if fraction is None:
        return None
    pct = fraction * 100.0
    if pct <= -20:
        return 10.0
    if pct <= -5:
        return linear_map(pct, in_lo=-20, in_hi=-5, out_lo=10, out_hi=35)
    if pct <= 0:
        return linear_map(pct, in_lo=-5, in_hi=0, out_lo=35, out_hi=50)
    if pct <= 10:
        return linear_map(pct, in_lo=0, in_hi=10, out_lo=50, out_hi=70)
    if pct <= 25:
        return linear_map(pct, in_lo=10, in_hi=25, out_lo=70, out_hi=90)
    if pct <= 40:
        return linear_map(pct, in_lo=25, in_hi=40, out_lo=90, out_hi=98)
    return 98.0


def _score_operating_margin(margin: Optional[float]) -> Optional[float]:
    if margin is None:
        return None
    pct = margin * 100.0
    if pct < 0:
        return linear_map(pct, in_lo=-30, in_hi=0, out_lo=5, out_hi=35)
    if pct <= 10:
        return linear_map(pct, in_lo=0, in_hi=10, out_lo=40, out_hi=60)
    if pct <= 20:
        return linear_map(pct, in_lo=10, in_hi=20, out_lo=60, out_hi=80)
    if pct <= 35:
        return linear_map(pct, in_lo=20, in_hi=35, out_lo=80, out_hi=95)
    return 95.0


def _score_fcf_margin(margin: Optional[float]) -> Optional[float]:
    if margin is None:
        return None
    pct = margin * 100.0
    if pct < 0:
        return linear_map(pct, in_lo=-20, in_hi=0, out_lo=10, out_hi=40)
    if pct <= 5:
        return linear_map(pct, in_lo=0, in_hi=5, out_lo=45, out_hi=60)
    if pct <= 15:
        return linear_map(pct, in_lo=5, in_hi=15, out_lo=60, out_hi=80)
    if pct <= 25:
        return linear_map(pct, in_lo=15, in_hi=25, out_lo=80, out_hi=95)
    return 95.0


def _score_debt_to_equity(dte: Optional[float]) -> Optional[float]:
    """Lower leverage scores higher."""
    if dte is None:
        return None
    if dte < 0:
        dte = 0.0
    if dte <= 0.3:
        return linear_map(dte, in_lo=0, in_hi=0.3, out_lo=95, out_hi=88)
    if dte <= 1.0:
        return linear_map(dte, in_lo=0.3, in_hi=1.0, out_lo=88, out_hi=70)
    if dte <= 2.5:
        return linear_map(dte, in_lo=1.0, in_hi=2.5, out_lo=70, out_hi=40)
    if dte <= 5.0:
        return linear_map(dte, in_lo=2.5, in_hi=5.0, out_lo=40, out_hi=15)
    return 10.0


def _score_roe(roe: Optional[float]) -> Optional[float]:
    if roe is None:
        return None
    pct = roe * 100.0 if abs(roe) <= 1.5 else roe  # yfinance may return fraction or %
    if pct < 0:
        return linear_map(pct, in_lo=-20, in_hi=0, out_lo=10, out_hi=35)
    if pct <= 10:
        return linear_map(pct, in_lo=0, in_hi=10, out_lo=40, out_hi=60)
    if pct <= 20:
        return linear_map(pct, in_lo=10, in_hi=20, out_lo=60, out_hi=82)
    if pct <= 35:
        return linear_map(pct, in_lo=20, in_hi=35, out_lo=82, out_hi=95)
    return 95.0


def _fmt_pct(fraction: Optional[float]) -> str:
    if fraction is None:
        return "N/A"
    return f"{fraction * 100:+.1f}%"


def _fmt_ratio(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def fetch_financial_inputs(symbol: str) -> Dict[str, Any]:
    """Pull raw inputs from yfinance (latest vs prior annual where needed)."""
    sym = (symbol or "").strip().upper()
    t = yf.Ticker(sym)
    info = t.info or {}

    bs = t.balance_sheet.transpose()
    inc = t.financials.transpose()
    cf = t.cashflow.transpose()

    latest_bs = bs.iloc[0] if len(bs) else None
    prior_bs = bs.iloc[1] if len(bs) > 1 else None
    latest_inc = inc.iloc[0] if len(inc) else None
    prior_inc = inc.iloc[1] if len(inc) > 1 else None
    latest_cf = cf.iloc[0] if len(cf) else None

    if latest_bs is None or latest_inc is None:
        raise ValueError("Missing balance sheet or income statement")

    revenue = latest_inc.get("Total Revenue")
    prior_revenue = prior_inc.get("Total Revenue") if prior_inc is not None else None

    net_income = latest_inc.get("Net Income")
    prior_net_income = prior_inc.get("Net Income") if prior_inc is not None else None

    operating_income = latest_inc.get("Operating Income") or latest_inc.get("EBIT")

    shares = info.get("sharesOutstanding")
    if not shares and info.get("marketCap") and info.get("currentPrice"):
        shares = safe_div(info.get("marketCap"), info.get("currentPrice"))

    eps = safe_div(net_income, shares) if shares else None
    prior_eps = None
    if prior_inc is not None and shares:
        prior_eps = safe_div(prior_inc.get("Net Income"), shares)

    # FCF: prefer Free Cash Flow line; else OCF - capex
    fcf = latest_cf.get("Free Cash Flow") if latest_cf is not None else None
    if fcf is None and latest_cf is not None:
        ocf = latest_cf.get("Total Cash From Operating Activities") or latest_cf.get(
            "Operating Cash Flow"
        )
        capex = latest_cf.get("Capital Expenditure") or latest_cf.get("Capital Expenditures")
        if ocf is not None and capex is not None:
            fcf = float(ocf) + float(capex)  # capex usually negative

    total_assets = latest_bs.get("Total Assets")
    total_liab = latest_bs.get("Total Liab") or latest_bs.get("Total Liabilities")
    ltd = latest_bs.get("Long Term Debt") or 0
    std = latest_bs.get("Short Long Term Debt") or latest_bs.get("Short Term Debt") or 0
    equity = (total_assets - total_liab) if total_assets and total_liab else latest_bs.get(
        "Stockholders Equity"
    ) or latest_bs.get("Total Stockholder Equity")

    debt = float(ltd or 0) + float(std or 0)
    debt_to_equity = safe_div(debt, equity)

    roe_info = info.get("returnOnEquity")
    roe_calc = safe_div(net_income, equity)

    return {
        "symbol": sym,
        "company": (info.get("shortName") or info.get("longName") or sym),
        "revenue_growth": pct_change(revenue, prior_revenue),
        "eps_growth": pct_change(eps, prior_eps) if eps is not None and prior_eps is not None else pct_change(net_income, prior_net_income),
        "operating_margin": safe_div(operating_income, revenue),
        "fcf_margin": safe_div(fcf, revenue),
        "debt_to_equity": debt_to_equity,
        "return_on_equity": roe_info if roe_info is not None else roe_calc,
        "revenue": revenue,
        "net_income": net_income,
        "fcf": fcf,
    }


def score_financial_health(symbol: str) -> FinancialHealthResult:
    """
    Compute financial health sub-score (0–100).

    Returns weighted average of normalized metrics; missing metrics are
    excluded and weights renormalized over available inputs.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return FinancialHealthResult(symbol="", score=None, error="empty symbol")

    try:
        raw = fetch_financial_inputs(sym)
    except Exception as e:
        return FinancialHealthResult(symbol=sym, score=None, error=str(e))

    builders = [
        (
            "revenue_growth",
            "Revenue growth (YoY)",
            METRIC_WEIGHTS["revenue_growth"],
            raw["revenue_growth"],
            _fmt_pct(raw["revenue_growth"]),
            _score_growth_yoy(raw["revenue_growth"]),
        ),
        (
            "eps_growth",
            "EPS / earnings growth (YoY)",
            METRIC_WEIGHTS["eps_growth"],
            raw["eps_growth"],
            _fmt_pct(raw["eps_growth"]),
            _score_growth_yoy(raw["eps_growth"]),
        ),
        (
            "operating_margin",
            "Operating margin",
            METRIC_WEIGHTS["operating_margin"],
            raw["operating_margin"],
            _fmt_pct(raw["operating_margin"]),
            _score_operating_margin(raw["operating_margin"]),
        ),
        (
            "fcf_margin",
            "Free cash flow margin",
            METRIC_WEIGHTS["fcf_margin"],
            raw["fcf_margin"],
            _fmt_pct(raw["fcf_margin"]),
            _score_fcf_margin(raw["fcf_margin"]),
        ),
        (
            "debt_to_equity",
            "Debt / equity",
            METRIC_WEIGHTS["debt_to_equity"],
            raw["debt_to_equity"],
            _fmt_ratio(raw["debt_to_equity"]),
            _score_debt_to_equity(raw["debt_to_equity"]),
        ),
        (
            "return_on_equity",
            "Return on equity",
            METRIC_WEIGHTS["return_on_equity"],
            raw["return_on_equity"],
            _fmt_pct(raw["return_on_equity"]) if raw["return_on_equity"] is not None and abs(raw["return_on_equity"]) <= 1.5 else (
                f"{raw['return_on_equity']:.1f}%" if raw["return_on_equity"] is not None else "N/A"
            ),
            _score_roe(raw["return_on_equity"]),
        ),
    ]

    metrics: List[MetricResult] = []
    missing: List[str] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for key, label, w, raw_val, raw_disp, sub_score in builders:
        m = MetricResult(key=key, label=label, weight=w, raw=raw_val, raw_display=raw_disp)
        if sub_score is None:
            m.note = "insufficient data"
            missing.append(key)
        else:
            m.score = round(sub_score, 1)
            weighted_sum += sub_score * w
            weight_total += w
        metrics.append(m)

    if weight_total <= 0:
        return FinancialHealthResult(
            symbol=sym,
            score=None,
            metrics=metrics,
            missing=missing,
            error="no scorable financial metrics",
        )

    composite = weighted_sum / weight_total
    return FinancialHealthResult(
        symbol=sym,
        score=round(composite, 1),
        metrics=metrics,
        missing=missing,
    )
