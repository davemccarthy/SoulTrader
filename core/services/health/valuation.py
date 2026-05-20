"""
Valuation component (20% of final buy score in v2 model).

Scores how expensive the stock is vs sector norms and absolute PEG.
Lower relative multiples → higher score. No LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yfinance as yf

from core.services.health._util import clip_score, linear_map, safe_div, score_relative_multiple
from core.services.health.sectors import resolve_valuation_benchmark

METRIC_WEIGHTS = {
    "forward_pe_vs_sector": 0.35,
    "ev_ebitda_vs_sector": 0.25,
    "peg_ratio": 0.25,
    "price_sales_vs_sector": 0.15,
}

COMPONENT_WEIGHT = 0.20


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
class ValuationHealthResult:
    symbol: str
    score: Optional[float]
    sector: str = ""
    sector_benchmark_key: str = "default"
    benchmarks: Dict[str, float] = field(default_factory=dict)
    metrics: List[MetricResult] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "component": "valuation",
            "component_weight": COMPONENT_WEIGHT,
            "sector": self.sector,
            "sector_benchmark_key": self.sector_benchmark_key,
            "benchmarks": self.benchmarks,
            "score": self.score,
            "metrics": [m.to_dict() for m in self.metrics],
            "missing": self.missing,
            "error": self.error,
        }


def _score_peg(peg: Optional[float]) -> Optional[float]:
    """Lower PEG is better; negative or zero earnings growth → skip."""
    if peg is None or peg <= 0:
        return None
    if peg <= 0.8:
        return 95.0
    if peg <= 1.2:
        return linear_map(peg, in_lo=0.8, in_hi=1.2, out_lo=90, out_hi=75)
    if peg <= 1.8:
        return linear_map(peg, in_lo=1.2, in_hi=1.8, out_lo=75, out_hi=55)
    if peg <= 2.5:
        return linear_map(peg, in_lo=1.8, in_hi=2.5, out_lo=55, out_hi=35)
    if peg <= 4.0:
        return linear_map(peg, in_lo=2.5, in_hi=4.0, out_lo=35, out_hi=15)
    return 10.0


def _f(info: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        v = info.get(k)
        if v is not None:
            try:
                fv = float(v)
                if fv == fv and fv not in (float("inf"), float("-inf")):
                    return fv
            except (TypeError, ValueError):
                continue
    return None


def _infer_peg(info: Dict[str, Any]) -> Optional[float]:
    peg = _f(info, "pegRatio")
    if peg is not None and peg > 0:
        return peg
    pe = _f(info, "forwardPE", "trailingPE")
    growth = _f(info, "earningsGrowth", "revenueGrowth")
    if pe is not None and growth is not None and growth > 0:
        # yfinance growth fields are often decimal (0.12 = 12%)
        g = growth * 100.0 if abs(growth) <= 1.5 else growth
        if g <= 0:
            return None
        return pe / g
    return None


def fetch_valuation_inputs(symbol: str) -> Dict[str, Any]:
    sym = (symbol or "").strip().upper()
    info = yf.Ticker(sym).info or {}
    sector_label, bench = resolve_valuation_benchmark(info.get("sector"))

    forward_pe = _f(info, "forwardPE")
    if forward_pe is None or forward_pe <= 0:
        forward_pe = _f(info, "trailingPE")

    ev_ebitda = _f(info, "enterpriseToEbitda")
    ps = _f(info, "priceToSalesTrailing12Months", "priceToSales")

    pe_rel = safe_div(forward_pe, bench["forward_pe"]) if forward_pe and forward_pe > 0 else None
    ev_rel = safe_div(ev_ebitda, bench["ev_ebitda"]) if ev_ebitda and ev_ebitda > 0 else None
    ps_rel = safe_div(ps, bench["ps"]) if ps and ps > 0 else None
    peg = _infer_peg(info)

    return {
        "symbol": sym,
        "company": (info.get("shortName") or info.get("longName") or sym),
        "sector": info.get("sector") or "",
        "sector_benchmark_key": sector_label,
        "benchmarks": bench,
        "forward_pe": forward_pe,
        "forward_pe_vs_sector": pe_rel,
        "ev_ebitda": ev_ebitda,
        "ev_ebitda_vs_sector": ev_rel,
        "price_sales": ps,
        "price_sales_vs_sector": ps_rel,
        "peg_ratio": peg,
    }


def score_valuation_health(symbol: str) -> ValuationHealthResult:
    sym = (symbol or "").strip().upper()
    if not sym:
        return ValuationHealthResult(symbol="", score=None, error="empty symbol")

    try:
        raw = fetch_valuation_inputs(sym)
    except Exception as e:
        return ValuationHealthResult(symbol=sym, score=None, error=str(e))

    bench = raw["benchmarks"]
    sector_key = raw["sector_benchmark_key"]

    def _rel_display(rel: Optional[float], stock_val: Optional[float], bench_val: float, suffix: str) -> str:
        if rel is None:
            return "N/A"
        stock_s = f"{stock_val:.1f}" if stock_val is not None else "?"
        return f"{stock_s}{suffix} vs {bench_val:.1f} ({rel:.2f}x)"

    builders = [
        (
            "forward_pe_vs_sector",
            "Forward P/E vs sector",
            METRIC_WEIGHTS["forward_pe_vs_sector"],
            raw["forward_pe_vs_sector"],
            _rel_display(raw["forward_pe_vs_sector"], raw["forward_pe"], bench["forward_pe"], "x"),
            score_relative_multiple(raw["forward_pe_vs_sector"]),
        ),
        (
            "ev_ebitda_vs_sector",
            "EV/EBITDA vs sector",
            METRIC_WEIGHTS["ev_ebitda_vs_sector"],
            raw["ev_ebitda_vs_sector"],
            _rel_display(raw["ev_ebitda_vs_sector"], raw["ev_ebitda"], bench["ev_ebitda"], "x"),
            score_relative_multiple(raw["ev_ebitda_vs_sector"]),
        ),
        (
            "peg_ratio",
            "PEG ratio",
            METRIC_WEIGHTS["peg_ratio"],
            raw["peg_ratio"],
            f"{raw['peg_ratio']:.2f}" if raw["peg_ratio"] is not None else "N/A",
            _score_peg(raw["peg_ratio"]),
        ),
        (
            "price_sales_vs_sector",
            "Price/sales vs sector",
            METRIC_WEIGHTS["price_sales_vs_sector"],
            raw["price_sales_vs_sector"],
            _rel_display(raw["price_sales_vs_sector"], raw["price_sales"], bench["ps"], "x"),
            score_relative_multiple(raw["price_sales_vs_sector"]),
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
        return ValuationHealthResult(
            symbol=sym,
            score=None,
            sector=raw.get("sector", ""),
            benchmarks=bench,
            metrics=metrics,
            missing=missing,
            error="no scorable valuation metrics",
        )

    return ValuationHealthResult(
        symbol=sym,
        score=round(weighted_sum / weight_total, 1),
        sector=raw.get("sector", ""),
        sector_benchmark_key=sector_key,
        benchmarks=bench,
        metrics=metrics,
        missing=missing,
    )
