"""
Run health v2 + SO assessment on a single ticker.

By default scores live (yfinance) and prints a readable report without writing
to the database. Use --save to persist an Assessment when the Stock exists.

Usage:
    python manage.py assess_stock KLRA
    python manage.py assess_stock PYPL --json
    python manage.py assess_stock KLRA --save
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Dict, Optional

from django.core.management.base import BaseCommand, CommandError

from core.models import Stock
from core.services.health.assess import (
    COMPONENT_MODEL_WEIGHTS,
    COMPONENT_SCORERS,
    composite_from_scores,
    create_assessment_for_stock,
    run_component_results,
)
from core.services.health.risk_matrix import (
    RISK_LEVELS,
    compute_so_snapshot,
    risk_fit_all,
    risk_floors_for,
)
from core.services.health.so_ratings import (
    score_to_opportunity_grade,
    score_to_stability_grade,
    so_composite_from_grades,
    so_grade_pair,
)

COMPONENT_LABELS = {
    "financial": "Financial health",
    "valuation": "Valuation",
    "intrinsic": "Intrinsic valuation",
    "price": "Price position",
    "consensus": "Analyst consensus",
    "sector": "Sector / industry",
}


def _fmt(value: Optional[float], width: int = 5) -> str:
    if value is None:
        return f"{'—':>{width}}"
    return f"{float(value):>{width}.1f}"


def _score_symbol(symbol: str) -> Dict[str, Any]:
    sym = (symbol or "").strip().upper()
    results = run_component_results(sym)
    scores: Dict[str, Optional[float]] = {}
    errors: Dict[str, str] = {}
    for key, _ in COMPONENT_SCORERS:
        result = results.get(key)
        if result is None:
            scores[key] = None
            continue
        scores[key] = getattr(result, "score", None)
        err = getattr(result, "error", None)
        if err:
            errors[key] = str(err)

    composite = composite_from_scores(scores)
    so = compute_so_snapshot(sym, results)
    stability = so.get("stability")
    opportunity = so.get("opportunity")
    stab_g = score_to_stability_grade(stability)
    opp_g = score_to_opportunity_grade(opportunity)
    composite_so = so_composite_from_grades(stab_g, opp_g)

    return {
        "symbol": sym,
        "composite": float(composite) if composite is not None else None,
        "stability": stability,
        "opportunity": opportunity,
        "stability_grade": stab_g.letter if stab_g else None,
        "opportunity_grade": opp_g.letter if opp_g else None,
        "so_pair": so_grade_pair(stab_g, opp_g),
        "so_grade": composite_so.get("letter") if composite_so else None,
        "so_composite_score": composite_so.get("score") if composite_so else None,
        "components": scores,
        "component_errors": errors,
        "so_submetrics": {
            "stab_debt_to_equity": so.get("stab_debt_to_equity"),
            "stab_fcf_margin": so.get("stab_fcf_margin"),
            "stab_operating_margin": so.get("stab_operating_margin"),
            "stab_durability": so.get("stab_durability"),
            "opp_fin_growth": so.get("opp_fin_growth"),
            "opp_price_blend": so.get("opp_price_blend"),
            "opp_valuation_blend": so.get("opp_valuation_blend"),
        },
        "risk_fit": risk_fit_all(stability, opportunity),
        "risk_floors": {
            risk: {
                "so_composite_floor": risk_floors_for(risk).get("so_composite_floor"),
                "so_floor_display": risk_floors_for(risk).get("so_floor_display"),
            }
            for risk in RISK_LEVELS
        },
    }


class Command(BaseCommand):
    help = "Run health v2 + SO assessment on a single stock ticker."

    def add_arguments(self, parser) -> None:
        parser.add_argument("symbol", type=str, help="Ticker symbol, e.g. KLRA or PYPL")
        parser.add_argument(
            "--json",
            action="store_true",
            help="Print full JSON payload instead of a readable report",
        )
        parser.add_argument(
            "--save",
            action="store_true",
            help="Persist a new Assessment row (Stock must already exist)",
        )

    def handle(self, *args, **options) -> None:
        symbol = (options["symbol"] or "").strip().upper()
        if not symbol:
            raise CommandError("symbol is required")

        try:
            payload = _score_symbol(symbol)
        except Exception as exc:
            raise CommandError(f"Assessment failed for {symbol}: {exc}") from exc

        assessment_id = None
        if options["save"]:
            stock = Stock.objects.filter(symbol=symbol).first()
            if stock is None:
                raise CommandError(
                    f"Stock {symbol} not found — create it first, or omit --save for a live score only"
                )
            assessment = create_assessment_for_stock(stock)
            if assessment is None:
                raise CommandError(f"create_assessment_for_stock returned None for {symbol}")
            assessment_id = assessment.id
            payload["assessment_id"] = assessment_id
            # Prefer freshly persisted SO axes when present.
            if assessment.stability is not None:
                payload["stability"] = float(assessment.stability)
            if assessment.opportunity is not None:
                payload["opportunity"] = float(assessment.opportunity)
            if assessment.score is not None:
                payload["composite"] = float(assessment.score)

        if options["json"]:
            self.stdout.write(json.dumps(payload, indent=2, default=str))
            return

        self._print_report(payload, saved_id=assessment_id)

    def _print_report(self, payload: Dict[str, Any], *, saved_id: Optional[int]) -> None:
        symbol = payload["symbol"]
        so_grade = payload.get("so_grade") or "—"
        self.stdout.write(self.style.MIGRATE_HEADING(f"Assessment  {symbol}"))
        if saved_id is not None:
            self.stdout.write(f"Saved Assessment id={saved_id}")
        self.stdout.write(f"GRADE  {so_grade}")
        self.stdout.write("")
        self.stdout.write("Axis         Grade   Score")
        self.stdout.write(
            f"Stability    {(payload.get('stability_grade') or '—'):<5}   "
            f"{_fmt(payload.get('stability'))}"
        )
        self.stdout.write(
            f"Opportunity  {(payload.get('opportunity_grade') or '—'):<5}   "
            f"{_fmt(payload.get('opportunity'))}"
        )
        self.stdout.write(f"Pair         {(payload.get('so_pair') or '—')}")
        self.stdout.write("")

        self.stdout.write("Details                    Wt     Score")
        components = payload.get("components") or {}
        for key, weight in COMPONENT_MODEL_WEIGHTS.items():
            label = COMPONENT_LABELS.get(key, key)
            pct = int(Decimal(str(weight)) * 100)
            self.stdout.write(f"{label:<26}{pct:>2}%    {_fmt(components.get(key))}")

        errors = payload.get("component_errors") or {}
        if errors:
            self.stdout.write("")
            self.stdout.write("Component errors:")
            for key, err in errors.items():
                self.stdout.write(f"  {key}: {err}")

        sub = payload.get("so_submetrics") or {}
        self.stdout.write("")
        self.stdout.write("SO submetrics")
        self.stdout.write(
            f"  debt/eq {_fmt(sub.get('stab_debt_to_equity'))}  "
            f"fcf {_fmt(sub.get('stab_fcf_margin'))}  "
            f"op {_fmt(sub.get('stab_operating_margin'))}  "
            f"dur {_fmt(sub.get('stab_durability'))}"
        )
        self.stdout.write(
            f"  growth {_fmt(sub.get('opp_fin_growth'))}  "
            f"price {_fmt(sub.get('opp_price_blend'))}  "
            f"val {_fmt(sub.get('opp_valuation_blend'))}"
        )

        self.stdout.write("")
        self.stdout.write("Risk band       Floor   Fit")
        fit = payload.get("risk_fit") or {}
        floors = payload.get("risk_floors") or {}
        for risk in RISK_LEVELS:
            floor = (floors.get(risk) or {}).get("so_composite_floor") or "—"
            self.stdout.write(f"{risk.capitalize():<15}{floor:<7} {fit.get(risk, '—')}")

        composite = payload.get("composite")
        if composite is not None:
            self.stdout.write("")
            self.stdout.write(f"Legacy composite score: {composite:.1f}")
