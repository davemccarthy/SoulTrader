"""
Compare discovery-time Assessment vs latest Assessment for a fund's holdings.

Useful for purge scoring experiments (thesis fade = score dropped since buy).

Delta uses a no-price composite (financial, valuation, intrinsic, consensus, sector).
Missing components are skipped and weights renormalized — not treated as 0.

Usage:
    python manage.py compare_assessments --fund AGR1
    python manage.py compare_assessments --fund AGR1 --refresh
    python manage.py compare_assessments --fund AGR1 --refresh --limit 15
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core.models import Assessment, Holding, Profile
from core.services.health.assess import COMPONENT_MODEL_WEIGHTS, create_assessment_for_stock

# Components used for fade comparison (price excluded — too noisy for cull ranking).
COMPARE_KEYS = (
    "financial",
    "valuation",
    "intrinsic",
    "consensus",
    "sector",
)

# Shown in detail rows only (includes price for diagnosis).
COMPONENT_KEYS = COMPARE_KEYS + ("price",)


def _dec(value) -> Optional[Decimal]:
    if value is None:
        return None
    return Decimal(str(value))


def _fmt(value: Optional[Decimal], width: int = 6) -> str:
    if value is None:
        return f"{'—':>{width}}"
    return f"{float(value):>{width}.1f}"


def _delta(a: Optional[Decimal], b: Optional[Decimal]) -> Optional[Decimal]:
    """latest - discovery (negative = deteriorated)."""
    if a is None or b is None:
        return None
    return a - b


def _no_price_score(assessment: Assessment) -> Optional[Decimal]:
    """
    Weighted composite excluding price. Missing components are skipped and
    remaining weights renormalized (no zero-fill).
    """
    num = Decimal("0")
    den = Decimal("0")
    for key in COMPARE_KEYS:
        raw = getattr(assessment, key, None)
        if raw is None:
            continue
        w = COMPONENT_MODEL_WEIGHTS[key]
        num += Decimal(str(raw)) * w
        den += w
    if den <= 0:
        return None
    return (num / den).quantize(Decimal("0.1"))


class Command(BaseCommand):
    help = (
        "Compare discovery Assessment vs latest Assessment for fund holdings "
        "(no-price composite; missing components skipped/renormalized); "
        "optionally refresh latest via --refresh."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--fund",
            type=str,
            required=True,
            help="Profile.name (exact match), e.g. AGR1",
        )
        parser.add_argument(
            "--refresh",
            action="store_true",
            help="Create a fresh Assessment for each holding stock before comparing.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Max holdings to process (0 = all). Applied after age sort (oldest first).",
        )
        parser.add_argument(
            "--min-delta",
            type=float,
            default=None,
            help="Only show rows where no-price score delta (latest-discovery) <= this (e.g. -2).",
        )

    def handle(self, *args, **options):
        fund_name = (options["fund"] or "").strip()
        fund = Profile.objects.filter(name=fund_name).first()
        if not fund:
            raise CommandError(f'Fund "{fund_name}" not found')

        holdings = list(
            Holding.objects.filter(fund=fund, shares__gt=0)
            .select_related(
                "stock",
                "discovery",
                "discovery__assessment",
                "discovery__advisor",
            )
            .order_by("discovery__created", "created", "id")
        )
        if options["limit"] and options["limit"] > 0:
            holdings = holdings[: options["limit"]]

        if not holdings:
            self.stdout.write(self.style.WARNING(f"{fund.name}: no holdings"))
            return

        refresh = bool(options["refresh"])
        min_delta = options.get("min_delta")
        min_delta_dec = Decimal(str(min_delta)) if min_delta is not None else None

        self.stdout.write(
            f"\n{fund.name}: {len(holdings)} holdings"
            f"{' (refreshing assessments)' if refresh else ' (existing latest only)'}"
            f" | delta = no-price composite (skip/renorm NULLs)\n"
        )

        rows: List[Dict[str, Any]] = []
        skipped = 0

        for i, holding in enumerate(holdings, 1):
            stock = holding.stock
            discovery = holding.discovery
            disc_a = discovery.assessment if discovery else None

            if refresh:
                self.stdout.write(f"  [{i}/{len(holdings)}] refresh {stock.symbol}…", ending="")
                self.stdout.flush()
                try:
                    latest = create_assessment_for_stock(stock)
                except Exception as exc:
                    self.stdout.write(self.style.ERROR(f" fail: {exc}"))
                    skipped += 1
                    continue
                if latest is None:
                    self.stdout.write(self.style.WARNING(" no scores"))
                    skipped += 1
                    continue
                self.stdout.write(f" score={latest.score}")
            else:
                latest = (
                    Assessment.objects.filter(stock=stock)
                    .order_by("-created")
                    .first()
                )

            if disc_a is None or latest is None:
                skipped += 1
                continue

            disc_score = _no_price_score(disc_a)
            latest_score = _no_price_score(latest)
            score_delta = _delta(latest_score, disc_score)

            if min_delta_dec is not None:
                if score_delta is None or score_delta > min_delta_dec:
                    continue

            now = timezone.now()
            created = discovery.created if discovery and discovery.created else holding.created
            days_held = (now - created).days if created else 0

            px = _dec(stock.price) or Decimal("0")
            avg = _dec(holding.average_price) or Decimal("0")
            shares = Decimal(str(holding.shares or 0))
            mv = px * shares
            cost = avg * shares
            pnl_dol = mv - cost

            row = {
                "symbol": stock.symbol,
                "days": days_held,
                "pnl_dol": pnl_dol,
                "disc_score": disc_score,
                "latest_score": latest_score,
                "score_delta": score_delta,
                "same_row": disc_a.id == latest.id,
                "disc_consensus": _dec(disc_a.consensus),
                "latest_consensus": _dec(latest.consensus),
                "consensus_delta": _delta(_dec(latest.consensus), _dec(disc_a.consensus)),
                "components": {
                    k: _delta(_dec(getattr(latest, k)), _dec(getattr(disc_a, k)))
                    for k in COMPONENT_KEYS
                },
                "advisor": (
                    discovery.advisor.name
                    if discovery and discovery.advisor_id
                    else "—"
                ),
            }
            rows.append(row)

        # Most deteriorated first; missing delta last
        rows.sort(
            key=lambda r: (
                r["score_delta"] is None,
                r["score_delta"] if r["score_delta"] is not None else Decimal("0"),
            )
        )

        self.stdout.write("")
        hdr = (
            f"{'SYM':6} {'Days':>4} {'P&L$':>8} "
            f"{'Disc':>6} {'Now':>6} {'Δ':>6} "
            f"{'CnsΔ':>6} {'Same?':>5}  Advisor"
        )
        self.stdout.write(hdr)
        self.stdout.write("-" * len(hdr))

        faded = 0
        improved = 0
        flat = 0
        for r in rows:
            d = r["score_delta"]
            if d is None:
                mark = ""
            elif d < 0:
                faded += 1
                mark = self.style.WARNING(_fmt(d))
            elif d > 0:
                improved += 1
                mark = self.style.SUCCESS(_fmt(d))
            else:
                flat += 1
                mark = _fmt(d)

            same = "yes" if r["same_row"] else "no"
            self.stdout.write(
                f"{r['symbol']:6} {r['days']:4d} {float(r['pnl_dol']):8.0f} "
                f"{_fmt(r['disc_score'])} {_fmt(r['latest_score'])} {mark} "
                f"{_fmt(r['consensus_delta'])} {same:>5}  {r['advisor']}"
            )

        self.stdout.write("")
        self.stdout.write(
            f"Compared {len(rows)} | faded {faded} | flat {flat} | "
            f"improved {improved} | skipped {skipped}"
        )

        # Detail for worst fades (no-price score drop); price shown for diagnosis only
        worst = [r for r in rows if r["score_delta"] is not None and r["score_delta"] < 0][:8]
        if worst:
            self.stdout.write(
                "\nComponent deltas (latest − discovery) for worst no-price fades "
                "(price shown for diagnosis only):"
            )
            for r in worst:
                parts = []
                for k in COMPONENT_KEYS:
                    cd = r["components"].get(k)
                    if cd is not None and cd != 0:
                        parts.append(f"{k[:3]}{_fmt(cd, 5)}")
                detail = " ".join(parts) if parts else "(no component moves)"
                self.stdout.write(
                    f"  {r['symbol']:6} scoreΔ={_fmt(r['score_delta'])}  {detail}"
                )
        self.stdout.write("")
