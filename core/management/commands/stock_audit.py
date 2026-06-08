"""
Stock audit: generic analyst-style consensus via Gemini (Vunder's ask_gemini for transport).

Holdings are scoped by fund (Profile), not user. Up to --limit tickers are chosen at random.

Usage:
    python manage.py stock_audit --dry-run --limit 5
    python manage.py stock_audit --fund EXP1 --limit 8
    python manage.py stock_audit --at-loss-only --limit 5 --dry-run
    python manage.py stock_audit --symbols AAPL,MSFT --dry-run
    python manage.py stock_audit --limit 3 --context "Extra framing appended to scenario." --dry-run
"""

from __future__ import annotations

import json
import random
from decimal import Decimal
from typing import Any, Dict, List

from django.core.management.base import BaseCommand

from core.models import Holding, Stock
from core.services.llm.gemini import ask_gemini as llm_ask_gemini


def build_consensus_prompt(*, scenario: str, context_lines: List[str]) -> str:
    """Institutional-style analyst consensus prompt (Vunder-compatible JSON shape)."""
    context_block = "\n".join(context_lines)
    return f"""
You are an equity research assistant producing institutional-quality summaries.

For each ticker below, infer the CURRENT consensus-style recommendation
("Strong Buy", "Buy", "Hold", "Sell") as seen in mainstream equity research.

IMPORTANT:
- Do NOT default to generic summaries.
- Explicitly weigh the dominant bull vs bear arguments before deciding.
- Reflect what would ACTUALLY drive an analyst's rating (growth, margins, valuation, macro sensitivity, etc.).
- Avoid vague phrasing like "mixed outlook" unless truly justified.

Use the FACT LINES only as supporting hints. If facts are sparse, infer from widely known characteristics of the company/sector and LOWER confidence accordingly.

Scenario: {scenario}

Stocks and context (one ticker per line):
{context_block}

TASK — for each ticker, return:
1) consensus: exactly one of "Strong Buy", "Buy", "Hold", "Sell".
2) summary: ONE sharp sentence that includes:
   - the PRIMARY driver of the rating
   - and the MAIN limiting factor or risk
3) confidence: number 0.0–1.0 based on:
   - 0.9+ = very strong, widely agreed consensus
   - 0.7–0.89 = solid but not unanimous
   - 0.5–0.69 = mixed or uncertain
   - <0.5 = weak/unclear consensus

CONSTRAINTS:
- No fluff or filler language
- No repetition of the ticker name
- No bullet points
- Keep summaries under 25 words
- Avoid trendy narratives unless clearly material (e.g., AI risk only if it meaningfully impacts fundamentals)

OUTPUT FORMAT:
Respond with ONLY a single JSON object.
Keys are ticker symbols. Each value is an object with:
- "consensus"
- "summary"
- "confidence" (number)

Example:
{{
  "AAPL": {{"consensus": "Buy", "summary": "Services growth supports margins but valuation limits upside.", "confidence": 0.82}},
  "MSFT": {{"consensus": "Strong Buy", "summary": "Cloud and AI leadership drive durable growth with minimal near-term risk.", "confidence": 0.9}}
}}
"""


def _ticker_context_line(sym: str, stock: Stock) -> str:
    """One line for the LLM: ticker, company name, price only."""
    price = stock.price or Decimal("0")
    company = (stock.company or "").strip() or "unknown"
    return f"  {sym}: {company!r}, ${float(price):.2f}"


def _holdings_aggregate(rows: List[Holding]) -> Dict[str, Dict[str, Any]]:
    """Merge holdings per symbol for context line + weighted average cost."""
    agg: Dict[str, Dict[str, Any]] = {}
    for h in rows:
        sym = h.stock.symbol.upper()
        if sym not in agg:
            agg[sym] = {
                "stock": h.stock,
                "total_shares": 0,
                "cost_num": Decimal("0"),
            }
        a = agg[sym]
        a["stock"] = h.stock
        a["total_shares"] += h.shares
        a["cost_num"] += Decimal(h.shares) * (h.average_price or Decimal("0"))
    return agg


def _holdings_lines_from_agg(agg: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    return {sym: _ticker_context_line(sym, a["stock"]) for sym, a in agg.items()}


def _symbols_strictly_at_loss(agg: Dict[str, Dict[str, Any]]) -> List[str]:
    """Symbols where current price < weighted average cost (merged across funds)."""
    out: List[str] = []
    for sym, a in agg.items():
        tsh = a["total_shares"]
        if tsh <= 0:
            continue
        avg_cost = a["cost_num"] / tsh
        price = a["stock"].price or Decimal("0")
        if price < avg_cost:
            out.append(sym)
    return out


class Command(BaseCommand):
    help = "Analyst-style consensus for holdings (or explicit tickers) via Gemini service"

    def add_arguments(self, parser):
        parser.add_argument(
            "--fund",
            type=str,
            help="Filter holdings to a profile/fund name (case-insensitive substring); "
            "omit to include all enabled funds",
        )
        parser.add_argument(
            "--symbols",
            type=str,
            help="Comma-separated tickers instead of DB holdings (e.g. AAPL,MSFT)",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=15,
            help="Max tickers to send in one request (default: 15)",
        )
        parser.add_argument(
            "--scenario",
            type=str,
            default="Portfolio holdings review",
            help="One-line scenario injected into the prompt",
        )
        parser.add_argument(
            "--context",
            type=str,
            default="",
            help="Optional text appended to Scenario after an em dash (same Scenario line).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print prompt only; do not call Gemini",
        )
        parser.add_argument(
            "--use-search",
            action="store_true",
            help="Enable Google Search grounding on Gemini (slower; may help recency)",
        )
        parser.add_argument(
            "--refresh",
            action="store_true",
            help="Refresh each stock price from yfinance before building context",
        )
        parser.add_argument(
            "--at-loss-only",
            action="store_true",
            help="Holdings mode only: sample only positions underwater (price < avg cost). "
            "If none, falls back to all symbols with a warning.",
        )

    def handle(self, *args, **options):
        symbols_arg = options.get("symbols")
        limit = max(1, int(options.get("limit") or 15))
        scenario = (options.get("scenario") or "Portfolio holdings review").strip()
        ctx = (options.get("context") or "").strip()
        if ctx:
            scenario = f"{scenario} — {ctx}".strip()
        dry_run = options.get("dry_run")
        use_search = options.get("use_search")
        do_refresh = options.get("refresh")
        at_loss_only = options.get("at_loss_only")

        context_lines: List[str] = []
        tickers: List[str] = []

        if symbols_arg:
            if at_loss_only:
                self.stdout.write(
                    self.style.WARNING("--at-loss-only is ignored when using --symbols")
                )
            raw = [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
            tickers = []
            for sym in raw:
                if len(tickers) >= limit:
                    break
                try:
                    st = Stock.objects.get(symbol__iexact=sym)
                except Stock.DoesNotExist:
                    self.stdout.write(self.style.WARNING(f"No Stock row for {sym}; skipping"))
                    continue
                tickers.append(sym)
                if do_refresh:
                    st.refresh()
                context_lines.append(_ticker_context_line(sym, st))
        else:
            qs = Holding.objects.filter(
                shares__gt=0,
                fund_id__isnull=False,
                fund__enabled=True,
            ).select_related("stock", "fund")
            fund_filter = (options.get("fund") or "").strip()
            if fund_filter:
                qs = qs.filter(fund__name__icontains=fund_filter)

            rows = list(qs)
            if not rows:
                self.stdout.write(
                    self.style.WARNING(
                        "No holdings matched (need fund-linked holdings, shares>0, fund enabled"
                        + (f", fund name contains {fund_filter!r}" if fund_filter else "")
                        + ")"
                    )
                )
                return

            if do_refresh:
                seen = set()
                for h in rows:
                    sid = h.stock_id
                    if sid not in seen:
                        seen.add(sid)
                        h.stock.refresh()

            agg = _holdings_aggregate(rows)
            lines_by_sym = _holdings_lines_from_agg(agg)
            symbols = list(lines_by_sym.keys())
            if at_loss_only:
                pool = _symbols_strictly_at_loss(agg)
                if not pool:
                    self.stdout.write(
                        self.style.WARNING(
                            "No underwater positions for --at-loss-only; using full holdings set."
                        )
                    )
                    pool = symbols
                else:
                    self.stdout.write(
                        self.style.NOTICE(
                            f"--at-loss-only: {len(pool)} underwater / {len(symbols)} symbols"
                        )
                    )
                symbols = pool
            random.shuffle(symbols)
            chosen = symbols[:limit]
            tickers = chosen
            context_lines = [lines_by_sym[s] for s in chosen]

        if not context_lines:
            self.stdout.write(self.style.ERROR("No tickers to audit"))
            return

        prompt = build_consensus_prompt(scenario=scenario, context_lines=context_lines)

        self.stdout.write(self.style.NOTICE("=== Prompt ===\n"))
        self.stdout.write(prompt)

        if dry_run:
            self.stdout.write(self.style.SUCCESS("\n(dry-run: no API call)"))
            return

        model, results, _next_model_idx, _next_key_idx = llm_ask_gemini(
            prompt=prompt,
            advisor_name="stock_audit",
            gemini_model_index=0,
            gemini_key_index=0,
            timeout=120.0,
            use_search=use_search,
        )

        if not results or not isinstance(results, dict):
            self.stdout.write(self.style.ERROR("No usable JSON from Gemini"))
            return

        self.stdout.write(self.style.SUCCESS(f"\n=== Model: {model} ===\n"))
        for sym in tickers:
            data = results.get(sym) or results.get(sym.upper()) or results.get(sym.lower())
            if not isinstance(data, dict):
                self.stdout.write(self.style.WARNING(f"{sym}: (missing in response)"))
                continue
            cons = (data.get("consensus") or "").strip()
            conf = data.get("confidence")
            summ = (data.get("summary") or "").strip()[:220]
            self.stdout.write(f"{sym:<8}  {cons:<12}  conf={conf}  {summ}")

        self.stdout.write("\n" + json.dumps(results, indent=2))
