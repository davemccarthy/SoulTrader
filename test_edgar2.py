"""
EDGAR Advisor using edgartools library - Clean interface implementation.

Usage:
    python test_edgar2.py --date 2026-01-16 [--exclude TICKER1 TICKER2] [--limit N] [--backtest]

TODO - Enhancements to Better Align with Market Reactions:
================================================================================
1. EPS BEAT VS. CONSENSUS ESTIMATES
   - Track EPS vs. consensus estimates (e.g., $0.19 vs. $0.12 expected = 58% beat)
   - Add bonus points for significant EPS beats (>20%, >50%)
   - Source: Market Data API (marketdata.app)
     * Endpoint: GET /v1/stocks/earnings/{symbol}/
     * Provides: estimatedEPS (consensus), reportedEPS (actual), surpriseEPS, surpriseEPSpct
     * Can query by date to match filing periods
     * ⚠️  LIMITATION: Earnings endpoint returns 402 Payment Required - requires paid plan
     * Free tier (100 API credits/day) may not include earnings data
     * Documentation: https://www.postman.com/marketdataapp/market-data/documentation/t8jjlx8/market-data-api-v1
   - Alternative: Use LLM (Gemini) to query consensus estimates as fallback (see item #14)
   - Example: TFIN beat by 58% but our system didn't capture this forward-looking signal
   - Example: LGCY reported $0.09 EPS, missed consensus $0.11 expected, causing -17.5% price drop
   - Compare actuals to expectations, not just historical growth

2. EXPENSE REDUCTION / COST DISCIPLINE
   - Increase weighting for margin expansion when revenue is stable (not declining)
   - Add bonus for margin expansion through cost-cutting (expense reduction)
   - Track operating expense changes explicitly
   - The market rewards cost discipline
   - Example: TFIN's 5% expense reduction through restructuring was key positive

3. SHARE BUYBACK DETECTION
   - Detect share repurchase programs from 10-Q/10-K filings
   - Parse financial statements or filing text for buyback announcements
   - Add bonus points for buyback programs (signals management confidence)
   - Example: TFIN's $30M buyback authorization was a positive signal

4. ANALYST SENTIMENT TRACKING
   - Track analyst upgrades/downgrades around earnings releases
   - Integrate with financial data APIs for analyst ratings
   - Add bonus/penalty based on analyst sentiment changes
   - Forward-looking indicator
   - Example: B. Riley upgrade and price target raises for TFIN

5. FORWARD GUIDANCE ANALYSIS
   - Parse management outlook from filings (MD&A section, earnings call transcripts)
   - Detect positive/negative guidance changes
   - Add bonus/penalty based on guidance direction
   - Example: PENG had strong FY 2025 results but weak FY 2026 outlook caused price drop

6. MARGIN TREND ANALYSIS
   - Detect margin compression vs. expansion trends (not just current vs. previous)
   - Multi-period margin trajectory analysis
   - Penalize declining margin trends even if current period improved
   - Example: PENG's narrowing gross margins were a market concern

7. CONSECUTIVE IMPROVEMENT TRACKING
   - Track multi-period improvement trends (2+ quarters of growth)
   - Add bonus for consecutive improvements (momentum signal)
   - Currently only compares current vs. previous period

8. INSIDER ACTIVITY DETECTION (Form 4)
   - Implement Form 4 parsing for insider trading activity
   - Factor in recent insider trades (buying vs. selling)
   - Penalize selling, bonus for buying
   - Already mentioned in code but not implemented
   - Use SEC Submissions API or edgartools Form 4 objects
   - Search recent daily index files for Form 4 filings with same CIK

9. PERIOD VALIDATION
   - Add validation to set delta to 0.0 if current_period_end == prev_period_end
   - Prevent incorrect comparisons when periods match

10. REVENUE QUALITY INDICATORS
    - Distinguish between organic growth vs. acquisition-driven growth
    - Flag revenue growth from one-time events
    - Better context for revenue changes

11. DEBT/LEVERAGE IMPROVEMENTS
    - Currently only tracks debt-to-equity reduction
    - Could add: debt paydown amounts, interest coverage improvements
    - More granular debt metrics

12. CASH POSITION ANALYSIS
    - Track cash and cash equivalents changes
    - Bonus for strong cash position improvements
    - Penalty for cash burn without revenue growth

13. 8-K FILING ANALYSIS (PROACTIVE EARNINGS DETECTION)
    - CRITICAL: Earnings announcements come via 8-K (Item 2.02) BEFORE detailed 10-K/10-Q
      * Example: RFIL 8-K on 2026-01-14 announced Q4 earnings (non-GAAP EPS $0.20 vs $0.09 consensus = 122% beat)
      * Market reacted to 8-K earnings release (+22.6% 1d), not the 10-K filed same day (annual GAAP EPS $0.01)
      * Example: RILY 8-K on 2026-01-14 announced Q3 earnings (EPS $2.98) → +48.54% pre-market surge
      * 8-K earnings press releases (Exhibit 99.1) contain the market-moving information
      * 10-K/10-Q provides detailed audited financials but often no new market-moving data
    - PROACTIVE APPROACH: Preempt strong price moves, don't just analyze after
      * Add 8-K to INVESTMENT_FORMS (currently only 10-K, 10-Q)
      * Prioritize 8-K Item 2.02 (Results of Operations and Financial Condition) for earnings
      * Parse press releases (Exhibit 99.1) immediately for: EPS, revenue, guidance, key metrics
      * LLM consensus check workflow:
        - Extract EPS/revenue from 8-K press release
        - Query LLM: "What was analyst consensus EPS for [TICKER] Q[X] [YEAR]?"
        - Calculate beat/miss percentage
        - Adjust score with beat/miss bonus/penalty
      * Flag high-confidence opportunities (big beats, strong guidance) for early entry
      * Alert BEFORE full market reaction (pre-market or early trading)
    - WORKFLOW: 8-K filed → Parse earnings → Query LLM consensus → Calculate beat/miss → 
      Score with beat/miss bonus → Alert if high score → Enter BEFORE full market reaction
    - OTHER 8-K ITEMS: Also analyze for corporate actions (mergers, acquisitions, material events)
      * Parse filing text for key events (Item 1.01, 2.01, 8.01, etc.)
      * Add scoring for positive corporate actions (acquisitions, partnerships)
      * Example: BTBT 8-K (merger) showed +18.5% 7d, +35.7% 30d price performance
      * Compare to old script's 8-K analysis approach (test_edgar.py)

14. LLM CONSENSUS ESTIMATES (Gemini Free Tier)
    - Use LLM to query analyst consensus estimates as fallback when paid APIs unavailable
    - Two approaches:
      a) Direct filing analysis: "What was analyst consensus EPS for LGCY Q4 2025?"
      b) Real-time lookup: "What's current analyst consensus EPS for GOOGL Q4 2025?"
    - Parse LLM response to extract consensus estimates (EPS, revenue, price target)
    - Advantages: Free tier available, works around API limitations
    - Considerations: Rate limits, accuracy validation, response parsing
    - Gemini Free Tier Models:
      * gemini-3-pro-preview
      * gemini-3-flash-preview
      * gemini-2.5-pro
      * gemini-2.5-flash
      * gemini-2.5-flash-lite

15. EPS PERIOD MATCHING
    - Match EPS calculation to correct period (quarterly vs annual)
    - Example: LGCY 10-K shows annual EPS $0.59, but Q4 EPS was $0.09 (quarterly)
    - Ensure shares outstanding matches same period as net income
    - Currently matching shares to report_period (fixed in recent update)
    - Distinguish between basic and diluted EPS (try diluted first, fallback to basic)
    - For 10-K filings, consider if quarterly EPS is needed vs annual EPS

16. LLM DISCONNECT ANALYSIS SCRIPT (Gemini)
    - Create standalone sister script to analyze filing/price disconnects
    - Input: filing reference (accession number)
    - Feed filing analysis data (metrics, scores, price performance) to Gemini LLM
    - Ask LLM why price changed (or didn't) despite good/bad filing
    - Example: SCHL had -0.16 delta (bad fundamentals) but +13.7% 7d, +43.0% price
    - Example: RSSS had 1.00 delta (perfect) but -22.7% price drop
    - Prompt needs work - consider: earnings beats/misses, guidance, analyst reactions,
      market expectations, sector trends, other catalysts
    - Use Gemini free tier models (see item #14 for model list)
    - Goal: Understand disconnects between filing fundamentals and market reactions
    - Output: LLM explanation of why the disconnect occurred
"""
import os
import time
import json
import re
import html
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal

import pandas as pd
import requests
import yfinance as yf
from edgar import get_filings, Company, Filing, find, set_identity

# Configuration
INVESTMENT_FORMS = ["10-K", "10-Q"]
_EDGAR_EMAIL = os.environ.get("EDGAR_USER_AGENT_EMAIL", "soultrader@example.com")
set_identity(f"SoulTrader {_EDGAR_EMAIL}")

# SEC API configuration
SEC_COMPANY_FACTS_API = "https://data.sec.gov/api/xbrl/companyfacts"
SEC_SUBMISSIONS_API = "https://data.sec.gov/submissions"
SEC_HEADERS = {
    "User-Agent": f"SoulTrader {_EDGAR_EMAIL}"
}

# Market Data API configuration (for consensus EPS estimates)
MARKETDATA_API_TOKEN = os.environ.get("MARKETDATA_API_TOKEN")
MARKETDATA_API_BASE = "https://api.marketdata.app/v1"

# Create a session for persistent connections (SEC requires rate limiting)
_session = requests.Session()
_session.headers.update({
    "User-Agent": f"SoulTrader {_EDGAR_EMAIL}",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
})

# Caching
_COMPANY_FACTS_CACHE = {}
_SUBMISSIONS_CACHE = {}
_CIK_TO_TICKER_CACHE = {}
_VALUATION_CACHE = {}


def analyze_single_filing(filing, detailed: bool = False, use_marketdata: bool = False) -> Optional[Dict]:
    """
    Analyze a single filing from edgartools.
    Uses SEC Company Facts API for metrics (same data source as old script).
    
    Args:
        filing: edgartools Filing object
        detailed: If True, print detailed breakdown
        
    Returns:
        Result dictionary or None if analysis fails
    """
    try:
        cik = str(filing.cik).zfill(10)
        ticker = None
        
        # Handle 8-K filings differently (event-driven, not period-based financials)
        if filing.form == "8-K":
            if detailed:
                print(f"  ⚠️  8-K filing detected - using event-based analysis (not financial metrics)")
            return analyze_8k_filing(filing, detailed=detailed)
        
        # Fetch company facts from SEC Company Facts API (aggregated, company-wide data)
        if detailed:
            print(f"  Fetching Company Facts API data for CIK {cik}...")
        facts_data = get_company_facts(cik)
        use_xbrl_fallback = False
        parent_cik = None
        
        # Check if this is a finance subsidiary - if so, try to get parent company's data
        company_name = getattr(filing, 'company', None) or ""
        if not facts_data and ("FINANCE" in company_name.upper() or "FINANCIAL" in company_name.upper()):
            if detailed:
                print(f"  ⚠️  Finance subsidiary detected: {company_name}")
                print(f"  Attempting to find parent company...")
            
            # Known mappings for finance subsidiaries to parent tickers
            known_subsidiaries = {
                "FERRELLGAS FINANCE CORP": "FGPR",
                "FERRELLGAS FINANCE": "FGPR",
            }
            
            parent_ticker = None
            for sub_name, ticker in known_subsidiaries.items():
                if sub_name in company_name.upper():
                    parent_ticker = ticker
                    break
            
            # Try to get parent company's CIK and Company Facts data
            if parent_ticker:
                try:
                    from edgar import Company
                    parent_company = Company(parent_ticker)
                    if parent_company and hasattr(parent_company, 'cik'):
                        parent_cik = str(parent_company.cik).zfill(10)
                        if detailed:
                            print(f"  ✓ Found parent company CIK: {parent_cik} (ticker: {parent_ticker})")
                        # Try to get parent company's Company Facts data
                        parent_facts_data = get_company_facts(parent_cik)
                        if parent_facts_data:
                            if detailed:
                                print(f"  ✓ Found parent company's Company Facts data - using consolidated financials")
                            facts_data = parent_facts_data
                            cik = parent_cik  # Update CIK to parent for period matching
                        else:
                            if detailed:
                                print(f"  ⚠️  Parent company also has no Company Facts data")
                except Exception as e:
                    if detailed:
                        print(f"  ⚠️  Could not get parent company CIK: {e}")
        
        if not facts_data:
            if detailed:
                print("  ⚠️  No Company Facts data available - falling back to XBRL extraction")
            use_xbrl_fallback = True
        
        # Get filing's report period (period end date)
        report_period = None
        if hasattr(filing, 'period_of_report'):
            report_period = filing.period_of_report
        elif hasattr(filing, 'report_date'):
            report_period = filing.report_date
        
        if detailed and report_period:
            print(f"  Filing report period: {report_period}")
        
        # Extract financial metrics using Company Facts API, filtered by actual report period
        # Or fall back to XBRL extraction if Company Facts not available
        if use_xbrl_fallback:
            # Try to get XBRL from filing
            try:
                xbrl = filing.xbrl()
                if xbrl:
                    metrics = extract_financial_metrics_from_xbrl(xbrl, filing.form, detailed=detailed)
                    if detailed:
                        print("  ✓ Extracted metrics from XBRL")
                else:
                    if detailed:
                        print("  ⚠️  No XBRL available in filing")
                    return None
            except Exception as e:
                if detailed:
                    print(f"  ⚠️  Error extracting from XBRL: {e}")
                return None
        else:
            metrics = extract_financial_metrics_from_company_facts(
                facts_data, filing.form, report_period=report_period, detailed=detailed
            )
        
        if not metrics or (not metrics.get("revenue") and not metrics.get("net_income")):
            if detailed:
                print("  ⚠️  No financial metrics extracted")
            return None
        
        # Calculate ratios and scores
        ratios = calculate_ratios(metrics)
        red_flags = detect_red_flags(metrics, ratios)
        
        # Get previous period metrics for delta comparison
        # Use SEC submissions API to find previous filing's report period
        prev_report_period = None
        yoy_report_period = None
        if report_period and not use_xbrl_fallback:
            prev_report_period = get_previous_filing_period(cik, filing.form, report_period)
            if detailed:
                if prev_report_period:
                    print(f"  Previous filing report period (sequential): {prev_report_period}")
                else:
                    print(f"  ⚠️  Could not find previous {filing.form} filing")
            
            # For 10-Q filings, also get YoY comparison (same quarter previous year)
            if filing.form == "10-Q":
                yoy_report_period = get_yoy_filing_period(cik, filing.form, report_period)
                if detailed:
                    if yoy_report_period:
                        print(f"  Year-over-year filing report period: {yoy_report_period}")
                    else:
                        print(f"  ⚠️  Could not find YoY {filing.form} filing")
        
        if use_xbrl_fallback:
            # Extract previous period from XBRL
            try:
                xbrl = filing.xbrl()
                if xbrl:
                    prev_metrics = extract_previous_period_metrics_from_xbrl(xbrl, filing.form, metrics)
                    if detailed and prev_metrics:
                        print(f"  ✓ Extracted previous period metrics from XBRL")
                else:
                    prev_metrics = {}
            except Exception as e:
                if detailed:
                    print(f"  ⚠️  Error extracting previous period from XBRL: {e}")
                prev_metrics = {}
            yoy_metrics = {}  # XBRL fallback doesn't support YoY yet
        else:
            prev_metrics = extract_previous_period_metrics_from_company_facts(
                facts_data, filing.form, metrics, prev_report_period=prev_report_period, detailed=detailed
            )
            # Extract YoY metrics for 10-Q filings
            yoy_metrics = {}
            if filing.form == "10-Q" and yoy_report_period:
                yoy_metrics = extract_previous_period_metrics_from_company_facts(
                    facts_data, filing.form, metrics, prev_report_period=yoy_report_period, detailed=detailed
                )
                if detailed and yoy_metrics:
                    print(f"  ✓ Extracted YoY metrics for comparison")
        
        # Get ticker (preferred but not required - allow analysis without ticker)
        # Try to get ticker from filing object first (if available)
        if not ticker:
            ticker = getattr(filing, 'ticker', None) or getattr(filing, 'symbol', None)
        # Fall back to CIK lookup
        if not ticker:
            ticker = cik_to_ticker(cik)
        
        # Get ticker (preferred but not required - allow analysis without ticker)
        # Try to get ticker from filing object first (if available)
        if not ticker:
            ticker = getattr(filing, 'ticker', None) or getattr(filing, 'symbol', None)
        
        # If we found parent ticker earlier, use it
        if not ticker and parent_cik:
            # Try to get ticker from parent CIK
            ticker = cik_to_ticker(parent_cik)
            if ticker and detailed:
                print(f"  ✓ Got parent company ticker from CIK: {ticker}")
        
        # Fall back to CIK lookup for original CIK if still no ticker
        if not ticker:
            ticker = cik_to_ticker(cik)
        
        # For finance subsidiaries, try to infer parent company ticker from company name
        if not ticker:
            # Known mappings for finance subsidiaries to parent tickers
            known_subsidiaries = {
                "FERRELLGAS FINANCE CORP": "FGPR",
                "FERRELLGAS FINANCE": "FGPR",
            }
            
            for sub_name, parent_ticker in known_subsidiaries.items():
                if sub_name in company_name.upper():
                    ticker = parent_ticker
                    if detailed:
                        print(f"  ✓ Using parent company ticker: {ticker}")
                    break
        
        if not ticker:
            if detailed:
                print("  ⚠️  No ticker found - analysis will proceed without valuation metrics")
        
        # Get valuation metrics (only if ticker available)
        valuation = get_valuation_metrics(ticker, metrics) if ticker else None
        if not ticker and detailed:
            print("  ⚠️  Valuation metrics unavailable (no ticker)")
        
        # Get consensus EPS from Market Data API (only if use_marketdata=True to conserve API calls)
        consensus_data = None
        if use_marketdata and ticker and report_period:
            if detailed:
                print(f"  Fetching consensus EPS from Market Data API for {ticker}...")
            consensus_data = get_consensus_eps_marketdata(ticker, report_period, filing.form, debug=detailed)
            if detailed and consensus_data:
                est_eps = consensus_data.get('estimated_eps')
                rep_eps = consensus_data.get('reported_eps')
                surprise_pct = consensus_data.get('surprise_eps_pct')
                if est_eps is not None:
                    print(f"  ✓ Consensus EPS: ${est_eps:.2f}")
                    if rep_eps is not None:
                        print(f"  ✓ Reported EPS: ${rep_eps:.2f}")
                        if surprise_pct is not None:
                            print(f"  ✓ Surprise: {surprise_pct:+.1f}%")
            elif detailed:
                print(f"  ⚠️  No consensus EPS data available")
                print(f"      (Market Data API earnings endpoint may require paid plan)")
        
        # Calculate scores
        absolute_score = calculate_investment_score(
            metrics, ratios, red_flags, filing.form, valuation, debug=detailed
        )
        # Calculate delta score - for 10-Q, prioritize YoY comparison
        if filing.form == "10-Q" and yoy_metrics and any(yoy_metrics.values()):
            # For 10-Q, use YoY comparison (weighted 70%) and sequential (weighted 30%)
            yoy_delta, _ = calculate_delta_score(metrics, ratios, yoy_metrics, filing_type=filing.form, debug=detailed)
            seq_delta, _ = calculate_delta_score(metrics, ratios, prev_metrics, filing_type=filing.form, debug=False)
            delta_score = (yoy_delta * 0.70) + (seq_delta * 0.30)
            delta_breakdown = None
            if detailed:
                print(f"  Delta Score: YoY={yoy_delta:.2f} (70%), Sequential={seq_delta:.2f} (30%), Combined={delta_score:.2f}")
        else:
            # For 10-K or if YoY not available, use sequential only
            delta_score, delta_breakdown = calculate_delta_score(metrics, ratios, prev_metrics, filing_type=filing.form, debug=detailed)
        delta_label = classify_filing_delta(delta_score)
        
        # Get price momentum
        filing_date_str = filing.filing_date.strftime("%Y%m%d") if hasattr(filing.filing_date, 'strftime') else str(filing.filing_date).replace("-", "")
        price_momentum = get_price_momentum(ticker, filing_date_str, delta_label) if ticker else None
        momentum_score = price_momentum.get("momentum_score", 0.5) if price_momentum else 0.5
        
        # Insider activity detection (to be implemented later)
        insider_activity = None
        
        # Calculate combined/final score
        if filing.form == "10-K":
            delta_weight, abs_weight = 0.40, 0.60
        elif filing.form == "10-Q":
            delta_weight, abs_weight = 0.35, 0.65
        else:
            delta_weight, abs_weight = 0.30, 0.70
        
        combined_score = (absolute_score * abs_weight) + (delta_score * delta_weight)
        
        # Final score
        final_score = (combined_score * 0.9) + (momentum_score * 0.1)
        
        # Get company name
        company_name = getattr(filing, 'company', None) or f"CIK-{cik}"
        
        # Build result
        result = {
            "cik": cik,
            "company": company_name,
            "ticker": ticker,
            "form": filing.form,
            "filing_date": filing.filing_date,
            "filing_ref": filing.accession_number,
            "score": final_score,
            "absolute_score": absolute_score,
            "delta_score": delta_score,
            "delta_label": delta_label,
            "combined_score": combined_score,
            "momentum_score": momentum_score,
            "price_momentum": price_momentum,
            "metrics": metrics,
            "ratios": ratios,
            "red_flags": red_flags,
            "valuation": valuation,
            "insider_activity": insider_activity,
            "consensus_data": consensus_data,
        }
        
        # Print detailed breakdown if requested
        if detailed:
            print_detailed_analysis(result, delta_breakdown, prev_metrics)
        
        return result
        
    except Exception as e:
        if detailed:
            print(f"  Error analyzing filing: {e}")
            import traceback
            traceback.print_exc()
        return None


def humanize_number(value: Optional[float]) -> str:
    """Format large numbers with K/M/B suffixes."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    elif abs(value) >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.2f}K"
    else:
        return f"${value:.2f}"


def print_detailed_analysis(result: Dict, delta_breakdown: Optional[Dict], prev_metrics: Dict):
    """Print detailed analysis breakdown similar to original script."""
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    print(f"\nSCORES:")
    print(f"  Final Score: {result['score']:.2f}")
    print(f"  Absolute: {result['absolute_score']:.2f}")
    print(f"  Delta: {result['delta_score']:.2f} ({result.get('delta_label', 'N/A')})")
    print(f"  Combined: {result['combined_score']:.2f}")
    
    # Delta breakdown
    if delta_breakdown and delta_breakdown.get("components"):
        print(f"\n  Delta Score Breakdown:")
        curr_period_end = result['metrics'].get("_period_end", "N/A")
        prev_period_end = prev_metrics.get("_period_end", "N/A")
        
        if curr_period_end and prev_period_end and curr_period_end == prev_period_end:
            print(f"    ⚠️  WARNING: Comparing same period ({curr_period_end})")
            print(f"       This suggests:")
            print(f"       - Previous period data may not be available")
            print(f"       - XBRL data may only have one period of this type")
            print(f"       - Delta score may be invalid (comparing period to itself)")
            print(f"       - If delta score > 0, it may be comparing different data points with same end date")
        elif not prev_period_end or prev_period_end == "N/A":
            print(f"    ⚠️  WARNING: Previous period date not available")
            print(f"       Current: ending {curr_period_end}")
            print(f"       Previous: (date unknown)")
        else:
            print(f"    Comparing: ending {curr_period_end} vs ending {prev_period_end}")
        
        for component, value in delta_breakdown["components"]:
            print(f"    +{value:.2f} - {component}")
        print(f"    = {result['delta_score']:.2f} total")
    
    # Delta comparison
    if prev_metrics and any(prev_metrics.values()):
        print(f"\n{'='*80}")
        print("DELTA COMPARISON (Current vs Previous Period)")
        print("=" * 80)
        
        metrics = result['metrics']
        ratios = result['ratios']
        
        # Revenue
        if metrics.get('revenue') and prev_metrics.get('revenue'):
            curr_rev = metrics['revenue']
            prev_rev = prev_metrics['revenue']
            rev_change = curr_rev - prev_rev
            rev_change_pct = (rev_change / prev_rev * 100) if prev_rev != 0 else 0
            print(f"\nRevenue:")
            print(f"  Current: {humanize_number(curr_rev)}")
            print(f"  Previous: {humanize_number(prev_rev)}")
            print(f"  Change: {humanize_number(rev_change)} ({rev_change_pct:+.1f}%)")
        
        # Net Income
        if metrics.get('net_income') is not None and prev_metrics.get('net_income') is not None:
            curr_ni = metrics['net_income']
            prev_ni = prev_metrics['net_income']
            ni_change = curr_ni - prev_ni
            ni_change_pct = (ni_change / abs(prev_ni) * 100) if prev_ni != 0 else (100 if curr_ni > 0 else 0)
            print(f"\nNet Income:")
            print(f"  Current: {humanize_number(curr_ni)}")
            print(f"  Previous: {humanize_number(prev_ni)}")
            print(f"  Change: {humanize_number(ni_change)} ({ni_change_pct:+.1f}%)")
            if prev_ni <= 0 < curr_ni:
                print(f"  ⚠️  Profitability inflection: Loss → Profit")
        
        # Operating Cash Flow
        if metrics.get('operating_cash_flow') is not None and prev_metrics.get('operating_cash_flow') is not None:
            curr_ocf = metrics['operating_cash_flow']
            prev_ocf = prev_metrics['operating_cash_flow']
            ocf_change = curr_ocf - prev_ocf
            ocf_change_pct = (ocf_change / abs(prev_ocf) * 100) if prev_ocf != 0 else (100 if curr_ocf > 0 else 0)
            print(f"\nOperating Cash Flow:")
            print(f"  Current: {humanize_number(curr_ocf)}")
            print(f"  Previous: {humanize_number(prev_ocf)}")
            print(f"  Change: {humanize_number(ocf_change)} ({ocf_change_pct:+.1f}%)")
        
        # Ratio changes - calculate prev_ratios from prev_metrics for comparison
        prev_ratios = calculate_ratios(prev_metrics) if prev_metrics else {}
        
        # Earnings Per Share (EPS) - compare after prev_ratios is calculated
        if ratios.get('eps') is not None and prev_ratios.get('eps') is not None:
            curr_eps = ratios['eps']
            prev_eps = prev_ratios['eps']
            eps_change = curr_eps - prev_eps
            eps_change_pct = (eps_change / abs(prev_eps) * 100) if prev_eps != 0 else (100 if curr_eps > 0 else 0)
            print(f"\nEarnings Per Share (EPS):")
            print(f"  Current: ${curr_eps:.2f}")
            print(f"  Previous: ${prev_eps:.2f}")
            print(f"  Change: ${eps_change:+.2f} ({eps_change_pct:+.1f}%)")
        if prev_ratios:
            print(f"\nRatio Changes:")
            if ratios.get('current_ratio') and prev_ratios.get('current_ratio'):
                curr_cr = ratios['current_ratio']
                prev_cr = prev_ratios['current_ratio']
                cr_change = curr_cr - prev_cr
                print(f"  Current Ratio: {curr_cr:.2f} (was {prev_cr:.2f}, {cr_change:+.2f})")
            
            if ratios.get('debt_to_equity') and prev_ratios.get('debt_to_equity'):
                curr_de = ratios['debt_to_equity']
                prev_de = prev_ratios['debt_to_equity']
                de_change = curr_de - prev_de
                print(f"  Debt/Equity: {curr_de:.2f} (was {prev_de:.2f}, {de_change:+.2f})")
                if de_change < 0:
                    print(f"    ✅ Debt reduction")
            
            if ratios.get('gross_margin') is not None and prev_ratios.get('gross_margin') is not None:
                curr_gm = ratios['gross_margin']
                prev_gm = prev_ratios['gross_margin']
                gm_change = curr_gm - prev_gm
                print(f"  Gross Margin: {curr_gm:.1f}% (was {prev_gm:.1f}%, {gm_change:+.1f}%)")
            
            if ratios.get('operating_margin') is not None and prev_ratios.get('operating_margin') is not None:
                curr_om = ratios['operating_margin']
                prev_om = prev_ratios['operating_margin']
                om_change = curr_om - prev_om
                print(f"  Operating Margin: {curr_om:.1f}% (was {prev_om:.1f}%, {om_change:+.1f}%)")
            
            if ratios.get('net_margin') is not None and prev_ratios.get('net_margin') is not None:
                curr_nm = ratios['net_margin']
                prev_nm = prev_ratios['net_margin']
                nm_change = curr_nm - prev_nm
                print(f"  Net Margin: {curr_nm:.1f}% (was {prev_nm:.1f}%, {nm_change:+.1f}%)")
            
            if ratios.get('cash_flow_margin') is not None and prev_ratios.get('cash_flow_margin') is not None:
                curr_cfm = ratios['cash_flow_margin']
                prev_cfm = prev_ratios['cash_flow_margin']
                cfm_change = curr_cfm - prev_cfm
                print(f"  Cash Flow Margin: {curr_cfm:.1f}% (was {prev_cfm:.1f}%, {cfm_change:+.1f}%)")
    
    # Current period metrics
    print(f"\n{'='*80}")
    print("CURRENT PERIOD METRICS")
    print("=" * 80)
    metrics = result['metrics']
    ratios = result['ratios']
    valuation = result.get('valuation')
    
    print(f"\nFINANCIAL METRICS:")
    if metrics.get('revenue'):
        print(f"  Revenue: {humanize_number(metrics['revenue'])}")
    if metrics.get('gross_profit'):
        print(f"  Gross Profit: {humanize_number(metrics['gross_profit'])}")
    if metrics.get('operating_income'):
        print(f"  Operating Income: {humanize_number(metrics['operating_income'])}")
    if metrics.get('operating_cash_flow'):
        print(f"  Operating Cash Flow: {humanize_number(metrics['operating_cash_flow'])}")
    if metrics.get('net_income'):
        print(f"  Net Income: {humanize_number(metrics['net_income'])}")
    
    print(f"\nFINANCIAL RATIOS:")
    if ratios.get('current_ratio') is not None:
        print(f"  Current Ratio: {ratios['current_ratio']:.2f}")
    if ratios.get('debt_to_equity') is not None:
        print(f"  Debt/Equity: {ratios['debt_to_equity']:.2f}")
    if ratios.get('gross_margin') is not None:
        print(f"  Gross Margin: {ratios['gross_margin']:.1f}%")
    if ratios.get('operating_margin') is not None:
        print(f"  Operating Margin: {ratios['operating_margin']:.1f}%")
    if ratios.get('net_margin') is not None:
        print(f"  Net Margin: {ratios['net_margin']:.1f}%")
    if ratios.get('cash_flow_margin') is not None:
        print(f"  Cash Flow Margin: {ratios['cash_flow_margin']:.1f}%")
    if ratios.get('eps') is not None:
        print(f"  Earnings Per Share (EPS): ${ratios['eps']:.2f}")
    
    # Show consensus EPS comparison if available
    consensus_data = result.get('consensus_data')
    if consensus_data:
        est_eps = consensus_data.get('estimated_eps')
        rep_eps = consensus_data.get('reported_eps')
        surprise_pct = consensus_data.get('surprise_eps_pct')
        if est_eps is not None:
            actual_eps = ratios.get('eps')
            print(f"\nEPS CONSENSUS COMPARISON:")
            print(f"  Actual EPS (from filing): ${actual_eps:.2f}" if actual_eps else "  Actual EPS: N/A")
            print(f"  Consensus Estimate: ${est_eps:.2f}")
            if rep_eps is not None:
                print(f"  Reported EPS (Market Data): ${rep_eps:.2f}")
            if surprise_pct is not None:
                beat_miss = "Beat" if surprise_pct > 0 else "Miss"
                print(f"  Surprise: {surprise_pct:+.1f}% ({beat_miss})")
            elif actual_eps and est_eps:
                # Calculate our own beat/miss
                surprise = ((actual_eps - est_eps) / est_eps) * 100 if est_eps > 0 else 0
                beat_miss = "Beat" if surprise > 0 else "Miss"
                print(f"  Beat/Miss: {surprise:+.1f}% ({beat_miss})")
    
    if valuation:
        print(f"\nVALUATION:")
        if valuation.get('current_price'):
            print(f"  Current Price: ${valuation['current_price']:.2f}")
        if valuation.get('pe_ratio'):
            print(f"  P/E: {valuation['pe_ratio']:.2f}")
        if valuation.get('pb_ratio'):
            print(f"  P/B: {valuation['pb_ratio']:.2f}")
        if valuation.get('ps_ratio'):
            print(f"  P/S: {valuation['ps_ratio']:.2f}")
        if valuation.get('market_cap'):
            print(f"  Market Cap: {humanize_number(valuation['market_cap'])}")
    
    red_flags = result.get('red_flags', [])
    if red_flags:
        print(f"\nRED FLAGS ({len(red_flags)}):")
        for flag_desc, penalty in red_flags:
            print(f"  - {flag_desc} (penalty: {penalty:.2f})")
    else:
        print(f"\nRED FLAGS: None")


def analyze_filing_by_accession(accession_number: str):
    """
    Analyze a specific filing by accession number using edgartools.
    
    Args:
        accession_number: Filing accession number (e.g., "0001660280-25-000128")
    """
    print("=" * 80)
    print(f"ANALYZING FILING: {accession_number}")
    print("=" * 80)
    
    # Use edgartools find() function for direct lookup by accession number
    print(f"Looking up filing by accession number...")
    try:
        filing = find(accession_number)
        if filing:
            print(f"✓ Found filing: {filing.company} - {filing.form} on {filing.filing_date}")
            
            # Analyze the filing with detailed output
            # Use Market Data API only in --analyze mode to conserve API calls (100/day limit)
            result = analyze_single_filing(filing, detailed=True, use_marketdata=True)
            if not result:
                print("\n⚠️  Analysis failed")
                return
        else:
            print(f"\n⚠️  Could not find filing {accession_number}")
            print(f"   Make sure the accession number is correct")
            return
    except Exception as e:
        print(f"\n⚠️  Error looking up filing: {e}")
        return


def analyze_edgar_filings(date_str: str) -> List[Dict]:
    """
    Main entry point: Analyze EDGAR filings for a specific date.
    
    Args:
        date_str: Date in YYYY-MM-DD format
        
    Returns:
        List of result dictionaries with scores and metrics
    """
    
    # Normalize date (handle weekends)
    filing_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    # Move to previous weekday if weekend
    while filing_date.weekday() >= 5:
        filing_date -= timedelta(days=1)
    
    date_str_normalized = filing_date.strftime("%Y-%m-%d")
    
    print(f"\nAnalyzing EDGAR filings for {date_str_normalized}...")
    
    # Get filings using edgartools
    try:
        filings = get_filings(filing_date=date_str_normalized, form=INVESTMENT_FORMS, amendments=False)
        print(f"Found {len(filings)} investment form filings")
        print(f"Processing filings...")
    except Exception as e:
        print(f"Error fetching filings: {e}")
        return []
    
    results = []
    filings_processed = 0
    filings_with_xbrl = 0
    filings_with_metrics = 0
    
    # Process each filing
    for filing in filings:
        filings_processed += 1
        xbrl = filing.xbrl()
        if xbrl:
            filings_with_xbrl += 1
        
        result = analyze_single_filing(filing, detailed=False)
        if result:
            metrics = result.get("metrics", {})
            if metrics and (metrics.get("revenue") or metrics.get("net_income")):
                filings_with_metrics += 1
            results.append(result)
    
    # Debug output
    print(f"\nProcessed: {filings_processed} filings, {filings_with_xbrl} with XBRL, {filings_with_metrics} with extracted metrics")
    
    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Deduplicate by ticker
    seen_tickers = {}
    deduplicated = []
    for result in results:
        ticker = result.get('ticker')
        if not ticker:
            deduplicated.append(result)
        elif ticker not in seen_tickers:
            seen_tickers[ticker] = result
            deduplicated.append(result)
    
    return deduplicated


def get_company_submissions(cik: str) -> Optional[Dict]:
    """
    Fetch company submissions (filing history) from SEC Submissions API.
    This provides filing dates, report periods, and accession numbers.
    
    Args:
        cik: Company CIK (10-digit, zero-padded)
    
    Returns:
        Dictionary with submissions data or None
    """
    if not cik:
        return None
    
    # Pad CIK to 10 digits
    cik = str(cik).zfill(10)
    
    # Check cache
    if cik in _SUBMISSIONS_CACHE:
        return _SUBMISSIONS_CACHE[cik]
    
    url = f"{SEC_SUBMISSIONS_API}/CIK{cik}.json"
    
    try:
        time.sleep(0.05)  # Rate limiting
        response = _session.get(url, timeout=30)
        
        if response.status_code == 404:
            _SUBMISSIONS_CACHE[cik] = None
            return None
        
        response.raise_for_status()
        data = response.json()
        _SUBMISSIONS_CACHE[cik] = data
        return data
        
    except requests.RequestException as e:
        _SUBMISSIONS_CACHE[cik] = None
        return None
    except json.JSONDecodeError as e:
        _SUBMISSIONS_CACHE[cik] = None
        return None


def get_previous_filing_period(cik: str, form: str, current_report_date: str) -> Optional[str]:
    """
    Get the report period end date of the previous filing of the same form type.
    
    Args:
        cik: Company CIK
        form: Form type (10-K, 10-Q)
        current_report_date: Current filing's report period end date (YYYY-MM-DD)
    
    Returns:
        Previous filing's report period end date (YYYY-MM-DD) or None
    """
    submissions = get_company_submissions(cik)
    if not submissions or "filings" not in submissions:
        return None
    
    filings = submissions.get("filings", {})
    recent = filings.get("recent", {})
    
    if not recent:
        return None
    
    forms = recent.get("form", [])
    report_dates = recent.get("reportDate", [])
    
    # Find previous filing of same form type with report date before current
    previous_dates = []
    for f, rd in zip(forms, report_dates):
        if f == form and rd and rd < current_report_date:
            previous_dates.append(rd)
    
    if previous_dates:
        # Return the most recent previous report date
        return sorted(previous_dates, reverse=True)[0]
    
    return None


def get_yoy_filing_period(cik: str, form: str, current_report_date: str) -> Optional[str]:
    """
    Get the report period end date of the same quarter/year from the previous year (YoY comparison).
    For 10-Q: Get same quarter from previous year (e.g., Q3 2025 -> Q3 2024)
    For 10-K: Get previous year's annual filing (e.g., FY 2025 -> FY 2024)
    
    Args:
        cik: Company CIK
        form: Form type (10-K, 10-Q)
        current_report_date: Current filing's report period end date (YYYY-MM-DD)
    
    Returns:
        Year-over-year filing's report period end date (YYYY-MM-DD) or None
    """
    try:
        current_date = datetime.strptime(current_report_date, "%Y-%m-%d").date()
        # Calculate YoY date (subtract 1 year)
        yoy_date = current_date.replace(year=current_date.year - 1)
        yoy_date_str = yoy_date.strftime("%Y-%m-%d")
        
        submissions = get_company_submissions(cik)
        if not submissions or "filings" not in submissions:
            return None
        
        filings = submissions.get("filings", {})
        recent = filings.get("recent", {})
        
        if not recent:
            return None
        
        forms = recent.get("form", [])
        report_dates = recent.get("reportDate", [])
        
        # Find filing of same form type with report date matching YoY date (or close)
        # Allow up to 5 days difference to handle slight variations
        for f, rd in zip(forms, report_dates):
            if f == form and rd:
                try:
                    rd_date = datetime.strptime(rd, "%Y-%m-%d").date()
                    days_diff = abs((rd_date - yoy_date).days)
                    # For quarterly filings, match same month (within 5 days)
                    # For annual filings, match same month (within 5 days)
                    if days_diff <= 5 and rd_date.month == yoy_date.month:
                        return rd
                except (ValueError, TypeError):
                    continue
        
        # If exact match not found, try to find closest match within same month
        closest_match = None
        min_days_diff = float('inf')
        for f, rd in zip(forms, report_dates):
            if f == form and rd:
                try:
                    rd_date = datetime.strptime(rd, "%Y-%m-%d").date()
                    if rd_date.month == yoy_date.month and rd_date.year == yoy_date.year:
                        days_diff = abs((rd_date - yoy_date).days)
                        if days_diff < min_days_diff:
                            min_days_diff = days_diff
                            closest_match = rd
                except (ValueError, TypeError):
                    continue
        
        return closest_match if min_days_diff <= 30 else None
        
    except (ValueError, TypeError) as e:
        return None


def get_company_facts(cik: str) -> Optional[Dict]:
    """
    Fetch company facts (XBRL data) from SEC Company Facts API.
    This is the aggregated company-wide data (not filing-specific).
    
    Args:
        cik: Company CIK (10-digit, zero-padded)
    
    Returns:
        Dictionary with company facts data or None
    """
    if not cik:
        return None
    
    # Pad CIK to 10 digits
    cik = str(cik).zfill(10)
    
    # Check cache
    if cik in _COMPANY_FACTS_CACHE:
        return _COMPANY_FACTS_CACHE[cik]
    
    url = f"{SEC_COMPANY_FACTS_API}/CIK{cik}.json"
    
    try:
        time.sleep(0.05)  # Rate limiting - SEC requires delays
        response = _session.get(url, timeout=30)
        
        if response.status_code == 404:
            # Company may not have XBRL data
            _COMPANY_FACTS_CACHE[cik] = None
            return None
        
        response.raise_for_status()
        data = response.json()
        _COMPANY_FACTS_CACHE[cik] = data
        return data
        
    except requests.RequestException as e:
        _COMPANY_FACTS_CACHE[cik] = None
        return None
    except json.JSONDecodeError as e:
        _COMPANY_FACTS_CACHE[cik] = None
        return None


def extract_financial_metrics_from_company_facts(facts_data: Dict, filing_type: str = None, report_period: str = None, detailed: bool = False) -> Dict:
    """
    Extract key financial metrics from SEC Company Facts API.
    Filters for period-specific data matching the filing's actual report period.
    
    Args:
        facts_data: XBRL company facts data from Company Facts API
        filing_type: Type of filing (10-K, 10-Q) - used to determine period type
        report_period: Filing's report period end date (YYYY-MM-DD) - if provided, filters to match this period
        detailed: If True, print debug output
    
    Returns:
        Dictionary with extracted metrics and period info
    """
    metrics = {
        "revenue": None,
        "revenue_growth": None,
        "gross_profit": None,
        "operating_income": None,
        "net_income": None,
        "operating_cash_flow": None,
        "total_assets": None,
        "total_liabilities": None,
        "current_assets": None,
        "current_liabilities": None,
        "shares_outstanding": None,
        "_period_end": None,  # Period end date
        "_fiscal_period": None,  # Fiscal period (FY, Q1, Q2, etc.)
        "_fiscal_year": None,  # Fiscal year (e.g., 2025)
        "_revenue_concept_used": None,  # Track which revenue concept was used
        "_cash_flow_concept_used": None,  # Track which cash flow concept was used
    }
    
    if not facts_data or "facts" not in facts_data:
        return metrics
    
    facts = facts_data.get("facts", {})
    us_gaap = facts.get("us-gaap", {})
    
    # Determine period filter: annual for 10-K, quarterly for 10-Q
    filter_annual = filing_type == "10-K" if filing_type else False
    
    if detailed:
        print(f"  DEBUG: Filing type: {filing_type}, filter_annual: {filter_annual}, report_period: {report_period}")
    
    # Helper to filter period-specific data (not cumulative) and match report period
    # For 10-K: filter by fp=="FY" and form=="10-K"
    # For 10-Q: filter by duration (85-95 days) and form=="10-Q"
    def filter_period_specific(data_list, target_period_end=None):
        period_specific = []
        for entry in data_list:
            start = entry.get("start", "")
            end = entry.get("end", "")
            fp = entry.get("fp", "")  # Fiscal period (FY, Q1, Q2, Q3, Q4)
            form = entry.get("form", "")  # Form type (10-K, 10-Q)
            
            # For 10-K: must have fp=="FY" and form=="10-K"
            if filter_annual:
                if fp != "FY" or form != "10-K":
                    continue
            
            # For 10-Q: must have form=="10-Q" (fp can be Q1, Q2, Q3, Q4)
            # BUT: if form is empty or doesn't match, still check duration as fallback
            else:
                # For 10-Q, be more lenient - check form if available, but don't exclude if form is missing
                if form and form != "10-Q":
                    continue
            
            if start and end:
                try:
                    if len(start) == 10 and len(end) == 10:  # YYYY-MM-DD format
                        start_date = datetime.strptime(start, "%Y-%m-%d").date()
                        end_date = datetime.strptime(end, "%Y-%m-%d").date()
                        duration_days = (end_date - start_date).days
                        
                        # Check duration matches period type (as additional validation)
                        duration_match = False
                        if not filter_annual:
                            # Quarterly: accept 85-95 days
                            if 85 <= duration_days <= 95:
                                duration_match = True
                        else:
                            # Annual: accept 360-370 days
                            if 360 <= duration_days <= 370:
                                duration_match = True
                        
                        # If duration matches (or we're using fp/form filtering), check if period end matches target
                        if duration_match or filter_annual:
                            if target_period_end:
                                # Match exact period end date
                                if end == target_period_end:
                                    period_specific.append(entry)
                            else:
                                # No target period - include all matching duration/form
                                period_specific.append(entry)
                except (ValueError, TypeError):
                    # If date parsing fails, skip for 10-K (need valid dates), include for 10-Q if no target
                    if not filter_annual and not target_period_end:
                        period_specific.append(entry)
            else:
                # If no start date, only include for 10-Q if no target period
                if not filter_annual and not target_period_end:
                    period_specific.append(entry)
        
        # If we were filtering by target_period_end and found no matches, return empty list
        # Otherwise, return filtered list or fallback to all data
        if target_period_end and not period_specific:
            return []  # No match found for target period - don't fall back to all data
        return period_specific if period_specific else data_list
    
    # Extract revenue (most recent PERIOD-SPECIFIC, not cumulative)
    # If report_period provided, filter to match that exact period
    if "Revenues" in us_gaap:
        rev_data = us_gaap["Revenues"].get("units", {}).get("USD", [])
        if rev_data:
            period_specific_rev = filter_period_specific(rev_data, target_period_end=report_period)
            if detailed:
                print(f"  DEBUG: Revenue data - total entries: {len(rev_data)}, period-specific after filter: {len(period_specific_rev)}")
                if report_period:
                    # Check if we found entries matching the report period
                    matching = [e for e in period_specific_rev if e.get("end") == report_period]
                    if matching:
                        print(f"  DEBUG: ✓ Found {len(matching)} entry(ies) matching report period {report_period}")
                    else:
                        print(f"  DEBUG: ⚠️  No entries found matching report period {report_period}")
                        # Show what period ends are actually available
                        all_period_ends = sorted(set(e.get("end") for e in rev_data if e.get("end") and len(e.get("end", "")) == 10), reverse=True)
                        print(f"  DEBUG: Available period ends in Company Facts (most recent 10): {all_period_ends[:10]}")
                        # Check if there's a close match (within a few days)
                        if all_period_ends:
                            try:
                                target_date = datetime.strptime(report_period, "%Y-%m-%d").date()
                                for period_end in all_period_ends[:10]:
                                    period_date = datetime.strptime(period_end, "%Y-%m-%d").date()
                                    days_diff = abs((period_date - target_date).days)
                                    if days_diff <= 5:
                                        print(f"  DEBUG: Found close match: {period_end} (within {days_diff} days of {report_period})")
                            except:
                                pass
                if len(rev_data) > 0:
                    # Show duration info for first few entries
                    for i, entry in enumerate(rev_data[:3]):
                        start = entry.get("start", "")
                        end = entry.get("end", "")
                        if start and end and len(start) == 10 and len(end) == 10:
                            try:
                                start_date = datetime.strptime(start, "%Y-%m-%d").date()
                                end_date = datetime.strptime(end, "%Y-%m-%d").date()
                                duration = (end_date - start_date).days
                                print(f"  DEBUG: Revenue entry {i+1}: {start} to {end} = {duration} days, val={entry.get('val')}")
                            except:
                                pass
            if period_specific_rev:
                # If we filtered by report_period, there should be exactly one match
                if report_period:
                    # Look for exact match first
                    exact_match = [e for e in period_specific_rev if e.get("end") == report_period]
                    if exact_match:
                        latest = exact_match[0]
                    elif len(period_specific_rev) == 1:
                        # Only one entry, use it but warn
                        latest = period_specific_rev[0]
                        if detailed:
                            print(f"  DEBUG: ⚠️  WARNING: Only one period-specific entry found, but period end {latest.get('end')} doesn't match report period {report_period}")
                    else:
                        # Multiple entries but none match - take most recent and warn
                        latest = sorted(period_specific_rev, key=lambda x: x.get("end", ""), reverse=True)[0]
                        if detailed:
                            print(f"  DEBUG: ⚠️  WARNING: No exact match for report period {report_period}, using most recent: {latest.get('end')}")
                else:
                    # No report period specified - take most recent
                    latest = sorted(period_specific_rev, key=lambda x: x.get("end", ""), reverse=True)[0]
                
                metrics["revenue"] = latest.get("val")
                metrics["_period_end"] = latest.get("end")
                metrics["_fiscal_period"] = latest.get("fp")
                metrics["_fiscal_year"] = latest.get("fy")  # Fiscal year (e.g., 2025)
                metrics["_revenue_concept_used"] = "Revenues"  # Track which concept was used
                if detailed:
                    print(f"  DEBUG: ✓ Extracted revenue: ${latest.get('val')/1e6:.2f}M for period ending {latest.get('end')}")
                    if report_period and latest.get("end") != report_period:
                        print(f"  DEBUG: ⚠️  WARNING: Report period mismatch! Expected {report_period}, got {latest.get('end')}")
            elif detailed:
                print(f"  DEBUG: ⚠️  No revenue found after filtering for period {report_period}")
                # Check what revenue concepts are available
                revenue_keys = [k for k in us_gaap.keys() if 'revenue' in k.lower() or 'sales' in k.lower()]
                if revenue_keys:
                    print(f"  DEBUG: Revenue-related concepts found: {revenue_keys[:5]}")
                    # Check first revenue concept for available periods
                    if revenue_keys:
                        rev_data = us_gaap[revenue_keys[0]].get("units", {}).get("USD", [])
                        if rev_data:
                            available_ends = sorted(set(e.get("end") for e in rev_data if e.get("end") and len(e.get("end", "")) == 10), reverse=True)[:5]
                            print(f"  DEBUG: Available revenue period ends in '{revenue_keys[0]}': {available_ends}")
                
                # Calculate growth if we have previous period-specific data
                if len(period_specific_rev) >= 2:
                    sorted_rev = sorted(period_specific_rev, key=lambda x: x.get("end", ""), reverse=True)
                    current = sorted_rev[0].get("val")
                    previous = sorted_rev[1].get("val")
                    if previous and previous > 0:
                        metrics["revenue_growth"] = ((current - previous) / previous) * 100
            elif detailed:
                print(f"  DEBUG: No period-specific revenue found after filtering")
                if len(rev_data) > 0:
                    print(f"  DEBUG: Sample revenue entry: {rev_data[0]}")
    elif detailed:
        # Search for revenue-related keys
        revenue_keys = [k for k in us_gaap.keys() if 'revenue' in k.lower() or 'sales' in k.lower()]
        print(f"  DEBUG: 'Revenues' key not found in us_gaap.")
        print(f"  DEBUG: Revenue-related keys found: {revenue_keys[:10]}")  # Limit output
        
        # For financial institutions, also search for income-related concepts
        income_keys = [k for k in us_gaap.keys() if ('income' in k.lower() and ('interest' in k.lower() or 'noninterest' in k.lower() or 'operating' in k.lower()))]
        if income_keys:
            print(f"  DEBUG: Income-related keys found (for financial institutions): {income_keys[:10]}")
        
        if report_period:
            print(f"  DEBUG: Checking all revenue/income concepts for report period {report_period}...")
        else:
            print(f"  DEBUG: Checking all revenue/income concepts...")
        
        # Check what values each revenue concept has for the report period
        for rev_key in revenue_keys[:5]:  # Limit to first 5 to avoid spam
            rev_data_all = us_gaap[rev_key].get("units", {}).get("USD", [])
            if rev_data_all:
                if report_period:
                    # Find entries matching the report period
                    for entry in rev_data_all:
                        if entry.get("end") == report_period:
                            val = entry.get("val")
                            start = entry.get("start", "")
                            end = entry.get("end", "")
                            print(f"  DEBUG:   {rev_key}: {start} to {end} = ${val/1e6:.2f}M (raw: {val})")
                else:
                    # No report period specified - show first few entries
                    for entry in rev_data_all[:3]:
                        val = entry.get("val")
                        start = entry.get("start", "")
                        end = entry.get("end", "")
                        print(f"  DEBUG:   {rev_key}: {start} to {end} = ${val/1e6:.2f}M (raw: {val})")
                    break  # Only show first concept if no report period
        
        # Also check income concepts for financial institutions
        for income_key in income_keys[:5]:  # Limit to first 5
            income_data_all = us_gaap[income_key].get("units", {}).get("USD", [])
            if income_data_all:
                if report_period:
                    # Find entries matching the report period
                    for entry in income_data_all:
                        if entry.get("end") == report_period:
                            val = entry.get("val")
                            start = entry.get("start", "")
                            end = entry.get("end", "")
                            print(f"  DEBUG:   {income_key}: {start} to {end} = ${val/1e6:.2f}M (raw: {val})")
    
    # Try alternative revenue keys if "Revenues" not found
    if metrics["revenue"] is None:
        # Common revenue tags in XBRL
        # For regular companies:
        revenue_concepts = [
            "RevenuesFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax", 
            "RevenuesFromContractWithCustomer",
            "RevenueFromContractWithCustomer",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
        ]
        
        # For financial institutions (banks), add income concepts:
        # Note: For banks, "revenue" is often reported as total interest + noninterest income
        financial_revenue_concepts = [
            "InterestAndFeeIncomeLoans",
            "InterestAndDividendIncomeOperating",
            "InterestIncome",
            "TotalInterestIncome",
            "NoninterestIncome",
            "InterestAndFeeIncomeLoansAndLeases",
            "InterestIncomeOperating",
            "OperatingIncomeLoss",  # For some financial institutions, this represents total income
        ]
        
        # Combine both lists
        revenue_concepts = revenue_concepts + financial_revenue_concepts
        
        for rev_concept in revenue_concepts:
            if rev_concept in us_gaap:
                rev_data = us_gaap[rev_concept].get("units", {}).get("USD", [])
                if rev_data:
                    period_specific_rev = filter_period_specific(rev_data, target_period_end=report_period)
                    if detailed:
                        print(f"  DEBUG: Found revenue in '{rev_concept}' - entries: {len(rev_data)}, period-specific: {len(period_specific_rev)}")
                        # Show all period-specific entries with their durations
                        if period_specific_rev:
                            print(f"  DEBUG: Period-specific revenue entries:")
                            for entry in sorted(period_specific_rev, key=lambda x: x.get("end", ""), reverse=True)[:5]:
                                start = entry.get("start", "")
                                end = entry.get("end", "")
                                val = entry.get("val")
                                fp = entry.get("fp", "")
                                if start and end:
                                    try:
                                        start_date = datetime.strptime(start, "%Y-%m-%d").date()
                                        end_date = datetime.strptime(end, "%Y-%m-%d").date()
                                        duration = (end_date - start_date).days
                                        print(f"    - {start} to {end} ({duration} days, fp={fp}): ${val/1e6:.2f}M")
                                    except:
                                        print(f"    - {start} to {end} (fp={fp}): ${val/1e6:.2f}M")
                    if period_specific_rev:
                        # If we filtered by report_period, there should be exactly one match
                        if report_period and len(period_specific_rev) == 1:
                            latest = period_specific_rev[0]
                        else:
                            latest = sorted(period_specific_rev, key=lambda x: x.get("end", ""), reverse=True)[0]
                        metrics["revenue"] = latest.get("val")
                        metrics["_period_end"] = latest.get("end")
                        metrics["_fiscal_period"] = latest.get("fp")
                        metrics["_revenue_concept_used"] = rev_concept  # Track which concept was used
                        if detailed:
                            print(f"  DEBUG: ✓ Extracted revenue from '{rev_concept}': {latest.get('val')} for period ending {latest.get('end')}")
                            if report_period and latest.get("end") != report_period:
                                print(f"  DEBUG: ⚠️  WARNING: Report period mismatch! Expected {report_period}, got {latest.get('end')}")
                        break
    
    # For financial institutions, if no revenue found, try using OperatingIncomeLoss as revenue
    # (For banks, operating income often represents total income/revenue)
    if metrics["revenue"] is None and "OperatingIncomeLoss" in us_gaap:
        oi_data = us_gaap["OperatingIncomeLoss"].get("units", {}).get("USD", [])
        if oi_data:
            period_specific_oi = filter_period_specific(oi_data, target_period_end=report_period)
            if period_specific_oi:
                if report_period:
                    exact_match = [e for e in period_specific_oi if e.get("end") == report_period]
                    if exact_match:
                        latest = exact_match[0]
                    elif len(period_specific_oi) == 1:
                        latest = period_specific_oi[0]
                    else:
                        latest = sorted(period_specific_oi, key=lambda x: x.get("end", ""), reverse=True)[0]
                else:
                    latest = sorted(period_specific_oi, key=lambda x: x.get("end", ""), reverse=True)[0]
                # Only use OperatingIncomeLoss as revenue if it's positive (income, not loss)
                if latest.get("val", 0) > 0:
                    metrics["revenue"] = latest.get("val")
                    metrics["_period_end"] = latest.get("end")
                    metrics["_fiscal_period"] = latest.get("fp")
                    metrics["_revenue_concept_used"] = "OperatingIncomeLoss"  # Track which concept was used
                    if detailed:
                        print(f"  DEBUG: ✓ Using OperatingIncomeLoss as revenue: ${latest.get('val')/1e6:.2f}M for period ending {latest.get('end')}")
    
    # Extract gross profit (period-specific)
    if "GrossProfit" in us_gaap:
        gp_data = us_gaap["GrossProfit"].get("units", {}).get("USD", [])
        if gp_data:
            period_specific_gp = filter_period_specific(gp_data, target_period_end=report_period)
            if period_specific_gp:
                if report_period:
                    exact_match = [e for e in period_specific_gp if e.get("end") == report_period]
                    if exact_match:
                        latest = exact_match[0]
                    elif len(period_specific_gp) == 1:
                        latest = period_specific_gp[0]
                    else:
                        latest = sorted(period_specific_gp, key=lambda x: x.get("end", ""), reverse=True)[0]
                else:
                    latest = sorted(period_specific_gp, key=lambda x: x.get("end", ""), reverse=True)[0]
                metrics["gross_profit"] = latest.get("val")
                if detailed:
                    print(f"  DEBUG: ✓ Extracted gross profit: ${latest.get('val')/1e6:.2f}M for period ending {latest.get('end')}")
            elif detailed:
                print(f"  DEBUG: ⚠️  No gross profit found for period {report_period}")
                # Check if GrossProfit exists in us_gaap
                if "GrossProfit" in us_gaap:
                    gp_data = us_gaap["GrossProfit"].get("units", {}).get("USD", [])
                    if gp_data:
                        print(f"  DEBUG: GrossProfit exists but no period-specific match. Total entries: {len(gp_data)}")
                        # Show available periods
                        available_ends = sorted(set(e.get("end") for e in gp_data if e.get("end") and len(e.get("end", "")) == 10), reverse=True)[:5]
                        print(f"  DEBUG: Available gross profit period ends: {available_ends}")
    
    # Extract operating income (period-specific)
    # For financial institutions, OperatingIncomeLoss might not exist, so try alternatives
    operating_income_concepts = [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",  # For banks
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",  # Alternative
    ]
    
    operating_income_found = False
    for oi_concept in operating_income_concepts:
        if oi_concept in us_gaap:
            oi_data = us_gaap[oi_concept].get("units", {}).get("USD", [])
            if oi_data:
                period_specific_oi = filter_period_specific(oi_data, target_period_end=report_period)
                if period_specific_oi:
                    if report_period:
                        exact_match = [e for e in period_specific_oi if e.get("end") == report_period]
                        if exact_match:
                            latest = exact_match[0]
                        elif len(period_specific_oi) == 1:
                            latest = period_specific_oi[0]
                        else:
                            latest = sorted(period_specific_oi, key=lambda x: x.get("end", ""), reverse=True)[0]
                    else:
                        latest = sorted(period_specific_oi, key=lambda x: x.get("end", ""), reverse=True)[0]
                    metrics["operating_income"] = latest.get("val")
                    metrics["_operating_income_concept_used"] = oi_concept  # Track which concept was used
                    if detailed:
                        print(f"  DEBUG: ✓ Extracted operating income from '{oi_concept}': ${latest.get('val')/1e6:.2f}M for period ending {latest.get('end')}")
                    operating_income_found = True
                    break
                elif detailed:
                    print(f"  DEBUG: ⚠️  '{oi_concept}' exists but no period-specific match for period {report_period}")
                    # Show available periods
                    available_ends = sorted(set(e.get("end") for e in oi_data if e.get("end") and len(e.get("end", "")) == 10), reverse=True)[:5]
                    print(f"  DEBUG: Available '{oi_concept}' period ends: {available_ends}")
    
    if not operating_income_found and detailed:
        print(f"  DEBUG: ⚠️  No operating income found - checked concepts: {operating_income_concepts}")
        # For financial institutions, check if we can calculate a proxy from income before taxes
        if "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest" in us_gaap:
            print(f"  DEBUG: Note: 'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest' exists but didn't match period")
    
    # Extract net income (period-specific)
    if "NetIncomeLoss" in us_gaap:
        ni_data = us_gaap["NetIncomeLoss"].get("units", {}).get("USD", [])
        if ni_data:
            period_specific_ni = filter_period_specific(ni_data, target_period_end=report_period)
            if period_specific_ni:
                if report_period and len(period_specific_ni) == 1:
                    latest = period_specific_ni[0]
                else:
                    latest = sorted(period_specific_ni, key=lambda x: x.get("end", ""), reverse=True)[0]
                metrics["net_income"] = latest.get("val")
    
    # Extract operating cash flow (period-specific)
    # Try primary concept first, then search for alternatives
    cash_flow_found = False
    cash_flow_concepts = [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "CashAndCashEquivalentsAtCarryingValue",  # Sometimes used as proxy
        "CashFlowFromOperatingActivities",  # Alternative name
    ]
    
    for cf_concept in cash_flow_concepts:
        if cf_concept in us_gaap:
            cf_data = us_gaap[cf_concept].get("units", {}).get("USD", [])
            if cf_data:
                # First, try to find exact period match (bypass duration filter)
                if report_period:
                    exact_match = [e for e in cf_data if e.get("end") == report_period]
                    if exact_match:
                        latest = exact_match[0]
                        metrics["operating_cash_flow"] = latest.get("val")
                        metrics["_cash_flow_concept_used"] = cf_concept
                        cash_flow_found = True
                        if detailed:
                            print(f"  DEBUG: ✓ Extracted operating cash flow from '{cf_concept}': ${latest.get('val')/1e6:.2f}M for period ending {latest.get('end')}")
                        break
                
                # If no exact match, try period-specific filter
                period_specific_cf = filter_period_specific(cf_data, target_period_end=report_period)
                if period_specific_cf:
                    if report_period:
                        # Look for exact match in filtered results
                        exact_match = [e for e in period_specific_cf if e.get("end") == report_period]
                        if exact_match:
                            latest = exact_match[0]
                        elif len(period_specific_cf) == 1:
                            latest = period_specific_cf[0]
                            if detailed:
                                print(f"  DEBUG: ⚠️  Operating cash flow period mismatch: expected {report_period}, got {latest.get('end')}")
                        else:
                            latest = sorted(period_specific_cf, key=lambda x: x.get("end", ""), reverse=True)[0]
                            if detailed:
                                print(f"  DEBUG: ⚠️  Operating cash flow: no exact match for {report_period}, using {latest.get('end')}")
                    else:
                        latest = sorted(period_specific_cf, key=lambda x: x.get("end", ""), reverse=True)[0]
                    metrics["operating_cash_flow"] = latest.get("val")
                    metrics["_cash_flow_concept_used"] = cf_concept
                    cash_flow_found = True
                    if detailed:
                        print(f"  DEBUG: ✓ Extracted operating cash flow from '{cf_concept}': ${latest.get('val')/1e6:.2f}M for period ending {latest.get('end')}")
                    break
    
    if not cash_flow_found and detailed:
        # Search for any cash flow related concepts
        cash_flow_keys = [k for k in us_gaap.keys() if 'cash' in k.lower() and ('operating' in k.lower() or 'flow' in k.lower())]
        if cash_flow_keys:
            print(f"  DEBUG: ⚠️  No operating cash flow found for period {report_period}")
            print(f"  DEBUG: Available cash flow concepts: {cash_flow_keys[:5]}")
            # Check if any have data for the report period
            for cf_key in cash_flow_keys[:3]:  # Check first 3
                cf_data = us_gaap[cf_key].get("units", {}).get("USD", [])
                if cf_data:
                    matching = [e for e in cf_data if e.get("end") == report_period]
                    if matching:
                        print(f"  DEBUG:   {cf_key} has data for {report_period}: ${matching[0].get('val')/1e6:.2f}M")
                        # Use it even though filter didn't catch it
                        metrics["operating_cash_flow"] = matching[0].get("val")
                        metrics["_cash_flow_concept_used"] = cf_key
                        cash_flow_found = True
                        if detailed:
                            print(f"  DEBUG: ✓ Using {cf_key} for operating cash flow: ${matching[0].get('val')/1e6:.2f}M")
                        break
        else:
            print(f"  DEBUG: ⚠️  No operating cash flow found for period {report_period} (no cash flow concepts in us_gaap)")
    
    # Extract balance sheet items (instant - match to report period if provided)
    # Balance sheet items are "instant" (point in time), so match by period end date
    if "Assets" in us_gaap:
        assets_data = us_gaap["Assets"].get("units", {}).get("USD", [])
        if assets_data:
            if report_period:
                # Find entry matching report period
                matched = [e for e in assets_data if e.get("end") == report_period]
                if matched:
                    latest = matched[0]
                else:
                    # Fallback to most recent
                    latest = sorted(assets_data, key=lambda x: x.get("end", ""), reverse=True)[0]
            else:
                latest = sorted(assets_data, key=lambda x: x.get("end", ""), reverse=True)[0]
            metrics["total_assets"] = latest.get("val")
    
    if "Liabilities" in us_gaap:
        liab_data = us_gaap["Liabilities"].get("units", {}).get("USD", [])
        if liab_data:
            if report_period:
                matched = [e for e in liab_data if e.get("end") == report_period]
                if matched:
                    latest = matched[0]
                else:
                    latest = sorted(liab_data, key=lambda x: x.get("end", ""), reverse=True)[0]
            else:
                latest = sorted(liab_data, key=lambda x: x.get("end", ""), reverse=True)[0]
            metrics["total_liabilities"] = latest.get("val")
    
    if "AssetsCurrent" in us_gaap:
        ca_data = us_gaap["AssetsCurrent"].get("units", {}).get("USD", [])
        if ca_data:
            if report_period:
                matched = [e for e in ca_data if e.get("end") == report_period]
                if matched:
                    latest = matched[0]
                else:
                    latest = sorted(ca_data, key=lambda x: x.get("end", ""), reverse=True)[0]
            else:
                latest = sorted(ca_data, key=lambda x: x.get("end", ""), reverse=True)[0]
            metrics["current_assets"] = latest.get("val")
    
    if "LiabilitiesCurrent" in us_gaap:
        cl_data = us_gaap["LiabilitiesCurrent"].get("units", {}).get("USD", [])
        if cl_data:
            if report_period:
                matched = [e for e in cl_data if e.get("end") == report_period]
                if matched:
                    latest = matched[0]
                else:
                    latest = sorted(cl_data, key=lambda x: x.get("end", ""), reverse=True)[0]
            else:
                latest = sorted(cl_data, key=lambda x: x.get("end", ""), reverse=True)[0]
            metrics["current_liabilities"] = latest.get("val")
    
    # Extract shares outstanding - match to same period as net income
    # Try diluted first, then basic
    shares_found = False
    for shares_concept in ["WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"]:
        if shares_concept in us_gaap:
            shares_data = us_gaap[shares_concept].get("units", {}).get("shares", [])
            if shares_data:
                if report_period:
                    # Match to same period as net income
                    matched = [e for e in shares_data if e.get("end") == report_period]
                    if matched:
                        metrics["shares_outstanding"] = matched[0].get("val")
                        shares_found = True
                        if detailed:
                            print(f"  DEBUG: ✓ Using {shares_concept} for period {report_period}: {matched[0].get('val'):,.0f} shares")
                        break
                    else:
                        # Fallback to most recent
                        latest = sorted(shares_data, key=lambda x: x.get("end", ""), reverse=True)[0]
                        metrics["shares_outstanding"] = latest.get("val")
                        shares_found = True
                        if detailed:
                            print(f"  DEBUG: ⚠️  Using most recent {shares_concept} (period {latest.get('end')}): {latest.get('val'):,.0f} shares")
                        break
                else:
                    # No report period specified, use most recent
                    latest = sorted(shares_data, key=lambda x: x.get("end", ""), reverse=True)[0]
                    metrics["shares_outstanding"] = latest.get("val")
                    shares_found = True
                    if detailed:
                        print(f"  DEBUG: ✓ Using most recent {shares_concept}: {latest.get('val'):,.0f} shares")
                    break
        if shares_found:
            break
    
    return metrics


def extract_previous_period_metrics_from_company_facts(facts_data: Dict, filing_type: str, current_metrics: Dict, prev_report_period: str = None, detailed: bool = False) -> Dict:
    """
    Extract previous period metrics from Company Facts API for delta comparison.
    Matches to the previous filing's report period (not just second-most-recent).
    
    Args:
        facts_data: XBRL company facts data from Company Facts API
        filing_type: Type of filing (10-K, 10-Q)
        current_metrics: Current period metrics (to identify current period_end)
        prev_report_period: Previous filing's report period end date (YYYY-MM-DD) - if provided, filters to match this period
        detailed: If True, print debug output
    
    Returns:
        Dictionary with previous period metrics
    """
    prev_metrics = {
        "revenue": None,
        "net_income": None,
        "operating_cash_flow": None,
        "total_assets": None,
        "total_liabilities": None,
        "current_assets": None,
        "current_liabilities": None,
        "_period_end": None,
        "_fiscal_period": None,
    }
    
    if not facts_data or "facts" not in facts_data:
        return prev_metrics
    
    facts = facts_data.get("facts", {})
    us_gaap = facts.get("us-gaap", {})
    
    # Determine period filter: annual for 10-K, quarterly for 10-Q
    filter_annual = filing_type == "10-K" if filing_type else False
    
    # Helper to filter period-specific data (same as extract_financial_metrics_from_company_facts)
    # For 10-K: filter by fp=="FY" and form=="10-K", and optionally by fiscal year
    # For 10-Q: filter by duration (85-95 days) and form=="10-Q"
    def filter_period_specific(data_list, target_period_end=None, target_fiscal_year=None):
        period_specific = []
        for entry in data_list:
            start = entry.get("start", "")
            end = entry.get("end", "")
            fp = entry.get("fp", "")  # Fiscal period (FY, Q1, Q2, Q3, Q4)
            form = entry.get("form", "")  # Form type (10-K, 10-Q)
            fy = entry.get("fy")  # Fiscal year
            
            # For 10-K: must have fp=="FY" and form=="10-K"
            if filter_annual:
                if fp != "FY" or form != "10-K":
                    continue
                # If target fiscal year specified, match it
                if target_fiscal_year and fy != target_fiscal_year:
                    continue
            
            # For 10-Q: must have form=="10-Q" (fp can be Q1, Q2, Q3, Q4)
            else:
                if form != "10-Q":
                    continue
            
            if start and end:
                try:
                    if len(start) == 10 and len(end) == 10:  # YYYY-MM-DD format
                        start_date = datetime.strptime(start, "%Y-%m-%d").date()
                        end_date = datetime.strptime(end, "%Y-%m-%d").date()
                        duration_days = (end_date - start_date).days
                        
                        duration_match = False
                        if not filter_annual:
                            # Quarterly: accept 85-95 days
                            if 85 <= duration_days <= 95:
                                duration_match = True
                        else:
                            # Annual: accept 360-370 days
                            if 360 <= duration_days <= 370:
                                duration_match = True
                        
                        if duration_match or filter_annual:
                            if target_period_end:
                                # Match exact period end date
                                if end == target_period_end:
                                    period_specific.append(entry)
                            else:
                                period_specific.append(entry)
                except (ValueError, TypeError):
                    if not filter_annual and not target_period_end:
                        period_specific.append(entry)
            else:
                if not filter_annual and not target_period_end:
                    period_specific.append(entry)
        
        # If we were filtering by target_period_end and found no matches, return empty list
        if target_period_end and not period_specific:
            return []  # No match found for target period
        return period_specific if period_specific else data_list
    
    # For 10-K, get previous fiscal year if available (calculate BEFORE using it)
    prev_fiscal_year = None
    if filter_annual and current_metrics.get("_fiscal_year"):
        try:
            prev_fiscal_year = int(current_metrics["_fiscal_year"]) - 1
        except (ValueError, TypeError):
            pass
    
    # Extract previous period revenue (match to prev_report_period if provided)
    # Use the same revenue concept that was used for current period
    prev_revenue_found = False
    revenue_concept_used = current_metrics.get("_revenue_concept_used")
    
    # Build list of concepts to try: same as current, then "Revenues", then others
    concepts_to_try = []
    if revenue_concept_used:
        concepts_to_try.append(revenue_concept_used)
    if "Revenues" not in concepts_to_try:
        concepts_to_try.append("Revenues")
    # Add other revenue concepts
    revenue_concepts = [
        "RevenuesFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax", 
        "RevenuesFromContractWithCustomer",
        "RevenueFromContractWithCustomer",
        "SalesRevenueNet",
    ]
    for rev_concept in revenue_concepts:
        if rev_concept not in concepts_to_try:
            concepts_to_try.append(rev_concept)
    
    for rev_concept in concepts_to_try:
        if rev_concept in us_gaap:
            rev_data = us_gaap[rev_concept].get("units", {}).get("USD", [])
            if rev_data:
                period_specific_rev = filter_period_specific(rev_data, target_period_end=prev_report_period, target_fiscal_year=prev_fiscal_year)
                if period_specific_rev:
                    if prev_report_period:
                        # Look for exact match first
                        exact_match = [e for e in period_specific_rev if e.get("end") == prev_report_period]
                        if exact_match:
                            prev_rev = exact_match[0]
                        elif len(period_specific_rev) == 1:
                            prev_rev = period_specific_rev[0]
                        else:
                            # Multiple entries - take most recent
                            prev_rev = sorted(period_specific_rev, key=lambda x: x.get("end", ""), reverse=True)[0]
                    else:
                        # No prev_report_period - get second-most-recent
                        sorted_rev = sorted(period_specific_rev, key=lambda x: x.get("end", ""), reverse=True)
                        if len(sorted_rev) >= 2:
                            prev_rev = sorted_rev[1]
                        elif len(sorted_rev) == 1:
                            prev_rev = sorted_rev[0]
                        else:
                            prev_rev = None
                    
                    if prev_rev:
                        prev_metrics["revenue"] = prev_rev.get("val")
                        prev_metrics["_period_end"] = prev_rev.get("end")
                        prev_metrics["_fiscal_period"] = prev_rev.get("fp")
                        prev_metrics["_fiscal_year"] = prev_rev.get("fy")
                        prev_revenue_found = True
                        if detailed:
                            print(f"  DEBUG: ✓ Found previous period revenue in '{rev_concept}': ${prev_rev.get('val')/1e6:.2f}M for period ending {prev_rev.get('end')}")
                            if prev_report_period and prev_rev.get("end") != prev_report_period:
                                print(f"  DEBUG: ⚠️  Previous period mismatch! Expected {prev_report_period}, got {prev_rev.get('end')}")
                        break
    
    if not prev_revenue_found and detailed:
        print(f"  DEBUG: ⚠️  No previous period revenue found for period {prev_report_period}")
    
    # Extract previous period gross profit
    if "GrossProfit" in us_gaap:
        gp_data = us_gaap["GrossProfit"].get("units", {}).get("USD", [])
        if gp_data:
            period_specific_gp = filter_period_specific(gp_data, target_period_end=prev_report_period, target_fiscal_year=prev_fiscal_year)
            if period_specific_gp:
                if prev_report_period:
                    exact_match = [e for e in period_specific_gp if e.get("end") == prev_report_period]
                    if exact_match:
                        prev_metrics["gross_profit"] = exact_match[0].get("val")
                    elif len(period_specific_gp) == 1:
                        prev_metrics["gross_profit"] = period_specific_gp[0].get("val")
                    else:
                        sorted_gp = sorted(period_specific_gp, key=lambda x: x.get("end", ""), reverse=True)
                        prev_metrics["gross_profit"] = sorted_gp[0].get("val")
                else:
                    sorted_gp = sorted(period_specific_gp, key=lambda x: x.get("end", ""), reverse=True)
                    if len(sorted_gp) >= 2:
                        prev_metrics["gross_profit"] = sorted_gp[1].get("val")
    
    # Extract previous period operating income
    # Use the same concept that was used for current period (if available)
    operating_income_concept_used = current_metrics.get("_operating_income_concept_used", "OperatingIncomeLoss")
    
    # Try the same concept first, then fall back to alternatives
    operating_income_concepts = [
        operating_income_concept_used,  # Use the same concept as current period
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    operating_income_concepts = [x for x in operating_income_concepts if not (x in seen or seen.add(x))]
    
    for oi_concept in operating_income_concepts:
        if oi_concept in us_gaap:
            oi_data = us_gaap[oi_concept].get("units", {}).get("USD", [])
            if oi_data:
                period_specific_oi = filter_period_specific(oi_data, target_period_end=prev_report_period, target_fiscal_year=prev_fiscal_year)
                if period_specific_oi:
                    if prev_report_period:
                        exact_match = [e for e in period_specific_oi if e.get("end") == prev_report_period]
                        if exact_match:
                            prev_metrics["operating_income"] = exact_match[0].get("val")
                        elif len(period_specific_oi) == 1:
                            prev_metrics["operating_income"] = period_specific_oi[0].get("val")
                        else:
                            sorted_oi = sorted(period_specific_oi, key=lambda x: x.get("end", ""), reverse=True)
                            prev_metrics["operating_income"] = sorted_oi[0].get("val")
                    else:
                        sorted_oi = sorted(period_specific_oi, key=lambda x: x.get("end", ""), reverse=True)
                        if len(sorted_oi) >= 2:
                            prev_metrics["operating_income"] = sorted_oi[1].get("val")
                    break  # Found it, stop searching
    
    # Extract previous period net income
    if "NetIncomeLoss" in us_gaap:
        ni_data = us_gaap["NetIncomeLoss"].get("units", {}).get("USD", [])
        if ni_data:
            period_specific_ni = filter_period_specific(ni_data, target_period_end=prev_report_period, target_fiscal_year=prev_fiscal_year)
            if period_specific_ni:
                if prev_report_period:
                    exact_match = [e for e in period_specific_ni if e.get("end") == prev_report_period]
                    if exact_match:
                        prev_metrics["net_income"] = exact_match[0].get("val")
                    elif len(period_specific_ni) == 1:
                        prev_metrics["net_income"] = period_specific_ni[0].get("val")
                    else:
                        sorted_ni = sorted(period_specific_ni, key=lambda x: x.get("end", ""), reverse=True)
                        prev_metrics["net_income"] = sorted_ni[0].get("val")
                else:
                    sorted_ni = sorted(period_specific_ni, key=lambda x: x.get("end", ""), reverse=True)
                    if len(sorted_ni) >= 2:
                        prev_metrics["net_income"] = sorted_ni[1].get("val")
    
    # Extract previous period operating cash flow
    # Use the same cash flow concept that was used for current period
    cash_flow_concept_used = current_metrics.get("_cash_flow_concept_used")
    cash_flow_concepts = []
    if cash_flow_concept_used:
        cash_flow_concepts.append(cash_flow_concept_used)
    cash_flow_concepts.extend([
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "CashAndCashEquivalentsAtCarryingValue",
        "CashFlowFromOperatingActivities",
    ])
    # Remove duplicates while preserving order
    cash_flow_concepts = list(dict.fromkeys(cash_flow_concepts))
    
    prev_cash_flow_found = False
    for cf_concept in cash_flow_concepts:
        if cf_concept in us_gaap:
            cf_data = us_gaap[cf_concept].get("units", {}).get("USD", [])
            if cf_data:
                # First, try to find exact period match (bypass duration filter)
                if prev_report_period:
                    exact_match = [e for e in cf_data if e.get("end") == prev_report_period]
                    if exact_match:
                        prev_metrics["operating_cash_flow"] = exact_match[0].get("val")
                        prev_cash_flow_found = True
                        if detailed:
                            print(f"  DEBUG: ✓ Found previous period operating cash flow in '{cf_concept}': ${prev_metrics['operating_cash_flow']/1e6:.2f}M for period ending {prev_report_period}")
                        break
                
                # If no exact match, try period-specific filter
                period_specific_cf = filter_period_specific(cf_data, target_period_end=prev_report_period)
                if period_specific_cf:
                    if prev_report_period:
                        # Look for exact match in filtered results
                        exact_match = [e for e in period_specific_cf if e.get("end") == prev_report_period]
                        if exact_match:
                            prev_metrics["operating_cash_flow"] = exact_match[0].get("val")
                        elif len(period_specific_cf) == 1:
                            prev_metrics["operating_cash_flow"] = period_specific_cf[0].get("val")
                        else:
                            sorted_cf = sorted(period_specific_cf, key=lambda x: x.get("end", ""), reverse=True)
                            prev_metrics["operating_cash_flow"] = sorted_cf[0].get("val")
                    else:
                        sorted_cf = sorted(period_specific_cf, key=lambda x: x.get("end", ""), reverse=True)
                        if len(sorted_cf) >= 2:
                            prev_metrics["operating_cash_flow"] = sorted_cf[1].get("val")
                    
                    if prev_metrics.get("operating_cash_flow"):
                        prev_cash_flow_found = True
                        if detailed:
                            print(f"  DEBUG: ✓ Found previous period operating cash flow in '{cf_concept}': ${prev_metrics['operating_cash_flow']/1e6:.2f}M")
                        break
    
    if not prev_cash_flow_found and detailed:
        print(f"  DEBUG: ⚠️  No previous period operating cash flow found for period {prev_report_period}")
    
    # Extract previous period balance sheet items (match to prev_report_period if provided)
    if "Assets" in us_gaap:
        assets_data = us_gaap["Assets"].get("units", {}).get("USD", [])
        if assets_data:
            if prev_report_period:
                matched = [e for e in assets_data if e.get("end") == prev_report_period]
                if matched:
                    prev_metrics["total_assets"] = matched[0].get("val")
                elif len(assets_data) >= 2:
                    sorted_assets = sorted(assets_data, key=lambda x: x.get("end", ""), reverse=True)
                    prev_metrics["total_assets"] = sorted_assets[1].get("val")
            elif len(assets_data) >= 2:
                sorted_assets = sorted(assets_data, key=lambda x: x.get("end", ""), reverse=True)
                prev_metrics["total_assets"] = sorted_assets[1].get("val")
    
    if "Liabilities" in us_gaap:
        liab_data = us_gaap["Liabilities"].get("units", {}).get("USD", [])
        if liab_data:
            if prev_report_period:
                matched = [e for e in liab_data if e.get("end") == prev_report_period]
                if matched:
                    prev_metrics["total_liabilities"] = matched[0].get("val")
                elif len(liab_data) >= 2:
                    sorted_liab = sorted(liab_data, key=lambda x: x.get("end", ""), reverse=True)
                    prev_metrics["total_liabilities"] = sorted_liab[1].get("val")
            elif len(liab_data) >= 2:
                sorted_liab = sorted(liab_data, key=lambda x: x.get("end", ""), reverse=True)
                prev_metrics["total_liabilities"] = sorted_liab[1].get("val")
    
    if "AssetsCurrent" in us_gaap:
        ca_data = us_gaap["AssetsCurrent"].get("units", {}).get("USD", [])
        if ca_data:
            if prev_report_period:
                matched = [e for e in ca_data if e.get("end") == prev_report_period]
                if matched:
                    prev_metrics["current_assets"] = matched[0].get("val")
                elif len(ca_data) >= 2:
                    sorted_ca = sorted(ca_data, key=lambda x: x.get("end", ""), reverse=True)
                    prev_metrics["current_assets"] = sorted_ca[1].get("val")
            elif len(ca_data) >= 2:
                sorted_ca = sorted(ca_data, key=lambda x: x.get("end", ""), reverse=True)
                prev_metrics["current_assets"] = sorted_ca[1].get("val")
    
    if "LiabilitiesCurrent" in us_gaap:
        cl_data = us_gaap["LiabilitiesCurrent"].get("units", {}).get("USD", [])
        if cl_data:
            if prev_report_period:
                matched = [e for e in cl_data if e.get("end") == prev_report_period]
                if matched:
                    prev_metrics["current_liabilities"] = matched[0].get("val")
                elif len(cl_data) >= 2:
                    sorted_cl = sorted(cl_data, key=lambda x: x.get("end", ""), reverse=True)
                    prev_metrics["current_liabilities"] = sorted_cl[1].get("val")
            elif len(cl_data) >= 2:
                sorted_cl = sorted(cl_data, key=lambda x: x.get("end", ""), reverse=True)
                prev_metrics["current_liabilities"] = sorted_cl[1].get("val")
    
    return prev_metrics


def extract_financial_metrics_from_xbrl(xbrl, filing_type: str, detailed: bool = False) -> Dict:
    """
    Extract financial metrics from edgartools XBRL object.
    
    Key learning: facts from xbrl.query().by_concept().execute() are DICTIONARIES,
    not objects. Access values via fact['numeric_value'], fact['period_end'], etc.
    """
    metrics = {
        "revenue": None,
        "gross_profit": None,
        "operating_income": None,
        "net_income": None,
        "operating_cash_flow": None,
        "total_assets": None,
        "total_liabilities": None,
        "current_assets": None,
        "current_liabilities": None,
        "_period_end": None,
        "_fiscal_period": None,
        "_fiscal_year": None,
        "_revenue_concept_used": None,  # Track which revenue concept was used
    }
    
    if not xbrl:
        return metrics
    
    # Try to use facts DataFrame for dimension-aware filtering
    # Fall back to query().by_concept() if DataFrame not available
    facts_df = None
    try:
        if hasattr(xbrl, 'facts'):
            facts_df = xbrl.facts
            # Verify it's a DataFrame-like object
            if not hasattr(facts_df, 'columns'):
                facts_df = None
    except Exception:
        facts_df = None
    
    # Helper to get latest fact value for a concept with dimension filtering
    def get_latest_fact(concept_name, period_type_filter='duration', target_period_end=None):
        """
        Get most recent fact value for a concept, preferring consolidated parent entity.
        
        Args:
            concept_name: Concept to query (e.g., "Revenue", "Assets")
            period_type_filter: 'duration' for income/cash flow, 'instant' for balance sheet, None for both
            target_period_end: If provided, filter facts to match this period_end (ensures consistency)
        """
        try:
            # Try DataFrame approach first (has dimension columns)
            if facts_df is not None and hasattr(facts_df, 'columns'):
                # Filter by concept (case-insensitive partial match)
                concept_col = None
                for col in facts_df.columns:
                    if col.lower() in ['concept', 'tag', 'conceptname']:
                        concept_col = col
                        break
                
                if concept_col:
                    # Find facts matching concept
                    concept_filter = facts_df[concept_col].str.contains(concept_name, case=False, na=False, regex=False)
                    matched_facts = facts_df[concept_filter].copy()
                    
                    if not matched_facts.empty:
                        # Filter by period type if specified
                        if 'period_type' in matched_facts.columns and period_type_filter:
                            period_type_mask = matched_facts['period_type'] == period_type_filter
                            if period_type_mask.any():
                                matched_facts = matched_facts[period_type_mask]
                        
                        # Filter by target_period_end if provided (for balance sheet consistency)
                        if target_period_end and 'period_end' in matched_facts.columns:
                            target_period_str = str(target_period_end)[:10]
                            period_mask = matched_facts['period_end'].astype(str).str[:10] == target_period_str
                            if period_mask.any():
                                matched_facts = matched_facts[period_mask]
                            # If no exact match for balance sheet items, keep all (period mismatch less critical)
                            # For income statement items, exact match is required
                        
                        # Prefer facts with no segment/entity dimensions (consolidated parent)
                        # Look for dimension columns and filter out facts with non-empty dimension values
                        dimension_cols = [col for col in matched_facts.columns if 'dimension' in col.lower() or 'segment' in col.lower() or 'entity' in col.lower()]
                        if dimension_cols:
                            # Prefer rows where all dimension columns are NaN/empty (consolidated)
                            has_dimensions = matched_facts[dimension_cols].notna().any(axis=1)
                            consolidated = matched_facts[~has_dimensions]
                            if not consolidated.empty:
                                matched_facts = consolidated
                        
                        if not matched_facts.empty:
                            # Sort by period_end descending
                            if 'period_end' in matched_facts.columns:
                                matched_facts = matched_facts.sort_values('period_end', ascending=False, na_position='last')
                            
                            # Get first row
                            latest = matched_facts.iloc[0]
                            
                            # Extract values
                            value = latest.get('value') or latest.get('numeric_value')
                            period_end = latest.get('period_end')
                            fiscal_period = latest.get('fp') or latest.get('fiscal_period')
                            
                            return value, period_end, fiscal_period
            
            # Fallback: Use query().by_concept() API (list of dicts)
            facts = xbrl.query().by_concept(concept_name, exact=False).execute()
            if not facts:
                return None, None, None
            
            # Filter by period type if specified
            if period_type_filter:
                filtered_facts = [f for f in facts if f.get('period_type') == period_type_filter]
            else:
                filtered_facts = facts
            
            if not filtered_facts:
                # Try the other period type as fallback
                if period_type_filter == 'duration':
                    filtered_facts = [f for f in facts if f.get('period_type') == 'instant']
                elif period_type_filter == 'instant':
                    filtered_facts = [f for f in facts if f.get('period_type') == 'duration']
            
            if not filtered_facts:
                return None, None, None
            
            # If target_period_end is provided, filter to that specific period
            if target_period_end:
                target_period_str = str(target_period_end)[:10]
                period_matched = []
                for f in filtered_facts:
                    fact_period = f.get('period_end', '')
                    if fact_period:
                        fact_period_str = str(fact_period)[:10]
                        if fact_period_str == target_period_str:
                            period_matched.append(f)
                
                if period_matched:
                    # Prefer facts with fewer dimension keys (consolidated parent)
                    # Facts with minimal dimension info are typically consolidated
                    def dimension_count(f):
                        count = 0
                        for k in f.keys():
                            if 'dimension' in k.lower() or 'segment' in k.lower() or 'entity' in k.lower():
                                if f.get(k):  # Non-empty dimension value
                                    count += 1
                        return count
                    
                    # Sort by dimension count (fewer = more likely consolidated)
                    period_matched.sort(key=dimension_count)
                    filtered_facts = period_matched
                elif target_period_end:
                    # If no exact period match but we have filtered_facts, 
                    # for balance sheet items we might need closest match instead of exact
                    # Return None only if we really have no facts at all
                    if not filtered_facts:
                        return None, None, None
                    # Otherwise, proceed with filtered_facts (will use latest by period_end)
            
            if not filtered_facts:
                return None, None, None
            
            # Sort by period_end descending (most recent first)
            sorted_facts = sorted(filtered_facts, key=lambda f: f.get('period_end', ''), reverse=True)
            latest = sorted_facts[0]
            
            # Extract values from dict
            value = latest.get('numeric_value') or latest.get('value')
            period_end = latest.get('period_end')
            fiscal_period = latest.get('fp')
            
            return value, period_end, fiscal_period
            
        except Exception as e:
            # Debug: uncomment to see errors
            # print(f"  Debug get_latest_fact({concept_name}): {e}")
            return None, None, None
    
    # Extract revenue first to get the filing's period_end
    revenue, period_end, fiscal_period = get_latest_fact("Revenue")
    if revenue is None:
        revenue, period_end, fiscal_period = get_latest_fact("Revenues")
    
    target_period_end = None
    if revenue is not None:
        try:
            metrics["revenue"] = float(revenue)
            if period_end:
                target_period_end = str(period_end)[:10] if period_end else None
                metrics["_period_end"] = target_period_end
            if fiscal_period:
                metrics["_fiscal_period"] = str(fiscal_period)
        except (ValueError, TypeError):
            pass
    
    # Extract gross profit (use same period as revenue if available)
    gross_profit, _, _ = get_latest_fact("GrossProfit", target_period_end=target_period_end)
    if gross_profit is not None:
        try:
            metrics["gross_profit"] = float(gross_profit)
            if detailed:
                print(f"  DEBUG: ✓ Extracted gross profit from XBRL: ${float(gross_profit)/1e6:.2f}M")
        except (ValueError, TypeError):
            pass
    
    # Extract operating income (use same period as revenue if available)
    operating_income, _, _ = get_latest_fact("OperatingIncomeLoss", target_period_end=target_period_end)
    if operating_income is not None:
        try:
            metrics["operating_income"] = float(operating_income)
            if detailed:
                print(f"  DEBUG: ✓ Extracted operating income from XBRL: ${float(operating_income)/1e6:.2f}M")
        except (ValueError, TypeError):
            pass
    
    # Extract net income (use same period as revenue if available)
    net_income, _, _ = get_latest_fact("NetIncomeLoss", target_period_end=target_period_end)
    if net_income is not None:
        try:
            metrics["net_income"] = float(net_income)
        except (ValueError, TypeError):
            pass
    
    # Extract operating cash flow (use same period as revenue if available)
    cash_flow, _, _ = get_latest_fact("NetCashProvidedByUsedInOperatingActivities", target_period_end=target_period_end)
    if cash_flow is not None:
        try:
            metrics["operating_cash_flow"] = float(cash_flow)
        except (ValueError, TypeError):
            pass
    
    # Extract balance sheet items using balance_sheet().get_raw_data() DataFrame
    # This ensures proper entity context and period matching
    try:
        balance_sheet = xbrl.statements.balance_sheet()
        
        if balance_sheet is not None and hasattr(balance_sheet, 'get_raw_data'):
            try:
                bs_data = balance_sheet.get_raw_data()
                
                # Debug: Show what get_raw_data() returns
                if detailed:
                    print(f"\n  DEBUG: balance_sheet.get_raw_data() type: {type(bs_data)}")
                    if isinstance(bs_data, list):
                        print(f"  DEBUG: List length: {len(bs_data)}")
                        if len(bs_data) > 0:
                            print(f"  DEBUG: First item type: {type(bs_data[0])}")
                            print(f"  DEBUG: First item: {bs_data[0]}")
                
                # get_raw_data() returns a list, not DataFrame
                # Convert to DataFrame if it's a list
                if isinstance(bs_data, list) and len(bs_data) > 0:
                    try:
                        # Try to convert list to DataFrame
                        bs_df = pd.DataFrame(bs_data)
                        
                        if detailed:
                            print(f"  DEBUG: Converted to DataFrame, shape: {bs_df.shape}")
                            print(f"  DEBUG: DataFrame columns: {list(bs_df.columns)}")
                            if hasattr(bs_df, 'index'):
                                print(f"  DEBUG: DataFrame index (first 10): {list(bs_df.index)[:10]}")
                        
                        if hasattr(bs_df, 'columns') and len(bs_df.columns) > 0:
                            # The values are in a 'values' dict column, not a 'value' column
                            # Each row has: concept, values (dict mapping periods to amounts), has_values (bool)
                            
                            if detailed:
                                print(f"  DEBUG: All columns: {list(bs_df.columns)}")
                            
                            # Iterate through rows and extract from 'values' dict
                            for idx, row in bs_df.iterrows():
                                # Skip abstract concepts (they don't have actual values)
                                if row.get('is_abstract', False) or not row.get('has_values', False):
                                    continue
                                
                                # Get concept name
                                concept = str(row.get('concept', '')).lower()
                                if not concept:
                                    continue
                                
                                # Get values dictionary
                                values_dict = row.get('values', {})
                                if not isinstance(values_dict, dict) or len(values_dict) == 0:
                                    continue
                                
                                # Extract the most recent value from the dict
                                # The dict keys are likely dates or periods, values are amounts
                                # Sort keys to get the most recent period
                                try:
                                    # Try to get the latest value (assuming dict keys are sortable)
                                    sorted_keys = sorted(values_dict.keys(), reverse=True)
                                    if not sorted_keys:
                                        continue
                                    
                                    # Get the value for the most recent period
                                    latest_key = sorted_keys[0]
                                    value = values_dict[latest_key]
                                    
                                    if value is None or pd.isna(value):
                                        continue
                                    
                                    value_float = float(value)
                                    
                                    # Match concepts - exclude "Abstract" concepts
                                    if 'abstract' in concept:
                                        continue
                                    
                                    # Match AssetsCurrent (but not AssetsCurrentAbstract)
                                    if concept.endswith('assetscurrent') and not metrics.get("current_assets"):
                                        metrics["current_assets"] = value_float
                                        if detailed:
                                            print(f"  DEBUG: ✓ Found AssetsCurrent from '{row.get('concept', idx)}': {value_float} (period: {latest_key})")
                                    # Match LiabilitiesCurrent (but not LiabilitiesCurrentAbstract)
                                    elif concept.endswith('liabilitiescurrent') and not metrics.get("current_liabilities"):
                                        metrics["current_liabilities"] = value_float
                                        if detailed:
                                            print(f"  DEBUG: ✓ Found LiabilitiesCurrent from '{row.get('concept', idx)}': {value_float} (period: {latest_key})")
                                    # Match Assets (exact match, not containing)
                                    elif concept == 'us-gaap_assets' and not metrics.get("total_assets"):
                                        metrics["total_assets"] = value_float
                                        if detailed:
                                            print(f"  DEBUG: ✓ Found Assets from '{row.get('concept', idx)}': {value_float} (period: {latest_key})")
                                    # Match Liabilities (exact match, not containing)
                                    elif concept == 'us-gaap_liabilities' and not metrics.get("total_liabilities"):
                                        metrics["total_liabilities"] = value_float
                                        if detailed:
                                            print(f"  DEBUG: ✓ Found Liabilities from '{row.get('concept', idx)}': {value_float} (period: {latest_key})")
                                except (ValueError, TypeError, KeyError) as e:
                                    if detailed:
                                        print(f"  DEBUG: Error extracting value for concept '{concept}': {e}")
                                    continue
                    except Exception as e:
                        if detailed:
                            print(f"  DEBUG: Error converting list to DataFrame: {e}")
                            import traceback
                            traceback.print_exc()
            except Exception as e:
                if detailed:
                    print(f"  DEBUG: Error getting balance_sheet.get_raw_data(): {e}")
                    import traceback
                    traceback.print_exc()
    except Exception as e:
        if detailed:
            print(f"  DEBUG: Error using statements API: {e}")
        # Fallback to individual fact extraction if statements API fails
        # Fallback to individual fact extraction if statements API fails
        assets, _, _ = get_latest_fact("Assets", period_type_filter='instant', target_period_end=target_period_end)
        if assets is not None:
            try:
                metrics["total_assets"] = float(assets)
            except (ValueError, TypeError):
                pass
        
        liabilities, _, _ = get_latest_fact("Liabilities", period_type_filter='instant', target_period_end=target_period_end)
        if liabilities is not None:
            try:
                metrics["total_liabilities"] = float(liabilities)
            except (ValueError, TypeError):
                pass
        
        current_assets, _, _ = get_latest_fact("AssetsCurrent", period_type_filter='instant', target_period_end=target_period_end)
        if current_assets is not None:
            try:
                metrics["current_assets"] = float(current_assets)
            except (ValueError, TypeError):
                pass
        
        current_liabilities, _, _ = get_latest_fact("LiabilitiesCurrent", period_type_filter='instant', target_period_end=target_period_end)
        if current_liabilities is not None:
            try:
                metrics["current_liabilities"] = float(current_liabilities)
            except (ValueError, TypeError):
                pass
    
    return metrics


def extract_previous_period_metrics_from_xbrl(xbrl, filing_type: str, current_metrics: Dict) -> Dict:
    """
    Extract previous period metrics from XBRL for delta comparison.
    
    Args:
        xbrl: edgartools XBRL object
        filing_type: Filing type (10-K, 10-Q)
        current_metrics: Current period metrics (to identify current period_end)
        
    Returns:
        Dictionary with previous period metrics
    """
    prev_metrics = {
        "revenue": None,
        "net_income": None,
        "operating_cash_flow": None,
        "total_assets": None,
        "total_liabilities": None,
        "current_assets": None,
        "current_liabilities": None,
        "_period_end": None,
        "_fiscal_period": None,
    }
    
    if not xbrl or not current_metrics:
        return prev_metrics
    
    current_period_end = current_metrics.get("_period_end")
    if not current_period_end:
        return prev_metrics  # Can't find previous period without knowing current
    
    # Helper to get previous period fact (second most recent with different period_end)
    def get_previous_fact(concept_name, period_type_filter='duration'):
        """Get previous period fact value (different period_end than current)."""
        try:
            facts = xbrl.query().by_concept(concept_name, exact=False).execute()
            if not facts:
                return None, None, None
            
            # Filter by period type
            if period_type_filter:
                filtered_facts = [f for f in facts if f.get('period_type') == period_type_filter]
            else:
                filtered_facts = facts
            
            if not filtered_facts:
                # Try fallback
                if period_type_filter == 'duration':
                    filtered_facts = [f for f in facts if f.get('period_type') == 'instant']
                elif period_type_filter == 'instant':
                    filtered_facts = [f for f in facts if f.get('period_type') == 'duration']
            
            if not filtered_facts:
                return None, None, None
            
            # Sort by period_end descending
            sorted_facts = sorted(filtered_facts, key=lambda f: f.get('period_end', ''), reverse=True)
            
            # Find first fact with different period_end than current
            for fact in sorted_facts:
                fact_period_end = str(fact.get('period_end', ''))[:10] if fact.get('period_end') else None
                if fact_period_end and fact_period_end != current_period_end:
                    value = fact.get('numeric_value') or fact.get('value')
                    period_end = fact.get('period_end')
                    fiscal_period = fact.get('fp')
                    return value, period_end, fiscal_period
            
            return None, None, None
            
        except Exception:
            return None, None, None
    
    # Extract previous period revenue
    prev_revenue, prev_period_end, prev_fiscal_period = get_previous_fact("Revenue")
    if prev_revenue is None:
        prev_revenue, prev_period_end, prev_fiscal_period = get_previous_fact("Revenues")
    
    if prev_revenue is not None:
        try:
            prev_metrics["revenue"] = float(prev_revenue)
            if prev_period_end:
                prev_metrics["_period_end"] = str(prev_period_end)[:10]
            if prev_fiscal_period:
                prev_metrics["_fiscal_period"] = str(prev_fiscal_period)
        except (ValueError, TypeError):
            pass
    
    # Extract previous period gross profit
    prev_gp, _, _ = get_previous_fact("GrossProfit")
    if prev_gp is not None:
        try:
            prev_metrics["gross_profit"] = float(prev_gp)
        except (ValueError, TypeError):
            pass
    
    # Extract previous period operating income
    prev_oi, _, _ = get_previous_fact("OperatingIncomeLoss")
    if prev_oi is not None:
        try:
            prev_metrics["operating_income"] = float(prev_oi)
        except (ValueError, TypeError):
            pass
    
    # Extract previous period net income
    prev_ni, _, _ = get_previous_fact("NetIncomeLoss")
    if prev_ni is not None:
        try:
            prev_metrics["net_income"] = float(prev_ni)
        except (ValueError, TypeError):
            pass
    
    # Extract previous period operating cash flow
    prev_cf, _, _ = get_previous_fact("NetCashProvidedByUsedInOperatingActivities")
    if prev_cf is not None:
        try:
            prev_metrics["operating_cash_flow"] = float(prev_cf)
        except (ValueError, TypeError):
            pass
    
    # Extract previous period balance sheet items
    prev_assets, _, _ = get_previous_fact("Assets", period_type_filter='instant')
    if prev_assets is not None:
        try:
            prev_metrics["total_assets"] = float(prev_assets)
        except (ValueError, TypeError):
            pass
    
    prev_liabilities, _, _ = get_previous_fact("Liabilities", period_type_filter='instant')
    if prev_liabilities is not None:
        try:
            prev_metrics["total_liabilities"] = float(prev_liabilities)
        except (ValueError, TypeError):
            pass
    
    prev_current_assets, _, _ = get_previous_fact("AssetsCurrent", period_type_filter='instant')
    if prev_current_assets is not None:
        try:
            prev_metrics["current_assets"] = float(prev_current_assets)
        except (ValueError, TypeError):
            pass
    
    prev_current_liabilities, _, _ = get_previous_fact("LiabilitiesCurrent", period_type_filter='instant')
    if prev_current_liabilities is not None:
        try:
            prev_metrics["current_liabilities"] = float(prev_current_liabilities)
        except (ValueError, TypeError):
            pass
    
    return prev_metrics


# Port helper functions from test_edgar.py
def calculate_ratios(metrics: Dict) -> Dict:
    """
    Calculate financial ratios from metrics.
    Now includes Gross Margin and Operating Margin.
    """
    """Calculate financial ratios from metrics."""
    ratios = {
        "current_ratio": None,
        "debt_to_equity": None,
        "cash_flow_margin": None,
        "gross_margin": None,
        "operating_margin": None,
        "net_margin": None,
    }
    
    if metrics.get("current_assets") and metrics.get("current_liabilities"):
        if metrics["current_liabilities"] > 0:
            ratios["current_ratio"] = metrics["current_assets"] / metrics["current_liabilities"]
    
    if metrics.get("total_assets") and metrics.get("total_liabilities"):
        equity = metrics["total_assets"] - metrics["total_liabilities"]
        if equity > 0:
            ratios["debt_to_equity"] = metrics["total_liabilities"] / equity
    
    if metrics.get("operating_cash_flow") and metrics.get("revenue"):
        if metrics["revenue"] > 0:
            ratios["cash_flow_margin"] = (metrics["operating_cash_flow"] / metrics["revenue"]) * 100
    
    # Gross Margin = Gross Profit / Revenue
    if metrics.get("gross_profit") and metrics.get("revenue"):
        if metrics["revenue"] > 0:
            ratios["gross_margin"] = (metrics["gross_profit"] / metrics["revenue"]) * 100
    
    # Operating Margin = Operating Income / Revenue
    if metrics.get("operating_income") and metrics.get("revenue"):
        if metrics["revenue"] > 0:
            ratios["operating_margin"] = (metrics["operating_income"] / metrics["revenue"]) * 100
    
    # Net Margin = Net Income / Revenue
    if metrics.get("net_income") and metrics.get("revenue"):
        if metrics["revenue"] > 0:
            ratios["net_margin"] = (metrics["net_income"] / metrics["revenue"]) * 100
    
    # Earnings Per Share (EPS) = Net Income / Shares Outstanding
    if metrics.get("net_income") and metrics.get("shares_outstanding"):
        if metrics["shares_outstanding"] > 0:
            ratios["eps"] = metrics["net_income"] / metrics["shares_outstanding"]
    
    return ratios


def detect_red_flags(metrics: Dict, ratios: Dict) -> List[Tuple[str, float]]:
    """Detect red flags from financial metrics."""
    flags = []
    
    if metrics.get("operating_cash_flow") is not None and metrics["operating_cash_flow"] < 0:
        flags.append(("Negative operating cash flow", -0.3))
    
    if ratios.get("current_ratio") is not None and ratios["current_ratio"] < 1.0:
        flags.append(("Current ratio < 1.0 (liquidity risk)", -0.15))
    
    if ratios.get("debt_to_equity") is not None and ratios["debt_to_equity"] > 2.0:
        flags.append(("High debt-to-equity ratio", -0.15))
    
    if metrics.get("revenue_growth") is not None and metrics["revenue_growth"] < -10:
        flags.append(("Revenue declining >10%", -0.1))
    
    if metrics.get("net_income") is not None and metrics["net_income"] < 0:
        if metrics.get("revenue") and metrics["revenue"] > 0:
            flags.append(("Negative net income", -0.1))
    
    return flags


def calculate_investment_score(metrics: Dict, ratios: Dict, red_flags: List[Tuple[str, float]], 
                                filing_type: str, valuation: Optional[Dict] = None, debug: bool = False) -> float:
    """Calculate investment score (0.0 to 1.5) based on financial metrics."""
    # Ported from test_edgar.py lines 942-1122
    # Financial Strength (0-1.0, 35% weight)
    financial_strength = 0.0
    ni = metrics.get("net_income")
    ocf = metrics.get("operating_cash_flow")
    rev = metrics.get("revenue")
    
    if ocf and ocf > 0:
        if ocf > rev * 0.2 if rev else False:
            financial_strength += 0.40
        elif ocf > rev * 0.1 if rev else False:
            financial_strength += 0.30
        elif ocf > 0:
            financial_strength += 0.15
    
    if ni and ni > 0:
        financial_strength += 0.30
    
    cr = ratios.get("current_ratio")
    if cr and cr > 2.0:
        financial_strength += 0.30
    elif cr and cr > 1.5:
        financial_strength += 0.20
    
    financial_strength = min(financial_strength, 1.0)
    
    # Business Quality (0-1.0, 25% weight)
    business_quality = 0.0
    cfm = ratios.get("cash_flow_margin")
    if cfm is not None:
        if cfm > 20:
            business_quality += 0.30
        elif cfm > 10:
            business_quality += 0.20
        elif cfm > 5:
            business_quality += 0.15
        elif cfm > 0:
            business_quality += 0.08
    
    if ni is not None and rev:
        pm = (ni / rev) * 100
        if pm > 20:
            business_quality += 0.30
        elif pm > 10:
            business_quality += 0.20
        elif pm > 5:
            business_quality += 0.15
        elif pm > 0:
            business_quality += 0.08
    
    business_quality = min(business_quality, 1.0)
    
    # Valuation Component (0-1.0, 20% weight)
    valuation_score = 0.0
    if valuation:
        pe = valuation.get('pe_ratio')
        if pe is not None and pe > 0:
            if pe < 10:
                valuation_score += 0.30
            elif pe < 15:
                valuation_score += 0.25
            elif pe < 20:
                valuation_score += 0.15
            elif pe < 30:
                valuation_score += 0.08
        
        pb = valuation.get('pb_ratio')
        if pb is not None and pb > 0:
            if pb < 1.0:
                valuation_score += 0.25
            elif pb < 1.5:
                valuation_score += 0.20
            elif pb < 2.5:
                valuation_score += 0.12
            elif pb < 4.0:
                valuation_score += 0.05
        
        ps = valuation.get('ps_ratio')
        if ps is not None and ps > 0:
            if ps < 1.0:
                valuation_score += 0.25
            elif ps < 2.0:
                valuation_score += 0.20
            elif ps < 3.0:
                valuation_score += 0.12
            elif ps < 5.0:
                valuation_score += 0.05
        
        market_cap = valuation.get('market_cap')
        if market_cap and market_cap < 50_000_000:
            valuation_score *= 0.5
    
    valuation_score = min(valuation_score, 1.0)
    
    # Filing Type Bonus
    filing_bonus = 0.0
    if filing_type == "10-K":
        filing_bonus = 0.10
    elif filing_type == "10-Q":
        filing_bonus = 0.05
    
    # Red Flag Penalties
    total_penalty = sum(penalty for _, penalty in red_flags)
    
    # Combine weighted scores
    score = (
        financial_strength * 0.35 +
        business_quality * 0.25 +
        valuation_score * 0.20 +
        filing_bonus +
        total_penalty * 0.5
    )
    
    return max(0.0, min(1.5, score))


def calculate_delta_score(current_metrics: Dict, current_ratios: Dict, prev_metrics: Dict, filing_type: str = None, debug: bool = False) -> Tuple[float, Optional[Dict]]:
    """
    Calculate delta score based on improvement vs previous filing.
    
    Args:
        current_metrics: Current period metrics
        current_ratios: Current period ratios
        prev_metrics: Previous period metrics (sequential or YoY)
        filing_type: Type of filing (10-K, 10-Q) - used for context in debug output
        debug: If True, print detailed breakdown
    
    Returns:
        Tuple of (delta_score, breakdown_dict)
    """
    if not prev_metrics or not any(prev_metrics.values()):
        return (0.0, None if not debug else {"components": []})
    
    if debug and filing_type:
        comparison_type = "YoY" if filing_type == "10-Q" else "Sequential"
        print(f"  Calculating {comparison_type} delta score...")
    
    score = 0.0
    
    # Revenue growth (magnitude-aware)
    curr_rev = current_metrics.get("revenue")
    prev_rev = prev_metrics.get("revenue")
    if curr_rev and prev_rev and curr_rev > 0 and prev_rev > 0:
        pct_change = ((curr_rev - prev_rev) / prev_rev) * 100
        if curr_rev > prev_rev:
            # Revenue growth - reward based on magnitude
            if pct_change >= 100:
                score += 0.40  # Massive growth (100%+)
                if debug:
                    print(f"    +0.40 - Revenue Growth ({pct_change:+.1f}%) - Massive")
            elif pct_change >= 50:
                score += 0.30  # Exceptional growth (50-100%)
                if debug:
                    print(f"    +0.30 - Revenue Growth ({pct_change:+.1f}%) - Exceptional")
            elif pct_change >= 20:
                score += 0.25  # Strong growth (20-50%)
                if debug:
                    print(f"    +0.25 - Revenue Growth ({pct_change:+.1f}%) - Strong")
            elif pct_change >= 10:
                score += 0.20  # Good growth (10-20%)
                if debug:
                    print(f"    +0.20 - Revenue Growth ({pct_change:+.1f}%) - Good")
            elif pct_change >= 5:
                score += 0.15  # Moderate growth (5-10%)
                if debug:
                    print(f"    +0.15 - Revenue Growth ({pct_change:+.1f}%) - Moderate")
            else:
                score += 0.10  # Small growth (0-5%)
                if debug:
                    print(f"    +0.10 - Revenue Growth ({pct_change:+.1f}%) - Small")
        elif curr_rev < prev_rev:
            # Revenue decline - penalize based on magnitude
            if pct_change <= -20:
                score -= 0.30  # Severe decline (-20%+)
                if debug:
                    print(f"    -0.30 - Revenue Decline ({pct_change:+.1f}%) - Severe")
            elif pct_change <= -10:
                score -= 0.20  # Significant decline (-10% to -20%)
                if debug:
                    print(f"    -0.20 - Revenue Decline ({pct_change:+.1f}%) - Significant")
            elif pct_change <= -5:
                score -= 0.15  # Moderate decline (-5% to -10%)
                if debug:
                    print(f"    -0.15 - Revenue Decline ({pct_change:+.1f}%) - Moderate")
            else:
                score -= 0.10  # Small decline (0% to -5%)
                if debug:
                    print(f"    -0.10 - Revenue Decline ({pct_change:+.1f}%) - Small")
    
    # Gross Profit growth (magnitude-aware)
    curr_gp = current_metrics.get("gross_profit")
    prev_gp = prev_metrics.get("gross_profit")
    if curr_gp and prev_gp and curr_gp > 0 and prev_gp > 0:
        if curr_gp > prev_gp:
            pct_change = ((curr_gp - prev_gp) / prev_gp) * 100
            if pct_change >= 100:
                score += 0.30  # Massive growth (100%+)
                if debug:
                    print(f"    +0.30 - Gross Profit Growth ({pct_change:+.1f}%) - Massive")
            elif pct_change >= 50:
                score += 0.20  # Exceptional growth (50-100%)
                if debug:
                    print(f"    +0.20 - Gross Profit Growth ({pct_change:+.1f}%) - Exceptional")
            elif pct_change >= 20:
                score += 0.15  # Strong growth (20-50%)
                if debug:
                    print(f"    +0.15 - Gross Profit Growth ({pct_change:+.1f}%) - Strong")
            elif pct_change >= 10:
                score += 0.12  # Good growth (10-20%)
                if debug:
                    print(f"    +0.12 - Gross Profit Growth ({pct_change:+.1f}%) - Good")
            elif pct_change >= 5:
                score += 0.10  # Moderate growth (5-10%)
                if debug:
                    print(f"    +0.10 - Gross Profit Growth ({pct_change:+.1f}%) - Moderate")
            else:
                score += 0.08  # Small growth (0-5%)
                if debug:
                    print(f"    +0.08 - Gross Profit Growth ({pct_change:+.1f}%) - Small")
    
    # Get net income for profitability check (define before using)
    curr_ni = current_metrics.get("net_income")
    prev_ni = prev_metrics.get("net_income")
    
    # Margin expansion (using operating margin if available, otherwise net margin) - magnitude-aware
    # Note: Margin expansion without revenue growth may indicate cost-cutting rather than sustainable growth
    curr_oi = current_metrics.get("operating_income")
    prev_oi = prev_metrics.get("operating_income")
    if curr_oi and curr_rev and prev_oi and prev_rev and curr_rev > 0 and prev_rev > 0:
        curr_margin = (curr_oi / curr_rev) * 100
        prev_margin = (prev_oi / prev_rev) * 100
        if curr_margin > prev_margin:
            margin_change = curr_margin - prev_margin
            # Check if revenue declined while margins expanded (potential red flag)
            revenue_declined = curr_rev < prev_rev
            if revenue_declined:
                # Margin expansion without revenue growth - reduce reward (cost-cutting vs. growth)
                if margin_change >= 10.0:
                    score += 0.20  # Massive expansion but revenue down (reduced from 0.30)
                    if debug:
                        print(f"    +0.20 - Operating Margin Expansion ({margin_change:+.1f}%) - Massive (revenue declined)")
                elif margin_change >= 5.0:
                    score += 0.15  # Exceptional expansion but revenue down (reduced from 0.25)
                    if debug:
                        print(f"    +0.15 - Operating Margin Expansion ({margin_change:+.1f}%) - Exceptional (revenue declined)")
                elif margin_change >= 2.0:
                    score += 0.12  # Strong expansion but revenue down (reduced from 0.20)
                    if debug:
                        print(f"    +0.12 - Operating Margin Expansion ({margin_change:+.1f}%) - Strong (revenue declined)")
                elif margin_change >= 1.0:
                    score += 0.10  # Good expansion but revenue down (reduced from 0.15)
                    if debug:
                        print(f"    +0.10 - Operating Margin Expansion ({margin_change:+.1f}%) - Good (revenue declined)")
                else:
                    score += 0.08  # Small expansion but revenue down (reduced from 0.10)
                    if debug:
                        print(f"    +0.08 - Operating Margin Expansion ({margin_change:+.1f}%) - Small (revenue declined)")
            else:
                # Margin expansion with revenue growth - full reward
                if margin_change >= 10.0:
                    score += 0.30  # Massive expansion (10%+)
                    if debug:
                        print(f"    +0.30 - Operating Margin Expansion ({margin_change:+.1f}%) - Massive")
                elif margin_change >= 5.0:
                    score += 0.25  # Exceptional expansion (5-10%)
                    if debug:
                        print(f"    +0.25 - Operating Margin Expansion ({margin_change:+.1f}%) - Exceptional")
                elif margin_change >= 2.0:
                    score += 0.20  # Strong expansion (2-5%)
                    if debug:
                        print(f"    +0.20 - Operating Margin Expansion ({margin_change:+.1f}%) - Strong")
                elif margin_change >= 1.0:
                    score += 0.15  # Good expansion (1-2%)
                    if debug:
                        print(f"    +0.15 - Operating Margin Expansion ({margin_change:+.1f}%) - Good")
                elif margin_change >= 0.5:
                    score += 0.12  # Moderate expansion (0.5-1%)
                    if debug:
                        print(f"    +0.12 - Operating Margin Expansion ({margin_change:+.1f}%) - Moderate")
                else:
                    score += 0.10  # Small expansion (0-0.5%)
                    if debug:
                        print(f"    +0.10 - Operating Margin Expansion ({margin_change:+.1f}%) - Small")
    elif curr_ni is not None and curr_rev and prev_ni is not None and prev_rev and curr_rev > 0 and prev_rev > 0:
        # Fallback to net margin if operating income not available
        curr_margin = (curr_ni / curr_rev) * 100
        prev_margin = (prev_ni / prev_rev) * 100
        if curr_margin > prev_margin:
            margin_change = curr_margin - prev_margin
            # Check if revenue declined while margins expanded (potential red flag)
            revenue_declined = curr_rev < prev_rev
            if revenue_declined:
                # Margin expansion without revenue growth - reduce reward
                if margin_change >= 10.0:
                    score += 0.20  # Massive expansion but revenue down (reduced from 0.30)
                    if debug:
                        print(f"    +0.20 - Net Margin Expansion ({margin_change:+.1f}%) - Massive (revenue declined)")
                elif margin_change >= 5.0:
                    score += 0.15  # Exceptional expansion but revenue down (reduced from 0.25)
                    if debug:
                        print(f"    +0.15 - Net Margin Expansion ({margin_change:+.1f}%) - Exceptional (revenue declined)")
                elif margin_change >= 2.0:
                    score += 0.12  # Strong expansion but revenue down (reduced from 0.20)
                    if debug:
                        print(f"    +0.12 - Net Margin Expansion ({margin_change:+.1f}%) - Strong (revenue declined)")
                elif margin_change >= 1.0:
                    score += 0.10  # Good expansion but revenue down (reduced from 0.15)
                    if debug:
                        print(f"    +0.10 - Net Margin Expansion ({margin_change:+.1f}%) - Good (revenue declined)")
                else:
                    score += 0.08  # Small expansion but revenue down (reduced from 0.10)
                    if debug:
                        print(f"    +0.08 - Net Margin Expansion ({margin_change:+.1f}%) - Small (revenue declined)")
            else:
                # Margin expansion with revenue growth - full reward
                if margin_change >= 10.0:
                    score += 0.30  # Massive expansion (10%+)
                    if debug:
                        print(f"    +0.30 - Net Margin Expansion ({margin_change:+.1f}%) - Massive")
                elif margin_change >= 5.0:
                    score += 0.25  # Exceptional expansion (5-10%)
                    if debug:
                        print(f"    +0.25 - Net Margin Expansion ({margin_change:+.1f}%) - Exceptional")
                elif margin_change >= 2.0:
                    score += 0.20  # Strong expansion (2-5%)
                    if debug:
                        print(f"    +0.20 - Net Margin Expansion ({margin_change:+.1f}%) - Strong")
                elif margin_change >= 1.0:
                    score += 0.15  # Good expansion (1-2%)
                    if debug:
                        print(f"    +0.15 - Net Margin Expansion ({margin_change:+.1f}%) - Good")
                elif margin_change >= 0.5:
                    score += 0.12  # Moderate expansion (0.5-1%)
                    if debug:
                        print(f"    +0.12 - Net Margin Expansion ({margin_change:+.1f}%) - Moderate")
                else:
                    score += 0.10  # Small expansion (0-0.5%)
                    if debug:
                        print(f"    +0.10 - Net Margin Expansion ({margin_change:+.1f}%) - Small")
    
    # Cash flow improvement (magnitude-aware)
    curr_ocf = current_metrics.get("operating_cash_flow")
    prev_ocf = prev_metrics.get("operating_cash_flow")
    if curr_ocf is not None and prev_ocf is not None:
        if prev_ocf <= 0 < curr_ocf:
            # Special case: turnaround from negative to positive
            score += 0.20
            if debug:
                print(f"    +0.20 - Cash Flow Turnaround (negative to positive)")
        elif curr_ocf > prev_ocf:
            ocf_change_pct = ((curr_ocf - prev_ocf) / abs(prev_ocf) * 100) if prev_ocf != 0 else 0
            if ocf_change_pct >= 200:
                score += 0.40  # Massive improvement (200%+)
                if debug:
                    print(f"    +0.40 - Cash Flow Improvement ({ocf_change_pct:+.1f}%) - Massive")
            elif ocf_change_pct >= 100:
                score += 0.35  # Exceptional improvement (100-200%)
                if debug:
                    print(f"    +0.35 - Cash Flow Improvement ({ocf_change_pct:+.1f}%) - Exceptional")
            elif ocf_change_pct >= 50:
                score += 0.25  # Strong improvement (50-100%)
                if debug:
                    print(f"    +0.25 - Cash Flow Improvement ({ocf_change_pct:+.1f}%) - Strong")
            elif ocf_change_pct >= 20:
                score += 0.20  # Good improvement (20-50%)
                if debug:
                    print(f"    +0.20 - Cash Flow Improvement ({ocf_change_pct:+.1f}%) - Good")
            elif ocf_change_pct >= 10:
                score += 0.15  # Moderate improvement (10-20%)
                if debug:
                    print(f"    +0.15 - Cash Flow Improvement ({ocf_change_pct:+.1f}%) - Moderate")
            elif ocf_change_pct >= 5:
                score += 0.12  # Small improvement (5-10%)
                if debug:
                    print(f"    +0.12 - Cash Flow Improvement ({ocf_change_pct:+.1f}%) - Small")
            else:
                score += 0.10  # Minimal improvement (0-5%)
                if debug:
                    print(f"    +0.10 - Cash Flow Improvement ({ocf_change_pct:+.1f}%) - Minimal")
    
    # Profitability (magnitude-aware)
    if curr_ni is not None and prev_ni is not None:
        if prev_ni <= 0 < curr_ni:
            # Turnaround from loss to profit
            score += 0.25
            if debug:
                print(f"    +0.25 - Profitability Turnaround (loss to profit)")
        elif curr_ni > prev_ni and prev_ni > 0:
            # Growth in profitability
            pct_change = ((curr_ni - prev_ni) / abs(prev_ni) * 100) if prev_ni != 0 else 0
            if pct_change >= 100:
                score += 0.20  # Massive growth (100%+)
                if debug:
                    print(f"    +0.20 - Net Income Growth ({pct_change:+.1f}%) - Massive")
            elif pct_change >= 50:
                score += 0.15  # Exceptional growth (50-100%)
                if debug:
                    print(f"    +0.15 - Net Income Growth ({pct_change:+.1f}%) - Exceptional")
            elif pct_change >= 25:
                score += 0.12  # Strong growth (25-50%)
                if debug:
                    print(f"    +0.12 - Net Income Growth ({pct_change:+.1f}%) - Strong")
            elif pct_change >= 10:
                score += 0.10  # Good growth (10-25%)
                if debug:
                    print(f"    +0.10 - Net Income Growth ({pct_change:+.1f}%) - Good")
            else:
                score += 0.08  # Small growth (0-10%)
                if debug:
                    print(f"    +0.08 - Net Income Growth ({pct_change:+.1f}%) - Small")
        elif curr_ni < prev_ni and prev_ni > 0 and curr_ni > 0:
            # Decline in profitability (both periods profitable) - penalize
            pct_change = ((curr_ni - prev_ni) / abs(prev_ni) * 100) if prev_ni != 0 else 0
            if pct_change <= -30:
                score -= 0.25  # Severe decline (-30%+)
                if debug:
                    print(f"    -0.25 - Net Income Decline ({pct_change:+.1f}%) - Severe")
            elif pct_change <= -20:
                score -= 0.20  # Significant decline (-20% to -30%)
                if debug:
                    print(f"    -0.20 - Net Income Decline ({pct_change:+.1f}%) - Significant")
            elif pct_change <= -10:
                score -= 0.15  # Moderate decline (-10% to -20%)
                if debug:
                    print(f"    -0.15 - Net Income Decline ({pct_change:+.1f}%) - Moderate")
            else:
                score -= 0.10  # Small decline (0% to -10%)
                if debug:
                    print(f"    -0.10 - Net Income Decline ({pct_change:+.1f}%) - Small")
    
    # Balance sheet
    curr_de = current_ratios.get("debt_to_equity")
    if prev_metrics.get("total_assets") and prev_metrics.get("total_liabilities"):
        prev_equity = prev_metrics["total_assets"] - prev_metrics["total_liabilities"]
        if prev_equity > 0:
            prev_de = prev_metrics["total_liabilities"] / prev_equity
            if curr_de is not None and curr_de < prev_de:
                score += 0.10
    
    # Ensure score doesn't go negative (but can be below 0.0 for severe declines)
    final_score = max(-0.5, min(score, 1.0))  # Cap negative at -0.5, positive at 1.0
    return (final_score, None)


def classify_filing_delta(delta_score: float) -> str:
    """Classify filing as Good, Neutral, or Bad based on delta score."""
    if delta_score >= 0.4:
        return "Good"
    elif delta_score >= 0.15:
        return "Neutral"
    else:
        return "Bad"


def cik_to_ticker(cik: str) -> Optional[str]:
    """Map CIK to ticker symbol using edgartools Company."""
    cik = str(cik).zfill(10)
    
    if cik in _CIK_TO_TICKER_CACHE:
        return _CIK_TO_TICKER_CACHE[cik]
    
    try:
        company = Company(cik)
        ticker = company.get_ticker()
        _CIK_TO_TICKER_CACHE[cik] = ticker
        return ticker
    except Exception as e:
        # Debug: show what went wrong
        print(f"  DEBUG: Error getting ticker for CIK {cik}: {e}")
        _CIK_TO_TICKER_CACHE[cik] = None
        return None


def get_valuation_metrics(ticker: str, metrics: Dict) -> Optional[Dict]:
    """Get valuation metrics (P/E, P/B, P/S) for a ticker."""
    if not ticker or ticker in _VALUATION_CACHE:
        return _VALUATION_CACHE.get(ticker) if ticker else None
    
    try:
        time.sleep(0.05)
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        market_cap = info.get('marketCap')
        
        if not current_price or current_price <= 0:
            _VALUATION_CACHE[ticker] = None
            return None
        
        valuation = {
            'current_price': current_price,
            'market_cap': market_cap,
            'pe_ratio': None,
            'pb_ratio': None,
            'ps_ratio': None,
        }
        
        if metrics.get('net_income') and metrics['net_income'] > 0:
            if market_cap:
                valuation['pe_ratio'] = market_cap / metrics['net_income']
        
        if market_cap and metrics.get('total_assets') and metrics.get('total_liabilities'):
            book_value = metrics['total_assets'] - metrics['total_liabilities']
            if book_value > 0:
                valuation['pb_ratio'] = market_cap / book_value
        
        if market_cap and metrics.get('revenue') and metrics['revenue'] > 0:
            valuation['ps_ratio'] = market_cap / metrics['revenue']
        
        _VALUATION_CACHE[ticker] = valuation
        return valuation
        
    except Exception:
        _VALUATION_CACHE[ticker] = None
        return None


def get_consensus_eps_marketdata(ticker: str, report_period_end: str, filing_type: str, debug: bool = False) -> Optional[Dict]:
    """
    Get consensus EPS estimate from Market Data API.
    Only use with --analyze to conserve API calls (100/day limit).
    
    Args:
        ticker: Stock ticker symbol
        report_period_end: Report period end date (YYYY-MM-DD)
        filing_type: Filing type (10-K, 10-Q)
        debug: Print debug information
    
    Returns:
        Dictionary with consensus EPS data, or None
    """
    if not ticker or not MARKETDATA_API_TOKEN:
        if debug:
            print(f"    DEBUG: Missing ticker or API token")
        return None
    
    try:
        # Parse report period to determine fiscal quarter/year
        try:
            period_date = datetime.strptime(report_period_end, "%Y-%m-%d").date()
            fiscal_year = period_date.year
            fiscal_quarter = (period_date.month - 1) // 3 + 1  # Q1=1, Q2=2, Q3=3, Q4=4
            if debug:
                print(f"    DEBUG: Looking for FY{fiscal_year} Q{fiscal_quarter} (period end: {report_period_end})")
        except (ValueError, TypeError) as e:
            if debug:
                print(f"    DEBUG: Error parsing report period: {e}")
            return None
        
        # Market Data API endpoint
        url = f"{MARKETDATA_API_BASE}/stocks/earnings/{ticker}/"
        params = {
            "format": "json"
        }
        
        # Try different authentication methods
        # Based on API docs and common patterns, try multiple approaches
        auth_methods = [
            # Method 1: Token as query parameter (most common)
            {"params": {"token": MARKETDATA_API_TOKEN, "format": "json"}, "headers": {}},
            # Method 2: X-CSRFToken header (what user tried in curl)
            {"params": {"format": "json"}, "headers": {"X-CSRFToken": MARKETDATA_API_TOKEN}},
            # Method 3: Authorization Bearer header
            {"params": {"format": "json"}, "headers": {"Authorization": f"Bearer {MARKETDATA_API_TOKEN}"}},
            # Method 4: Authorization Token header
            {"params": {"format": "json"}, "headers": {"Authorization": f"Token {MARKETDATA_API_TOKEN}"}},
            # Method 5: X-API-Key header
            {"params": {"format": "json"}, "headers": {"X-API-Key": MARKETDATA_API_TOKEN}},
        ]
        
        response = None
        auth_success = False
        
        for i, auth_method in enumerate(auth_methods, 1):
            if debug:
                method_desc = f"query param" if "token" in auth_method["params"] else f"{list(auth_method['headers'].keys())[0]} header"
                print(f"    DEBUG: Trying authentication method {i}: {method_desc}")
            
            time.sleep(0.2)  # Rate limiting
            try:
                response = _session.get(url, params=auth_method["params"], headers=auth_method["headers"], timeout=30)
                
                # Check if authentication succeeded
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if data.get("s") == "ok":
                            auth_success = True
                            if debug:
                                print(f"    DEBUG: ✓ Authentication successful with method {i}")
                            break
                        elif data.get("s") == "error" and "token" in data.get("errmsg", "").lower():
                            # Token error, try next method
                            if debug:
                                print(f"    DEBUG: ✗ Token error: {data.get('errmsg')}")
                            continue
                    except:
                        # If we can't parse JSON but got 200, might be success
                        if debug:
                            print(f"    DEBUG: ⚠️  Got 200 but couldn't parse response, trying next method...")
                        continue
                elif response.status_code == 401:
                    if debug:
                        print(f"    DEBUG: ✗ 401 Unauthorized, trying next method...")
                    continue
                elif response.status_code == 402:
                    # Payment required - stop trying other methods
                    if debug:
                        print(f"    DEBUG: ⚠️  API returned 402 Payment Required")
                        print(f"    DEBUG: Earnings data endpoint requires paid plan (not available in free tier)")
                    return None
                else:
                    # Other error, might be valid response
                    if debug:
                        print(f"    DEBUG: Got status {response.status_code}, checking response...")
                    try:
                        data = response.json()
                        if data.get("s") == "ok":
                            auth_success = True
                            break
                    except:
                        pass
            except requests.RequestException as e:
                if debug:
                    print(f"    DEBUG: Request exception: {e}")
                continue
        
        if not auth_success or response is None:
            if debug:
                print(f"    DEBUG: ✗ All authentication methods failed")
            return None
        
        # Parse the response
        try:
            data = response.json()
        except Exception as e:
            if debug:
                print(f"    DEBUG: Error parsing JSON response: {e}")
            return None
        
        if debug:
            print(f"    DEBUG: API response status: {data.get('s')}")
            print(f"    DEBUG: Response keys: {list(data.keys())[:10]}")  # Show first 10 keys
        
        if data.get("s") != "ok":
            if debug:
                print(f"    DEBUG: API returned non-ok status: {data.get('s')}")
                if data.get("message"):
                    print(f"    DEBUG: API message: {data.get('message')}")
            return None
        
        # Find matching earnings data for the fiscal period
        # Try different possible field names
        fiscal_years = data.get("fiscalYear", []) or data.get("fiscal_year", []) or data.get("fiscalYearEnd", [])
        fiscal_quarters = data.get("fiscalQuarter", []) or data.get("fiscal_quarter", []) or data.get("quarter", [])
        estimated_eps = data.get("estimatedEPS", []) or data.get("estimated_eps", []) or data.get("consensusEPS", [])
        reported_eps = data.get("reportedEPS", []) or data.get("reported_eps", []) or data.get("actualEPS", [])
        surprise_eps = data.get("surpriseEPS", []) or data.get("surprise_eps", [])
        surprise_eps_pct = data.get("surpriseEPSpct", []) or data.get("surprise_eps_pct", []) or data.get("surpriseEPSPercent", [])
        report_dates = data.get("reportDate", []) or data.get("report_date", []) or data.get("date", [])
        
        if debug:
            print(f"    DEBUG: Found {len(fiscal_years)} earnings records")
            if fiscal_years:
                print(f"    DEBUG: Sample fiscal years: {fiscal_years[:5]}")
                print(f"    DEBUG: Sample fiscal quarters: {fiscal_quarters[:5] if fiscal_quarters else 'N/A'}")
        
        # If no data arrays, check if it's a single record format
        if not fiscal_years and isinstance(data, dict):
            # Might be a single record or different structure
            if "fiscalYear" in data or "fiscal_year" in data:
                # Single record format
                fy = data.get("fiscalYear") or data.get("fiscal_year")
                fq = data.get("fiscalQuarter") or data.get("fiscal_quarter")
                if fy == fiscal_year and (not fq or fq == fiscal_quarter):
                    result = {
                        "fiscal_year": fy,
                        "fiscal_quarter": fq,
                        "estimated_eps": data.get("estimatedEPS") or data.get("estimated_eps") or data.get("consensusEPS"),
                        "reported_eps": data.get("reportedEPS") or data.get("reported_eps") or data.get("actualEPS"),
                        "surprise_eps": data.get("surpriseEPS") or data.get("surprise_eps"),
                        "surprise_eps_pct": data.get("surpriseEPSpct") or data.get("surprise_eps_pct") or data.get("surpriseEPSPercent"),
                        "report_date": data.get("reportDate") or data.get("report_date") or data.get("date"),
                    }
                    if debug:
                        print(f"    DEBUG: Found single record match")
                    return result
        
        # Find entry matching our fiscal period
        if fiscal_years and fiscal_quarters:
            for i, (fy, fq) in enumerate(zip(fiscal_years, fiscal_quarters)):
                if debug and i < 3:
                    print(f"    DEBUG: Checking record {i}: FY{fy} Q{fq}")
                if fy == fiscal_year and fq == fiscal_quarter:
                    # Found matching period
                    result = {
                        "fiscal_year": fy,
                        "fiscal_quarter": fq,
                        "estimated_eps": estimated_eps[i] if i < len(estimated_eps) else None,
                        "reported_eps": reported_eps[i] if i < len(reported_eps) else None,
                        "surprise_eps": surprise_eps[i] if i < len(surprise_eps) else None,
                        "surprise_eps_pct": surprise_eps_pct[i] if i < len(surprise_eps_pct) else None,
                        "report_date": report_dates[i] if i < len(report_dates) else None,
                    }
                    if debug:
                        print(f"    DEBUG: ✓ Found match at index {i}")
                    return result
        
        # If no exact match, try to find closest period (within same fiscal year)
        if fiscal_years and fiscal_quarters:
            for i, (fy, fq) in enumerate(zip(fiscal_years, fiscal_quarters)):
                if fy == fiscal_year:
                    # Same fiscal year, use closest quarter
                    result = {
                        "fiscal_year": fy,
                        "fiscal_quarter": fq,
                        "estimated_eps": estimated_eps[i] if i < len(estimated_eps) else None,
                        "reported_eps": reported_eps[i] if i < len(reported_eps) else None,
                        "surprise_eps": surprise_eps[i] if i < len(surprise_eps) else None,
                        "surprise_eps_pct": surprise_eps_pct[i] if i < len(surprise_eps_pct) else None,
                        "report_date": report_dates[i] if i < len(report_dates) else None,
                    }
                    if debug:
                        print(f"    DEBUG: ⚠️  Using closest match: FY{fy} Q{fq} (requested Q{fiscal_quarter})")
                    return result
        
        if debug:
            print(f"    DEBUG: ✗ No matching fiscal period found")
            if fiscal_years:
                print(f"    DEBUG: Available fiscal years: {set(fiscal_years)}")
        
        return None
        
    except requests.RequestException as e:
        if debug:
            print(f"    DEBUG: Request error: {e}")
        return None
    except Exception as e:
        if debug:
            print(f"    DEBUG: Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        return None


def get_8k_exhibit_99_1(filing, detailed: bool = False) -> Optional[str]:
    """
    Get Exhibit 99.1 (press release) content from 8-K filing.
    
    Exhibit 99.1 is a separate file, not embedded in the main 8-K text.
    We need to:
    1. Get the filing index (index.json) to find all documents
    2. Identify Exhibit 99.1 from the index
    3. Fetch the separate Exhibit 99.1 file
    
    Args:
        filing: edgartools Filing object (8-K form)
        detailed: If True, print detailed information
        
    Returns:
        Exhibit 99.1 content as string, or None if not found
    """
    try:
        cik = str(filing.cik).zfill(10)
        accession = filing.accession_number
        
        # Remove dashes from accession number for URL
        accession_nodash = accession.replace("-", "")
        
        # Step 1: Get filing index
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/index.json"
        
        if detailed:
            print(f"  Fetching filing index: {index_url}")
        
        time.sleep(0.1)  # Rate limiting
        response = _session.get(index_url, timeout=30)
        response.raise_for_status()
        index_data = response.json()
        
        # Step 2: Find Exhibit 99.1 in the directory
        directory_items = index_data.get("directory", {}).get("item", [])
        
        if detailed:
            print(f"  Found {len(directory_items)} files in filing")
        
        exhibit_99_1 = None
        for item in directory_items:
            item_type = item.get("type", "").upper()
            item_name = item.get("name", "").lower()
            item_desc = item.get("description", "").lower() if item.get("description") else ""
            
            # Look for EX-99.1 type (best match)
            if item_type == "EX-99.1":
                exhibit_99_1 = item
                if detailed:
                    print(f"  ✓ Found Exhibit 99.1 by type: {item.get('name')}")
                break
            
            # Fallback: Look for filename containing 99 and press/earn
            if "99" in item_name and ("press" in item_name or "earn" in item_name or "ex99" in item_name):
                exhibit_99_1 = item
                if detailed:
                    print(f"  ✓ Found Exhibit 99.1 by filename: {item.get('name')}")
                break
            
            # Fallback: Look for description containing press/earnings
            if item_desc and ("press" in item_desc or "earnings" in item_desc or "99.1" in item_desc):
                exhibit_99_1 = item
                if detailed:
                    print(f"  ✓ Found Exhibit 99.1 by description: {item.get('name')}")
                break
        
        if not exhibit_99_1:
            if detailed:
                print(f"  ⚠️  Exhibit 99.1 not found in filing index")
            return None
        
        # Step 3: Fetch the exhibit content
        exhibit_name = exhibit_99_1.get("name")
        exhibit_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{exhibit_name}"
        
        if detailed:
            print(f"  Fetching Exhibit 99.1: {exhibit_url}")
        
        time.sleep(0.1)  # Rate limiting
        response = _session.get(exhibit_url, timeout=30)
        response.raise_for_status()
        exhibit_content = response.text
        
        if detailed:
            print(f"  ✓ Fetched Exhibit 99.1 ({len(exhibit_content)} chars)")
        
        return exhibit_content
        
    except Exception as e:
        if detailed:
            print(f"  ❌ Error fetching Exhibit 99.1: {e}")
            import traceback
            traceback.print_exc()
        return None


def parse_8k_items(filing, detailed: bool = False) -> Dict:
    """
    Parse 8-K filing content to identify which items are present.
    
    Common 8-K items:
    - Item 1.01: Entry into a Material Definitive Agreement (acquisitions, mergers)
    - Item 2.02: Results of Operations and Financial Condition (earnings announcements)
    - Item 8.01: Other Events
    
    Args:
        filing: edgartools Filing object (8-K form)
        detailed: If True, print detailed information
        
    Returns:
        Dictionary with item types found and their content snippets
    """
    items_found = {}
    
    try:
        # Try to get filing text content
        content = None
        
        # Method 1: Try text_url and fetch directly
        if hasattr(filing, 'text_url') and filing.text_url:
            try:
                if detailed:
                    print(f"  Fetching 8-K content from: {filing.text_url}")
                time.sleep(0.1)  # Rate limiting
                response = _session.get(filing.text_url, timeout=30)
                if response.status_code == 200:
                    content = response.text
                    if detailed:
                        print(f"  ✓ Fetched {len(content)} characters of content")
            except Exception as e:
                if detailed:
                    print(f"  ⚠️  Error fetching from text_url: {e}")
        
        # Method 2: Try html() method if available
        if not content and hasattr(filing, 'html'):
            try:
                content = filing.html() if callable(filing.html) else filing.html
            except:
                pass
        
        # Method 3: Try text() method if available
        if not content and hasattr(filing, 'text'):
            try:
                content = filing.text() if callable(filing.text) else filing.text
            except:
                pass
        
        if not content:
            if detailed:
                print(f"  ⚠️  Could not access 8-K content")
            return {"items_found": {}, "content_available": False}
        
        # Search for item markers (case-insensitive)
        content_upper = content.upper()
        
        # Common 8-K item patterns
        item_patterns = {
            "1.01": ["ITEM 1.01", "ITEM 1.01.", "ENTRY INTO A MATERIAL DEFINITIVE AGREEMENT"],
            "2.02": ["ITEM 2.02", "ITEM 2.02.", "RESULTS OF OPERATIONS AND FINANCIAL CONDITION"],
            "8.01": ["ITEM 8.01", "ITEM 8.01.", "OTHER EVENTS"],
            "1.02": ["ITEM 1.02", "ITEM 1.02.", "TERMINATION OF A MATERIAL DEFINITIVE AGREEMENT"],
            "2.01": ["ITEM 2.01", "ITEM 2.01.", "COMPLETION OF ACQUISITION"],
            "5.02": ["ITEM 5.02", "ITEM 5.02.", "DEPARTURE OF DIRECTORS"],
        }
        
        for item_num, patterns in item_patterns.items():
            for pattern in patterns:
                if pattern in content_upper:
                    # Find the position and extract a snippet
                    idx = content_upper.find(pattern)
                    if idx != -1:
                        # Extract ~500 chars after the item marker
                        snippet_start = idx
                        snippet_end = min(len(content), idx + len(pattern) + 500)
                        snippet = content[snippet_start:snippet_end].strip()
                        
                        # Clean up snippet (remove excessive whitespace)
                        snippet = ' '.join(snippet.split())
                        if len(snippet) > 400:
                            snippet = snippet[:400] + "..."
                        
                        items_found[item_num] = {
                            "pattern_matched": pattern,
                            "snippet": snippet
                        }
                        
                        if detailed:
                            print(f"  ✓ Found Item {item_num}: {pattern}")
                        break  # Found this item, move to next
        
        if detailed:
            if items_found:
                print(f"  ✓ Identified {len(items_found)} item(s): {', '.join(items_found.keys())}")
            else:
                print(f"  ⚠️  No standard items found in 8-K")
        
        return {
            "items_found": items_found,
            "content_available": True,
            "content_length": len(content),
            "content": content  # Return content for further parsing
        }
        
    except Exception as e:
        if detailed:
            print(f"  ❌ Error parsing 8-K items: {e}")
            import traceback
            traceback.print_exc()
        return {"items_found": {}, "content_available": False, "error": str(e)}


def extract_earnings_from_item_202(content: str, detailed: bool = False, search_full_content: bool = False) -> Dict:
    """
    Extract earnings data from Item 2.02 section of 8-K filing.
    
    Looks for:
    - EPS (earnings per share) - GAAP and non-GAAP
    - Revenue/sales
    - Net income
    - Period information (quarter, fiscal year)
    - Guidance (forward-looking statements)
    
    Also checks Exhibit 99.1 (press release) which often contains the actual earnings data.
    
    Args:
        content: Full 8-K filing text content
        detailed: If True, print detailed extraction info
        
    Returns:
        Dictionary with extracted earnings data
    """
    earnings_data = {
        "eps_gaap": None,
        "eps_nongaap": None,
        "eps_adjusted": None,
        "revenue": None,
        "net_income": None,
        "period": None,
        "fiscal_quarter": None,
        "fiscal_year": None,
        "guidance_mentioned": False,
    }
    
    try:
        # If searching full content, skip section extraction
        if search_full_content:
            search_text = content
            if detailed:
                print(f"  Searching full content ({len(content)} chars)")
        else:
            # First, try to find Exhibit 99.1 (press release) - this often has the actual earnings data
            exhibit_991_markers = ["EXHIBIT 99.1", "EXHIBIT 99.1.", "EXHIBIT 99-1"]
            exhibit_start = -1
            content_upper = content.upper()
            
            for marker in exhibit_991_markers:
                idx = content_upper.find(marker)
                if idx != -1:
                    exhibit_start = idx
                    break
            
            # Extract text to search (prioritize Exhibit 99.1 if found, otherwise Item 2.02)
            search_text = content
            if exhibit_start != -1:
                # Extract Exhibit 99.1 section (usually goes to end or next exhibit)
                # Take a larger section to capture the full press release
                exhibit_end = min(len(content), exhibit_start + 50000)  # Up to 50KB for press release
                next_exhibit_markers = ["EXHIBIT 99.2", "EXHIBIT 10", "SIGNATURE"]
                for marker in next_exhibit_markers:
                    idx = content_upper.find(marker, exhibit_start + 100)
                    if idx != -1 and idx < exhibit_end:
                        exhibit_end = idx
                search_text = content[exhibit_start:exhibit_end]
                
                # Decode HTML entities
                try:
                    search_text = html.unescape(search_text)
                except:
                    pass
                
                # Remove HTML tags (basic cleanup) - be more aggressive
                search_text = re.sub(r'<script[^>]*>.*?</script>', ' ', search_text, flags=re.DOTALL | re.IGNORECASE)
                search_text = re.sub(r'<style[^>]*>.*?</style>', ' ', search_text, flags=re.DOTALL | re.IGNORECASE)
                search_text = re.sub(r'<[^>]+>', ' ', search_text)
                search_text = re.sub(r'&[a-z]+;', ' ', search_text, flags=re.IGNORECASE)  # Remove remaining entities
                search_text = ' '.join(search_text.split())  # Normalize whitespace
                
                if detailed:
                    print(f"  ✓ Found Exhibit 99.1 (press release) - extracting from there ({len(search_text)} chars)")
            else:
                # Fall back to Item 2.02 section
                item_202_markers = ["ITEM 2.02", "ITEM 2.02.", "RESULTS OF OPERATIONS AND FINANCIAL CONDITION"]
                item_start = -1
                for marker in item_202_markers:
                    idx = content_upper.find(marker)
                    if idx != -1:
                        item_start = idx
                        break
                
                if item_start == -1:
                    if detailed:
                        print(f"  ⚠️  Could not find Item 2.02 section or Exhibit 99.1")
                    return earnings_data
                
                # Extract Item 2.02 section (up to next Item or end of document)
                next_item_markers = ["ITEM 1.01", "ITEM 2.01", "ITEM 8.01", "ITEM 9.01"]
                item_end = len(content)
                for marker in next_item_markers:
                    idx = content_upper.find(marker, item_start + 100)
                    if idx != -1 and idx < item_end:
                        item_end = idx
                search_text = content[item_start:item_end]
                if detailed:
                    print(f"  Extracting from Item 2.02 section ({len(search_text)} chars)")
        
        # Clean up search_text (decode HTML, remove tags) if not already done
        if search_full_content:
            try:
                search_text = html.unescape(search_text)
            except:
                pass
            search_text = re.sub(r'<script[^>]*>.*?</script>', ' ', search_text, flags=re.DOTALL | re.IGNORECASE)
            search_text = re.sub(r'<style[^>]*>.*?</style>', ' ', search_text, flags=re.DOTALL | re.IGNORECASE)
            search_text = re.sub(r'<[^>]+>', ' ', search_text)
            search_text = re.sub(r'&[a-z]+;', ' ', search_text, flags=re.IGNORECASE)
            search_text = ' '.join(search_text.split())
        
        # Extract EPS - look for patterns like "$0.20", "EPS of $0.20", "earnings per share of $0.20"
        # EPS patterns (GAAP, non-GAAP, adjusted)
        # More specific patterns to avoid matching par values
        # Handle various formats: "EPS were $2.91", "earnings per share of $2.91", "$2.91 per share"
        eps_patterns = [
            r'(?:GAAP\s+)?EPS\s+(?:of\s+|were\s+)?\$?\s*([\d,]+\.?\d{2,})',  # "EPS were $2.91" or "EPS of $2.91"
            r'earnings\s+per\s+(?:common\s+)?share\s+(?:\(?EPS\)?)?\s+(?:of\s+|were\s+)?\$?\s*([\d,]+\.?\d{2,})',  # "earnings per share (EPS) were $2.91"
            r'(?:basic|diluted)\s+(?:and\s+(?:basic|diluted)\s+)?(?:earnings\s+per\s+(?:common\s+)?share|EPS)\s+(?:of\s+|were\s+)?\$?\s*([\d,]+\.?\d{2,})',  # "Basic and diluted earnings per common share (EPS) were $2.91"
            r'(?:diluted|basic)\s+EPS\s+(?:of\s+|were\s+)?\$?\s*([\d,]+\.?\d{2,})',
            r'\$([\d,]+\.?\d{2,})\s+per\s+(?:diluted|basic)?\s*(?:share|diluted share|common share)',  # "$2.91 per share"
            r'EPS\s+\(?(?:diluted|basic|GAAP|non-GAAP)\)?\s*(?:of\s+|were\s+)?\$?\s*([\d,]+\.?\d{2,})',
        ]
        
        # Non-GAAP/Adjusted EPS patterns
        nongaap_patterns = [
            r'(?:non[-\s]?GAAP|adjusted)\s+EPS\s+(?:of\s+)?\$?\s*([\d,]+\.?\d{2,})',
            r'(?:non[-\s]?GAAP|adjusted)\s+earnings\s+per\s+share\s+(?:of\s+)?\$?\s*([\d,]+\.?\d{2,})',
            r'Adjusted\s+EPS\s+(?:of\s+)?\$?\s*([\d,]+\.?\d{2,})',
        ]
        
        # Try to find EPS values
        for pattern in eps_patterns:
            matches = re.finditer(pattern, search_text, re.IGNORECASE)
            for match in matches:
                value_str = match.group(1).replace(',', '')
                try:
                    value = float(value_str)
                    # Check if it's labeled as non-GAAP or adjusted
                    context = search_text[max(0, match.start()-50):match.end()+50].lower()
                    if 'nongaap' in context or 'non-gaap' in context or 'adjusted' in context:
                        if earnings_data["eps_nongaap"] is None:
                            earnings_data["eps_nongaap"] = value
                            if detailed:
                                print(f"  ✓ Found non-GAAP EPS: ${value:.2f}")
                    elif earnings_data["eps_gaap"] is None:
                        earnings_data["eps_gaap"] = value
                        if detailed:
                            print(f"  ✓ Found GAAP EPS: ${value:.2f}")
                except ValueError:
                    pass
        
        # Try non-GAAP specific patterns
        for pattern in nongaap_patterns:
            matches = re.finditer(pattern, search_text, re.IGNORECASE)
            for match in matches:
                value_str = match.group(1).replace(',', '')
                try:
                    value = float(value_str)
                    if earnings_data["eps_nongaap"] is None:
                        earnings_data["eps_nongaap"] = value
                        if detailed:
                            print(f"  ✓ Found non-GAAP EPS: ${value:.2f}")
                except ValueError:
                    pass
        
        # Extract Revenue - look for patterns like "$1.52 million", "revenue of $1.52M"
        revenue_patterns = [
            r'(?:total\s+)?(?:net\s+)?(?:sales\s+)?revenue\s+(?:of\s+)?\$?([\d,]+\.?\d*)\s*(?:million|M|billion|B|thousand|K)',
            r'(?:total\s+)?(?:net\s+)?sales\s+(?:of\s+)?\$?([\d,]+\.?\d*)\s*(?:million|M|billion|B|thousand|K)',
            r'revenue\s+of\s+\$?([\d,]+\.?\d*)\s*(?:million|M|billion|B)',
        ]
        
        for pattern in revenue_patterns:
            matches = re.finditer(pattern, search_text, re.IGNORECASE)
            for match in matches:
                value_str = match.group(1).replace(',', '')
                multiplier_str = search_text[match.start():match.end()].upper()
                try:
                    value = float(value_str)
                    # Apply multiplier
                    if 'BILLION' in multiplier_str or 'B' in multiplier_str:
                        value *= 1_000_000_000
                    elif 'MILLION' in multiplier_str or 'M' in multiplier_str:
                        value *= 1_000_000
                    elif 'THOUSAND' in multiplier_str or 'K' in multiplier_str:
                        value *= 1_000
                    
                    if earnings_data["revenue"] is None:
                        earnings_data["revenue"] = value
                        if detailed:
                            print(f"  ✓ Found Revenue: ${value/1_000_000:.2f}M")
                except ValueError:
                    pass
        
        # Extract period information (Q4 2025, fiscal year 2025, etc.)
        period_patterns = [
            r'(?:fiscal\s+)?(?:fourth\s+)?quarter\s+(?:ended\s+)?(?:fiscal\s+year\s+)?(\d{4})',
            r'Q(\d)\s+(?:of\s+)?(?:fiscal\s+year\s+)?(\d{4})',
            r'fiscal\s+year\s+(\d{4})',
        ]
        
        for pattern in period_patterns:
            matches = re.finditer(pattern, search_text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) == 2:  # Q4 2025 format
                    quarter = int(match.group(1))
                    year = int(match.group(2))
                    earnings_data["fiscal_quarter"] = quarter
                    earnings_data["fiscal_year"] = year
                    if detailed:
                        print(f"  ✓ Found period: Q{quarter} {year}")
                    break
                elif len(match.groups()) == 1:  # Just year
                    year = int(match.group(1))
                    earnings_data["fiscal_year"] = year
                    if detailed:
                        print(f"  ✓ Found fiscal year: {year}")
                    break
        
        # Check for guidance mentions and extract snippets
        guidance_keywords = ['guidance', 'outlook', 'forecast', 'expect', 'anticipate', 'project']
        search_text_lower = search_text.lower()
        guidance_snippets = []
        
        # Boilerplate/disclaimer phrases to filter out
        disclaimer_phrases = [
            'may fail to achieve',
            'forward-looking statements',
            'risks and uncertainties',
            'no assurance',
            'not a forecast',
            'not intended to',
            'actual results may differ',
            'we cannot guarantee',
            'subject to risks',
            'could cause actual',
            'may not be',
            'uncertainties include',
        ]
        
        # Positive sentiment indicators (prioritize these)
        positive_indicators = [
            'raising', 'increased', 'strong', 'momentum', 'growth',
            'positive', 'exceeded', 'beat', 'above', 'higher',
            'record', 'milestone', 'accelerat'
        ]
        
        def is_disclaimer(text):
            """Check if snippet is likely boilerplate disclaimer."""
            text_lower = text.lower()
            return any(phrase in text_lower for phrase in disclaimer_phrases)
        
        def has_numbers(text):
            """Check if snippet contains dollar amounts or percentages."""
            return bool(re.search(r'\$[\d,]+|\d+\s*%|\d+\s*(million|billion|M|B)', text, re.IGNORECASE))
        
        def has_positive_sentiment(text):
            """Check if snippet has positive sentiment words."""
            text_lower = text.lower()
            return any(word in text_lower for word in positive_indicators)
        
        for keyword in guidance_keywords:
            idx = 0
            while True:
                idx = search_text_lower.find(keyword, idx)
                if idx == -1:
                    break
                # Extract surrounding context (100 chars before, 200 chars after)
                start = max(0, idx - 100)
                end = min(len(search_text), idx + 200)
                snippet = search_text[start:end].strip()
                # Clean up the snippet
                snippet = ' '.join(snippet.split())  # Normalize whitespace
                
                # Filter out disclaimers
                if snippet and snippet not in guidance_snippets and not is_disclaimer(snippet):
                    guidance_snippets.append(snippet)
                idx += len(keyword)
                # Limit to 3 snippets per keyword
                if len([s for s in guidance_snippets if keyword in s.lower()]) >= 2:
                    break
        
        # Sort snippets: prioritize those with numbers or positive sentiment
        def snippet_score(s):
            score = 0
            if has_numbers(s):
                score += 2
            if has_positive_sentiment(s):
                score += 1
            return score
        
        guidance_snippets.sort(key=snippet_score, reverse=True)
        
        if guidance_snippets:
            earnings_data["guidance_mentioned"] = True
            earnings_data["guidance_snippets"] = guidance_snippets[:5]  # Keep top 5
            if detailed:
                print(f"  ✓ Guidance/outlook mentioned in filing")
                print(f"  GUIDANCE SNIPPETS (filtered, sorted by relevance):")
                for i, snippet in enumerate(guidance_snippets[:5], 1):
                    # Truncate long snippets
                    wrapped = snippet[:250] if len(snippet) > 250 else snippet
                    # Add indicator if has numbers or positive sentiment
                    indicators = []
                    if has_numbers(snippet):
                        indicators.append("$")
                    if has_positive_sentiment(snippet):
                        indicators.append("+")
                    indicator_str = f"[{','.join(indicators)}] " if indicators else ""
                    print(f"    {i}. {indicator_str}{wrapped}")
        
        # If nothing found, show a sample of the text for debugging
        if detailed and not any([earnings_data["eps_gaap"], earnings_data["eps_nongaap"], earnings_data["revenue"]]):
            print(f"  ⚠️  No EPS or revenue found in extracted text")
            print(f"  Sample text (first 500 chars): {search_text[:500]}...")
        
        return earnings_data
        
    except Exception as e:
        if detailed:
            print(f"  ❌ Error extracting earnings data: {e}")
            import traceback
            traceback.print_exc()
        return earnings_data


def analyze_8k_filing(filing, detailed: bool = False) -> Optional[Dict]:
    """
    Analyze an 8-K filing (event-driven, not period-based financials).
    
    Args:
        filing: edgartools Filing object (8-K form)
        detailed: If True, print detailed breakdown
        
    Returns:
        Result dictionary or None if analysis fails
    """
    try:
        cik = str(filing.cik).zfill(10)
        company_name = getattr(filing, 'company', None) or f"CIK-{cik}"
        
        if detailed:
            print(f"  Analyzing 8-K filing for {company_name}")
            print(f"  Filing Date: {filing.filing_date}")
            print(f"  Accession: {filing.accession_number}")
        
        # Get ticker
        ticker = cik_to_ticker(cik)
        if not ticker and detailed:
            print(f"  ⚠️  No ticker found")
        
        # Parse 8-K items (Item 2.02, Item 1.01, etc.)
        items_info = parse_8k_items(filing, detailed=detailed)
        items_found = items_info.get("items_found", {})
        content = items_info.get("content", "")
        
        # Get Exhibit 99.1 if Item 2.02 (earnings announcement)
        exhibit_99_1_content = None
        if "2.02" in items_found:
            exhibit_99_1_content = get_8k_exhibit_99_1(filing, detailed=detailed)
        
        # Get full content if we have it (for fallback search)
        if not content and items_info.get("content_available"):
            # Re-fetch content for extraction
            if hasattr(filing, 'text_url') and filing.text_url:
                try:
                    time.sleep(0.1)
                    response = _session.get(filing.text_url, timeout=30)
                    if response.status_code == 200:
                        content = response.text
                except:
                    pass
        
        # Determine primary item type
        primary_item = None
        if "2.02" in items_found:
            primary_item = "2.02"  # Earnings announcement
        elif "1.01" in items_found:
            primary_item = "1.01"  # Acquisition/merger
        elif "8.01" in items_found:
            primary_item = "8.01"  # Other events
        elif items_found:
            primary_item = list(items_found.keys())[0]  # First item found
        
        if detailed:
            if primary_item:
                print(f"  Primary item type: Item {primary_item}")
            else:
                print(f"  ⚠️  Could not identify primary item type")
        
        # Extract earnings data if Item 2.02
        earnings_data = {}
        if primary_item == "2.02":
            # Prioritize Exhibit 99.1 content (the actual press release)
            if exhibit_99_1_content:
                if detailed:
                    print(f"  Extracting earnings data from Exhibit 99.1 (press release)...")
                earnings_data = extract_earnings_from_item_202(exhibit_99_1_content, detailed=detailed, search_full_content=True)
            elif content:
                # Fallback to main content if Exhibit 99.1 not available
                if detailed:
                    print(f"  ⚠️  Exhibit 99.1 not found, searching main filing content...")
                earnings_data = extract_earnings_from_item_202(content, detailed=detailed)
            
            # If still no data found, search entire main content as last resort
            if not any([earnings_data.get("eps_gaap"), earnings_data.get("eps_nongaap"), earnings_data.get("revenue")]) and content:
                if detailed:
                    print(f"  ⚠️  No earnings data found, searching entire filing content...")
                earnings_data_full = extract_earnings_from_item_202(content, detailed=False, search_full_content=True)
                # Merge results (prefer original, but fill in from full search)
                for key in earnings_data:
                    if earnings_data[key] is None and earnings_data_full.get(key):
                        earnings_data[key] = earnings_data_full[key]
                        if detailed:
                            value = earnings_data_full[key]
                            if isinstance(value, (int, float)):
                                if key.startswith("eps"):
                                    print(f"  ✓ Found {key}: ${value:.2f}")
                                elif key == "revenue":
                                    print(f"  ✓ Found {key}: ${value/1_000_000:.2f}M")
                                else:
                                    print(f"  ✓ Found {key}: {value}")
                            else:
                                print(f"  ✓ Found {key}: {value}")
        
        # Return result with item information
        return {
            "cik": cik,
            "company": company_name,
            "ticker": ticker,
            "form": "8-K",
            "filing_date": filing.filing_date,
            "filing_ref": filing.accession_number,
            "score": 0.0,  # Placeholder - will be calculated based on 8-K content
            "absolute_score": 0.0,
            "delta_score": 0.0,
            "delta_label": "Neutral",
            "combined_score": 0.0,
            "momentum_score": 0.5,
            "price_momentum": get_price_momentum(ticker, filing.filing_date.strftime("%Y%m%d"), "Neutral") if ticker else None,
            "metrics": {},  # 8-Ks don't have period-based metrics
            "ratios": {},
            "red_flags": [],
            "valuation": get_valuation_metrics(ticker, {}) if ticker else None,
            "insider_activity": None,
            "8k_items": items_found,
            "8k_primary_item": primary_item,
            "8k_earnings_data": earnings_data if primary_item == "2.02" else None,
        }
        
    except Exception as e:
        if detailed:
            print(f"  ❌ Error analyzing 8-K: {e}")
            import traceback
            traceback.print_exc()
        return None


def get_price_momentum(ticker: str, filing_date_str: str, delta_label: str) -> Optional[Dict]:
    """Calculate price momentum relative to filing date."""
    if not ticker:
        return None
    
    try:
        filing_date = datetime.strptime(filing_date_str, "%Y%m%d").date()
        start_date = filing_date - timedelta(days=30)
        end_date = filing_date + timedelta(days=90)
        
        today = datetime.now().date()
        if end_date > today:
            end_date = today
        
        time.sleep(0.05)
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            return None
        
        filing_prices = hist[hist.index.date <= filing_date]
        if filing_prices.empty:
            return None
        
        filing_price = filing_prices.iloc[-1]["Close"]
        
        # Get 1-day price (next trading day after filing)
        after_1d_date = filing_date + timedelta(days=1)
        after_1d_prices = hist[hist.index.date >= after_1d_date]
        after_1d_price = after_1d_prices.iloc[0]["Close"] if not after_1d_prices.empty else None
        
        after_7d_date = filing_date + timedelta(days=7)
        after_30d_date = filing_date + timedelta(days=30)
        after_60d_date = filing_date + timedelta(days=60)
        after_90d_date = filing_date + timedelta(days=90)
        
        after_7d_prices = hist[hist.index.date >= after_7d_date]
        after_7d_price = after_7d_prices.iloc[0]["Close"] if not after_7d_prices.empty else None
        
        after_30d_prices = hist[hist.index.date >= after_30d_date]
        after_30d_price = after_30d_prices.iloc[0]["Close"] if not after_30d_prices.empty else None
        
        after_60d_prices = hist[hist.index.date >= after_60d_date]
        after_60d_price = after_60d_prices.iloc[0]["Close"] if not after_60d_prices.empty else None
        
        after_90d_prices = hist[hist.index.date >= after_90d_date]
        after_90d_price = after_90d_prices.iloc[0]["Close"] if not after_90d_prices.empty else None
        
        momentum_score = 0.5
        
        if delta_label == "Good":
            if after_7d_price:
                change_7d = ((after_7d_price - filing_price) / filing_price) * 100
                if change_7d < -5:
                    momentum_score = 0.9
                elif change_7d < 0:
                    momentum_score = 0.8
                elif change_7d < 10:
                    momentum_score = 0.6
        
        price_change_1d = ((after_1d_price - filing_price) / filing_price) * 100 if after_1d_price else None
        price_change_7d = ((after_7d_price - filing_price) / filing_price) * 100 if after_7d_price else None
        price_change_30d = ((after_30d_price - filing_price) / filing_price) * 100 if after_30d_price else None
        price_change_60d = ((after_60d_price - filing_price) / filing_price) * 100 if after_60d_price else None
        price_change_90d = ((after_90d_price - filing_price) / filing_price) * 100 if after_90d_price else None
        
        return {
            "momentum_score": momentum_score,
            "filing_price": filing_price,
            "price_change_1d": price_change_1d,
            "price_change_7d": price_change_7d,
            "price_change_30d": price_change_30d,
            "price_change_60d": price_change_60d,
            "price_change_90d": price_change_90d,
        }
    except Exception:
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EDGAR filing analysis using edgartools")
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    parser.add_argument("--analyze", type=str, help="Analyze specific filing by accession number (e.g., --analyze 0001660280-25-000128)")
    parser.add_argument("--limit", type=int, help="Limit number of results")
    parser.add_argument("--backtest", action="store_true", help="Enable backtesting mode")
    parser.add_argument("--sort-by-7d", action="store_true", help="Sort results by 7-day price change instead of score")
    
    args = parser.parse_args()
    
    if args.analyze:
        # Single filing analysis mode
        analyze_filing_by_accession(args.analyze)
    else:
        if not args.date:
            parser.error("--date is required unless using --analyze")
        
        results = analyze_edgar_filings(args.date)
        
        # Sort results
        if args.sort_by_7d:
            # Sort by 7D price change (descending), handling None values
            def get_7d_change(result):
                pm = result.get('price_momentum') or {}
                change_7d = pm.get('price_change_7d')
                return change_7d if change_7d is not None else -999  # Put None values at end
            results = sorted(results, key=get_7d_change, reverse=True)
            sort_label = "7-Day Price Change"
        else:
            # Default: sort by score (already sorted, but ensure descending)
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            sort_label = "Score"
        
        # Format and print results
        threshold = 0.55
        above_threshold = [r for r in results if r['score'] >= threshold]
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total filings analyzed: {len(results)}")
        print(f"Above threshold (≥{threshold}): {len(above_threshold)}")
        print(f"Sorted by: {sort_label}")
        
        if results:
            limit = args.limit or 10
            top_results = results[:limit]
            
            print(f"\nTop {len(top_results)} by {sort_label}:")
            print("=" * 120)
            
            # Column headers (Ref column width is flexible)
            header = f"{'#':<3} {'Score':<6} {'Company':<35} {'Form':<6} {'Delta':<8} {'Start $':<8} {'1d %':<7} {'7d %':<7} {'30d %':<8} {'60d %':<8} {'Now %':<8} {'Ref':<20} {'✅'}"
            print(header)
            print("-" * 130)
            
            for i, result in enumerate(top_results, 1):
                ticker_str = f" ({result['ticker']})" if result['ticker'] else ""
                company_name = f"{result['company']}{ticker_str}"
                if len(company_name) > 34:
                    company_name = company_name[:31] + "..."
                
                score_rounded = round(result['score'], 2)
                threshold_marker = "✅" if score_rounded >= threshold else " "
                delta_score = result['delta_score']
                delta_label = result.get('delta_label', 'N/A')
                delta_str = f"{delta_score:.2f}({delta_label[:1]})"
                
                # Get price data
                pm = result.get('price_momentum') or {}
                valuation = result.get('valuation') or {}
                current_price = valuation.get('current_price') if valuation else None
                filing_price = pm.get('filing_price') if pm else None
                
                # Format filing price
                start_price_str = f"${filing_price:.2f}" if filing_price and filing_price > 0 else "N/A"
                
                # Format price changes
                price_1d = f"{pm['price_change_1d']:+.1f}%" if pm.get('price_change_1d') is not None else "N/A"
                price_7d = f"{pm['price_change_7d']:+.1f}%" if pm.get('price_change_7d') is not None else "N/A"
                price_30d = f"{pm['price_change_30d']:+.1f}%" if pm.get('price_change_30d') is not None else "N/A"
                price_60d = f"{pm['price_change_60d']:+.1f}%" if pm.get('price_change_60d') is not None else "N/A"
                
                # Calculate current price change
                price_now = None
                if current_price and filing_price and filing_price > 0:
                    price_now = ((current_price - filing_price) / filing_price) * 100
                price_now_str = f"{price_now:+.1f}%" if price_now is not None else "N/A"
                
                filing_ref = result.get('filing_ref', 'N/A')
                form_str = result['form']
                
                # Print main row - use first 20 chars of ref for alignment, but we'll print full ref
                ref_display = filing_ref[:20] if len(filing_ref) > 20 else filing_ref
                row = f"{i:<3} {score_rounded:<6.2f} {company_name:<35} {form_str:<6} {delta_str:<8} {start_price_str:<8} {price_1d:<7} {price_7d:<7} {price_30d:<8} {price_60d:<8} {price_now_str:<8} {ref_display:<20} {threshold_marker}"
                print(row)
                
                # If reference is longer than 20 chars, print full reference on continuation line (indented to align with Ref column)
                if len(filing_ref) > 20:
                    # Calculate position where Ref column starts (sum of previous column widths)
                    ref_col_start = 3 + 6 + 35 + 6 + 8 + 8 + 7 + 7 + 8 + 8 + 8 + 1  # All columns + 1 space (updated for 1d column)
                    print(f"{'':>{ref_col_start}} {filing_ref}")