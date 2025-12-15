# Yahoo Advisor Strategy Comparison

## Test Script Results (Low Volume Strategy)
**Date:** Current run
**Stocks Found:** 6 undervalued stocks

| Symbol | Current Price | Notional Price | Discount | Upside | Method |
|--------|--------------|----------------|----------|--------|--------|
| LITB   | $1.84        | $4.56          | 59.6%    | 147.8% | P/E    |
| CIG-C  | $2.53        | $6.72          | 62.4%    | 165.6% | P/E    |
| IH     | $2.26        | $6.09          | 62.9%    | 169.5% | P/E    |
| GHG    | $1.72        | $2.94          | 41.5%    | 71.0%  | EV/EBITDA |
| VIOT   | $2.23        | $7.41          | 69.9%    | 232.3% | P/E    |
| DDL    | $2.04        | $3.82          | 46.7%    | 87.5%  | EV/EBITDA |

## Actual Holdings (Yahoo Advisor)
**Total Holdings:** 47
**Aggregate Return:** +5.9%
**Average Return:** +5.0%
**Total Cost:** $134,873.36
**Current Value:** $142,813.04

### Top Performing Stocks:
- **ODV**: +27.2% (best performer)
- **ASA**: +11.4% (multiple users)
- **KLXE**: +16.9%
- **WDH**: +4.7% (multiple users)
- **CIG**: +1.0% (multiple users)

### Underperforming Stocks:
- **IH**: -14.7% (multiple users) ⚠️
- **CYH**: -3.6% (multiple users)
- **BBDO**: -0.3% (multiple users)

## Key Findings

### 1. **Direct Match: IH Stock** ⭐ CRITICAL FINDING
- **Test Script (Today):** Found IH at $2.26 with 62.9% discount, 169.5% upside potential (Notional: $6.09)
- **Actual Holdings:** IH purchased at $2.65-$2.70 on 2025-12-02, currently at $2.26 (-14.7% return)
- **Discovery History:** 
  - Discovered on 2025-12-02 at $2.65-$2.70
  - Notional price was $6.09 (same as today!)
  - Discount was 55.7-56.0% (similar to today's 62.9%)
- **Insight:** This is a **smoking gun** finding:
  - The advisor found IH 2+ weeks ago at $2.65-$2.70
  - Stock has declined to $2.26 (-14.7%)
  - **Test script finds it AGAIN at $2.26** with same notional price ($6.09)
  - This suggests:
    * The notional price calculation is **consistent** (found same value twice)
    * The stock is **still undervalued** according to the model
    * Either: (a) Market hasn't recognized value yet, OR (b) Notional price is too optimistic
    * The decline may be temporary or the stock needs more time

### 2. **Related Match: CIG vs CIG-C**
- **Test Script:** Found CIG-C (preferred shares) at $2.53
- **Actual Holdings:** CIG (common shares) at $2.05, purchased at $2.03 (+1.0%)
- **Insight:** Different share classes, but same company. CIG common shares are performing slightly better than break-even.

### 3. **WDH - Passed Filters But Not Final**
- **Test Script:** WDH passed all filters but didn't make final 6 (discount ratio was 31.2%, above 0.66 threshold)
- **Actual Holdings:** WDH at $1.79, purchased at $1.71 (+4.7% return)
- **Insight:** WDH is performing well despite not being in the "most undervalued" category. This suggests the 0.66 threshold might be too strict.

### 4. **Performance Analysis**
- **Average Return:** +5.0% is solid but not exceptional
- **Best Performers:** ODV (+27.2%), ASA (+11.4%), KLXE (+16.9%)
- **Worst Performers:** IH (-14.7%), CYH (-3.6%)
- **Win Rate:** Most stocks are positive or near break-even

## Strategic Insights

### Low Volume Strategy Validation
1. **IH Match:** The test script found IH using low-volume strategy, and it's currently in holdings. However, it's down -14.7%, suggesting:
   - Low-volume stocks may take longer to realize value
   - OR the notional price calculation needs refinement
   - OR timing matters (when was IH originally discovered?)

2. **Diversification:** Actual holdings show 8 different stocks (IH, ASA, BBDO, CYH, ODV, WDH, CIG, KLXE), suggesting the advisor has been finding variety.

3. **Consistency:** The test script's methodology (notional price, fundamentals, trend filter) appears to be working, as evidenced by:
   - +5.9% aggregate return
   - Most stocks in positive territory
   - Some strong winners (ODV, ASA, KLXE)

### Recommendations

1. **IH - Double Discovery Signal:** ⚠️ HIGH PRIORITY
   - **Fact:** Advisor found IH at $2.65-$2.70 on Dec 2, now at $2.26 (-14.7%)
   - **Fact:** Test script finds it AGAIN at $2.26 with same notional ($6.09)
   - **Decision Point:**
     * If notional price is accurate → This is a **buying opportunity** (stock even cheaper now)
     * If notional price is wrong → Need to investigate why P/E method gives $6.09
   - **Action:** 
     * Check IH fundamentals: Has profit margin, revenue, or EPS changed?
     * Compare IH's P/E to sector average (advisor uses sector P/E for notional)
     * Consider: Is this a value trap or true undervaluation?

2. **Adjust Threshold:** WDH performed well (+4.7%) but didn't make final 6 due to 31.2% discount (threshold is 0.66 = 66%). Consider:
   - Relaxing threshold slightly (e.g., 0.70) to capture more opportunities
   - OR keeping strict threshold but acknowledging some good stocks will be missed

3. **Volume Strategy:** The low-volume approach found IH, which is already in holdings. This suggests:
   - Low-volume strategy may find similar stocks to high-volume
   - OR it may find different opportunities
   - Need more data to validate if low-volume performs better

4. **Notional Price Accuracy:** 
   - IH: Notional $6.09, Actual $2.26 (2.7x difference)
   - This large gap suggests either:
     * The P/E method is too optimistic
     * The stock is truly undervalued and needs time
     * Market conditions have deteriorated

## Strategic Insights Summary

### The IH Case Study
**Timeline:**
- Dec 2, 2025: Yahoo advisor discovers IH at $2.65-$2.70 (notional $6.09, 56% discount)
- Dec 2-16, 2025: Stock declines to $2.26 (-14.7% from purchase)
- Today: Test script finds IH again at $2.26 (same notional $6.09, 63% discount)

**Interpretation:**
1. **Consistency:** Notional price calculation is stable ($6.09 both times)
2. **Validation:** Test script validates the original discovery logic
3. **Opportunity:** Stock is now cheaper with same notional value (bigger discount)
4. **Risk:** Stock has declined, suggesting either:
   - Market doesn't agree with $6.09 valuation
   - Temporary market conditions
   - Fundamental issues not captured by filters

### Low Volume Strategy Performance
- **Test found 6 stocks:** LITB, CIG-C, IH, GHG, VIOT, DDL
- **IH overlap:** Confirms low-volume strategy finds similar opportunities
- **WDH note:** Passed filters but didn't make final 6, yet performing +4.7%
- **Conclusion:** Low-volume strategy appears to work, but may need threshold adjustment

### Overall Advisor Performance
- **+5.9% aggregate return** is solid
- **+5.0% average return** per stock
- **Winners:** ODV (+27.2%), ASA (+11.4%), KLXE (+16.9%)
- **Losers:** IH (-14.7%), CYH (-3.6%)
- **Win rate:** ~70% positive or break-even

## Next Steps

1. ✅ **IH Analysis Complete:** Discovered Dec 2 at $2.65, found again today at $2.26
2. **Investigate IH Fundamentals:** Why has it declined? Check recent earnings, news, sector trends
3. **Test Threshold Adjustment:** Relax discount threshold (0.66 → 0.70) to capture WDH-like opportunities
4. **Monitor New Discoveries:** Track the 6 new stocks (LITB, CIG-C, GHG, VIOT, DDL) vs historical performance
5. **Volume Strategy Validation:** Compare low-volume vs high-volume discovery performance over time

