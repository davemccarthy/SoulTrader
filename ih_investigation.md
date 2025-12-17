# IH Investigation: Why Did It Decline -14.7%?

## Executive Summary

**IH was discovered on Dec 2, 2025 at $2.65-$2.70, now trading at $2.26 (-14.7%).**
The investigation reveals **critical gaps in our filters** that allowed a declining company to pass.

## Key Findings

### ✅ What Passed Our Filters (Correctly)

1. **Profit Margin: 12.46%** ✓
   - Profitable company, well above 0% threshold

2. **Market Cap: $116M** ✓
   - Above $25M minimum

3. **52 Week Change: 18.23%** ✓
   - Positive yearly performance, well above -50% threshold

4. **Notional Price: $6.09** ✓
   - Using P/E method with sector average (21.0)
   - Company P/E is 7.79 (0.37x sector average = very cheap)

5. **Recent Trend: -10.32%** ✓
   - Negative trend is fine (filter only excludes >5% gains)

### ❌ What Our Filters MISSED (Critical Red Flags)

1. **Revenue Growth: -6.9%** ⚠️ **CRITICAL**
   - Company revenue is **declining**
   - No filter checks for revenue growth
   - This is a major red flag for value traps

2. **Recent Price Decline: -25.9% (3 months), -19.86% (30 days)** ⚠️
   - Stock has been in a **sustained downtrend**
   - Our 5-day trend filter only checks short-term
   - No filter for longer-term price momentum

3. **Forward EPS Lower Than Trailing: 0.19 vs 0.29** ⚠️
   - Earnings are expected to **decline**
   - Forward P/E (11.89) > Trailing P/E (7.79)
   - Suggests earnings contraction

4. **52 Week High: $3.60, Current: $2.26** ⚠️
   - Stock is down **37% from 52-week high**
   - Trading near 52-week low ($1.55)
   - Suggests fundamental deterioration

## Root Cause Analysis

### Why Did IH Pass?

1. **Notional Price Calculation Issue:**
   - Uses **sector average P/E (21.0)** for Consumer Defensive
   - Company's actual P/E is **7.79** (very low)
   - Low P/E suggests market knows something is wrong
   - **The notional price ($6.09) may be too optimistic**

2. **Missing Revenue Growth Filter:**
   - We check profit margin (current profitability)
   - We DON'T check revenue growth (future sustainability)
   - **Declining revenue is a major red flag**

3. **Missing Longer-Term Trend Filter:**
   - 5-day trend filter only catches recent spikes
   - **No filter for sustained declines (30-day, 90-day)**

4. **No Forward Earnings Check:**
   - We use trailing EPS for notional calculation
   - **Forward EPS is lower, suggesting earnings decline**

## Recommended Filter Improvements

### 1. **Add Revenue Growth Filter** (HIGH PRIORITY)
```python
MIN_REVENUE_GROWTH = -5.0  # Reject companies with >5% revenue decline
```
- **Rationale:** Declining revenue often precedes earnings problems
- **IH Impact:** Would have been rejected (-6.9% revenue growth)

### 2. **Add Longer-Term Trend Filter** (HIGH PRIORITY)
```python
MAX_30_DAY_DECLINE = -15.0  # Reject stocks down >15% in last 30 days
MAX_90_DAY_DECLINE = -20.0  # Reject stocks down >20% in last 90 days
```
- **Rationale:** Sustained declines suggest fundamental issues
- **IH Impact:** Would have been rejected (-19.86% in 30 days, -25.9% in 90 days)

### 3. **Add Forward Earnings Quality Check** (MEDIUM PRIORITY)
```python
# Reject if forward EPS is significantly lower than trailing
if forward_eps and trailing_eps:
    eps_decline = (forward_eps - trailing_eps) / trailing_eps
    if eps_decline < -0.20:  # >20% earnings decline expected
        reject
```
- **Rationale:** Declining earnings expectations are a red flag
- **IH Impact:** Would have been flagged (34% earnings decline expected)

### 4. **Improve Notional Price Calculation** (MEDIUM PRIORITY)
- **Current:** Uses sector average P/E (may be too optimistic for troubled companies)
- **Suggestion:** Use a blend of company P/E and sector P/E, or cap notional price
- **Alternative:** If company P/E is <50% of sector average, investigate further

### 5. **Add 52-Week High/Low Context** (LOW PRIORITY)
```python
# Reject if stock is down >30% from 52-week high
if current_price < (fifty_two_week_high * 0.70):
    # Additional scrutiny or rejection
```
- **Rationale:** Stocks near 52-week lows may have fundamental issues
- **IH Impact:** Would have been flagged (down 37% from high)

## Implementation Priority

### Phase 1 (Immediate - High Impact)
1. ✅ **Revenue Growth Filter** - Would have caught IH
2. ✅ **30-Day Trend Filter** - Would have caught IH

### Phase 2 (Short-term - Medium Impact)
3. Forward Earnings Quality Check
4. Notional Price Calculation Refinement

### Phase 3 (Long-term - Low Impact)
5. 52-Week High/Low Context

## Expected Impact

### IH Case Study
- **Current Filters:** Passed all 6 filters ✓
- **With New Filters:** Would have been rejected ❌
  - Revenue growth filter: -6.9% < -5% threshold ❌
  - 30-day trend filter: -19.86% < -15% threshold ❌

### Trade-offs
- **Pros:** 
  - Catches declining companies earlier
  - Reduces value traps
  - Improves overall portfolio quality
  
- **Cons:**
  - May reject some legitimate value opportunities
  - More conservative (fewer discoveries)
  - Need to tune thresholds carefully

## Next Steps

1. **Implement Phase 1 filters** (Revenue Growth + 30-Day Trend)
2. **Test on historical discoveries** - How many would have been rejected?
3. **Backtest performance** - Would filtered portfolio perform better?
4. **Tune thresholds** - Balance between catching problems and not being too strict
5. **Monitor IH** - See if it recovers or continues declining (validates filters)

## Conclusion

**IH passed our filters but had clear red flags:**
- Declining revenue (-6.9%)
- Sustained price decline (-19.86% in 30 days)
- Declining earnings expectations

**The missing filters would have caught this:**
- Revenue growth filter: ❌
- 30-day trend filter: ❌

**Recommendation:** Implement Phase 1 filters immediately to prevent similar issues.









