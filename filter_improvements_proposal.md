# Filter Improvements Proposal Based on IH Investigation

## Investigation Summary

**IH declined -14.7% after discovery** despite passing all 6 filters. Investigation revealed:

### Red Flags That Were Missed:
1. ❌ **Revenue declining -6.9%** (no revenue growth filter)
2. ❌ **30-day price decline -19.86%** (only 5-day trend checked)
3. ❌ **90-day price decline -25.9%** (no longer-term trend filter)
4. ⚠️ **Forward EPS lower than trailing** (34% earnings decline expected)

### What Passed (Correctly):
- ✅ Profit margin: 12.46%
- ✅ Market cap: $116M
- ✅ 52-week change: +18.23%
- ✅ Notional price: $6.09 (P/E method)
- ✅ 5-day trend: -10.32% (negative is fine)

## Proposed Filter Improvements

### 1. Revenue Growth Filter (HIGH PRIORITY)

**Rationale:** Declining revenue often precedes earnings problems and value traps.

**Implementation:**
```python
MIN_REVENUE_GROWTH = -5.0  # Reject if revenue decline >5%
```

**IH Impact:** Would have been rejected (-6.9% < -5%)

**Code Location:** Add to `_filter_fundamentals()` method

### 2. 30-Day Trend Filter (HIGH PRIORITY)

**Rationale:** Sustained declines suggest fundamental deterioration, not just short-term volatility.

**Implementation:**
```python
MAX_30_DAY_DECLINE = -15.0  # Reject if down >15% in last 30 days
```

**IH Impact:** Would have been rejected (-19.86% < -15%)

**Code Location:** Add new method `_filter_longer_term_trend()` called after `_filter_recent_trend()`

### 3. Forward Earnings Quality Check (MEDIUM PRIORITY)

**Rationale:** Declining earnings expectations are a red flag.

**Implementation:**
```python
MAX_EPS_DECLINE = -0.20  # Reject if forward EPS decline >20% vs trailing
```

**IH Impact:** Would have been flagged (34% decline expected)

**Code Location:** Add to `_filter_fundamentals()` method

### 4. Lower Threshold to 0.70 (As Requested)

**Rationale:** WDH performed +4.7% but didn't make final cut (31.2% discount vs 66% threshold).

**Implementation:**
```python
UNDERVALUED_RATIO_THRESHOLD = 0.70  # Changed from 0.66
```

**Impact:** Would capture more opportunities like WDH

## Implementation Plan

### Phase 1: Critical Filters (Immediate)
1. Revenue Growth Filter
2. 30-Day Trend Filter
3. Lower threshold to 0.70

### Phase 2: Additional Quality Checks (Short-term)
4. Forward Earnings Quality Check

## Expected Outcomes

### Positive:
- ✅ Catch declining companies earlier (like IH)
- ✅ Reduce value traps
- ✅ Improve overall portfolio quality
- ✅ Capture more opportunities (threshold 0.70)

### Trade-offs:
- ⚠️ More conservative (fewer discoveries)
- ⚠️ May reject some legitimate value opportunities
- ⚠️ Need to tune thresholds based on backtesting

## Code Changes Required

### 1. Add Constants
```python
# Add to constants section
MIN_REVENUE_GROWTH = -5.0
MAX_30_DAY_DECLINE = -15.0
MAX_EPS_DECLINE = -0.20
UNDERVALUED_RATIO_THRESHOLD = 0.70  # Changed from 0.66
```

### 2. Update `_filter_fundamentals()` Method
- Add revenue growth check
- Add forward EPS quality check

### 3. Add `_filter_longer_term_trend()` Method
- Calculate 30-day trend
- Filter out stocks with sustained declines

### 4. Update `discover()` Method
- Call new `_filter_longer_term_trend()` after `_filter_recent_trend()`
- Extract revenue growth and forward EPS in data collection phase

## Testing Plan

1. **Backtest on Historical Discoveries:**
   - How many would have been rejected?
   - What was the performance of rejected vs accepted?

2. **Test on Current Holdings:**
   - Would IH have been rejected? (Yes)
   - Would other holdings still pass?

3. **Monitor New Discoveries:**
   - Compare performance of filtered vs unfiltered
   - Adjust thresholds based on results

## Risk Assessment

### Low Risk:
- Revenue growth filter (clear metric)
- 30-day trend filter (objective price data)

### Medium Risk:
- Forward EPS check (may reject some legitimate value plays)
- Threshold change to 0.70 (may capture lower-quality opportunities)

### Mitigation:
- Start with conservative thresholds
- Monitor and adjust based on results
- Keep detailed logs of why stocks were rejected





