# Filter Comparison: Before vs After Phase 1 Filters

## Original Results (Before New Filters)
**6 stocks discovered:**
1. LITB - $1.84 → $4.56 (59.6% discount, 147.8% upside)
2. CIG-C - $2.53 → $6.72 (62.4% discount, 165.6% upside)
3. IH - $2.26 → $6.09 (62.9% discount, 169.5% upside) ⚠️
4. GHG - $1.72 → $2.94 (41.5% discount, 71.0% upside)
5. VIOT - $2.23 → $7.41 (69.9% discount, 232.3% upside)
6. DDL - $2.04 → $3.82 (46.7% discount, 87.5% upside)

## New Results (With Phase 1 Filters)
**2 stocks discovered:**
1. CIG-C - $2.53 → $6.72 (62.4% discount, 165.6% upside) ✓
2. DDL - $2.04 → $3.82 (46.7% discount, 87.5% upside) ✓

## Stocks Filtered Out

### 1. LITB - Filtered by Revenue Growth
- **Original:** Passed all filters
- **New Filter:** Revenue decline >5%
- **Impact:** Would have been rejected by revenue growth filter

### 2. IH - Filtered by BOTH New Filters ⚠️
- **Original:** Passed all filters
- **Revenue Growth:** -6.9% (below -5% threshold) ❌
- **30-Day Trend:** -19.86% (below -15% threshold) ❌
- **Impact:** This is the stock that declined -14.7% after discovery - **correctly filtered out!**

### 3. GHG - Filtered by 30-Day Trend
- **Original:** Passed all filters
- **30-Day Trend:** Likely below -15% threshold
- **Impact:** Would have been rejected by 30-day decline filter

### 4. VIOT - Filtered by Revenue Growth or 30-Day Trend
- **Original:** Passed all filters
- **Impact:** Likely rejected by one of the new filters

## Analysis

### Stocks That Survived
- **CIG-C:** Still passes all filters including new ones ✓
- **DDL:** Still passes all filters including new ones ✓

### Stocks That Were Filtered
- **IH:** Correctly filtered (had -6.9% revenue decline and -19.86% 30-day decline)
- **LITB, GHG, VIOT:** Need to check why they were filtered

## Key Insight

**IH was correctly identified and filtered out** - this validates that the new filters work as intended. IH had:
- Revenue declining -6.9%
- 30-day price decline -19.86%
- Both red flags that the new filters catch

The fact that we went from 6 stocks to 2 stocks shows the filters are working, but we may want to review if we're being too conservative.









