# Detailed Analysis: Stocks Filtered Out by Phase 1 Filters

## Summary
**Before:** 6 stocks discovered  
**After:** 2 stocks discovered  
**Filtered Out:** 4 stocks

## Stocks That Survived ✓

1. **CIG-C** - $2.53 → $6.72 (62.4% discount)
   - Revenue Growth: Passed
   - 30-Day Trend: -6.30% (above -15% threshold) ✓
   - **Status:** Still discovered

2. **DDL** - $2.04 → $3.82 (46.7% discount)
   - Revenue Growth: Passed
   - 30-Day Trend: +10.87% (positive!) ✓
   - **Status:** Still discovered

## Stocks Filtered Out ❌

### 1. **LITB** - Filtered by 30-Day Trend
- **Original:** $1.84 → $4.56 (59.6% discount, 147.8% upside)
- **Revenue Growth:** -2.70% (passed - above -5% threshold)
- **30-Day Trend:** -50.54% ❌ (below -15% threshold)
- **Reason:** Severe 30-day decline (-50.54%)
- **Assessment:** Correctly filtered - massive decline suggests problems

### 2. **IH** - Filtered by BOTH Filters ⚠️ **VALIDATION**
- **Original:** $2.26 → $6.09 (62.9% discount, 169.5% upside)
- **Revenue Growth:** -6.90% ❌ (below -5% threshold)
- **30-Day Trend:** -19.86% ❌ (below -15% threshold)
- **Reason:** Declining revenue AND sustained price decline
- **Assessment:** **CORRECTLY FILTERED** - This is the stock that declined -14.7% after discovery. The filters worked perfectly!

### 3. **GHG** - Filtered by Revenue Growth
- **Original:** $1.72 → $2.94 (41.5% discount, 71.0% upside)
- **Revenue Growth:** -11.30% ❌ (below -5% threshold)
- **30-Day Trend:** -14.85% (just above -15% threshold, but close)
- **Reason:** Significant revenue decline (-11.30%)
- **Assessment:** Correctly filtered - declining revenue is a red flag

### 4. **VIOT** - Filtered by 30-Day Trend
- **Original:** $2.23 → $7.41 (69.9% discount, 232.3% upside)
- **Revenue Growth:** +42.10% ✓ (excellent growth!)
- **30-Day Trend:** -28.75% ❌ (below -15% threshold)
- **Reason:** Despite strong revenue growth, severe 30-day price decline
- **Assessment:** Correctly filtered - even with good fundamentals, -28.75% decline suggests market concerns

## Key Insights

### 1. **IH Validation** ✅
- IH was correctly identified and filtered out
- Had both red flags: declining revenue (-6.9%) and sustained decline (-19.86%)
- This proves the filters work as intended

### 2. **Filter Effectiveness**
- **Revenue Growth Filter:** Caught IH and GHG (declining revenue)
- **30-Day Trend Filter:** Caught LITB, IH, and VIOT (sustained declines)
- **Combined:** Both filters caught IH (the problem stock)

### 3. **Trade-offs**
- **More Conservative:** Only 2 stocks vs 6 stocks
- **Higher Quality:** All filtered stocks had legitimate concerns
- **VIOT Case:** Even with +42% revenue growth, -28.75% decline was correctly flagged

### 4. **Potential Adjustments**
- **VIOT** had excellent revenue growth (+42%) but severe decline (-28.75%)
  - Question: Should we allow exceptions for strong revenue growth?
  - Answer: Probably not - market knows something we don't
- **GHG** was close on 30-day trend (-14.85%, threshold is -15%)
  - This is working as intended (just above threshold)

## Conclusion

**The Phase 1 filters are working correctly:**
- ✅ Caught IH (the known problem stock)
- ✅ Filtered out stocks with legitimate concerns
- ✅ More conservative approach (2 vs 6 stocks)
- ✅ Higher quality discoveries

**Recommendation:** Keep the filters as-is. The reduction from 6 to 2 stocks is acceptable given:
1. IH (the problem stock) was correctly filtered
2. All filtered stocks had real concerns
3. The 2 remaining stocks (CIG-C, DDL) are solid candidates





