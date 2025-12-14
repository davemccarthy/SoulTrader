# Vunder Advisor - The Search for Undervalued Stocks

## Overview

**Vunder** is a new advisor based on the refined Yahoo advisor methodology, focused on discovering undervalued stocks through comprehensive fundamental and technical filtering.

## Key Features

### 1. Notional Price Calculation
Uses multiple valuation methods to determine intrinsic value:
- **DCF** (Discounted Cash Flow)
- **P/E** (Price-to-Earnings with sector averages)
- **EV/EBITDA** (Enterprise Value to EBITDA)
- **Revenue Multiple** (Price-to-Sales)
- **P/B** (Price-to-Book)

Selects the best method based on data availability and reasonableness.

### 2. Comprehensive Filtering Pipeline

#### Step 1: Fundamental Filters
- ✅ **Profit Margin** >= 0% (profitable or break-even)
- ✅ **Yearly Change** >= -50% (not catastrophic losses)
- ✅ **Market Cap** >= $25M (minimum liquidity)
- ✅ **Revenue Growth** >= -5% (reject declining revenue) - **Phase 1**

#### Step 2: Trend Filters
- ✅ **5-Day Trend** <= 5% (avoid stocks that already ran up)
- ✅ **30-Day Trend** >= -15% (reject sustained declines) - **Phase 1**

#### Step 3: Average Low Filter
- ✅ **50-Day SMA** - Price <= 110% of 50-day moving average (avoid elevated prices)

#### Step 4: Undervalued Selection
- ✅ **Discount Ratio** <= 0.70 (actual/notional price ratio)

## Configuration

### Current Settings
```python
UNDERVALUED_RATIO_THRESHOLD = 0.70  # Discount threshold
MIN_PROFIT_MARGIN = 0.0
MAX_YEARLY_LOSS = -50.0
MIN_MARKET_CAP = 25_000_000
MAX_RECENT_TREND_PCT = 5.0
MIN_REVENUE_GROWTH = -5.0  # Phase 1
MAX_30_DAY_DECLINE = -15.0  # Phase 1
MAX_PRICE_VS_SMA50 = 1.10  # Average low filter
```

### Volume Strategy
Currently uses **high volume** strategy (most active stocks). Can be configured for low volume strategy (less discovered stocks) by modifying `_get_active_stocks()` call in `discover()` method.

## Performance Expectations

Based on test script results with current filters:
- **Input:** ~100 stocks
- **After Fundamentals:** ~22 stocks
- **After 5-Day Trend:** ~14 stocks
- **After 30-Day Trend:** ~11 stocks
- **After Average Low:** ~10 stocks
- **Final Discovered:** ~2-3 stocks

## Advantages Over Original Yahoo Advisor

1. **Revenue Growth Filter** - Catches declining companies (e.g., IH case)
2. **30-Day Trend Filter** - Catches sustained declines
3. **Average Low Filter** - Avoids stocks trading too far above average
4. **Refined Threshold** - 0.70 vs 0.66 (captures more opportunities like WDH)
5. **Clean Implementation** - Based on tested methodology

## Usage

The advisor is automatically registered and available in the system. To use:

1. Ensure advisor is enabled in Django admin
2. Run discovery via `smartanalyse` command
3. Advisor will discover undervalued stocks matching all criteria

## Testing

Use the test script to validate:
```bash
python test_yahoo.py --limit 100 --low-volume
```

This mimics the Vunder advisor's discovery logic.

## Future Enhancements

- Make volume strategy configurable (high vs low)
- Add configuration options for filter thresholds
- Performance tracking and backtesting
- Additional quality filters based on results

## Files

- **Implementation:** `core/services/advisors/vunder.py`
- **Registration:** Auto-registered via `register()` function
- **Test Script:** `test_yahoo.py` (mimics Vunder logic)




