# Oscilla Trading Parameters - Snapshot
**Date:** 2025-01-XX  
**Purpose:** Document current parameters before multi-period testing

## Entry Filters
- `MIN_PRICE = 8` - Minimum stock price
- `MAX_PRICE = 80` - Maximum stock price
- `MIN_VOLUME = 1,000,000` - Minimum daily volume
- `MIN_AVG_VOLUME = 2,000,000` - Minimum average volume over lookback
- `REL_VOLUME_MIN = 0.8` - Minimum relative volume (vs average)
- `REL_VOLUME_MAX = 1.3` - Maximum relative volume (vs average)
- `LOOKBACK_DAYS = 40` - Days to look back for analysis
- `MIN_RR = 1.8` - Minimum reward:risk ratio

## Risk Management
- `MIN_STOP_BUFFER_PCT = 0.10` - Stop distance from entry (10% = stop at 90% of entry)
  - **NOTE:** Results shown with 0.50 (50% stop buffer, stops at 50% of entry) - verify which is active
- `MAX_TARGET_PCT = 0.10` - Maximum target as % gain (10% when diminishing disabled)

## Exit Strategy Configuration
- `TARGET_DIMINISHING_ENABLED = True` - Enable diminishing target over time
- `TARGET_DIMINISHING_MULTIPLIER = 0.75` - Cap target at 75% of calculated target
- `STOP_AUGMENTING_ENABLED = True` - Enable trailing stop over time
- `MAX_DAYS = 40` - Maximum days to hold trade

## STOP_SLIDE (Structural Exit)
- `STOP_SLIDE_ENABLED = True` - Enable STOP_SLIDE exit
- `STOP_SLIDE_CONFIRM_BARS = 1` - Number of bars to confirm signal
- **Mode:** Profit-only (only triggers when `close > buy_price`)

## Disabled Features
- `WAVE_EXIT_ENABLED = False` - Wave exhaustion exits (Plan C, deprecated)
- `DESC_TREND_ENABLED = False` - Descending trend exits (deprecated)

## Recent Test Results Reference
**Date Tested:** 2025-04-01  
**Stop Buffer Used:** 0.50 (50% - stops at 50% of entry price)

Results:
- Total Trades: 27
- Win Rate: 18/27 (66.7%)
- Average Return: -1.50%
- Average Win: +3.76%
- Average Loss: -19.17% ⚠️ (very large due to 50% stop buffer)
- Average Days Held: 27.2 days
- Expected Value: -1.61% per trade

**Key Issue:** Wide stops (50%) reduce win rate but dramatically increase loss size when stops are hit (-25%, -41.25% losses observed)

## Multi-Period Testing

To test parameter robustness across multiple periods and avoid overfitting:

```bash
python3 test_multiple_periods.py
```

This will test the strategy across multiple distinct time periods and provide:
- Results for each period
- Aggregate statistics across all periods
- Consistency metrics (std dev, ranges)
- Win rate and return distributions

The script tests monthly periods from Oct 2024 through Apr 2025 by default. Adjust the `test_periods` list in the script to test different date ranges.

