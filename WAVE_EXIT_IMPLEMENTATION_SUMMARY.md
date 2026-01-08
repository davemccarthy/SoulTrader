# Wave Exit Implementation Summary for ChatGPT

## Current Code State

The `test_oscilla.py` script implements a wave-based trading strategy using wavelet analysis. The code has been updated to return complete wave state information from `wavelet_trade_engine()`.

## Wave State Now Returned (After Enhancement)

The `wavelet_trade_engine()` function now returns the following wave state fields in addition to trade signals:

```python
{
    "accepted": True,
    "dominant_period_days": int,        # Dominant cycle period (e.g., 42 days)
    "half_period": int,                  # Smoothing window = max(3, dominant_period // 2)
    "consistency": float,                # Wave pattern consistency (0.0-1.0)
    "wave_position": float,              # Entry wave position (0=trough, 1=peak)
    "buy": float,                        # Entry price
    "stop": float,                       # Stop loss price
    "target": float,                     # Target price (currently aspirational)
    "reward_risk": float,                # Reward:Risk ratio
    # NEW: Wave state for exit detection
    "avg_trough": float,                 # Average of smoothed troughs (wave baseline)
    "avg_peak": float,                   # Average of smoothed peaks (wave ceiling)
    "wave_range": float,                 # avg_peak - avg_trough (wave amplitude)
    "log": list                          # Debug log messages
}
```

## Wave State Calculation (Current Implementation)

The wave state is calculated in `wavelet_trade_engine()` as follows:

1. **Wavelet Analysis** (lines 275-293):
   - Detrends price series using 20-day rolling mean
   - Performs continuous wavelet transform (CWT) using Morlet wavelet
   - Finds dominant scale/period using maximum average power
   - Calculates consistency metric

2. **Peak/Trough Detection** (lines 300-312):
   - Smooths price using `half_period` window (centered rolling mean)
   - Uses `scipy.signal.find_peaks` to detect peaks and troughs in smoothed series
   - Calculates `avg_peak` and `avg_trough` as means of detected peaks/troughs
   - `wave_range = avg_peak - avg_trough`

3. **Wave Position** (line 322):
   - `wave_position = (current_price - avg_trough) / wave_range`
   - Value of 0 = at trough, 1 = at peak
   - Entry typically occurs at wave_position ≈ 0.25-0.35

## Current Holding State Structure (For Backtesting)

In the current backtesting implementation, holdings are simulated but not explicitly persisted. When a trade is accepted, the candidate DataFrame contains all wave state. Here's the structure:

**At Entry** (from `generate_trading_candidates()` result):
```python
candidate_row = {
    'ticker': 'AAPL',
    'price': 150.25,
    'buy': 150.25,                      # Entry price
    'stop': 135.00,                     # Stop loss
    'target': 165.00,                   # Target price
    'dominant_period_days': 42,         # ✅ Already returned
    'half_period': 21,                  # ✅ Now returned
    'consistency': 0.452,
    'wave_position': 0.318,             # Entry wave position
    'reward_risk': 2.50,
    # NEW wave state fields:
    'avg_trough': 140.00,               # ✅ Now returned
    'avg_peak': 160.00,                 # ✅ Now returned
    'wave_range': 20.00,                # ✅ Now returned
}
```

**Minimal Wave State for Exit Detection** (what needs to be stored/passed):
```python
entry_wave_state = {
    'avg_trough': float,        # Required for wave_position calculation
    'avg_peak': float,          # Required for wave_position calculation
    'wave_range': float,        # Required for wave_position calculation
    'dominant_period_days': int,  # Required for time expiry
    'half_period': int,          # Required for smoothed rollover detection
}
```

## Current Exit Logic (What We're Replacing)

Currently, `backtest_signal()` uses fixed exit criteria:
1. **Stop Loss**: Price hits stop price (with optional augmenting stop)
2. **Target Price**: Price hits target price (with optional diminishing target)
3. **Timeout**: Maximum days held (currently 40 days)

The problem: Target prices are often aspirational and not reached, leading to:
- Trades that hit stop-loss after stalling near peak
- Missed profit opportunities when wave exhausts before target

## What We Need: `detect_wave_peak()` Function

Based on your guidance, we need a function that detects wave exhaustion using **structural signals** (not predictions). The function should:

### Input Parameters:
```python
def detect_wave_exhaustion(
    current_price: float,
    entry_wave_state: dict,  # {avg_trough, avg_peak, wave_range, dominant_period_days, half_period}
    price_series_since_entry: pd.Series,  # Prices since entry (for smoothed rollover)
    entry_date: datetime/date,
    current_date: datetime/date,
    config: dict = None  # Exit thresholds (optional)
) -> dict:
```

### Exit Signals (Priority Order):

1. **🥇 Wave Phase Exhaustion** (Primary - earliest off-ramp):
   - Calculate: `wave_position = (current_price - avg_trough) / wave_range`
   - Exit when: `wave_position >= 0.70-0.85` (configurable)
   - Rationale: Capture 70-85% of wave expansion, avoid rollover risk

2. **🥈 Smoothed Rollover** (Confirmation):
   - Use same smoothing window as entry: `half_period`
   - Detect: Price crosses below rising smoothed line
   - Signal: `price < smoothed_price AND previous_smoothed < current_smoothed`
   - Rationale: Confirms wave has stopped accelerating

3. **🥉 Time Expiry** (Cycle Complete):
   - Calculate: `days_held >= 1.2 × dominant_period_days`
   - Rationale: One full cycle has completed

### Return Structure:
```python
{
    'exhausted': bool,          # True if any exit signal triggered
    'signal': str,              # 'wave_phase_exit' | 'smoothed_rollover' | 'time_expiry' | None
    'wave_position': float,     # Current wave position
    'details': dict             # Signal-specific details
}
```

## Integration Points

### 1. In `backtest_signal()`:
- Extract `entry_wave_state` from candidate row
- Call `detect_wave_exhaustion()` each day during backtest
- Check wave exhaustion **before** stop/target/timeout checks
- Exit at current price when exhaustion detected

### 2. In `run_backtest()`:
- Extract wave state from candidate DataFrame row
- Pass to `backtest_signal()` as `entry_wave_state` parameter

### 3. Future: Real Trading System:
- Store `entry_wave_state` when trade is executed
- Monitor holdings periodically with `detect_wave_exhaustion()`
- Exit when exhaustion detected

## Configuration Constants Needed

Suggested additions to CONFIG section:
```python
WAVE_EXIT_ENABLED = True
EXIT_WAVE_POSITION = 0.75  # Conservative: 0.65, Balanced: 0.75, Aggressive: 0.85
WAVE_TIME_EXPIRY_MULTIPLIER = 1.2  # Exit after 1.2 × dominant_period
ROLLOVER_LOOKBACK = 3  # Days to look back for smoothed rollover confirmation
```

## Questions for Implementation

1. **Wave Power Decay**: You mentioned tracking dominant scale power over time. Should we also store initial wavelet power at entry for comparison, or is wave phase + rollover sufficient?

2. **Exit Price**: When wave exhaustion is detected, should we:
   - Exit at current close price (simplest)
   - Exit at smoothed price (more conservative)
   - Exit at recent high (capture peak, but requires lookback)

3. **Hybrid Exit Priority**: Should we check wave exhaustion signals in parallel with stop-loss, or should wave exhaustion override stop-loss (e.g., if wave_position > 0.75 but price is above entry, exit at profit even if below stop)?

## Code Location Reference

- **Wave calculation**: `wavelet_trade_engine()` - lines 259-376
- **Candidate generation**: `generate_trading_candidates()` - lines 374-481
- **Backtesting**: `backtest_signal()` - lines 526-652, `run_backtest()` - lines 655-816

---

**Ready for**: Implementation of `detect_wave_exhaustion()` function with integration into existing backtesting framework.







































