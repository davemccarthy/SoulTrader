# Wave Exhaustion Exit Testing Results

## Test Setup

Testing wave exhaustion exit signals against fixed target exits to validate the concept.

**Configuration:**
- `WAVE_EXIT_ENABLED = True`
- `MAX_TARGET_PCT = 0.50` (50% - intentionally high to test wave exhaustion)
- `EXIT_WAVE_POSITION`: Tested at multiple thresholds
- Test Date: November 4, 2025 (known good performance period)
- Total Trades: 20

## Test Results Summary

### Baseline: Fixed 10% Target (MAX_TARGET_PCT = 0.10)
```
Total Trades: 20
Profitable Trades: 16 (80.0%)
  - Wins: 8 (40.0%)
  - Wave Exits: 8 (40.0%)
Losses: 3 (15.0%)
Timeouts: 1 (5.0%)
Average Return: 3.60% ($+36.01)
Total Return: $+720.28
```

**Key stocks that hit 10% target:**
- UVIX: +10.0% (3 days)
- ALC: +10.0% (8 days)
- SPXS: +10.0% (16 days)

### Test 1: EXIT_WAVE_POSITION = 0.75 (75% threshold)
```
Total Trades: 20
Profitable Trades: 14 (70.0%)
  - Wins: 5 (25.0%)
  - Wave Exits: 9 (45.0%)
Losses: 4 (20.0%)
Timeouts: 2 (10.0%)
Average Return: 2.16% ($+21.65)
Average Wave Exit Return: 4.81% ($+48.11)
Total Return: $+432.93
```

**Performance vs 10% target:**
- UVIX: +10.0% → **-8.5%** (hit stop-loss, -18.5% worse)
- ALC: +10.0% → **+6.74% timeout** (-3.26% worse)
- SPXS: +10.0% → **+3.03% wave_exit** (-6.97% worse)

**Total degradation: -28.73% on these three stocks**

### Test 2: EXIT_WAVE_POSITION = 0.90 (90% threshold)
```
Total Trades: 20
Profitable Trades: 14 (70.0%)
  - Wins: 10 (50.0%)
  - Wave Exits: 4 (20.0%)
Losses: 4 (20.0%)
Timeouts: 2 (10.0%)
Average Return: 2.10% ($+21.02)
Average Wave Exit Return: 6.90% ($+68.97) - Higher but fewer exits
Total Return: $+420.49
```

**Stocks that changed from wave_exit to win:**
- KBE: 1.67% wave_exit → 1.76% win (+0.09%)
- TSN: 3.58% wave_exit → 3.39% win (-0.19%)
- SH: 3.74% wave_exit → 3.36% win (-0.38%)
- EQNR: 3.11% wave_exit → 2.80% win (-0.31%)
- SPDN: 3.61% wave_exit → 3.17% win (-0.44%)

**Observation:** With 0.90 threshold, wave exhaustion rarely triggers. Stocks that previously exited via wave exhaustion now exit via target/stop, often at lower returns.

### Test 3: EXIT_WAVE_POSITION = 0.60 (60% threshold)
```
Total Trades: 20
Profitable Trades: 14 (70.0%)
  - Wins: 2 (10.0%)
  - Wave Exits: 12 (60.0%) - Very high trigger rate
Losses: 4 (20.0%)
Timeouts: 2 (10.0%)
Average Return: 2.15% ($+21.50)
Average Wave Exit Return: 4.68% ($+46.84)
Total Return: $+429.93
```

**Observation:** 60% of trades exit via wave exhaustion (most aggressive threshold yet). Average wave exit return similar to 0.75 threshold, but overall performance still below baseline.

**Stocks that changed:**
- PR: Now wave_exit at 6.82% (was win at 6.11% with 10% target) - slightly better
- TIGR: Now wave_exit at 3.05% (was win at 4.24% with 10% target) - worse
- DOG: Now wave_exit at 3.03% (was win at 2.86% with 10% target) - similar

## Performance Comparison Summary

| Configuration | Avg Return | Total Return | Wave Exits | Wave Exit Avg | Profitable Rate |
|--------------|------------|--------------|------------|---------------|-----------------|
| **Baseline: 10% Target** | **3.60%** | **$720.28** | 8 (40%) | 4.81% | 80% |
| 0.60 Threshold | 2.15% | $429.93 | 12 (60%) | 4.68% | 70% |
| 0.75 Threshold | 2.16% | $432.93 | 9 (45%) | 4.81% | 70% |
| 0.90 Threshold | 2.10% | $420.49 | 4 (20%) | 6.90% | 70% |

**Conclusion:** All wave exhaustion thresholds underperform the fixed 10% target baseline by approximately 40% ($430-433 vs $720).

## Key Findings

### 1. Wave Exhaustion Performance Issues

**Problem:** Wave exhaustion exits consistently underperform fixed 10% target exits.

**Evidence:**
- All three stocks (UVIX, ALC, SPXS) that hit 10% targets performed worse with wave exhaustion
- Average returns lower: 2.10-2.16% vs 3.60% baseline
- Total return lower: $420-433 vs $720 baseline

### 2. Wave Position Calculation May Be Mismatched

**Hypothesis:** The `wave_position` metric may not align with actual price performance for fast-moving stocks.

**Calculation:**
```python
wave_position = (current_price - avg_trough) / wave_range
```

**Issue:** Stocks that achieve 10% price gains may not reach high wave_position values if:
- Wave range is large (making the ratio small)
- Price moves quickly (historical averages lag)
- The calculation uses smoothed averages that don't capture rapid moves

**Example:**
- UVIX hit +10% gain but wave exhaustion didn't trigger before stop-loss hit
- Suggests wave_position was still low despite significant price appreciation

### 3. Threshold Sensitivity

**Observations:**
- **0.60 threshold:** Very aggressive, 60% of trades exit via wave exhaustion, average 4.68% return, overall 2.15% (similar to other thresholds)
- **0.75 threshold:** Balanced trigger rate (45%), average 4.81% wave exit return, overall 2.16%
- **0.90 threshold:** Rarely triggers (20%), average 6.90% wave exit return (higher but fewer), overall 2.10%
- **All thresholds underperform 10% fixed target baseline (3.60% overall return)**

**Key Finding:** None of the tested thresholds (0.60, 0.75, 0.90) achieve the baseline performance of fixed 10% target exits.

### 4. Different Exit Signals Show Different Performance

**Wave Phase Exit (wave_position >= threshold):**
- Most common signal
- Performance varies significantly with threshold

**Smoothed Rollover:**
- Triggered for SPXS and ACI
- Early exits (3-4% range)
- May be catching reversals but also missing continued moves

**Time Expiry:**
- Triggered for BIZD (38 days = 1.2 × dominant_period)
- Worked well (4.14% return)

## Questions for ChatGPT

1. **Wave Position Mismatch:** Why do stocks with 10% price gains not trigger wave_position >= 0.75? Is the calculation appropriate for fast momentum moves?

2. **Threshold Selection:** What is the optimal EXIT_WAVE_POSITION threshold, or is a fixed threshold the wrong approach entirely?

3. **Hybrid Approach:** Should we use a hybrid exit strategy (e.g., exit at 10% OR wave exhaustion, whichever comes first)?

4. **Fast vs Slow Moves:** Does wave exhaustion work better for slow, extended waves vs. fast momentum spikes? Should we disable it for certain stock types?

5. **Entry Wave Position:** All tested stocks had negative entry wave_position values (e.g., -4.85, -2.04, -1.03). Does this affect exit calculations?

6. **Fundamental Concept:** Is wave exhaustion exit the right approach for Oscilla, or should we stick with fixed targets for simplicity and performance?

## Current Hypothesis

Wave exhaustion may be conceptually sound but:
- The wave_position metric may not accurately reflect price performance for rapid moves
- Fixed targets (10%) may be more reliable for capturing gains
- Wave exhaustion might work better for slower, more extended cycles

## Next Steps

1. ✅ **Completed:** Tested 0.60, 0.75, and 0.90 thresholds - all underperform baseline
2. **Consult ChatGPT:** Get expert analysis on why wave exhaustion underperforms and how to fix it
3. **Consider hybrid approach:** Wave exhaustion OR fixed target (whichever triggers first) - might combine best of both
4. **Analyze correlation:** Between entry wave_position and exit performance
5. **Review calculation:** Wave_position calculation to understand why it doesn't align with price gains
6. **Alternative:** Consider disabling wave exhaustion and using fixed targets, or making it conditional on trade characteristics

