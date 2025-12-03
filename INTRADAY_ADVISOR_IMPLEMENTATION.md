# Intraday Advisor Implementation

## Overview
The Intraday Momentum Trading Advisor has been integrated into the SoulTrader system for EXPERIMENTAL risk users only.

## Key Features

### Discovery Window
- **Time**: 15:30-16:30 GMT (10:30-11:30 ET) - Hours 1-2 of market opening
- **Frequency**: Once per day per SA session
- **Validation**: Automatically checks if within discovery window before running

### Entry Signals
The advisor discovers stocks that meet ALL of these conditions:
- Price > VWAP (intraday value)
- Short EMA (8) > Long EMA (21) - momentum confirmation
- RSI < 75 (not overbought)
- Volume > 1.5x average (volume surge)
- Market cap > $100M (liquidity filter)

### Exit Instructions
Each discovery automatically includes sell instructions:
- **STOP_LOSS**: Entry price - (2 × ATR)
- **TARGET_PRICE**: Entry price + (3 × ATR)

These are calculated as multipliers (e.g., 0.98 for stop loss = 98% of entry price).

### Risk Profile
- Only available for **EXPERIMENTAL** risk users
- Configured in `core/models.py` Profile.RISK['EXPERIMENTAL']['advisors']

## Files Created/Modified

### New Files
1. **`core/services/advisors/intraday.py`**
   - Main advisor implementation
   - Time window checking
   - Stock scanning and signal evaluation
   - Discovery creation with sell instructions

### Modified Files
1. **`core/services/advisors/__init__.py`**
   - Added import for intraday advisor
   - Registered in advisors namespace

2. **`core/models.py`**
   - Added "Intraday" to EXPERIMENTAL risk profile advisors list

## Usage

### Automatic Discovery
The advisor will automatically run during discovery phase when:
1. SA session runs during 10:30-11:30 ET
2. User has EXPERIMENTAL risk profile
3. No discoveries already made in this SA session

### Manual Testing
To test the advisor:
```python
from core.services.advisors.intraday import Intraday
from core.models import Advisor, SmartAnalysis

# Get or create advisor
advisor_obj = Advisor.objects.get_or_create(
    name="Intraday",
    defaults={'python_class': 'Intraday', 'enabled': True}
)[0]

# Create SA session
sa = SmartAnalysis.objects.create(username="test_user")

# Create advisor instance
intraday = Intraday(advisor_obj)

# Run discovery
intraday.discover(sa)
```

## Technical Details

### Time Checking
- Uses `pytz` for timezone handling
- Checks Eastern Time (ET) for market hours
- Validates weekday (excludes weekends)
- Returns helpful error messages if outside discovery window

### Stock Scanning
- Fetches top 50 most active stocks from Yahoo Finance
- Sorts by volume after fetching
- Fallback list if screener fails

### Indicators Calculated
- EMA (8-period and 21-period)
- RSI (14-period)
- ATR (14-period)
- VWAP (cumulative)
- Volume moving average (20-period)

### Once-Per-Day Logic
Checks if Intraday discoveries already exist in the current SA session. Since SA sessions typically run once per day, this ensures discoveries are made only once per day.

## Future Enhancements

### Sell Instructions (Future)
- Manual sell for winning stocks
- "Sell if up after 5 hours" instruction (can be added later)

### Performance Tracking
- Use `analyze_intraday_performance.py` script to track daily results
- Monitor win rate and refine entry criteria based on data

## Testing Notes

1. **Time Zone**: Ensure server timezone is correctly configured
2. **Market Hours**: Discovery only runs during specified window
3. **User Risk**: Must be set to EXPERIMENTAL for advisor to run
4. **Once Per Day**: Won't discover again if already discovered in same SA session

## Integration with Main System

The advisor integrates seamlessly with the existing discovery system:
- Uses standard `AdvisorBase.discovered()` method
- Creates `Discovery` records with proper relationships
- Adds `SellInstruction` records for TP/SL
- Filtered by risk profile in `analyze_discovery()`

## See Also

- `test_intraday_trade.py` - Standalone testing script
- `analyze_intraday_performance.py` - Performance analysis tool
- Discovery results are tracked in CSV files (from test script) or in database (from advisor)






