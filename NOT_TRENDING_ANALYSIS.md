# NOT_TRENDING Sell Instruction Analysis

## Current Status

### ✅ Currently Uses NOT_TRENDING
1. **Flux** - Uses NOT_TRENDING (volume-based momentum advisor)

### ❌ Does NOT Use NOT_TRENDING

#### Advisors with sell_instructions (should consider adding):
1. **Intraday** - Uses: `STOP_PRICE`, `TARGET_PRICE`, `END_DAY`
   - **Recommendation: ✅ ADD** - Momentum-based advisor, volume is critical
   
2. **FDA** - Uses: `TARGET_PERCENTAGE`, `STOP_PERCENTAGE`, `AFTER_DAYS`, `DESCENDING_TREND`
   - **Recommendation: ✅ ADD** - Event-driven, volume often drops after initial pop
   
3. **Insider** - Uses: `STOP_PERCENTAGE`, `TARGET_PERCENTAGE`, `AFTER_DAYS`, `DESCENDING_TREND`
   - **Recommendation: ✅ ADD** - Insider buying often comes with volume, should exit when volume dries up
   
4. **StockStory** (via `AdvisorBase.analyze`) - Uses: `TARGET_PERCENTAGE`, `STOP_PERCENTAGE`, `DESCENDING_TREND`
   - **Recommendation: ✅ ADD** - News-driven moves often have volume, should exit when interest wanes
   
5. **Polygon** (via `AdvisorBase.news_flash` → `analyze`) - Uses same as StockStory
   - **Recommendation: ✅ ADD** - Same reasoning as StockStory (news-driven)

#### Advisors without sell_instructions (lower priority):
6. **Yahoo** - No sell_instructions
   - **Recommendation: ⚠️ CONSIDER** - Value-based advisor, less volume-dependent, but could still benefit
   
7. **User** - No sell_instructions
   - **Recommendation: ❌ SKIP** - User-selected stocks, let user decide exits

## Recommendation

**Add NOT_TRENDING to all momentum/event-driven advisors:**
- ✅ Intraday
- ✅ FDA  
- ✅ Insider
- ✅ StockStory
- ✅ Polygon

**Rationale:**
- These advisors discover stocks based on momentum, events, or news
- Volume is a key indicator of continued interest
- When volume drops significantly, the catalyst may have played out
- NOT_TRENDING provides an automatic exit when the trade loses momentum

**Implementation:**
Add `("NOT_TRENDING", None)` to the `sell_instructions` list in each advisor's discovery method.












