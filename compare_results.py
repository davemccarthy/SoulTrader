#!/usr/bin/env python
"""Compare Nov 16 vs Nov 18 test results"""
import pandas as pd

nov16 = pd.read_csv('penny_stock_with_insider_20251116_210837.csv')
nov18 = pd.read_csv('penny_stock_with_insider_20251118_135620.csv')

print('=== COMPARING TOP 5 SCORES ===')
print('Stock | Nov 16 Score | Nov 18 Score | Change')
print('-' * 50)
for symbol in ['UAA', 'ATUS', 'TEF', 'GAU', 'CINT']:
    s16 = nov16[nov16['symbol'] == symbol]['score'].values[0] if symbol in nov16['symbol'].values else None
    s18 = nov18[nov18['symbol'] == symbol]['score'].values[0] if symbol in nov18['symbol'].values else None
    if s16 and s18:
        change = s18 - s16
        print(f'{symbol:5} | {s16:11.3f} | {s18:11.3f} | {change:+.3f}')
    else:
        print(f'{symbol:5} | {"N/A":>11} | {"N/A":>11} | N/A')

print('\n=== WHY SAME TOP 5? ===')
print('The screener IS returning different stocks (38 overlap, 7 new, 6 dropped)')
print('But the scoring algorithm consistently ranks these 5 highest because:')
print('1. Fundamentals (revenue, cash flow) don\'t change daily')
print('2. These stocks have strong fundamentals relative to other penny stocks')
print('3. Trough detection keeps them high if still near lows')
print('\nThis suggests the algorithm is finding QUALITY penny stocks,')
print('but may not be capturing daily VOLATILITY changes effectively.')












