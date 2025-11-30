"""
Intraday Trading Strategy - Performance Analysis Tool

Analyzes daily discovery and exit results to track strategy performance over time.
Helps identify patterns and optimize entry criteria.

Usage:
    python analyze_intraday_performance.py
    
    Or analyze specific date range:
    python analyze_intraday_performance.py --start-date 2025-11-26 --end-date 2025-11-28
"""

import pandas as pd
import numpy as np
import glob
import os
import argparse
from datetime import datetime, timedelta
from collections import defaultdict

def extract_date_from_filename(filename):
    """Extract date from filename like intraday_discoveries_20251126_152833.csv or intraday_eod_exits_20251126_201617.csv"""
    try:
        basename = os.path.basename(filename)
        # Format: intraday_*_YYYYMMDD_HHMMSS.csv
        # Extract YYYYMMDD from anywhere in filename
        import re
        match = re.search(r'(\d{8})', basename)
        if match:
            date_str = match.group(1)  # YYYYMMDD
            return datetime.strptime(date_str, '%Y%m%d').date()
    except Exception as e:
        pass
    return None

def find_matching_files():
    """Find all discovery and EOD exit CSV files and match them by date"""
    discovery_files = glob.glob('intraday_discoveries_*.csv')
    exit_files = glob.glob('intraday_eod_exits_*.csv')
    held_files = glob.glob('intraday_eod_held_*.csv')
    
    # Group by date
    daily_data = defaultdict(lambda: {
        'discoveries': None,
        'exits': None,
        'held': None,
        'date': None
    })
    
    for f in discovery_files:
        date = extract_date_from_filename(f)
        if date:
            daily_data[date]['discoveries'] = f
            daily_data[date]['date'] = date
    
    for f in exit_files:
        date = extract_date_from_filename(f)
        if date:
            daily_data[date]['exits'] = f
    
    for f in held_files:
        date = extract_date_from_filename(f)
        if date:
            daily_data[date]['held'] = f
    
    return daily_data

def analyze_daily_performance(date, discoveries_file, exits_file, held_file):
    """Analyze performance for a single trading day"""
    result = {
        'date': date,
        'total_discoveries': 0,
        'winners': 0,
        'losers': 0,
        'win_rate': 0.0,
        'avg_gain_winners': 0.0,
        'avg_loss_losers': 0.0,
        'best_winner_pct': 0.0,
        'worst_loser_pct': 0.0,
        'total_pnl_pct': 0.0,
        'avg_volume_ratio': 0.0,
        'avg_rsi': 0.0,
        'avg_vwap_spread_pct': 0.0,
        'winners_avg_volume_ratio': 0.0,
        'winners_avg_rsi': 0.0,
        'losers_avg_volume_ratio': 0.0,
        'losers_avg_rsi': 0.0,
    }
    
    try:
        # Load discoveries
        df_disc = pd.read_csv(discoveries_file)
        result['total_discoveries'] = len(df_disc)
        
        # Calculate VWAP spread
        df_disc['vwap_spread_pct'] = ((df_disc['entry_price'] - df_disc['vwap']) / df_disc['vwap']) * 100
        
        result['avg_volume_ratio'] = df_disc['volume_ratio'].mean()
        result['avg_rsi'] = df_disc['rsi'].mean()
        result['avg_vwap_spread_pct'] = df_disc['vwap_spread_pct'].mean()
        
        # Load exits and held
        winners = []
        losers = []
        
        if exits_file and os.path.exists(exits_file):
            df_exits = pd.read_csv(exits_file)
            winners = df_exits['pnl_pct'].tolist()
            result['avg_gain_winners'] = df_exits['pnl_pct'].mean() if len(df_exits) > 0 else 0.0
            result['best_winner_pct'] = df_exits['pnl_pct'].max() if len(df_exits) > 0 else 0.0
        
        if held_file and os.path.exists(held_file):
            df_held = pd.read_csv(held_file)
            losers = df_held['pnl_pct'].tolist()
            result['avg_loss_losers'] = df_held['pnl_pct'].mean() if len(df_held) > 0 else 0.0
            result['worst_loser_pct'] = df_held['pnl_pct'].min() if len(df_held) > 0 else 0.0
        
        result['winners'] = len(winners)
        result['losers'] = len(losers)
        total = result['winners'] + result['losers']
        result['win_rate'] = (result['winners'] / total * 100) if total > 0 else 0.0
        
        # Calculate weighted average P&L
        total_pnl = sum(winners) + sum(losers)
        result['total_pnl_pct'] = total_pnl / total if total > 0 else 0.0
        
        # Compare winners vs losers characteristics
        if exits_file and os.path.exists(exits_file):
            df_exits_merged = df_disc.merge(
                pd.read_csv(exits_file)[['symbol', 'pnl_pct']],
                on='symbol',
                how='inner'
            )
            if len(df_exits_merged) > 0:
                result['winners_avg_volume_ratio'] = df_exits_merged['volume_ratio'].mean()
                result['winners_avg_rsi'] = df_exits_merged['rsi'].mean()
        
        if held_file and os.path.exists(held_file):
            df_held_merged = df_disc.merge(
                pd.read_csv(held_file)[['symbol', 'pnl_pct']],
                on='symbol',
                how='inner'
            )
            if len(df_held_merged) > 0:
                result['losers_avg_volume_ratio'] = df_held_merged['volume_ratio'].mean()
                result['losers_avg_rsi'] = df_held_merged['rsi'].mean()
        
    except Exception as e:
        print(f"⚠️  Error analyzing {date}: {e}")
    
    return result

def print_daily_summary(results):
    """Print summary table of daily performance"""
    if not results:
        print("❌ No data found to analyze")
        return
    
    print("=" * 120)
    print("INTRADAY STRATEGY - DAILY PERFORMANCE SUMMARY")
    print("=" * 120)
    print()
    
    # Daily breakdown
    print(f"{'Date':<12} {'Disc':<5} {'Win':<4} {'Loss':<5} {'WR%':<6} {'Avg Win%':<9} {'Avg Loss%':<10} {'Best%':<7} {'Worst%':<8} {'Total%':<8}")
    print("-" * 120)
    
    for r in sorted(results, key=lambda x: x['date']):
        print(f"{r['date']}  {r['total_discoveries']:<5} {r['winners']:<4} {r['losers']:<5} "
              f"{r['win_rate']:>5.1f}%  {r['avg_gain_winners']:>7.2f}%  {r['avg_loss_losers']:>8.2f}%  "
              f"{r['best_winner_pct']:>5.2f}%  {r['worst_loser_pct']:>6.2f}%  {r['total_pnl_pct']:>6.2f}%")
    
    print()
    print("=" * 120)
    
    # Aggregate statistics
    print("\n📊 AGGREGATE STATISTICS:")
    print("-" * 120)
    
    total_days = len(results)
    total_discoveries = sum(r['total_discoveries'] for r in results)
    total_winners = sum(r['winners'] for r in results)
    total_losers = sum(r['losers'] for r in results)
    total_trades = total_winners + total_losers
    
    overall_win_rate = (total_winners / total_trades * 100) if total_trades > 0 else 0.0
    
    all_winners = []
    all_losers = []
    for r in results:
        if r['winners'] > 0:
            # Approximate from avg (will be slightly off but gives ballpark)
            all_winners.extend([r['avg_gain_winners']] * r['winners'])
        if r['losers'] > 0:
            all_losers.extend([r['avg_loss_losers']] * r['losers'])
    
    avg_win = np.mean(all_winners) if all_winners else 0.0
    avg_loss = np.mean(all_losers) if all_losers else 0.0
    best_win = max((r['best_winner_pct'] for r in results), default=0.0)
    worst_loss = min((r['worst_loser_pct'] for r in results), default=0.0)
    
    print(f"  Trading Days Analyzed: {total_days}")
    print(f"  Total Discoveries: {total_discoveries}")
    print(f"  Total Trades Closed: {total_trades} (Winners: {total_winners}, Losers: {total_losers})")
    print(f"  Overall Win Rate: {overall_win_rate:.1f}%")
    print(f"  Average Winner: +{avg_win:.2f}%")
    print(f"  Average Loser: {avg_loss:.2f}%")
    print(f"  Best Winner: +{best_win:.2f}%")
    print(f"  Worst Loser: {worst_loss:.2f}%")
    
    if avg_loss != 0:
        risk_reward = abs(avg_win / avg_loss)
        print(f"  Risk/Reward Ratio: {risk_reward:.2f}")
    
    print()

def print_pattern_analysis(results):
    """Analyze patterns in winners vs losers"""
    print("=" * 120)
    print("PATTERN ANALYSIS: Winners vs Losers")
    print("=" * 120)
    print()
    
    # Collect all winner/loser characteristics
    winners_vol = []
    winners_rsi = []
    losers_vol = []
    losers_rsi = []
    
    for r in results:
        if r['winners_avg_volume_ratio'] > 0:
            winners_vol.append(r['winners_avg_volume_ratio'])
        if r['winners_avg_rsi'] > 0:
            winners_rsi.append(r['winners_avg_rsi'])
        if r['losers_avg_volume_ratio'] > 0:
            losers_vol.append(r['losers_avg_volume_ratio'])
        if r['losers_avg_rsi'] > 0:
            losers_rsi.append(r['losers_avg_rsi'])
    
    if winners_vol and losers_vol:
        print("📊 VOLUME RATIO:")
        print(f"   Winners Avg: {np.mean(winners_vol):.2f}x")
        print(f"   Losers Avg:  {np.mean(losers_vol):.2f}x")
        print(f"   Difference:  {np.mean(winners_vol) - np.mean(losers_vol):+.2f}x")
        if np.mean(winners_vol) > np.mean(losers_vol):
            threshold = np.mean(losers_vol)
            print(f"   💡 Suggestion: Consider minimum volume_ratio > {threshold:.1f}x")
        print()
    
    if winners_rsi and losers_rsi:
        print("📈 RSI:")
        print(f"   Winners Avg: {np.mean(winners_rsi):.1f}")
        print(f"   Losers Avg:  {np.mean(losers_rsi):.1f}")
        print(f"   Difference:  {np.mean(winners_rsi) - np.mean(losers_rsi):+.1f}")
        if abs(np.mean(winners_rsi) - np.mean(losers_rsi)) > 3:
            print(f"   💡 Suggestion: Consider RSI range {min(winners_rsi):.0f}-{max(winners_rsi):.0f}")
        print()
    
    print()

def print_daily_characteristics(results):
    """Show average daily characteristics"""
    print("=" * 120)
    print("DAILY CHARACTERISTICS")
    print("=" * 120)
    print()
    
    print(f"{'Date':<12} {'Avg Vol':<9} {'Avg RSI':<9} {'Avg Spread%':<12}")
    print("-" * 120)
    
    for r in sorted(results, key=lambda x: x['date']):
        print(f"{r['date']}  {r['avg_volume_ratio']:>7.2f}x  {r['avg_rsi']:>7.1f}  {r['avg_vwap_spread_pct']:>10.2f}%")
    
    print()

def main():
    parser = argparse.ArgumentParser(description='Analyze intraday trading strategy performance')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)', default=None)
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)', default=None)
    parser.add_argument('--save', type=str, help='Save results to CSV file', default=None)
    
    args = parser.parse_args()
    
    # Find all matching files
    daily_data = find_matching_files()
    
    if not daily_data:
        print("❌ No discovery files found. Run discovery mode first.")
        return
    
    # Filter by date range if specified
    if args.start_date:
        start = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        daily_data = {k: v for k, v in daily_data.items() if k >= start}
    
    if args.end_date:
        end = datetime.strptime(args.end_date, '%Y-%m-%d').date()
        daily_data = {k: v for k, v in daily_data.items() if k <= end}
    
    if not daily_data:
        print("❌ No data found for specified date range")
        return
    
    print(f"📂 Found {len(daily_data)} trading day(s) with data")
    print()
    
    # Analyze each day
    results = []
    for date, files in sorted(daily_data.items()):
        if files['discoveries']:
            result = analyze_daily_performance(
                date,
                files['discoveries'],
                files['exits'],
                files['held']
            )
            results.append(result)
    
    if not results:
        print("❌ No valid data found to analyze")
        return
    
    # Print reports
    print_daily_summary(results)
    print_daily_characteristics(results)
    
    if len(results) >= 2:
        print_pattern_analysis(results)
    
    # Save to CSV if requested
    if args.save:
        df_results = pd.DataFrame(results)
        df_results.to_csv(args.save, index=False)
        print(f"💾 Results saved to: {args.save}")
        print()

if __name__ == '__main__':
    main()

