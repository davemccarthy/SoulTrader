#!/usr/bin/env python
"""
Quick script to run multiple backtests and summarize results.
"""
import subprocess
import sys
from datetime import datetime

def run_backtest(weeks_ago, peak_passed_threshold=None, peak_volume_multiplier=None, descending_trend_threshold=None, descending_trend_lookback=None, stop_loss=None, min_cycles=2, buy_threshold=0.3):
    """Run a single backtest and extract summary stats."""
    cmd = [
        sys.executable,
        'test_cyclical_patterns.py',
        '--backtest',
        '--limit', '30',
        '--lookback', '180',
        '--min-cycles', str(min_cycles),
        '--buy-threshold', str(buy_threshold),
        '--weeks-ago', str(weeks_ago)
    ]
    if peak_passed_threshold is not None:
        cmd.extend(['--peak-passed-threshold', str(peak_passed_threshold)])
    if peak_volume_multiplier is not None:
        cmd.extend(['--peak-volume-multiplier', str(peak_volume_multiplier)])
    if descending_trend_threshold is not None:
        cmd.extend(['--descending-trend-threshold', str(descending_trend_threshold)])
    if descending_trend_lookback is not None:
        cmd.extend(['--descending-trend-lookback', str(descending_trend_lookback)])
    if stop_loss is not None:
        cmd.extend(['--stop-loss', str(stop_loss)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        
        # Extract summary stats
        stats = {}
        
        # Check if backtest was aborted
        if 'BACKTEST ABORTED' in output:
            for line in output.split('\n'):
                if 'Found' in line and 'BUY signal' in line:
                    # Extract number of signals found
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.isdigit() and i < len(parts) - 1 and 'signal' in parts[i+1]:
                            stats['total_signals'] = part
                            stats['aborted'] = True
                            break
            return stats
        
        for line in output.split('\n'):
            if 'Total BUY signals tested:' in line:
                stats['total_signals'] = line.split(':')[1].strip()
            elif 'Valid price data:' in line:
                stats['valid_data'] = line.split(':')[1].strip()
            elif 'Stocks sold:' in line:
                stats['stocks_sold'] = line.split(':')[1].strip().split()[0]
            elif 'Winners:' in line:
                stats['winners'] = line.split(':')[1].strip().split()[0]
            elif 'Losers:' in line:
                stats['losers'] = line.split(':')[1].strip().split()[0]
            elif 'Average return:' in line:
                stats['avg_return'] = line.split(':')[1].strip()
            elif 'Win rate:' in line:
                stats['win_rate'] = line.split(':')[1].strip()
            elif 'Best performer:' in line:
                stats['best'] = line.split(':')[1].strip()
            elif 'Worst performer:' in line:
                stats['worst'] = line.split(':')[1].strip()
            elif 'Average days held:' in line:
                stats['avg_days'] = line.split(':')[1].strip()
        
        return stats
    except Exception as e:
        return {'error': str(e)}

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run batch backtests')
    parser.add_argument('--peak-passed-threshold', type=float, default=None,
                       help='PEAK_PASSED threshold to test (default: None = no PEAK_PASSED)')
    parser.add_argument('--peak-volume-multiplier', type=float, default=None,
                       help='Volume multiplier for volume-confirmed peaks (e.g., 1.5 = 150%% of avg volume)')
    parser.add_argument('--descending-trend-threshold', type=float, default=None,
                       help='DESCENDING_TREND threshold to test (e.g., -5.0 = 5%%, default: None = no DESCENDING_TREND)')
    parser.add_argument('--descending-trend-lookback', type=int, default=None,
                       help='DESCENDING_TREND lookback days (e.g., 7 = 7-day average, default: None = no DESCENDING_TREND)')
    parser.add_argument('--stop-loss', type=float, default=None,
                       help='Stop loss multiplier relative to buy price (e.g., 0.95 = 5%% below, 0.78 = 22%% below, default: None = disabled)')
    parser.add_argument('--min-cycles', type=int, default=2,
                       help='Minimum number of cycles required (default: 2, use 3 for stricter filtering)')
    parser.add_argument('--buy-threshold', type=float, default=0.3,
                       help='BUY threshold - position must be < this value (default: 0.3 = 30%%, use 0.2 for 20%%)')
    args = parser.parse_args()
    
    peak_threshold = args.peak_passed_threshold
    peak_volume_mult = args.peak_volume_multiplier
    descending_threshold = args.descending_trend_threshold
    descending_lookback = args.descending_trend_lookback
    stop_loss = args.stop_loss
    min_cycles = args.min_cycles
    buy_threshold = args.buy_threshold
    
    title = "COMPREHENSIVE BACKTEST ANALYSIS"
    if stop_loss is not None:
        stop_loss_pct = (1 - stop_loss) * 100
        title += f" (with STOP_LOSS {stop_loss_pct:.1f}% below buy price)"
    elif peak_threshold is not None:
        title += f" (with PEAK_PASSED {peak_threshold}%"
        if peak_volume_mult is not None:
            title += f", volume-confirmed {peak_volume_mult}x avg"
        title += ")"
    elif descending_threshold is not None and descending_lookback is not None:
        title += f" (with DESCENDING_TREND {abs(descending_threshold)}% below {descending_lookback}-day avg)"
    else:
        title += " (without STOP_LOSS, PEAK_PASSED or DESCENDING_TREND)"
    
    print("="*100)
    print(title)
    print("="*100)
    print(f"Running backtests across multiple time periods...\n")
    
    weeks_to_test = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]
    results = []
    
    for weeks in weeks_to_test:
        print(f"Testing {weeks} weeks ago...", end=' ', flush=True)
        stats = run_backtest(weeks, peak_passed_threshold=peak_threshold, peak_volume_multiplier=peak_volume_mult, 
                            descending_trend_threshold=descending_threshold, descending_trend_lookback=descending_lookback,
                            stop_loss=stop_loss, min_cycles=min_cycles, buy_threshold=buy_threshold)
        if 'error' not in stats:
            results.append((weeks, stats))
            if stats.get('aborted'):
                print(f"⚠ ABORTED ({stats.get('total_signals', '?')} signals - below minimum)")
            else:
                print(f"✓ ({stats.get('total_signals', '?')} signals, {stats.get('avg_return', '?')} avg return)")
        else:
            print(f"✗ Error: {stats['error']}")
    
    print("\n" + "="*100)
    print("SUMMARY RESULTS")
    print("="*100)
    print(f"\n{'Weeks':<8} {'Signals':<10} {'Sold':<8} {'Winners':<10} {'Avg Return':<12} {'Win Rate':<10} {'Best':<12} {'Worst':<12} {'Days Held':<10}")
    print("-"*100)
    
    for weeks, stats in results:
        if stats.get('aborted'):
            print(f"{weeks:<8} {stats.get('total_signals', 'N/A'):<10} {'ABORTED':<8} {'-':<10} {'-':<12} {'-':<10} {'-':<12} {'-':<12} {'-':<10}")
        else:
            print(f"{weeks:<8} {stats.get('total_signals', 'N/A'):<10} {stats.get('stocks_sold', 'N/A'):<8} "
                  f"{stats.get('winners', 'N/A'):<10} {stats.get('avg_return', 'N/A'):<12} "
                  f"{stats.get('win_rate', 'N/A'):<10} {stats.get('best', 'N/A'):<12} "
                  f"{stats.get('worst', 'N/A'):<12} {stats.get('avg_days', 'N/A'):<10}")
    
    # Calculate overall stats
    if results:
        avg_returns = []
        win_rates = []
        total_signals = 0
        total_winners = 0
        total_losers = 0
        
        for weeks, stats in results:
            try:
                avg_return_str = stats.get('avg_return', '').replace('%', '').replace('+', '').strip()
                if avg_return_str and avg_return_str != 'N/A':
                    avg_returns.append(float(avg_return_str))
            except:
                pass
            
            try:
                win_rate_str = stats.get('win_rate', '').replace('%', '').strip()
                if win_rate_str and win_rate_str != 'N/A':
                    win_rates.append(float(win_rate_str))
            except:
                pass
            
            try:
                total_signals += int(stats.get('total_signals', 0))
                total_winners += int(stats.get('winners', 0))
                total_losers += int(stats.get('losers', 0))
            except:
                pass
        
        print("\n" + "="*100)
        print("OVERALL STATISTICS")
        print("="*100)
        if avg_returns:
            print(f"Average return across all periods: {sum(avg_returns)/len(avg_returns):+.1f}%")
        if win_rates:
            print(f"Average win rate across all periods: {sum(win_rates)/len(win_rates):.1f}%")
        if total_signals > 0:
            overall_win_rate = (total_winners / (total_winners + total_losers) * 100) if (total_winners + total_losers) > 0 else 0
            print(f"Total signals tested: {total_signals}")
            print(f"Total winners: {total_winners}")
            print(f"Total losers: {total_losers}")
            print(f"Overall win rate: {overall_win_rate:.1f}%")
        
        # Find best and worst periods
        if avg_returns:
            best_idx = avg_returns.index(max(avg_returns))
            worst_idx = avg_returns.index(min(avg_returns))
            print(f"\nBest period: {results[best_idx][0]} weeks ago ({results[best_idx][1].get('avg_return', 'N/A')} avg return)")
            print(f"Worst period: {results[worst_idx][0]} weeks ago ({results[worst_idx][1].get('avg_return', 'N/A')} avg return)")
    
    print("\n" + "="*100)

if __name__ == "__main__":
    main()

