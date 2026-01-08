#!/usr/bin/env python3
"""
Test Oscilla trading strategy across multiple distinct periods to assess parameter robustness.

This script runs backtests on different time periods to avoid overfitting to a single date.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the current directory to path to import test_oscilla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_oscilla
import pandas as pd

def test_multiple_periods(test_periods, max_stocks_per_period=20, verbose=False):
    """
    Test strategy across multiple distinct time periods.
    
    Args:
        test_periods: List of (start_date, end_date, period_name) tuples
        max_stocks_per_period: Maximum stocks to analyze per period (for speed)
        verbose: If True, show detailed output for each period
    
    Returns:
        Dictionary with results for each period and summary statistics
    """
    all_results = {}
    period_summaries = []
    
    print("=" * 80)
    print("MULTI-PERIOD BACKTEST - Parameter Robustness Test")
    print("=" * 80)
    print(f"Testing {len(test_periods)} distinct periods\n")
    
    for i, (start_date, end_date, period_name) in enumerate(test_periods, 1):
        print(f"\n{'='*80}")
        print(f"PERIOD {i}/{len(test_periods)}: {period_name}")
        print(f"Date Range: {start_date} to {end_date}")
        print(f"{'='*80}\n")
        
        # Run backtest for this period
        # Calculate approximate number of dates (weekly sampling)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days_diff = (end_dt - start_dt).days
        num_dates = max(5, min(20, days_diff // 7))  # Sample approximately weekly
        
        df_results = test_oscilla.run_backtest(
            start_date=start_date,
            end_date=end_date,
            num_dates=num_dates,
            max_stocks=max_stocks_per_period,
            verbose=verbose
        )
        
        if df_results.empty:
            print(f"⚠️  No results for {period_name}")
            all_results[period_name] = pd.DataFrame()
            continue
        
        # Calculate summary statistics for this period
        total_trades = len(df_results)
        wins = len(df_results[df_results['outcome'] == 'win'])
        losses = len(df_results[df_results['outcome'] == 'loss'])
        timeouts = len(df_results[df_results['outcome'] == 'timeout'])
        stop_slides = len(df_results[df_results['outcome'] == 'stop_slide'])
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        avg_return = df_results['return_pct'].mean()
        avg_win = df_results[df_results['outcome'] == 'win']['return_pct'].mean() if wins > 0 else 0
        avg_loss = df_results[df_results['outcome'] == 'loss']['return_pct'].mean() if losses > 0 else 0
        avg_days_held = df_results['days_held'].mean()
        total_return = df_results['return_dollars'].sum()
        expected_value = df_results['return_pct'].mean()
        
        period_summary = {
            'period': period_name,
            'start_date': start_date,
            'end_date': end_date,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'timeouts': timeouts,
            'stop_slides': stop_slides,
            'win_rate_pct': round(win_rate, 1),
            'avg_return_pct': round(avg_return, 2),
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'avg_days_held': round(avg_days_held, 1),
            'total_return_dollars': round(total_return, 2),
            'expected_value_pct': round(expected_value, 2)
        }
        
        period_summaries.append(period_summary)
        all_results[period_name] = df_results
        
        # Print period summary
        print(f"\n📊 {period_name} Results:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Win Rate: {win_rate:.1f}% ({wins} wins, {losses} losses, {timeouts} timeouts)")
        if stop_slides > 0:
            print(f"   STOP_SLIDE exits: {stop_slides}")
        print(f"   Average Return: {avg_return:.2f}%")
        print(f"   Average Win: {avg_win:.2f}%")
        print(f"   Average Loss: {avg_loss:.2f}%")
        print(f"   Average Days Held: {avg_days_held:.1f}")
        print(f"   Total Return: ${total_return:,.2f}")
        print(f"   Expected Value: {expected_value:.2f}% per trade")
    
    # Calculate cross-period statistics
    print(f"\n{'='*80}")
    print("CROSS-PERIOD SUMMARY")
    print(f"{'='*80}\n")
    
    df_summary = pd.DataFrame(period_summaries)
    
    if not df_summary.empty:
        print("Period-by-Period Results:")
        print(df_summary.to_string(index=False))
        
        print(f"\n📈 Aggregate Statistics Across All Periods:")
        total_trades_all = df_summary['total_trades'].sum()
        total_wins_all = df_summary['wins'].sum()
        total_losses_all = df_summary['losses'].sum()
        overall_win_rate = (total_wins_all / total_trades_all * 100) if total_trades_all > 0 else 0
        
        # Weighted averages
        weighted_avg_return = (df_summary['avg_return_pct'] * df_summary['total_trades']).sum() / total_trades_all if total_trades_all > 0 else 0
        weighted_avg_win = (df_summary['avg_win_pct'] * df_summary['wins']).sum() / total_wins_all if total_wins_all > 0 else 0
        weighted_avg_loss = (df_summary['avg_loss_pct'] * df_summary['losses']).sum() / total_losses_all if total_losses_all > 0 else 0
        weighted_avg_days = (df_summary['avg_days_held'] * df_summary['total_trades']).sum() / total_trades_all if total_trades_all > 0 else 0
        total_return_all = df_summary['total_return_dollars'].sum()
        
        print(f"   Total Trades Across All Periods: {total_trades_all}")
        print(f"   Overall Win Rate: {overall_win_rate:.1f}% ({total_wins_all} wins, {total_losses_all} losses)")
        print(f"   Weighted Average Return: {weighted_avg_return:.2f}%")
        print(f"   Weighted Average Win: {weighted_avg_win:.2f}%")
        print(f"   Weighted Average Loss: {weighted_avg_loss:.2f}%")
        print(f"   Weighted Average Days Held: {weighted_avg_days:.1f}")
        print(f"   Total Return Across All Periods: ${total_return_all:,.2f}")
        print(f"   Expected Value Per Trade: {weighted_avg_return:.2f}%")
        
        # Consistency metrics
        print(f"\n📊 Consistency Metrics:")
        print(f"   Win Rate Range: {df_summary['win_rate_pct'].min():.1f}% - {df_summary['win_rate_pct'].max():.1f}%")
        print(f"   Avg Return Range: {df_summary['avg_return_pct'].min():.2f}% - {df_summary['avg_return_pct'].max():.2f}%")
        print(f"   Std Dev of Win Rates: {df_summary['win_rate_pct'].std():.1f}%")
        print(f"   Std Dev of Avg Returns: {df_summary['avg_return_pct'].std():.2f}%")
        
        # Periods with positive vs negative returns
        positive_periods = len(df_summary[df_summary['avg_return_pct'] > 0])
        negative_periods = len(df_summary[df_summary['avg_return_pct'] <= 0])
        print(f"\n   Periods with Positive Returns: {positive_periods}/{len(df_summary)} ({positive_periods/len(df_summary)*100:.1f}%)")
        print(f"   Periods with Negative Returns: {negative_periods}/{len(df_summary)} ({negative_periods/len(df_summary)*100:.1f}%)")
    
    return {
        'period_results': all_results,
        'summary': df_summary,
        'period_summaries': period_summaries
    }


def main():
    """Define test periods and run multi-period backtest."""
    
    # Define distinct test periods (spread across different time frames)
    # Adjust these dates based on available historical data
    test_periods = [
        # Q4 2024 periods
        ("2024-10-01", "2024-10-31", "October 2024"),
        ("2024-11-01", "2024-11-30", "November 2024"),
        ("2024-12-01", "2024-12-31", "December 2024"),
        
        # Q1 2025 periods
        ("2025-01-01", "2025-01-31", "January 2025"),
        ("2025-02-01", "2025-02-28", "February 2025"),
        ("2025-03-01", "2025-03-31", "March 2025"),
        ("2025-04-01", "2025-04-30", "April 2025"),
        
        # Additional periods if you have data
        # ("2024-07-01", "2024-07-31", "July 2024"),
        # ("2024-08-01", "2024-08-31", "August 2024"),
        # ("2024-09-01", "2024-09-30", "September 2024"),
    ]
    
    # Run multi-period test
    results = test_multiple_periods(
        test_periods=test_periods,
        max_stocks_per_period=20,  # Limit per period for speed
        verbose=False  # Set to True for detailed output per period
    )
    
    print(f"\n{'='*80}")
    print("Multi-Period Testing Complete")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
































