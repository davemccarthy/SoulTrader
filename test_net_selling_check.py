#!/usr/bin/env python
"""
Test script to verify the net selling check works correctly.
"""

import os
import sys
import django

# Setup Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.services.advisors.insider import check_net_selling

def test_net_selling_check():
    """Test the net selling check for known stocks."""
    
    test_cases = [
        ('AISP', True, 'Should have net selling'),
        ('SRTS', True, 'Should have net selling'),
        ('STRZ', False, 'Should NOT have net selling (net buying)'),
        ('RCG', False, 'Should NOT have net selling (net buying)'),
        ('ANVS', False, 'Should NOT have net selling (from comparison group)'),
        ('TPVG', False, 'Should NOT have net selling (from comparison group)'),
    ]
    
    print("="*80)
    print("TESTING NET SELLING CHECK")
    print("="*80)
    print()
    
    results = []
    for ticker, expected_net_selling, description in test_cases:
        print(f"Testing {ticker}: {description}")
        print(f"  Fetching data from openinsider.com...")
        
        has_net_selling, total_purchase_val, total_sale_val = check_net_selling(ticker)
        
        status = "✓ PASS" if has_net_selling == expected_net_selling else "✗ FAIL"
        results.append((ticker, has_net_selling, expected_net_selling, status))
        
        print(f"  Result: {status}")
        print(f"    Has net selling: {has_net_selling}")
        print(f"    Expected: {expected_net_selling}")
        print(f"    Total purchases: ${total_purchase_val:,.0f}")
        print(f"    Total sales: ${total_sale_val:,.0f}")
        if total_sale_val > 0 or total_purchase_val > 0:
            net = total_sale_val - total_purchase_val
            print(f"    Net: ${net:,.0f} ({'selling' if net > 0 else 'buying'})")
        print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    passed = sum(1 for _, _, _, status in results if "PASS" in status)
    total = len(results)
    
    for ticker, actual, expected, status in results:
        print(f"{status} {ticker}: actual={actual}, expected={expected}")
    
    print()
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = test_net_selling_check()
    sys.exit(exit_code)

