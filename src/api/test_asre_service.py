"""
Test ASRE Service
Validates all core functionality
"""
from services.asre_service import ASREService
import time

print("=" * 80)
print("ASRE SERVICE TEST SUITE")
print("=" * 80)

# Test 1: Health Check
print("\n[TEST 1] Health Check")
health = ASREService.health_check()
print(f"  ASRE Available: {health['asre_available']}")
print(f"  Supported Stocks: {health['supported_stocks_count']}")
print(f"  Cache Size: {health['cache_size']}")

# Test 2: Single Stock Rating
print("\n[TEST 2] Get Single Stock Rating (NVDA)")
print("  Processing... (may take 30-60 seconds)")
start = time.time()
try:
    rating = ASREService.get_stock_rating("NVDA")
    elapsed = time.time() - start
    print(f"  [OK] NVDA Rating: {rating['rfinal']:.1f}")
    print(f"  Signal: {rating['signal']}")
    print(f"  F-Score: {rating['fscore']:.1f}")
    print(f"  Category: {rating['category']}")
    print(f"  Time: {elapsed:.2f}s")
except Exception as e:
    print(f"  [ERROR] {e}")

# Test 3: Cache Hit
print("\n[TEST 3] Cache Performance")
start = time.time()
try:
    rating = ASREService.get_stock_rating("NVDA")  # Should be cached
    elapsed = time.time() - start
    print(f"  [OK] Cached fetch: {elapsed:.3f}s (should be < 0.1s)")
    if elapsed < 0.1:
        print("  [OK] Cache working correctly!")
    else:
        print("  [WARN] Cache may not be working")
except Exception as e:
    print(f"  [ERROR] {e}")

# Test 4: Multiple Stocks
print("\n[TEST 4] Batch Processing (3 stocks)")
try:
    results = ASREService.get_multiple_stocks(["MSFT", "GOOGL", "META"])
    print(f"  [OK] Processed {len(results)} stocks")
    for r in results:
        if r.get('rfinal'):
            print(f"    {r['ticker']}: {r['rfinal']:.1f} ({r['signal']})")
        else:
            print(f"    {r['ticker']}: ERROR - {r.get('error')}")
except Exception as e:
    print(f"  [ERROR] {e}")

# Test 5: Stock Comparison
print("\n[TEST 5] Stock Comparison & Ranking")
try:
    comparison = ASREService.compare_stocks(["NVDA", "MSFT", "GOOGL"])
    print(f"  [OK] Ranked {len(comparison)} stocks:")
    for stock in comparison[:5]:  # Top 5
        print(f"    #{stock['rank']}: {stock['ticker']} - {stock['rfinal']:.1f}")
except Exception as e:
    print(f"  [ERROR] {e}")

# Test 6: Invalid Ticker
print("\n[TEST 6] Error Handling (Invalid Ticker)")
try:
    ASREService.get_stock_rating("INVALID123")
    print("  [ERROR] Should have raised exception!")
except ValueError as e:
    print(f"  [OK] Correctly rejected: {e}")

# Test 7: Cache Stats
print("\n[TEST 7] Cache Statistics")
stats = ASREService.get_cache_stats()
print(f"  Cache Size: {stats['size']}")
print(f"  Cached Tickers: {', '.join(stats['tickers'][:5])}")

print("\n" + "=" * 80)
print("[OK] ALL TESTS COMPLETED")
print("=" * 80)
