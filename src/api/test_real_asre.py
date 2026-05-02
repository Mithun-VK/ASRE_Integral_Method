from services.asre_service import ASREService
from datetime import datetime, timedelta

print("=" * 80)
print("TESTING REAL ASRE INTEGRATION")
print("=" * 80)

# Check health
print("\n[1] Health Check:")
health = ASREService.health_check()
print(f"    ASRE Available: {health['asre_available']}")

# Test single stock (NVDA) WITH DATES
print("\n[2] Fetching NVDA rating (may take 30-60 seconds)...")
try:
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    rating = ASREService.get_stock_rating(
        "NVDA",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        force_refresh=True
    )

    print(f"\n SUCCESS!")
    print(f"    Ticker:     {rating['ticker']}")
    print(f"    R_Final:    {rating['rfinal']:.2f}")
    print(f"    Signal:     {rating['signal']}")
    print(f"    F-Score:    {rating['fscore']:.2f}")
    print(f"    T-Score:    {rating['tscore']:.2f}")
    print(f"    M-Score:    {rating['mscore']:.2f}")
    print(f"    Category:   {rating['category']}")
    print(f"    Dip Stage:  {rating['dip_stage']}")
    print(f"    Context:    {rating['context']}")
    print(f"    Timestamp:  {rating['timestamp']}")

    # Check if it's real data (not mock)
    if rating.get('_data_points', 0) > 100:
        print(f"\n REAL ASRE DATA ({rating['_data_points']} data points)")
    else:
        print(f"\n  MOCK DATA (ASRE may have failed)")

except Exception as e:
    print(f"\n ERROR: {e}")

# Test cache
print("\n[3] Testing cache (should be instant)...")
import time
start = time.time()
cached_rating = ASREService.get_stock_rating("NVDA")
elapsed = time.time() - start
print(f"    Cache fetch time: {elapsed:.3f}s")
if elapsed < 0.1:
    print(" Cache working!")
else:
    print("  Cache may not be working")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
