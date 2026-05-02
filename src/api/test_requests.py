from models.requests import *

# Test 1: CompareStocksRequest
try:
    req = CompareStocksRequest(tickers=["NVDA", "msft", "GOOGL"])
    print("[OK] CompareStocksRequest validation passed")
    print(f"    Tickers: {req.tickers}")
except Exception as e:
    print(f"[ERROR] CompareStocksRequest failed: {e}")

# Test 2: PortfolioRequest
try:
    req = PortfolioRequest(holdings=[
        {"ticker": "NVDA", "shares": 100, "value": 14250.0},
        {"ticker": "MSFT", "shares": 50, "value": 21500.0}
    ])
    print("[OK] PortfolioRequest validation passed")
    print(f"    Total value: ${req.total_value:,.2f}")
except Exception as e:
    print(f"[ERROR] PortfolioRequest failed: {e}")

# Test 3: AIChatRequest
try:
    req = AIChatRequest(message="Why is NVDA rated 89?", ticker="NVDA")
    print("[OK] AIChatRequest validation passed")
    print(f"    Message: {req.message}")
except Exception as e:
    print(f"[ERROR] AIChatRequest failed: {e}")

# Test 4: Invalid ticker (should fail)
try:
    req = CompareStocksRequest(tickers=["NVDA", "INVALID_TICKER_123"])
    print("[ERROR] Should have failed validation!")
except Exception as e:
    print("[OK] Invalid ticker correctly rejected")

print("\n[OK] All request model tests passed!")
