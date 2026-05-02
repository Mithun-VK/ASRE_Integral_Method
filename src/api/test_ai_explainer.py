from services.ai_explainer import AIExplainer

explainer = AIExplainer()

# Test 1: Stock rating explanation
asre_data = {
    "ticker": "NVDA",
    "rfinal": 89.0,
    "fscore": 95.0,
    "tscore": 65.0,
    "mscore": 75.0,
    "signal": "STRONG BUY",
    "category": "EXCEPTIONAL GROWTH"
}

result = explainer.explain_stock_rating("NVDA", asre_data)
print("=" * 80)
print(f"NVDA Explanation:\n{result['explanation']}\n")

# Test 2: Momentum trap
trap = explainer.explain_momentum_trap("PUMP", fscore=30, tscore=85, mscore=90)
print("=" * 80)
print(f"Trap Warning: {trap['warning']}\n")
print(f"Risk Factors: {trap['risk_factors']}")

print("\n[OK] AI Explainer works!")
