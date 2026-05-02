from services.momentum_trap import analyze_trap_risk, MarketRegion

# Test 1: NVDA (Strong fundamentals, moderate technicals)
result = analyze_trap_risk("NVDA", fscore=95, tscore=65, mscore=75, rfinal=89)
print(f"NVDA: {result['severity_level']} - {result['visual_indicator']}")
print(f"Warning: {result['warning']}\n")

# Test 2: Pump stock (Weak fundamentals, high technicals)
result = analyze_trap_risk("PUMP", fscore=30, tscore=85, mscore=90, rfinal=45)
print(f"PUMP: {result['severity_level']} - {result['visual_indicator']}")
print(f"Warning: {result['warning']}\n")
print(f"Recommendations:")
for rec in result['recommendations'][:3]:
    print(f"  {rec}")
