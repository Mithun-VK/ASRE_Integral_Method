from services.portfolio_analyzer import PortfolioAnalyzer

analyzer = PortfolioAnalyzer()

# Test portfolio
holdings = [
    {"ticker": "NVDA", "shares": 100, "value": 14250.00},
    {"ticker": "MSFT", "shares": 50, "value": 21500.00},
    {"ticker": "GOOGL", "shares": 75, "value": 10500.00},
    {"ticker": "META", "shares": 40, "value": 8000.00},
]

result = analyzer.analyze_portfolio(holdings)

print("=" * 80)
print(f"Portfolio Health: {result['health_level']}")
print(f"Overall Score: {result['overall_score']:.1f}/100")
print(f"Total Value: ${result['total_value']:,.2f}\n")

print("Risk Breakdown:")
for category, data in result['risk_breakdown'].items():
    print(f"  {category.upper()}: {data['percentage']:.1f}% (${data['value']:,.2f})")

print(f"\nHigh-Risk Stocks: {len(result['high_risk_stocks'])}")

print(f"\nTop Recommendations:")
for rec in result['recommendations'][:5]:
    print(f"  {rec}")

print("\n[OK] Portfolio Analyzer works!")
