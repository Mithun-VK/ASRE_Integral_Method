import requests
import json

# Test portfolio analysis
portfolio_data = {
    "holdings": [
        {"ticker": "NVDA", "shares": 100, "value": 14250.00},
        {"ticker": "MSFT", "shares": 50, "value": 21500.00},
        {"ticker": "GOOGL", "shares": 75, "value": 10500.00}
    ]
}

response = requests.post(
    "http://localhost:8000/api/portfolio/analyze",
    json=portfolio_data
)

if response.status_code == 200:
    result = response.json()
    print(" Portfolio Analysis:")
    print(f"  Health: {result['health_level']}")
    print(f"  Score: {result['overall_score']:.1f}/100")
    print(f"  Total Value: ${result['total_value']:,.2f}")
    print(f"\n  Top Recommendations:")
    for rec in result['recommendations'][:3]:
        print(f"    - {rec}")
else:
    print(f" Error: {response.status_code}")
    print(response.text)
