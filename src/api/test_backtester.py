from services.backtester import BacktesterService
import pandas as pd

# Create test CSV
test_csv = """date,ticker,action,quantity,price
2024-01-15,NVDA,buy,10,450.00
2024-02-20,NVDA,sell,10,720.00
2024-01-20,MSFT,buy,20,380.00
2024-03-15,MSFT,sell,20,420.00"""

backtester = BacktesterService()
results = backtester.run_backtest(test_csv, initial_capital=50000)

print("=" * 80)
print(" BACKTEST RESULTS")
print("=" * 80)
print(f"\nUser Performance:")
print(f"  Total Return: ${results['user_metrics']['total_return']:,.2f}")
print(f"  Return %: {results['user_metrics']['total_return_pct']:.2f}%")
print(f"  Win Rate: {results['user_metrics']['win_rate']:.2f}%")
print(f"  Sharpe Ratio: {results['user_metrics']['sharpe_ratio']:.2f}")

print(f"\nASRE Performance:")
print(f"  Total Return: ${results['asre_metrics']['total_return']:,.2f}")
print(f"  Return %: {results['asre_metrics']['total_return_pct']:.2f}%")
print(f"  Win Rate: {results['asre_metrics']['win_rate']:.2f}%")
print(f"  Sharpe Ratio: {results['asre_metrics']['sharpe_ratio']:.2f}")

print(f"\nComparison:")
print(f"  Winner: {results['comparison']['winner']}")
print(f"  Message: {results['comparison']['message']}")

print("\n[OK] Backtester works!")
