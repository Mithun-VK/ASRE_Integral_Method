"""
Example: Run Dynamic Backtest v3

This example shows how to run an institutional-grade backtest
with dynamic R_ASRE calculation and point-in-time data.
"""

from asre.data.fundamental_fetcher import FundamentalFetcher
from asre.data.point_in_time import PointInTimeData
from asre.backtest.engine_v3 import DynamicBacktestEngine
from asre.data import data_loader

def run_example():
    # 1. Fetch fundamental data
    print("Fetching fundamental data...")
    fetcher = FundamentalFetcher()
    fundamentals = fetcher.fetch_quarterly_fundamentals('NVDA', '2020-01-01', '2026-01-01')

    # 2. Fetch price data
    print("Fetching price data...")
    prices = data_loader.fetch_stock_data('NVDA', '2020-01-01', '2026-01-01')

    # 3. Run dynamic backtest
    print("Running backtest...")
    engine = DynamicBacktestEngine(
        initial_capital=100000,
        rebalance_frequency='Q',
        threshold_buy=70
    )

    results = engine.run_backtest(
        'NVDA',
        prices,
        fundamentals,
        '2020-01-01',
        '2026-01-01'
    )

    # 4. Display results
    print(f"\nTotal Return: {results['total_return']:.2f}%")
    print(f"CAGR: {results['cagr']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")

    # 5. Verify dynamic R_ASRE
    r_asre_std = results['r_asre_history']['r_asre'].std()
    print(f"\nR_ASRE Std Dev: {r_asre_std:.2f}")
    if r_asre_std > 0:
        print(" R_ASRE is DYNAMIC (not static)!")
    else:
        print(" WARNING: R_ASRE appears static!")

if __name__ == "__main__":
    run_example()
