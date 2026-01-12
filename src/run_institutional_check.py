"""
Run Institutional Backtest Comparison
Compare Retail (Close Execution) vs Institutional (Next-Open Execution)
"""
import sys
import pandas as pd
import numpy as np
from asre.data_loader import DataLoader
from signal_engine import SignalEngine
from asre.backtest import compute_strategy_returns, generate_backtest_report
from asre_institutional import compute_institutional_returns, print_backtest_report
from asre.composite import compute_complete_asre

def run_comparison(ticker, start_date, end_date):
    print(f"\n{'='*80}")
    print(f"RUNNING REALISM CHECK FOR {ticker}")
    print(f"{'='*80}")

    # 1. Load Data
    loader = DataLoader()
    df = loader.load_stock_data(ticker, start_date, end_date)

    # 2. Compute Ratings
    df = compute_complete_asre(df, medallion=True)

    # 3. Generate Signals (Hybrid Engine)
    engine = SignalEngine(floor=55.0)
    df_signals = engine.generate_signals(df, rating_col='r_final')

    # ---------------------------------------------------------
    # RETAIL MODE (Old Way: Close-to-Close, Fixed Slippage)
    # ---------------------------------------------------------
    df_retail = compute_strategy_returns(
        df_signals.reset_index(),
        signal_col='signal',
        price_col='close',
        transaction_cost=0.001,
        slippage=0.0005
    )
    report_retail = generate_backtest_report(df_retail)

    # ---------------------------------------------------------
    # INSTITUTIONAL MODE (New Way: Next-Open, Dynamic Slippage)
    # ---------------------------------------------------------
    df_inst = compute_institutional_returns(
        df_signals.reset_index(),
        signal_col='signal',
        open_col='open',
        close_col='close',
        vol_col='vix' if 'vix' in df_signals.columns else None
    )
    report_inst = generate_backtest_report(df_inst)

    # ---------------------------------------------------------
    # PRINT COMPARISON
    # ---------------------------------------------------------
    print(f"\n{'METRIC':<20} | {'RETAIL (Fantasy)':<20} | {'INSTITUTIONAL (Real)':<20} | {'REALITY GAP':<15}")
    print("-" * 85)

    metrics = ['total_return', 'cagr', 'sharpe_ratio', 'max_drawdown', 'win_rate']

    for m in metrics:
        val_ret = report_retail.get(m, 0)
        val_inst = report_inst.get(m, 0)

        # Formatting
        if m in ['sharpe_ratio', 'profit_factor']:
            fmt = "{:.3f}"
        else:
            fmt = "{:.2%}"

        diff = val_inst - val_ret

        print(f"{m:<20} | {fmt.format(val_ret):<20} | {fmt.format(val_inst):<20} | {diff:+.3f}")

    print("-" * 85)
    print(f"\n✅ REALITY CHECK COMPLETE.")

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    start = "2024-01-01"
    end = "2025-12-31"
    run_comparison(ticker, start, end)
