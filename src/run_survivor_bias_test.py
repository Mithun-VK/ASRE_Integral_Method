"""
Run Survivor Bias Test
Tests the strategy against companies that went BANKRUPT or were REMOVED from S&P 500.

Goal:
1. Ensure the strategy does NOT blow up on SIVB/FRC.
2. Ensure the strategy sells before the collapse (if using Trailing Stop/Floor).
"""
import sys
import pandas as pd
from asre.data_loader import DataLoader
from signal_engine import SignalEngine
from asre_institutional import compute_institutional_returns, print_backtest_report
from asre.composite import compute_complete_asre
from survivor_bias_data import DEATH_LIST

def run_test():
    print(f"\n{'='*80}")
    print(f"💀 SURVIVOR BIAS STRESS TEST")
    print(f"{'='*80}")

    results = []

    for company in DEATH_LIST:
        ticker = company['ticker']
        fail_date = company['fail_date']
        print(f"\nTesting {ticker} ({company['status']} on {fail_date})...")

        try:
            # 1. Load Data (Handle missing data gracefully)
            # We assume data exists up to the failure point
            loader = DataLoader()
            # Try to fetch 1 year before failure
            start_date = str(int(fail_date[:4]) - 2) + "-01-01"
            end_date = fail_date

            try:
                df = loader.load_stock_data(ticker, start_date, end_date)
            except Exception as e:
                print(f"  ❌ Data load failed (Expected for delisted): {e}")
                results.append({'Ticker': ticker, 'Result': 'NO_DATA_FOUND', 'Return': 'N/A'})
                continue

            if len(df) < 50:
                print(f"  ❌ Insufficient data ({len(df)} rows)")
                results.append({'Ticker': ticker, 'Result': 'INSUFFICIENT_DATA', 'Return': 'N/A'})
                continue

            # 2. Compute Ratings
            df = compute_complete_asre(df, medallion=True)

            # 3. Generate Signals
            engine = SignalEngine(floor=55.0)
            df_signals = engine.generate_signals(df, rating_col='r_final')

            # 4. Institutional Backtest
            df_inst = compute_institutional_returns(
                df_signals.reset_index(),
                signal_col='signal',
                open_col='open',
                close_col='close'
            )

            # 5. Check Performance
            final_return = df_inst['cumulative_return'].iloc[-1] - 1
            max_dd = df_inst['drawdown'].min()

            # Did we survive?
            # If Return > -20% on a bankrupt stock, that's a HUGE win.
            print(f"  ✅ Test Complete: Return={final_return:.2%} | MaxDD={max_dd:.2%}")

            results.append({
                'Ticker': ticker, 
                'Status': company['status'],
                'Return': f"{final_return:.2%}",
                'MaxDD': f"{max_dd:.2%}",
                'Verdict': 'SURVIVED' if final_return > -0.20 else 'KILLED'
            })

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({'Ticker': ticker, 'Result': 'ERROR', 'Return': 'N/A'})

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY RESULTS")
    print(f"{'='*80}")
    print(f"{'Ticker':<10} | {'Status':<10} | {'Return':<10} | {'Verdict':<10}")
    print("-" * 50)
    for r in results:
        if 'Verdict' in r:
            print(f"{r['Ticker']:<10} | {r['Status']:<10} | {r['Return']:<10} | {r['Verdict']:<10}")
        else:
            print(f"{r['Ticker']:<10} | {r['Result']:<10} | {'N/A':<10} | N/A")

if __name__ == "__main__":
    run_test()
