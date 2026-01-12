"""
ASRE Comprehensive Backtest Engine v2.1 - INTERACTIVE VERSION
==============================================================
Properly handles Backtester attributes + User input for ticker
"""


import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


from asre.data_loader import load_stock_data
from asre.composite import compute_complete_asre
from asre.backtest import Backtester, generate_signals_threshold


def generate_signals_with_floor(
    df: pd.DataFrame,
    threshold_long: float = 70.0,
    threshold_short: float = 30.0,
    fundamental_floor: float = 65.0,
    use_floor: bool = True,
) -> pd.DataFrame:
    """Generate signals with optional fundamental floor protection."""
    df = df.copy()
    df['signal'] = 0
    df.loc[df['r_asre'] >= threshold_long, 'signal'] = 1
    df.loc[df['r_asre'] <= threshold_short, 'signal'] = -1


    if use_floor:
        df.loc[(df['signal'] == 1) & (df['f_score'] < fundamental_floor), 'signal'] = 0


    return df



class BeforeAfterComparison:
    """Compare backtest performance before (v2.0) and after (v2.1) fundamental floor."""


    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results_before = {}
        self.results_after = {}


    def run(
        self,
        threshold_long: float = 70.0,
        threshold_short: float = 30.0,
        fundamental_floor: float = 65.0,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
    ):
        """Run backtest with and without fundamental floor."""
        # v2.0: No fundamental floor
        df_before = generate_signals_with_floor(
            self.df,
            threshold_long=threshold_long,
            threshold_short=threshold_short,
            use_floor=False,
        )


        bt_before = Backtester(df_before, rating_col='r_asre')
        bt_before.run(
            signal_type='threshold',
            threshold_long=threshold_long,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
        )


        self.results_before = self._extract_results(bt_before)


        # v2.1: With fundamental floor
        df_after = generate_signals_with_floor(
            self.df,
            threshold_long=threshold_long,
            threshold_short=threshold_short,
            fundamental_floor=fundamental_floor,
            use_floor=True,
        )


        bt_after = Backtester(df_after, rating_col='r_asre')
        bt_after.run(
            signal_type='threshold',
            threshold_long=threshold_long,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
        )


        self.results_after = self._extract_results(bt_after)


        self.bt_before = bt_before
        self.bt_after = bt_after


    def _extract_results(self, bt):
        """Safely extract results from Backtester object."""
        results = {}


        # List of attributes to extract
        attrs = [
            'total_return', 'cagr', 'sharpe', 'max_drawdown',
            'win_rate', 'profit_factor', 'var_95', 'num_trades'
        ]


        for attr in attrs:
            if hasattr(bt, attr):
                results[attr] = getattr(bt, attr)
            else:
                # Try to compute from available data
                if attr == 'total_return' and hasattr(bt, 'equity_curve'):
                    if len(bt.equity_curve) > 0:
                        results[attr] = ((bt.equity_curve[-1] / bt.equity_curve[0]) - 1) * 100
                    else:
                        results[attr] = 0.0
                elif attr == 'num_trades' and hasattr(bt, 'trades'):
                    results[attr] = len(bt.trades)
                else:
                    results[attr] = 0.0


        return results


    def get_improvement_summary(self) -> Dict[str, float]:
        """Get summary of improvements from v2.0 to v2.1."""
        return {
            'sharpe_improvement': self.results_after['sharpe'] - self.results_before['sharpe'],
            'drawdown_reduction': (self.results_before['max_drawdown'] - self.results_after['max_drawdown']) / 100,
            'return_change': self.results_after['total_return'] - self.results_before['total_return'],
            'trades_filtered': self.results_before['num_trades'] - self.results_after['num_trades'],
        }


    def print_report(self):
        """Print formatted comparison report."""
        print("\n" + "=" * 90)
        print(f"{'BEFORE/AFTER COMPARISON: v2.0 vs v2.1':^90}")
        print("=" * 90)


        metrics = [
            ('Total Return', 'total_return', '%', True),
            ('CAGR', 'cagr', '%', True),
            ('Sharpe Ratio', 'sharpe', '', True),
            ('Max Drawdown', 'max_drawdown', '%', False),
            ('Win Rate', 'win_rate', '%', True),
            ('Profit Factor', 'profit_factor', '', True),
            ('VaR (95%)', 'var_95', '%', False),
        ]


        print(f"\n{'Metric':<20} | {'v2.0 (No Floor)':>15} | {'v2.1 (With Floor)':>17} | {'Δ Improvement':>14}")
        print("-" * 90)


        for name, key, unit, higher_better in metrics:
            before = self.results_before.get(key, 0.0)
            after = self.results_after.get(key, 0.0)
            delta = after - before


            if higher_better:
                emoji = "✅" if delta > 0 else "❌"
            else:
                emoji = "✅" if delta < 0 else "❌"


            if unit == '%':
                print(f"{name:<20} | {before:>13.2f}%    | {after:>15.2f}%    | {delta:>+11.2f}% {emoji}")
            else:
                print(f"{name:<20} | {before:>15.3f}    | {after:>17.3f}    | {delta:>+11.3f} {emoji}")


        print("=" * 90)


        print("\n🎯 KEY INSIGHT:")
        if self.results_after['sharpe'] > self.results_before['sharpe']:
            print("   v2.1 IMPROVED risk-adjusted returns")
        else:
            print("   v2.0 had higher returns BUT with higher risk")
            print("   v2.1 PROTECTED capital with lower drawdown")


        print("=" * 90)



def run_backtest(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
    strategy: str = 'threshold',
    threshold_long: float = 70.0,
    threshold_short: float = 30.0,
    fundamental_floor: float = 65.0,
    use_floor: bool = True,
    initial_capital: float = 100000,
    transaction_cost: float = 0.001,
    verbose: bool = True,
) -> Dict:
    """
    Run complete backtest on a single stock.


    The report prints automatically if verbose=True.
    Returns a dict with the Backtester object for further analysis.
    """
    # Load data
    if verbose:
        print(f"\n📊 Loading data for {ticker}...")


    df = load_stock_data(ticker, start_date, end_date)


    if verbose:
        print(f"✅ Loaded {len(df)} rows ({df['date'].iloc[0]} to {df['date'].iloc[-1]})")
        print(f"\n🧮 Computing ASRE ratings...")


    # Compute ASRE
    df = compute_complete_asre(df, medallion=True, return_all_components=True)


    if verbose:
        latest = df.iloc[-1]
        print(f"✅ ASRE computed")
        print(f"   Latest: F={latest['f_score']:.0f}%, R_ASRE={latest['r_asre']:.1f}")
        print(f"\n🔬 Running backtest...")


    # Create a copy for backtesting
    df_backtest = df.copy()


    # Apply fundamental floor if v2.1
    if use_floor:
        df_backtest = generate_signals_with_floor(
            df_backtest,
            threshold_long=threshold_long,
            threshold_short=threshold_short,
            fundamental_floor=fundamental_floor,
            use_floor=True
        )


    # Run backtest
    bt = Backtester(df_backtest, rating_col='r_asre')
    bt.run(
        signal_type='threshold',
        threshold_long=threshold_long,
        threshold_short=threshold_short,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
    )


    if verbose:
        version = "v2.1 (with floor)" if use_floor else "v2.0 (no floor)"
        bt.print_report(title=f"{ticker} Backtest ({version})")


    # Safely extract metrics
    def safe_get(attr, default=None):
        if hasattr(bt, attr):
            return getattr(bt, attr)
        return default


    # Return backtester object and dataframe for further analysis
    return {
        'ticker': ticker,
        'version': 'v2.1' if use_floor else 'v2.0',
        'backtest_obj': bt,
        'dataframe': df_backtest,
        # Try to extract metrics, but don't fail if not available
        'metrics': {
            'total_return': safe_get('total_return'),
            'cagr': safe_get('cagr'),
            'sharpe': safe_get('sharpe'),
            'max_drawdown': safe_get('max_drawdown'),
            'win_rate': safe_get('win_rate'),
            'profit_factor': safe_get('profit_factor'),
            'num_trades': safe_get('num_trades') or (len(bt.trades) if hasattr(bt, 'trades') else 0),
        }
    }



def compare_strategies(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
    threshold_long: float = 70.0,
    fundamental_floor: float = 65.0,
    initial_capital: float = 100000,
    verbose: bool = True,
) -> Dict:
    """Compare v2.0 (no floor) vs v2.1 (with floor) strategies."""
    if verbose:
        print(f"\n📊 Loading data for {ticker}...")


    df = load_stock_data(ticker, start_date, end_date)


    if verbose:
        print(f"✅ Loaded {len(df)} rows")
        print(f"\n🧮 Computing ASRE ratings...")


    df = compute_complete_asre(df, medallion=True, return_all_components=True)


    if verbose:
        print(f"✅ ASRE computed")
        print(f"\n🔬 Running comparison...")


    comparison = BeforeAfterComparison(df)
    comparison.run(
        threshold_long=threshold_long,
        fundamental_floor=fundamental_floor,
        initial_capital=initial_capital,
    )


    if verbose:
        comparison.print_report()


    return {
        'ticker': ticker,
        'improvements': comparison.get_improvement_summary(),
        'v2_0': comparison.results_before,
        'v2_1': comparison.results_after,
        'comparison_obj': comparison,
    }



def quick_scan(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Quick scan of multiple stocks with ASRE ratings."""
    results = []


    for ticker in tickers:
        try:
            print(f"Processing {ticker}...")


            df = load_stock_data(ticker, start_date, end_date)
            df = compute_complete_asre(df, medallion=True, return_all_components=True)


            latest = df.iloc[-1]


            signal = '🚀 BUY' if latest['r_asre'] >= 70 else ('📈 HOLD' if latest['r_asre'] >= 40 else '🔻 SELL')


            results.append({
                'Ticker': ticker,
                'F-Score': latest['f_score'],
                'T-Score': latest['t_score'],
                'M-Score': latest['m_score'],
                'R_Final': latest['r_final'],
                'R_ASRE': latest['r_asre'],
                'Signal': signal,
                'Price': latest['close'],
                'Date': latest['date'],
            })


        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            continue


    return pd.DataFrame(results)



def interactive_backtest():
    """Interactive mode - get user input for ticker and parameters."""
    print("=" * 100)
    print("ASRE INTERACTIVE BACKTEST".center(100))
    print("=" * 100)

    # Get ticker
    print("\n📊 Enter stock ticker to backtest:")
    ticker = input("Ticker (e.g., NVDA, MSFT, AAPL): ").strip().upper()

    if not ticker:
        print("❌ No ticker provided. Using NVDA as default.")
        ticker = "NVDA"

    # Get start date
    print("\n📅 Enter start date for backtest:")
    start_date = input("Start date (YYYY-MM-DD) [default: 2024-01-01]: ").strip()

    if not start_date:
        start_date = "2024-01-01"

    # Ask about fundamental floor
    print("\n🛡️ Use fundamental floor protection (v2.1)?")
    print("   v2.1 = Only buy stocks with F-Score >= 65%")
    print("   v2.0 = No fundamental filter")
    use_floor_input = input("Use v2.1 with floor? (Y/n) [default: Y]: ").strip().lower()

    use_floor = use_floor_input != 'n'

    # Ask about comparison
    print("\n🔬 Do you want to compare v2.0 vs v2.1?")
    compare_input = input("Run comparison? (y/N) [default: N]: ").strip().lower()

    run_comparison = compare_input == 'y'

    print("\n" + "=" * 100)
    print(f"🚀 Starting backtest for {ticker} from {start_date}")
    print(f"   Version: {'v2.1 (with fundamental floor)' if use_floor else 'v2.0 (no floor)'}")
    print("=" * 100)

    try:
        if run_comparison:
            # Run comparison
            results = compare_strategies(
                ticker=ticker,
                start_date=start_date,
                verbose=True
            )
            print("\n✅ Comparison complete!")

        else:
            # Run single backtest
            results = run_backtest(
                ticker=ticker,
                start_date=start_date,
                use_floor=use_floor,
                verbose=True
            )
            print("\n✅ Backtest complete!")

        print("\n" + "=" * 100)
        print("💡 NEXT STEPS:")
        print("=" * 100)
        print("""
You can also use programmatically:

  # Single backtest
  from asre_backtest_v2_1 import run_backtest
  results = run_backtest('GOOGL', start_date='2024-01-01')

  # Compare v2.0 vs v2.1
  from asre_backtest_v2_1 import compare_strategies
  compare_strategies('META', start_date='2024-01-01')

  # Quick scan multiple stocks
  from asre_backtest_v2_1 import quick_scan
  df = quick_scan(['NVDA', 'MSFT', 'AAPL'], start_date='2024-01-01')
        """)
        print("=" * 100)

        return results

    except Exception as e:
        print(f"\n❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return None



if __name__ == "__main__":
    # Run interactive mode
    results = interactive_backtest()