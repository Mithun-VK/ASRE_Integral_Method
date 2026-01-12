"""
Dynamic R_ASRE Backtest - HYBRID (Absolute Floor + Risk Management)

✅ NEW FEATURES:
1. Dynamic Thresholds (adapts to market conditions)
2. Custom Buy/Sell Floors (Buy>=65, Sell<=55)
3. Solves "Best of the Worst" problem (INTC case)
4. Uses SignalEngine for robust signal generation
5. ✅ NEW: 15% Stop Loss + 30% Take Profit (2:1 R/R)
6. ✅ NEW: 10% Trailing Stop
7. ✅ NEW: Trade Exit Reasons + P&L Tracking

Author: ASRE Project
Date: January 2026
"""

import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import argparse  # ✅ NEW

# Import your existing modules
from asre.data.fundamental_fetcher import FundamentalFetcher
from asre.data_loader import DataLoader
from asre.composite import compute_complete_asre
from asre.backtest import generate_backtest_report, print_backtest_report
from signal_engine import SignalEngine

# ✅ [Keep compute_strategy_returns UNCHANGED - Same as before]
def compute_strategy_returns(
    df: pd.DataFrame,
    signal_col: str = 'signal',
    price_col: str = 'close',
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    stop_loss_pct: float = 0.15,
    take_profit_pct: float = 0.30,
    trailing_stop_pct: float = 0.10,
) -> pd.DataFrame:
    """[Same implementation as before - no changes needed]"""
    result_df = df.copy()
    result_df['entry_signal'] = result_df[signal_col].shift(1).fillna(0)
    result_df['price_return'] = result_df[price_col].pct_change()
    result_df['position'] = 0.0
    result_df['entry_price'] = np.nan
    result_df['position_high'] = np.nan
    result_df['position_low'] = np.nan
    result_df['exit_reason'] = ''
    
    trade_groups = (result_df['entry_signal'].diff() != 0).cumsum()
    
    for group_id, group_df in result_df.groupby(trade_groups):
        if len(group_df) == 0:
            continue
            
        first_signal = group_df['entry_signal'].iloc[0]
        if first_signal == 0:
            continue
            
        entry_row = group_df.iloc[0]
        result_df.loc[entry_row.name, 'position'] = first_signal
        result_df.loc[entry_row.name, 'entry_price'] = entry_row[price_col]
        
        if first_signal > 0:
            result_df.loc[entry_row.name, 'position_high'] = entry_row[price_col]
        else:
            result_df.loc[entry_row.name, 'position_low'] = entry_row[price_col]
        
        for i in range(1, len(group_df)):
            row_idx = group_df.index[i]
            current_price = result_df.at[row_idx, price_col]
            
            if first_signal > 0:
                prev_high = result_df.at[group_df.index[i-1], 'position_high']
                result_df.at[row_idx, 'position_high'] = max(prev_high, current_price)
                pos_high = result_df.at[row_idx, 'position_high']
            else:
                prev_low = result_df.at[group_df.index[i-1], 'position_low']
                result_df.at[row_idx, 'position_low'] = min(prev_low, current_price)
                pos_low = result_df.at[row_idx, 'position_low']
            
            exit_triggered = False
            
            if first_signal > 0:
                sl_price = result_df.at[entry_row.name, 'entry_price'] * (1 - stop_loss_pct)
                tp_price = result_df.at[entry_row.name, 'entry_price'] * (1 + take_profit_pct)
                trailing_price = pos_high * (1 - trailing_stop_pct)
                
                if current_price <= sl_price:
                    result_df.at[row_idx, 'exit_reason'] = 'Stop_Loss'
                    exit_triggered = True
                elif current_price >= tp_price:
                    result_df.at[row_idx, 'exit_reason'] = 'Take_Profit'
                    exit_triggered = True
                elif current_price <= trailing_price:
                    result_df.at[row_idx, 'exit_reason'] = 'Trailing_Stop'
                    exit_triggered = True
            else:
                sl_price = result_df.at[entry_row.name, 'entry_price'] * (1 + stop_loss_pct)
                tp_price = result_df.at[entry_row.name, 'entry_price'] * (1 - take_profit_pct)
                trailing_price = pos_low * (1 + trailing_stop_pct)
                
                if current_price >= sl_price:
                    result_df.at[row_idx, 'exit_reason'] = 'Stop_Loss'
                    exit_triggered = True
                elif current_price <= tp_price:
                    result_df.at[row_idx, 'exit_reason'] = 'Take_Profit'
                    exit_triggered = True
                elif current_price >= trailing_price:
                    result_df.at[row_idx, 'exit_reason'] = 'Trailing_Stop'
                    exit_triggered = True
            
            if exit_triggered:
                result_df.at[row_idx, 'position'] = 0
                break
            else:
                result_df.at[row_idx, 'position'] = first_signal
    
    result_df['strategy_return'] = result_df['position'].shift(1) * result_df['price_return']
    entry_trades = result_df['entry_signal'].diff().abs() > 0
    exit_trades = (result_df['position'].diff().abs() > 0) & (result_df['position'] == 0)
    all_trades = entry_trades | exit_trades
    total_cost = transaction_cost + slippage
    result_df['transaction_cost_incurred'] = np.where(all_trades, total_cost, 0.0)
    result_df['net_return'] = result_df['strategy_return'] - result_df['transaction_cost_incurred']
    result_df['cumulative_return'] = (1 + result_df['net_return']).cumprod()
    cummax = result_df['cumulative_return'].cummax()
    result_df['drawdown'] = (result_df['cumulative_return'] - cummax) / cummax
    result_df['trade_pnl'] = np.where(exit_trades, result_df['cumulative_return'].diff(), 0)
    
    return result_df


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def fetch_all_data(ticker: str, start_date: str, end_date: str):
    """[Same as before - no changes]"""
    logger.info(f"\n{'='*80}")
    logger.info(f"FETCHING DATA FOR {ticker}")
    logger.info(f"{'='*80}")

    logger.info(f"\n📊 Step 1: Fetching Quarterly Fundamentals...")
    fetcher = FundamentalFetcher()
    try:
        df_fundamentals = fetcher.fetch_quarterly_fundamentals(ticker, start_date, end_date)
        logger.info(f"   ✅ Fetched {len(df_fundamentals)} quarters")
    except Exception as e:
        logger.error(f"   ❌ Error fetching quarterly fundamentals: {e}")
        df_fundamentals = None

    logger.info(f"\n📈 Step 2: Loading Price Data & Merging Fundamentals...")
    loader = DataLoader()
    try:
        df = loader.load_stock_data(ticker, start_date, end_date, quarterly_fundamentals=df_fundamentals)
        logger.info(f"   ✅ Loaded {len(df)} days of complete data")
        if 'pe' in df.columns:
            pe_std = df['pe'].std()
            if pe_std > 0.1:
                logger.info(f"   ✅ Fundamentals are TIME-VARYING (PE std={pe_std:.2f})")
    except Exception as e:
        logger.error(f"   ❌ Error loading data: {e}")
        raise

    logger.info(f"\n🎯 Step 3: Computing ASRE Ratings...")
    try:
        df_complete = compute_complete_asre(df, medallion=True, return_all_components=True)
        logger.info(f"   ✅ ASRE ratings computed")
        df_complete['date'] = pd.to_datetime(df_complete['date'])
        df_complete = df_complete.set_index('date')
    except Exception as e:
        logger.error(f"   ❌ Error computing ASRE: {e}")
        raise

    return df_complete


def run_hybrid_backtest_engine(df: pd.DataFrame, rating_col: str = 'r_final', 
                              buy_floor: float = 65.0, sell_floor: float = 55.0):  # ✅ NEW
    """
    Run backtest using SignalEngine with CUSTOM Buy/Sell thresholds.
    
    Args:
        buy_floor: Buy when R_final >= buy_floor (DEFAULT: 65)
        sell_floor: Sell when R_final <= sell_floor (DEFAULT: 55)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"HYBRID BACKTEST ENGINE (BUY>={buy_floor}, SELL<={sell_floor})")  # ✅ NEW
    logger.info(f"{'='*80}")

    # ✅ Initialize Signal Engine with BOTH thresholds
    engine = SignalEngine(
        floor=buy_floor,         # ✅ NEW: Buy threshold
        sell_threshold=sell_floor,  # ✅ NEW: Sell threshold
        sensitivity=0.5, 
        window=63
    )

    df_signals = engine.generate_signals(df, rating_col=rating_col)
    df_reset = df_signals.reset_index()

    df_results = compute_strategy_returns(
        df_reset,
        signal_col='signal',
        price_col='close',
        transaction_cost=0.001,
        slippage=0.0005,
        stop_loss_pct=0.15,
        take_profit_pct=0.30,
        trailing_stop_pct=0.10
    )

    report = generate_backtest_report(df_results)
    return df_results, report


def main():
    # ✅ NEW: Argument parser for custom floors
    parser = argparse.ArgumentParser(description="Hybrid R_ASRE Backtest with Custom Floors")
    parser.add_argument("ticker", help="Stock ticker (e.g., AAPL, RELIANCE.NS)")
    parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date (YYYY-MM-DD)")
    parser.add_argument("rating_col", nargs='?', default='r_final', help="Rating column (default: r_final)")
    parser.add_argument("--buy", type=float, default=65.0, help="Buy floor (default: 65)")
    parser.add_argument("--sell", type=float, default=55.0, help="Sell floor (default: 55)")
    
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"HYBRID R_ASRE BACKTEST (BUY>={args.buy}, SELL<={args.sell})")  # ✅ NEW
    print(f"{'='*80}")

    try:
        df = fetch_all_data(args.ticker, args.start_date, args.end_date)
        
        df_hybrid, report_hybrid = run_hybrid_backtest_engine(
            df, 
            rating_col=args.rating_col,
            buy_floor=args.buy,   # ✅ NEW
            sell_floor=args.sell  # ✅ NEW
        )

        print_backtest_report(report_hybrid, f"Hybrid Strategy ({args.ticker})")
        print(f"\n✅ BACKTEST COMPLETE! (Buy>={args.buy}, Sell<={args.sell})")

    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
