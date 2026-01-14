"""
Investment-Focused ASRE Backtest - PURE INVESTING MODE
============================================================
✅ PHILOSOPHY: Exact Allocation Tracking (Invest fully, no "noise filter").
✅ GOAL: Replicate the 'Signal' behavior but with real dollars.
✅ CASH: Assumes fractional shares allowed (fully invested).
"""

import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import argparse

# Import ASRE modules
from asre.data.fundamental_fetcher import FundamentalFetcher
from asre.data_loader import DataLoader
from asre.composite import compute_complete_asre
from asre.backtest import generate_backtest_report

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ===========================================================================
# INVESTMENT SIGNAL ENGINE (UNCHANGED)
# ===========================================================================

class InvestmentSignalEngine:
    def __init__(self, 
                 overweight_threshold: float = 75.0,
                 full_threshold: float = 60.0,
                 reduce_threshold: float = 45.0,
                 exit_threshold: float = 30.0,
                 rebalance_period: str = 'W'): 
        self.overweight = overweight_threshold
        self.full = full_threshold
        self.reduce = reduce_threshold
        self.exit = exit_threshold
        self.rebalance_period = rebalance_period

    def generate_signals(self, df: pd.DataFrame, rating_col: str = 'r_asre') -> pd.DataFrame:
        """Generate tiered allocation signals (FIXED: merge_asof)."""
        df = df.copy()

        resample_rule = self.rebalance_period.replace('M', 'ME')
        df_resampled = df.resample(resample_rule).last()

        def get_allocation(rating):
            if pd.isna(rating): return 0.0
            elif rating >= self.overweight: return 1.20
            elif rating >= self.full: return 1.00
            elif rating >= self.reduce: return 0.50
            elif rating >= self.exit: return 0.25
            else: return 0.00

        df_resampled['target_allocation'] = df_resampled[rating_col].apply(get_allocation)

        df_resampled = df_resampled.sort_index()
        temp_alloc = df_resampled[['target_allocation']].reset_index()
        temp_alloc.columns = ['rebalance_date', 'target_allocation']

        df_reset = df.reset_index()

        df_merged = pd.merge_asof(
            df_reset, 
            temp_alloc, 
            left_on='date', 
            right_on='rebalance_date', 
            direction='backward'
        )

        df_merged.index = df.index
        df['allocation'] = df_merged['target_allocation'].fillna(method='ffill').fillna(0.0)

        df['prev_allocation'] = df['allocation'].shift(1).fillna(0.0)
        df['is_rebalance_date'] = df['allocation'] != df['prev_allocation']

        if df['allocation'].iloc[0] > 0:
            df.iloc[0, df.columns.get_loc('is_rebalance_date')] = True

        return df

# ===========================================================================
# INVESTMENT POSITION MANAGER (PURE INVESTING MODE)
# ===========================================================================

def compute_investment_returns(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,
    allocation_col: str = 'allocation',
    price_col: str = 'close',
    transaction_cost: float = 0.0005,
    stop_loss_pct: float = 0.25,
    trailing_stop_pct: float = 0.20,
    ltcg_tax: float = 0.10,
    stcg_tax: float = 0.15,
) -> pd.DataFrame:

    result_df = df.copy()
    result_df['position'] = 0.0
    result_df['entry_price'] = np.nan
    result_df['entry_date'] = pd.NaT
    result_df['position_high'] = np.nan
    result_df['exit_reason'] = ''
    result_df['trade_pnl'] = 0.0
    result_df['days_held'] = 0
    result_df['tax_applied'] = 0.0

    result_df['portfolio_value'] = initial_capital
    result_df['cash'] = initial_capital
    result_df['shares'] = 0.0

    trade_log = []

    # State variables
    current_cash = initial_capital
    current_shares = 0.0

    entry_price = np.nan
    entry_date = None
    position_high = np.nan

    for idx, row in result_df.iterrows():
        target_allocation = row[allocation_col]
        current_price = row[price_col]

        # Mark to market
        current_position_val = current_shares * current_price
        total_portfolio_val = current_cash + current_position_val

        # Track High
        if pd.isna(position_high) and current_shares > 0:
            position_high = current_price
        elif current_shares > 0:
            position_high = max(position_high, current_price)

        exit_triggered = False
        exit_reason = ''

        # --- RISK MANAGEMENT ---
        if current_shares > 0:
            sl_price = entry_price * (1 - stop_loss_pct)
            if current_price <= sl_price:
                exit_triggered = True
                exit_reason = 'Stop_Loss'
            elif not pd.isna(position_high):
                trailing_price = position_high * (1 - trailing_stop_pct)
                if current_price <= trailing_price:
                    exit_triggered = True
                    exit_reason = 'Trailing_Stop'

        # --- FORCED EXIT ---
        if exit_triggered:
            sale_proceeds = current_shares * current_price

            days_held = (idx - entry_date).days if entry_date else 0
            tax_rate = stcg_tax if days_held < 365 else ltcg_tax

            raw_profit = sale_proceeds - (current_shares * entry_price)
            tax_amount = max(0, raw_profit * tax_rate)
            net_proceeds = sale_proceeds - tax_amount
            net_pnl_pct = (net_proceeds - (current_shares * entry_price)) / (current_shares * entry_price)

            current_cash += net_proceeds
            current_shares = 0.0

            result_df.at[idx, 'trade_pnl'] = net_pnl_pct
            result_df.at[idx, 'exit_reason'] = exit_reason
            result_df.at[idx, 'days_held'] = days_held
            result_df.at[idx, 'tax_applied'] = tax_amount

            trade_log.append({
                'Date': idx.strftime('%Y-%m-%d'),
                'Action': 'SELL (Risk Exit)',
                'Reason': exit_reason,
                'Price': f"{current_price:.2f}",
                'Size': '0%',
                'PnL': f"${raw_profit - tax_amount:.2f} ({net_pnl_pct*100:.1f}%)",
                'Portfolio': f"${current_cash:.0f}"
            })

            entry_price = np.nan
            entry_date = None
            position_high = np.nan

            # Recalculate portfolio value after crash exit
            total_portfolio_val = current_cash 

        # --- REBALANCING LOGIC (PURE INVESTING - NO THRESHOLD FILTER) ---
        if row.get('is_rebalance_date', False) or exit_triggered:

            # Calculate EXACT target value
            target_val = total_portfolio_val * target_allocation
            current_holdings_val = current_shares * current_price

            val_diff = target_val - current_holdings_val

            # Allow even small adjustments (Invest fully)
            # Only skip if diff is essentially zero (< $1)
            if abs(val_diff) > 1.0:

                # BUYING
                if val_diff > 0:
                    cost = val_diff * (1 + transaction_cost)
                    # Allow using margin/cash if available (Investing usually cash only, but strategy allows 120%)
                    # If we need more than cash, we are borrowing (Implicit Margin)
                    # For simple "Investing", we cap at Cash? 
                    # NO, strategy says 120% "Overweight". This implies Margin.
                    # We will allow "Simulated Margin" by letting cash go negative?
                    # OR, we cap at 100% Cash?
                    # Standard investing = Cap at Cash. 
                    # If target is 120%, we buy up to max cash.

                    max_buy_val = current_cash / (1 + transaction_cost)
                    actual_buy_val = min(val_diff, max_buy_val)

                    if actual_buy_val > 0:
                        shares_to_buy = actual_buy_val / current_price
                        current_shares += shares_to_buy
                        current_cash -= actual_buy_val * (1 + transaction_cost)

                        if pd.isna(entry_price):
                            entry_price = current_price
                            entry_date = idx
                            position_high = current_price

                        trade_log.append({
                            'Date': idx.strftime('%Y-%m-%d'),
                            'Action': f"BUY (+{target_allocation*100:.0f}%)",
                            'Reason': 'Entry/Increase',
                            'Price': f"{current_price:.2f}",
                            'Size': f"{target_allocation*100:.0f}%",
                            'PnL': '-',
                            'Portfolio': f"${current_cash + (current_shares*current_price):.0f}"
                        })

                # SELLING
                elif val_diff < 0:
                    shares_to_sell = abs(val_diff) / current_price
                    shares_to_sell = min(shares_to_sell, current_shares)

                    if shares_to_sell > 0:
                        sale_proceeds = shares_to_sell * current_price

                        days_held = (idx - entry_date).days if entry_date else 0
                        tax_rate = stcg_tax if days_held < 365 else ltcg_tax

                        raw_profit = (current_price - entry_price) * shares_to_sell
                        tax_amount = max(0, raw_profit * tax_rate)

                        net_proceeds = sale_proceeds - tax_amount - (sale_proceeds * transaction_cost)

                        current_cash += net_proceeds
                        current_shares -= shares_to_sell

                        action_str = "SELL (Full)" if current_shares < 0.001 else f"SELL (To {target_allocation*100:.0f}%)"
                        reason_str = "Signal < Exit" if target_allocation == 0 else "Reduce Exposure"

                        trade_log.append({
                            'Date': idx.strftime('%Y-%m-%d'),
                            'Action': action_str,
                            'Reason': reason_str,
                            'Price': f"{current_price:.2f}",
                            'Size': f"{target_allocation*100:.0f}%",
                            'PnL': f"${raw_profit - tax_amount:.2f}",
                            'Portfolio': f"${current_cash + (current_shares*current_price):.0f}"
                        })

                        if current_shares < 0.001:
                            entry_price = np.nan
                            entry_date = None
                            position_high = np.nan
                            current_shares = 0

        # Update row
        result_df.at[idx, 'portfolio_value'] = current_cash + (current_shares * current_price)
        result_df.at[idx, 'cash'] = current_cash
        result_df.at[idx, 'shares'] = current_shares
        result_df.at[idx, 'position'] = (current_shares * current_price) / result_df.at[idx, 'portfolio_value']

        result_df.at[idx, 'entry_price'] = entry_price
        result_df.at[idx, 'entry_date'] = entry_date
        result_df.at[idx, 'position_high'] = position_high

    result_df['net_return'] = result_df['portfolio_value'].pct_change()
    result_df['cumulative_return'] = result_df['portfolio_value'] / initial_capital

    cummax = result_df['cumulative_return'].cummax()
    result_df['drawdown'] = (result_df['cumulative_return'] - cummax) / cummax

    return result_df, trade_log

# ===========================================================================
# DATA FETCHING (UNCHANGED)
# ===========================================================================

def fetch_all_data(ticker: str, start_date: str, end_date: str):
    logger.info(f"\n{'='*80}")
    logger.info(f"FETCHING DATA FOR {ticker}")
    logger.info(f"{'='*80}")

    logger.info(f"\n📊 Step 1: Fetching Quarterly Fundamentals...")
    fetcher = FundamentalFetcher()
    try:
        df_fundamentals = fetcher.fetch_quarterly_fundamentals(ticker, start_date, end_date)
        logger.info(f"   ✅ Fetched {len(df_fundamentals)} quarters")
    except Exception as e:
        logger.error(f"   ❌ Error fetching fundamentals: {e}")
        df_fundamentals = None

    logger.info(f"\n📈 Step 2: Loading Price Data & Merging Fundamentals...")
    loader = DataLoader()
    try:
        df = loader.load_stock_data(ticker, start_date, end_date, quarterly_fundamentals=df_fundamentals)
        logger.info(f"   ✅ Loaded {len(df)} days of data")
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

# ===========================================================================
# INVESTMENT BACKTEST ENGINE (UPDATED)
# ===========================================================================

def run_investment_backtest(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,
    rating_col: str = 'r_asre',
    overweight_threshold: float = 75.0,
    full_threshold: float = 60.0,
    reduce_threshold: float = 45.0,
    exit_threshold: float = 30.0,
    rebalance_period: str = 'W',
):
    logger.info(f"\n{'='*80}")
    logger.info(f"INVESTMENT BACKTEST ENGINE ($10,000 Start)")
    logger.info(f"Rating: {rating_col} | Rebalance: {rebalance_period}")
    logger.info(f"Tiers: Exit<{exit_threshold}, Reduce<{reduce_threshold}, Full<{full_threshold}, OW>={overweight_threshold}")
    logger.info(f"{'='*80}")

    engine = InvestmentSignalEngine(
        overweight_threshold=overweight_threshold,
        full_threshold=full_threshold,
        reduce_threshold=reduce_threshold,
        exit_threshold=exit_threshold,
        rebalance_period=rebalance_period
    )

    df_signals = engine.generate_signals(df, rating_col=rating_col)

    df_results, trade_log = compute_investment_returns(
        df_signals,
        initial_capital=initial_capital,
        allocation_col='allocation',
        price_col='close',
        transaction_cost=0.0005,
        stop_loss_pct=0.25,
        trailing_stop_pct=0.20,
        ltcg_tax=0.10,
        stcg_tax=0.15
    )

    report = generate_investment_report(df_results, initial_capital)
    return df_results, report, trade_log

# ===========================================================================
# REPORT GENERATION (UPDATED FOR DOLLAR VALUES)
# ===========================================================================

def generate_investment_report(df: pd.DataFrame, initial_capital: float) -> dict:
    final_val = df['portfolio_value'].iloc[-1]
    total_return = ((final_val - initial_capital) / initial_capital) * 100

    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = ((final_val / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    volatility = df['net_return'].std() * np.sqrt(252) * 100
    sharpe = (cagr / volatility) if volatility > 0 else 0
    max_dd = df['drawdown'].min() * 100
    avg_dd = df[df['drawdown'] < 0]['drawdown'].mean() * 100 if (df['drawdown'] < 0).any() else 0

    trades = df[df['trade_pnl'] != 0]
    num_trades = len(trades)

    wins = trades[trades['trade_pnl'] > 0]
    losses = trades[trades['trade_pnl'] < 0]

    win_rate = (len(wins) / num_trades * 100) if num_trades > 0 else 0

    return {
        'initial_capital': initial_capital,
        'final_value': final_val,
        'profit': final_val - initial_capital,
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'avg_drawdown': avg_dd,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'years': years
    }

def print_investment_report(report: dict, trade_log: list, ticker: str):
    print(f"\n{'='*80}")
    print(f"INVESTMENT PERFORMANCE REPORT - {ticker}")
    print(f"{'='*80}\n")
    print(f"💰 Financial Summary")
    print(f"  Initial Capital:         ${report['initial_capital']:,.2f}")
    print(f"  Final Portfolio:         ${report['final_value']:,.2f}")
    print(f"  Net Profit:              ${report['profit']:,.2f}")
    print(f"\n📈 Return Metrics")
    print(f"  Total Return:            {report['total_return']:.2f}%")
    print(f"  CAGR:                    {report['cagr']:.2f}%")
    print(f"  Volatility (Annual):     {report['volatility']:.2f}%")
    print(f"  Sharpe Ratio:            {report['sharpe']:.3f}")
    print(f"\n📉 Risk Metrics")
    print(f"  Max Drawdown:            {report['max_drawdown']:.2f}%")
    print(f"  Avg Drawdown:            {report['avg_drawdown']:.2f}%")
    print(f"\n🎯 Trade Statistics")
    print(f"  Win Rate:                {report['win_rate']:.2f}%")
    print(f"  Trades Executed:         {report['num_trades']}")

    print(f"\n{'='*80}")
    print(f"📝 DETAILED TRADE LOG ({len(trade_log)} Events)")
    print(f"{'='*80}")
    print(f"{'Date':<12} {'Action':<20} {'Price':<10} {'Size':<8} {'Portfolio':<12} {'PnL'}")
    print(f"{'-'*12} {'-'*20} {'-'*10} {'-'*8} {'-'*12} {'-'*6}")

    for t in trade_log:
        print(f"{t['Date']:<12} {t['Action']:<20} {t['Price']:<10} {t['Size']:<8} {t['Portfolio']:<12} {t['PnL']}")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description="Investment-Focused ASRE Backtest")
    parser.add_argument("ticker", help="Stock ticker (e.g., AAPL, META)")
    parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial Capital (default: 10000)")
    parser.add_argument("--rating", default='r_asre', choices=['r_asre', 'r_final'], 
                       help="Rating column (default: r_asre)")
    parser.add_argument("--rebalance", default='W', choices=['W', 'M', 'Q'],
                       help="Rebalancing frequency (default: W)")
    parser.add_argument("--overweight", type=float, default=75.0, help="Overweight threshold")
    parser.add_argument("--full", type=float, default=60.0, help="Full position threshold")
    parser.add_argument("--reduce", type=float, default=45.0, help="Reduce position threshold")
    parser.add_argument("--exit", type=float, default=30.0, help="Exit threshold")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"🏦 INVESTMENT BACKTEST - {args.ticker}")
    print(f"{'='*80}")

    try:
        df = fetch_all_data(args.ticker, args.start_date, args.end_date)
        df_results, report, trade_log = run_investment_backtest(
            df,
            initial_capital=args.capital,
            rating_col=args.rating,
            overweight_threshold=args.overweight,
            full_threshold=args.full,
            reduce_threshold=args.reduce,
            exit_threshold=args.exit,
            rebalance_period=args.rebalance
        )
        print_investment_report(report, trade_log, args.ticker)
        output_file = f"investment_backtest_{args.ticker}_{args.rating}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_results.to_csv(output_file)
        print(f"\n✅ Results saved to {output_file}")
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()