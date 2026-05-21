"""
Investment-Focused ASRE Backtest - WITH TRADE LOG & CAPITAL DISPLAY
============================================================
✅ ADDED: Initial capital parameter for dollar value display
✅ CALCULATION: Remains percentage-based (unchanged)
✅ DISPLAY: Shows both percentage and dollar values
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
# INVESTMENT POSITION MANAGER (UNCHANGED - PERCENTAGE-BASED)
# ===========================================================================

def compute_investment_returns(
    df: pd.DataFrame,
    allocation_col: str = 'allocation',
    price_col: str = 'close',
    transaction_cost: float = 0.0005,
    stop_loss_pct: float = 0.25,
    trailing_stop_pct: float = 0.15,
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

    # ✅ NEW: Trade Log List
    trade_log = []

    current_position = 0.0
    entry_price = np.nan
    entry_date = None
    position_high = np.nan

    for idx, row in result_df.iterrows():
        target_allocation = row[allocation_col]
        current_price = row[price_col]

        if pd.isna(position_high) and current_position > 0:
            position_high = current_price
        elif current_position > 0:
            position_high = max(position_high, current_price)

        exit_triggered = False
        exit_reason = ''

        if current_position > 0:
            sl_price = entry_price * (1 - stop_loss_pct)
            if current_price <= sl_price:
                exit_triggered = True
                exit_reason = 'Stop_Loss'
            elif not pd.isna(position_high):
                trailing_price = position_high * (1 - trailing_stop_pct)
                if current_price <= trailing_price:
                    exit_triggered = True
                    exit_reason = 'Trailing_Stop'

        if exit_triggered:
            trade_return = (current_price - entry_price) / entry_price
            days_held = (idx - entry_date).days if entry_date else 0
            tax_rate = stcg_tax if days_held < 365 else ltcg_tax
            net_pnl = trade_return * (1 - tax_rate) if trade_return > 0 else trade_return

            result_df.at[idx, 'trade_pnl'] = net_pnl * current_position
            result_df.at[idx, 'exit_reason'] = exit_reason
            result_df.at[idx, 'days_held'] = days_held
            result_df.at[idx, 'tax_applied'] = tax_rate if trade_return > 0 else 0.0

            # ✅ LOG TRADE
            trade_log.append({
                'Date': idx.strftime('%Y-%m-%d'),
                'Action': 'SELL (Risk Exit)',
                'Reason': exit_reason,
                'Price': f"{current_price:.2f}",
                'Size': f"{current_position*100:.0f}%",
                'PnL': f"{net_pnl*100:.2f}%"
            })

            current_position = 0.0
            entry_price = np.nan
            entry_date = None
            position_high = np.nan

        if row.get('is_rebalance_date', False) or exit_triggered:
            position_change = target_allocation - current_position

            if abs(position_change) > 0.01: # Filter small noise
                action = 'BUY' if position_change > 0 else 'SELL'
                reason = 'Rebalance'

                # If increasing position (or new entry)
                if position_change > 0:
                    if current_position == 0:
                        entry_price = current_price
                        entry_date = idx
                        position_high = current_price

                    # ✅ LOG TRADE (Entry/Add)
                    trade_log.append({
                        'Date': idx.strftime('%Y-%m-%d'),
                        'Action': f"{action} (+{position_change*100:.0f}%)",
                        'Reason': 'Entry/Increase',
                        'Price': f"{current_price:.2f}",
                        'Size': f"{target_allocation*100:.0f}%",
                        'PnL': '-'
                    })

                    current_position = target_allocation

                # If reducing/exiting
                elif position_change < 0:
                    # If full exit
                    if target_allocation == 0:
                        trade_return = (current_price - entry_price) / entry_price if not pd.isna(entry_price) else 0
                        days_held = (idx - entry_date).days if entry_date else 0
                        tax_rate = stcg_tax if days_held < 365 else ltcg_tax
                        net_pnl = trade_return * (1 - tax_rate) if trade_return > 0 else trade_return

                        result_df.at[idx, 'trade_pnl'] = net_pnl * current_position
                        result_df.at[idx, 'exit_reason'] = 'Signal_Exit'
                        result_df.at[idx, 'days_held'] = days_held
                        result_df.at[idx, 'tax_applied'] = tax_rate if trade_return > 0 else 0.0

                        # ✅ LOG TRADE (Full Exit)
                        trade_log.append({
                            'Date': idx.strftime('%Y-%m-%d'),
                            'Action': 'SELL (Full Exit)',
                            'Reason': 'Signal < Exit',
                            'Price': f"{current_price:.2f}",
                            'Size': '0%',
                            'PnL': f"{net_pnl*100:.2f}%"
                        })

                        entry_price = np.nan
                        entry_date = None
                        position_high = np.nan

                    else:
                        # Partial reduction (no PnL realization logic for partials yet, just log)
                        trade_log.append({
                            'Date': idx.strftime('%Y-%m-%d'),
                            'Action': f"SELL ({position_change*100:.0f}%)",
                            'Reason': 'Reduce Exposure',
                            'Price': f"{current_price:.2f}",
                            'Size': f"{target_allocation*100:.0f}%",
                            'PnL': '-'
                        })

                    current_position = target_allocation

        result_df.at[idx, 'position'] = current_position
        result_df.at[idx, 'entry_price'] = entry_price
        result_df.at[idx, 'entry_date'] = entry_date
        result_df.at[idx, 'position_high'] = position_high

    result_df['price_return'] = result_df[price_col].pct_change()
    result_df['strategy_return'] = result_df['position'].shift(1) * result_df['price_return']
    result_df['position_change'] = result_df['position'].diff().abs()
    result_df['transaction_cost_incurred'] = result_df['position_change'] * transaction_cost
    result_df['net_return'] = result_df['strategy_return'] - result_df['transaction_cost_incurred']
    result_df['cumulative_return'] = (1 + result_df['net_return']).cumprod()

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
# INVESTMENT BACKTEST ENGINE (UPDATED WITH CAPITAL PARAMETER)
# ===========================================================================

def run_investment_backtest(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,  # ✅ ADDED
    rating_col: str = 'r_asre',
    overweight_threshold: float = 75.0,
    full_threshold: float = 60.0,
    reduce_threshold: float = 45.0,
    exit_threshold: float = 30.0,
    rebalance_period: str = 'W',
):
    logger.info(f"\n{'='*80}")
    logger.info(f"INVESTMENT BACKTEST ENGINE (${initial_capital:,.0f} Start)")  # ✅ UPDATED
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
        allocation_col='allocation',
        price_col='close',
        transaction_cost=0.0005,
        stop_loss_pct=0.25,
        trailing_stop_pct=0.15,
        ltcg_tax=0.10,
        stcg_tax=0.15
    )

    report = generate_investment_report(df_results, initial_capital)  # ✅ PASS CAPITAL
    return df_results, report, trade_log

# ===========================================================================
# REPORT GENERATION (UPDATED WITH DOLLAR VALUES)
# ===========================================================================

def generate_investment_report(df: pd.DataFrame, initial_capital: float = 10000.0) -> dict:  # ✅ ADDED PARAMETER
    total_return = (df['cumulative_return'].iloc[-1] - 1) * 100
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = ((df['cumulative_return'].iloc[-1]) ** (1 / years) - 1) * 100 if years > 0 else 0
    volatility = df['net_return'].std() * np.sqrt(252) * 100
    sharpe = (cagr / volatility) if volatility > 0 else 0
    max_dd = df['drawdown'].min() * 100
    avg_dd = df[df['drawdown'] < 0]['drawdown'].mean() * 100 if (df['drawdown'] < 0).any() else 0

    trades = df[df['trade_pnl'] != 0]
    num_trades = len(trades)
    wins = trades[trades['trade_pnl'] > 0]
    losses = trades[trades['trade_pnl'] < 0]

    win_rate = (len(wins) / num_trades * 100) if num_trades > 0 else 0
    avg_win = wins['trade_pnl'].mean() * 100 if len(wins) > 0 else 0
    avg_loss = losses['trade_pnl'].mean() * 100 if len(losses) > 0 else 0
    profit_factor = abs(wins['trade_pnl'].sum() / losses['trade_pnl'].sum()) if len(losses) > 0 and losses['trade_pnl'].sum() != 0 else 0
    total_tax = trades['tax_applied'].sum() if 'tax_applied' in trades.columns else 0

    total_position_change = df['position_change'].sum()
    annual_turnover = (total_position_change / years) * 100 if years > 0 else 0
    time_in_market = (df['position'] > 0).sum() / len(df) * 100

    # ✅ ADDED: Calculate dollar values
    final_value = initial_capital * df['cumulative_return'].iloc[-1]
    profit = final_value - initial_capital

    return {
        'initial_capital': initial_capital,  # ✅ ADDED
        'final_value': final_value,  # ✅ ADDED
        'profit': profit,  # ✅ ADDED
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'avg_drawdown': avg_dd,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_tax_rate': total_tax,
        'annual_turnover': annual_turnover,
        'time_in_market': time_in_market,
        'years': years
    }

def print_investment_report(report: dict, trade_log: list, ticker: str):
    print(f"\n{'='*80}")
    print(f"INVESTMENT PERFORMANCE REPORT - {ticker}")
    print(f"{'='*80}\n")
    
    # ✅ ADDED: Financial Summary Section
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
    print(f"\n💼 Investment Behavior")
    print(f"  Time in Market:          {report['time_in_market']:.1f}%")
    print(f"  Annual Turnover:         {report['annual_turnover']:.1f}%")
    print(f"  Number of Rebalances:    {report['num_trades']}")
    print(f"\n🎯 Trade Statistics")
    print(f"  Win Rate:                {report['win_rate']:.2f}%")
    print(f"  Avg Win:                 {report['avg_win']:.2f}%")
    print(f"  Avg Loss:                {report['avg_loss']:.2f}%")
    print(f"  Profit Factor:           {report['profit_factor']:.2f}")
    print(f"\n💰 Tax Impact")
    print(f"  Total Tax Drag:          {report['total_tax_rate']:.2f}%")

    print(f"\n{'='*80}")
    print(f"📝 DETAILED TRADE LOG ({len(trade_log)} Events)")
    print(f"{'='*80}")
    print(f"{'Date':<12} {'Action':<20} {'Price':<10} {'Size':<8} {'Reason':<20} {'PnL'}")
    print(f"{'-'*12} {'-'*20} {'-'*10} {'-'*8} {'-'*20} {'-'*6}")

    for t in trade_log:
        print(f"{t['Date']:<12} {t['Action']:<20} {t['Price']:<10} {t['Size']:<8} {t['Reason']:<20} {t['PnL']}")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description="Investment-Focused ASRE Backtest")
    parser.add_argument("ticker", help="Stock ticker (e.g., AAPL, META)")
    parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial Capital (default: 10000)")  # ✅ ADDED
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
            initial_capital=args.capital,  # ✅ ADDED
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
