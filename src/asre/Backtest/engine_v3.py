"""
Dynamic Backtest Engine v3.0 - Production Ready

Institutional-grade backtesting with dynamic R_ASRE calculation and NO look-ahead bias.

Key Features:
- Recalculates R_ASRE quarterly (not static!)
- Uses point-in-time data only
- Integrates with existing ASRE score calculators
- Comprehensive performance tracking

Author: ASRE Project
Date: January 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Import ASRE data modules
from asre.data.fundamental_fetcher import FundamentalFetcher
from asre.data.point_in_time import PointInTimeData

# Import existing ASRE score calculators
from asre import composite, fundamentals, technical, momentum, data_loader


class BacktestEngineError(Exception):
    """Custom exception for backtest engine errors"""
    pass


class DynamicBacktestEngine:
    """
    Dynamic Backtest Engine with point-in-time R_ASRE calculation.

    Core Innovation:
    - R_ASRE is recalculated at each rebalance date
    - Uses ONLY data available as of that date (no look-ahead bias)
    - R_ASRE changes over time (not static!)

    Usage:
        engine = DynamicBacktestEngine(initial_capital=100000)
        results = engine.run_backtest('NVDA', '2020-01-01', '2025-01-01')
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        rebalance_frequency: str = 'Q',
        threshold_buy: float = 70,
        threshold_sell: float = 40,
        threshold_fundamental_floor: float = 65,
        transaction_cost: float = 0.001,
        verbose: bool = True
    ):
        """
        Initialize Dynamic Backtest Engine.

        Args:
            initial_capital: Starting capital
            rebalance_frequency: 'Q' (quarterly), 'M' (monthly)
            threshold_buy: Buy if R_ASRE >= this
            threshold_sell: Sell if R_ASRE < this
            threshold_fundamental_floor: Require F-Score >= this
            transaction_cost: Transaction cost (0.001 = 0.1%)
            verbose: Print detailed progress
        """
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self.threshold_fundamental_floor = threshold_fundamental_floor
        self.transaction_cost = transaction_cost
        self.verbose = verbose

        # Initialize components
        self.fetcher = FundamentalFetcher()

        # Initialize score calculators
        self.composite_calc = composite.CompositeRating()
        self.f_score_calc = fundamentals.FScore()
        self.t_score_calc = technical.TScore()
        self.m_score_calc = momentum.MScore()

        # State tracking
        self.position = 0  # Current position (0 or 1 for single stock)
        self.cash = initial_capital
        self.shares = 0
        self.equity_curve = []
        self.trade_history = []
        self.r_asre_history = []

        if self.verbose:
            print(f"✅ DynamicBacktestEngine initialized")
            print(f"   Initial capital: ${initial_capital:,.0f}")
            print(f"   Rebalance: {rebalance_frequency}")
            print(f"   Thresholds: BUY>={threshold_buy}, SELL<{threshold_sell}")

    def run_backtest(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        fundamentals_df: Optional[pd.DataFrame] = None,
        prices_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Run dynamic backtest with quarterly R_ASRE recalculation.

        Args:
            ticker: Stock ticker
            start_date: Backtest start date
            end_date: Backtest end date
            fundamentals_df: Pre-fetched fundamentals (optional)
            prices_df: Pre-fetched prices (optional)

        Returns:
            Dictionary with backtest results
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🚀 STARTING DYNAMIC BACKTEST: {ticker}")
            print(f"{'='*80}")
            print(f"Period: {start_date} to {end_date}")
            print(f"Rebalance: {self.rebalance_frequency}")

        # Reset state
        self._reset_state()

        # Fetch data if not provided
        if fundamentals_df is None:
            if self.verbose:
                print(f"\n📊 Fetching fundamental data...")
            fundamentals_df = self.fetcher.fetch_quarterly_fundamentals(
                ticker, start_date, end_date
            )

        if prices_df is None:
            if self.verbose:
                print(f"\n📈 Fetching price data...")
            prices_df = data_loader.fetch_stock_data(ticker, start_date, end_date)

        # Initialize point-in-time data manager
        if self.verbose:
            print(f"\n🔒 Initializing point-in-time data manager...")
        pit = PointInTimeData(fundamentals_df)

        # Validate no look-ahead bias
        if self.verbose:
            print(f"\n🔍 Validating no look-ahead bias...")
        pit.validate_no_lookahead(start_date, end_date, self.rebalance_frequency)

        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates(
            start_date, end_date, self.rebalance_frequency
        )

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"📅 QUARTERLY REBALANCING ({len(rebalance_dates)} periods)")
            print(f"{'='*80}")

        # Run backtest loop
        for i, rebalance_date in enumerate(rebalance_dates):
            if self.verbose:
                print(f"\n--- Period {i+1}/{len(rebalance_dates)}: {rebalance_date.date()} ---")

            try:
                # Get point-in-time data
                pit_data = pit.get_data_as_of(rebalance_date)

                # Get prices up to rebalance date
                prices_up_to_date = prices_df[prices_df.index <= rebalance_date]

                if prices_up_to_date.empty:
                    if self.verbose:
                        print(f"⏭️  No price data yet, skipping...")
                    continue

                # Calculate R_ASRE dynamically
                scores = self._calculate_r_asre_point_in_time(
                    pit_data, prices_up_to_date, rebalance_date
                )

                # Get current price
                current_price = prices_up_to_date['Close'].iloc[-1]

                # Generate signal
                signal = self._generate_signal(scores)

                # Execute trade
                self._execute_trade(signal, current_price, rebalance_date)

                # Update equity curve
                portfolio_value = self._calculate_portfolio_value(current_price)
                self.equity_curve.append({
                    'date': rebalance_date,
                    'portfolio_value': portfolio_value,
                    'price': current_price,
                    'position': self.position
                })

                # Store R_ASRE history
                self.r_asre_history.append({
                    'date': rebalance_date,
                    'quarter_date': pit_data['date'],
                    'announced_date': pit_data['announced_date'],
                    'r_asre': scores['r_asre'],
                    'f_score': scores['f_score'],
                    't_score': scores['t_score'],
                    'm_score': scores['m_score'],
                    'signal': signal
                })

                if self.verbose:
                    print(f"   Quarter: {pit_data['date'].date()} (announced {pit_data['announced_date'].date()})")
                    print(f"   R_ASRE: {scores['r_asre']:.1f} (F:{scores['f_score']:.0f}, T:{scores['t_score']:.0f}, M:{scores['m_score']:.0f})")
                    print(f"   Signal: {signal} | Position: {self.position} | Price: ${current_price:.2f}")
                    print(f"   Portfolio: ${portfolio_value:,.0f}")

            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Error at {rebalance_date}: {e}")
                continue

        # Calculate final results
        results = self._calculate_results(ticker, prices_df)

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"✅ BACKTEST COMPLETE")
            print(f"{'='*80}")
            self._print_results(results)

        return results

    def _reset_state(self):
        """Reset backtest state"""
        self.position = 0
        self.cash = self.initial_capital
        self.shares = 0
        self.equity_curve = []
        self.trade_history = []
        self.r_asre_history = []

    def _generate_rebalance_dates(
        self,
        start_date: str,
        end_date: str,
        frequency: str
    ) -> List[pd.Timestamp]:
        """Generate rebalance dates"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        if frequency == 'Q':
            dates = pd.date_range(start, end, freq='QS')
        elif frequency == 'M':
            dates = pd.date_range(start, end, freq='MS')
        else:
            raise ValueError(f"Invalid frequency: {frequency}")

        return list(dates)

    def _calculate_r_asre_point_in_time(
        self,
        fundamentals_row: pd.Series,
        prices: pd.DataFrame,
        as_of_date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Calculate R_ASRE using only data available as of date.

        This is the CRITICAL method that prevents look-ahead bias.
        """
        # Calculate F-Score from fundamentals
        f_score = self.f_score_calc.calculate(fundamentals_row)

        # Calculate T-Score from prices (up to as_of_date only!)
        t_score = self.t_score_calc.calculate(prices)

        # Calculate M-Score from prices (up to as_of_date only!)
        m_score = self.m_score_calc.calculate(prices)

        # Combine into R_ASRE using composite calculator
        r_asre = self.composite_calc.calculate_composite_score(
            f_score, t_score, m_score
        )

        return {
            'r_asre': r_asre,
            'f_score': f_score,
            't_score': t_score,
            'm_score': m_score
        }

    def _generate_signal(self, scores: Dict[str, float]) -> str:
        """
        Generate trading signal based on scores.

        Rules:
        - BUY if R_ASRE >= threshold_buy AND F-Score >= fundamental_floor
        - SELL if R_ASRE < threshold_sell
        - HOLD otherwise
        """
        r_asre = scores['r_asre']
        f_score = scores['f_score']

        # Check fundamental floor
        if f_score < self.threshold_fundamental_floor:
            return 'HOLD'  # Don't buy if fundamentals weak

        # Generate signal
        if r_asre >= self.threshold_buy:
            return 'BUY'
        elif r_asre < self.threshold_sell:
            return 'SELL'
        else:
            return 'HOLD'

    def _execute_trade(
        self,
        signal: str,
        price: float,
        date: pd.Timestamp
    ):
        """Execute trade based on signal"""
        if signal == 'BUY' and self.position == 0:
            # Buy: Invest all cash
            cost = self.cash * self.transaction_cost
            self.shares = (self.cash - cost) / price
            self.cash = 0
            self.position = 1

            self.trade_history.append({
                'date': date,
                'action': 'BUY',
                'price': price,
                'shares': self.shares,
                'cost': cost
            })

        elif signal == 'SELL' and self.position == 1:
            # Sell: Convert shares to cash
            proceeds = self.shares * price
            cost = proceeds * self.transaction_cost
            self.cash = proceeds - cost
            self.shares = 0
            self.position = 0

            self.trade_history.append({
                'date': date,
                'action': 'SELL',
                'price': price,
                'shares': self.shares,
                'cost': cost
            })

    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value"""
        if self.position == 1:
            return self.shares * current_price
        else:
            return self.cash

    def _calculate_results(
        self,
        ticker: str,
        prices_df: pd.DataFrame
    ) -> Dict:
        """Calculate comprehensive backtest results"""
        equity_df = pd.DataFrame(self.equity_curve)
        r_asre_df = pd.DataFrame(self.r_asre_history)

        if equity_df.empty:
            raise BacktestEngineError("No backtest data generated")

        # Calculate returns
        equity_df['returns'] = equity_df['portfolio_value'].pct_change()

        # Final portfolio value
        final_value = equity_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        # Calculate CAGR
        days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
        years = days / 365.25
        cagr = ((final_value / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Calculate Sharpe ratio (annualized)
        returns = equity_df['returns'].dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Calculate max drawdown
        equity_df['cummax'] = equity_df['portfolio_value'].cummax()
        equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['cummax']) / equity_df['cummax'] * 100
        max_drawdown = equity_df['drawdown'].min()

        # Calculate buy & hold returns
        buy_hold_return = self._calculate_buy_hold_return(prices_df)

        # R_ASRE statistics (CRITICAL: Proves it's dynamic!)
        r_asre_mean = r_asre_df['r_asre'].mean()
        r_asre_std = r_asre_df['r_asre'].std()
        r_asre_min = r_asre_df['r_asre'].min()
        r_asre_max = r_asre_df['r_asre'].max()

        # Count signals
        signals = r_asre_df['signal'].value_counts().to_dict()

        return {
            'ticker': ticker,
            'start_date': equity_df['date'].iloc[0],
            'end_date': equity_df['date'].iloc[-1],
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'buy_hold_return': buy_hold_return,
            'alpha': total_return - buy_hold_return,
            'num_trades': len(self.trade_history),
            'num_periods': len(equity_df),
            'r_asre_mean': r_asre_mean,
            'r_asre_std': r_asre_std,
            'r_asre_min': r_asre_min,
            'r_asre_max': r_asre_max,
            'signals': signals,
            'equity_curve': equity_df,
            'trade_history': pd.DataFrame(self.trade_history),
            'r_asre_history': r_asre_df
        }

    def _calculate_buy_hold_return(self, prices_df: pd.DataFrame) -> float:
        """Calculate buy & hold benchmark return"""
        if len(prices_df) < 2:
            return 0

        start_price = prices_df['Close'].iloc[0]
        end_price = prices_df['Close'].iloc[-1]

        return (end_price - start_price) / start_price * 100

    def _print_results(self, results: Dict):
        """Print formatted results"""
        print(f"\n📊 PERFORMANCE SUMMARY")
        print(f"{'─'*80}")
        print(f"Total Return:        {results['total_return']:>8.2f}%")
        print(f"CAGR:                {results['cagr']:>8.2f}%")
        print(f"Sharpe Ratio:        {results['sharpe_ratio']:>8.2f}")
        print(f"Max Drawdown:        {results['max_drawdown']:>8.2f}%")
        print(f"Buy & Hold Return:   {results['buy_hold_return']:>8.2f}%")
        print(f"Alpha:               {results['alpha']:>8.2f}%")
        print(f"\n📈 TRADING ACTIVITY")
        print(f"{'─'*80}")
        print(f"Number of Trades:    {results['num_trades']:>8}")
        print(f"Number of Periods:   {results['num_periods']:>8}")
        print(f"Signals: {results['signals']}")
        print(f"\n🎯 R_ASRE STATISTICS (Dynamic Proof!)")
        print(f"{'─'*80}")
        print(f"Mean:                {results['r_asre_mean']:>8.2f}")
        print(f"Std Dev:             {results['r_asre_std']:>8.2f}  ← MUST BE > 0!")
        print(f"Min:                 {results['r_asre_min']:>8.2f}")
        print(f"Max:                 {results['r_asre_max']:>8.2f}")

        if results['r_asre_std'] > 0:
            print(f"\n✅ R_ASRE IS DYNAMIC (std dev = {results['r_asre_std']:.2f})")
        else:
            print(f"\n❌ WARNING: R_ASRE APPEARS STATIC (std dev = 0)")


# Example usage
if __name__ == '__main__':
    """
    Example: Run dynamic backtest on NVDA
    """
    print("="*80)
    print("DYNAMIC BACKTEST ENGINE v3.0 - EXAMPLE")
    print("="*80)

    try:
        # Initialize engine
        engine = DynamicBacktestEngine(
            initial_capital=100000,
            rebalance_frequency='Q',
            threshold_buy=70,
            threshold_sell=40,
            verbose=True
        )

        # Run backtest
        results = engine.run_backtest(
            ticker='NVDA',
            start_date='2023-01-01',
            end_date='2024-12-31'
        )

        # Show R_ASRE evolution
        print(f"\n📊 R_ASRE EVOLUTION (First 10 periods):")
        print(results['r_asre_history'][['date', 'r_asre', 'signal']].head(10).to_string(index=False))

        # Verify dynamic R_ASRE
        print(f"\n✅ VERIFICATION:")
        print(f"   R_ASRE changed: {results['r_asre_history']['r_asre'].nunique()} unique values")
        print(f"   R_ASRE std dev: {results['r_asre_std']:.2f}")

        if results['r_asre_std'] > 0:
            print(f"   ✅ DYNAMIC R_ASRE CONFIRMED!")
        else:
            print(f"   ❌ WARNING: R_ASRE appears static")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()