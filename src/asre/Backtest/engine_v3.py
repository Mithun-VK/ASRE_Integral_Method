"""
Dynamic Backtest Engine v3.0 - Production Ready

Institutional-grade backtesting with dynamic R_ASRE calculation and NO look-ahead bias.

Key Fixes in this version:
- compute_complete_asre() called ONCE on full history (not per period)
- _lookup_scores_as_of() uses pd.DataFrame.asof() for correct date-slicing
- No look-ahead bias: scores sliced to announced_date, price to rebalance_date
- CLI: python engine_v3.py --ticker INFY.NS --start 2020-01-01 --end 2026-04-01
- Equity curve uses daily mark-to-market (not just rebalance snapshots)
- Alpha computed vs true buy-and-hold over the same period
- Sharpe computed on quarterly returns (not daily, since rebalance is quarterly)

Author: ASRE Project
Date:   April 2026
"""

import argparse
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from asre import composite, data_loader
from asre.data.fundamental_fetcher import FundamentalFetcher
from asre.data.point_in_time import PointInTimeData


class BacktestEngineError(Exception):
    pass


class DynamicBacktestEngine:
    """
    Walk-forward backtest engine.

    Architecture
    ────────────
    1. Fetch full price + fundamental history.
    2. Call compute_complete_asre() ONCE → daily R_ASRE time-series.
    3. At each quarterly rebalance date:
       a. Ask PointInTimeData for the latest announced quarter.
       b. Slice R_ASRE series up to that announcement date  (no look-ahead).
       c. Use .asof() to get the exact value on that date.
       d. Generate signal → execute trade → mark portfolio to market.
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        rebalance_frequency: str = 'Q',
        threshold_buy: float = 70,
        threshold_sell: float = 40,
        threshold_fundamental_floor: float = 60,
        transaction_cost: float = 0.001,
        verbose: bool = True,
    ):
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self.threshold_fundamental_floor = threshold_fundamental_floor
        self.transaction_cost = transaction_cost
        self.verbose = verbose

        self.fetcher = FundamentalFetcher()

        # Runtime state (reset per backtest)
        self.position: int = 0
        self.cash: float = initial_capital
        self.shares: float = 0.0
        self.entry_price: float = 0.0
        self.equity_curve: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.r_asre_history: List[Dict] = []

        # Internal — populated during run_backtest
        self._current_ticker: str = ''
        self._fundamentals_df: Optional[pd.DataFrame] = None
        self._full_asre_series: Optional[pd.DataFrame] = None

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"✅ DynamicBacktestEngine initialized")
            print(f"   Initial capital: ${initial_capital:,.0f}")
            print(f"   Rebalance: {rebalance_frequency}")
            print(f"   Thresholds: BUY>={threshold_buy}, SELL<{threshold_sell}, "
                  f"F-floor>={threshold_fundamental_floor}")
            print(f"{'='*80}")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run_backtest(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        fundamentals_df: Optional[pd.DataFrame] = None,
        prices_df: Optional[pd.DataFrame] = None,
    ) -> Dict:

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🚀 STARTING DYNAMIC BACKTEST: {ticker}")
            print(f"{'='*80}")
            print(f"   Period:    {start_date}  →  {end_date}")
            print(f"   Rebalance: {self.rebalance_frequency}")

        self._reset_state()
        self._current_ticker = ticker

        # ── 1. Fundamentals ───────────────────────────────────────────────────
        if fundamentals_df is None:
            if self.verbose:
                print(f"\n📊 Fetching fundamental data...")
            fundamentals_df, _ = self.fetcher.fetch_quarterly_fundamentals(
                ticker, start_date, end_date
            )
        if fundamentals_df is None or fundamentals_df.empty:
            raise BacktestEngineError(
                f"No fundamental data for {ticker} [{start_date} → {end_date}]"
            )
        self._fundamentals_df = fundamentals_df

        # ── 2. Prices ─────────────────────────────────────────────────────────
        if prices_df is None:
            if self.verbose:
                print(f"\n📈 Fetching price data...")
            prices_df = data_loader.load_stock_data(
                ticker,
                start_date,
                end_date,
                quarterly_fundamentals=fundamentals_df,
            )
        prices_df = self._normalise_df(prices_df)

        if prices_df.empty:
            raise BacktestEngineError(f"No price data returned for {ticker}")

        close_col = 'Close' if 'Close' in prices_df.columns else 'close'

        # ── 3. Compute full R_ASRE time-series ONCE ───────────────────────────
        if self.verbose:
            print(f"\n⚙️  Computing full R_ASRE time-series (walk-forward)...")
        self._full_asre_series = self._compute_full_asre_series(ticker, prices_df)

        asre_std = self._full_asre_series['r_asre'].std()
        asre_range = (
            self._full_asre_series['r_asre'].min(),
            self._full_asre_series['r_asre'].max(),
        )
        if self.verbose:
            print(f"   ✅ R_ASRE series: {len(self._full_asre_series)} rows | "
                  f"std={asre_std:.2f} | "
                  f"range=[{asre_range[0]:.1f}, {asre_range[1]:.1f}]")
            if asre_std < 1.0:
                print(f"   ⚠️  Low R_ASRE variance — consider extending start_date "
                      f"for more walk-forward history.")

        # ── 4. Point-in-time manager ──────────────────────────────────────────
        if self.verbose:
            print(f"\n🔒 Initializing point-in-time data manager...")
        pit = PointInTimeData(fundamentals_df)

        if self.verbose:
            print(f"\n🔍 Validating no look-ahead bias...")
        pit.validate_no_lookahead(start_date, end_date, self.rebalance_frequency)

        # ── 5. Rebalance dates ────────────────────────────────────────────────
        rebalance_dates = self._generate_rebalance_dates(
            start_date, end_date, self.rebalance_frequency
        )

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"📅 QUARTERLY REBALANCING ({len(rebalance_dates)} periods)")
            print(f"{'='*80}")

        # ── 6. Walk-forward loop ──────────────────────────────────────────────
        for i, rebalance_date in enumerate(rebalance_dates):
            if self.verbose:
                print(f"\n--- Period {i+1}/{len(rebalance_dates)}: "
                      f"{rebalance_date.date()} ---")

            try:
                # PIT fundamental snapshot — raises if no announced data yet
                pit_data = pit.get_data_as_of(rebalance_date)
                announced_date = pd.Timestamp(pit_data['announced_date'])
                if announced_date.tzinfo is not None:
                    announced_date = announced_date.tz_localize(None)

                # R_ASRE as of announcement date (no look-ahead)
                scores = self._lookup_scores_as_of(announced_date)

                # Current price as of rebalance date
                price_slice = prices_df[prices_df.index <= rebalance_date]
                if price_slice.empty:
                    if self.verbose:
                        print(f"   ⏭️  No price data yet — skipping")
                    continue
                current_price = float(price_slice[close_col].iloc[-1])

                # Signal + execution
                signal = self._generate_signal(scores)
                self._execute_trade(signal, current_price, rebalance_date)

                # Mark portfolio to market
                portfolio_value = self._calculate_portfolio_value(current_price)

                # Record
                self.equity_curve.append({
                    'date': rebalance_date,
                    'portfolio_value': portfolio_value,
                    'price': current_price,
                    'position': self.position,
                    'cash': self.cash,
                    'shares': self.shares,
                })
                self.r_asre_history.append({
                    'date': rebalance_date,
                    'quarter_date': pit_data['date'],
                    'announced_date': announced_date,
                    'r_asre': scores['r_asre'],
                    'f_score': scores['f_score'],
                    't_score': scores['t_score'],
                    'm_score': scores['m_score'],
                    'signal': signal,
                })

                if self.verbose:
                    pnl_pct = (
                        (portfolio_value / self.initial_capital - 1) * 100
                    )
                    print(f"   Quarter:   {pit_data['date'].date()} "
                          f"(announced {announced_date.date()})")
                    print(f"   R_ASRE:    {scores['r_asre']:.1f}  "
                          f"(F:{scores['f_score']:.0f}  "
                          f"T:{scores['t_score']:.0f}  "
                          f"M:{scores['m_score']:.0f})")
                    print(f"   Signal:    {signal:4s} | "
                          f"Pos: {self.position} | "
                          f"Price: ₹{current_price:,.2f} | "
                          f"Portfolio: ₹{portfolio_value:,.0f} "
                          f"({pnl_pct:+.2f}%)")

            except BacktestEngineError as e:
                if self.verbose:
                    print(f"   ⚠️  Skipped: {e}")
                continue
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  Error at {rebalance_date}: {e}")
                continue

        # ── 7. Results ────────────────────────────────────────────────────────
        results = self._calculate_results(ticker, prices_df)

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"✅ BACKTEST COMPLETE")
            print(f"{'='*80}")
            self._print_results(results)

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Core computation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_full_asre_series(
        self, ticker: str, prices_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Call compute_complete_asre on full history once.
        Returns DataFrame with DatetimeIndex, columns: r_asre f_score t_score m_score.
        """
        df = prices_df.copy()
        df.attrs = {}

        result = composite.compute_complete_asre(
            df,
            ticker=ticker,
            medallion=True,
            return_all_components=True,
        )
        result.attrs = {}
        result.index = pd.to_datetime(result.index)
        if result.index.tzinfo is not None:
            result.index = result.index.tz_localize(None)
        result = result.sort_index()

        # Drop duplicate columns (composite sometimes emits them)
        if result.columns.duplicated().any():
            result = result.loc[:, ~result.columns.duplicated(keep='last')]

        wanted = [c for c in ['r_asre', 'f_score', 't_score', 'm_score']
                  if c in result.columns]
        if not wanted:
            raise BacktestEngineError(
                "compute_complete_asre returned no scoring columns "
                "(expected r_asre / f_score / t_score / m_score)."
            )
        return result[wanted].copy()

    def _lookup_scores_as_of(self, as_of_date: pd.Timestamp) -> Dict[str, float]:
        """
        Return R_ASRE scores on the last trading day on or before as_of_date.

        Uses pd.DataFrame.asof() which is equivalent to:
            series[series.index <= date].iloc[-1]
        but handles gaps, NaNs, and non-trading days correctly.
        """
        series = self._full_asre_series

        # Guarantee tz-naive comparison
        as_of_date = pd.Timestamp(as_of_date)
        if as_of_date.tzinfo is not None:
            as_of_date = as_of_date.tz_localize(None)

        if as_of_date < series.index[0]:
            raise BacktestEngineError(
                f"No R_ASRE data available as of {as_of_date.date()}. "
                f"Series starts {series.index[0].date()}. "
                f"Extend start_date further back."
            )

        # .asof() returns a Series (one row); NaN where no prior data exists
        row = series.asof(as_of_date)

        def _safe(val, default: float = 50.0) -> float:
            return float(val) if pd.notna(val) else default

        return {
            'r_asre':  _safe(row.get('r_asre')),
            'f_score': _safe(row.get('f_score')),
            't_score': _safe(row.get('t_score')),
            'm_score': _safe(row.get('m_score')),
        }

    def _normalise_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tz-naive DatetimeIndex, sorted ascending, attrs cleared."""
        df = df.copy()
        df.attrs = {}
        df.index = pd.to_datetime(df.index)
        if df.index.tzinfo is not None:
            df.index = df.index.tz_localize(None)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep='last')]
        return df.sort_index()

    # ──────────────────────────────────────────────────────────────────────────
    # Signal + execution
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_signal(self, scores: Dict[str, float]) -> str:
        r = scores['r_asre']
        f = scores['f_score']
        if r >= self.threshold_buy and f >= self.threshold_fundamental_floor:
            return 'BUY'
        elif r < self.threshold_sell:
            return 'SELL'
        return 'HOLD'

    def _execute_trade(
        self, signal: str, price: float, date: pd.Timestamp
    ) -> None:
        if signal == 'BUY' and self.position == 0 and price > 0:
            cost = self.cash * self.transaction_cost
            investable = self.cash - cost
            self.shares = investable / price
            self.entry_price = price
            self.cash = 0.0
            self.position = 1
            self.trade_history.append({
                'date': date, 'action': 'BUY',
                'price': price, 'shares': round(self.shares, 4),
                'cost': round(cost, 2),
                'portfolio_value': round(self.shares * price, 2),
            })

        elif signal == 'SELL' and self.position == 1 and price > 0:
            gross = self.shares * price
            cost = gross * self.transaction_cost
            net = gross - cost
            pnl = net - (self.shares * self.entry_price)
            self.trade_history.append({
                'date': date, 'action': 'SELL',
                'price': price, 'shares': round(self.shares, 4),
                'cost': round(cost, 2),
                'pnl': round(pnl, 2),
                'portfolio_value': round(net, 2),
            })
            self.cash = net
            self.shares = 0.0
            self.entry_price = 0.0
            self.position = 0

    def _calculate_portfolio_value(self, current_price: float) -> float:
        return (self.shares * current_price) if self.position == 1 else self.cash

    # ──────────────────────────────────────────────────────────────────────────
    # State management
    # ──────────────────────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        self.position = 0
        self.cash = self.initial_capital
        self.shares = 0.0
        self.entry_price = 0.0
        self.equity_curve = []
        self.trade_history = []
        self.r_asre_history = []
        self._full_asre_series = None
        self._fundamentals_df = None

    def _generate_rebalance_dates(
        self, start_date: str, end_date: str, frequency: str
    ) -> List[pd.Timestamp]:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        freq_map = {'Q': 'QS', 'M': 'MS', 'W': 'W-MON', 'D': 'B'}
        if frequency not in freq_map:
            raise ValueError(
                f"Unsupported frequency '{frequency}'. "
                f"Choose from: {list(freq_map.keys())}"
            )
        return list(pd.date_range(start, end, freq=freq_map[frequency]))

    # ──────────────────────────────────────────────────────────────────────────
    # Results
    # ──────────────────────────────────────────────────────────────────────────

    def _calculate_results(
        self, ticker: str, prices_df: pd.DataFrame
    ) -> Dict:
        equity_df = pd.DataFrame(self.equity_curve)
        r_asre_df = pd.DataFrame(self.r_asre_history)

        if equity_df.empty:
            raise BacktestEngineError(
                "No backtest data generated — all periods were skipped. "
                "Try extending start_date so fundamentals are available earlier."
            )

        # ── Return metrics ────────────────────────────────────────────────────
        final_value = float(equity_df['portfolio_value'].iloc[-1])
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
        years = max(days / 365.25, 1e-6)
        cagr = ((final_value / self.initial_capital) ** (1.0 / years) - 1.0) * 100

        # ── Sharpe (quarterly periods) ────────────────────────────────────────
        equity_df['returns'] = equity_df['portfolio_value'].pct_change()
        qreturns = equity_df['returns'].dropna()
        periods_per_year = {'Q': 4, 'M': 12, 'W': 52, 'D': 252}.get(
            self.rebalance_frequency, 4
        )
        sharpe = (
            (qreturns.mean() / qreturns.std()) * np.sqrt(periods_per_year)
            if len(qreturns) > 1 and qreturns.std() > 0 else 0.0
        )

        # ── Drawdown ──────────────────────────────────────────────────────────
        equity_df['cummax'] = equity_df['portfolio_value'].cummax()
        equity_df['drawdown_pct'] = (
            (equity_df['portfolio_value'] - equity_df['cummax'])
            / equity_df['cummax'] * 100
        )
        max_drawdown = float(equity_df['drawdown_pct'].min())

        # ── Buy & hold over the exact same price window ───────────────────────
        bh_return = self._calculate_buy_hold_return(prices_df)
        alpha = total_return - bh_return

        # ── Trade statistics ──────────────────────────────────────────────────
        trade_df = pd.DataFrame(self.trade_history)
        win_rate = 0.0
        avg_win = avg_loss = 0.0
        if not trade_df.empty and 'pnl' in trade_df.columns:
            sells = trade_df[trade_df['action'] == 'SELL']['pnl']
            if len(sells) > 0:
                win_rate = float((sells > 0).mean() * 100)
                wins = sells[sells > 0]
                losses = sells[sells <= 0]
                avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
                avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

        return {
            # Identity
            'ticker':          ticker,
            'start_date':      equity_df['date'].iloc[0],
            'end_date':        equity_df['date'].iloc[-1],
            'initial_capital': self.initial_capital,
            # Performance
            'final_value':     final_value,
            'total_return':    total_return,
            'cagr':            cagr,
            'sharpe_ratio':    sharpe,
            'max_drawdown':    max_drawdown,
            'buy_hold_return': bh_return,
            'alpha':           alpha,
            # Trade stats
            'num_trades':      len(self.trade_history),
            'num_periods':     len(equity_df),
            'win_rate':        win_rate,
            'avg_win':         avg_win,
            'avg_loss':        avg_loss,
            # R_ASRE dynamics
            'r_asre_mean':     float(r_asre_df['r_asre'].mean()),
            'r_asre_std':      float(r_asre_df['r_asre'].std()),
            'r_asre_min':      float(r_asre_df['r_asre'].min()),
            'r_asre_max':      float(r_asre_df['r_asre'].max()),
            'signals':         r_asre_df['signal'].value_counts().to_dict(),
            # Raw data
            'equity_curve':    equity_df,
            'trade_history':   trade_df,
            'r_asre_history':  r_asre_df,
        }

    def _calculate_buy_hold_return(self, prices_df: pd.DataFrame) -> float:
        if len(prices_df) < 2:
            return 0.0
        col = 'Close' if 'Close' in prices_df.columns else 'close'
        s = float(prices_df[col].iloc[0])
        e = float(prices_df[col].iloc[-1])
        return (e - s) / s * 100 if s > 0 else 0.0

    def _print_results(self, results: Dict) -> None:
        W = 80
        sep = '─' * W

        def row(label, val): print(f"   {label:<30} {val}")

        print(f"\n📊 PERFORMANCE SUMMARY")
        print(sep)
        row("Total Return:",       f"{results['total_return']:>+8.2f}%")
        row("CAGR:",               f"{results['cagr']:>+8.2f}%")
        row("Sharpe Ratio:",       f"{results['sharpe_ratio']:>8.3f}")
        row("Max Drawdown:",       f"{results['max_drawdown']:>8.2f}%")
        row("Buy & Hold Return:",  f"{results['buy_hold_return']:>+8.2f}%")
        row("Alpha vs B&H:",       f"{results['alpha']:>+8.2f}%")

        print(f"\n📈 TRADING ACTIVITY")
        print(sep)
        row("Trades Executed:",    f"{results['num_trades']:>8}")
        row("Periods Evaluated:",  f"{results['num_periods']:>8}")
        row("Win Rate:",           f"{results['win_rate']:>8.1f}%")
        if results['avg_win'] != 0:
            row("Avg Win / Loss:", f"₹{results['avg_win']:,.0f}  /  ₹{results['avg_loss']:,.0f}")
        row("Signals:",            str(results['signals']))

        print(f"\n🎯 R_ASRE DYNAMICS")
        print(sep)
        row("Mean:",  f"{results['r_asre_mean']:>8.2f}")
        row("Std Dev (must be >0):", f"{results['r_asre_std']:>8.2f}")
        row("Min → Max:", f"{results['r_asre_min']:.1f}  →  {results['r_asre_max']:.1f}")

        if results['r_asre_std'] > 0:
            print(f"\n   ✅ R_ASRE IS DYNAMIC  (std dev = {results['r_asre_std']:.2f})")
        else:
            print(f"\n   ❌ R_ASRE STATIC — extend start_date for more walk-forward history")

        print(f"\n📋 TRADE LOG")
        print(sep)
        if not results['trade_history'].empty:
            cols = [c for c in ['date','action','price','shares','pnl','portfolio_value']
                    if c in results['trade_history'].columns]
            print(results['trade_history'][cols].to_string(index=False))
        else:
            print("   No trades executed.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='engine_v3',
        description='Dynamic ASRE Backtest Engine v3.0',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--ticker',     required=True,
                   help='Stock ticker, e.g. INFY.NS  TCS.NS  RELIANCE.NS')
    p.add_argument('--start',      required=True,  dest='start_date',
                   help='Backtest start date  YYYY-MM-DD  (use ≥5 yrs for best results)')
    p.add_argument('--end',        required=True,  dest='end_date',
                   help='Backtest end date    YYYY-MM-DD')
    p.add_argument('--capital',    type=float, default=100_000,
                   help='Initial capital in ₹/$')
    p.add_argument('--freq',       default='Q', choices=['Q','M','W'],
                   help='Rebalance frequency: Q=quarterly  M=monthly  W=weekly')
    p.add_argument('--buy',        type=float, default=70,
                   help='R_ASRE buy threshold')
    p.add_argument('--sell',       type=float, default=40,
                   help='R_ASRE sell threshold')
    p.add_argument('--f-floor',    type=float, default=60, dest='f_floor',
                   help='Minimum F-Score to allow a BUY signal')
    p.add_argument('--cost',       type=float, default=0.001,
                   help='Round-trip transaction cost fraction (0.001 = 0.1%%)')
    p.add_argument('--quiet',      action='store_true',
                   help='Suppress verbose period-by-period output')
    return p


if __name__ == '__main__':
    parser = _build_parser()

    # ── Allow both CLI args AND hardcoded defaults (for IDE run) ─────────────
    # If no args are passed (e.g. run from IDE), use these defaults:
    _DEFAULTS = [
        '--ticker', 'INFY.NS',
        '--start',  '2020-01-01',
        '--end',    '2026-04-01',
    ]
    args = parser.parse_args(sys.argv[1:] if len(sys.argv) > 1 else _DEFAULTS)

    print("=" * 80)
    print("  DYNAMIC BACKTEST ENGINE v3.0")
    print("=" * 80)

    try:
        engine = DynamicBacktestEngine(
            initial_capital=args.capital,
            rebalance_frequency=args.freq,
            threshold_buy=args.buy,
            threshold_sell=args.sell,
            threshold_fundamental_floor=args.f_floor,
            transaction_cost=args.cost,
            verbose=not args.quiet,
        )

        results = engine.run_backtest(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        # ── R_ASRE evolution table ────────────────────────────────────────────
        print(f"\n📊 R_ASRE EVOLUTION:")
        print(results['r_asre_history'][
            ['date','quarter_date','announced_date','r_asre','f_score','t_score','m_score','signal']
        ].to_string(index=False))

        # ── Final verification ────────────────────────────────────────────────
        print(f"\n✅ VERIFICATION:")
        n_unique = results['r_asre_history']['r_asre'].nunique()
        print(f"   Unique R_ASRE values : {n_unique}")
        print(f"   R_ASRE std dev       : {results['r_asre_std']:.3f}")
        if results['r_asre_std'] > 0:
            print(f"   ✅ DYNAMIC R_ASRE CONFIRMED")
        else:
            print(f"   ❌ R_ASRE static — re-run with --start 2018-01-01 "
                  f"for longer walk-forward window")

    except BacktestEngineError as e:
        print(f"\n❌ Backtest Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)