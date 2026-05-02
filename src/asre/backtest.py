"""
ASRE Backtesting Framework

Fix Log:
  FIX-LOOP-1: Harden prev_high / prev_low extraction in simulation loop —
              result_df.at[] can return a 0-d numpy array when dtype is
              contaminated; pd.notna(0d_array) returns a 0-d bool array,
              making the `if` statement raise "truth value of a DataFrame
              is ambiguous". Use try/float() instead of pd.notna().

  FIX-LOOP-2: Force-cast all working columns to float64 AFTER creation
              (after step-5 pre-drop + step-6 re-assignment) to prevent
              numpy block-manager dtype contamination from TA libraries.

  FIX-LOOP-3: Expand the pre-drop set to include common TA-library column
              names (signal, buy_signal, sell_signal, trade_signal) so
              their object-dtype values never survive into the loop.

  FIX-BT-1:  Backtester.__init__ — .str.strip().str.lower() (order matters)
  FIX-BT-2:  Backtester.__init__ — log duplicate column names before dropping
  FIX-BT-3:  Backtester.run()   — drop pre-existing signal/position columns
  FIX-BT-4:  BeforeAfterComparison.__init__ — same strip+lower fix
  FIX-BT-5:  BacktesterV2.run_dip_quality_strategy — same pre-drop guard
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    CompositeConfig,
    FundamentalsConfig,
    TechnicalConfig,
    MomentumConfig,
    BacktestConfig,
)

from .composite import (
    compute_complete_asre,
    validate_asre_rating,
    get_asre_rating,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# FIX-LOOP-3: Full set of column names to drop before compute_strategy_returns
# assigns its own clean versions. Includes common TA-library output names
# that carry object dtype and contaminate the numpy block manager.
_WORKING_COLS = frozenset((
    'position', 'entry_signal', 'entry_price',
    'position_high', 'position_low', 'exit_reason',
    'price_return', 'strategy_return', 'net_return',
    'cumulative_return', 'drawdown', 'trade_pnl',
    'transaction_cost_incurred',
    # TA-library output column names:
    'signal', 'buy_signal', 'sell_signal', 'trade_signal',
))


def _safe_float(val, fallback: float) -> float:
    """
    FIX-LOOP-1 helper.
    Safely convert result_df.at[] output to a Python float.
    .at[] can return:
      - a Python float  (normal case)
      - a 0-d numpy array  (when the column block was dtype-contaminated)
      - np.nan / NaN
    pd.notna() on a 0-d array returns a 0-d bool array whose truth value
    raises "The truth value of a DataFrame is ambiguous."
    Using try/float() avoids that entirely.
    """
    try:
        f = float(val)
        return fallback if np.isnan(f) else f
    except (TypeError, ValueError):
        return fallback


# ---------------------------------------------------------------------------
# Signal Generators
# ---------------------------------------------------------------------------

def generate_signals_threshold(
    ratings: pd.Series,
    threshold_long: float = 70.0,
    threshold_short: float = 30.0,
    confidence_lower: Optional[pd.Series] = None,
    confidence_upper: Optional[pd.Series] = None,
    use_confidence: bool = False,
) -> pd.Series:
    ratings = ratings.squeeze()
    signals = pd.Series(0, index=ratings.index, dtype=int)

    if use_confidence and confidence_lower is not None and confidence_upper is not None:
        cl = confidence_lower.squeeze()
        cu = confidence_upper.squeeze()
        signals[cl >= threshold_long]  = 1
        signals[cu <= threshold_short] = -1
    else:
        signals[ratings >= threshold_long]  = 1
        signals[ratings <= threshold_short] = -1

    return signals


def generate_signals_quantile(
    ratings: pd.Series,
    top_quantile: float = 0.8,
    bottom_quantile: float = 0.2,
    window: int = 252,
) -> pd.Series:
    ratings = ratings.squeeze()
    upper_threshold = ratings.rolling(window=window).quantile(top_quantile)
    lower_threshold = ratings.rolling(window=window).quantile(bottom_quantile)

    signals = pd.Series(0, index=ratings.index, dtype=int)
    signals[ratings >= upper_threshold] = 1
    signals[ratings <= lower_threshold] = -1

    return signals


def generate_signals_regime(
    ratings: pd.Series,
    vix: pd.Series,
    vix_low: float = 15.0,
    vix_high: float = 25.0,
    rating_threshold: float = 60.0,
) -> pd.Series:
    ratings = ratings.squeeze()
    vix     = vix.squeeze()
    signals = pd.Series(0, index=ratings.index, dtype=int)

    low_vol_mask    = vix < vix_low
    normal_vol_mask = (vix >= vix_low) & (vix <= vix_high)
    high_vol_mask   = vix > vix_high

    signals[low_vol_mask    & (ratings > rating_threshold - 10)] = 1
    signals[normal_vol_mask & (ratings > rating_threshold)]      = 1
    signals[high_vol_mask   & (ratings > rating_threshold + 10)] = 1

    signals[low_vol_mask    & (ratings < 100 - rating_threshold + 10)] = -1
    signals[normal_vol_mask & (ratings < 100 - rating_threshold)]      = -1
    signals[high_vol_mask   & (ratings < 100 - rating_threshold - 10)] = -1

    return signals


def apply_position_sizing(
    signals: pd.Series,
    ratings: pd.Series,
    max_position: float = 1.0,
    scale_by_confidence: bool = True,
    confidence_lower: Optional[pd.Series] = None,
    confidence_upper: Optional[pd.Series] = None,
) -> pd.Series:
    if isinstance(signals, pd.DataFrame):
        signals = signals.iloc[:, -1]
    if isinstance(ratings, pd.DataFrame):
        ratings = ratings.iloc[:, -1]

    if not scale_by_confidence:
        return (signals * max_position).rename(None)

    if confidence_lower is not None and confidence_upper is not None:
        cl = confidence_lower.iloc[:, -1] if isinstance(confidence_lower, pd.DataFrame) else confidence_lower
        cu = confidence_upper.iloc[:, -1] if isinstance(confidence_upper, pd.DataFrame) else confidence_upper
        ci_width  = cu - cl
        max_width = float(ci_width.max())
        confidence = (
            1 - (ci_width / max_width)
            if max_width > 0
            else pd.Series(1.0, index=signals.index)
        )
    else:
        confidence = np.abs(ratings - 50) / 50

    if isinstance(confidence, pd.DataFrame):
        confidence = confidence.iloc[:, -1]

    positions = signals * confidence * max_position
    positions = np.clip(positions, -max_position, max_position)
    return pd.Series(positions, index=signals.index)


# ---------------------------------------------------------------------------
# Core Return Engine
# ---------------------------------------------------------------------------

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
    result_df = df.copy()

    # ══ FIX-ATTRS ══════════════════════════════════════════════════════════════
    # pandas propagates df.attrs through .copy(). compute_complete_asre stores
    # DataFrames (component scores, intermediate results) in df.attrs.
    # When groupby() internally calls concat() for error/repr paths, pandas
    # compares attrs dicts via `obj.attrs == attrs`, which calls DataFrame.__eq__
    # returning a DataFrame, then bool() on it raises:
    #   "The truth value of a DataFrame is ambiguous"
    # Wiping attrs here is safe — attrs is metadata only, never used by backtest.
    result_df.attrs = {}
    # ══════════════════════════════════════════════════════════════════════════

    # ── Step 1: strip THEN lowercase ─────────────────────────────────────────
    result_df.columns = result_df.columns.str.strip().str.lower()
    price_col  = price_col.strip().lower()
    signal_col = signal_col.strip().lower()

    # ── Step 2: deduplicate index ─────────────────────────────────────────────
    if result_df.index.duplicated().any():
        result_df = result_df[~result_df.index.duplicated(keep='last')]
        result_df = result_df.sort_index()

    # ── Step 3: deduplicate columns ───────────────────────────────────────────
    if result_df.columns.duplicated().any():
        dupes = result_df.columns[result_df.columns.duplicated(keep=False)].unique().tolist()
        logger.warning("compute_strategy_returns: duplicate columns %s — keeping last", dupes)
        result_df = result_df.loc[:, ~result_df.columns.duplicated(keep='last')]

    # ── Step 4: squeeze any column that is still a DataFrame slice ────────────
    for _col in [signal_col, price_col]:
        if _col in result_df.columns:
            val = result_df[_col]
            if isinstance(val, pd.DataFrame):
                result_df[_col] = val.iloc[:, -1]

    # ── Step 5: enforce float64 on price and signal ───────────────────────────
    try:
        result_df[price_col]  = pd.to_numeric(result_df[price_col],  errors='coerce').astype('float64')
        result_df[signal_col] = pd.to_numeric(result_df[signal_col], errors='coerce').fillna(0.0).astype('float64')
    except Exception as _e:
        raise ValueError(
            f"compute_strategy_returns: could not coerce '{price_col}' or "
            f"'{signal_col}' to float64. "
            f"Columns present: {list(result_df.columns)[:20]}"
        ) from _e

    # ── Step 6: FIX-LOOP-3 — drop ALL known working columns before creation ──
    # This removes any object-dtype 'position', 'signal', 'buy_signal' etc.
    # injected by TA libraries or earlier ASRE pipeline steps. Must happen
    # BEFORE the assignments below so we start from a clean float64 slate.
    _cols_to_drop = [c for c in _WORKING_COLS if c in result_df.columns
                     and c not in (signal_col, price_col)]
    if _cols_to_drop:
        logger.debug("compute_strategy_returns: pre-dropping columns %s", _cols_to_drop)
        result_df = result_df.drop(columns=_cols_to_drop)

    # ── Step 7: create working columns ───────────────────────────────────────
    result_df['entry_signal']  = result_df[signal_col].shift(1).fillna(0)
    result_df['price_return']  = result_df[price_col].pct_change()
    result_df['position']      = 0.0
    result_df['entry_price']   = np.nan
    result_df['position_high'] = np.nan
    result_df['position_low']  = np.nan
    result_df['exit_reason']   = ''

    # ── Step 8: FIX-LOOP-2 — force float64 on ALL scalar-access columns ──────
    # Prevents object-dtype block contamination from prior mixed-type
    # assignments causing .at[] to return 0-d arrays instead of scalars.
    for _c in ('position', 'entry_price', 'position_high', 'position_low', 'entry_signal'):
        result_df[_c] = result_df[_c].astype('float64')

    # ── Step 9: simulation loop ───────────────────────────────────────────────
    trade_groups = (result_df['entry_signal'].diff() != 0).cumsum()

    for group_id, group_df in result_df.groupby(trade_groups):
        if len(group_df) == 0:
            continue

        first_signal = float(group_df['entry_signal'].iloc[0])
        if first_signal == 0:
            continue

        entry_idx   = group_df.index[0]
        entry_price = float(result_df.at[entry_idx, price_col])

        result_df.at[entry_idx, 'position']    = first_signal
        result_df.at[entry_idx, 'entry_price'] = entry_price

        if first_signal > 0:
            result_df.at[entry_idx, 'position_high'] = entry_price
        else:
            result_df.at[entry_idx, 'position_low']  = entry_price

        for i in range(1, len(group_df)):
            row_idx  = group_df.index[i]
            prev_idx = group_df.index[i - 1]

            current_price      = float(result_df.at[row_idx,   price_col])
            stored_entry_price = float(result_df.at[entry_idx, 'entry_price'])
            exit_triggered     = False

            if first_signal > 0:
                # FIX-LOOP-1: use _safe_float instead of pd.notna()
                prev_high = _safe_float(result_df.at[prev_idx, 'position_high'], entry_price)
                pos_high  = max(prev_high, current_price)
                result_df.at[row_idx, 'position_high'] = pos_high

                sl_price       = stored_entry_price * (1 - stop_loss_pct)
                tp_price       = stored_entry_price * (1 + take_profit_pct)
                trailing_price = pos_high           * (1 - trailing_stop_pct)

                if   current_price <= sl_price:      result_df.at[row_idx, 'exit_reason'] = 'Stop_Loss';     exit_triggered = True
                elif current_price >= tp_price:       result_df.at[row_idx, 'exit_reason'] = 'Take_Profit';   exit_triggered = True
                elif current_price <= trailing_price: result_df.at[row_idx, 'exit_reason'] = 'Trailing_Stop'; exit_triggered = True

            else:
                # FIX-LOOP-1: use _safe_float instead of pd.notna()
                prev_low = _safe_float(result_df.at[prev_idx, 'position_low'], entry_price)
                pos_low  = min(prev_low, current_price)
                result_df.at[row_idx, 'position_low'] = pos_low

                sl_price       = stored_entry_price * (1 + stop_loss_pct)
                tp_price       = stored_entry_price * (1 - take_profit_pct)
                trailing_price = pos_low            * (1 + trailing_stop_pct)

                if   current_price >= sl_price:      result_df.at[row_idx, 'exit_reason'] = 'Stop_Loss';     exit_triggered = True
                elif current_price <= tp_price:       result_df.at[row_idx, 'exit_reason'] = 'Take_Profit';   exit_triggered = True
                elif current_price >= trailing_price: result_df.at[row_idx, 'exit_reason'] = 'Trailing_Stop'; exit_triggered = True

            if exit_triggered:
                result_df.at[row_idx, 'position'] = 0.0
                break
            else:
                result_df.at[row_idx, 'position'] = first_signal

    # ── Step 10: post-loop metrics ────────────────────────────────────────────
    result_df['strategy_return'] = result_df['position'].shift(1) * result_df['price_return']

    entry_trades = (result_df['entry_signal'].diff().abs() > 0).fillna(False)
    exit_trades  = ((result_df['position'].diff().abs() > 0) & (result_df['position'] == 0)).fillna(False)
    all_trades   = entry_trades | exit_trades

    total_cost = transaction_cost + slippage
    result_df['transaction_cost_incurred'] = np.where(all_trades, total_cost, 0.0)
    result_df['net_return']        = result_df['strategy_return'] - result_df['transaction_cost_incurred']
    result_df['cumulative_return'] = (1 + result_df['net_return']).cumprod()

    cummax = result_df['cumulative_return'].cummax()
    result_df['drawdown'] = (result_df['cumulative_return'] - cummax) / cummax

    result_df['trade_pnl'] = np.where(
        exit_trades,
        result_df['cumulative_return'].diff().fillna(0),
        0,
    )

    return result_df


# ---------------------------------------------------------------------------
# Portfolio Aggregation
# ---------------------------------------------------------------------------

def compute_portfolio_returns(
    portfolio: Dict[str, pd.DataFrame],
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    tickers = list(portfolio.keys())

    if weights is None:
        weights = {ticker: 1.0 / len(tickers) for ticker in tickers}

    returns_dict = {}
    for ticker in tickers:
        if 'net_return' in portfolio[ticker].columns:
            returns_dict[ticker] = portfolio[ticker]['net_return']

    returns_df        = pd.DataFrame(returns_dict)
    portfolio_returns = sum(returns_df[ticker] * weights[ticker] for ticker in tickers)

    result_df = pd.DataFrame({
        'portfolio_return':  portfolio_returns,
        'cumulative_return': (1 + portfolio_returns).cumprod(),
    })

    cummax = result_df['cumulative_return'].cummax()
    result_df['drawdown'] = (result_df['cumulative_return'] - cummax) / cummax

    return result_df


# ---------------------------------------------------------------------------
# Risk / Performance Metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    excess_returns = returns - risk_free_rate / periods_per_year
    if excess_returns.std() == 0:
        return 0.0
    return excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)


def sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    excess_returns   = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    return excess_returns.mean() / downside_returns.std() * np.sqrt(periods_per_year)


def calmar_ratio(returns, periods_per_year=252):
    total_return = (1 + returns).prod()
    n_years      = len(returns) / periods_per_year
    cagr_val     = total_return ** (1 / n_years) - 1 if n_years > 0 else 0
    cumulative   = (1 + returns).cumprod()
    running_max  = cumulative.cummax()
    drawdown     = (cumulative - running_max) / running_max
    max_dd       = drawdown.min()
    if max_dd == 0:
        return 0.0
    return cagr_val / abs(max_dd)


def information_ratio(returns, benchmark_returns, periods_per_year=252):
    aligned = pd.DataFrame({'strategy': returns, 'benchmark': benchmark_returns}).dropna()
    if len(aligned) < 2:
        return 0.0
    excess         = aligned['strategy'] - aligned['benchmark']
    tracking_error = excess.std()
    if tracking_error == 0:
        return 0.0
    return excess.mean() / tracking_error * np.sqrt(periods_per_year)


def max_drawdown(returns):
    cumulative  = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown    = (cumulative - running_max) / running_max
    return abs(drawdown.min())


def cagr(returns, periods_per_year=252):
    total_return = (1 + returns).prod()
    n_years      = len(returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    return total_return ** (1 / n_years) - 1


def win_rate(returns):
    positive = (returns > 0).sum()
    total    = len(returns[returns != 0])
    if total == 0:
        return 0.0
    return positive / total


def profit_factor(returns):
    gains  = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    return gains / losses


def value_at_risk(returns, confidence=0.95):
    return abs(np.percentile(returns, (1 - confidence) * 100))


def conditional_var(returns, confidence=0.95):
    var         = value_at_risk(returns, confidence)
    tail_losses = returns[returns < -var]
    if len(tail_losses) == 0:
        return var
    return abs(tail_losses.mean())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def generate_backtest_report(df, benchmark_returns=None, periods_per_year=252):
    returns = df['net_return'].dropna()

    report = {
        'total_return':   (1 + returns).prod() - 1,
        'cagr':           cagr(returns, periods_per_year),
        'mean_return':    returns.mean() * periods_per_year,
        'volatility':     returns.std()  * np.sqrt(periods_per_year),
        'sharpe_ratio':   sharpe_ratio(returns,  periods_per_year=periods_per_year),
        'sortino_ratio':  sortino_ratio(returns, periods_per_year=periods_per_year),
        'calmar_ratio':   calmar_ratio(returns,  periods_per_year),
        'max_drawdown':   max_drawdown(returns),
        'avg_drawdown':   abs(df['drawdown'].mean()),
        'win_rate':       win_rate(returns),
        'profit_factor':  profit_factor(returns),
        'best_day':       returns.max(),
        'worst_day':      returns.min(),
        'var_95':         value_at_risk(returns, 0.95),
        'cvar_95':        conditional_var(returns, 0.95),
        'skewness':       stats.skew(returns),
        'kurtosis':       stats.kurtosis(returns),
        'num_trades':     int(df['transaction_cost_incurred'].astype(bool).sum()),
        'avg_trade_cost': df['transaction_cost_incurred'].mean(),
    }

    if benchmark_returns is not None:
        bench = benchmark_returns.squeeze()
        report['information_ratio'] = information_ratio(returns, bench, periods_per_year)
        aligned = pd.DataFrame({'strategy': returns, 'benchmark': bench}).dropna()
        if len(aligned) > 1:
            report['beta']        = np.cov(aligned['strategy'], aligned['benchmark'])[0, 1] / np.var(aligned['benchmark'])
            report['alpha']       = report['cagr'] - report['beta'] * cagr(aligned['benchmark'], periods_per_year)
            report['correlation'] = aligned.corr().iloc[0, 1]

    return report


def print_backtest_report(report, title="Backtest Report"):
    print("=" * 70)
    print(f"{title:^70}")
    print("=" * 70)
    print(f"\n  Total Return:   {report['total_return']:>10.2%}")
    print(f"  CAGR:           {report['cagr']:>10.2%}")
    print(f"  Sharpe Ratio:   {report['sharpe_ratio']:>10.3f}")
    print(f"  Sortino Ratio:  {report['sortino_ratio']:>10.3f}")
    print(f"  Calmar Ratio:   {report['calmar_ratio']:>10.3f}")
    print(f"  Max Drawdown:   {report['max_drawdown']:>10.2%}")
    print(f"  Avg Drawdown:   {report['avg_drawdown']:>10.2%}")
    print(f"  Win Rate:       {report['win_rate']:>10.2%}")
    print(f"  Profit Factor:  {report['profit_factor']:>10.3f}")
    print(f"  Volatility:     {report['volatility']:>10.2%}")
    print(f"  VaR (95%):      {report['var_95']:>10.2%}")
    print(f"  CVaR (95%):     {report['cvar_95']:>10.2%}")
    print(f"  Skewness:       {report['skewness']:>10.3f}")
    print(f"  Kurtosis:       {report['kurtosis']:>10.3f}")
    print(f"  Num Trades:     {report['num_trades']:>10}")
    if 'alpha' in report:
        print(f"  Alpha:          {report['alpha']:>10.4f}")
        print(f"  Beta:           {report['beta']:>10.4f}")
        print(f"  Info Ratio:     {report['information_ratio']:>10.3f}")
        print(f"  Correlation:    {report['correlation']:>10.3f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Backtester Class
# ---------------------------------------------------------------------------

class Backtester:
    def __init__(
        self,
        df: pd.DataFrame,
        rating_col: str = 'r_asre',
        price_col: str = 'close',
        benchmark_col: Optional[str] = None,
        config: Optional[BacktestConfig] = None,
    ):
        self.df = df.copy()

        # FIX-BT-1: strip THEN lowercase
        self.df.columns = self.df.columns.str.strip().str.lower()

        # FIX-BT-2: log + deduplicate
        if self.df.columns.duplicated().any():
            dupes = self.df.columns[self.df.columns.duplicated(keep=False)].unique().tolist()
            logger.warning("Backtester.__init__: duplicate columns %s — keeping last", dupes)
            self.df = self.df.loc[:, ~self.df.columns.duplicated(keep='last')]

        self.rating_col    = rating_col.strip().lower()
        self.price_col     = price_col.strip().lower()
        self.benchmark_col = benchmark_col.strip().lower() if benchmark_col else None
        self.config        = config or BacktestConfig()

        self.results_df: Optional[pd.DataFrame] = None
        self.report:     Optional[Dict]          = None

        if self.rating_col not in self.df.columns:
            raise ValueError(f"Rating column '{self.rating_col}' not found in DataFrame")

        is_valid, msg = validate_asre_rating(self.df, self.rating_col)
        if not is_valid:
            logger.warning("Rating validation warning: %s", msg)

    def run(
        self,
        signal_type: str = 'threshold',
        threshold_long: Optional[float] = None,
        threshold_short: Optional[float] = None,
        transaction_cost: Optional[float] = None,
        slippage: Optional[float] = None,
        max_position: Optional[float] = None,
        use_confidence_bounds: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        logger.info("Running backtest with %s signals...", signal_type)

        threshold_long   = threshold_long   or self.config.threshold_long
        threshold_short  = threshold_short  or self.config.threshold_short
        transaction_cost = transaction_cost if transaction_cost is not None else self.config.transaction_cost
        slippage         = slippage         if slippage         is not None else self.config.slippage
        max_position     = max_position     if max_position     is not None else self.config.max_position

        ratings = get_asre_rating(
            self.df,
            rating_type='medallion' if self.rating_col == 'r_asre' else 'final',
        )
        if isinstance(ratings, pd.DataFrame):
            ratings = ratings.iloc[:, 0]
        ratings = ratings.squeeze()
        if isinstance(ratings, pd.DataFrame):
            ratings = pd.Series(
                ratings.values.flatten(),
                index=self.df.index[:len(ratings.values.flatten())]
            )

        confidence_lower = (
            self.df['confidence_lower'].squeeze()
            if 'confidence_lower' in self.df.columns else None
        )
        confidence_upper = (
            self.df['confidence_upper'].squeeze()
            if 'confidence_upper' in self.df.columns else None
        )

        if signal_type == 'threshold':
            signals = generate_signals_threshold(
                ratings, threshold_long, threshold_short,
                confidence_lower=confidence_lower,
                confidence_upper=confidence_upper,
                use_confidence=use_confidence_bounds,
            )
        elif signal_type == 'quantile':
            signals = generate_signals_quantile(ratings, **kwargs)
        elif signal_type == 'regime':
            if 'vix' not in self.df.columns:
                raise ValueError("VIX column required for regime-based signals")
            signals = generate_signals_regime(ratings, self.df['vix'].squeeze(), **kwargs)
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

        logger.info("Generated %d active signals", (signals != 0).sum())

        positions = apply_position_sizing(
            signals, ratings,
            max_position=max_position,
            scale_by_confidence=True,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
        )

        # FIX-BT-3: drop pre-existing signal/position columns before writing
        for _col in _WORKING_COLS:
            if _col in self.df.columns:
                self.df = self.df.drop(columns=_col)

        self.df['signal']   = signals
        self.df['position'] = positions

        self.results_df = compute_strategy_returns(
            self.df,
            signal_col='position',
            price_col=self.price_col,
            transaction_cost=transaction_cost,
            slippage=slippage,
        )

        logger.info("Backtest complete")
        return self.results_df

    def get_report(self) -> Dict:
        if self.results_df is None:
            raise ValueError("Run backtest first with .run()")
        benchmark_returns = None
        if self.benchmark_col and self.benchmark_col in self.df.columns:
            benchmark_returns = self.df[self.benchmark_col]
        self.report = generate_backtest_report(self.results_df, benchmark_returns=benchmark_returns)
        return self.report

    def print_report(self, title: str = "ASRE Strategy Backtest"):
        if self.report is None:
            self.get_report()
        print_backtest_report(self.report, title)

    def get_equity_curve(self) -> pd.Series:
        if self.results_df is None:
            raise ValueError("Run backtest first")
        return self.results_df['cumulative_return']

    def get_drawdown_series(self) -> pd.Series:
        if self.results_df is None:
            raise ValueError("Run backtest first")
        return self.results_df['drawdown']

    def get_trade_log(self) -> pd.DataFrame:
        if self.results_df is None:
            raise ValueError("Run backtest first")
        trades = self.results_df[self.results_df['transaction_cost_incurred'] > 0].copy()
        return trades[['signal', 'position', self.price_col, 'net_return', 'cumulative_return']]


# ---------------------------------------------------------------------------
# Dip-Quality Signal Generator
# ---------------------------------------------------------------------------

def generate_signals_dip_quality(
    df: pd.DataFrame,
    min_dip_quality: float = 70.0,
    min_fundamental: float = 65.0,
    allowed_stages: List[str] = ["EARLY", "MID"],
    use_r_asre: bool = True,
) -> pd.Series:
    signals = pd.Series(0, index=df.index, dtype=int)

    required_cols = ['dip_dip_quality_score', 'dip_dip_stage', 'f_score']
    if not all(col in df.columns for col in required_cols):
        logger.warning("Dip quality columns not found. Computing them now...")
        df = compute_complete_asre(df, medallion=True, return_all_components=True)

    dip_quality = df['dip_dip_quality_score'].squeeze()
    dip_stage   = df['dip_dip_stage'].squeeze()
    f_score     = df['f_score'].squeeze()
    rating = (
        df['r_asre'].squeeze()
        if use_r_asre and 'r_asre' in df.columns
        else df['r_final'].squeeze()
    )

    buy_condition = (
        (dip_quality >= min_dip_quality) &
        (f_score     >= min_fundamental) &
        (dip_stage.isin(allowed_stages)) &
        (rating      >= 60)
    )
    signals[buy_condition] = 1

    sell_condition = (
        (f_score  < min_fundamental) |
        (dip_stage.isin(["LATE", "RECOVERY"])) |
        (rating  <= 25)
    )
    signals[sell_condition] = -1

    logger.info(
        "Dip Quality Signals: %d LONG, %d SHORT, %d NEUTRAL",
        (signals == 1).sum(), (signals == -1).sum(), (signals == 0).sum()
    )
    return signals


def analyze_entry_quality_performance(df: pd.DataFrame, holding_period: int = 20) -> pd.DataFrame:
    if 'dip_dip_stage' not in df.columns:
        logger.warning("No dip stage data available")
        return pd.DataFrame()

    df = df.copy()
    df['forward_return'] = df['close'].pct_change(holding_period).shift(-holding_period)
    stages  = ['EARLY', 'MID', 'LATE', 'RECOVERY']
    results = []

    for stage in stages:
        stage_data = df[df['dip_dip_stage'] == stage]['forward_return'].dropna()
        if len(stage_data) == 0:
            continue
        results.append({
            'Stage':         stage,
            'Count':         len(stage_data),
            'Win Rate':      (stage_data > 0).sum() / len(stage_data),
            'Avg Return':    stage_data.mean(),
            'Median Return': stage_data.median(),
            'Std Dev':       stage_data.std(),
            'Sharpe': (
                stage_data.mean() / stage_data.std() * np.sqrt(252 / holding_period)
                if stage_data.std() > 0 else 0
            ),
            'Best':  stage_data.max(),
            'Worst': stage_data.min(),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Before/After Comparison
# ---------------------------------------------------------------------------

class BeforeAfterComparison:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # FIX-BT-4: strip THEN lowercase
        self.df.columns = self.df.columns.str.strip().str.lower()
        if self.df.columns.duplicated().any():
            dupes = self.df.columns[self.df.columns.duplicated(keep=False)].unique().tolist()
            logger.warning("BeforeAfterComparison.__init__: duplicate columns %s — keeping last", dupes)
            self.df = self.df.loc[:, ~self.df.columns.duplicated(keep='last')]
        self.results = {}

    def simulate_v2_0_rating(self) -> pd.Series:
        df          = self.df.copy()
        r_asre_v21  = df['r_asre'].squeeze().copy()
        momentum_trap = (df['f_score'].squeeze() < 65) & (df['m_score'].squeeze() > 80)
        r_asre_v20  = r_asre_v21.copy()
        for idx in df[momentum_trap].index:
            f = float(df.loc[idx, 'f_score'])
            t = float(df.loc[idx, 't_score'])
            m = float(df.loc[idx, 'm_score'])
            simulated_score    = 0.2 * f + 0.2 * t + 0.6 * m
            r_asre_v20.loc[idx] = np.clip(simulated_score, 50, 95)
        return r_asre_v20

    def run(self, signal_type='threshold', threshold_long=70.0, transaction_cost=0.001):
        logger.info("Running Before/After Comparison...")
        df_v20           = self.df.copy()
        df_v20['r_asre'] = self.simulate_v2_0_rating()

        bt_v20 = Backtester(df_v20, rating_col='r_asre')
        bt_v20.run(signal_type=signal_type, threshold_long=threshold_long, transaction_cost=transaction_cost)

        bt_v21 = Backtester(self.df, rating_col='r_asre')
        bt_v21.run(signal_type=signal_type, threshold_long=threshold_long, transaction_cost=transaction_cost)

        self.results = {
            'v2.0':        bt_v20.get_report(),
            'v2.1':        bt_v21.get_report(),
            'v2.0_equity': bt_v20.get_equity_curve(),
            'v2.1_equity': bt_v21.get_equity_curve(),
        }

    def print_report(self):
        if not self.results:
            raise ValueError("Run comparison first with .run()")
        v20 = self.results['v2.0']
        v21 = self.results['v2.1']
        print("=" * 90)
        print(f"{'BEFORE/AFTER COMPARISON: v2.0 vs v2.1':^90}")
        print("=" * 90)
        print(f"  {'Metric':<25} {'v2.0 (Before)':>20} {'v2.1 (After)':>20} {'Delta':>15}")
        print("-" * 90)
        metrics = [
            ('Total Return',  'total_return',  '{:.2%}'),
            ('CAGR',          'cagr',          '{:.2%}'),
            ('Sharpe Ratio',  'sharpe_ratio',  '{:.3f}'),
            ('Sortino Ratio', 'sortino_ratio', '{:.3f}'),
            ('Max Drawdown',  'max_drawdown',  '{:.2%}'),
            ('Win Rate',      'win_rate',      '{:.2%}'),
            ('Profit Factor', 'profit_factor', '{:.3f}'),
        ]
        for label, key, fmt in metrics:
            v0_val = v20.get(key, 0)
            v1_val = v21.get(key, 0)
            delta  = v1_val - v0_val
            sign   = '+' if delta >= 0 else ''
            print(
                f"  {label:<25} {fmt.format(v0_val):>20} "
                f"{fmt.format(v1_val):>20} {sign}{fmt.format(delta):>14}"
            )
        print("=" * 90)

    def get_improvement_summary(self) -> Dict:
        if not self.results:
            raise ValueError("Run comparison first")
        v20 = self.results['v2.0']
        v21 = self.results['v2.1']
        return {
            'return_improvement':   v21['total_return']  - v20['total_return'],
            'sharpe_improvement':   v21['sharpe_ratio']  - v20['sharpe_ratio'],
            'drawdown_reduction':   v20['max_drawdown']  - v21['max_drawdown'],
            'win_rate_improvement': v21['win_rate']      - v20['win_rate'],
        }


# ---------------------------------------------------------------------------
# BacktesterV2 — Dip Quality Strategy
# ---------------------------------------------------------------------------

class BacktesterV2(Backtester):
    def run_dip_quality_strategy(
        self,
        min_dip_quality: float = 70.0,
        min_fundamental: float = 65.0,
        allowed_stages: List[str] = ["EARLY", "MID"],
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
    ):
        logger.info("Running Dip Quality Strategy (v2.1)...")
        signals = generate_signals_dip_quality(
            self.df,
            min_dip_quality=min_dip_quality,
            min_fundamental=min_fundamental,
            allowed_stages=allowed_stages,
        )

        # FIX-BT-5: drop pre-existing working columns before writing
        for _col in _WORKING_COLS:
            if _col in self.df.columns:
                self.df = self.df.drop(columns=_col)

        self.df['signal']   = signals
        self.df['position'] = signals.astype(float)

        self.results_df = compute_strategy_returns(
            self.df,
            signal_col='position',
            price_col=self.price_col,
            transaction_cost=transaction_cost,
            slippage=slippage,
        )

        logger.info("Dip Quality Strategy complete!")
        return self.results_df

    def analyze_entry_timing(self, holding_period: int = 20) -> pd.DataFrame:
        return analyze_entry_quality_performance(self.df, holding_period)

    def print_entry_timing_report(self):
        timing_df = self.analyze_entry_timing()
        if timing_df.empty:
            print("No dip quality data available")
            return
        print("\n" + "=" * 90)
        print(f"{'ENTRY TIMING ANALYSIS (20-day Forward Returns)':^90}")
        print("=" * 90)
        print(f"  {'Stage':<10} | {'Count':>6} | {'Win Rate':>8} | {'Avg Return':>10} | {'Sharpe':>7}")
        print("-" * 90)
        for _, row in timing_df.iterrows():
            stage_emoji = {'EARLY': '🎯', 'MID': '✅', 'LATE': '⚠️', 'RECOVERY': '❌'}.get(row['Stage'], '')
            print(
                f"  {stage_emoji} {row['Stage']:<9} | "
                f"{row['Count']:>6.0f} | "
                f"{row['Win Rate']:>8.1%} | "
                f"{row['Avg Return']:>10.2%} | "
                f"{row['Sharpe']:>7.2f}"
            )
        print("=" * 90)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    'generate_signals_threshold',
    'generate_signals_quantile',
    'generate_signals_regime',
    'apply_position_sizing',
    'compute_strategy_returns',
    'compute_portfolio_returns',
    'sharpe_ratio',
    'sortino_ratio',
    'calmar_ratio',
    'information_ratio',
    'max_drawdown',
    'cagr',
    'win_rate',
    'profit_factor',
    'value_at_risk',
    'conditional_var',
    'generate_backtest_report',
    'print_backtest_report',
    'Backtester',
    'generate_signals_dip_quality',
    'analyze_entry_quality_performance',
    'BeforeAfterComparison',
    'BacktesterV2',
]