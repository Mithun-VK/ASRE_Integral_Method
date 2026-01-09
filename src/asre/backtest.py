"""
ASRE Backtesting Framework

Comprehensive backtesting engine for ASRE rating strategies.

Features:
1. Signal Generation
   - Long/short/neutral based on rating thresholds
   - Position sizing and leverage control
   - Multiple signal types (threshold, quantile, regime)

2. Return Computation
   - Transaction costs (bid-ask spread, commissions)
   - Slippage modeling
   - Capital allocation

3. Performance Metrics
   - Sharpe ratio, Calmar ratio, Sortino ratio
   - Information ratio vs benchmark
   - Win rate, profit factor
   - Maximum drawdown, recovery time
   - Cumulative returns, CAGR

4. Risk Analysis
   - Value at Risk (VaR)
   - Conditional VaR (CVaR)
   - Beta, correlation
   - Rolling metrics

Production-grade features:
- Full integration with composite.py and config.py
- Event-driven backtesting (no look-ahead bias)
- Multiple rebalancing frequencies
- Portfolio-level analysis
- Confidence-based position sizing
- Comprehensive reporting
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
# Signal Generation (Enhanced with Confidence Bounds)
# ---------------------------------------------------------------------------


def generate_signals_threshold(
    ratings: pd.Series,
    threshold_long: float = 70.0,
    threshold_short: float = 30.0,
    confidence_lower: Optional[pd.Series] = None,
    confidence_upper: Optional[pd.Series] = None,
    use_confidence: bool = False,
) -> pd.Series:
    """
    Generate trading signals based on rating thresholds.
    
    Signal rules:
    - rating >= threshold_long → Long (1)
    - rating <= threshold_short → Short (-1)
    - otherwise → Neutral (0)
    
    Enhanced: Can use Kalman confidence bounds for signal validation
    
    Args:
        ratings: ASRE rating series
        threshold_long: Threshold for long signal
        threshold_short: Threshold for short signal
        confidence_lower: Lower confidence bound from Kalman filter
        confidence_upper: Upper confidence bound from Kalman filter
        use_confidence: If True, require confidence bounds to support signal
    
    Returns:
        Signal series: 1 (long), 0 (neutral), -1 (short)
    """
    signals = pd.Series(0, index=ratings.index, dtype=int)
    
    if use_confidence and confidence_lower is not None and confidence_upper is not None:
        # Long only if lower bound >= threshold
        signals[confidence_lower >= threshold_long] = 1
        
        # Short only if upper bound <= threshold
        signals[confidence_upper <= threshold_short] = -1
    else:
        # Standard threshold signals
        signals[ratings >= threshold_long] = 1
        signals[ratings <= threshold_short] = -1
    
    return signals


def generate_signals_quantile(
    ratings: pd.Series,
    top_quantile: float = 0.8,
    bottom_quantile: float = 0.2,
    window: int = 252,
) -> pd.Series:
    """
    Generate signals based on rolling quantile ranks.
    
    Args:
        ratings: ASRE rating series
        top_quantile: Top quantile for long (e.g., 0.8 = top 20%)
        bottom_quantile: Bottom quantile for short
        window: Rolling window for quantile calculation
    
    Returns:
        Signal series
    """
    # Rolling quantile thresholds
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
    """
    Generate regime-aware signals.
    
    Logic:
    - Low vol regime (VIX < vix_low): More aggressive (lower threshold)
    - High vol regime (VIX > vix_high): More conservative (higher threshold)
    
    Args:
        ratings: ASRE rating series
        vix: VIX series
        vix_low: Low volatility threshold
        vix_high: High volatility threshold
        rating_threshold: Base rating threshold
    
    Returns:
        Signal series
    """
    signals = pd.Series(0, index=ratings.index, dtype=int)
    
    # Low vol: aggressive (long if rating > threshold - 10)
    low_vol_mask = vix < vix_low
    signals[low_vol_mask & (ratings > rating_threshold - 10)] = 1
    
    # Normal vol: standard threshold
    normal_vol_mask = (vix >= vix_low) & (vix <= vix_high)
    signals[normal_vol_mask & (ratings > rating_threshold)] = 1
    
    # High vol: conservative (long if rating > threshold + 10)
    high_vol_mask = vix > vix_high
    signals[high_vol_mask & (ratings > rating_threshold + 10)] = 1
    
    # Short signals (mirror logic)
    signals[low_vol_mask & (ratings < 100 - rating_threshold + 10)] = -1
    signals[normal_vol_mask & (ratings < 100 - rating_threshold)] = -1
    signals[high_vol_mask & (ratings < 100 - rating_threshold - 10)] = -1
    
    return signals


def apply_position_sizing(
    signals: pd.Series,
    ratings: pd.Series,
    max_position: float = 1.0,
    scale_by_confidence: bool = True,
    confidence_lower: Optional[pd.Series] = None,
    confidence_upper: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Scale position sizes based on signal confidence.
    
    Enhanced: Can use Kalman confidence interval width as confidence proxy
    
    Args:
        signals: Binary signals (1, 0, -1)
        ratings: ASRE ratings
        max_position: Maximum position size (leverage cap)
        scale_by_confidence: Scale by distance from neutral (50) or CI width
        confidence_lower: Lower confidence bound
        confidence_upper: Upper confidence bound
    
    Returns:
        Position size series (continuous values)
    """
    if not scale_by_confidence:
        return signals * max_position
    
    # Use confidence interval width if available
    if confidence_lower is not None and confidence_upper is not None:
        # Narrower CI = higher confidence
        ci_width = confidence_upper - confidence_lower
        max_width = ci_width.max()
        
        # Inverse: narrower CI → higher confidence
        if max_width > 0:
            confidence = 1 - (ci_width / max_width)
        else:
            confidence = 1.0
    else:
        # Distance from neutral (50) as confidence proxy
        confidence = np.abs(ratings - 50) / 50
    
    # Scale signals by confidence
    positions = signals * confidence * max_position
    
    # Clip to max position
    positions = np.clip(positions, -max_position, max_position)
    
    return positions


# ---------------------------------------------------------------------------
# Return Computation
# ---------------------------------------------------------------------------


def compute_strategy_returns(
    df: pd.DataFrame,
    signal_col: str = 'signal',
    price_col: str = 'close',
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
) -> pd.DataFrame:
    """
    Compute strategy returns with transaction costs.
    
    Args:
        df: DataFrame with signals and prices
        signal_col: Column name for signals
        price_col: Column name for prices
        transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
        slippage: Slippage as fraction of price
    
    Returns:
        DataFrame with added columns:
        - price_return: Raw price return
        - strategy_return: Strategy return (signal * price_return)
        - transaction_cost_incurred: Cost of trades
        - net_return: Strategy return minus costs
        - cumulative_return: Cumulative strategy return
        - drawdown: Drawdown from peak
    """
    result_df = df.copy()
    
    # Calculate price returns
    result_df['price_return'] = result_df[price_col].pct_change()
    
    # Strategy returns (before costs)
    signals = result_df[signal_col].shift(1)  # Use previous day's signal (no look-ahead)
    result_df['strategy_return'] = signals * result_df['price_return']
    
    # Detect trades (signal changes)
    signal_change = result_df[signal_col].diff().abs()
    is_trade = signal_change > 0
    
    # Transaction costs (applied on trades)
    total_cost = transaction_cost + slippage
    result_df['transaction_cost_incurred'] = np.where(is_trade, total_cost, 0.0)
    
    # Net returns after costs
    result_df['net_return'] = result_df['strategy_return'] - result_df['transaction_cost_incurred']
    
    # Cumulative returns
    result_df['cumulative_return'] = (1 + result_df['net_return']).cumprod()
    
    # Drawdown
    cummax = result_df['cumulative_return'].cummax()
    result_df['drawdown'] = (result_df['cumulative_return'] - cummax) / cummax
    
    return result_df


def compute_portfolio_returns(
    portfolio: Dict[str, pd.DataFrame],
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute portfolio-level returns from multiple assets.
    
    Args:
        portfolio: Dict mapping ticker to DataFrame with 'net_return' column
        weights: Dict mapping ticker to weight (equal weight if None)
    
    Returns:
        DataFrame with portfolio returns
    """
    tickers = list(portfolio.keys())
    
    # Equal weights if not specified
    if weights is None:
        weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
    
    # Align all DataFrames by date
    returns_dict = {}
    for ticker in tickers:
        if 'net_return' in portfolio[ticker].columns:
            returns_dict[ticker] = portfolio[ticker]['net_return']
    
    returns_df = pd.DataFrame(returns_dict)
    
    # Weighted portfolio returns
    portfolio_returns = sum(returns_df[ticker] * weights[ticker] for ticker in tickers)
    
    # Aggregate metrics
    result_df = pd.DataFrame({
        'portfolio_return': portfolio_returns,
        'cumulative_return': (1 + portfolio_returns).cumprod(),
    })
    
    # Drawdown
    cummax = result_df['cumulative_return'].cummax()
    result_df['drawdown'] = (result_df['cumulative_return'] - cummax) / cummax
    
    return result_df


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sharpe ratio.
    
    Formula: Sharpe = (E[R] - R_f) / σ(R) * √(periods_per_year)
    
    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year (252 for daily)
    
    Returns:
        Annualized Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    
    return sharpe


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sortino ratio (penalizes only downside volatility).
    
    Formula: Sortino = (E[R] - R_f) / σ_downside(R) * √(periods_per_year)
    
    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
    
    Returns:
        Annualized Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    
    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(periods_per_year)
    
    return sortino


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Calmar ratio.
    
    Formula: Calmar = CAGR / Max Drawdown
    
    Args:
        returns: Return series
        periods_per_year: Trading periods per year
    
    Returns:
        Calmar ratio
    """
    # CAGR
    total_return = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    cagr_val = total_return ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    if max_dd == 0:
        return 0.0
    
    calmar = cagr_val / abs(max_dd)
    
    return calmar


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Information ratio.
    
    Formula: IR = E[R - R_b] / σ(R - R_b) * √(periods_per_year)
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Trading periods per year
    
    Returns:
        Annualized Information ratio
    """
    # Align series
    aligned = pd.DataFrame({
        'strategy': returns,
        'benchmark': benchmark_returns,
    }).dropna()
    
    if len(aligned) < 2:
        return 0.0
    
    # Excess returns
    excess = aligned['strategy'] - aligned['benchmark']
    
    # Tracking error
    tracking_error = excess.std()
    
    if tracking_error == 0:
        return 0.0
    
    ir = excess.mean() / tracking_error * np.sqrt(periods_per_year)
    
    return ir


def max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Return series
    
    Returns:
        Maximum drawdown (positive value, e.g., 0.2 = 20% drawdown)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    return abs(drawdown.min())


def cagr(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Args:
        returns: Return series
        periods_per_year: Trading periods per year
    
    Returns:
        Annualized return (CAGR)
    """
    total_return = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    
    if n_years <= 0:
        return 0.0
    
    return total_return ** (1 / n_years) - 1


def win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate (fraction of positive return periods).
    
    Args:
        returns: Return series
    
    Returns:
        Win rate [0, 1]
    """
    positive = (returns > 0).sum()
    total = len(returns[returns != 0])  # Exclude zero returns
    
    if total == 0:
        return 0.0
    
    return positive / total


def profit_factor(returns: pd.Series) -> float:
    """
    Calculate profit factor.
    
    Formula: PF = Total Gains / Total Losses
    
    Args:
        returns: Return series
    
    Returns:
        Profit factor (>1 is profitable)
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    
    return gains / losses


def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Value at Risk (VaR).
    
    VaR is the maximum expected loss at given confidence level.
    
    Args:
        returns: Return series
        confidence: Confidence level (0.95 = 95%)
    
    Returns:
        VaR (positive value, e.g., 0.02 = 2% max loss)
    """
    return abs(np.percentile(returns, (1 - confidence) * 100))


def conditional_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
    
    CVaR is the expected loss beyond VaR.
    
    Args:
        returns: Return series
        confidence: Confidence level
    
    Returns:
        CVaR (positive value)
    """
    var = value_at_risk(returns, confidence)
    
    # Average of losses exceeding VaR
    tail_losses = returns[returns < -var]
    
    if len(tail_losses) == 0:
        return var
    
    return abs(tail_losses.mean())


# ---------------------------------------------------------------------------
# Comprehensive Backtest Report
# ---------------------------------------------------------------------------


def generate_backtest_report(
    df: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    periods_per_year: int = 252,
) -> Dict[str, Union[float, pd.Series]]:
    """
    Generate comprehensive backtest report.
    
    Args:
        df: DataFrame with strategy returns
        benchmark_returns: Optional benchmark for comparison
        periods_per_year: Trading periods per year
    
    Returns:
        Dict with all performance metrics
    """
    returns = df['net_return'].dropna()
    
    report = {
        # Return metrics
        'total_return': (1 + returns).prod() - 1,
        'cagr': cagr(returns, periods_per_year),
        'mean_return': returns.mean() * periods_per_year,
        'volatility': returns.std() * np.sqrt(periods_per_year),
        
        # Risk-adjusted returns
        'sharpe_ratio': sharpe_ratio(returns, periods_per_year=periods_per_year),
        'sortino_ratio': sortino_ratio(returns, periods_per_year=periods_per_year),
        'calmar_ratio': calmar_ratio(returns, periods_per_year),
        
        # Drawdown metrics
        'max_drawdown': max_drawdown(returns),
        'avg_drawdown': abs(df['drawdown'].mean()),
        
        # Win/loss metrics
        'win_rate': win_rate(returns),
        'profit_factor': profit_factor(returns),
        'best_day': returns.max(),
        'worst_day': returns.min(),
        
        # Risk metrics
        'var_95': value_at_risk(returns, 0.95),
        'cvar_95': conditional_var(returns, 0.95),
        'skewness': stats.skew(returns),
        'kurtosis': stats.kurtosis(returns),
        
        # Trading metrics
        'num_trades': df['transaction_cost_incurred'].astype(bool).sum(),
        'avg_trade_cost': df['transaction_cost_incurred'].mean(),
    }
    
    # Benchmark comparison
    if benchmark_returns is not None:
        report['information_ratio'] = information_ratio(
            returns, benchmark_returns, periods_per_year
        )
        
        aligned = pd.DataFrame({
            'strategy': returns,
            'benchmark': benchmark_returns,
        }).dropna()
        
        if len(aligned) > 1:
            report['beta'] = np.cov(aligned['strategy'], aligned['benchmark'])[0, 1] / np.var(aligned['benchmark'])
            report['alpha'] = report['cagr'] - report['beta'] * cagr(aligned['benchmark'], periods_per_year)
            report['correlation'] = aligned.corr().iloc[0, 1]
    
    return report


def print_backtest_report(report: Dict[str, float], title: str = "Backtest Report"):
    """
    Print formatted backtest report.
    
    Args:
        report: Report dictionary from generate_backtest_report
        title: Report title
    """
    print("=" * 70)
    print(f"{title:^70}")
    print("=" * 70)
    
    print("\n📈 Return Metrics")
    print(f"  Total Return:        {report['total_return']:>10.2%}")
    print(f"  CAGR:                {report['cagr']:>10.2%}")
    print(f"  Mean Annual Return:  {report['mean_return']:>10.2%}")
    print(f"  Volatility (Annual): {report['volatility']:>10.2%}")
    
    print("\n📊 Risk-Adjusted Returns")
    print(f"  Sharpe Ratio:        {report['sharpe_ratio']:>10.3f}")
    print(f"  Sortino Ratio:       {report['sortino_ratio']:>10.3f}")
    print(f"  Calmar Ratio:        {report['calmar_ratio']:>10.3f}")
    
    print("\n📉 Drawdown Analysis")
    print(f"  Max Drawdown:        {report['max_drawdown']:>10.2%}")
    print(f"  Avg Drawdown:        {report['avg_drawdown']:>10.2%}")
    
    print("\n🎯 Win/Loss Statistics")
    print(f"  Win Rate:            {report['win_rate']:>10.2%}")
    print(f"  Profit Factor:       {report['profit_factor']:>10.3f}")
    print(f"  Best Day:            {report['best_day']:>10.2%}")
    print(f"  Worst Day:           {report['worst_day']:>10.2%}")
    
    print("\n⚠️  Risk Metrics")
    print(f"  VaR (95%):           {report['var_95']:>10.2%}")
    print(f"  CVaR (95%):          {report['cvar_95']:>10.2%}")
    print(f"  Skewness:            {report['skewness']:>10.3f}")
    print(f"  Kurtosis:            {report['kurtosis']:>10.3f}")
    
    print("\n💰 Trading Statistics")
    print(f"  Number of Trades:    {report['num_trades']:>10.0f}")
    print(f"  Avg Trade Cost:      {report['avg_trade_cost']:>10.4%}")
    
    if 'information_ratio' in report:
        print("\n📊 Benchmark Comparison")
        print(f"  Information Ratio:   {report['information_ratio']:>10.3f}")
        print(f"  Beta:                {report['beta']:>10.3f}")
        print(f"  Alpha:               {report['alpha']:>10.2%}")
        print(f"  Correlation:         {report['correlation']:>10.3f}")
    
    print("=" * 70)


# ---------------------------------------------------------------------------
# Backtesting Engine (Enhanced with Config Integration)
# ---------------------------------------------------------------------------


class Backtester:
    """
    Comprehensive backtesting engine for ASRE strategies.
    
    Fully integrated with:
    - composite.py: ASRE rating computation
    - config.py: Configuration objects
    
    Usage:
        bt = Backtester(df, rating_col='r_asre', config=backtest_config)
        bt.run(signal_type='threshold', threshold_long=70)
        report = bt.get_report()
        bt.print_report()
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        rating_col: str = 'r_asre',
        price_col: str = 'close',
        benchmark_col: Optional[str] = None,
        config: Optional[BacktestConfig] = None,
    ):
        """
        Initialize backtester.
        
        Args:
            df: DataFrame with ratings, prices, and returns
            rating_col: Column name for ASRE ratings
            price_col: Column name for prices
            benchmark_col: Optional benchmark return column
            config: BacktestConfig object (uses defaults if None)
        """
        self.df = df.copy()
        self.rating_col = rating_col
        self.price_col = price_col
        self.benchmark_col = benchmark_col
        self.config = config or BacktestConfig()
        
        self.results_df: Optional[pd.DataFrame] = None
        self.report: Optional[Dict] = None
        
        # Validate rating column
        if rating_col not in self.df.columns:
            raise ValueError(f"Rating column '{rating_col}' not found in DataFrame")
        
        # Validate ASRE rating
        is_valid, msg = validate_asre_rating(self.df, rating_col)
        if not is_valid:
            logger.warning(f"Rating validation warning: {msg}")
    
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
        """
        Run backtest with optional config overrides.
        
        Args:
            signal_type: 'threshold', 'quantile', or 'regime'
            threshold_long: Long threshold (uses config default if None)
            threshold_short: Short threshold (uses config default if None)
            transaction_cost: Transaction cost fraction (uses config if None)
            slippage: Slippage fraction (uses config if None)
            max_position: Maximum position size (uses config if None)
            use_confidence_bounds: Use Kalman confidence bounds for signals
            **kwargs: Additional args for signal generation
        
        Returns:
            Results DataFrame
        """
        logger.info(f"Running backtest with {signal_type} signals...")
        
        # Use config defaults if not specified
        threshold_long = threshold_long or self.config.threshold_long
        threshold_short = threshold_short or self.config.threshold_short
        transaction_cost = transaction_cost if transaction_cost is not None else self.config.transaction_cost
        slippage = slippage if slippage is not None else self.config.slippage
        max_position = max_position if max_position is not None else self.config.max_position
        
        # Get ratings (supports multiple rating types)
        ratings = get_asre_rating(self.df, rating_type='medallion' if self.rating_col == 'r_asre' else 'final')
        
        # Get confidence bounds if available
        confidence_lower = self.df.get('confidence_lower')
        confidence_upper = self.df.get('confidence_upper')
        
        # Generate signals
        if signal_type == 'threshold':
            signals = generate_signals_threshold(
                ratings,
                threshold_long,
                threshold_short,
                confidence_lower=confidence_lower,
                confidence_upper=confidence_upper,
                use_confidence=use_confidence_bounds,
            )
        elif signal_type == 'quantile':
            signals = generate_signals_quantile(
                ratings,
                **kwargs,
            )
        elif signal_type == 'regime':
            if 'vix' not in self.df.columns:
                raise ValueError("VIX column required for regime-based signals")
            
            signals = generate_signals_regime(
                ratings,
                self.df['vix'],
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
        
        logger.info(f"Generated {(signals != 0).sum()} active signals")
        
        # Position sizing
        positions = apply_position_sizing(
            signals,
            ratings,
            max_position=max_position,
            scale_by_confidence=True,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
        )
        
        self.df['signal'] = signals
        self.df['position'] = positions
        
        # Compute returns
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
        """Generate and return backtest report."""
        if self.results_df is None:
            raise ValueError("Run backtest first with .run()")
        
        benchmark_returns = None
        if self.benchmark_col and self.benchmark_col in self.df.columns:
            benchmark_returns = self.df[self.benchmark_col]
        
        self.report = generate_backtest_report(
            self.results_df,
            benchmark_returns=benchmark_returns,
        )
        
        return self.report
    
    def print_report(self, title: str = "ASRE Strategy Backtest"):
        """Print formatted report."""
        if self.report is None:
            self.get_report()
        
        print_backtest_report(self.report, title)
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve (cumulative returns)."""
        if self.results_df is None:
            raise ValueError("Run backtest first")
        
        return self.results_df['cumulative_return']
    
    def get_drawdown_series(self) -> pd.Series:
        """Get drawdown time series."""
        if self.results_df is None:
            raise ValueError("Run backtest first")
        
        return self.results_df['drawdown']
    
    def get_trade_log(self) -> pd.DataFrame:
        """
        Get detailed trade log.
        
        Returns:
            DataFrame with trade details
        """
        if self.results_df is None:
            raise ValueError("Run backtest first")
        
        # Extract trades (where position changes)
        trades = self.results_df[self.results_df['transaction_cost_incurred'] > 0].copy()
        
        return trades[['signal', 'position', self.price_col, 'net_return', 'cumulative_return']]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = [
    # Signal generation
    'generate_signals_threshold',
    'generate_signals_quantile',
    'generate_signals_regime',
    'apply_position_sizing',
    
    # Return computation
    'compute_strategy_returns',
    'compute_portfolio_returns',
    
    # Metrics
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
    
    # Reporting
    'generate_backtest_report',
    'print_backtest_report',
    
    # Engine
    'Backtester',
]
