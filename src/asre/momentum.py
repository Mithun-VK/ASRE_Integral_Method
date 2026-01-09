"""
ASRE Momentum Score (M-Score) Implementation

Implements the complete M-Score formula:

M(t) = 50 + 50 * [∫₀ᵗ R(τ)·e^(-κ(t-τ)) dτ / √(∫₀ᵗ R²(τ)·e^(-2κ(t-τ)) dτ)] + β_m · ρ_autocorr(60)

Components:
1. Log returns R(t) = log(P(t)/P(t-60))
2. Exponential decay convolution (numerator)
3. Volatility normalization (denominator)
4. Autocorrelation mean-reversion term
5. Sharpe adjustment: M_adj(t) = M(t) * √(0.15/σ_60d(t))

Production-grade features:
- Exact formula implementation from ASRE specification
- Numerical stability (clipping, epsilon handling)
- Vectorized operations for performance
- Comprehensive validation
- Memory-efficient rolling calculations
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import MomentumConfig
from .indicators import (
    log_returns,
    rolling_autocorrelation,
    rolling_volatility,
    exponential_decay_convolution,
    volatility_normalization,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# M-Score Core Implementation
# ---------------------------------------------------------------------------


def compute_momentum_score(
    df: pd.DataFrame,
    config: Optional[MomentumConfig] = None,
    return_components: bool = False,
) -> pd.DataFrame:
    """
    Compute Momentum Score (M-Score) for a stock.
    
    Formula: M(t) = 50 + 50 * [Numerator / Denominator] + β_m · ρ_autocorr(60)
    
    Args:
        df: DataFrame with 'close' price column
        config: MomentumConfig object (uses defaults if None)
        return_components: If True, return all intermediate components
    
    Returns:
        DataFrame with added columns:
        - m_score: Base momentum score [0, 100]
        - m_score_adj: Sharpe-adjusted momentum score
        
        If return_components=True, also includes:
        - log_returns_60d: 60-day log returns
        - decay_convolution: Exponential decay convolution (numerator)
        - vol_normalization: Volatility normalization (denominator)
        - autocorr_60d: 60-day autocorrelation
        - vol_60d: 60-day rolling volatility
    
    Raises:
        ValueError: If required columns missing
    """
    
    if config is None:
        config = MomentumConfig()
    
    # Validate input
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
    
    if len(df) < config.window_60d:
        logger.warning(
            f"DataFrame has {len(df)} rows, but M-Score needs at least "
            f"{config.window_60d} rows for reliable calculation"
        )
    
    result_df = df.copy()
    
    # Step 1: Calculate 60-day log returns
    logger.debug("Computing 60-day log returns...")
    log_returns_60d = log_returns(df['close'], periods=config.window_60d, fillna=False)
    
    # Step 2: Exponential decay convolution (numerator)
    logger.debug(f"Computing exponential decay convolution (κ={config.kappa})...")
    numerator = exponential_decay_convolution(
        log_returns_60d,
        kappa=config.kappa,
        window=config.window_60d,
    )
    
    # Step 3: Volatility normalization (denominator)
    logger.debug("Computing volatility normalization...")
    denominator = volatility_normalization(
        log_returns_60d,
        kappa=config.kappa,
        window=config.window_60d,
    )
    
    # Step 4: Compute ratio (with numerical stability)
    ratio = numerator / (denominator + 1e-10)
    
    # Clip extreme values for stability
    ratio = np.clip(ratio, -3, 3)  # ±3 sigma bounds
    
    # Step 5: Calculate 60-day autocorrelation (mean reversion)
    logger.debug("Computing autocorrelation for mean reversion...")
    # Use daily returns for autocorrelation, not 60-day returns
    daily_returns = log_returns(df['close'], periods=1, fillna=False)
    autocorr_60d = rolling_autocorrelation(
        daily_returns,
        lag=60,
        window=config.window_60d,
    )
    
    # Fill NaN in autocorrelation with 0 (neutral)
    autocorr_60d = autocorr_60d.fillna(0.0)
    
    # Step 6: Assemble base M-Score
    logger.debug("Assembling M-Score...")
    m_score = 50 + 50 * ratio + config.beta_m * autocorr_60d
    
    # Clip to valid range [0, 100]
    m_score = np.clip(m_score, 0, 100)
    
    # Step 7: Sharpe adjustment
    logger.debug("Applying Sharpe adjustment...")
    vol_60d = rolling_volatility(daily_returns, window=config.window_60d)
    
    # Sharpe adjustment factor: sqrt(0.15 / σ_60d)
    sharpe_factor = np.sqrt(0.15 / (vol_60d + 1e-10))
    
    # Clip sharpe factor to reasonable range
    sharpe_factor = np.clip(sharpe_factor, 0.5, 2.0)
    
    m_score_adj = m_score * sharpe_factor
    
    # Clip adjusted score to [0, 100]
    m_score_adj = np.clip(m_score_adj, 0, 100)
    
    # Add to result DataFrame
    result_df['m_score'] = m_score
    result_df['m_score_adj'] = m_score_adj
    
    # Optionally return all components
    if return_components:
        result_df['log_returns_60d'] = log_returns_60d
        result_df['decay_convolution'] = numerator
        result_df['vol_normalization'] = denominator
        result_df['autocorr_60d'] = autocorr_60d
        result_df['vol_60d'] = vol_60d
        result_df['sharpe_factor'] = sharpe_factor
        result_df['momentum_ratio'] = ratio
    
    logger.info(
        f"M-Score computed: mean={m_score_adj.mean():.2f}, "
        f"std={m_score_adj.std():.2f}, "
        f"range=[{m_score_adj.min():.2f}, {m_score_adj.max():.2f}]"
    )
    
    return result_df


# ---------------------------------------------------------------------------
# Alternative implementation with explicit formula breakdown
# ---------------------------------------------------------------------------


class MomentumScoreCalculator:
    """
    Object-oriented M-Score calculator for step-by-step computation.
    
    Useful for debugging, analysis, and understanding component contributions.
    
    Usage:
        calc = MomentumScoreCalculator(df, config)
        calc.compute()
        print(calc.m_score)
        print(calc.components)  # Access all intermediate values
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[MomentumConfig] = None,
    ):
        """
        Initialize calculator.
        
        Args:
            df: DataFrame with 'close' column
            config: MomentumConfig (uses defaults if None)
        """
        if config is None:
            config = MomentumConfig()
        
        self.df = df
        self.config = config
        self.prices = df['close']
        
        # Results (populated after compute())
        self.log_returns_60d: Optional[pd.Series] = None
        self.numerator: Optional[pd.Series] = None
        self.denominator: Optional[pd.Series] = None
        self.ratio: Optional[pd.Series] = None
        self.autocorr_60d: Optional[pd.Series] = None
        self.vol_60d: Optional[pd.Series] = None
        self.sharpe_factor: Optional[pd.Series] = None
        self.m_score: Optional[pd.Series] = None
        self.m_score_adj: Optional[pd.Series] = None
    
    def compute_log_returns(self) -> pd.Series:
        """
        Step 1: Compute 60-day log returns.
        
        Formula: R(t) = log(P(t) / P(t-60))
        """
        self.log_returns_60d = log_returns(
            self.prices,
            periods=self.config.window_60d,
            fillna=False,
        )
        logger.debug(f"Log returns: {len(self.log_returns_60d.dropna())} valid values")
        return self.log_returns_60d
    
    def compute_numerator(self) -> pd.Series:
        """
        Step 2: Exponential decay convolution (numerator).
        
        Formula: ∫₀ᵗ R(τ)·e^(-κ(t-τ)) dτ
        
        Approximation: Weighted sum with exponential decay weights.
        """
        if self.log_returns_60d is None:
            self.compute_log_returns()
        
        self.numerator = exponential_decay_convolution(
            self.log_returns_60d,
            kappa=self.config.kappa,
            window=self.config.window_60d,
        )
        
        logger.debug(
            f"Numerator: mean={self.numerator.mean():.6f}, "
            f"std={self.numerator.std():.6f}"
        )
        return self.numerator
    
    def compute_denominator(self) -> pd.Series:
        """
        Step 3: Volatility normalization (denominator).
        
        Formula: √(∫₀ᵗ R²(τ)·e^(-2κ(t-τ)) dτ)
        """
        if self.log_returns_60d is None:
            self.compute_log_returns()
        
        self.denominator = volatility_normalization(
            self.log_returns_60d,
            kappa=self.config.kappa,
            window=self.config.window_60d,
        )
        
        logger.debug(
            f"Denominator: mean={self.denominator.mean():.6f}, "
            f"std={self.denominator.std():.6f}"
        )
        return self.denominator
    
    def compute_ratio(self) -> pd.Series:
        """
        Step 4: Compute ratio of numerator / denominator.
        
        This is the core momentum signal (before scaling).
        """
        if self.numerator is None:
            self.compute_numerator()
        if self.denominator is None:
            self.compute_denominator()
        
        self.ratio = self.numerator / (self.denominator + 1e-10)
        
        # Clip extreme values
        self.ratio = np.clip(self.ratio, -3, 3)
        
        logger.debug(
            f"Momentum ratio: mean={self.ratio.mean():.4f}, "
            f"std={self.ratio.std():.4f}"
        )
        return self.ratio
    
    def compute_autocorrelation(self) -> pd.Series:
        """
        Step 5: Compute 60-day autocorrelation (mean reversion signal).
        
        Formula: ρ_autocorr(k) = Σ(R_i - R̄)(R_{i-k} - R̄) / Σ(R_i - R̄)²
        
        Negative autocorrelation indicates mean reversion.
        """
        # Use daily returns for autocorrelation
        daily_returns = log_returns(self.prices, periods=1, fillna=False)
        
        self.autocorr_60d = rolling_autocorrelation(
            daily_returns,
            lag=60,
            window=self.config.window_60d,
        )
        
        # Fill NaN with 0 (neutral)
        self.autocorr_60d = self.autocorr_60d.fillna(0.0)
        
        logger.debug(
            f"Autocorrelation: mean={self.autocorr_60d.mean():.4f}, "
            f"range=[{self.autocorr_60d.min():.4f}, {self.autocorr_60d.max():.4f}]"
        )
        return self.autocorr_60d
    
    def compute_base_score(self) -> pd.Series:
        """
        Step 6: Assemble base M-Score.
        
        Formula: M(t) = 50 + 50 * ratio + β_m · ρ_autocorr(60)
        
        Range: [0, 100]
        """
        if self.ratio is None:
            self.compute_ratio()
        if self.autocorr_60d is None:
            self.compute_autocorrelation()
        
        # Center at 50, scale ratio by 50
        self.m_score = 50 + 50 * self.ratio + self.config.beta_m * self.autocorr_60d
        
        # Clip to valid range
        self.m_score = np.clip(self.m_score, 0, 100)
        
        logger.debug(
            f"Base M-Score: mean={self.m_score.mean():.2f}, "
            f"std={self.m_score.std():.2f}"
        )
        return self.m_score
    
    def compute_sharpe_adjustment(self) -> pd.Series:
        """
        Step 7: Apply Sharpe adjustment.
        
        Formula: M_adj(t) = M(t) * √(0.15 / σ_60d(t))
        
        Penalizes high volatility, rewards stable momentum.
        """
        if self.m_score is None:
            self.compute_base_score()
        
        # Calculate 60-day volatility
        daily_returns = log_returns(self.prices, periods=1, fillna=False)
        self.vol_60d = rolling_volatility(daily_returns, window=self.config.window_60d)
        
        # Sharpe factor: sqrt(target_vol / actual_vol)
        self.sharpe_factor = np.sqrt(0.15 / (self.vol_60d + 1e-10))
        
        # Clip to reasonable range
        self.sharpe_factor = np.clip(self.sharpe_factor, 0.5, 2.0)
        
        # Apply adjustment
        self.m_score_adj = self.m_score * self.sharpe_factor
        
        # Clip adjusted score
        self.m_score_adj = np.clip(self.m_score_adj, 0, 100)
        
        logger.debug(
            f"Sharpe-adjusted M-Score: mean={self.m_score_adj.mean():.2f}, "
            f"std={self.m_score_adj.std():.2f}"
        )
        return self.m_score_adj
    
    def compute(self) -> pd.Series:
        """
        Run complete M-Score calculation pipeline.
        
        Returns:
            Sharpe-adjusted M-Score series
        """
        logger.info("Computing Momentum Score (M-Score)...")
        
        self.compute_log_returns()
        self.compute_numerator()
        self.compute_denominator()
        self.compute_ratio()
        self.compute_autocorrelation()
        self.compute_base_score()
        self.compute_sharpe_adjustment()
        
        logger.info("M-Score computation complete")
        
        return self.m_score_adj
    
    @property
    def components(self) -> dict:
        """
        Get all computed components as dictionary.
        
        Returns:
            Dict with all intermediate values
        """
        return {
            'log_returns_60d': self.log_returns_60d,
            'numerator': self.numerator,
            'denominator': self.denominator,
            'ratio': self.ratio,
            'autocorr_60d': self.autocorr_60d,
            'vol_60d': self.vol_60d,
            'sharpe_factor': self.sharpe_factor,
            'm_score': self.m_score,
            'm_score_adj': self.m_score_adj,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Export all components to DataFrame.
        
        Returns:
            DataFrame with all M-Score components
        """
        result = self.df.copy()
        
        for name, series in self.components.items():
            if series is not None:
                result[name] = series
        
        return result


# ---------------------------------------------------------------------------
# Validation and Quality Checks
# ---------------------------------------------------------------------------


def validate_momentum_score(
    df: pd.DataFrame,
    score_col: str = 'm_score_adj',
) -> Tuple[bool, str]:
    """
    Validate M-Score computation quality.
    
    Checks:
    - Score in valid range [0, 100]
    - Not too many NaN values
    - Reasonable distribution (not all same value)
    - No infinite values
    
    Args:
        df: DataFrame with M-Score
        score_col: Column name for M-Score
    
    Returns:
        Tuple of (is_valid, message)
    """
    if score_col not in df.columns:
        return False, f"Column '{score_col}' not found"
    
    score = df[score_col].dropna()
    
    if len(score) == 0:
        return False, "All M-Score values are NaN"
    
    # Check range
    if (score < 0).any() or (score > 100).any():
        return False, f"M-Score out of range [0, 100]: [{score.min():.2f}, {score.max():.2f}]"
    
    # Check for infinite values
    if np.isinf(score).any():
        return False, "M-Score contains infinite values"
    
    # Check NaN percentage
    nan_pct = (df[score_col].isna().sum() / len(df)) * 100
    if nan_pct > 50:
        return False, f"Too many NaN values: {nan_pct:.1f}%"
    
    # Check variance (not all same value)
    if score.std() < 0.1:
        return False, f"M-Score has no variance: std={score.std():.4f}"
    
    return True, f"M-Score valid: mean={score.mean():.2f}, std={score.std():.2f}"


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def compute_momentum_score_simple(
    prices: pd.Series,
    kappa: float = 0.03,
    beta_m: float = 0.2,
    window: int = 60,
) -> pd.Series:
    """
    Simplified M-Score computation from price series.
    
    Args:
        prices: Price series
        kappa: Decay rate
        beta_m: Mean reversion weight
        window: Window size
    
    Returns:
        Sharpe-adjusted M-Score series
    """
    df = pd.DataFrame({'close': prices})
    config = MomentumConfig(kappa=kappa, beta_m=beta_m, window_60d=window)
    result_df = compute_momentum_score(df, config)
    return result_df['m_score_adj']


def momentum_signal(
    m_score: pd.Series,
    threshold_long: float = 70.0,
    threshold_short: float = 30.0,
) -> pd.Series:
    """
    Generate trading signals from M-Score.
    
    Args:
        m_score: M-Score series
        threshold_long: Long threshold
        threshold_short: Short threshold
    
    Returns:
        Signal series: 1 (long), 0 (neutral), -1 (short)
    """
    signals = pd.Series(0, index=m_score.index)
    signals[m_score >= threshold_long] = 1
    signals[m_score <= threshold_short] = -1
    return signals


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = [
    'compute_momentum_score',
    'MomentumScoreCalculator',
    'validate_momentum_score',
    'compute_momentum_score_simple',
    'momentum_signal',
]
