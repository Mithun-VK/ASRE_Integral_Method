"""
ASRE Momentum Score (M-Score) Implementation (ENHANCED)

✅ ENHANCEMENTS:
- Rolling z-score normalization (prevents saturation)
- Adaptive volatility floor (prevents flat distributions)
- Soft clamping (preserves regime sensitivity)
- Numerical safety throughout
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
# Numerical Safety Utilities (ENHANCED)
# ---------------------------------------------------------------------------

def safe_rolling_zscore_momentum(
    series: pd.Series,
    window: int = 60,
    min_periods: int = 30,
    vol_floor: float = 0.01,
) -> pd.Series:
    """
    Compute rolling z-score with adaptive volatility floor for momentum.

    Prevents saturation by using rolling statistics instead of global normalization.

    Args:
        series: Input series (e.g., momentum ratio)
        window: Rolling window size
        min_periods: Minimum observations required
        vol_floor: Minimum volatility (adaptive to data scale)

    Returns:
        Z-score series with numerical safety
    """
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()

    # Adaptive volatility floor based on global statistics
    global_std = series.std()
    adaptive_floor = max(vol_floor, global_std * 0.05)  # 5% of global std

    rolling_std_safe = rolling_std.clip(lower=adaptive_floor)

    # Compute z-score
    z_score = (series - rolling_mean) / rolling_std_safe

    # Fill initial NaN with global z-score
    if z_score.isna().any():
        global_mean = series.mean()
        global_std_safe = max(series.std(), adaptive_floor)
        global_z = (series - global_mean) / global_std_safe
        z_score = z_score.fillna(global_z)

    return z_score


def soft_clamp_momentum(
    z: pd.Series,
    lower: float = -3.0,
    upper: float = 3.0,
    smoothness: float = 0.3,
) -> pd.Series:
    """
    Soft clamping using smooth sigmoid transition for momentum scores.

    More aggressive smoothness for momentum to prevent oscillations.

    Args:
        z: Z-score series
        lower: Lower threshold
        upper: Upper threshold
        smoothness: Transition smoothness (0.1-0.5)

    Returns:
        Soft-clamped z-score
    """
    # Soft lower bound
    z_lower = lower + smoothness * np.log(1 + np.exp((z - lower) / smoothness))

    # Soft upper bound
    z_clamped = upper - smoothness * np.log(1 + np.exp((upper - z_lower) / smoothness))

    return z_clamped


# ---------------------------------------------------------------------------
# M-Score Core Implementation (ENHANCED)
# ---------------------------------------------------------------------------

def compute_momentum_score(
    df: pd.DataFrame,
    config: Optional[MomentumConfig] = None,
    return_components: bool = False,
) -> pd.DataFrame:
    """
    Compute Momentum Score (M-Score) with saturation prevention.

    ✅ ENHANCEMENTS:
    - Rolling z-score normalization of momentum ratio
    - Adaptive volatility floor for denominator
    - Soft clamping of extreme values
    - Numerical stability throughout

    Formula: M(t) = 50 + 50 * [Numerator / Denominator] + β_m · ρ_autocorr(60)

    Args:
        df: DataFrame with 'close' price column
        config: MomentumConfig object (uses defaults if None)
        return_components: If True, return all intermediate components

    Returns:
        DataFrame with added columns:
        - m_score: Base momentum score [0, 100]
        - m_score_adj: Sharpe-adjusted momentum score

    Raises:
        ValueError: If required columns missing
    """

    if config is None:
        config = MomentumConfig()

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

    # -------------------------------------------------------------------
    # ✅ FIX 1: ADAPTIVE VOLATILITY FLOOR FOR DENOMINATOR
    # -------------------------------------------------------------------
    logger.debug("Computing volatility normalization with adaptive floor...")
    denominator = volatility_normalization(
        log_returns_60d,
        kappa=config.kappa,
        window=config.window_60d,
    )

    # Apply adaptive floor to prevent near-zero denominator
    global_vol = log_returns_60d.std()
    adaptive_floor = max(0.001, global_vol * 0.01)  # 1% of global vol
    denominator_safe = denominator.clip(lower=adaptive_floor)

    # -------------------------------------------------------------------
    # ✅ FIX 2: COMPUTE RATIO WITH SAFE DENOMINATOR
    # -------------------------------------------------------------------
    ratio_raw = numerator / denominator_safe

    # -------------------------------------------------------------------
    # ✅ FIX 3: ROLLING Z-SCORE NORMALIZATION (not hard clipping!)
    # -------------------------------------------------------------------
    logger.debug("Applying rolling z-score normalization to momentum ratio...")

    ratio_zscore = safe_rolling_zscore_momentum(
        ratio_raw,
        window=60,
        min_periods=30,
        vol_floor=0.01,
    )

    # -------------------------------------------------------------------
    # ✅ FIX 4: SOFT CLAMPING (preserves extreme moves)
    # -------------------------------------------------------------------
    ratio_clamped = soft_clamp_momentum(
        ratio_zscore,
        lower=-2.5,
        upper=2.5,
        smoothness=0.3,
    )

    # Step 5: Calculate 60-day autocorrelation (mean reversion)
    logger.debug("Computing autocorrelation for mean reversion...")
    daily_returns = log_returns(df['close'], periods=1, fillna=False)
    autocorr_60d = rolling_autocorrelation(
        daily_returns,
        lag=60,
        window=config.window_60d,
    )
    autocorr_60d = autocorr_60d.fillna(0.0)

    # -------------------------------------------------------------------
    # ✅ FIX 5: ASSEMBLE WITH ROLLING-NORMALIZED RATIO
    # -------------------------------------------------------------------
    logger.debug("Assembling M-Score with normalized components...")

    # Map clamped z-score to momentum contribution
    # Use tanh for smooth bounded transformation
    momentum_component = 50 * np.tanh(ratio_clamped / 2.0)

    m_score = 50 + momentum_component + config.beta_m * autocorr_60d

    # Clip to valid range [0, 100]
    m_score = np.clip(m_score, 0, 100)

    # -------------------------------------------------------------------
    # ✅ FIX 6: SHARPE ADJUSTMENT WITH ADAPTIVE FLOOR
    # -------------------------------------------------------------------
    logger.debug("Applying Sharpe adjustment with numerical safety...")
    vol_60d = rolling_volatility(daily_returns, window=config.window_60d)

    # Adaptive floor for volatility
    vol_floor_adaptive = max(0.01, daily_returns.std() * 0.05)
    vol_60d_safe = vol_60d.clip(lower=vol_floor_adaptive)

    # Sharpe adjustment factor with safe volatility
    sharpe_factor = np.sqrt(0.15 / vol_60d_safe)
    sharpe_factor = np.clip(sharpe_factor, 0.5, 2.0)

    m_score_adj = m_score * sharpe_factor
    m_score_adj = np.clip(m_score_adj, 0, 100)

    # Add to result DataFrame
    result_df['m_score'] = m_score
    result_df['m_score_adj'] = m_score_adj

    # Optionally return all components
    if return_components:
        result_df['log_returns_60d'] = log_returns_60d
        result_df['decay_convolution'] = numerator
        result_df['vol_normalization'] = denominator_safe
        result_df['ratio_raw'] = ratio_raw
        result_df['ratio_zscore'] = ratio_zscore
        result_df['ratio_clamped'] = ratio_clamped
        result_df['momentum_component'] = momentum_component
        result_df['autocorr_60d'] = autocorr_60d
        result_df['vol_60d'] = vol_60d_safe
        result_df['sharpe_factor'] = sharpe_factor

    logger.info(
        f"M-Score computed: mean={m_score_adj.mean():.2f}, "
        f"std={m_score_adj.std():.2f}, "
        f"range=[{m_score_adj.min():.2f}, {m_score_adj.max():.2f}]"
    )

    return result_df


# ---------------------------------------------------------------------------
# Alternative implementation with explicit formula breakdown (ENHANCED)
# ---------------------------------------------------------------------------

class MomentumScoreCalculator:
    """
    Object-oriented M-Score calculator with saturation fixes.

    Usage:
        calc = MomentumScoreCalculator(df, config)
        calc.compute()
        print(calc.m_score_adj)
    """

    def __init__(self, df: pd.DataFrame, config: Optional[MomentumConfig] = None):
        if config is None:
            config = MomentumConfig()

        self.df = df
        self.config = config
        self.prices = df['close']

        # Results
        self.log_returns_60d: Optional[pd.Series] = None
        self.numerator: Optional[pd.Series] = None
        self.denominator: Optional[pd.Series] = None
        self.ratio_raw: Optional[pd.Series] = None
        self.ratio_zscore: Optional[pd.Series] = None
        self.ratio_clamped: Optional[pd.Series] = None
        self.autocorr_60d: Optional[pd.Series] = None
        self.vol_60d: Optional[pd.Series] = None
        self.sharpe_factor: Optional[pd.Series] = None
        self.m_score: Optional[pd.Series] = None
        self.m_score_adj: Optional[pd.Series] = None

    def compute(self) -> pd.Series:
        """Run complete M-Score calculation with enhancements."""
        logger.info("Computing Momentum Score (M-Score) - Enhanced...")

        # Step 1: Log returns
        self.log_returns_60d = log_returns(
            self.prices,
            periods=self.config.window_60d,
            fillna=False,
        )

        # Step 2: Numerator
        self.numerator = exponential_decay_convolution(
            self.log_returns_60d,
            kappa=self.config.kappa,
            window=self.config.window_60d,
        )

        # Step 3: Denominator with adaptive floor
        self.denominator = volatility_normalization(
            self.log_returns_60d,
            kappa=self.config.kappa,
            window=self.config.window_60d,
        )

        global_vol = self.log_returns_60d.std()
        adaptive_floor = max(0.001, global_vol * 0.01)
        self.denominator = self.denominator.clip(lower=adaptive_floor)

        # Step 4: Raw ratio
        self.ratio_raw = self.numerator / self.denominator

        # Step 5: Rolling z-score normalization
        self.ratio_zscore = safe_rolling_zscore_momentum(
            self.ratio_raw,
            window=60,
            min_periods=30,
        )

        # Step 6: Soft clamping
        self.ratio_clamped = soft_clamp_momentum(
            self.ratio_zscore,
            lower=-2.5,
            upper=2.5,
        )

        # Step 7: Autocorrelation
        daily_returns = log_returns(self.prices, periods=1, fillna=False)
        self.autocorr_60d = rolling_autocorrelation(
            daily_returns,
            lag=60,
            window=self.config.window_60d,
        )
        self.autocorr_60d = self.autocorr_60d.fillna(0.0)

        # Step 8: Assemble base score
        momentum_component = 50 * np.tanh(self.ratio_clamped / 2.0)
        self.m_score = 50 + momentum_component + self.config.beta_m * self.autocorr_60d
        self.m_score = np.clip(self.m_score, 0, 100)

        # Step 9: Sharpe adjustment
        self.vol_60d = rolling_volatility(daily_returns, window=self.config.window_60d)
        vol_floor = max(0.01, daily_returns.std() * 0.05)
        self.vol_60d = self.vol_60d.clip(lower=vol_floor)

        self.sharpe_factor = np.sqrt(0.15 / self.vol_60d)
        self.sharpe_factor = np.clip(self.sharpe_factor, 0.5, 2.0)

        self.m_score_adj = self.m_score * self.sharpe_factor
        self.m_score_adj = np.clip(self.m_score_adj, 0, 100)

        logger.info("M-Score computation complete (enhanced)")

        return self.m_score_adj


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_momentum_score(
    df: pd.DataFrame,
    score_col: str = 'm_score_adj',
) -> Tuple[bool, str]:
    """Validate M-Score computation quality."""
    if score_col not in df.columns:
        return False, f"Column '{score_col}' not found"

    score = df[score_col].dropna()

    if len(score) == 0:
        return False, "All M-Score values are NaN"

    if (score < 0).any() or (score > 100).any():
        return False, f"M-Score out of range: [{score.min():.2f}, {score.max():.2f}]"

    if np.isinf(score).any():
        return False, "M-Score contains infinite values"

    nan_pct = (df[score_col].isna().sum() / len(df)) * 100
    if nan_pct > 50:
        return False, f"Too many NaN values: {nan_pct:.1f}%"

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
    """Simplified M-Score computation from price series."""
    df = pd.DataFrame({'close': prices})
    config = MomentumConfig(kappa=kappa, beta_m=beta_m, window_60d=window)
    result_df = compute_momentum_score(df, config)
    return result_df['m_score_adj']


def momentum_signal(
    m_score: pd.Series,
    threshold_long: float = 70.0,
    threshold_short: float = 30.0,
) -> pd.Series:
    """Generate trading signals from M-Score."""
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
    'safe_rolling_zscore_momentum',
    'soft_clamp_momentum',
]
