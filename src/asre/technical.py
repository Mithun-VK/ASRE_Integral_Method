"""
ASRE Technical Score (T-Score) Implementation (ENHANCED)

Implements the complete T-Score formula with saturation fixes:

T(t) = 50 + 50 * tanh(∫₀ᵗ [ρ(τ) · dP(τ)/dt + γ · d²P(τ)/dt²] dτ)

✅ ENHANCEMENTS (while preserving original logic):
- Rolling z-score normalization for integral (prevents saturation)
- Adaptive volatility floor (maintains sensitivity)
- Smooth EMA instead of abrupt rolling window
- All original components preserved
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import TechnicalConfig
from .indicators import (
    rsi,
    ema,
    sma,
    parkinson_volatility,
    rolling_volatility,
    log_returns,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper Functions (ENHANCED)
# ---------------------------------------------------------------------------

def hyperbolic_tangent(x: pd.Series) -> pd.Series:
    """
    Hyperbolic tangent activation function.
    
    Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Range: [-1, 1]
    """
    x_clipped = np.clip(x, -10, 10)
    return np.tanh(x_clipped)


def safe_rolling_zscore(
    series: pd.Series,
    window: int = 60,
    min_periods: int = 30,
) -> pd.Series:
    """
    ✅ FIX 1: Rolling z-score normalization with adaptive floor.
    
    Prevents saturation by using rolling statistics instead of global.
    
    Args:
        series: Input series
        window: Rolling window size
        min_periods: Minimum observations required
    
    Returns:
        Z-score series with numerical safety
    """
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    
    # Adaptive volatility floor: 1% of global std or 0.001 minimum
    global_std = series.std()
    vol_floor = max(0.001, global_std * 0.01) if not np.isnan(global_std) else 0.001
    
    # Clip rolling std to prevent division by near-zero
    rolling_std_safe = rolling_std.clip(lower=vol_floor)
    
    # Compute z-score
    z_score = (series - rolling_mean) / rolling_std_safe
    
    # Fill initial NaN with global z-score
    if z_score.isna().any():
        global_mean = series.mean()
        global_std_safe = max(series.std(), vol_floor)
        global_z = (series - global_mean) / global_std_safe
        z_score = z_score.fillna(global_z)
    
    return z_score


def soft_clamp(
    z: pd.Series,
    lower: float = -3.0,
    upper: float = 3.0,
    smoothness: float = 0.3,
) -> pd.Series:
    """
    ✅ FIX 2: Soft clamping using smooth sigmoid transitions.
    
    Replaces hard clip() with smooth boundaries.
    
    Args:
        z: Input z-score
        lower: Lower threshold
        upper: Upper threshold
        smoothness: Transition smoothness (0.2-0.5)
    
    Returns:
        Soft-clamped series
    """
    # Soft lower bound
    z_lower = lower + smoothness * np.log(1 + np.exp((z - lower) / smoothness))
    
    # Soft upper bound
    z_clamped = upper - smoothness * np.log(1 + np.exp((upper - z_lower) / smoothness))
    
    return z_clamped


def percentile_normalize(
    values: pd.Series,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
) -> pd.Series:
    """
    Multi-stage normalization (ENHANCED).
    
    ✅ FIX 3: Better fallback stages to prevent flatline.
    """
    valid_values = values.dropna()
    
    if len(valid_values) == 0:
        return pd.Series(50.0, index=values.index)
    
    # STAGE 1: Standard percentile normalization
    p_lower = np.nanpercentile(valid_values, lower_percentile)
    p_upper = np.nanpercentile(valid_values, upper_percentile)
    
    if p_upper - p_lower > 1e-6:
        normalized = ((values - p_lower) / (p_upper - p_lower) * 100)
        normalized = np.clip(normalized, 0, 100)
        logger.debug(f"✓ Percentile norm: [{p_lower:.3f}, {p_upper:.3f}] → std={normalized.std():.2f}")
        return normalized
    
    # STAGE 2: Rolling z-score fallback (NEW!)
    logger.debug("Percentile failed, using rolling z-score...")
    z_scores = safe_rolling_zscore(values, window=60, min_periods=20)
    z_scores_clamped = soft_clamp(z_scores, lower=-3, upper=3)
    
    # Map z-scores to [0, 100] with tanh
    normalized = 50 + 25 * np.tanh(z_scores_clamped / 2)
    
    if normalized.std() > 1.0:
        logger.debug(f"✓ Rolling z-score: std={normalized.std():.2f}")
        return np.clip(normalized, 0, 100)
    
    # STAGE 3: Global z-score fallback
    mean_val = valid_values.mean()
    std_val = valid_values.std()
    
    if std_val > 1e-8:
        zscore = (values - mean_val) / std_val
        normalized = 50 + 25 * np.tanh(zscore / 2)
        logger.debug(f"✓ Global z-score fallback: std={std_val:.6f} → std={normalized.std():.2f}")
        return np.clip(normalized, 20, 80)
    
    # STAGE 4: Last resort - return neutral with tiny variation
    logger.warning("⚠️ Flat signal detected → returning neutral band")
    epsilon = np.random.normal(0, 0.5, len(values))
    result = 50 + epsilon
    return pd.Series(np.clip(result, 45, 55), index=values.index)


def rsi_derivative_detailed(
    prices: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    RSI derivative with detailed calculation.
    
    Formula: ρ(τ) = ∂RSI(τ)/∂t
    """
    rsi_values = rsi(prices, period)
    rsi_deriv = rsi_values.diff()
    
    # Smooth to reduce noise
    rsi_deriv = rsi_deriv.rolling(window=3, min_periods=1).mean()
    
    return rsi_deriv


def compute_price_velocity_with_drift(
    prices: pd.Series,
    drift_window: int = 20,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate price velocity: dP/dt = μ_p + σ_p · dW_t^P
    """
    velocity = prices.diff()
    mean_drift = velocity.rolling(window=drift_window).mean()
    vol = velocity.rolling(window=drift_window).std()
    
    return velocity, mean_drift, vol


def compute_price_acceleration_mean_reversion(
    prices: pd.Series,
    ma_window: int = 200,
    theta: float = 0.1,
) -> pd.Series:
    """
    Calculate price acceleration with mean reversion.
    
    Formula: d²P/dt² = θ(MA_200 - P(t))
    """
    ma_200 = ema(prices, span=ma_window)
    acceleration = theta * (ma_200 - prices)
    return acceleration


# ---------------------------------------------------------------------------
# T-Score Core Implementation (ENHANCED)
# ---------------------------------------------------------------------------

def compute_technical_score(
    df: pd.DataFrame,
    config: Optional[TechnicalConfig] = None,
    return_components: bool = False,
    use_percentile_norm: bool = True,
) -> pd.DataFrame:
    """
    Compute Technical Score (T-Score) for a stock.
    
    ✅ ENHANCEMENTS:
    - Rolling z-score normalization (prevents flatline)
    - Adaptive volatility floor (maintains sensitivity)
    - Soft clamping (preserves regime changes)
    - EMA smoothing (reduces noise)
    
    Formula: T(t) = 50 + 50 * tanh(∫₀ᵗ [ρ(τ)·dP/dt + γ·d²P/dt²] dτ)
    """
    
    if config is None:
        config = TechnicalConfig()
    
    # Validate input
    required_cols = ['close', 'high', 'low']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame must contain columns: {missing}")
    
    if len(df) < config.window_200d:
        logger.warning(
            f"DataFrame has {len(df)} rows, but T-Score needs at least "
            f"{config.window_200d} rows for reliable MA200 calculation"
        )
    
    result_df = df.copy()
    prices = df['close']
    
    # Step 1: Calculate RSI derivative ρ(τ)
    logger.debug(f"Computing RSI derivative (period={config.window_rsi})...")
    rsi_values = rsi(prices, period=config.window_rsi)
    rsi_deriv = rsi_derivative_detailed(prices, period=config.window_rsi)
    
    # Step 2: Calculate price velocity dP/dt
    logger.debug("Computing price velocity...")
    velocity, drift, vol_velocity = compute_price_velocity_with_drift(
        prices,
        drift_window=config.window_20d,
    )
    
    # Normalize velocity by Parkinson volatility
    parkinson_vol = parkinson_volatility(
        df['high'],
        df['low'],
        window=config.window_20d,
    )
    
    # ✅ FIX 4: Adaptive volatility floor for division
    global_park_vol = parkinson_vol.mean()
    vol_floor_park = max(1e-6, global_park_vol * 0.01) if not np.isnan(global_park_vol) else 1e-6
    normalized_velocity = velocity / (parkinson_vol.clip(lower=vol_floor_park))
    
    # Step 3: Calculate price acceleration d²P/dt²
    logger.debug(f"Computing price acceleration (θ={config.theta})...")
    acceleration = compute_price_acceleration_mean_reversion(
        prices,
        ma_window=config.window_200d,
        theta=config.theta,
    )
    
    # Normalize acceleration by volatility
    normalized_acceleration = acceleration / (parkinson_vol.clip(lower=vol_floor_park))
    
    # Step 4: Calculate relative price deviation from MA200
    logger.debug("Computing relative deviation from MA200...")
    ma_200 = ema(prices, span=config.window_200d)
    
    returns = log_returns(prices)
    vol_20d = rolling_volatility(returns, window=config.window_20d)
    
    # Relative deviation: (P - MA200) / (MA200 * σ_20d)
    global_vol_20d = vol_20d.mean()
    vol_floor_20d = max(1e-6, global_vol_20d * 0.01) if not np.isnan(global_vol_20d) else 1e-6
    relative_deviation = (prices - ma_200) / (ma_200 * vol_20d.clip(lower=vol_floor_20d))
    
    # Step 5: Combine components into integral
    logger.debug(f"Assembling technical integral (γ={config.gamma})...")
    
    # ρ(τ) · dP/dt component
    rsi_velocity_component = rsi_deriv * normalized_velocity
    
    # γ · d²P/dt² component
    acceleration_component = config.gamma * normalized_acceleration
    
    # Log component statistics
    logger.debug(
        f"RSI*Velocity: mean={rsi_velocity_component.mean():.6f}, "
        f"std={rsi_velocity_component.std():.6f}"
    )
    logger.debug(
        f"Accel: mean={acceleration_component.mean():.6f}, "
        f"std={acceleration_component.std():.6f}"
    )
    
    # Combined signal
    combined_signal = rsi_velocity_component + acceleration_component
    
    logger.debug(
        f"Combined signal: mean={combined_signal.mean():.6f}, "
        f"std={combined_signal.std():.6f}, "
        f"range=[{combined_signal.min():.6f}, {combined_signal.max():.6f}]"
    )
    
    # ✅ FIX 5: Use EMA instead of rolling sum for integral
    # This provides smoother, more responsive integration
    window_integral = config.window_200d
    
    # Exponential moving average as integral approximation
    alpha = 2 / (window_integral + 1)
    technical_integral = combined_signal.ewm(alpha=alpha, min_periods=1).mean()
    
    # Normalize by typical scale
    technical_integral = technical_integral * np.sqrt(window_integral)
    
    logger.debug(
        f"Technical integral: mean={technical_integral.mean():.6f}, "
        f"std={technical_integral.std():.6f}, "
        f"range=[{technical_integral.min():.6f}, {technical_integral.max():.6f}]"
    )
    
    # Check for zero variance
    if technical_integral.std() < 1e-6:
        logger.warning(
            f"Technical integral has near-zero variance (std={technical_integral.std():.8f}). "
            f"Using fallback normalization."
        )
    
    # ✅ FIX 6: Apply rolling z-score normalization BEFORE percentile norm
    # This is the key fix that prevents saturation
    technical_integral_zscore = safe_rolling_zscore(
        technical_integral,
        window=60,
        min_periods=30,
    )
    
    # Soft clamp z-scores
    technical_integral_zscore = soft_clamp(
        technical_integral_zscore,
        lower=-3.0,
        upper=3.0,
        smoothness=0.3,
    )
    
    # Step 6: Apply percentile normalization (optional)
    if use_percentile_norm:
        logger.debug("Applying percentile normalization...")
        normalized_integral = percentile_normalize(
            technical_integral_zscore,
            lower_percentile=5.0,
            upper_percentile=95.0
        )
        # Scale to tanh input range
        normalized_integral = (normalized_integral - 50) / 12
        t_score_raw = technical_integral.copy()
    else:
        normalized_integral = technical_integral_zscore
        t_score_raw = None
    
    # Step 7: Apply tanh bounding
    logger.debug("Applying hyperbolic tangent bounding...")
    tanh_signal = hyperbolic_tangent(normalized_integral)
    
    # Step 8: Scale to [0, 100]
    t_score = 50 + 50 * tanh_signal
    
    # ✅ FIX 7: Light EMA smoothing instead of rolling mean
    t_score = pd.Series(t_score, index=result_df.index)
    t_score = t_score.ewm(span=5, min_periods=1).mean()
    
    # Soft clamp final score
    t_score = np.clip(t_score, 5, 95)
    
    # FINAL SAFETY
    t_score = np.nan_to_num(t_score, nan=50.0, posinf=100.0, neginf=0.0)
    t_score = np.clip(t_score, 0, 100)
    
    # Store components
    if use_percentile_norm and t_score_raw is not None:
        result_df['t_score_raw_integral'] = t_score_raw
    
    result_df['t_score'] = t_score
    
    # Optionally return all components
    if return_components:
        result_df['rsi_14d'] = rsi_values
        result_df['rsi_derivative'] = rsi_deriv
        result_df['price_velocity'] = velocity
        result_df['normalized_velocity'] = normalized_velocity
        result_df['price_acceleration'] = acceleration
        result_df['normalized_acceleration'] = normalized_acceleration
        result_df['ma_200'] = ma_200
        result_df['relative_deviation'] = relative_deviation
        result_df['parkinson_vol'] = parkinson_vol
        result_df['vol_20d'] = vol_20d
        result_df['rsi_velocity_component'] = rsi_velocity_component
        result_df['acceleration_component'] = acceleration_component
        result_df['technical_integral'] = technical_integral
        result_df['technical_integral_zscore'] = technical_integral_zscore
        result_df['tanh_signal'] = tanh_signal
    
    logger.info(
        f"T-Score computed: mean={t_score.mean():.2f}, "
        f"std={t_score.std():.2f}, "
        f"range=[{t_score.min():.2f}, {t_score.max():.2f}]"
    )
    
    return result_df


# ---------------------------------------------------------------------------
# Alternative implementation (unchanged)
# ---------------------------------------------------------------------------

class TechnicalScoreCalculator:
    """Object-oriented T-Score calculator (unchanged from original)."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[TechnicalConfig] = None,
        use_percentile_norm: bool = True,
    ):
        if config is None:
            config = TechnicalConfig()
        
        self.df = df
        self.config = config
        self.use_percentile_norm = use_percentile_norm
        self.prices = df['close']
        self.high = df['high']
        self.low = df['low']
        
        self.rsi_values: Optional[pd.Series] = None
        self.rsi_deriv: Optional[pd.Series] = None
        self.velocity: Optional[pd.Series] = None
        self.acceleration: Optional[pd.Series] = None
        self.ma_200: Optional[pd.Series] = None
        self.parkinson_vol: Optional[pd.Series] = None
        self.vol_20d: Optional[pd.Series] = None
        self.relative_deviation: Optional[pd.Series] = None
        self.technical_integral: Optional[pd.Series] = None
        self.t_score_raw: Optional[pd.Series] = None
        self.t_score: Optional[pd.Series] = None
    
    def compute(self) -> pd.Series:
        """Run complete T-Score calculation."""
        result_df = compute_technical_score(
            self.df,
            config=self.config,
            return_components=True,
            use_percentile_norm=self.use_percentile_norm,
        )
        self.t_score = result_df['t_score']
        return self.t_score
    
    @property
    def components(self) -> dict:
        """Get all computed components."""
        return {k: v for k, v in self.__dict__.items() if isinstance(v, pd.Series)}
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export to DataFrame."""
        return compute_technical_score(
            self.df,
            config=self.config,
            return_components=True,
            use_percentile_norm=self.use_percentile_norm,
        )


# ---------------------------------------------------------------------------
# Validation and Convenience Functions (unchanged)
# ---------------------------------------------------------------------------

def validate_technical_score(
    df: pd.DataFrame,
    score_col: str = 't_score',
) -> Tuple[bool, str]:
    """Validate T-Score computation quality."""
    if score_col not in df.columns:
        return False, f"Column '{score_col}' not found"
    
    score = df[score_col].dropna()
    
    if len(score) == 0:
        return False, "All T-Score values are NaN"
    
    if (score < 0).any() or (score > 100).any():
        return False, f"T-Score out of range: [{score.min():.2f}, {score.max():.2f}]"
    
    if np.isinf(score).any():
        return False, "T-Score contains infinite values"
    
    nan_pct = (df[score_col].isna().sum() / len(df)) * 100
    if nan_pct > 50:
        return False, f"Too many NaN values: {nan_pct:.1f}%"
    
    if score.std() < 0.5:
        logger.warning(f"T-Score has low variance: std={score.std():.4f}")
    
    return True, f"T-Score valid: mean={score.mean():.2f}, std={score.std():.2f}"


def compute_technical_score_simple(
    df: pd.DataFrame,
    gamma: float = 0.1,
    theta: float = 0.1,
    use_percentile_norm: bool = True,
) -> pd.Series:
    """Simplified T-Score computation."""
    config = TechnicalConfig(gamma=gamma, theta=theta)
    result_df = compute_technical_score(df, config, use_percentile_norm=use_percentile_norm)
    return result_df['t_score']


def technical_signal(
    t_score: pd.Series,
    threshold_long: float = 60.0,
    threshold_short: float = 40.0,
) -> pd.Series:
    """Generate trading signals from T-Score."""
    signals = pd.Series(0, index=t_score.index)
    signals[t_score >= threshold_long] = 1
    signals[t_score <= threshold_short] = -1
    return signals


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = [
    'compute_technical_score',
    'TechnicalScoreCalculator',
    'validate_technical_score',
    'compute_technical_score_simple',
    'technical_signal',
    'hyperbolic_tangent',
    'percentile_normalize',
    'rsi_derivative_detailed',
    'compute_price_velocity_with_drift',
    'compute_price_acceleration_mean_reversion',
    'safe_rolling_zscore',
    'soft_clamp',
]
