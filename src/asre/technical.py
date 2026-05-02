"""
ASRE Technical Score (T-Score) Implementation (ENHANCED)

Implements the complete T-Score formula with saturation fixes:

T(t) = 50 + 50 * tanh(∫₀ᵗ [ρ(τ) · dP(τ)/dt + γ · d²P(τ)/dt²] dτ)

✅ ENHANCEMENTS (while preserving original logic):
- Rolling z-score normalization for integral (prevents saturation)
- Adaptive volatility floor (maintains sensitivity)
- Smooth EMA instead of abrupt rolling window
- All original components preserved

Version: B2-Rc (rolling percentile norm + divisor fix)

Changes from previous version:
  B2-Rc-1: ROLLING_NORM_WINDOW = 120 constant added.
  B2-Rc-2: rolling_percentile_normalize() added — replaces global
            percentile_normalize() call in compute_technical_score().
            Uses a 120-day sliding window so the normalization anchor
            moves with the stock; prevents prolonged trend periods from
            consuming the global 95th percentile and pinning T-score
            at exactly 95 for 30-40 consecutive rows.
  B2-Rc-3: Scalar changed /12 → /20 in compute_technical_score().
            Maps percentile-normalized [0,100] → [-2.5, 2.5] tanh input
            instead of [-4.17, 4.17]; tanh stays in near-linear range for
            typical values, saturates only for true outlier days.
  All other code unchanged — percentile_normalize() retained for
  use in Stage 2 fallback of percentile_normalize() itself.
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
# B2-Rc-1: Rolling normalisation window constant
# ---------------------------------------------------------------------------
# 120 trading days = ~6 months of context for percentile anchor.
# Shorter windows (e.g. 60) are too reactive — a single strong month
# inflates the local ceiling. Longer (e.g. 252) introduces too much lag
# when a stock regime genuinely changes. 120 is the calibrated midpoint.
ROLLING_NORM_WINDOW = 120    # ← B2-Rc NEW


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
    Rolling z-score normalization with adaptive floor. UNCHANGED.

    Prevents saturation by using rolling statistics instead of global.
    """
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std  = series.rolling(window=window, min_periods=min_periods).std()

    global_std = series.std()
    vol_floor  = max(0.001, global_std * 0.01) if not np.isnan(global_std) else 0.001

    rolling_std_safe = rolling_std.clip(lower=vol_floor)
    z_score = (series - rolling_mean) / rolling_std_safe

    if z_score.isna().any():
        global_mean      = series.mean()
        global_std_safe  = max(series.std(), vol_floor)
        global_z         = (series - global_mean) / global_std_safe
        z_score          = z_score.fillna(global_z)

    return z_score


def soft_clamp(
    z: pd.Series,
    lower: float = -3.0,
    upper: float = 3.0,
    smoothness: float = 0.3,
) -> pd.Series:
    """
    Soft clamping using smooth sigmoid transitions. UNCHANGED.

    Replaces hard clip() with smooth boundaries.
    """
    z_lower   = lower + smoothness * np.log(1 + np.exp((z - lower) / smoothness))
    z_clamped = upper - smoothness * np.log(1 + np.exp((upper - z_lower) / smoothness))
    return z_clamped


def percentile_normalize(
    values: pd.Series,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
) -> pd.Series:
    """
    Multi-stage global normalization. UNCHANGED — retained as fallback
    inside rolling_percentile_normalize() Stage 3 and for external use.
    No longer called directly in compute_technical_score() — see B2-Rc-2.
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
        logger.debug(
            "✓ Percentile norm: [%.3f, %.3f] → std=%.2f",
            p_lower, p_upper, normalized.std(),
        )
        return normalized

    # STAGE 2: Rolling z-score fallback
    logger.debug("Percentile failed, using rolling z-score...")
    z_scores         = safe_rolling_zscore(values, window=60, min_periods=20)
    z_scores_clamped = soft_clamp(z_scores, lower=-3, upper=3)
    normalized       = 50 + 25 * np.tanh(z_scores_clamped / 2)

    if normalized.std() > 1.0:
        logger.debug("✓ Rolling z-score: std=%.2f", normalized.std())
        return np.clip(normalized, 0, 100)

    # STAGE 3: Global z-score fallback
    mean_val = valid_values.mean()
    std_val  = valid_values.std()

    if std_val > 1e-8:
        zscore     = (values - mean_val) / std_val
        normalized = 50 + 25 * np.tanh(zscore / 2)
        logger.debug(
            "✓ Global z-score fallback: std=%.6f → std=%.2f",
            std_val, normalized.std(),
        )
        return np.clip(normalized, 20, 80)

    # STAGE 4: Neutral band
    logger.warning("⚠️ Flat signal detected → returning neutral band")
    epsilon = np.random.normal(0, 0.5, len(values))
    result  = 50 + epsilon
    return pd.Series(np.clip(result, 45, 55), index=values.index)


# ---------------------------------------------------------------------------
# B2-Rc-2: Rolling percentile normalisation (NEW)
# ---------------------------------------------------------------------------

def rolling_percentile_normalize(
    values: pd.Series,
    window: int = ROLLING_NORM_WINDOW,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
) -> pd.Series:
    """
    B2-Rc-2: Rolling percentile normalization.

    Replaces global percentile_normalize() in compute_technical_score()
    to prevent prolonged trend periods from anchoring the global
    percentile ceiling and causing T-score saturation at 95.

    For each row t, normalization uses only the trailing `window` rows
    [t-window, t). This keeps the 5th/95th percentile anchor moving
    with the stock's recent regime — a 39-day uptrend cannot represent
    the global 95th percentile if the window is 120 days.

    Fallback: if fewer than window//2 rows are available in the trailing
    window (warm-up period), falls back to global percentile_normalize()
    for that row, then transitions to rolling once min_periods is met.

    Args:
        values:            Input series (typically soft-clamped z-scores)
        window:            Rolling context window (default 120 = 6 months)
        lower_percentile:  Lower bound percentile (default 5.0)
        upper_percentile:  Upper bound percentile (default 95.0)

    Returns:
        pd.Series normalised to [0, 100] per rolling window context.
    """
    min_periods = window // 2    # 60 rows minimum for rolling to kick in

    def _row_norm(window_vals: np.ndarray) -> float:
        """Normalise the last value in window_vals against the window distribution."""
        current = window_vals[-1]
        if np.isnan(current):
            return 50.0

        clean = window_vals[~np.isnan(window_vals)]
        if len(clean) < 10:
            return 50.0

        p_lo = np.nanpercentile(clean, lower_percentile)
        p_hi = np.nanpercentile(clean, upper_percentile)

        if p_hi - p_lo > 1e-6:
            return float(np.clip((current - p_lo) / (p_hi - p_lo) * 100, 0, 100))

        # Flat window — return position as global z-score mapped to [20, 80]
        mean_w = clean.mean()
        std_w  = clean.std()
        if std_w > 1e-8:
            z = (current - mean_w) / std_w
            return float(np.clip(50 + 25 * np.tanh(z / 2), 20, 80))

        return 50.0

    # Rolling apply: each row sees trailing `window` values including itself
    rolling_norm = values.rolling(
        window=window,
        min_periods=min_periods,
    ).apply(_row_norm, raw=True)

    # Warm-up fallback: rows where rolling hasn't reached min_periods yet
    warm_up_mask = rolling_norm.isna()
    if warm_up_mask.any():
        global_norm     = percentile_normalize(values, lower_percentile, upper_percentile)
        rolling_norm    = rolling_norm.copy()
        rolling_norm[warm_up_mask] = global_norm[warm_up_mask]

    logger.debug(
        "B2-Rc rolling_percentile_normalize: window=%d, range=[%.1f, %.1f], std=%.2f",
        window, rolling_norm.min(), rolling_norm.max(), rolling_norm.std(),
    )
    return rolling_norm


# ---------------------------------------------------------------------------
# Helper functions (UNCHANGED)
# ---------------------------------------------------------------------------

def rsi_derivative_detailed(
    prices: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    RSI derivative with detailed calculation. UNCHANGED.

    Formula: ρ(τ) = ∂RSI(τ)/∂t
    """
    rsi_values = rsi(prices, period)
    rsi_deriv  = rsi_values.diff()
    rsi_deriv  = rsi_deriv.rolling(window=3, min_periods=1).mean()
    return rsi_deriv


def compute_price_velocity_with_drift(
    prices: pd.Series,
    drift_window: int = 20,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate price velocity: dP/dt = μ_p + σ_p · dW_t^P. UNCHANGED.
    """
    velocity   = prices.diff()
    mean_drift = velocity.rolling(window=drift_window).mean()
    vol        = velocity.rolling(window=drift_window).std()
    return velocity, mean_drift, vol


def compute_price_acceleration_mean_reversion(
    prices: pd.Series,
    ma_window: int = 200,
    theta: float = 0.1,
) -> pd.Series:
    """
    Calculate price acceleration with mean reversion. UNCHANGED.

    Formula: d²P/dt² = θ(MA_200 - P(t))
    """
    ma_200       = ema(prices, span=ma_window)
    acceleration = theta * (ma_200 - prices)
    return acceleration


# ---------------------------------------------------------------------------
# T-Score Core Implementation — B2-Rc-2 and B2-Rc-3 applied
# ---------------------------------------------------------------------------

def compute_technical_score(
    df: pd.DataFrame,
    config: Optional[TechnicalConfig] = None,
    return_components: bool = False,
    use_percentile_norm: bool = True,
) -> pd.DataFrame:
    """
    Compute Technical Score (T-Score) for a stock.

    ✅ ENHANCEMENTS (from previous version, retained):
    - Rolling z-score normalization (prevents flatline)
    - Adaptive volatility floor (maintains sensitivity)
    - Soft clamping (preserves regime changes)
    - EMA smoothing (reduces noise)

    ✅ B2-Rc changes (this version):
    - B2-Rc-2: percentile_normalize() → rolling_percentile_normalize()
               Prevents global percentile ceiling from binding for
               prolonged trend periods (fixes SBIN T=95 for 39 rows).
    - B2-Rc-3: scalar /12 → /20
               Maps [0,100] → [-2.5, 2.5] tanh input; keeps tanh in
               near-linear range for typical values.

    Formula: T(t) = 50 + 50 * tanh(∫₀ᵗ [ρ(τ)·dP/dt + γ·d²P/dt²] dτ)
    """
    if config is None:
        config = TechnicalConfig()

    required_cols = ['close', 'high', 'low']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame must contain columns: {missing}")

    if len(df) < config.window_200d:
        logger.warning(
            "DataFrame has %d rows, but T-Score needs at least %d rows "
            "for reliable MA200 calculation", len(df), config.window_200d,
        )

    result_df = df.copy()
    prices    = df['close']

    # Step 1: RSI derivative ρ(τ)
    logger.debug("Computing RSI derivative (period=%d)...", config.window_rsi)
    rsi_values = rsi(prices, period=config.window_rsi)
    rsi_deriv  = rsi_derivative_detailed(prices, period=config.window_rsi)

    # Step 2: Price velocity dP/dt
    logger.debug("Computing price velocity...")
    velocity, drift, vol_velocity = compute_price_velocity_with_drift(
        prices, drift_window=config.window_20d,
    )

    parkinson_vol   = parkinson_volatility(df['high'], df['low'], window=config.window_20d)
    global_park_vol = parkinson_vol.mean()
    vol_floor_park  = max(1e-6, global_park_vol * 0.01) if not np.isnan(global_park_vol) else 1e-6

    normalized_velocity = velocity / (parkinson_vol.clip(lower=vol_floor_park))

    # Step 3: Price acceleration d²P/dt²
    logger.debug("Computing price acceleration (θ=%.2f)...", config.theta)
    acceleration           = compute_price_acceleration_mean_reversion(
        prices, ma_window=config.window_200d, theta=config.theta,
    )
    normalized_acceleration = acceleration / (parkinson_vol.clip(lower=vol_floor_park))

    # Step 4: Relative deviation from MA200
    logger.debug("Computing relative deviation from MA200...")
    ma_200   = ema(prices, span=config.window_200d)
    returns  = log_returns(prices)
    vol_20d  = rolling_volatility(returns, window=config.window_20d)

    global_vol_20d  = vol_20d.mean()
    vol_floor_20d   = max(1e-6, global_vol_20d * 0.01) if not np.isnan(global_vol_20d) else 1e-6
    relative_deviation = (prices - ma_200) / (ma_200 * vol_20d.clip(lower=vol_floor_20d))

    # Step 5: Combine into integral
    logger.debug("Assembling technical integral (γ=%.2f)...", config.gamma)
    rsi_velocity_component  = rsi_deriv * normalized_velocity
    acceleration_component  = config.gamma * normalized_acceleration

    logger.debug(
        "RSI*Velocity: mean=%.6f, std=%.6f",
        rsi_velocity_component.mean(), rsi_velocity_component.std(),
    )
    logger.debug(
        "Accel: mean=%.6f, std=%.6f",
        acceleration_component.mean(), acceleration_component.std(),
    )

    combined_signal = rsi_velocity_component + acceleration_component

    logger.debug(
        "Combined signal: mean=%.6f, std=%.6f, range=[%.6f, %.6f]",
        combined_signal.mean(), combined_signal.std(),
        combined_signal.min(), combined_signal.max(),
    )

    # Step 5b: EMA integral
    window_integral    = config.window_200d
    alpha              = 2 / (window_integral + 1)
    technical_integral = combined_signal.ewm(alpha=alpha, min_periods=1).mean()
    technical_integral = technical_integral * np.sqrt(window_integral)

    logger.debug(
        "Technical integral: mean=%.6f, std=%.6f, range=[%.6f, %.6f]",
        technical_integral.mean(), technical_integral.std(),
        technical_integral.min(), technical_integral.max(),
    )

    if technical_integral.std() < 1e-6:
        logger.warning(
            "Technical integral has near-zero variance (std=%.8f). "
            "Using fallback normalization.", technical_integral.std(),
        )

    # Step 6: Rolling z-score + soft clamp (UNCHANGED)
    technical_integral_zscore = safe_rolling_zscore(
        technical_integral, window=60, min_periods=30,
    )
    technical_integral_zscore = soft_clamp(
        technical_integral_zscore, lower=-3.0, upper=3.0, smoothness=0.3,
    )

    # Step 7: Percentile normalization
    if use_percentile_norm:
        logger.debug(
            "Applying rolling percentile normalization (window=%d)...",
            ROLLING_NORM_WINDOW,
        )

        # ── B2-Rc-2 PATCH ────────────────────────────────────────────────────
        # WAS: normalized_integral = percentile_normalize(
        #          technical_integral_zscore,
        #          lower_percentile=5.0, upper_percentile=95.0)
        #
        # Problem: global percentile binds SBIN's 39-day uptrend to the
        # global 95th percentile ceiling → T-score = 95 for 39 rows.
        #
        # Fix: rolling_percentile_normalize with window=120 (6 months).
        # The normalization anchor moves with the stock's recent regime.
        normalized_integral = rolling_percentile_normalize(
            technical_integral_zscore,
            window=ROLLING_NORM_WINDOW,        # ← B2-Rc-2 CHANGED (was global call)
            lower_percentile=5.0,
            upper_percentile=95.0,
        )
        # ── END B2-Rc-2 PATCH ────────────────────────────────────────────────

        # ── B2-Rc-3 PATCH ────────────────────────────────────────────────────
        # WAS: normalized_integral = (normalized_integral - 50) / 12
        # Problem: /12 maps [0,100] → [-4.17, 4.17]; tanh(4.17)=0.9996
        #          → virtually every strong period maps to 50±50 = 0 or 100
        #          → clips to 5 or 95.
        # Fix: /20 maps [0,100] → [-2.5, 2.5]; tanh(2.5)=0.987
        #      → max score before clip = 49.35+50 = 99.35 → clips to 95
        #        only for true outliers, not prolonged moderate trends.
        normalized_integral = (normalized_integral - 50) / 20   # ← B2-Rc-3 CHANGED (was /12)
        # ── END B2-Rc-3 PATCH ────────────────────────────────────────────────

        t_score_raw = technical_integral.copy()
    else:
        normalized_integral = technical_integral_zscore
        t_score_raw         = None

    # Step 8: tanh bounding (UNCHANGED)
    logger.debug("Applying hyperbolic tangent bounding...")
    tanh_signal = hyperbolic_tangent(normalized_integral)

    # Step 9: Scale to [0, 100] (UNCHANGED)
    t_score = 50 + 50 * tanh_signal

    # Step 10: EMA smoothing (UNCHANGED)
    t_score = pd.Series(t_score, index=result_df.index)
    t_score = t_score.ewm(span=5, min_periods=1).mean()

    # Final clips (UNCHANGED — clip values stay at 5 and 95)
    t_score = np.clip(t_score, 5, 95)
    t_score = np.nan_to_num(t_score, nan=50.0, posinf=100.0, neginf=0.0)
    t_score = np.clip(t_score, 0, 100)

    # Store optional components
    if use_percentile_norm and t_score_raw is not None:
        result_df['t_score_raw_integral'] = t_score_raw

    result_df['t_score'] = t_score

    if return_components:
        result_df['rsi_14d']                    = rsi_values
        result_df['rsi_derivative']             = rsi_deriv
        result_df['price_velocity']             = velocity
        result_df['normalized_velocity']        = normalized_velocity
        result_df['price_acceleration']         = acceleration
        result_df['normalized_acceleration']    = normalized_acceleration
        result_df['ma_200']                     = ma_200
        result_df['relative_deviation']         = relative_deviation
        result_df['parkinson_vol']              = parkinson_vol
        result_df['vol_20d']                    = vol_20d
        result_df['rsi_velocity_component']     = rsi_velocity_component
        result_df['acceleration_component']     = acceleration_component
        result_df['technical_integral']         = technical_integral
        result_df['technical_integral_zscore']  = technical_integral_zscore
        result_df['tanh_signal']                = tanh_signal

    logger.info(
        "T-Score computed: mean=%.2f, std=%.2f, range=[%.2f, %.2f]",
        t_score.mean(), t_score.std(), t_score.min(), t_score.max(),
    )

    return result_df


# ---------------------------------------------------------------------------
# Alternative implementation (UNCHANGED)
# ---------------------------------------------------------------------------

class TechnicalScoreCalculator:
    """Object-oriented T-Score calculator. UNCHANGED."""

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[TechnicalConfig] = None,
        use_percentile_norm: bool = True,
    ):
        if config is None:
            config = TechnicalConfig()

        self.df                  = df
        self.config              = config
        self.use_percentile_norm = use_percentile_norm
        self.prices              = df['close']
        self.high                = df['high']
        self.low                 = df['low']

        self.rsi_values:           Optional[pd.Series] = None
        self.rsi_deriv:            Optional[pd.Series] = None
        self.velocity:             Optional[pd.Series] = None
        self.acceleration:         Optional[pd.Series] = None
        self.ma_200:               Optional[pd.Series] = None
        self.parkinson_vol:        Optional[pd.Series] = None
        self.vol_20d:              Optional[pd.Series] = None
        self.relative_deviation:   Optional[pd.Series] = None
        self.technical_integral:   Optional[pd.Series] = None
        self.t_score_raw:          Optional[pd.Series] = None
        self.t_score:              Optional[pd.Series] = None

    def compute(self) -> pd.Series:
        """Run complete T-Score calculation."""
        result_df   = compute_technical_score(
            self.df,
            config=self.config,
            return_components=True,
            use_percentile_norm=self.use_percentile_norm,
        )
        self.t_score = result_df['t_score']
        return self.t_score

    @property
    def components(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if isinstance(v, pd.Series)}

    def to_dataframe(self) -> pd.DataFrame:
        return compute_technical_score(
            self.df,
            config=self.config,
            return_components=True,
            use_percentile_norm=self.use_percentile_norm,
        )


# ---------------------------------------------------------------------------
# Validation and Convenience Functions (UNCHANGED)
# ---------------------------------------------------------------------------

def validate_technical_score(
    df: pd.DataFrame,
    score_col: str = 't_score',
) -> Tuple[bool, str]:
    """Validate T-Score computation quality. UNCHANGED."""
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
        logger.warning("T-Score has low variance: std=%.4f", score.std())

    return True, f"T-Score valid: mean={score.mean():.2f}, std={score.std():.2f}"


def compute_technical_score_simple(
    df: pd.DataFrame,
    gamma: float = 0.1,
    theta: float = 0.1,
    use_percentile_norm: bool = True,
) -> pd.Series:
    """Simplified T-Score computation. UNCHANGED."""
    config    = TechnicalConfig(gamma=gamma, theta=theta)
    result_df = compute_technical_score(df, config, use_percentile_norm=use_percentile_norm)
    return result_df['t_score']


def technical_signal(
    t_score: pd.Series,
    threshold_long: float = 60.0,
    threshold_short: float = 40.0,
) -> pd.Series:
    """Generate trading signals from T-Score. UNCHANGED."""
    signals = pd.Series(0, index=t_score.index)
    signals[t_score >= threshold_long]  = 1
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
    'rolling_percentile_normalize',    # ← B2-Rc NEW
    'rsi_derivative_detailed',
    'compute_price_velocity_with_drift',
    'compute_price_acceleration_mean_reversion',
    'safe_rolling_zscore',
    'soft_clamp',
    'ROLLING_NORM_WINDOW',             # ← B2-Rc NEW
]
