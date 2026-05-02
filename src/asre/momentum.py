"""
ASRE Momentum Score (M-Score) Implementation (ENHANCED v2.0)

✅ ORIGINAL ENHANCEMENTS:
- Rolling z-score normalization (prevents saturation)
- Adaptive volatility floor (prevents flat distributions)
- Soft clamping (preserves regime sensitivity)
- Numerical safety throughout

✅ NEW ENHANCEMENTS (v2.0):
- Trend continuation detection (captures strong moves like AAPL +72%)
- Adaptive thresholds (better timing, adjusts to market conditions)
- Trade filters (reduces friction from 8-11 trades to 3-5)
- Minimum holding period enforcement
- Signal confirmation requirements
- Backward compatible (existing code works unchanged)

Version: B2-Rc (Sharpe additive-delta + freeze detection + norm constant)

Changes from previous version:
  B2-Rc-M1: MOMENTUM_NORM_WINDOW = 60 constant added.
             window=60 was hardcoded in 4 places; unified to this constant.
  B2-Rc-M2: Sharpe adjustment changed from multiplicative to additive delta.
             WAS: m_score_adj = m_score * sharpe_factor  (factor ∈ [0.5, 2.0])
             Problem: For stable large-cap NSE stocks (ICICIBANK, BPCL,
             MARUTI) where vol_60d ≈ 0.004, sharpe_factor clips to 2.0.
             Any m_score > 50 becomes > 100 → frozen at 100.
             FIX: sharpe_delta = (sharpe_factor - 1.0) * 15.0 → [-7.5, +15]
             additive points; m_score_adj = clip(m_score + sharpe_delta, 0, 100).
             A score of 92 + 15 = 107 → clips to 100 only at extreme outlier
             (sharpe_factor = 2.0 AND m_score ≥ 85). Typical case:
             m_score=75, sharpe_factor=1.8 → 75 + 12 = 87. No saturation.
  B2-Rc-M3: M-score freeze detection added.
             Parallel to T-score SCORE FREEZE warning (which fires for SBIN
             t_score=95 for 39 rows). Fires when rolling std(m_score_adj, 10)
             < 0.1 — distinguishes genuine saturation from data freezes.
  B2-Rc-M4: 4× inline window=60 replaced with MOMENTUM_NORM_WINDOW constant.
  All other code unchanged.

Usage:
    # Original usage (unchanged):
    result = compute_momentum_score(df)
    signals = momentum_signal(result['m_score_adj'])

    # Enhanced usage (opt-in):
    config = MomentumConfig(
        use_enhancements=True,
        min_holding_days=12,
        confirmation_days=3
    )
    result = compute_momentum_score(df, config, return_components=True)
    signals = momentum_signal(
        result['m_score_adj'],
        use_enhancements=True,
        trend_strength=result.get('trend_strength'),
        prices=df['close']
    )
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
# B2-Rc-M1: Rolling normalisation window constant
# ---------------------------------------------------------------------------
# 60 trading days = 3 months — the window for momentum z-score and Sharpe
# volatility estimation. Unifying the 4 inline window=60 literals here
# means a single change adjusts all normalisation simultaneously.
MOMENTUM_NORM_WINDOW = 60    # ← B2-Rc-M1 NEW


# ===========================================================================
# ENHANCEMENT LAYER: Trend Detection & Filters (UNCHANGED from v2.0)
# ===========================================================================

def _calculate_trend_strength(
    prices: pd.Series,
    short_window: int = 60,
    long_window: int = 50,
) -> pd.Series:
    """
    Calculate trend strength score (0-100). UNCHANGED.

    High score = strong, sustainable trend
    Low score = weak or overextended trend
    """
    ma_short = prices.rolling(short_window).mean()
    ma_long  = prices.rolling(long_window).mean()

    price_above_short = (prices > ma_short).astype(int)
    price_above_long  = (prices > ma_long).astype(int)

    returns_short  = prices.pct_change(short_window)
    distance_short = (prices - ma_short) / ma_short

    trend_score = (
        price_above_short * 25
        + price_above_long * 25
        + np.clip(returns_short * 100, -25, 25)
        + np.clip((1 - abs(distance_short) * 5) * 25, 0, 25)
    )

    return np.clip(trend_score, 0, 100)


def _calculate_trend_maturity(
    prices: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Calculate trend maturity (0-1). UNCHANGED.

    0 = young trend (likely to continue)
    1 = mature trend (may reverse)
    """
    returns           = prices.pct_change()
    cumulative_return = prices.pct_change(window)

    volatility = returns.rolling(window).std()
    vol_ratio  = volatility / volatility.rolling(window * 2).mean()
    vol_score  = np.clip(vol_ratio, 0, 2) / 2

    abs_return   = abs(cumulative_return)
    return_score = np.clip(abs_return / 0.5, 0, 1)

    x                  = np.arange(window)
    linearity_scores   = []

    for i in range(len(prices)):
        if i < window:
            linearity_scores.append(0.5)
        else:
            y = prices.iloc[i - window + 1 : i + 1].values
            if len(y) == window and not np.any(np.isnan(y)):
                correlation = np.corrcoef(x, y)[0, 1]
                linearity_scores.append(abs(correlation))
            else:
                linearity_scores.append(0.5)

    linearity = pd.Series(linearity_scores, index=prices.index)

    maturity = vol_score * 0.3 + return_score * 0.4 + linearity * 0.3
    return maturity.fillna(0.5)


def _calculate_adaptive_thresholds(
    m_score: pd.Series,
    trend_strength: pd.Series,
    volatility: pd.Series,
    base_long: float = 70,
    base_short: float = 30,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate adaptive entry/exit thresholds. UNCHANGED.

    Strong trend → lower threshold (easier to stay in)
    High volatility → wider bands (reduce whipsaws)
    """
    trend_adjustment = (trend_strength - 50) / 50 * 10

    vol_mean       = volatility.rolling(60).mean()
    vol_std        = volatility.rolling(60).std()
    vol_normalized = (volatility - vol_mean) / (vol_std + 1e-6)
    vol_adjustment = np.clip(vol_normalized * 5, -5, 5)

    long_threshold  = base_long  - trend_adjustment + vol_adjustment
    short_threshold = base_short + trend_adjustment - vol_adjustment

    long_threshold  = np.clip(long_threshold,  65, 85)
    short_threshold = np.clip(short_threshold, 15, 35)

    return long_threshold, short_threshold


def _apply_confirmation_filter(
    signals: pd.Series,
    m_score: pd.Series,
    confirmation_days: int = 3,
) -> pd.Series:
    """
    Require signal to persist for N days before acting. UNCHANGED.
    """
    if confirmation_days <= 0:
        return signals

    confirmed = pd.Series(0, index=signals.index)

    for i in range(min(confirmation_days, len(signals))):
        confirmed.iloc[i] = signals.iloc[i]

    for i in range(confirmation_days, len(signals)):
        recent = signals.iloc[i - confirmation_days : i + 1]
        if all(recent == 1):
            confirmed.iloc[i] = 1
        elif all(recent == -1):
            confirmed.iloc[i] = -1
        elif all(recent == 0):
            confirmed.iloc[i] = 0
        else:
            confirmed.iloc[i] = confirmed.iloc[i - 1]

    return confirmed


def _apply_holding_period_filter(
    signals: pd.Series,
    min_holding_days: int = 10,
) -> pd.Series:
    """
    Enforce minimum holding period once in position. UNCHANGED.
    """
    if min_holding_days <= 0:
        return signals

    filtered           = signals.copy()
    current_position   = 0
    days_in_position   = 0

    for i in range(len(signals)):
        proposed = signals.iloc[i]

        if current_position == 0:
            filtered.iloc[i] = proposed
            if proposed != 0:
                current_position   = proposed
                days_in_position   = 1
        else:
            days_in_position += 1
            if days_in_position < min_holding_days:
                filtered.iloc[i] = current_position
            else:
                if proposed != current_position:
                    filtered.iloc[i]   = proposed
                    current_position   = proposed
                    days_in_position   = 1 if proposed != 0 else 0
                else:
                    filtered.iloc[i] = current_position

    return filtered


def _apply_trend_continuation_filter(
    signals: pd.Series,
    trend_strength: pd.Series,
    trend_maturity: pd.Series,
    m_score: pd.Series,
    continuation_threshold: float = 70,
) -> pd.Series:
    """
    Override exit signals when strong trend is continuing. UNCHANGED.
    """
    filtered = signals.copy()

    for i in range(1, len(signals)):
        prev_signal    = filtered.iloc[i - 1]
        current_signal = signals.iloc[i]

        if prev_signal == 1 and current_signal != 1:
            strong_trend = trend_strength.iloc[i] > continuation_threshold
            immature     = trend_maturity.iloc[i] < 0.7
            high_score   = m_score.iloc[i] > 50
            if strong_trend and immature and high_score:
                filtered.iloc[i] = 1

        elif prev_signal == -1 and current_signal != -1:
            strong_downtrend = trend_strength.iloc[i] < (100 - continuation_threshold)
            immature         = trend_maturity.iloc[i] < 0.7
            low_score        = m_score.iloc[i] < 50
            if strong_downtrend and immature and low_score:
                filtered.iloc[i] = -1

    return filtered


# ===========================================================================
# Numerical Safety Utilities (UNCHANGED)
# ===========================================================================

def safe_rolling_zscore_momentum(
    series: pd.Series,
    window: int = MOMENTUM_NORM_WINDOW,          # ← B2-Rc-M4 (was hardcoded 60)
    min_periods: int = MOMENTUM_NORM_WINDOW // 2, # ← B2-Rc-M4 (was hardcoded 30)
    vol_floor: float = 0.01,
) -> pd.Series:
    """
    Compute rolling z-score with adaptive volatility floor for momentum.
    UNCHANGED except default args now reference MOMENTUM_NORM_WINDOW.
    """
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std  = series.rolling(window=window, min_periods=min_periods).std()

    global_std     = series.std()
    adaptive_floor = max(vol_floor, global_std * 0.05)

    rolling_std_safe = rolling_std.clip(lower=adaptive_floor)
    z_score          = (series - rolling_mean) / rolling_std_safe

    if z_score.isna().any():
        global_mean      = series.mean()
        global_std_safe  = max(series.std(), adaptive_floor)
        global_z         = (series - global_mean) / global_std_safe
        z_score          = z_score.fillna(global_z)

    return z_score


def soft_clamp_momentum(
    z: pd.Series,
    lower: float = -3.0,
    upper: float = 3.0,
    smoothness: float = 0.3,
) -> pd.Series:
    """
    Soft clamping for momentum scores. UNCHANGED.
    """
    z_lower   = lower + smoothness * np.log(1 + np.exp((z - lower) / smoothness))
    z_clamped = upper - smoothness * np.log(1 + np.exp((upper - z_lower) / smoothness))
    return z_clamped


# ===========================================================================
# M-Score Core Implementation — B2-Rc-M2, M3, M4 applied
# ===========================================================================

def compute_momentum_score(
    df: pd.DataFrame,
    config: Optional[MomentumConfig] = None,
    return_components: bool = False,
) -> pd.DataFrame:
    """
    Compute Momentum Score (M-Score) with optional enhancements.

    ✅ ORIGINAL ENHANCEMENTS (retained):
    - Rolling z-score normalization of momentum ratio
    - Adaptive volatility floor for denominator
    - Soft clamping of extreme values
    - Numerical stability throughout

    ✅ NEW ENHANCEMENTS (v2.0) - Enabled via config.use_enhancements:
    - Trend strength detection (captures strong continuations)
    - Trend maturity detection (identifies overextension)

    ✅ B2-Rc changes (this version):
    - B2-Rc-M2: Sharpe multiplicative → additive delta.
    - B2-Rc-M3: M-score freeze detection.
    - B2-Rc-M4: MOMENTUM_NORM_WINDOW constant replaces inline literals.

    Formula: M(t) = 50 + 50 * tanh(ratio_clamped/2) + β_m · ρ_autocorr(60)
    """
    if config is None:
        config = MomentumConfig()

    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    if len(df) < config.window_60d:
        logger.warning(
            "DataFrame has %d rows, but M-Score needs at least %d rows "
            "for reliable calculation", len(df), config.window_60d,
        )

    result_df = df.copy()

    # Step 1: 60-day log returns
    logger.debug("Computing 60-day log returns...")
    log_returns_60d = log_returns(df['close'], periods=config.window_60d, fillna=False)

    # Step 2: Exponential decay convolution (numerator)
    logger.debug("Computing exponential decay convolution (κ=%.3f)...", config.kappa)
    numerator = exponential_decay_convolution(
        log_returns_60d,
        kappa=config.kappa,
        window=config.window_60d,
    )

    # Step 3: Volatility normalization (denominator)
    logger.debug("Computing volatility normalization with adaptive floor...")
    denominator = volatility_normalization(
        log_returns_60d,
        kappa=config.kappa,
        window=config.window_60d,
    )

    global_vol       = log_returns_60d.std()
    adaptive_floor   = max(0.001, global_vol * 0.01)
    denominator_safe = denominator.clip(lower=adaptive_floor)

    # Step 4: Raw ratio
    ratio_raw = numerator / denominator_safe

    # Step 5: Rolling z-score normalization
    logger.debug("Applying rolling z-score normalization to momentum ratio...")
    ratio_zscore = safe_rolling_zscore_momentum(
        ratio_raw,
        window=MOMENTUM_NORM_WINDOW,           # ← B2-Rc-M4
        min_periods=MOMENTUM_NORM_WINDOW // 2, # ← B2-Rc-M4
        vol_floor=0.01,
    )

    # Step 6: Soft clamping
    ratio_clamped = soft_clamp_momentum(
        ratio_zscore,
        lower=-2.5,
        upper=2.5,
        smoothness=0.3,
    )

    # Step 7: Autocorrelation
    logger.debug("Computing autocorrelation for mean reversion...")
    daily_returns = log_returns(df['close'], periods=1, fillna=False)
    autocorr_60d  = rolling_autocorrelation(
        daily_returns,
        lag=MOMENTUM_NORM_WINDOW,              # ← B2-Rc-M4
        window=config.window_60d,
    )
    autocorr_60d = autocorr_60d.fillna(0.0)

    # Step 8: Base M-Score assembly (UNCHANGED)
    logger.debug("Assembling M-Score with normalized components...")
    momentum_component = 50 * np.tanh(ratio_clamped / 2.0)
    m_score            = 50 + momentum_component + config.beta_m * autocorr_60d
    m_score            = np.clip(m_score, 0, 100)

    # Step 9: Sharpe adjustment
    # ── B2-Rc-M2 PATCH ───────────────────────────────────────────────────────
    # WAS (multiplicative — causes saturation):
    #   sharpe_factor = np.sqrt(0.15 / vol_60d_safe)          # ∈ [0.5, 2.0]
    #   sharpe_factor = np.clip(sharpe_factor, 0.5, 2.0)
    #   m_score_adj   = m_score * sharpe_factor
    #   m_score_adj   = np.clip(m_score_adj, 0, 100)
    #
    # Problem: For stable NSE large-caps (ICICIBANK, BPCL, MARUTI):
    #   vol_60d ≈ 0.004 → sharpe_factor = sqrt(0.15/0.004) = 6.1 → clips to 2.0
    #   m_score ≈ 92  → m_score_adj = 92 * 2.0 = 184 → clips to 100
    #   This keeps m_score_adj = 100 for the entire low-volatility period.
    #
    # Fix (additive delta):
    #   sharpe_delta = (sharpe_factor - 1.0) * 15.0  →  ∈ [-7.5, +15.0] pts
    #   m_score_adj  = clip(m_score + sharpe_delta, 0, 100)
    #
    # New behaviour:
    #   sharpe_factor=2.0 (min vol) →  +15 pts: m_score=92 → adj=107 → 100
    #     (only clips if m_score already > 85, not the typical ~65-75 range)
    #   sharpe_factor=1.0 (neutral) →   +0 pts: m_score unchanged
    #   sharpe_factor=0.5 (high vol) →  -7.5 pts: m_score=65 → adj=57.5
    logger.debug("Applying Sharpe adjustment (additive delta)...")
    vol_60d = rolling_volatility(daily_returns, window=MOMENTUM_NORM_WINDOW)  # ← B2-Rc-M4

    vol_floor_adaptive = max(0.01, daily_returns.std() * 0.05)
    vol_60d_safe       = vol_60d.clip(lower=vol_floor_adaptive)

    sharpe_factor = np.sqrt(0.15 / vol_60d_safe)
    sharpe_factor = np.clip(sharpe_factor, 0.5, 2.0)

    sharpe_delta  = (sharpe_factor - 1.0) * 15.0           # ← B2-Rc-M2 NEW: [-7.5, +15]
    m_score_adj   = np.clip(m_score + sharpe_delta, 0, 100) # ← B2-Rc-M2 CHANGED
    # ── END B2-Rc-M2 PATCH ───────────────────────────────────────────────────

    # ── B2-Rc-M3 PATCH: M-score freeze detection ─────────────────────────────
    # Parallel to T-score SCORE FREEZE warning that fires for SBIN t=95×39 rows.
    # Threshold: rolling std < 0.1 over 10 rows → genuine freeze or saturation.
    _FREEZE_WINDOW   = 10
    _FREEZE_STD_GATE = 0.1

    if len(m_score_adj) >= _FREEZE_WINDOW:
        recent_std = m_score_adj.rolling(_FREEZE_WINDOW).std().iloc[-1]
        if not np.isnan(recent_std) and recent_std < _FREEZE_STD_GATE:
            logger.warning(
                "SCORE FREEZE detected [M-Score]: m_score_adj std=%.4f over "
                "last %d rows (value=%.1f) — possible stale price data or "
                "Sharpe saturation. Run with --clear-cache to force rescan.",
                recent_std, _FREEZE_WINDOW, m_score_adj.iloc[-1],
            )
    # ── END B2-Rc-M3 PATCH ───────────────────────────────────────────────────

    result_df['m_score']     = m_score
    result_df['m_score_adj'] = m_score_adj

    # =========================================================================
    # Enhancement Layer (v2.0) — UNCHANGED
    # =========================================================================
    if hasattr(config, 'use_enhancements') and config.use_enhancements:
        logger.debug("Computing enhancement metrics...")

        trend_strength = _calculate_trend_strength(df['close'])
        trend_maturity = _calculate_trend_maturity(df['close'])

        result_df['trend_strength'] = trend_strength
        result_df['trend_maturity'] = trend_maturity

        if hasattr(config, 'use_adaptive_thresholds') and config.use_adaptive_thresholds:
            long_thresh, short_thresh = _calculate_adaptive_thresholds(
                m_score_adj,
                trend_strength,
                vol_60d_safe,
                base_long=getattr(config, 'base_long_threshold',  70),
                base_short=getattr(config, 'base_short_threshold', 30),
            )
            result_df['adaptive_long_threshold']  = long_thresh
            result_df['adaptive_short_threshold'] = short_thresh

    # Optional component export (UNCHANGED)
    if return_components:
        result_df['log_returns_60d']   = log_returns_60d
        result_df['decay_convolution'] = numerator
        result_df['vol_normalization'] = denominator_safe
        result_df['ratio_raw']         = ratio_raw
        result_df['ratio_zscore']      = ratio_zscore
        result_df['ratio_clamped']     = ratio_clamped
        result_df['momentum_component']= momentum_component
        result_df['autocorr_60d']      = autocorr_60d
        result_df['vol_60d']           = vol_60d_safe
        result_df['sharpe_factor']     = sharpe_factor
        result_df['sharpe_delta']      = sharpe_delta   # ← B2-Rc-M2: new component

    logger.info(
        "M-Score computed: mean=%.2f, std=%.2f, range=[%.2f, %.2f]",
        m_score_adj.mean(), m_score_adj.std(),
        m_score_adj.min(), m_score_adj.max(),
    )

    return result_df


# ===========================================================================
# Alternative implementation (ENHANCED v2.0) — Sharpe fix applied
# ===========================================================================

class MomentumScoreCalculator:
    """
    Object-oriented M-Score calculator with enhancements.
    B2-Rc-M2: sharpe_delta stored as new attribute.
    """

    def __init__(self, df: pd.DataFrame, config: Optional[MomentumConfig] = None):
        if config is None:
            config = MomentumConfig()

        self.df     = df
        self.config = config
        self.prices = df['close']

        self.log_returns_60d:  Optional[pd.Series] = None
        self.numerator:        Optional[pd.Series] = None
        self.denominator:      Optional[pd.Series] = None
        self.ratio_raw:        Optional[pd.Series] = None
        self.ratio_zscore:     Optional[pd.Series] = None
        self.ratio_clamped:    Optional[pd.Series] = None
        self.autocorr_60d:     Optional[pd.Series] = None
        self.vol_60d:          Optional[pd.Series] = None
        self.sharpe_factor:    Optional[pd.Series] = None
        self.sharpe_delta:     Optional[pd.Series] = None  # ← B2-Rc-M2 NEW
        self.m_score:          Optional[pd.Series] = None
        self.m_score_adj:      Optional[pd.Series] = None
        self.trend_strength:   Optional[pd.Series] = None
        self.trend_maturity:   Optional[pd.Series] = None

    def compute(self) -> pd.Series:
        """Run complete M-Score calculation — delegates to functional API."""
        logger.info("Computing Momentum Score (M-Score) — Enhanced v2.0...")

        result_df      = compute_momentum_score(
            self.df,
            config=self.config,
            return_components=True,
        )

        # Populate instance attributes from result
        self.log_returns_60d   = result_df.get('log_returns_60d')
        self.numerator         = result_df.get('decay_convolution')
        self.denominator       = result_df.get('vol_normalization')
        self.ratio_raw         = result_df.get('ratio_raw')
        self.ratio_zscore      = result_df.get('ratio_zscore')
        self.ratio_clamped     = result_df.get('ratio_clamped')
        self.autocorr_60d      = result_df.get('autocorr_60d')
        self.vol_60d           = result_df.get('vol_60d')
        self.sharpe_factor     = result_df.get('sharpe_factor')
        self.sharpe_delta      = result_df.get('sharpe_delta')   # ← B2-Rc-M2
        self.m_score           = result_df['m_score']
        self.m_score_adj       = result_df['m_score_adj']

        if 'trend_strength' in result_df:
            self.trend_strength = result_df['trend_strength']
            self.trend_maturity = result_df['trend_maturity']

        logger.info("M-Score computation complete (Enhanced v2.0 / B2-Rc)")
        return self.m_score_adj


# ===========================================================================
# Validation (UNCHANGED)
# ===========================================================================

def validate_momentum_score(
    df: pd.DataFrame,
    score_col: str = 'm_score_adj',
) -> Tuple[bool, str]:
    """Validate M-Score computation quality. UNCHANGED."""
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


# ===========================================================================
# Convenience Functions (UNCHANGED)
# ===========================================================================

def compute_momentum_score_simple(
    prices: pd.Series,
    kappa: float = 0.03,
    beta_m: float = 0.2,
    window: int = MOMENTUM_NORM_WINDOW,    # ← B2-Rc-M4
) -> pd.Series:
    """Simplified M-Score from price series. UNCHANGED."""
    df        = pd.DataFrame({'close': prices})
    config    = MomentumConfig(kappa=kappa, beta_m=beta_m, window_60d=window)
    result_df = compute_momentum_score(df, config)
    return result_df['m_score_adj']


def momentum_signal(
    m_score: pd.Series,
    threshold_long: float  = 70.0,
    threshold_short: float = 30.0,
    use_enhancements: bool = False,
    trend_strength: Optional[pd.Series] = None,
    trend_maturity: Optional[pd.Series] = None,
    prices: Optional[pd.Series] = None,
    config: Optional[MomentumConfig] = None,
) -> pd.Series:
    """
    Generate trading signals from M-Score with optional enhancements.
    UNCHANGED — signal logic unaffected by B2-Rc score fixes.
    """
    signals = pd.Series(0, index=m_score.index)
    signals[m_score >= threshold_long]  = 1
    signals[m_score <= threshold_short] = -1

    if not use_enhancements:
        return signals

    if config is None:
        config = MomentumConfig()

    confirmation_days      = getattr(config, 'confirmation_days',               3)
    min_holding_days       = getattr(config, 'min_holding_days',                12)
    continuation_threshold = getattr(config, 'trend_continuation_threshold',    70)

    logger.debug("Applying signal enhancements...")

    if trend_strength is not None and trend_maturity is not None:
        signals = _apply_trend_continuation_filter(
            signals, trend_strength, trend_maturity, m_score, continuation_threshold,
        )
        logger.debug("  After trend continuation: %d changes", (signals.diff() != 0).sum())

    if confirmation_days > 0:
        signals = _apply_confirmation_filter(signals, m_score, confirmation_days)
        logger.debug("  After confirmation: %d changes", (signals.diff() != 0).sum())

    if min_holding_days > 0:
        signals = _apply_holding_period_filter(signals, min_holding_days)
        logger.debug("  After holding period: %d changes", (signals.diff() != 0).sum())

    return signals


# ===========================================================================
# Export
# ===========================================================================

__all__ = [
    'compute_momentum_score',
    'MomentumScoreCalculator',
    'validate_momentum_score',
    'compute_momentum_score_simple',
    'momentum_signal',
    'safe_rolling_zscore_momentum',
    'soft_clamp_momentum',
    'MOMENTUM_NORM_WINDOW',           # ← B2-Rc-M1 NEW
]
