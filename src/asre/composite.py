"""
ASRE Composite Score & ASRE-Medallion Implementation (ENHANCED v4.0)

✅ CRITICAL FIXES APPLIED:
1. Buy-the-dip detection (High F + Low T = Use F-score directly)
2. Component divergence handling (bypass aggressive normalization)
3. More sensitive z-score normalization (window: 60 → 50)
4. Better smoothness in soft clamping (0.5 → 0.4)
5. Reduced SCALE for more responsive tanh (2.0 → 1.8)
6. Relaxed Kalman validation (percentile-based jump detection)
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from .config import CompositeConfig, FundamentalsConfig, TechnicalConfig, MomentumConfig
from .momentum import compute_momentum_score
from .technical import compute_technical_score
from .fundamentals import compute_fundamental_score_universal as compute_fundamental_score
from .indicators import rolling_volatility, log_returns
from .calibration import (
    KalmanFilter1D,
    apply_kalman_filter_to_series,
    calibrate_logistic_weights,
    calibrate_noise_parameters,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numerical Safety Utilities (ENHANCED)
# ---------------------------------------------------------------------------

def safe_rolling_zscore(
    series: pd.Series,
    window: int = 50,
    min_periods: int = 25,
    vol_floor: float = 0.3,
) -> pd.Series:
    """
    Compute rolling z-score with numerical safety (ENHANCED for sensitivity).

    ✅ CHANGES:
    - Shorter window (50 vs 60) → more responsive to recent changes
    - Lower vol_floor (0.3 vs 0.5) → more sensitive to small variations
    - Lower min_periods (25 vs 30) → faster stabilization
    """
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()

    # Adaptive volatility floor
    global_std = series.std()
    adaptive_floor = max(vol_floor, global_std * 0.03)

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


def soft_clamp_zscore(
    z: pd.Series,
    lower: float = -4.0,
    upper: float = 4.0,
    smoothness: float = 0.4,
) -> pd.Series:
    """
    Soft clamping using smooth sigmoid transition (ENHANCED).

    ✅ CHANGES:
    - Lower smoothness (0.4 vs 0.5) → sharper response to changes
    """
    z_lower = lower + smoothness * np.log(1 + np.exp((z - lower) / smoothness))
    z_clamped = upper - smoothness * np.log(1 + np.exp((upper - z_lower) / smoothness))
    return z_clamped


def validate_kalman_input(
    series: pd.Series,
    max_value: float = 1000.0,
    max_diff: float = 40.0,
) -> Tuple[bool, str]:
    """
    Validate series before Kalman filtering (RELAXED).

    ✅ CHANGES:
    - Changed max_diff logic to use 90th percentile instead of absolute max
    - This allows occasional large jumps without rejecting the entire series
    """
    clean_series = series.dropna()

    if len(clean_series) == 0:
        return False, "All values are NaN"

    if np.isinf(clean_series).any():
        return False, "Series contains infinite values"

    if (clean_series.abs() > max_value).any():
        return False, f"Values exceed max_value={max_value}"

    # ✅ NEW: Use percentile-based jump detection (more forgiving)
    diffs = clean_series.diff().abs()
    p90_jump = np.nanpercentile(diffs.dropna(), 90)

    # Only reject if 90th percentile of jumps exceeds threshold
    if p90_jump > max_diff:
        return False, f"Extreme jump detected: {p90_jump:.2f}"

    return True, "Valid"


# ---------------------------------------------------------------------------
# ✅ NEW: Component Divergence Detection
# ---------------------------------------------------------------------------

def detect_component_divergence(
    scores: pd.DataFrame,
) -> Dict[str, float]:
    """
    Detect extreme divergence between F/T/M scores.

    Returns:
        Dict with divergence metrics and recommended handling strategy
    """
    latest_f = scores['f_score'].iloc[-1]
    latest_t = scores['t_score'].iloc[-1]
    latest_m = scores['m_score'].iloc[-1]

    # Calculate component variance
    component_variance = np.std([latest_f, latest_t, latest_m])

    # Detect buy-the-dip scenario (High F + Low T)
    is_buy_dip = (latest_f >= 70) and (latest_t <= 20)

    # Detect overbought weak fundamentals (Low F + High T)
    is_pump_risk = (latest_f <= 50) and (latest_t >= 80)

    # Detect balanced scenario
    is_balanced = component_variance < 25

    return {
        'variance': component_variance,
        'f_score': latest_f,
        't_score': latest_t,
        'm_score': latest_m,
        'is_buy_dip': is_buy_dip,
        'is_pump_risk': is_pump_risk,
        'is_balanced': is_balanced,
    }


def compute_direct_weighted_score(
    f_score: float,
    t_score: float,
    m_score: float,
    weights: np.ndarray,
) -> float:
    """
    Compute direct weighted score without normalization.

    Used for high-divergence scenarios (buy-the-dip, pump-risk).
    """
    return weights[0] * f_score + weights[1] * t_score + weights[2] * m_score


# ---------------------------------------------------------------------------
# Weight Optimization (UNCHANGED)
# ---------------------------------------------------------------------------

def optimize_weights(
    scores: pd.DataFrame,
    returns: Optional[pd.Series] = None,
    lambda_risk: float = 2.0,
    use_mle: bool = True,
) -> np.ndarray:
    """Optimize component weights using Markowitz mean-variance with optional MLE calibration."""
    score_cols = ['f_score', 't_score', 'm_score']
    missing = [col for col in score_cols if col not in scores.columns]
    if missing:
        logger.warning(f"Missing scores: {missing}, using equal weights")
        return np.array([1/3, 1/3, 1/3])

    S = scores[score_cols].dropna().values

    if len(S) < 30:
        logger.warning("Insufficient data for optimization, using default weights")
        return np.array([0.4, 0.35, 0.25])

    # MLE-based initial guess
    if use_mle and returns is not None:
        logger.debug("Using MLE for initial weight calibration...")
        try:
            binary_target = (returns > 0).astype(int)
            aligned = pd.concat([scores[score_cols], binary_target], axis=1).dropna()

            if len(aligned) >= 50:
                mle_weights, auc = calibrate_logistic_weights(
                    aligned[score_cols],
                    aligned.iloc[:, -1],
                    feature_names=score_cols,
                )
                w0 = np.clip(mle_weights, 0.05, 0.9)
                w0 = w0 / w0.sum()
                logger.info(f"MLE weights: {w0}, AUC: {auc:.3f}")
            else:
                w0 = np.array([0.4, 0.35, 0.25])
        except Exception as e:
            logger.warning(f"MLE calibration failed: {e}, using defaults")
            w0 = np.array([0.4, 0.35, 0.25])
    else:
        w0 = np.array([0.4, 0.35, 0.25])

    # Calculate covariance matrix
    cov_matrix = np.cov(S.T)
    mean_returns = S.mean(axis=0)

    def objective(w):
        portfolio_var = w @ cov_matrix @ w
        portfolio_return = w @ mean_returns
        portfolio_std = np.sqrt(portfolio_var)

        if portfolio_std < 1e-6:
            sharpe = 0
        else:
            sharpe = portfolio_return / portfolio_std

        return portfolio_var - lambda_risk * sharpe

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.1, 0.8) for _ in range(3)]

    try:
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100},
        )

        if result.success:
            optimal_w = result.x
            logger.info(
                f"Optimized weights: F={optimal_w[0]:.3f}, "
                f"T={optimal_w[1]:.3f}, M={optimal_w[2]:.3f}"
            )
            return optimal_w
        else:
            logger.warning("Optimization failed, using MLE/default weights")
            return w0

    except Exception as e:
        logger.error(f"Optimization error: {e}, using MLE/default weights")
        return w0


def adjust_weights_for_regime(
    base_weights: np.ndarray,
    vix: float,
    correlation_mt: float,
    phi: float = 0.1,
    psi: float = 0.1,
) -> np.ndarray:
    """Adjust weights based on market regime and score correlation."""
    w_F, w_T, w_M = base_weights

    vix_neutral = 20.0
    iv_regime = (vix - vix_neutral) / vix_neutral
    iv_regime = np.clip(iv_regime, -1, 1)

    w_F_adj = w_F * (1 + phi * iv_regime)
    w_M_adj = w_M * (1 - psi * correlation_mt)
    w_T_adj = w_T

    adjusted = np.array([w_F_adj, w_T_adj, w_M_adj])
    adjusted = adjusted / adjusted.sum()

    logger.debug(
        f"Regime-adjusted weights: F={adjusted[0]:.3f}, "
        f"T={adjusted[1]:.3f}, M={adjusted[2]:.3f} "
        f"(VIX={vix:.1f}, ρ_MT={correlation_mt:.3f})"
    )

    return adjusted


def compute_regime_adjustment(
    sector_beta: pd.Series,
    eta: float = 0.02,
    window: int = 60,
) -> pd.Series:
    """Compute regime adjustment from sector beta."""
    beta_sign = np.sign(sector_beta)

    decay_weights = np.exp(-eta * np.arange(window)[::-1])
    decay_weights = decay_weights / decay_weights.sum()

    def weighted_sum(values):
        if len(values) < window:
            return 0.0
        return np.dot(values, decay_weights)

    regime_adj = beta_sign.rolling(window=window).apply(weighted_sum, raw=True)
    regime_adj = regime_adj * 5
    regime_adj = np.clip(regime_adj, -10, 10)

    return regime_adj


# ---------------------------------------------------------------------------
# R_final Computation (ENHANCED WITH BUY-THE-DIP LOGIC)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# NEW: DIP QUALITY SCORE DATA CLASS
# ---------------------------------------------------------------------------

@dataclass
class DipQualityMetrics:
    """
    Comprehensive dip quality assessment metrics.

    Attributes:
        dip_quality_score: Overall dip quality (0-100)
        entry_timing_score: How early in the dip (0-100, higher = earlier)
        momentum_divergence: Momentum vs technical divergence indicator
        fundamental_strength: F-Score quality assessment
        expected_upside: Estimated upside potential (%)
        risk_reward_ratio: Risk/reward ratio
        dip_stage: Stage of dip (EARLY, MID, LATE, RECOVERY)
        confidence: Statistical confidence in assessment (0-100)
    """
    dip_quality_score: float
    entry_timing_score: float
    momentum_divergence: float
    fundamental_strength: float
    expected_upside: float
    risk_reward_ratio: float
    dip_stage: str
    confidence: float

    def to_dict(self) -> Dict:
        return {
            'dip_quality_score': round(self.dip_quality_score, 2),
            'entry_timing_score': round(self.entry_timing_score, 2),
            'momentum_divergence': round(self.momentum_divergence, 2),
            'fundamental_strength': round(self.fundamental_strength, 2),
            'expected_upside': round(self.expected_upside, 2),
            'risk_reward_ratio': round(self.risk_reward_ratio, 2),
            'dip_stage': self.dip_stage,
            'confidence': round(self.confidence, 2)
        }


# ---------------------------------------------------------------------------
# NEW: COMPUTE DIP QUALITY SCORE (✅ BUG FIXED)
# ---------------------------------------------------------------------------

def compute_dip_quality_score(
    f_score: float,
    t_score: float,
    m_score: float,
    variance: float,
    historical_volatility: Optional[float] = None,
    sector_beta: Optional[float] = None,
) -> DipQualityMetrics:
    """
    Compute comprehensive dip quality score.

    HIGH quality dip = Strong fundamentals + Early entry + Low momentum recovery
    LOW quality dip = Weak fundamentals OR Late entry OR Already recovered

    Args:
        f_score: Fundamental score (0-100)
        t_score: Technical score (0-100)
        m_score: Momentum score (0-100)
        variance: Component variance
        historical_volatility: Optional historical volatility (annualized %)
        sector_beta: Optional sector beta

    Returns:
        DipQualityMetrics object with all assessment metrics
    """

    # Step 1: Entry Timing Score (inversely related to momentum)
    # Lower momentum = earlier in dip = better entry timing
    entry_timing_score = 100.0 - m_score

    # Bonus for extremely low momentum (< 20%)
    if m_score < 20:
        entry_timing_score = min(100, entry_timing_score * 1.2)

    # Penalty for high momentum (> 80%)
    if m_score > 80:
        entry_timing_score = max(0, entry_timing_score * 0.5)


    # Step 2: Momentum Divergence Score
    # Measures how oversold technicals are vs momentum
    momentum_divergence = abs(t_score - m_score)

    # ✅ FIX: Initialize both variables at the start
    divergence_penalty = 0.0
    divergence_bonus = 0.0

    # High divergence with low T + high M = late-stage recovery (bad)
    # High divergence with low T + low M = early-stage dip (good)
    if t_score < 20 and m_score > 70:
        divergence_penalty = -30.0  # Late recovery
    elif t_score < 20 and m_score < 30:
        divergence_bonus = 20.0  # Perfect early dip


    # Step 3: Fundamental Strength Assessment
    # Strong fundamentals (F > 70%) increase dip quality
    if f_score >= 85:
        fundamental_strength = 100.0
        fundamental_multiplier = 1.3
    elif f_score >= 70:
        fundamental_strength = 85.0
        fundamental_multiplier = 1.15
    elif f_score >= 60:
        fundamental_strength = 70.0
        fundamental_multiplier = 1.0
    elif f_score >= 50:
        fundamental_strength = 50.0
        fundamental_multiplier = 0.85
    else:
        fundamental_strength = 30.0
        fundamental_multiplier = 0.6


    # Step 4: Core Dip Quality Formula
    # Base formula: F × (1 - M/100) × Entry_Timing_Factor
    base_quality = f_score * (1.0 - m_score / 100.0)

    # Apply entry timing boost
    entry_timing_factor = 1.0 + (entry_timing_score / 100.0) * 0.5

    # Apply fundamental multiplier
    dip_quality_raw = base_quality * entry_timing_factor * fundamental_multiplier

    # Add divergence adjustment
    dip_quality_raw += divergence_bonus
    dip_quality_raw += divergence_penalty

    # Clip to 0-100 range
    dip_quality_score = np.clip(dip_quality_raw, 0, 100)


    # Step 5: Determine Dip Stage
    if t_score < 10 and m_score < 20:
        dip_stage = "EARLY"  # Best entry point
    elif t_score < 20 and m_score < 40:
        dip_stage = "MID"  # Good entry point
    elif t_score < 20 and m_score >= 40:
        dip_stage = "LATE"  # Marginal entry point
    elif t_score < 30 and m_score >= 70:
        dip_stage = "RECOVERY"  # Already bouncing, too late
    else:
        dip_stage = "UNDEFINED"


    # Step 6: Expected Upside Calculation
    # Based on technical oversold level and fundamental strength
    technical_upside = (20.0 - t_score) * 0.5 if t_score < 20 else 0.0
    fundamental_upside = (f_score - 50.0) * 0.3 if f_score > 50 else 0.0

    expected_upside = technical_upside + fundamental_upside
    expected_upside = max(0, expected_upside)


    # Step 7: Risk-Reward Ratio
    # Downside risk increases with high momentum (already recovered)
    downside_risk = max(5.0, m_score * 0.15)  # Higher M = higher risk of pullback

    # Adjust for volatility if available
    if historical_volatility is not None:
        vol_adjustment = min(1.5, historical_volatility / 20.0)
        downside_risk *= vol_adjustment

    risk_reward_ratio = expected_upside / downside_risk if downside_risk > 0 else 0.0


    # Step 8: Confidence Score
    # Higher confidence when:
    # - Low component variance (scores agree)
    # - Clear dip stage identification
    # - Strong fundamentals

    variance_confidence = 100.0 - min(100, variance * 2.0)
    stage_confidence = {
        "EARLY": 95.0,
        "MID": 85.0,
        "LATE": 70.0,
        "RECOVERY": 60.0,
        "UNDEFINED": 40.0
    }.get(dip_stage, 50.0)

    fundamental_confidence = min(100, f_score * 1.2)

    confidence = (variance_confidence * 0.3 + 
                  stage_confidence * 0.4 + 
                  fundamental_confidence * 0.3)


    return DipQualityMetrics(
        dip_quality_score=dip_quality_score,
        entry_timing_score=entry_timing_score,
        momentum_divergence=momentum_divergence,
        fundamental_strength=fundamental_strength,
        expected_upside=expected_upside,
        risk_reward_ratio=risk_reward_ratio,
        dip_stage=dip_stage,
        confidence=confidence
    )

# ---------------------------------------------------------------------------
# NEW: ENHANCED MARKET CONTEXT MESSAGE
# ---------------------------------------------------------------------------

def generate_market_context_message(
    divergence: Dict,
    dip_quality: Optional[DipQualityMetrics] = None
) -> str:
    """
    Generate intelligent market context message based on dip quality.

    Args:
        divergence: Component divergence dict
        dip_quality: DipQualityMetrics object

    Returns:
        Formatted market context string
    """

    f_score = divergence['f_score']
    t_score = divergence['t_score']
    m_score = divergence['m_score']
    is_buy_dip = divergence['is_buy_dip']
    is_pump_risk = divergence['is_pump_risk']

    if is_pump_risk:
        return (
            f"⚠️ PUMP RISK\n"
            f"Weak fundamentals (F={f_score:.0f}%) with elevated technicals (T={t_score:.0f}%).\n"
            f"High risk of correction. Avoid entry."
        )

    if not is_buy_dip:
        return (
            f"📊 BALANCED SCENARIO\n"
            f"Components aligned: F={f_score:.0f}%, T={t_score:.0f}%, M={m_score:.0f}%"
        )

    # Buy-the-dip scenario - use Dip Quality Score
    if dip_quality is None:
        # Fallback if dip_quality not computed
        return (
            f"🎯 BUY THE DIP\n"
            f"Strong fundamentals ({f_score:.0f}%) at oversold levels ({t_score:.0f}%)"
        )

    # Enhanced message with Dip Quality Score
    if dip_quality.dip_stage == "EARLY":
        return (
            f"🎯 HIGH QUALITY DIP (Score: {dip_quality.dip_quality_score:.0f}/100)\n"
            f"Stage: {dip_quality.dip_stage} - Excellent entry timing\n"
            f"F={f_score:.0f}%, T={t_score:.0f}%, M={m_score:.0f}%\n"
            f"Expected upside: {dip_quality.expected_upside:.1f}% | R/R: {dip_quality.risk_reward_ratio:.2f}"
        )

    elif dip_quality.dip_stage == "MID":
        return (
            f"🎯 GOOD DIP OPPORTUNITY (Score: {dip_quality.dip_quality_score:.0f}/100)\n"
            f"Stage: {dip_quality.dip_stage} - Solid entry point\n"
            f"F={f_score:.0f}%, T={t_score:.0f}%, M={m_score:.0f}%\n"
            f"Expected upside: {dip_quality.expected_upside:.1f}% | R/R: {dip_quality.risk_reward_ratio:.2f}"
        )

    elif dip_quality.dip_stage == "LATE":
        return (
            f"⚠️ LATE-STAGE DIP (Score: {dip_quality.dip_quality_score:.0f}/100)\n"
            f"Stage: {dip_quality.dip_stage} - Momentum building (M={m_score:.0f}%)\n"
            f"Entry acceptable but not ideal. Consider waiting for pullback.\n"
            f"Expected upside: {dip_quality.expected_upside:.1f}% | R/R: {dip_quality.risk_reward_ratio:.2f}"
        )

    elif dip_quality.dip_stage == "RECOVERY":
        return (
            f"⚠️ DIP ALREADY RECOVERING (Score: {dip_quality.dip_quality_score:.0f}/100)\n"
            f"Stage: {dip_quality.dip_stage} - Momentum maxed (M={m_score:.0f}%)\n"
            f"Entry NOT recommended - wait for next dip or consolidation.\n"
            f"Risk/Reward unfavorable: {dip_quality.risk_reward_ratio:.2f}"
        )

    else:
        return (
            f"📊 DIP DETECTED (Score: {dip_quality.dip_quality_score:.0f}/100)\n"
            f"F={f_score:.0f}%, T={t_score:.0f}%, M={m_score:.0f}%\n"
            f"Stage: {dip_quality.dip_stage} | Confidence: {dip_quality.confidence:.0f}%"
        )


# ===========================================================================
# FULLY FIXED ASRE COMPOSITE FUNCTIONS WITH FUNDAMENTAL FLOOR PROTECTION
# ===========================================================================

def compute_asre_rating(
    df: pd.DataFrame,
    config: Optional[CompositeConfig] = None,
    fundamentals_config: Optional[FundamentalsConfig] = None,
    technical_config: Optional[TechnicalConfig] = None,
    momentum_config: Optional[MomentumConfig] = None,
    optimize_weights_flag: bool = True,
    use_kalman: bool = True,
    auto_calibrate_noise: bool = True,
    return_components: bool = False,
    compute_dip_quality: bool = True,
) -> pd.DataFrame:
    """
    Compute ASRE composite rating (R_final) with COMPLETE fundamental floor protection.

    ✅ INSTITUTIONAL-GRADE FIXES:
    1. F >= 65% fundamental floor for buy-the-dip scenarios
    2. Explicit momentum trap detection: F < 65 AND M > 80
    3. Lower variance threshold (30) to catch edge cases like TSLA
    4. Dip Quality Score integration (0-100)
    5. Severe penalties for distressed stocks (max 30/100)
    """

    if config is None:
        config = CompositeConfig()

    result_df = df.copy()

    # Step 1: Compute all three component scores
    logger.info("Computing component scores...")

    if 'f_score' not in result_df.columns:
        logger.debug("Computing F-Score...")
        result_df = compute_fundamental_score(result_df, config=fundamentals_config)

    if 't_score' not in result_df.columns:
        logger.debug("Computing T-Score...")
        result_df = compute_technical_score(result_df, config=technical_config)

    if 'm_score_adj' not in result_df.columns:
        logger.debug("Computing M-Score...")
        result_df = compute_momentum_score(result_df, config=momentum_config)
        result_df['m_score'] = result_df['m_score_adj']

    score_cols = ['f_score', 't_score', 'm_score']
    scores = result_df[score_cols].copy()

    # Step 2: Optimize or use base weights
    if optimize_weights_flag:
        logger.info("Optimizing component weights...")
        returns = None
        if 'close' in result_df.columns:
            returns = result_df['close'].pct_change().shift(-1)

        optimal_w = optimize_weights(
            scores,
            returns=returns,
            lambda_risk=config.lambda_risk,
            use_mle=True,
        )
    else:
        optimal_w = np.array([config.w_f_base, config.w_t_base, config.w_m_base])
        logger.info(f"Using base weights: {optimal_w}")

    # Step 3: Adjust weights for regime
    logger.debug("Adjusting weights for market regime...")

    vix = result_df['vix'].iloc[-1] if 'vix' in result_df.columns else 20.0
    corr_mt = scores[['t_score', 'm_score']].corr().iloc[0, 1]
    if np.isnan(corr_mt):
        corr_mt = 0.0

    adjusted_w = adjust_weights_for_regime(
        optimal_w,
        vix=vix,
        correlation_mt=corr_mt,
        phi=config.phi,
        psi=config.psi,
    )

    # ✅ DETECT COMPONENT DIVERGENCE
    logger.debug("Analyzing component divergence...")
    divergence = detect_component_divergence(scores)

    logger.info(
        f"📊 Component Analysis: F={divergence['f_score']:.0f}%, "
        f"T={divergence['t_score']:.0f}%, M={divergence['m_score']:.0f}%, "
        f"Variance={divergence['variance']:.1f}"
    )

    # ✅ COMPUTE DIP QUALITY SCORE
    dip_quality_metrics = None
    if compute_dip_quality and (divergence['is_buy_dip'] or divergence['variance'] > 30):
        logger.info("\n" + "="*60)
        logger.info("Computing Dip Quality Score...")

        hist_vol = None
        if 'close' in result_df.columns:
            returns_series = result_df['close'].pct_change()
            hist_vol = returns_series.std() * np.sqrt(252) * 100

        sector_beta = result_df['sector_beta'].iloc[-1] if 'sector_beta' in result_df.columns else None

        dip_quality_metrics = compute_dip_quality_score(
            f_score=divergence['f_score'],
            t_score=divergence['t_score'],
            m_score=divergence['m_score'],
            variance=divergence['variance'],
            historical_volatility=hist_vol,
            sector_beta=sector_beta
        )

        logger.info(f"🎯 DIP QUALITY ASSESSMENT:")
        logger.info(f"   Overall Score: {dip_quality_metrics.dip_quality_score:.1f}/100")
        logger.info(f"   Dip Stage: {dip_quality_metrics.dip_stage}")
        logger.info(f"   Entry Timing: {dip_quality_metrics.entry_timing_score:.1f}/100")
        logger.info(f"   Expected Upside: {dip_quality_metrics.expected_upside:.1f}%")
        logger.info(f"   Risk/Reward: {dip_quality_metrics.risk_reward_ratio:.2f}")
        logger.info(f"   Confidence: {dip_quality_metrics.confidence:.1f}%")
        logger.info("="*60 + "\n")

        if return_components:
            for key, value in dip_quality_metrics.to_dict().items():
                result_df[f'dip_{key}'] = value

    # Step 4: Compute weighted composite
    logger.debug("Computing weighted composite score...")

    weighted_scores = scores.values @ adjusted_w
    weighted_scores_series = pd.Series(weighted_scores, index=result_df.index)

    # ✅ CRITICAL: FUNDAMENTAL FLOOR PROTECTION + MOMENTUM TRAP DETECTION
    # Lower threshold from 35 to 30 to catch edge cases like TSLA
    if divergence['variance'] > 30 or divergence['is_buy_dip'] or divergence['is_pump_risk'] or \
       (divergence['f_score'] < 65 and divergence['m_score'] > 80):

        if divergence['is_buy_dip']:
            # FUNDAMENTAL QUALITY FLOOR: F >= 65% required
            if divergence['f_score'] >= 65 and dip_quality_metrics and dip_quality_metrics.dip_stage in ["EARLY", "MID"]:
                buy_dip_weights = np.array([0.85, 0.05, 0.10])
                logger.info(
                    f"🎯 HIGH QUALITY BUY-THE-DIP (F>65%, Score={dip_quality_metrics.dip_quality_score:.0f}): "
                    f"F={divergence['f_score']:.0f}%, T={divergence['t_score']:.0f}%, "
                    f"Stage={dip_quality_metrics.dip_stage}"
                )
            elif divergence['f_score'] >= 65 and dip_quality_metrics and dip_quality_metrics.dip_stage == "LATE":
                buy_dip_weights = np.array([0.70, 0.10, 0.20])
                logger.warning(
                    f"⚠️ MARGINAL DIP (F>=65%, LATE stage, Score={dip_quality_metrics.dip_quality_score:.0f}): "
                    f"F={divergence['f_score']:.0f}%, M={divergence['m_score']:.0f}%"
                )
            elif divergence['f_score'] < 65:
                # ❌ FUNDAMENTAL FLOOR VIOLATION
                logger.error(
                    f"❌ FUNDAMENTAL FLOOR VIOLATION: F={divergence['f_score']:.0f}% < 65% "
                    f"→ REJECTING MOMENTUM TRAP (M={divergence['m_score']:.0f}%)"
                )
                penalty_multiplier = 0.3 * (divergence['f_score'] / 65.0)
                composite_base = weighted_scores_series * penalty_multiplier
                composite_base = composite_base.clip(1, 30)  # Max 30 for distressed

                # Store penalty flag for Medallion to use
                result_df['_fundamental_floor_violated'] = True
                result_df['_penalty_multiplier'] = penalty_multiplier
            else:
                buy_dip_weights = np.array([0.70, 0.15, 0.15])
                logger.warning("🎯 DEFAULT BUY-THE-DIP weights applied")

            if divergence['f_score'] >= 65:
                direct_score = compute_direct_weighted_score(
                    divergence['f_score'],
                    divergence['t_score'],
                    divergence['m_score'],
                    buy_dip_weights
                )
                composite_base = pd.Series(direct_score, index=result_df.index)
                composite_base = composite_base.clip(1, 95)
                result_df['_fundamental_floor_violated'] = False

        elif divergence['is_pump_risk']:
            logger.warning(
                f"⚠️ PUMP RISK detected: F={divergence['f_score']:.0f}%, "
                f"T={divergence['t_score']:.0f}% → Applying heavy penalty"
            )
            pump_penalty = 0.6
            direct_score = compute_direct_weighted_score(
                divergence['f_score'],
                divergence['t_score'],
                divergence['m_score'],
                adjusted_w
            ) * pump_penalty
            composite_base = pd.Series(direct_score, index=result_df.index)
            composite_base = composite_base.clip(1, 50)
            result_df['_fundamental_floor_violated'] = False

        else:
            # High divergence or momentum trap detected
            if divergence['f_score'] < 65 and divergence['m_score'] > 80:
                logger.error(
                    f"❌ MOMENTUM TRAP DETECTED: F={divergence['f_score']:.0f}%, M={divergence['m_score']:.0f}% "
                    f"→ Applying distressed penalty"
                )
                penalty_multiplier = 0.3 * (divergence['f_score'] / 65.0)
                composite_base = weighted_scores_series * penalty_multiplier
                composite_base = composite_base.clip(1, 30)
                result_df['_fundamental_floor_violated'] = True
                result_df['_penalty_multiplier'] = penalty_multiplier
            else:
                logger.warning(
                    f"⚠️ HIGH DIVERGENCE detected (σ={divergence['variance']:.1f}) "
                    f"→ Using direct weighted score"
                )
                direct_score = compute_direct_weighted_score(
                    divergence['f_score'],
                    divergence['t_score'],
                    divergence['m_score'],
                    adjusted_w
                )
                composite_base = pd.Series(direct_score, index=result_df.index)
                composite_base = composite_base.clip(1, 95)
                result_df['_fundamental_floor_violated'] = False

    else:
        # Standard normalization for balanced scenarios
        logger.debug("Applying rolling z-score normalization...")

        z_rolling = safe_rolling_zscore(
            weighted_scores_series,
            window=50,
            min_periods=25,
            vol_floor=0.3,
        )

        z_clamped = soft_clamp_zscore(
            z_rolling,
            lower=-3.5,
            upper=3.5,
            smoothness=0.4,
        )

        SCALE = 1.8
        composite_base = 50.0 + 40.0 * np.tanh(z_clamped / SCALE)
        composite_base = composite_base.fillna(50.0)
        result_df['_fundamental_floor_violated'] = False

    # Step 5: Compute regime adjustment
    if 'sector_beta' in result_df.columns:
        logger.debug("Computing regime adjustment...")
        regime_adj = compute_regime_adjustment(
            result_df['sector_beta'],
            eta=config.eta,
        )
    else:
        logger.warning("sector_beta not found, skipping regime adjustment")
        regime_adj = pd.Series(0.0, index=result_df.index)

    # Step 6: Add regime adjustment
    r_final = composite_base + regime_adj
    r_final = np.clip(r_final, 1, 99)

    # Kalman filter
    if use_kalman:
        logger.debug("Validating Kalman filter input...")

        is_valid, msg = validate_kalman_input(
            r_final,
            max_value=100.0,
            max_diff=40.0,
        )

        if not is_valid:
            logger.warning(f"Kalman validation failed: {msg}, skipping filter")
            result_df['r_final'] = r_final
            result_df['r_final_kalman'] = r_final
        else:
            logger.debug("Applying Kalman filter...")

            process_noise = 0.05
            measurement_noise = config.sigma_obs

            if auto_calibrate_noise and len(r_final.dropna()) > 100:
                try:
                    ma_50 = r_final.rolling(50).mean()
                    residuals = r_final - ma_50
                    innovations = r_final.diff()

                    Q_hat, R_hat = calibrate_noise_parameters(
                        residuals.dropna(),
                        innovations.dropna(),
                    )

                    process_noise = np.clip(Q_hat, 0.001, 0.5)
                    measurement_noise = np.clip(R_hat, 0.01, 2.0)

                    logger.info(f"Calibrated noise: Q={process_noise:.4f}, R={measurement_noise:.4f}")
                except Exception as e:
                    logger.warning(f"Noise calibration failed: {e}, using defaults")

            try:
                filtered_df = apply_kalman_filter_to_series(
                    r_final,
                    process_noise=process_noise,
                    measurement_noise=measurement_noise,
                )

                result_df['r_final'] = r_final
                result_df['r_final_kalman'] = filtered_df['filtered_state']
                result_df['confidence_lower'] = filtered_df['lower_ci']
                result_df['confidence_upper'] = filtered_df['upper_ci']
                result_df['kalman_covariance'] = filtered_df['covariance']
            except Exception as e:
                logger.error(f"Kalman filter failed: {e}, using unfiltered rating")
                result_df['r_final'] = r_final
                result_df['r_final_kalman'] = r_final
    else:
        result_df['r_final'] = r_final
        result_df['r_final_kalman'] = r_final

    # Optional components
    if return_components:
        result_df['composite_base'] = composite_base
        result_df['regime_adjustment'] = regime_adj
        result_df['weighted_scores'] = weighted_scores_series
        result_df['weight_f'] = adjusted_w[0]
        result_df['weight_t'] = adjusted_w[1]
        result_df['weight_m'] = adjusted_w[2]
        result_df['vix_regime'] = (vix - 20.0) / 20.0
        result_df['corr_mt'] = corr_mt
        result_df['component_variance'] = divergence['variance']
        result_df['is_buy_dip'] = divergence['is_buy_dip']
        result_df['is_pump_risk'] = divergence['is_pump_risk']

        if dip_quality_metrics:
            result_df['market_context'] = generate_market_context_message(
                divergence, dip_quality_metrics
            )

    logger.info(
        f"R_final computed: mean={r_final.mean():.2f}, "
        f"std={r_final.std():.2f}, "
        f"range=[{r_final.min():.2f}, {r_final.max():.2f}]"
    )

    return result_df


# ===========================================================================
# FULLY FIXED MEDALLION FUNCTION WITH FUNDAMENTAL FLOOR
# ===========================================================================

def compute_asre_medallion(
    df: pd.DataFrame,
    config: Optional[CompositeConfig] = None,
    fundamentals_config: Optional[FundamentalsConfig] = None,
    technical_config: Optional[TechnicalConfig] = None,
    momentum_config: Optional[MomentumConfig] = None,
    return_components: bool = False,
    compute_dip_quality: bool = True,
) -> pd.DataFrame:
    """
    Compute ASRE-Medallion rating with SAME fundamental floor protection as R_Final.

    ✅ CRITICAL: Applies SAME penalties as compute_asre_rating()
    ✅ Checks for _fundamental_floor_violated flag from R_Final computation
    ✅ Rejects momentum traps (F < 65, M > 80) just like R_Final
    """

    if config is None:
        config = CompositeConfig()

    result_df = df.copy()

    # Ensure component scores exist
    if 'f_score' not in result_df.columns:
        result_df = compute_fundamental_score(result_df)
    if 't_score' not in result_df.columns:
        result_df = compute_technical_score(result_df, config=technical_config)
    if 'm_score_adj' not in result_df.columns:
        result_df = compute_momentum_score(result_df, config=momentum_config)
        result_df['m_score'] = result_df['m_score_adj']

    score_cols = ['f_score', 't_score', 'm_score']
    scores = result_df[score_cols]

    # Detect divergence for Medallion
    divergence = detect_component_divergence(scores)

    # ✅ CRITICAL CHECK: Did R_Final detect fundamental floor violation?
    if '_fundamental_floor_violated' in result_df.columns and result_df['_fundamental_floor_violated'].iloc[-1]:
        logger.error(
            f"❌ Medallion: FUNDAMENTAL FLOOR VIOLATION inherited from R_Final "
            f"(F={divergence['f_score']:.0f}%, M={divergence['m_score']:.0f}%)"
        )

        # Apply SAME penalty as R_Final
        penalty_multiplier = result_df['_penalty_multiplier'].iloc[-1] if '_penalty_multiplier' in result_df.columns else 0.2
        r_asre_base = pd.Series(divergence['f_score'] * penalty_multiplier, index=result_df.index)
        r_asre_base = r_asre_base.clip(1, 25)  # Even stricter for Medallion: max 25/100

        result_df['r_asre'] = r_asre_base

        logger.info(
            f"R_ASRE computed: mean={r_asre_base.mean():.2f}, "
            f"range=[{r_asre_base.min():.2f}, {r_asre_base.max():.2f}]"
        )

        return result_df

    # ✅ EXPLICIT MOMENTUM TRAP CHECK (same as R_Final)
    if divergence['f_score'] < 65 and divergence['m_score'] > 80:
        logger.error(
            f"❌ Medallion: MOMENTUM TRAP DETECTED "
            f"(F={divergence['f_score']:.0f}%, M={divergence['m_score']:.0f}%)"
        )

        penalty_multiplier = 0.25 * (divergence['f_score'] / 65.0)
        r_asre_base = pd.Series(divergence['f_score'] * penalty_multiplier, index=result_df.index)
        r_asre_base = r_asre_base.clip(1, 25)

        result_df['r_asre'] = r_asre_base

        logger.info(
            f"R_ASRE computed: mean={r_asre_base.mean():.2f}, "
            f"range=[{r_asre_base.min():.2f}, {r_asre_base.max():.2f}]"
        )

        return result_df

    # ✅ COMPUTE DIP QUALITY FOR MEDALLION
    dip_quality_metrics = None
    if compute_dip_quality and (divergence['is_buy_dip'] or divergence['variance'] > 30):
        logger.info("Computing Dip Quality Score for Medallion...")

        hist_vol = None
        if 'close' in result_df.columns:
            returns_series = result_df['close'].pct_change()
            hist_vol = returns_series.std() * np.sqrt(252) * 100

        sector_beta = result_df['sector_beta'].iloc[-1] if 'sector_beta' in result_df.columns else None

        dip_quality_metrics = compute_dip_quality_score(
            f_score=divergence['f_score'],
            t_score=divergence['t_score'],
            m_score=divergence['m_score'],
            variance=divergence['variance'],
            historical_volatility=hist_vol,
            sector_beta=sector_beta
        )

    # Calculate inverse volatility (risk parity)
    logger.debug("Computing inverse volatility weights (risk parity)...")

    vol_window = 20

    sigma_f = scores['f_score'].rolling(window=vol_window).std()
    sigma_t = scores['t_score'].rolling(window=vol_window).std()
    sigma_m = scores['m_score'].rolling(window=vol_window).std()

    f_floor = max(0.1, scores['f_score'].std() * 0.1)
    t_floor = max(0.1, scores['t_score'].std() * 0.1)
    m_floor = max(0.1, scores['m_score'].std() * 0.1)

    sigma_f = sigma_f.clip(lower=f_floor).fillna(scores['f_score'].std())
    sigma_t = sigma_t.clip(lower=t_floor).fillna(scores['t_score'].std())
    sigma_m = sigma_m.clip(lower=m_floor).fillna(scores['m_score'].std())

    inv_vol_f = 1 / sigma_f
    inv_vol_t = 1 / sigma_t
    inv_vol_m = 1 / sigma_m

    # Risk-parity weighted composite
    logger.debug("Computing risk-parity composite...")

    w_f = config.w_f_base
    w_t = config.w_t_base
    w_m = config.w_m_base

    numerator = (
        w_f * scores['f_score'] * inv_vol_f +
        w_t * scores['t_score'] * inv_vol_t +
        w_m * scores['m_score'] * inv_vol_m
    )

    denominator = inv_vol_f + inv_vol_t + inv_vol_m
    base_raw = numerator / (denominator + 1e-8)

    # ✅ APPLY DIP QUALITY LOGIC TO MEDALLION (with fundamental floor check)
    if divergence['variance'] > 30 or divergence['is_buy_dip']:
        if divergence['is_buy_dip'] and divergence['f_score'] >= 65:
            # Only apply dip logic if fundamentals pass
            if dip_quality_metrics:
                if dip_quality_metrics.dip_stage == "EARLY":
                    medallion_boost = dip_quality_metrics.dip_quality_score * 0.9
                    logger.info(
                        f"🎯 Medallion: EARLY DIP boost "
                        f"(Quality={dip_quality_metrics.dip_quality_score:.0f})"
                    )
                elif dip_quality_metrics.dip_stage == "MID":
                    medallion_boost = dip_quality_metrics.dip_quality_score * 0.75
                    logger.info(
                        f"🎯 Medallion: MID DIP adjustment "
                        f"(Quality={dip_quality_metrics.dip_quality_score:.0f})"
                    )
                elif dip_quality_metrics.dip_stage == "LATE":
                    medallion_boost = dip_quality_metrics.dip_quality_score * 0.60
                    logger.info(
                        f"⚠️ Medallion: LATE DIP (Quality={dip_quality_metrics.dip_quality_score:.0f})"
                    )
                else:  # RECOVERY
                    medallion_boost = dip_quality_metrics.dip_quality_score * 0.45
                    logger.warning(
                        f"⚠️ Medallion: RECOVERY STAGE "
                        f"(Quality={dip_quality_metrics.dip_quality_score:.0f})"
                    )

                r_asre_base = pd.Series(medallion_boost, index=result_df.index)
                r_asre_base = r_asre_base.clip(20, 95)
            else:
                logger.info("🎯 Medallion: Buy-the-dip adjustment applied")
                r_asre_base = pd.Series(divergence['f_score'] * 0.8, index=result_df.index)
                r_asre_base = r_asre_base.clip(50, 95)
        else:
            # High divergence - use direct mapping
            r_asre_base = base_raw.clip(20, 90)
    else:
        # Standard z-score normalization
        logger.debug("Applying rolling z-score to risk-parity composite...")

        z_medallion = safe_rolling_zscore(
            base_raw,
            window=50,
            min_periods=25,
            vol_floor=0.25,
        )

        z_medallion_clamped = soft_clamp_zscore(
            z_medallion,
            lower=-3.5,
            upper=3.5,
            smoothness=0.4,
        )

        r_asre_base = 50.0 + 40.0 * np.tanh(z_medallion_clamped / 1.8)
        r_asre_base = r_asre_base.fillna(50.0)

    # Compute acceleration term
    logger.debug("Computing rating acceleration term...")

    if 'r_final' in result_df.columns:
        rating_series = result_df['r_final']
    else:
        rating_series = r_asre_base

    velocity = rating_series.diff().fillna(0.0)
    acceleration = velocity.diff().fillna(0.0)

    accel_median = acceleration.median()
    accel_mad = (acceleration - accel_median).abs().median()
    accel_threshold = accel_median + 5 * accel_mad

    acceleration = acceleration.clip(lower=-accel_threshold, upper=accel_threshold)

    lag = 5
    lagged_acceleration = acceleration.shift(lag).fillna(0.0)

    lambda_accel = 0.08
    accel_term = lambda_accel * lagged_acceleration
    accel_term = np.clip(accel_term, -5, 5)

    # Final ASRE-Medallion rating
    r_asre = r_asre_base + accel_term
    r_asre = np.clip(r_asre, 2, 98)

    result_df['r_asre'] = r_asre

    if return_components:
        result_df['r_asre_base'] = r_asre_base
        result_df['sigma_f'] = sigma_f
        result_df['sigma_t'] = sigma_t
        result_df['sigma_m'] = sigma_m
        result_df['inv_vol_f'] = inv_vol_f
        result_df['inv_vol_t'] = inv_vol_t
        result_df['inv_vol_m'] = inv_vol_m
        result_df['rating_velocity'] = velocity
        result_df['rating_acceleration'] = acceleration
        result_df['accel_term'] = accel_term

        if dip_quality_metrics:
            for key, value in dip_quality_metrics.to_dict().items():
                result_df[f'medallion_dip_{key}'] = value

    logger.info(
        f"R_ASRE computed: mean={r_asre.mean():.2f}, "
        f"std={r_asre.std():.2f}, "
        f"range=[{r_asre.min():.2f}, {r_asre.max():.2f}]"
    )

    return result_df

# ---------------------------------------------------------------------------
# Complete Pipeline, Validation, Convenience (UNCHANGED)
# ---------------------------------------------------------------------------

def compute_complete_asre(
    df: pd.DataFrame,
    config: Optional[CompositeConfig] = None,
    fundamentals_config: Optional[FundamentalsConfig] = None,
    technical_config: Optional[TechnicalConfig] = None,
    momentum_config: Optional[MomentumConfig] = None,
    medallion: bool = True,
    return_all_components: bool = False,
) -> pd.DataFrame:
    """Compute complete ASRE rating system with full integration."""

    logger.info("=" * 70)
    logger.info("ASRE Complete Rating System - Enhanced (Saturation Fixed)")
    logger.info("=" * 70)

    result_df = compute_asre_rating(
        df,
        config=config,
        fundamentals_config=fundamentals_config,
        technical_config=technical_config,
        momentum_config=momentum_config,
        optimize_weights_flag=True,
        use_kalman=True,
        auto_calibrate_noise=True,
        return_components=return_all_components,
    )

    if medallion:
        logger.info("\nComputing ASRE-Medallion rating...")
        result_df = compute_asre_medallion(
            result_df,
            config=config,
            fundamentals_config=fundamentals_config,
            technical_config=technical_config,
            momentum_config=momentum_config,
            return_components=return_all_components,
        )

    logger.info("=" * 70)
    logger.info("ASRE computation complete")
    logger.info("=" * 70)

    return result_df


def validate_asre_rating(
    df: pd.DataFrame,
    rating_col: str = 'r_final',
) -> Tuple[bool, str]:
    """Validate ASRE rating quality."""
    if rating_col not in df.columns:
        return False, f"Column '{rating_col}' not found"

    rating = df[rating_col].dropna()

    if len(rating) == 0:
        return False, "All rating values are NaN"

    if (rating < 0).any() or (rating > 100).any():
        return False, f"Rating out of range: [{rating.min():.2f}, {rating.max():.2f}]"

    if np.isinf(rating).any():
        return False, "Rating contains infinite values"

    nan_pct = (df[rating_col].isna().sum() / len(df)) * 100
    if nan_pct > 50:
        return False, f"Too many NaN values: {nan_pct:.1f}%"

    if rating.std() < 1.0:
        logger.warning(f"Rating has low variance: std={rating.std():.4f}")

    return True, f"Rating valid: mean={rating.mean():.2f}, std={rating.std():.2f}"


def get_asre_rating(
    df: pd.DataFrame,
    rating_type: str = 'medallion',
) -> pd.Series:
    """Get specific ASRE rating."""
    if rating_type == 'final':
        return df['r_final']
    elif rating_type == 'kalman':
        return df['r_final_kalman']
    elif rating_type == 'medallion':
        return df['r_asre']
    else:
        raise ValueError(f"Unknown rating type: {rating_type}")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = [
    'compute_asre_rating',
    'compute_asre_medallion',
    'compute_complete_asre',
    'optimize_weights',
    'adjust_weights_for_regime',
    'compute_regime_adjustment',
    'validate_asre_rating',
    'get_asre_rating',
    'safe_rolling_zscore',
    'soft_clamp_zscore',
    'validate_kalman_input',
    'detect_component_divergence',
    'compute_direct_weighted_score',
]
