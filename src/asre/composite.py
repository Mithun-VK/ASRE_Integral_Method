"""
ASRE Composite Score & ASRE-Medallion Implementation (ENHANCED v4.2)

v4.2 Changes (FIX B2 — Score Freeze Detection & Cache Recovery):
  FIX B2a — check_and_warn_freeze():
    Scans the last N rows of every score column (f_score, t_score, m_score)
    after all three component scores are computed, before the weight
    optimisation step.  If any score is identical for freeze_window (5)
    or more consecutive rows it fires a WARNING in the run log and
    injects a human-readable 'score_freeze_flag' column into the DataFrame.
    Example flag: "⚑ SCORE FREEZE: T_SCORE frozen 10d @ 5.0"
    This directly catches the ICICIBANK T=5/M=100 lockup seen in testing
    (10 consecutive identical values — confirmed data pipeline freeze).

  FIX B2b — recover_stale_cache():
    Utility function that deletes .pkl / .json / .parquet cache files
    matching a given ticker from ~/.asre/cache (or a custom directory).
    Called automatically when --clear-cache is passed to the CLI, or
    manually after check_and_warn_freeze() has fired.
    CLI usage:
        python -m asre compute ICICIBANK.NS --clear-cache --export-pdf --mode ia

  FIX B2 call-site:
    check_and_warn_freeze(result_df, ticker) is called inside
    compute_asre_rating() immediately after all three component scores are
    present (after m_score assignment) and before the weight optimisation
    step, so any frozen score is caught and flagged in the PDF output.

Previously applied fixes (all retained unchanged):
  v4.1 — Fix 4.1 (IA label renames), Fix 4.2 (NaN guard on r_final)
  v4.0 — Buy-the-dip, divergence, z-score window, smoothness, SCALE, Kalman
  v4.3 — BUG-1 warm-up NaN fill, BUG-2 HIGH DIVERGENCE std=0, BUG-3 AUC gate
"""

from __future__ import annotations

from dataclasses import dataclass
import glob as _glob
import logging
import os as _os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    apply_walk_forward_weights,
    calibrate_logistic_weights,
    calibrate_noise_parameters,
    compute_walk_forward_weights,
    get_latest_weights,
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

    Changes vs v3.x:
    - Shorter window (50 vs 60) → more responsive to recent changes
    - Lower vol_floor (0.3 vs 0.5) → more sensitive to small variations
    - Lower min_periods (25 vs 30) → faster stabilization
    """
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std  = series.rolling(window=window, min_periods=min_periods).std()

    global_std     = series.std()
    adaptive_floor = max(vol_floor, global_std * 0.03)

    rolling_std_safe = rolling_std.clip(lower=adaptive_floor)
    z_score = (series - rolling_mean) / rolling_std_safe

    if z_score.isna().any():
        global_mean     = series.mean()
        global_std_safe = max(series.std(), adaptive_floor)
        global_z        = (series - global_mean) / global_std_safe
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

    Change: lower smoothness (0.4 vs 0.5) → sharper response to changes.
    """
    z_lower   = lower + smoothness * np.log(1 + np.exp((z - lower) / smoothness))
    z_clamped = upper - smoothness * np.log(1 + np.exp((upper - z_lower) / smoothness))
    return z_clamped


def validate_kalman_input(
    series: pd.Series,
    max_value: float = 1000.0,
    max_diff: float = 40.0,
) -> Tuple[bool, str]:
    """
    Validate series before Kalman filtering (RELAXED).

    Change: uses 90th-percentile jump detection instead of absolute max,
    allowing occasional large jumps without rejecting the entire series.
    """
    clean_series = series.dropna()

    if len(clean_series) == 0:
        return False, "All values are NaN"

    if np.isinf(clean_series).any():
        return False, "Series contains infinite values"

    if (clean_series.abs() > max_value).any():
        return False, f"Values exceed max_value={max_value}"

    diffs    = clean_series.diff().abs()
    p90_jump = np.nanpercentile(diffs.dropna(), 90)

    if p90_jump > max_diff:
        return False, f"Extreme jump detected: {p90_jump:.2f}"

    return True, "Valid"


# ---------------------------------------------------------------------------
# FIX B2a — Score Freeze Detection
# ---------------------------------------------------------------------------

def check_and_warn_freeze(
    df: pd.DataFrame,
    ticker: str = "",
    score_cols: Optional[List[str]] = None,
    freeze_window: int = 5,
) -> pd.DataFrame:
    """
    FIX B2a: Detect and warn about frozen (stuck) score values.

    A score is considered frozen if it is identical for `freeze_window` or
    more consecutive trailing rows.  This directly catches the ICICIBANK
    T=5 / M=100 lockup observed in ASRE v3.1 testing (10 consecutive
    identical rows — confirmed data pipeline / cache freeze).

    Behaviour
    ---------
    - Logs a WARNING per frozen score (visible in run log + CLI output)
      before PDF generation so the analyst is alerted before signing off.
    - Injects / updates 'score_freeze_flag' column in the returned DataFrame.
      Format per frozen score: "⚑ SCORE FREEZE: T_SCORE frozen 10d @ 5.0"
      Multiple frozen scores are pipe-separated.
      Empty string when no freeze is detected (safe for PDF rendering).

    Call site
    ---------
    Called inside compute_asre_rating() immediately after all three component
    scores (f_score, t_score, m_score) are present and before the weight
    optimisation step.  Caller passes result_df and ticker.

    Args:
        df:           DataFrame containing score columns.
        ticker:       Ticker symbol for log context (e.g. "ICICIBANK.NS").
        score_cols:   Columns to inspect (default: f_score, t_score, m_score).
        freeze_window: Minimum consecutive identical trailing rows to trigger.

    Returns:
        df with 'score_freeze_flag' column added or updated.
    """
    if score_cols is None:
        score_cols = ["f_score", "t_score", "m_score"]

    freeze_flags: List[str] = []

    for col in score_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if len(series) < freeze_window:
            continue

        last_val    = series.iloc[-1]
        freeze_count = 0
        for val in reversed(series.values):
            if val == last_val:
                freeze_count += 1
            else:
                break

        if freeze_count >= freeze_window:
            flag_str = (
                f"\u26d1 SCORE FREEZE: {col.upper()} "
                f"frozen {freeze_count}d @ {last_val:.1f}"
            )
            freeze_flags.append(flag_str)
            logger.warning(
                "SCORE FREEZE detected [%s]: %s identical for %d consecutive "
                "rows (value=%.1f) — possible stale price data or cache lock. "
                "Run with --clear-cache to force rescan.",
                ticker or "UNKNOWN", col, freeze_count, float(last_val),
            )

    df = df.copy()
    df["score_freeze_flag"] = " | ".join(freeze_flags) if freeze_flags else ""

    if not freeze_flags:
        logger.debug(
            "[%s] Score freeze check passed — no frozen scores detected.",
            ticker or "UNKNOWN",
        )

    return df


# ---------------------------------------------------------------------------
# FIX B2b — Stale Cache Recovery
# ---------------------------------------------------------------------------

def recover_stale_cache(
    ticker: str,
    cache_dir: Optional[str] = None,
    extensions: Tuple[str, ...] = (".pkl", ".json", ".parquet"),
    dry_run: bool = False,
) -> int:
    """
    FIX B2b: Delete stale cache files for a specific ticker.

    Called automatically when --clear-cache is passed to the CLI, or
    manually after check_and_warn_freeze() has fired a WARNING.

    File matching
    -------------
    Files are matched case-insensitively by ticker prefix after normalising
    dots to underscores so both forms are caught:
        ICICIBANK_NS_scores.pkl
        icicibank_ns_fundamentals.json

    CLI equivalent
    --------------
        python -m asre compute ICICIBANK.NS --clear-cache --export-pdf --mode ia

    Args:
        ticker:     Ticker symbol (e.g. "ICICIBANK.NS" or "ICICIBANK_NS").
        cache_dir:  Directory to search. Default: ~/.asre/cache
        extensions: File extensions to remove.
        dry_run:    If True, log matches but do not delete (preview mode).

    Returns:
        Number of files deleted (or would-delete count in dry_run mode).
    """
    if cache_dir is None:
        cache_dir = str(Path.home() / ".asre" / "cache")

    ticker_safe = ticker.upper().replace(".", "_")
    deleted     = 0

    for ext in extensions:
        pattern = _os.path.join(cache_dir, f"*{ticker_safe}*{ext}")
        matches = _glob.glob(pattern, recursive=False)
        for filepath in matches:
            if dry_run:
                logger.info("DRY-RUN recover_stale_cache: would delete %s", filepath)
                deleted += 1
            else:
                try:
                    _os.remove(filepath)
                    logger.info("Cache cleared: %s", filepath)
                    deleted += 1
                except OSError as exc:
                    logger.warning("Could not delete %s: %s", filepath, exc)

    action = "Would remove" if dry_run else "Removed"
    if deleted:
        logger.info(
            "recover_stale_cache [%s]: %s %d file(s) in %s",
            ticker, action, deleted, cache_dir,
        )
    else:
        logger.info(
            "recover_stale_cache [%s]: no cache files found in %s",
            ticker, cache_dir,
        )

    return deleted


# ---------------------------------------------------------------------------
# Component Divergence Detection
# ---------------------------------------------------------------------------

def detect_component_divergence(
    scores: pd.DataFrame,
) -> Dict[str, float]:
    """
    Detect extreme divergence between F/T/M scores.

    Returns dict with divergence metrics and recommended handling strategy.
    Internal keys ``is_pump_risk`` and ``is_buy_dip`` are code-internal
    booleans; they are not surfaced as user-visible strings anywhere.
    """
    latest_f = scores["f_score"].iloc[-1]
    latest_t = scores["t_score"].iloc[-1]
    latest_m = scores["m_score"].iloc[-1]

    component_variance = np.std([latest_f, latest_t, latest_m])

    is_buy_dip   = (latest_f >= 70) and (latest_t <= 20)
    is_pump_risk = (latest_f <= 50) and (latest_t >= 80)
    is_balanced  = component_variance < 25

    return {
        "variance":     component_variance,
        "f_score":      latest_f,
        "t_score":      latest_t,
        "m_score":      latest_m,
        "is_buy_dip":   is_buy_dip,
        "is_pump_risk": is_pump_risk,
        "is_balanced":  is_balanced,
    }


def compute_direct_weighted_score(
    f_score: float,
    t_score: float,
    m_score: float,
    weights: np.ndarray,
) -> float:
    """
    Compute direct weighted score without normalization.

    Used for high-divergence scenarios (dip opportunity, divergence alert).
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
    """Optimise component weights using Markowitz mean-variance with optional MLE calibration."""
    score_cols = ["f_score", "t_score", "m_score"]
    missing = [col for col in score_cols if col not in scores.columns]
    if missing:
        logger.warning("Missing scores: %s, using equal weights", missing)
        return np.array([1 / 3, 1 / 3, 1 / 3])

    S = scores[score_cols].dropna().values

    if len(S) < 30:
        logger.warning("Insufficient data for optimization, using default weights")
        return np.array([0.40, 0.30, 0.30])

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
                logger.debug("MLE weights (clipped+normalised): %s, AUC: %.3f", w0, auc)

                if auc >= 0.55:
                    logger.info(
                        "Optimized weights: F=%.3f, T=%.3f, M=%.3f "
                        "(MLE direct — AUC=%.3f)",
                        w0[0], w0[1], w0[2], auc,
                    )
                    return w0
            else:
                w0 = np.array([0.40, 0.30, 0.30])
        except Exception as exc:
            logger.warning("MLE calibration failed: %s, using defaults", exc)
            w0 = np.array([0.40, 0.30, 0.30])
    else:
        w0 = np.array([0.40, 0.30, 0.30])

    cov_matrix   = np.cov(S.T)
    mean_returns = S.mean(axis=0)

    def objective(w):
        portfolio_var    = w @ cov_matrix @ w
        portfolio_return = w @ mean_returns
        portfolio_std    = np.sqrt(portfolio_var)
        sharpe = 0 if portfolio_std < 1e-6 else portfolio_return / portfolio_std
        return portfolio_var - lambda_risk * sharpe

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.10, 0.80), (0.05, 0.60), (0.05, 0.60)]

    try:
        result = minimize(
            objective, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200},
        )
        if result.success:
            optimal_w = result.x
            logger.info(
                "Optimized weights: F=%.3f, T=%.3f, M=%.3f",
                optimal_w[0], optimal_w[1], optimal_w[2],
            )
            return optimal_w
        else:
            logger.warning("Optimization failed, using MLE/default weights")
            return w0
    except Exception as exc:
        logger.error("Optimization error: %s, using MLE/default weights", exc)
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
    iv_regime   = np.clip((vix - vix_neutral) / vix_neutral, -1, 1)

    w_F_adj = w_F * (1 + phi * iv_regime)
    w_M_adj = w_M * (1 - psi * correlation_mt)
    w_T_adj = w_T

    adjusted = np.array([w_F_adj, w_T_adj, w_M_adj])
    adjusted = adjusted / adjusted.sum()

    logger.debug(
        "Regime-adjusted weights: F=%.3f, T=%.3f, M=%.3f (VIX=%.1f, rho_MT=%.3f)",
        adjusted[0], adjusted[1], adjusted[2], vix, correlation_mt,
    )
    return adjusted


def compute_regime_adjustment(
    sector_beta: pd.Series,
    eta: float = 0.02,
    window: int = 60,
) -> pd.Series:
    """Compute regime adjustment from sector beta."""
    beta_sign = np.sign(sector_beta)

    decay_weights  = np.exp(-eta * np.arange(window)[::-1])
    decay_weights /= decay_weights.sum()

    def weighted_sum(values):
        if len(values) < window:
            return 0.0
        return np.dot(values, decay_weights)

    regime_adj = beta_sign.rolling(window=window).apply(weighted_sum, raw=True)
    regime_adj = np.clip(regime_adj * 5, -10, 10)
    return regime_adj


# ---------------------------------------------------------------------------
# DipQualityMetrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class DipQualityMetrics:
    """
    Comprehensive dip quality assessment metrics.

    Attributes:
        dip_quality_score:   Overall dip quality (0-100)
        entry_timing_score:  How early in the dip (0-100, higher = earlier)
        momentum_divergence: Momentum vs technical divergence indicator
        fundamental_strength: F-Score quality assessment
        expected_upside:     Estimated upside potential (%)
        risk_reward_ratio:   Risk/reward ratio
        dip_stage:           Stage of dip (EARLY, MID, LATE, RECOVERY)
        confidence:          Statistical confidence in assessment (0-100)
    """
    dip_quality_score:    float
    entry_timing_score:   float
    momentum_divergence:  float
    fundamental_strength: float
    expected_upside:      float
    risk_reward_ratio:    float
    dip_stage:            str
    confidence:           float

    def to_dict(self) -> Dict:
        return {
            "dip_quality_score":    round(self.dip_quality_score,    2),
            "entry_timing_score":   round(self.entry_timing_score,   2),
            "momentum_divergence":  round(self.momentum_divergence,  2),
            "fundamental_strength": round(self.fundamental_strength, 2),
            "expected_upside":      round(self.expected_upside,      2),
            "risk_reward_ratio":    round(self.risk_reward_ratio,    2),
            "dip_stage":            self.dip_stage,
            "confidence":           round(self.confidence,           2),
        }


# ---------------------------------------------------------------------------
# Dip Stage (tier-aware)
# ---------------------------------------------------------------------------

def get_dip_stage_tier_aware(
    f_score: float,
    m_score: float,
    t_score: float,
) -> Tuple[str, float]:
    """
    Determine dip stage with tier-aware thresholds.

    Higher quality stocks (S/A-tier) get more lenient momentum thresholds.
    Lower quality stocks (C/D-tier) require stricter entry timing.
    """
    if f_score >= 85:
        if m_score < 60:    return "EARLY",    1.00
        elif m_score < 75:  return "MID",      0.90
        elif m_score < 90:  return "LATE",     0.70
        else:               return "RECOVERY", 0.20
    elif f_score >= 70:
        if m_score < 50:    return "EARLY",    1.00
        elif m_score < 70:  return "MID",      0.85
        elif m_score < 85:  return "LATE",     0.60
        else:               return "RECOVERY", 0.20
    elif f_score >= 55:
        if m_score < 40:    return "EARLY",    0.95
        elif m_score < 60:  return "MID",      0.75
        elif m_score < 80:  return "LATE",     0.50
        else:               return "RECOVERY", 0.20
    else:
        if m_score < 30:    return "EARLY",    0.80
        elif m_score < 50:  return "MID",      0.60
        else:               return "LATE",     0.30


# ---------------------------------------------------------------------------
# Dip Quality Score
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
    Compute comprehensive dip quality score (TIER-AWARE).

    HIGH quality dip = Strong fundamentals + Early entry + Low momentum recovery
    LOW  quality dip = Weak fundamentals OR Late entry OR Already recovered
    """
    entry_timing_score = 100.0 - m_score
    if m_score < 20:
        entry_timing_score = min(100, entry_timing_score * 1.2)
    if m_score > 80:
        entry_timing_score = max(0, entry_timing_score * 0.5)

    momentum_divergence = abs(t_score - m_score)
    divergence_penalty  = 0.0
    divergence_bonus    = 0.0
    if t_score < 20 and m_score > 70:
        divergence_penalty = -30.0
    elif t_score < 20 and m_score < 30:
        divergence_bonus   = 20.0

    if f_score >= 85:
        fundamental_strength = 100.0; fundamental_multiplier = 1.40
    elif f_score >= 70:
        fundamental_strength = 90.0;  fundamental_multiplier = 1.25
    elif f_score >= 60:
        fundamental_strength = 75.0;  fundamental_multiplier = 1.00
    elif f_score >= 50:
        fundamental_strength = 55.0;  fundamental_multiplier = 0.85
    else:
        fundamental_strength = 35.0;  fundamental_multiplier = 0.60

    base_quality        = f_score * (1.0 - m_score / 100.0)
    entry_timing_factor = 1.0 + (entry_timing_score / 100.0) * 0.5
    dip_quality_raw     = base_quality * entry_timing_factor * fundamental_multiplier
    dip_quality_raw    += divergence_bonus + divergence_penalty
    dip_quality_score   = float(np.clip(dip_quality_raw, 0, 100))

    dip_stage, stage_multiplier = get_dip_stage_tier_aware(f_score, m_score, t_score)

    technical_upside   = (20.0 - t_score) * 0.5 if t_score < 20 else 0.0
    fundamental_upside = (f_score - 50.0) * 0.3  if f_score > 50 else 0.0
    expected_upside    = max(0, technical_upside + fundamental_upside)

    downside_risk = max(5.0, m_score * 0.15)
    if historical_volatility is not None:
        downside_risk *= min(1.5, historical_volatility / 20.0)
    risk_reward_ratio = expected_upside / downside_risk if downside_risk > 0 else 0.0

    variance_confidence    = 100.0 - min(100, variance * 2.0)
    stage_confidence = {
        "EARLY": 95.0, "MID": 85.0, "LATE": 70.0,
        "RECOVERY": 60.0, "UNDEFINED": 40.0,
    }.get(dip_stage, 50.0)
    fundamental_confidence = min(100, f_score * 1.2)
    confidence = (
        variance_confidence    * 0.3 +
        stage_confidence       * 0.4 +
        fundamental_confidence * 0.3
    )

    return DipQualityMetrics(
        dip_quality_score=dip_quality_score,
        entry_timing_score=entry_timing_score,
        momentum_divergence=momentum_divergence,
        fundamental_strength=fundamental_strength,
        expected_upside=expected_upside,
        risk_reward_ratio=risk_reward_ratio,
        dip_stage=dip_stage,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# FIX 4.1 — generate_market_context_message
# "BUY THE DIP"  → "DIP OPPORTUNITY"
# "PUMP RISK"    → "DIVERGENCE ALERT"
# ---------------------------------------------------------------------------

def generate_market_context_message(
    divergence: Dict,
    dip_quality: Optional[DipQualityMetrics] = None,
) -> str:
    """
    Generate intelligent market context message based on dip quality.

    Fix 4.1 (v4.1): IA-prohibited signal labels replaced:
      "PUMP RISK"   → "DIVERGENCE ALERT"
      "BUY THE DIP" → "DIP OPPORTUNITY"
    """
    f_score      = divergence["f_score"]
    t_score      = divergence["t_score"]
    m_score      = divergence["m_score"]
    is_buy_dip   = divergence["is_buy_dip"]
    is_pump_risk = divergence["is_pump_risk"]

    if is_pump_risk:
        return (
            f"⚠️ DIVERGENCE ALERT\n"
            f"Weak fundamentals (F={f_score:.0f}%) with elevated technicals "
            f"(T={t_score:.0f}%).\n"
            f"High risk of correction. Avoid entry."
        )

    if not is_buy_dip:
        return (
            f"📊 BALANCED SCENARIO\n"
            f"Components aligned: F={f_score:.0f}%, T={t_score:.0f}%, M={m_score:.0f}%"
        )

    if dip_quality is None:
        return (
            f"🎯 DIP OPPORTUNITY\n"
            f"Strong fundamentals ({f_score:.0f}%) at oversold levels ({t_score:.0f}%)"
        )

    if dip_quality.dip_stage == "EARLY":
        return (
            f"🎯 HIGH QUALITY DIP (Score: {dip_quality.dip_quality_score:.0f}/100)\n"
            f"Stage: {dip_quality.dip_stage} - Excellent entry timing\n"
            f"F={f_score:.0f}%, T={t_score:.0f}%, M={m_score:.0f}%\n"
            f"Expected upside: {dip_quality.expected_upside:.1f}% "
            f"| R/R: {dip_quality.risk_reward_ratio:.2f}"
        )
    elif dip_quality.dip_stage == "MID":
        return (
            f"🎯 GOOD DIP OPPORTUNITY (Score: {dip_quality.dip_quality_score:.0f}/100)\n"
            f"Stage: {dip_quality.dip_stage} - Solid entry point\n"
            f"F={f_score:.0f}%, T={t_score:.0f}%, M={m_score:.0f}%\n"
            f"Expected upside: {dip_quality.expected_upside:.1f}% "
            f"| R/R: {dip_quality.risk_reward_ratio:.2f}"
        )
    elif dip_quality.dip_stage == "LATE":
        return (
            f"⚠️ LATE-STAGE DIP (Score: {dip_quality.dip_quality_score:.0f}/100)\n"
            f"Stage: {dip_quality.dip_stage} - Momentum building (M={m_score:.0f}%)\n"
            f"Entry acceptable but not ideal. Consider waiting for pullback.\n"
            f"Expected upside: {dip_quality.expected_upside:.1f}% "
            f"| R/R: {dip_quality.risk_reward_ratio:.2f}"
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


# ---------------------------------------------------------------------------
# compute_asre_rating (FIX B2 call-site added; all v4.3 fixes retained)
# ---------------------------------------------------------------------------

def compute_asre_rating(
    df: pd.DataFrame,
    ticker: str,
    config: Optional[CompositeConfig] = None,
    fundamentals_config: Optional[FundamentalsConfig] = None,
    technical_config: Optional[TechnicalConfig] = None,
    momentum_config: Optional[MomentumConfig] = None,
    optimize_weights_flag: bool = True,
    use_kalman: bool = True,
    auto_calibrate_noise: bool = True,
    return_components: bool = False,
    compute_dip_quality: bool = True,
    walk_forward_weights: bool = True,
    wf_train_window: int = 252,
    wf_min_train_rows: int = 50,
) -> pd.DataFrame:
    """
    Compute ASRE composite rating (R_final) with fundamental floor protection.

    v4.2 FIX B2 (this version):
      check_and_warn_freeze() called after all three component scores are
      present (after m_score assignment) and before the weight optimisation
      step.  A 'score_freeze_flag' column is injected into result_df;
      its value appears on the PDF analyst notes page when non-empty.
      To recover from a freeze, run:
          python -m asre compute TICKER.NS --clear-cache --export-pdf --mode ia

    v4.3 fixes (retained):
      BUG-1 — NaN warm-up bleed (ffill/bfill on weighted_scores_series)
      BUG-2 — HIGH DIVERGENCE branch std=0 (use per-date series, not scalar)
      BUG-3 — AUC gate epsilon (>= AUC_GATE - 1e-4)

    v4.2 / v4.1 fixes (retained):
      Walk-forward zero-look-ahead weights; NaN guard; label renames.
    """
    if config is None:
        config = CompositeConfig()

    result_df = df.copy()

    # ── Step 1: Component scores ──────────────────────────────────────────
    logger.info("Computing component scores...")

    if "f_score" not in result_df.columns:
        logger.debug("Computing F-Score...")
        result_df = compute_fundamental_score(result_df, ticker=ticker, config=fundamentals_config)

    if "t_score" not in result_df.columns:
        logger.debug("Computing T-Score...")
        result_df = compute_technical_score(result_df, config=technical_config)

    if "m_score_adj" not in result_df.columns:
        logger.debug("Computing M-Score...")
        result_df = compute_momentum_score(result_df, config=momentum_config)
        result_df["m_score"] = result_df["m_score_adj"]

    # ── FIX B2: Score freeze detection ────────────────────────────────────
    # Runs immediately after all three scores are available.
    # Injects score_freeze_flag column; WARNING fires if any score is
    # identical for 5+ consecutive rows (ICICIBANK T=5/M=100 pattern).
    result_df = check_and_warn_freeze(result_df, ticker=ticker)

    score_cols = ["f_score", "t_score", "m_score"]
    scores     = result_df[score_cols].copy()

    # ── Step 2: Weight optimisation ───────────────────────────────────────
    logger.info("Optimizing component weights...")

    returns = (
        result_df["close"].pct_change().shift(-1)
        if "close" in result_df.columns else None
    )

    if optimize_weights_flag and walk_forward_weights and returns is not None:
        weight_df = compute_walk_forward_weights(
            scores, returns,
            feature_names=score_cols,
            train_window=wf_train_window,
            min_train_rows=wf_min_train_rows,
        )
        weighted_scores_series = apply_walk_forward_weights(
            scores, weight_df, feature_names=score_cols
        )
        optimal_w, latest_auc, wf_source = get_latest_weights(weight_df)
        logger.info(
            "Optimized weights: F=%.3f, T=%.3f, M=%.3f (%s — AUC=%.3f)",
            optimal_w[0], optimal_w[1], optimal_w[2], wf_source, latest_auc,
        )
        result_df.attrs["weight_df"] = weight_df

    elif optimize_weights_flag:
        optimal_w = optimize_weights(
            scores, returns=returns,
            lambda_risk=config.lambda_risk, use_mle=True,
        )
        weighted_scores_series = pd.Series(
            scores.values @ optimal_w, index=result_df.index
        )

    else:
        optimal_w = np.array([config.w_f_base, config.w_t_base, config.w_m_base])
        logger.info("Using base weights: %s", optimal_w)
        weighted_scores_series = pd.Series(
            scores.values @ optimal_w, index=result_df.index
        )

    # ── BUG-1 FIX: Forward-fill NaN warm-up rows ─────────────────────────
    wss_nan_before = weighted_scores_series.isna().sum()
    if wss_nan_before > 0:
        weighted_scores_series = (
            weighted_scores_series
            .ffill()
            .bfill()
        )
        wss_nan_after = weighted_scores_series.isna().sum()
        logger.info(
            "Warm-up NaN fill: %d → %d NaN rows in weighted_scores_series "
            "(bfill/ffill applied to %d rows).",
            wss_nan_before, wss_nan_after, wss_nan_before - wss_nan_after,
        )

    # ── Step 3: Regime adjustment ─────────────────────────────────────────
    vix     = result_df["vix"].iloc[-1] if "vix" in result_df.columns else 20.0
    corr_mt = scores[["t_score", "m_score"]].corr().iloc[0, 1]
    if np.isnan(corr_mt):
        corr_mt = 0.0

    adjusted_w = adjust_weights_for_regime(
        optimal_w, vix=vix, correlation_mt=corr_mt,
        phi=config.phi, psi=config.psi,
    )

    divergence = detect_component_divergence(scores)

    logger.info(
        "📊 Component Analysis: F=%.0f%%, T=%.0f%%, M=%.0f%%, Variance=%.1f",
        divergence["f_score"], divergence["t_score"],
        divergence["m_score"], divergence["variance"],
    )

    # ── Dip quality ───────────────────────────────────────────────────────
    dip_quality_metrics = None
    if compute_dip_quality and (divergence["is_buy_dip"] or divergence["variance"] > 30):
        logger.info("\n" + "=" * 60)
        logger.info("Computing Dip Quality Score...")

        hist_vol = None
        if "close" in result_df.columns:
            hist_vol = result_df["close"].pct_change().std() * np.sqrt(252) * 100

        sector_beta = (
            result_df["sector_beta"].iloc[-1]
            if "sector_beta" in result_df.columns else None
        )

        dip_quality_metrics = compute_dip_quality_score(
            f_score=divergence["f_score"],
            t_score=divergence["t_score"],
            m_score=divergence["m_score"],
            variance=divergence["variance"],
            historical_volatility=hist_vol,
            sector_beta=sector_beta,
        )

        logger.info("🎯 DIP QUALITY ASSESSMENT:")
        logger.info("   Overall Score: %.1f/100",   dip_quality_metrics.dip_quality_score)
        logger.info("   Dip Stage: %s",              dip_quality_metrics.dip_stage)
        logger.info("   Entry Timing: %.1f/100",     dip_quality_metrics.entry_timing_score)
        logger.info("   Expected Upside: %.1f%%",    dip_quality_metrics.expected_upside)
        logger.info("   Risk/Reward: %.2f",          dip_quality_metrics.risk_reward_ratio)
        logger.info("   Confidence: %.1f%%",         dip_quality_metrics.confidence)
        logger.info("=" * 60 + "\n")

        if return_components:
            for key, value in dip_quality_metrics.to_dict().items():
                result_df[f"dip_{key}"] = value

    # ── Step 4: Weighted composite + floor protection ─────────────────────
    if (
        divergence["variance"] > 30
        or divergence["is_buy_dip"]
        or divergence["is_pump_risk"]
        or (divergence["f_score"] < 65 and divergence["m_score"] > 80)
    ):
        if divergence["is_buy_dip"]:
            if divergence["f_score"] >= 65 and dip_quality_metrics and                     dip_quality_metrics.dip_stage in ["EARLY", "MID"]:
                buy_dip_weights = np.array([0.85, 0.05, 0.10])
                logger.info(
                    "🎯 HIGH QUALITY DIP OPPORTUNITY (F>65%%, Score=%.0f): "
                    "F=%.0f%%, T=%.0f%%, Stage=%s",
                    dip_quality_metrics.dip_quality_score,
                    divergence["f_score"], divergence["t_score"],
                    dip_quality_metrics.dip_stage,
                )
            elif divergence["f_score"] >= 65 and dip_quality_metrics and                     dip_quality_metrics.dip_stage == "LATE":
                buy_dip_weights = np.array([0.70, 0.10, 0.20])
                logger.warning(
                    "⚠️ MARGINAL DIP (F>=65%%, LATE stage, Score=%.0f): "
                    "F=%.0f%%, M=%.0f%%",
                    dip_quality_metrics.dip_quality_score,
                    divergence["f_score"], divergence["m_score"],
                )
            elif divergence["f_score"] < 65:
                logger.error(
                    "❌ FUNDAMENTAL FLOOR VIOLATION: F=%.0f%% < 65%% "
                    "→ REJECTING MOMENTUM TRAP (M=%.0f%%)",
                    divergence["f_score"], divergence["m_score"],
                )
                penalty_multiplier = 0.3 * (divergence["f_score"] / 65.0)
                composite_base     = (weighted_scores_series * penalty_multiplier).clip(1, 30)
                result_df["_fundamental_floor_violated"] = True
                result_df["_penalty_multiplier"]          = penalty_multiplier
            else:
                buy_dip_weights = np.array([0.70, 0.15, 0.15])
                logger.warning("🎯 DEFAULT DIP OPPORTUNITY weights applied")

            if divergence["f_score"] >= 65:
                direct_score   = compute_direct_weighted_score(
                    divergence["f_score"], divergence["t_score"],
                    divergence["m_score"], buy_dip_weights,
                )
                composite_base = pd.Series(direct_score, index=result_df.index).clip(1, 95)
                result_df["_fundamental_floor_violated"] = False

        elif divergence["is_pump_risk"]:
            logger.warning(
                "⚠️ DIVERGENCE ALERT: Weak F (%.0f%%), elevated T (%.0f%%) "
                "→ Applying heavy penalty",
                divergence["f_score"], divergence["t_score"],
            )
            pump_penalty   = 0.6
            direct_score   = compute_direct_weighted_score(
                divergence["f_score"], divergence["t_score"],
                divergence["m_score"], adjusted_w,
            ) * pump_penalty
            composite_base = pd.Series(direct_score, index=result_df.index).clip(1, 50)
            result_df["_fundamental_floor_violated"] = False

        else:
            if divergence["f_score"] < 65 and divergence["m_score"] > 80:
                logger.error(
                    "❌ MOMENTUM TRAP DETECTED: F=%.0f%%, M=%.0f%% "
                    "→ Applying distressed penalty",
                    divergence["f_score"], divergence["m_score"],
                )
                penalty_multiplier = 0.3 * (divergence["f_score"] / 65.0)
                composite_base     = (weighted_scores_series * penalty_multiplier).clip(1, 30)
                result_df["_fundamental_floor_violated"] = True
                result_df["_penalty_multiplier"]          = penalty_multiplier
            else:
                # BUG-2 FIX: use per-date weighted_scores_series, not scalar broadcast
                logger.warning(
                    "⚠️ HIGH DIVERGENCE detected (variance=%.1f) "
                    "→ Using walk-forward weighted score (per-date)",
                    divergence["variance"],
                )
                composite_base = weighted_scores_series.clip(1, 95)
                result_df["_fundamental_floor_violated"] = False

    else:
        if walk_forward_weights and optimize_weights_flag and returns is not None:
            composite_base = weighted_scores_series.clip(1, 99)
            composite_base = composite_base.ewm(span=15, min_periods=1).mean()
            logger.debug("Walk-forward composite used directly (no z-score normalisation).")
        else:
            z_rolling  = safe_rolling_zscore(
                weighted_scores_series, window=25, min_periods=12, vol_floor=0.3
            )
            z_clamped  = soft_clamp_zscore(z_rolling, lower=-3.5, upper=3.5, smoothness=0.4)
            SCALE      = 1.8
            composite_base = 50.0 + 40.0 * np.tanh(z_clamped / SCALE)
            composite_base = composite_base.ewm(span=15, min_periods=1).mean()
        result_df["_fundamental_floor_violated"] = False

    # ── Step 5: Regime adjustment ─────────────────────────────────────────
    if "sector_beta" in result_df.columns:
        regime_adj = compute_regime_adjustment(result_df["sector_beta"], eta=config.eta)
    else:
        logger.warning("sector_beta not found, skipping regime adjustment")
        regime_adj = pd.Series(0.0, index=result_df.index)

    # ── Step 6: Build r_final ─────────────────────────────────────────────
    regime_adj = regime_adj.fillna(0.0)
    r_final    = np.clip(composite_base + regime_adj, 1, 99)

    # Fix 4.2 — NaN guard
    nan_count = r_final.isna().sum()
    if nan_count == len(r_final):
        raise ValueError(
            "R_final is entirely NaN. All composite scores failed. "
            "Likely cause: insufficient price rows or all-NaN FTM inputs. "
            f"DataFrame has {len(r_final)} rows."
        )
    elif nan_count > 0:
        logger.warning(
            "R_final has %d/%d NaN rows. Partial data. Results may be unreliable.",
            nan_count, len(r_final),
        )

    # ── Kalman filter ─────────────────────────────────────────────────────
    if use_kalman:
        is_valid, msg = validate_kalman_input(r_final, max_value=100.0, max_diff=40.0)

        if not is_valid:
            logger.warning("Kalman validation failed: %s, skipping filter", msg)
            result_df["r_final"]        = r_final
            result_df["r_final_kalman"] = r_final
        else:
            process_noise     = 0.05
            measurement_noise = config.sigma_obs

            if auto_calibrate_noise and len(r_final.dropna()) > 100:
                try:
                    ma_50       = r_final.rolling(50).mean()
                    residuals   = r_final - ma_50
                    innovations = r_final.diff()
                    Q_hat, R_hat = calibrate_noise_parameters(
                        residuals.dropna(), innovations.dropna()
                    )
                    process_noise     = np.clip(Q_hat, 0.001, 2.0)
                    measurement_noise = np.clip(R_hat, 0.01,  2.0)
                    logger.info(
                        "Calibrated noise: Q=%.4f, R=%.4f",
                        process_noise, measurement_noise,
                    )
                except Exception as exc:
                    logger.warning("Noise calibration failed: %s, using defaults", exc)

            try:
                filtered_df = apply_kalman_filter_to_series(
                    r_final,
                    process_noise=process_noise,
                    measurement_noise=measurement_noise,
                )
                result_df["r_final"]           = r_final
                result_df["r_final_kalman"]    = filtered_df["filtered_state"]
                result_df["confidence_lower"]  = filtered_df["lower_ci"]
                result_df["confidence_upper"]  = filtered_df["upper_ci"]
                result_df["kalman_covariance"] = filtered_df["covariance"]
            except Exception as exc:
                logger.error("Kalman filter failed: %s, using unfiltered rating", exc)
                result_df["r_final"]        = r_final
                result_df["r_final_kalman"] = r_final
    else:
        result_df["r_final"]        = r_final
        result_df["r_final_kalman"] = r_final

    if return_components:
        result_df["composite_base"]     = composite_base
        result_df["regime_adjustment"]  = regime_adj
        result_df["weighted_scores"]    = weighted_scores_series
        result_df["weight_f"]           = adjusted_w[0]
        result_df["weight_t"]           = adjusted_w[1]
        result_df["weight_m"]           = adjusted_w[2]
        result_df["vix_regime"]         = (vix - 20.0) / 20.0
        result_df["corr_mt"]            = corr_mt
        result_df["component_variance"] = divergence["variance"]
        result_df["is_buy_dip"]         = divergence["is_buy_dip"]
        result_df["is_pump_risk"]       = divergence["is_pump_risk"]

        if dip_quality_metrics:
            result_df["market_context"] = generate_market_context_message(
                divergence, dip_quality_metrics
            )

    logger.info(
        "R_final computed: mean=%.2f, std=%.2f, range=[%.2f, %.2f]",
        r_final.mean(), r_final.std(), r_final.min(), r_final.max(),
    )

    return result_df


# ---------------------------------------------------------------------------
# compute_asre_medallion (unchanged logic; log messages updated for Fix 4.1)
# ---------------------------------------------------------------------------

def compute_asre_medallion(
    df: pd.DataFrame,
    ticker: str,
    config: Optional[CompositeConfig] = None,
    fundamentals_config: Optional[FundamentalsConfig] = None,
    technical_config: Optional[TechnicalConfig] = None,
    momentum_config: Optional[MomentumConfig] = None,
    return_components: bool = False,
    compute_dip_quality: bool = True,
) -> pd.DataFrame:
    """
    Compute ASRE-Medallion rating with SAME fundamental floor protection as R_Final.

    Applies the same penalties as compute_asre_rating():
    - Inherits _fundamental_floor_violated flag
    - Rejects momentum traps (F < 65, M > 80)
    - Rejects divergence alerts (F < 50, T > 80)
    """
    if config is None:
        config = CompositeConfig()

    result_df = df.copy()

    if "f_score" not in result_df.columns:
        result_df = compute_fundamental_score(result_df, ticker=ticker, config=fundamentals_config)
    if "t_score" not in result_df.columns:
        result_df = compute_technical_score(result_df, config=technical_config)
    if "m_score_adj" not in result_df.columns:
        result_df = compute_momentum_score(result_df, config=momentum_config)
        result_df["m_score"] = result_df["m_score_adj"]

    score_cols = ["f_score", "t_score", "m_score"]
    scores     = result_df[score_cols]
    divergence = detect_component_divergence(scores)

    if "_fundamental_floor_violated" in result_df.columns and             result_df["_fundamental_floor_violated"].iloc[-1]:
        logger.error(
            "❌ Medallion: FUNDAMENTAL FLOOR VIOLATION inherited from R_Final "
            "(F=%.0f%%, M=%.0f%%)",
            divergence["f_score"], divergence["m_score"],
        )
        penalty_multiplier = (
            result_df["_penalty_multiplier"].iloc[-1]
            if "_penalty_multiplier" in result_df.columns else 0.2
        )
        r_asre_base = pd.Series(
            divergence["f_score"] * penalty_multiplier, index=result_df.index
        ).clip(1, 25)
        result_df["r_asre"] = r_asre_base
        logger.info(
            "R_ASRE computed: mean=%.2f, range=[%.2f, %.2f]",
            r_asre_base.mean(), r_asre_base.min(), r_asre_base.max(),
        )
        return result_df

    if divergence["f_score"] < 65 and divergence["m_score"] > 80:
        logger.error(
            "❌ Medallion: MOMENTUM TRAP DETECTED (F=%.0f%%, M=%.0f%%)",
            divergence["f_score"], divergence["m_score"],
        )
        penalty_multiplier = 0.25 * (divergence["f_score"] / 65.0)
        r_asre_base = pd.Series(
            divergence["f_score"] * penalty_multiplier, index=result_df.index
        ).clip(1, 25)
        result_df["r_asre"] = r_asre_base
        logger.info(
            "R_ASRE computed: mean=%.2f, range=[%.2f, %.2f]",
            r_asre_base.mean(), r_asre_base.min(), r_asre_base.max(),
        )
        return result_df

    if divergence["is_pump_risk"]:
        logger.warning(
            "⚠️ Medallion: DIVERGENCE ALERT (F=%.0f%%, T=%.0f%%)",
            divergence["f_score"], divergence["t_score"],
        )
        r_asre_base = pd.Series(
            divergence["f_score"] * 0.6, index=result_df.index
        ).clip(1, 30)
        result_df["r_asre"] = r_asre_base
        logger.info(
            "R_ASRE computed: mean=%.2f, range=[%.2f, %.2f]",
            r_asre_base.mean(), r_asre_base.min(), r_asre_base.max(),
        )
        return result_df

    dip_quality_metrics = None
    if compute_dip_quality and (divergence["is_buy_dip"] or divergence["variance"] > 30):
        logger.info("Computing Dip Quality Score for Medallion...")
        hist_vol    = None
        if "close" in result_df.columns:
            hist_vol = result_df["close"].pct_change().std() * np.sqrt(252) * 100
        sector_beta = (
            result_df["sector_beta"].iloc[-1]
            if "sector_beta" in result_df.columns else None
        )
        dip_quality_metrics = compute_dip_quality_score(
            f_score=divergence["f_score"], t_score=divergence["t_score"],
            m_score=divergence["m_score"],  variance=divergence["variance"],
            historical_volatility=hist_vol, sector_beta=sector_beta,
        )

    vol_window = 20
    sigma_f = scores["f_score"].rolling(window=vol_window).std()
    sigma_t = scores["t_score"].rolling(window=vol_window).std()
    sigma_m = scores["m_score"].rolling(window=vol_window).std()

    f_floor = max(0.1, scores["f_score"].std() * 0.1)
    t_floor = max(0.1, scores["t_score"].std() * 0.1)
    m_floor = max(0.1, scores["m_score"].std() * 0.1)

    sigma_f = sigma_f.clip(lower=f_floor).fillna(scores["f_score"].std())
    sigma_t = sigma_t.clip(lower=t_floor).fillna(scores["t_score"].std())
    sigma_m = sigma_m.clip(lower=m_floor).fillna(scores["m_score"].std())

    inv_vol_f = 1 / sigma_f
    inv_vol_t = 1 / sigma_t
    inv_vol_m = 1 / sigma_m

    w_f = config.w_f_base
    w_t = config.w_t_base
    w_m = config.w_m_base

    numerator = (
        w_f * scores["f_score"] * inv_vol_f +
        w_t * scores["t_score"] * inv_vol_t +
        w_m * scores["m_score"] * inv_vol_m
    )
    denominator = (w_f * inv_vol_f + w_t * inv_vol_t + w_m * inv_vol_m)
    base_raw    = numerator / (denominator + 1e-8)

    if divergence["variance"] > 30 or divergence["is_buy_dip"]:
        if divergence["is_buy_dip"] and divergence["f_score"] >= 65:
            if dip_quality_metrics:
                stage_boost = {
                    "EARLY":    dip_quality_metrics.dip_quality_score * 0.90,
                    "MID":      dip_quality_metrics.dip_quality_score * 0.75,
                    "LATE":     dip_quality_metrics.dip_quality_score * 0.60,
                    "RECOVERY": dip_quality_metrics.dip_quality_score * 0.45,
                }.get(dip_quality_metrics.dip_stage,
                      dip_quality_metrics.dip_quality_score * 0.60)

                stage_label = {
                    "EARLY":    "🎯 Medallion: EARLY DIP boost",
                    "MID":      "🎯 Medallion: MID DIP adjustment",
                    "LATE":     "⚠️ Medallion: LATE DIP",
                    "RECOVERY": "⚠️ Medallion: RECOVERY STAGE",
                }.get(dip_quality_metrics.dip_stage, "🎯 Medallion: DIP")

                logger.info("%s (Quality=%.0f)", stage_label, dip_quality_metrics.dip_quality_score)
                dip_multiplier = stage_boost / max(divergence["f_score"], 1.0)
                r_asre_base = (scores["f_score"] * dip_multiplier).clip(20, 95)
            else:
                logger.info("🎯 Medallion: Dip opportunity adjustment applied")
                r_asre_base = (scores["f_score"] * 0.8).clip(50, 95)
        else:
            r_asre_base = base_raw.clip(20, 90)
    else:
        z_medallion         = safe_rolling_zscore(base_raw, window=50, min_periods=25, vol_floor=0.25)
        z_medallion_clamped = soft_clamp_zscore(z_medallion, lower=-3.5, upper=3.5, smoothness=0.4)
        r_asre_base         = 50.0 + 40.0 * np.tanh(z_medallion_clamped / 1.8)
        r_asre_base         = r_asre_base.fillna(50.0)

    rating_series = result_df["r_final"] if "r_final" in result_df.columns else r_asre_base
    velocity      = rating_series.diff().fillna(0.0)
    acceleration  = velocity.diff().fillna(0.0)

    accel_median    = acceleration.median()
    accel_mad       = (acceleration - accel_median).abs().median()
    accel_threshold = accel_median + 5 * accel_mad
    acceleration    = acceleration.clip(lower=-accel_threshold, upper=accel_threshold)

    accel_term = np.clip(0.08 * acceleration.shift(5).fillna(0.0), -5, 5)

    r_asre = np.clip(r_asre_base + accel_term, 2, 98)
    result_df["r_asre"] = r_asre

    if return_components:
        result_df["r_asre_base"]         = r_asre_base
        result_df["sigma_f"]             = sigma_f
        result_df["sigma_t"]             = sigma_t
        result_df["sigma_m"]             = sigma_m
        result_df["inv_vol_f"]           = inv_vol_f
        result_df["inv_vol_t"]           = inv_vol_t
        result_df["inv_vol_m"]           = inv_vol_m
        result_df["rating_velocity"]     = velocity
        result_df["rating_acceleration"] = acceleration
        result_df["accel_term"]          = accel_term
        if dip_quality_metrics:
            for key, value in dip_quality_metrics.to_dict().items():
                result_df[f"medallion_dip_{key}"] = value

    logger.info(
        "R_ASRE computed: mean=%.2f, std=%.2f, range=[%.2f, %.2f]",
        r_asre.mean(), r_asre.std(), r_asre.min(), r_asre.max(),
    )
    return result_df


# ---------------------------------------------------------------------------
# Complete Pipeline
# ---------------------------------------------------------------------------

def compute_complete_asre(
    df: pd.DataFrame,
    ticker: str,
    config: Optional[CompositeConfig] = None,
    fundamentals_config: Optional[FundamentalsConfig] = None,
    technical_config: Optional[TechnicalConfig] = None,
    momentum_config: Optional[MomentumConfig] = None,
    medallion: bool = True,
    return_all_components: bool = True,
) -> pd.DataFrame:
    """Compute complete ASRE rating system with full integration."""
    logger.info("=" * 70)
    logger.info("ASRE Complete Rating System - Enhanced (Saturation Fixed)")
    logger.info("=" * 70)

    result_df = compute_asre_rating(
        df,
        ticker=ticker,
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
            ticker=ticker,
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


# ---------------------------------------------------------------------------
# Validation & Convenience
# ---------------------------------------------------------------------------

def validate_asre_rating(
    df: pd.DataFrame,
    rating_col: str = "r_final",
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
        logger.warning("Rating has low variance: std=%.4f", rating.std())

    return True, f"Rating valid: mean={rating.mean():.2f}, std={rating.std():.2f}"


def get_asre_rating(
    df: pd.DataFrame,
    rating_type: str = "medallion",
) -> pd.Series:
    """Get a specific ASRE rating series by type."""
    if rating_type == "final":
        return df["r_final"]
    elif rating_type == "kalman":
        return df["r_final_kalman"]
    elif rating_type == "medallion":
        return df["r_asre"]
    else:
        raise ValueError(f"Unknown rating type: '{rating_type}'")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = [
    "compute_asre_rating",
    "compute_asre_medallion",
    "compute_complete_asre",
    "optimize_weights",
    "adjust_weights_for_regime",
    "compute_regime_adjustment",
    "validate_asre_rating",
    "get_asre_rating",
    "safe_rolling_zscore",
    "soft_clamp_zscore",
    "validate_kalman_input",
    "detect_component_divergence",
    "compute_direct_weighted_score",
    "DipQualityMetrics",
    "compute_dip_quality_score",
    "get_dip_stage_tier_aware",
    "generate_market_context_message",
    # FIX B2
    "check_and_warn_freeze",
    "recover_stale_cache",
]
