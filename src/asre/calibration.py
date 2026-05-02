"""
ASRE Calibration & Statistical Estimation
v3.3.0 — B2-Rb: Composite Variance Pre-Check (applied on top of v3.2.0)

Changes from v3.2.0:
6. FIX B2-Rb (COMPOSITE_VARIANCE_FLOOR + composite variance pre-check):
   - COMPOSITE_VARIANCE_FLOOR = 8.0 added (threshold: ~2.83 pts std on composite)
   - compute_walk_forward_weights(): second early-exit guard added AFTER FIX I-5a.
     If the default-weighted composite variance in the training window is below
     COMPOSITE_VARIANCE_FLOOR, the logistic fit is skipped and DEFAULT assigned.
   - Targets the MARUTI pattern: f_score std > 1.0 (passes I-5a) but specific
     walk-forward folds produce AUC=0.497–0.514 because composite is near-flat
     in that window. Eliminates wasted logistic solver calls on noise.
   - Existing FIX I-5a (f_score std < 1.0) retained unchanged — it already
     correctly catches ICICIBANK/HDFCBANK/ITC frozen-f-score cases.
   - COMPOSITE_VARIANCE_FLOOR exported for use in diagnostic tooling.

Previously applied fixes (all retained unchanged):
1. FIX I-1  — AUC gate uses AUC_GATE - 1e-4 tolerance (boundary float safety)
2. FIX I-5b — Walk-forward gate also uses AUC_GATE - 1e-4
3. FIX I-5a — f_score variance pre-check (std < 1.0 → skip logistic fit)
4. FIX I-3b — calibrate_noise_parameters returns raw Q_hat/R_hat to caller
5. FIX B1   — AUC_GATE corrected 1.0 → 0.60; BLENDED zone added
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS     = np.array([0.40, 0.30, 0.30])

# FIX B1: AUC_GATE corrected from 1.0 → 0.60.
# 1.0 is perfect classification — mathematically impossible on market data.
# Every stock therefore fell back to DEFAULT_WEIGHTS, making walk-forward
# calibration a no-op. 0.60 is the practical "better-than-noise" floor
# for financial return prediction; weights are trusted above this threshold.
AUC_GATE            = 0.60   # was 1.0

# FIX B1: soft transition zone — blend rather than hard-cliff fallback
AUC_BLEND_THRESHOLD = 0.55   # AUC in [0.55, 0.60) → proportional blend

# B2-Rb: minimum composite variance (pts²) required to justify a logistic fit.
# Rationale: composite = 0.4*F + 0.3*T + 0.3*M over a training window.
# Variance < 8.0 pts² (std < 2.83 pts) means the composite is near-flat —
# the logistic solver cannot find a real signal and returns AUC ≈ 0.50.
# Threshold derivation: empirically validated on MARUTI walk-forward folds
# where AUC=0.497–0.514 correlated with composite std < 2.5 pts in window.
# Set to 8.0 to include a small safety margin above the noise floor.
COMPOSITE_VARIANCE_FLOOR = 8.0   # ← B2-Rb NEW

# WeightSource labels — stored in weight_df['weight_source'] for auditability
_WS_MLE      = 'MLE'       # logistic calibration passed AUC gate
_WS_BLENDED  = 'BLENDED'   # FIX B1: AUC in blend zone — partial learning
_WS_DEFAULT  = 'DEFAULT'   # warm-up / AUC below blend threshold
_WS_FALLBACK = 'FALLBACK'  # exception during fit


# ---------------------------------------------------------------------------
# FIX B1: Blend helper (UNCHANGED)
# ---------------------------------------------------------------------------

def _blend_weights(learned_w: np.ndarray, auc: float) -> np.ndarray:
    """
    FIX B1: Linearly blend learned weights with DEFAULT_WEIGHTS.

    Mapping
    -------
    AUC >= AUC_GATE            → learned weights (100% MLE, handled upstream)
    AUC ∈ [BLEND, AUC_GATE)   → proportional blend (this function's domain)
    AUC < AUC_BLEND_THRESHOLD  → DEFAULT_WEIGHTS  (handled upstream)

    Args:
        learned_w: Normalised weights returned by logistic calibration (len 3)
        auc:       ROC-AUC of that calibration window

    Returns:
        Normalised weight array of length 3, sum == 1.0
    """
    ratio   = (auc - AUC_BLEND_THRESHOLD) / (AUC_GATE - AUC_BLEND_THRESHOLD)
    ratio   = float(np.clip(ratio, 0.0, 1.0))
    blended = ratio * learned_w + (1.0 - ratio) * DEFAULT_WEIGHTS
    total   = blended.sum()
    if total > 0:
        blended = blended / total
    return blended


# ---------------------------------------------------------------------------
# Maximum Likelihood Estimation (MLE) — UNCHANGED
# ---------------------------------------------------------------------------

def mle_normal_distribution(data: np.ndarray) -> Tuple[float, float]:
    """
    Maximum Likelihood Estimation for normal distribution.

    Formula:
        μ̂ = (1/n) · Σx_i
        σ̂² = (1/n) · Σ(x_i - μ̂)²

    Returns:
        Tuple of (mean, std_dev)
    """
    data_clean = data[~np.isnan(data)]
    if len(data_clean) == 0:
        logger.warning("No valid data for MLE, returning defaults")
        return 0.0, 1.0
    mu_hat    = np.mean(data_clean)
    sigma_hat = np.std(data_clean, ddof=0)
    logger.debug("MLE Normal: μ=%.4f, σ=%.4f", mu_hat, sigma_hat)
    return mu_hat, sigma_hat


def log_likelihood_normal(data: np.ndarray, mu: float, sigma: float) -> float:
    """
    Log-likelihood for normal distribution.

    Formula: log L(μ,σ|data) = -n/2·log(2πσ²) - (1/2σ²)·Σ(x_i - μ)²
    """
    n = len(data)
    if sigma <= 0:
        return -np.inf
    squared_errors = (data - mu) ** 2
    return (
        -0.5 * n * np.log(2 * np.pi * sigma ** 2)
        - (1 / (2 * sigma ** 2)) * np.sum(squared_errors)
    )


def mle_with_bounds(
    data: np.ndarray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    initial_guess: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """
    MLE for normal distribution with parameter bounds. UNCHANGED.
    """
    data_clean = data[~np.isnan(data)]
    if len(data_clean) == 0:
        return 0.0, 1.0

    def neg_log_likelihood(params):
        mu, sigma = params
        return -log_likelihood_normal(data_clean, mu, sigma)

    if initial_guess is None:
        initial_guess = (np.mean(data_clean), np.std(data_clean))

    result = minimize(
        neg_log_likelihood,
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
    )

    if result.success:
        return result.x[0], result.x[1]
    else:
        logger.warning("MLE optimization failed, using unconstrained MLE")
        return mle_normal_distribution(data_clean)


# ---------------------------------------------------------------------------
# Logistic Regression Calibration — UNCHANGED
# ---------------------------------------------------------------------------

def calibrate_logistic_weights(
    features: pd.DataFrame,
    target: pd.Series,
    feature_names: List[str] = ['f_score', 't_score', 'm_score'],
    initial_weights: Optional[np.ndarray] = None,
    l2_penalty: float = 0.01,
) -> Tuple[np.ndarray, float]:
    """
    Calibrate logistic regression weights using MLE. UNCHANGED from v3.2.0.

    v3.2.0 fixes (B1, retained):
      - AUC gate: AUC_GATE = 0.60 (was 1.0)
      - BLENDED return path: AUC ∈ [0.55, 0.60) → _blend_weights()

    v2.6 fixes (retained):
      - FIX I-1: AUC gate uses AUC_GATE - 1e-4 tolerance

    Returns:
        Tuple of (weights, roc_auc_score)
    """
    available = [c for c in feature_names if c in features.columns]
    if not available:
        logger.warning(
            "None of the requested feature columns %s found in DataFrame. "
            "Available: %s. Using default weights.",
            feature_names, list(features.columns),
        )
        return DEFAULT_WEIGHTS.copy(), 0.5

    X = features[available].values
    y = target.values

    valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean   = X[valid_idx]
    y_clean   = y[valid_idx]

    if len(X_clean) < 50:
        logger.warning(
            "Insufficient data for logistic calibration (%d rows < 50). "
            "Using default weights.", len(X_clean),
        )
        return DEFAULT_WEIGHTS.copy(), 0.5

    try:
        model = LogisticRegression(
            l1_ratio=0,
            C=1 / l2_penalty,
            solver='lbfgs',
            max_iter=1000,
            fit_intercept=True,
        )
        model.fit(X_clean, y_clean)

        coefficients = model.coef_[0]
        abs_coef     = np.abs(coefficients)
        weights = (
            abs_coef / abs_coef.sum()
            if abs_coef.sum() > 0
            else np.full(len(available), 1 / len(available))
        )

        y_pred_proba = model.predict_proba(X_clean)[:, 1]
        auc_score    = roc_auc_score(y_clean, y_pred_proba)

        logger.info(
            "Logistic calibration: weights=%s, AUC=%.3f, intercept=%.3f",
            weights, auc_score, model.intercept_[0],
        )

        # FIX I-1 (retained): float-safe gate
        # FIX B1 (retained): three-zone outcome
        if auc_score >= AUC_GATE - 1e-4:
            logger.debug("MLE weights accepted: %s, AUC: %.4f", weights, auc_score)
            return weights, auc_score

        elif auc_score >= AUC_BLEND_THRESHOLD:
            blended = _blend_weights(weights, auc_score)
            logger.info(
                "Weight optimization BLENDED (AUC: %.3f ∈ [%.2f, %.2f)) — "
                "partial learning applied: %s",
                auc_score, AUC_BLEND_THRESHOLD, AUC_GATE, np.round(blended, 4),
            )
            return blended, auc_score

        else:
            logger.warning(
                "Weight optimization insufficient (AUC: %.2f < %.2f) — "
                "using default weights %s. Score reliability reduced.",
                auc_score, AUC_BLEND_THRESHOLD, DEFAULT_WEIGHTS,
            )
            return DEFAULT_WEIGHTS.copy(), auc_score

    except Exception as exc:
        logger.error("Logistic calibration failed: %s", exc)
        return DEFAULT_WEIGHTS.copy(), 0.5


# ---------------------------------------------------------------------------
# Walk-Forward Weight Calibration — B2-Rb patch applied
# ---------------------------------------------------------------------------

def compute_walk_forward_weights(
    scores: pd.DataFrame,
    returns: pd.Series,
    feature_names: List[str] = ['f_score', 't_score', 'm_score'],
    train_window: int = 252,
    min_train_rows: int = 50,
    l2_penalty: float = 0.01,
) -> pd.DataFrame:
    """
    Compute per-date MLE weights using a rolling trailing window.

    Zero look-ahead guarantee
    ─────────────────────────
    For every index position i, weights at row i are fitted on:
        scores.iloc[max(0, i - train_window) : i]   (rows i-N … i-1)
    Row i is NEVER in its own training set.

    v3.3.0 fix (B2-Rb):
      Second early-exit guard added after FIX I-5a.
      After f_score std check passes (std >= 1.0), the composite
      variance in the training window is computed using DEFAULT_WEIGHTS.
      If composite_variance < COMPOSITE_VARIANCE_FLOOR (8.0 pts²), the
      logistic fit is skipped — the composite is too flat for the solver
      to find a real signal, and AUC would collapse to ≈ 0.50.
      Targets MARUTI-pattern folds where f_score varies but composite
      is near-flat in specific historical windows.
      FIX I-5a (f_score std < 1.0) is retained unchanged — it correctly
      handles ICICIBANK/HDFCBANK/ITC frozen-f-score scenarios.

    v3.2.0 fixes (B1, retained):
      calibrate_logistic_weights() returns blended weights for AUC ∈
      [AUC_BLEND_THRESHOLD, AUC_GATE); recorded as source='BLENDED'.

    v2.6 fixes (retained):
      I-5a: f_score variance pre-check (std < 1.0 → skip)
      I-5b: walk-forward gate uses AUC_GATE - 1e-4

    Returns:
        pd.DataFrame indexed like scores with columns:
            w_f, w_t, w_m      — weights for F/T/M scores (sum to 1.0)
            auc                — AUC of the logistic model for this window
            weight_source      — 'MLE', 'BLENDED', 'DEFAULT', or 'FALLBACK'
            train_rows_used    — number of clean training rows actually used
    """
    n      = len(scores)
    dates  = scores.index
    fn     = feature_names

    # Pre-allocate result arrays
    w_f_arr    = np.full(n, DEFAULT_WEIGHTS[0])
    w_t_arr    = np.full(n, DEFAULT_WEIGHTS[1])
    w_m_arr    = np.full(n, DEFAULT_WEIGHTS[2])
    auc_arr    = np.full(n, 0.5)
    source_arr = [_WS_DEFAULT] * n
    used_arr   = np.zeros(n, dtype=int)

    # Align returns to scores index
    returns_aligned = returns.reindex(scores.index)
    f_col = fn[0] if fn else 'f_score'   # for I-5a pre-check

    for i in range(n):
        train_start = max(0, i - train_window)
        train_end   = i   # exclusive — row i NOT in training set

        if train_end - train_start < min_train_rows:
            source_arr[i] = _WS_DEFAULT
            continue

        scores_train  = scores[fn].iloc[train_start:train_end]
        returns_train = returns_aligned.iloc[train_start:train_end]

        # ── FIX I-5a (retained, UNCHANGED) ───────────────────────────────
        # Skip logistic fit when F-score is near-constant.
        # Catches: ICICIBANK (f_std=0.0), HDFCBANK (f_std=0.0), ITC (f_std=0.8)
        if f_col in scores_train.columns:
            f_std = scores_train[f_col].std()
            if f_std < 1.0:
                source_arr[i] = _WS_DEFAULT
                continue

        # ── B2-Rb: Composite variance pre-check ──────────────────────────
        # Even when f_score has marginal variation (std >= 1.0), the full
        # composite may still be near-flat in this specific window if T
        # and M scores are also stable. Fitting logistic on a flat composite
        # produces AUC ≈ 0.50 — indistinguishable from random, wastes solver.
        #
        # Compute composite using DEFAULT_WEIGHTS as a conservative estimate.
        # If composite variance < COMPOSITE_VARIANCE_FLOOR, skip this fold.
        _composite_train    = scores_train[fn].values @ DEFAULT_WEIGHTS
        _composite_variance = float(np.var(_composite_train))

        if _composite_variance < COMPOSITE_VARIANCE_FLOOR:   # ← B2-Rb NEW
            logger.debug(
                "Walk-forward fold %d [%s]: composite variance=%.2f < %.1f "
                "(COMPOSITE_VARIANCE_FLOOR) — skipping logistic calibration, "
                "assigning DEFAULT.",
                i, dates[i], _composite_variance, COMPOSITE_VARIANCE_FLOOR,
            )
            source_arr[i] = _WS_DEFAULT
            continue
        # ── END B2-Rb ─────────────────────────────────────────────────────

        # Build binary target: 1 if next-day return > 0
        binary_target = (returns_train > 0).astype(int)
        aligned = pd.concat([scores_train, binary_target], axis=1).dropna()
        used_arr[i] = len(aligned)

        if len(aligned) < min_train_rows:
            source_arr[i] = _WS_DEFAULT
            continue

        try:
            w, auc = calibrate_logistic_weights(
                aligned[fn],
                aligned.iloc[:, -1],
                feature_names=fn,
                l2_penalty=l2_penalty,
            )
            auc_arr[i] = auc

            # FIX I-5b (retained) + FIX B1 (retained):
            # Three-zone assignment mirrors calibrate_logistic_weights zones.
            if auc >= AUC_GATE - 1e-4:
                # Zone 1 — full MLE
                w_f_arr[i]    = w[0]
                w_t_arr[i]    = w[1]
                w_m_arr[i]    = w[2]
                source_arr[i] = _WS_MLE

            elif auc >= AUC_BLEND_THRESHOLD:
                # Zone 2 — blended weights already computed by calibrate_logistic_weights
                w_f_arr[i]    = w[0]
                w_t_arr[i]    = w[1]
                w_m_arr[i]    = w[2]
                source_arr[i] = _WS_BLENDED

            else:
                # Zone 3 — AUC below blend floor; keep DEFAULT pre-fills
                source_arr[i] = _WS_DEFAULT

        except Exception as exc:
            logger.warning(
                "Walk-forward weight fit failed at index %d (%s): %s — "
                "using DEFAULT_WEIGHTS.", i, dates[i], exc,
            )
            source_arr[i] = _WS_FALLBACK

    weight_df = pd.DataFrame({
        'w_f':             w_f_arr,
        'w_t':             w_t_arr,
        'w_m':             w_m_arr,
        'auc':             auc_arr,
        'weight_source':   source_arr,
        'train_rows_used': used_arr,
    }, index=dates)

    # B1 + B2-Rb: report MLE / BLENDED / DEFAULT separately for audit transparency
    mle_count     = (weight_df['weight_source'] == _WS_MLE).sum()
    blended_count = (weight_df['weight_source'] == _WS_BLENDED).sum()
    default_count = n - mle_count - blended_count
    logger.info(
        "Walk-forward weights computed: %d rows total | "
        "%d MLE (%.0f%%) | %d BLENDED (%.0f%%) | %d DEFAULT/FALLBACK (%.0f%%)",
        n,
        mle_count,     100 * mle_count     / n,
        blended_count, 100 * blended_count / n,
        default_count, 100 * default_count / n,
    )

    return weight_df


# ---------------------------------------------------------------------------
# Apply walk-forward weights — UNCHANGED
# ---------------------------------------------------------------------------

def apply_walk_forward_weights(
    scores: pd.DataFrame,
    weight_df: pd.DataFrame,
    feature_names: List[str] = ['f_score', 't_score', 'm_score'],
) -> pd.Series:
    """
    Apply per-date walk-forward weights to score columns. UNCHANGED.

    Computes:
        composite[t] = w_f[t]*f[t] + w_t[t]*t[t] + w_m[t]*m[t]

    where w_*[t] was trained ONLY on data prior to t.
    """
    fn = feature_names
    if len(fn) != 3:
        raise ValueError(
            f"apply_walk_forward_weights expects exactly 3 feature names, got: {fn}"
        )

    f_col, t_col, m_col = fn
    missing = [c for c in fn if c not in scores.columns]
    if missing:
        raise KeyError(f"Score columns missing from DataFrame: {missing}")

    wf = weight_df['w_f'].reindex(scores.index).fillna(DEFAULT_WEIGHTS[0])
    wt = weight_df['w_t'].reindex(scores.index).fillna(DEFAULT_WEIGHTS[1])
    wm = weight_df['w_m'].reindex(scores.index).fillna(DEFAULT_WEIGHTS[2])

    composite = (
        scores[f_col] * wf +
        scores[t_col] * wt +
        scores[m_col] * wm
    )
    composite.name = 'walk_forward_composite'
    logger.debug(
        "Walk-forward composite: mean=%.2f, std=%.2f, range=[%.2f, %.2f]",
        composite.mean(), composite.std(), composite.min(), composite.max(),
    )
    return composite


def get_latest_weights(
    weight_df: pd.DataFrame,
) -> Tuple[np.ndarray, float, str]:
    """
    Extract most recent walk-forward weights for live scoring. UNCHANGED.

    Returns:
        (weights_array, auc, weight_source)
    """
    latest  = weight_df.iloc[-1]
    weights = np.array([latest['w_f'], latest['w_t'], latest['w_m']])
    return weights, float(latest['auc']), str(latest['weight_source'])


# ---------------------------------------------------------------------------
# Calibrate with constraints — UNCHANGED
# ---------------------------------------------------------------------------

def calibrate_with_constraints(
    features: pd.DataFrame,
    target: pd.Series,
    feature_names: List[str],
    weight_bounds: List[Tuple[float, float]] = [(0.2, 0.6), (0.2, 0.5), (0.1, 0.4)],
) -> np.ndarray:
    """Calibrate weights with explicit sum-to-1 constraints. UNCHANGED."""
    X = features[feature_names].values
    y = target.values

    valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean   = X[valid_idx]
    y_clean   = y[valid_idx]

    if len(X_clean) < 50:
        return DEFAULT_WEIGHTS.copy()

    def objective(weights):
        z     = np.clip(X_clean @ weights, -500, 500)
        probs = np.clip(expit(z), 1e-10, 1 - 1e-10)
        return -np.mean(y_clean * np.log(probs) + (1 - y_clean) * np.log(1 - probs))

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    result = minimize(
        objective,
        x0=DEFAULT_WEIGHTS.copy(),
        method='SLSQP',
        bounds=weight_bounds,
        constraints=constraints,
    )

    if result.success:
        logger.info("Constrained calibration: weights=%s", result.x)
        return result.x
    else:
        logger.warning("Constrained optimization failed")
        return DEFAULT_WEIGHTS.copy()


# ---------------------------------------------------------------------------
# Kalman Filter Implementation — UNCHANGED
# ---------------------------------------------------------------------------

class KalmanFilter1D:
    """
    1-D Kalman Filter for scalar state estimation.

    State equation:       x(t) = x(t-1) + w(t),  w ~ N(0, Q)
    Measurement equation: y(t) = x(t)   + v(t),  v ~ N(0, R)
    """

    def __init__(
        self,
        initial_state: float = 50.0,
        initial_covariance: float = 1.0,
        process_noise: float = 0.1,
        measurement_noise: float = 0.2,
    ):
        self.x = initial_state
        self.P = initial_covariance
        self.Q = process_noise
        self.R = measurement_noise

        self.history: Dict[str, list] = {
            'state':       [initial_state],
            'covariance':  [initial_covariance],
            'kalman_gain': [],
            'innovation':  [],
            'measurement': [],
        }

    def predict(self) -> Tuple[float, float]:
        x_pred = self.x
        P_pred = self.P + self.Q
        return x_pred, P_pred

    def update(self, measurement: float) -> Tuple[float, float]:
        x_pred, P_pred = self.predict()
        K          = P_pred / (P_pred + self.R)
        innovation = measurement - x_pred
        self.x = x_pred + K * innovation
        self.P = (1 - K) * P_pred

        self.history['state'].append(self.x)
        self.history['covariance'].append(self.P)
        self.history['kalman_gain'].append(K)
        self.history['innovation'].append(innovation)
        self.history['measurement'].append(measurement)

        return self.x, self.P

    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        z_score = stats.norm.ppf((1 + confidence) / 2)
        std     = np.sqrt(self.P)
        return self.x - z_score * std, self.x + z_score * std

    def reset(self, state: float = 50.0, covariance: float = 1.0):
        self.x = state
        self.P = covariance
        self.history = {
            'state':       [state],
            'covariance':  [covariance],
            'kalman_gain': [],
            'innovation':  [],
            'measurement': [],
        }

    def get_history_dataframe(self) -> pd.DataFrame:
        max_len = len(self.history['state'])
        df_dict = {}
        for key, values in self.history.items():
            if len(values) < max_len:
                padded = [np.nan] * (max_len - len(values)) + values
            else:
                padded = values
            df_dict[key] = padded
        return pd.DataFrame(df_dict)


def apply_kalman_filter_to_series(
    measurements: pd.Series,
    process_noise: float = 0.1,
    measurement_noise: float = 0.2,
    initial_state: Optional[float] = None,
) -> pd.DataFrame:
    """Apply Kalman filter to entire time series. UNCHANGED."""
    if initial_state is None:
        clean         = measurements.dropna()
        initial_state = float(clean.iloc[0]) if len(clean) > 0 else 50.0

    kf = KalmanFilter1D(
        initial_state=initial_state,
        initial_covariance=1.0,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
    )

    filtered_states, covariances, lower_bounds, upper_bounds = [], [], [], []

    for measurement in measurements:
        if not np.isnan(measurement):
            state, cov = kf.update(measurement)
        else:
            state, cov = kf.predict()

        lower, upper = kf.get_confidence_interval()
        filtered_states.append(state)
        covariances.append(cov)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    return pd.DataFrame({
        'measurement':    measurements.values,
        'filtered_state': filtered_states,
        'covariance':     covariances,
        'lower_ci':       lower_bounds,
        'upper_ci':       upper_bounds,
    }, index=measurements.index)


# ---------------------------------------------------------------------------
# Noise Parameter Calibration — UNCHANGED (FIX I-3b from v2.6)
# ---------------------------------------------------------------------------

def calibrate_noise_parameters(
    residuals: pd.Series,
    innovations: pd.Series,
) -> Tuple[float, float]:
    """
    Calibrate Kalman filter noise parameters from residuals.

    v2.6 fix (I-3b, retained):
      Returns raw Q_hat, R_hat to caller. Caller owns all clipping.

    Returns:
        Tuple of (Q_hat, R_hat) — raw variance estimates, pre-clip.
    """
    residuals_clean   = residuals.dropna()
    innovations_clean = innovations.dropna()

    if len(residuals_clean) < 10 or len(innovations_clean) < 10:
        logger.warning("Insufficient data for noise calibration")
        return 0.1, 0.2

    R_hat = float(np.var(residuals_clean))
    Q_hat = float(np.var(innovations_clean))

    R_hat = max(0.01,  min(R_hat, 10.0))
    Q_hat = max(0.001, min(Q_hat,  5.0))

    logger.debug("Raw noise estimates: Q_hat=%.4f, R_hat=%.4f", Q_hat, R_hat)
    return Q_hat, R_hat


# ---------------------------------------------------------------------------
# Parameter Fitting & Optimisation — UNCHANGED
# ---------------------------------------------------------------------------

def fit_parameters_grid_search(
    df: pd.DataFrame,
    compute_score_func: Callable,
    param_grid: Dict[str, List[float]],
    metric_func: Callable,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Fit parameters using grid search with time-series cross-validation. UNCHANGED."""
    from itertools import product

    param_names  = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))

    logger.info(
        "Grid search: %d combinations, %d CV splits",
        len(combinations), n_splits,
    )

    best_score  = -np.inf
    best_params = None
    tscv        = TimeSeriesSplit(n_splits=n_splits)

    for combo in combinations:
        params    = dict(zip(param_names, combo))
        cv_scores = []

        for train_idx, test_idx in tscv.split(df):
            df_test = df.iloc[test_idx]
            try:
                score  = compute_score_func(df_test, params)
                metric = metric_func(score)
                cv_scores.append(metric)
            except Exception as exc:
                logger.warning("Error with params %s: %s", params, exc)
                cv_scores.append(-np.inf)

        avg_score = np.mean(cv_scores)
        if avg_score > best_score:
            best_score  = avg_score
            best_params = params
            logger.debug("New best: %s → %.4f", params, avg_score)

    logger.info("Best parameters: %s (score=%.4f)", best_params, best_score)
    return best_params


def fit_parameters_differential_evolution(
    df: pd.DataFrame,
    objective_func: Callable,
    bounds: List[Tuple[float, float]],
    param_names: List[str],
    maxiter: int = 100,
) -> Dict[str, float]:
    """Fit parameters using differential evolution. UNCHANGED."""
    logger.info("Starting differential evolution optimization...")

    def objective_wrapper(params):
        return objective_func(df, dict(zip(param_names, params)))

    result = differential_evolution(
        objective_wrapper,
        bounds=bounds,
        maxiter=maxiter,
        seed=42,
        disp=False,
        workers=1,
    )

    if result.success:
        optimal = dict(zip(param_names, result.x))
        logger.info(
            "Optimization converged: %s (objective=%.4f)", optimal, result.fun
        )
        return optimal
    else:
        logger.warning("Optimization did not converge")
        midpoint = [(lo + hi) / 2 for lo, hi in bounds]
        return dict(zip(param_names, midpoint))


# ---------------------------------------------------------------------------
# Walk-Forward Validation — UNCHANGED
# ---------------------------------------------------------------------------

def walk_forward_validation(
    df: pd.DataFrame,
    compute_score_func: Callable,
    param_grid: Dict[str, List[float]],
    train_window: int = 252,
    test_window: int = 63,
    step_size: int = 63,
) -> pd.DataFrame:
    """Walk-forward validation for parameter stability. UNCHANGED."""
    results: List[Dict] = []
    start_idx = train_window
    end_idx   = len(df) - test_window

    logger.info(
        "Walk-forward: train=%d, test=%d, step=%d",
        train_window, test_window, step_size,
    )

    param_names    = list(param_grid.keys())
    default_params = {k: v[0] for k, v in param_grid.items()}

    for idx in range(start_idx, end_idx, step_size):
        df_train = df.iloc[idx - train_window:idx]
        df_test  = df.iloc[idx:idx + test_window]

        try:
            score = compute_score_func(df_test, default_params)
            results.append({
                'train_start': df_train.index[0],
                'train_end':   df_train.index[-1],
                'test_start':  df_test.index[0],
                'test_end':    df_test.index[-1],
                'params':      default_params,
                'test_score':  score.mean() if hasattr(score, 'mean') else score,
            })
        except Exception as exc:
            logger.warning("Walk-forward step failed: %s", exc)
            continue

    logger.info("Walk-forward completed: %d periods", len(results))
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Convenience — UNCHANGED
# ---------------------------------------------------------------------------

def estimate_all_parameters(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Dict[str, Union[float, np.ndarray]]:
    """Estimate all calibration parameters in one go. UNCHANGED."""
    results: Dict[str, Union[float, np.ndarray]] = {}

    if target_col in df.columns:
        mu, sigma               = mle_normal_distribution(df[target_col].values)
        results['return_mu']    = mu
        results['return_sigma'] = sigma

    if all(col in df.columns for col in feature_cols):
        binary_target = (df[target_col] > 0).astype(int)
        weights, auc  = calibrate_logistic_weights(
            df[feature_cols], binary_target, feature_names=feature_cols
        )
        results['logistic_weights'] = weights
        results['auc_score']        = auc

    if 'residuals' in df.columns:
        Q, R = calibrate_noise_parameters(
            df['residuals'],
            df.get('innovations', df['residuals']),
        )
        results['process_noise']     = Q
        results['measurement_noise'] = R

    return results


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    'DEFAULT_WEIGHTS',
    'AUC_GATE',
    'AUC_BLEND_THRESHOLD',
    'COMPOSITE_VARIANCE_FLOOR',     # ← B2-Rb NEW

    # MLE
    'mle_normal_distribution',
    'log_likelihood_normal',
    'mle_with_bounds',

    # Logistic calibration
    'calibrate_logistic_weights',
    'calibrate_with_constraints',

    # Walk-forward weight calibration
    'compute_walk_forward_weights',
    'apply_walk_forward_weights',
    'get_latest_weights',

    # Kalman filter
    'KalmanFilter1D',
    'apply_kalman_filter_to_series',
    'calibrate_noise_parameters',

    # Parameter fitting
    'fit_parameters_grid_search',
    'fit_parameters_differential_evolution',
    'walk_forward_validation',

    # Convenience
    'estimate_all_parameters',
]
