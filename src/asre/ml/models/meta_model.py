"""
asre/ml/models/meta_model.py — ASRE Second-Stage Meta-Model (v1.0)

Purpose
-------
Builds a second-stage meta-model on top of the F, T, and M factor scores
plus optional contextual features (regime, volatility bucket, sector, etc.).
The meta-model learns how to combine base factor scores into a calibrated
probability of outperformance, conditioned on market context.

Design decisions
----------------
1. **Interpretable by default**: Logistic regression is the only supported
   model type. Coefficient inspection is a first-class output. Tree-based
   or neural models are explicitly excluded — SEBI audit-readiness requires
   that every model coefficient can be explained in plain English.

2. **Walk-forward only**: This module does NOT perform any train/test
   splitting. The caller (walk_forward_engine.py) is responsible for
   supplying temporally ordered folds. This prevents look-ahead bias.

3. **Safe degradation**: If the training set has fewer than MIN_TRAIN_SAMPLES
   (100) observations, the module returns a fallback result with is_fallback=True
   and predict_meta_proba returns 0.5 (maximum uncertainty) for all inputs.
   This ensures the composite score reverts to the theoretical prior rather
   than trusting a noisy model.

4. **Preprocessing is explicit**: build_meta_features() handles imputation,
   encoding, and standardisation in a single auditable pipeline. No hidden
   transforms occur inside train_meta_model().

Integration points
------------------
  walk_forward_engine.py:
      X, feature_names = build_meta_features(fold_df)
      result = train_meta_model(X, y, feature_names, config)
      probas = predict_meta_proba(result, X_test)

  calibrator.py:
      Raw probas from predict_meta_proba are passed to fit_calibrator()
      for post-hoc probability calibration.

  calibration_report.py:
      calibrated probas are evaluated via summarize_calibration().

References
----------
Hastie, T., Tibshirani, R., Friedman, J. (2009). The Elements of Statistical
  Learning. Springer. Chapter 4: Linear Methods for Classification.
Fama, E., French, K. (2015). A five-factor asset pricing model.
  Journal of Financial Economics, 116(1), 1-22.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic memoization (pure-function caches)
# ---------------------------------------------------------------------------
# build_meta_features / train_meta_model / transform_meta_features are pure
# functions of their inputs. composite.py calls them inside a per-row
# walk-forward loop (~571 calls per compute), which dominates rating latency
# (~33s). Caching by a SHA-1 hash of the inputs lets a *repeat* compute of the
# same ticker (e.g. the dip / backtest endpoints, or warm-cache requests) skip
# the entire loop. Caching never changes outputs — a hit returns a copy of the
# exact value the impl would have produced; on hash failure the key is None and
# the impl simply runs.
#
# NOTE ON SIZE: unlike calibration._WFW_CACHE (one entry per compute), these are
# called once PER ROW, so a full ticker needs ~500-600 entries to stay warm. A
# 64-entry cap would thrash and yield no cross-run reuse, so the cap is sized to
# hold several tickers' worth of per-row calls. A lock guards the check-evict-
# insert sequence because the scan endpoints fan compute out across threads.
_META_CACHE_MAX = 4096
_META_CACHE_LOCK = threading.Lock()
_META_BUILD_CACHE: "Dict[str, Tuple[np.ndarray, List[str], StandardScaler]]" = {}
_META_TRAIN_CACHE: "Dict[str, MetaModelResult]" = {}
_META_TRANSFORM_CACHE: "Dict[str, np.ndarray]" = {}


def _hash_obj(obj) -> str:
    """Stable content hash for an ndarray / Series / DataFrame / None."""
    if obj is None:
        return "none"
    try:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            h = pd.util.hash_pandas_object(obj, index=True).values
            return hashlib.sha1(h.tobytes()).hexdigest()
        arr = np.ascontiguousarray(np.asarray(obj, dtype=np.float64))
        return hashlib.sha1(arr.tobytes()).hexdigest()
    except Exception:
        return "unhashable"


def _hash_scaler(scaler: Optional[StandardScaler]) -> str:
    """Hash a fitted StandardScaler by its learned parameters."""
    if scaler is None:
        return "none"
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    if mean is None or scale is None:
        return "unfitted"
    return _hash_obj(np.concatenate([np.ravel(mean), np.ravel(scale)]))


def _cache_get(cache: dict, key: Optional[str]):
    if key is None:
        return None
    with _META_CACHE_LOCK:
        return cache.get(key)


def _cache_put(cache: dict, key: Optional[str], value) -> None:
    if key is None:
        return
    with _META_CACHE_LOCK:
        if key in cache:
            return
        if len(cache) >= _META_CACHE_MAX:
            cache.pop(next(iter(cache)))  # FIFO eviction
        cache[key] = value

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_TRAIN_SAMPLES: int = 100       # Absolute minimum to attempt fitting
MIN_RECOMMENDED_SAMPLES: int = 252  # 1 trading year — preferred minimum

META_MODEL_VERSION: str = "1.0"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class MetaModelConfig:
    """Configuration for the second-stage meta-model.

    Attributes:
        model_type: Model algorithm. Only 'logistic' is supported.
        regularization: Inverse regularization strength (C parameter for
            LogisticRegression). Smaller values = stronger regularization.
        handle_imbalance: If True, use class_weight='balanced' to handle
            class imbalance in the target variable.
        max_iter: Maximum iterations for the logistic regression solver.
        random_state: Random seed for deterministic behaviour.
        numeric_features: List of numeric feature column names expected
            in the input DataFrame.
        categorical_features: List of categorical feature column names
            to be one-hot encoded.
    """

    model_type: str = "logistic"
    regularization: float = 1.0
    handle_imbalance: bool = False
    max_iter: int = 1000
    random_state: int = 42
    numeric_features: List[str] = field(
        default_factory=lambda: ["f_score", "t_score", "m_score"]
    )
    categorical_features: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MetaModelResult:
    """Complete result from meta-model training.

    Attributes:
        model: The fitted sklearn LogisticRegression model, or None if
            fallback was triggered.
        scaler: The fitted StandardScaler used for numeric features, or
            None if fallback.
        feature_names: Ordered list of feature names after encoding.
        coefficients: Mapping of feature name to logistic regression
            coefficient for interpretability audit.
        intercept: Logistic regression intercept term.
        n_train_samples: Number of samples used for training.
        n_features: Number of features after encoding.
        training_metrics: In-fold metrics dict with keys like 'accuracy',
            'auc' (if computable).
        warnings: List of non-fatal warning strings.
        is_fallback: True if model could not be fitted and the result
            should be treated as a no-information prior.
    """

    model: Any
    scaler: Any
    feature_names: List[str]
    coefficients: Dict[str, float]
    intercept: float
    n_train_samples: int
    n_features: int
    training_metrics: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    is_fallback: bool = False


# ---------------------------------------------------------------------------
# build_meta_features
# ---------------------------------------------------------------------------

def build_meta_features(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], StandardScaler]:
    """Memoized wrapper around :func:`_build_meta_features_impl`.

    Pure function of (df, numeric_cols, categorical_cols); see the module-level
    cache notes. Returns a copy of the cached matrix/names on hit so callers
    can never mutate cached state.
    """
    key = None
    try:
        key = hashlib.sha1(
            "|".join([
                _hash_obj(df),
                str(numeric_cols),
                str(categorical_cols),
            ]).encode()
        ).hexdigest()
    except Exception:
        key = None

    cached = _cache_get(_META_BUILD_CACHE, key)
    if cached is not None:
        matrix, names, scaler = cached
        return matrix.copy(), list(names), scaler

    result = _build_meta_features_impl(df, numeric_cols, categorical_cols)
    matrix, names, scaler = result
    _cache_put(_META_BUILD_CACHE, key, (matrix.copy(), list(names), scaler))
    return result


def _build_meta_features_impl(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], StandardScaler]:
    """Build a feature matrix from raw DataFrame columns.

    Performs median imputation on numeric columns, one-hot encoding on
    categorical columns, and StandardScaler standardisation on numerics.
    Returns a dense float64 matrix suitable for LogisticRegression.

    Args:
        df: Input DataFrame containing raw feature columns.
        numeric_cols: Numeric column names to extract. If None, defaults
            to ['f_score', 't_score', 'm_score'].
        categorical_cols: Categorical column names to one-hot encode.
            If None, defaults to empty list.

    Returns:
        Tuple of (feature_matrix, feature_name_list, fitted_scaler).
        The scaler is returned so it can be reapplied to test data.

    Raises:
        ValueError: If df is empty or no valid features remain after
            cleaning.
    """
    if df is None or len(df) == 0:
        raise ValueError("build_meta_features: input DataFrame is empty.")

    if numeric_cols is None:
        numeric_cols = ["f_score", "t_score", "m_score"]
    if categorical_cols is None:
        categorical_cols = []

    warnings_: List[str] = []
    feature_names: List[str] = []
    feature_arrays: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Numeric features: impute → standardise
    # ------------------------------------------------------------------
    valid_numeric_cols: List[str] = []
    for col in numeric_cols:
        if col not in df.columns:
            warnings_.append(
                f"Numeric column '{col}' not found in DataFrame, skipping."
            )
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        # Skip all-NaN columns
        if series.isna().all():
            warnings_.append(
                f"Numeric column '{col}' is entirely NaN, skipping."
            )
            continue
        # Skip constant columns (zero variance after imputation)
        if series.dropna().nunique() <= 1:
            warnings_.append(
                f"Numeric column '{col}' is constant, skipping."
            )
            continue
        valid_numeric_cols.append(col)

    if valid_numeric_cols:
        numeric_df = df[valid_numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        # Median imputation for missing values
        medians = numeric_df.median()
        numeric_df = numeric_df.fillna(medians)

        scaler = StandardScaler()
        numeric_arr = scaler.fit_transform(numeric_df.values)

        feature_arrays.append(numeric_arr)
        feature_names.extend(valid_numeric_cols)
    else:
        scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Categorical features: one-hot encode
    # ------------------------------------------------------------------
    for col in categorical_cols:
        if col not in df.columns:
            warnings_.append(
                f"Categorical column '{col}' not found in DataFrame, skipping."
            )
            continue
        categories = df[col].fillna("_MISSING_").astype(str)
        unique_vals = sorted(categories.unique())

        if len(unique_vals) <= 1:
            warnings_.append(
                f"Categorical column '{col}' has only one unique value, "
                "skipping."
            )
            continue

        # Drop-first encoding to avoid multicollinearity
        for val in unique_vals[1:]:
            encoded = (categories == val).astype(np.float64).values
            feature_arrays.append(encoded.reshape(-1, 1))
            feature_names.append(f"{col}_{val}")

    if len(feature_arrays) == 0:
        raise ValueError(
            "build_meta_features: no valid features remain after cleaning. "
            f"Warnings: {warnings_}"
        )

    if warnings_:
        for w in warnings_:
            logger.warning("build_meta_features: %s", w)

    feature_matrix = np.hstack(feature_arrays)
    logger.debug(
        "build_meta_features: built %d features from %d samples. "
        "Numeric: %d, Categorical encoded: %d.",
        len(feature_names),
        feature_matrix.shape[0],
        len(valid_numeric_cols),
        len(feature_names) - len(valid_numeric_cols),
    )
    return feature_matrix, feature_names, scaler


def transform_meta_features(
    df: pd.DataFrame,
    feature_names: List[str],
    scaler: StandardScaler,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """Memoized wrapper around :func:`_transform_meta_features_impl`.

    Pure function of (df, feature_names, scaler params, numeric/categorical
    cols). Returns a copy of the cached matrix on hit.
    """
    key = None
    try:
        key = hashlib.sha1(
            "|".join([
                _hash_obj(df),
                str(list(feature_names)),
                _hash_scaler(scaler),
                str(numeric_cols),
                str(categorical_cols),
            ]).encode()
        ).hexdigest()
    except Exception:
        key = None

    cached = _cache_get(_META_TRANSFORM_CACHE, key)
    if cached is not None:
        return cached.copy()

    result = _transform_meta_features_impl(
        df, feature_names, scaler, numeric_cols, categorical_cols
    )
    _cache_put(_META_TRANSFORM_CACHE, key, result.copy())
    return result


def _transform_meta_features_impl(
    df: pd.DataFrame,
    feature_names: List[str],
    scaler: StandardScaler,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """Transform new data using a previously fitted scaler and feature set.

    Mirrors the transformations in build_meta_features but applies the
    already-fitted scaler rather than fitting a new one. Used for
    out-of-sample prediction.

    Args:
        df: New DataFrame to transform.
        feature_names: Feature names from the training build_meta_features
            call, defining the expected column order.
        scaler: Fitted StandardScaler from the training call.
        numeric_cols: Numeric column names (must match training).
        categorical_cols: Categorical column names (must match training).

    Returns:
        Feature matrix with the same column order as training.

    Raises:
        ValueError: If df is empty or required columns are missing.
    """
    if df is None or len(df) == 0:
        raise ValueError("transform_meta_features: input DataFrame is empty.")

    if numeric_cols is None:
        numeric_cols = ["f_score", "t_score", "m_score"]
    if categorical_cols is None:
        categorical_cols = []

    feature_arrays: List[np.ndarray] = []

    # Numeric: impute with median, apply pre-fitted scaler
    valid_numeric_cols = [c for c in numeric_cols if c in df.columns]
    if valid_numeric_cols:
        numeric_df = df[valid_numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        medians = numeric_df.median()
        numeric_df = numeric_df.fillna(medians)
        numeric_arr = scaler.transform(numeric_df.values)
        feature_arrays.append(numeric_arr)

    # Categorical: one-hot encode to match training features
    for col in categorical_cols:
        if col not in df.columns:
            continue
        categories = df[col].fillna("_MISSING_").astype(str)
        unique_from_training = [
            fn.replace(f"{col}_", "", 1)
            for fn in feature_names
            if fn.startswith(f"{col}_")
        ]
        for val in unique_from_training:
            encoded = (categories == val).astype(np.float64).values
            feature_arrays.append(encoded.reshape(-1, 1))

    if len(feature_arrays) == 0:
        raise ValueError(
            "transform_meta_features: no valid features could be constructed."
        )

    return np.hstack(feature_arrays)


# ---------------------------------------------------------------------------
# train_meta_model
# ---------------------------------------------------------------------------

def _config_repr(config: Optional[MetaModelConfig]) -> str:
    """Stable string representation of a MetaModelConfig for cache keying."""
    c = config or MetaModelConfig()
    return "|".join([
        str(c.model_type), f"{float(c.regularization):.6g}",
        str(bool(c.handle_imbalance)), str(int(c.max_iter)),
        str(int(c.random_state)), str(list(c.numeric_features)),
        str(list(c.categorical_features)),
    ])


def train_meta_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    config: Optional[MetaModelConfig] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> MetaModelResult:
    """Memoized wrapper around :func:`_train_meta_model_impl`.

    Pure function of (X, y, feature_names, config, sample_weight). Returns a
    deep copy of the cached :class:`MetaModelResult` on hit (the fitted model is
    tiny — 3 features — so the copy is cheap) so callers never share state.
    """
    key = None
    try:
        key = hashlib.sha1(
            "|".join([
                _hash_obj(X),
                _hash_obj(y),
                str(list(feature_names)),
                _config_repr(config),
                _hash_obj(sample_weight),
            ]).encode()
        ).hexdigest()
    except Exception:
        key = None

    cached = _cache_get(_META_TRAIN_CACHE, key)
    if cached is not None:
        return copy.deepcopy(cached)

    result = _train_meta_model_impl(X, y, feature_names, config, sample_weight)
    _cache_put(_META_TRAIN_CACHE, key, copy.deepcopy(result))
    return result


def _train_meta_model_impl(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    config: Optional[MetaModelConfig] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> MetaModelResult:
    """Train a second-stage meta-model on pre-built feature matrix.

    Fits an L2-regularised logistic regression by default. Returns a
    MetaModelResult containing the fitted model, interpretable coefficients,
    and in-sample quality metrics.

    If the sample size is below MIN_TRAIN_SAMPLES, returns a fallback
    result with is_fallback=True. Downstream consumers should check this
    flag and revert to the theoretical weight prior.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Binary target array of shape (n_samples,). Values must be 0 or 1.
        feature_names: Ordered feature names matching X columns.
        config: MetaModelConfig. If None, uses default configuration.
        sample_weight: Optional per-sample weights for training.

    Returns:
        MetaModelResult with fitted model or fallback.

    Raises:
        ValueError: If X and y have incompatible shapes, or y is not binary.
    """
    if config is None:
        config = MetaModelConfig()

    warnings_: List[str] = []

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    if X.ndim != 2:
        raise ValueError(
            f"train_meta_model: X must be 2-D, got shape {X.shape}."
        )
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"train_meta_model: X has {X.shape[0]} samples but y has "
            f"{y.shape[0]}."
        )
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"train_meta_model: X has {X.shape[1]} features but "
            f"feature_names has {len(feature_names)} entries."
        )

    # Check y is binary
    unique_y = np.unique(y[np.isfinite(y)])
    if not np.array_equal(unique_y, np.array([0.0, 1.0])):
        if len(unique_y) == 1:
            warnings_.append(
                f"Target y has only one class ({unique_y[0]}). "
                "Cannot train a classifier — returning fallback."
            )
            return _build_fallback_result(
                feature_names=feature_names,
                n_samples=X.shape[0],
                warnings_=warnings_,
            )
        raise ValueError(
            f"train_meta_model: y must be binary (0/1), got unique values "
            f"{unique_y}."
        )

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()
        if sample_weight.shape[0] != X.shape[0]:
            raise ValueError(
                f"train_meta_model: sample_weight has {sample_weight.shape[0]} "
                f"entries but X has {X.shape[0]} samples."
            )

    n_samples = X.shape[0]

    # ------------------------------------------------------------------
    # Check for NaN/Inf in features
    # ------------------------------------------------------------------
    nan_mask = ~np.isfinite(X).all(axis=1) | ~np.isfinite(y)
    if nan_mask.any():
        n_dropped = int(nan_mask.sum())
        warnings_.append(
            f"Dropped {n_dropped} rows with NaN/Inf values from training data."
        )
        valid = ~nan_mask
        X = X[valid]
        y = y[valid]
        if sample_weight is not None:
            sample_weight = sample_weight[valid]
        n_samples = X.shape[0]

    # ------------------------------------------------------------------
    # Sample size gate
    # ------------------------------------------------------------------
    if n_samples < MIN_TRAIN_SAMPLES:
        warnings_.append(
            f"Insufficient training samples ({n_samples} < "
            f"{MIN_TRAIN_SAMPLES} minimum). Returning fallback model."
        )
        logger.debug(
            "train_meta_model: only %d samples (< %d minimum). "
            "Returning fallback (is_fallback=True).",
            n_samples,
            MIN_TRAIN_SAMPLES,
        )
        return _build_fallback_result(
            feature_names=feature_names,
            n_samples=n_samples,
            warnings_=warnings_,
        )

    if n_samples < MIN_RECOMMENDED_SAMPLES:
        warnings_.append(
            f"Training with {n_samples} samples (< {MIN_RECOMMENDED_SAMPLES} "
            f"recommended). Model coefficients may be unstable."
        )

    # ------------------------------------------------------------------
    # Model type validation
    # ------------------------------------------------------------------
    if config.model_type != "logistic":
        raise ValueError(
            f"train_meta_model: unsupported model_type '{config.model_type}'. "
            "Only 'logistic' is supported."
        )

    # ------------------------------------------------------------------
    # Fit logistic regression
    # ------------------------------------------------------------------
    class_weight = "balanced" if config.handle_imbalance else None

    lr = LogisticRegression(
        C=config.regularization,
        class_weight=class_weight,
        max_iter=config.max_iter,
        random_state=config.random_state,
        solver="lbfgs",
    )

    try:
        lr.fit(X, y, sample_weight=sample_weight)
    except Exception as exc:
        warnings_.append(
            f"LogisticRegression.fit() failed: {exc}. Returning fallback."
        )
        logger.error(
            "train_meta_model: fit failed with %s. Returning fallback.",
            exc,
        )
        return _build_fallback_result(
            feature_names=feature_names,
            n_samples=n_samples,
            warnings_=warnings_,
        )

    # ------------------------------------------------------------------
    # Extract coefficients
    # ------------------------------------------------------------------
    coefs = lr.coef_.ravel()
    coefficients: Dict[str, float] = {
        name: float(coefs[i]) for i, name in enumerate(feature_names)
    }
    intercept = float(lr.intercept_[0])

    # ------------------------------------------------------------------
    # In-sample metrics
    # ------------------------------------------------------------------
    training_metrics: Dict[str, float] = {}
    y_pred = lr.predict(X)
    training_metrics["accuracy"] = float(accuracy_score(y, y_pred))

    try:
        y_proba = lr.predict_proba(X)[:, 1]
        training_metrics["auc"] = float(roc_auc_score(y, y_proba))
    except (ValueError, IndexError):
        warnings_.append(
            "Could not compute in-sample AUC (possibly single-class fold)."
        )
        training_metrics["auc"] = float("nan")

    # Class distribution
    n_positive = int(y.sum())
    n_negative = n_samples - n_positive
    training_metrics["prevalence"] = float(n_positive / n_samples)
    training_metrics["n_positive"] = float(n_positive)
    training_metrics["n_negative"] = float(n_negative)

    logger.debug(
        "train_meta_model: fitted on %d samples (%d pos / %d neg). "
        "AUC=%.4f, accuracy=%.4f. Features: %s.",
        n_samples,
        n_positive,
        n_negative,
        training_metrics.get("auc", float("nan")),
        training_metrics["accuracy"],
        feature_names,
    )

    # Log top coefficients for interpretability
    sorted_coefs = sorted(
        coefficients.items(), key=lambda kv: abs(kv[1]), reverse=True
    )
    for name, coef in sorted_coefs[:5]:
        logger.debug(
            "  coeff %-25s = %+.4f", name, coef
        )

    return MetaModelResult(
        model=lr,
        scaler=None,  # Scaler is managed externally via build_meta_features
        feature_names=list(feature_names),
        coefficients=coefficients,
        intercept=intercept,
        n_train_samples=n_samples,
        n_features=X.shape[1],
        training_metrics=training_metrics,
        warnings=warnings_,
        is_fallback=False,
    )


# ---------------------------------------------------------------------------
# predict_meta_proba
# ---------------------------------------------------------------------------

def predict_meta_proba(
    model_result: MetaModelResult,
    X: np.ndarray,
) -> np.ndarray:
    """Predict outperformance probability using a fitted meta-model.

    If the model_result is a fallback (is_fallback=True), returns an array
    of 0.5 for all samples — representing maximum uncertainty.

    Args:
        model_result: MetaModelResult from train_meta_model.
        X: Feature matrix of shape (n_samples, n_features).

    Returns:
        1-D array of predicted probabilities in [0, 1].

    Raises:
        ValueError: If X shape does not match the model's expected features.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if model_result.is_fallback:
        logger.info(
            "predict_meta_proba: model is fallback — returning 0.5 for "
            "%d samples.",
            X.shape[0],
        )
        return np.full(X.shape[0], 0.5, dtype=np.float64)

    if X.shape[1] != model_result.n_features:
        raise ValueError(
            f"predict_meta_proba: X has {X.shape[1]} features but model "
            f"expects {model_result.n_features}."
        )

    # Handle NaN/Inf in prediction input by replacing with 0 (standardised)
    nan_count = int((~np.isfinite(X)).sum())
    if nan_count > 0:
        logger.warning(
            "predict_meta_proba: %d NaN/Inf values in input — replacing "
            "with 0.0 (standardised mean).",
            nan_count,
        )
        X = np.where(np.isfinite(X), X, 0.0)

    probas = model_result.model.predict_proba(X)[:, 1]
    return probas.astype(np.float64)


# ---------------------------------------------------------------------------
# extract_model_metadata
# ---------------------------------------------------------------------------

def extract_model_metadata(model_result: MetaModelResult) -> Dict[str, Any]:
    """Extract serializable metadata from a MetaModelResult for audit logging.

    Returns a plain dict (JSON-safe) containing all model parameters,
    coefficients, training metrics, and warnings. This is written to the
    JSONL decision log by the walk-forward engine.

    Args:
        model_result: MetaModelResult to extract metadata from.

    Returns:
        Dict with keys: version, model_type, is_fallback, feature_names,
        coefficients, intercept, n_train_samples, n_features,
        training_metrics, warnings.
    """
    return {
        "version": META_MODEL_VERSION,
        "model_type": "logistic" if not model_result.is_fallback else "fallback",
        "is_fallback": model_result.is_fallback,
        "feature_names": list(model_result.feature_names),
        "coefficients": {
            k: round(v, 6) for k, v in model_result.coefficients.items()
        },
        "intercept": round(model_result.intercept, 6),
        "n_train_samples": model_result.n_train_samples,
        "n_features": model_result.n_features,
        "training_metrics": {
            k: round(v, 6) if np.isfinite(v) else None
            for k, v in model_result.training_metrics.items()
        },
        "warnings": list(model_result.warnings),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_fallback_result(
    feature_names: List[str],
    n_samples: int,
    warnings_: List[str],
) -> MetaModelResult:
    """Build a fallback MetaModelResult when training is not possible.

    The fallback result has is_fallback=True, zero coefficients, and
    predict_meta_proba will return 0.5 for all inputs.

    Args:
        feature_names: Feature name list from the caller.
        n_samples: Number of available training samples.
        warnings_: Accumulated warning strings.

    Returns:
        MetaModelResult with is_fallback=True.
    """
    return MetaModelResult(
        model=None,
        scaler=None,
        feature_names=list(feature_names),
        coefficients={name: 0.0 for name in feature_names},
        intercept=0.0,
        n_train_samples=n_samples,
        n_features=len(feature_names),
        training_metrics={
            "accuracy": float("nan"),
            "auc": float("nan"),
            "prevalence": float("nan"),
        },
        warnings=warnings_,
        is_fallback=True,
    )


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "MetaModelConfig",
    "MetaModelResult",
    "build_meta_features",
    "transform_meta_features",
    "train_meta_model",
    "predict_meta_proba",
    "extract_model_metadata",
    "MIN_TRAIN_SAMPLES",
    "MIN_RECOMMENDED_SAMPLES",
    "META_MODEL_VERSION",
]
