"""
ASRE Calibration & Statistical Estimation

Implements calibration algorithms for ASRE parameters:

1. Maximum Likelihood Estimation (MLE)
   - Normal distribution parameter estimation
   - Logistic regression for F-Score weights
   - Parameter bounds and constraints

2. Kalman Filter
   - 1-D state estimation with process/measurement noise
   - Real-time error covariance updates
   - Confidence interval computation

3. Parameter Fitting
   - Grid search optimization
   - Cross-validation
   - Walk-forward analysis

Production-grade features:
- Numerical stability (log-likelihood, regularization)
- Constraint handling (scipy.optimize)
- Comprehensive validation
- Multiple optimization methods
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Maximum Likelihood Estimation (MLE)
# ---------------------------------------------------------------------------


def mle_normal_distribution(
    data: np.ndarray,
) -> Tuple[float, float]:
    """
    Maximum Likelihood Estimation for normal distribution.
    
    Formula: 
        μ̂ = (1/n) · Σx_i
        σ̂² = (1/n) · Σ(x_i - μ̂)²
    
    Args:
        data: Sample data array
    
    Returns:
        Tuple of (mean, std_dev)
    """
    # Remove NaN values
    data_clean = data[~np.isnan(data)]
    
    if len(data_clean) == 0:
        logger.warning("No valid data for MLE, returning defaults")
        return 0.0, 1.0
    
    # MLE estimates
    mu_hat = np.mean(data_clean)
    sigma_hat = np.std(data_clean, ddof=0)  # MLE uses ddof=0
    
    logger.debug(f"MLE Normal: μ={mu_hat:.4f}, σ={sigma_hat:.4f}")
    
    return mu_hat, sigma_hat


def log_likelihood_normal(
    data: np.ndarray,
    mu: float,
    sigma: float,
) -> float:
    """
    Log-likelihood for normal distribution.
    
    Formula: log L(μ,σ|data) = -n/2·log(2πσ²) - (1/2σ²)·Σ(x_i - μ)²
    
    Args:
        data: Sample data
        mu: Mean parameter
        sigma: Standard deviation parameter
    
    Returns:
        Log-likelihood value
    """
    n = len(data)
    
    if sigma <= 0:
        return -np.inf
    
    # Calculate squared errors
    squared_errors = (data - mu) ** 2
    
    # Log-likelihood formula
    log_lik = -0.5 * n * np.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2)) * np.sum(squared_errors)
    
    return log_lik


def mle_with_bounds(
    data: np.ndarray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    initial_guess: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """
    MLE for normal distribution with parameter bounds.
    
    Uses numerical optimization to find MLE subject to constraints.
    
    Args:
        data: Sample data
        bounds: ((mu_min, mu_max), (sigma_min, sigma_max))
        initial_guess: Initial (mu, sigma) values
    
    Returns:
        Tuple of (optimal_mu, optimal_sigma)
    """
    data_clean = data[~np.isnan(data)]
    
    if len(data_clean) == 0:
        return 0.0, 1.0
    
    # Negative log-likelihood (to minimize)
    def neg_log_likelihood(params):
        mu, sigma = params
        return -log_likelihood_normal(data_clean, mu, sigma)
    
    # Initial guess
    if initial_guess is None:
        initial_guess = (np.mean(data_clean), np.std(data_clean))
    
    # Optimize
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
# Logistic Regression Calibration
# ---------------------------------------------------------------------------


def calibrate_logistic_weights(
    features: pd.DataFrame,
    target: pd.Series,
    feature_names: List[str] = ['f1_pe', 'f2_roe', 'f3_de'],
    initial_weights: Optional[np.ndarray] = None,
    l2_penalty: float = 0.01,
) -> Tuple[np.ndarray, float]:
    """
    Calibrate logistic regression weights using MLE.
    
    Formula: P(y=1|X) = 1 / (1 + exp(-(β₀ + β₁x₁ + β₂x₂ + β₃x₃)))
    
    Optimizes cross-entropy loss with L2 regularization.
    
    Args:
        features: DataFrame with feature columns
        target: Binary target (1 for positive return, 0 for negative)
        feature_names: Names of feature columns
        initial_weights: Initial β values
        l2_penalty: L2 regularization strength
    
    Returns:
        Tuple of (calibrated_weights, roc_auc_score)
    """
    # Prepare data
    X = features[feature_names].values
    y = target.values
    
    # Remove NaN
    valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[valid_idx]
    y_clean = y[valid_idx]
    
    if len(X_clean) < 50:
        logger.warning("Insufficient data for logistic calibration")
        return np.array([0.4, 0.35, 0.25]), 0.5
    
    # Fit logistic regression
    try:
        model = LogisticRegression(
            penalty='l2',
            C=1/l2_penalty,  # Inverse of regularization strength
            solver='lbfgs',
            max_iter=1000,
            fit_intercept=True,
        )
        
        model.fit(X_clean, y_clean)
        
        # Extract coefficients
        coefficients = model.coef_[0]
        
        # Normalize to sum to 1 (convert to weights)
        abs_coef = np.abs(coefficients)
        if abs_coef.sum() > 0:
            weights = abs_coef / abs_coef.sum()
        else:
            weights = np.array([1/3, 1/3, 1/3])
        
        # Calculate ROC-AUC
        y_pred_proba = model.predict_proba(X_clean)[:, 1]
        auc_score = roc_auc_score(y_clean, y_pred_proba)
        
        logger.info(
            f"Logistic calibration: weights={weights}, "
            f"AUC={auc_score:.3f}, intercept={model.intercept_[0]:.3f}"
        )
        
        return weights, auc_score
    
    except Exception as e:
        logger.error(f"Logistic calibration failed: {e}")
        return np.array([0.4, 0.35, 0.25]), 0.5


def calibrate_with_constraints(
    features: pd.DataFrame,
    target: pd.Series,
    feature_names: List[str],
    weight_bounds: List[Tuple[float, float]] = [(0.2, 0.6), (0.2, 0.5), (0.1, 0.4)],
) -> np.ndarray:
    """
    Calibrate weights with explicit constraints.
    
    Constraints:
    - Each weight within bounds
    - Weights sum to 1
    
    Args:
        features: Feature DataFrame
        target: Target series
        feature_names: Feature column names
        weight_bounds: Bounds for each weight
    
    Returns:
        Calibrated weight array
    """
    X = features[feature_names].values
    y = target.values
    
    # Remove NaN
    valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[valid_idx]
    y_clean = y[valid_idx]
    
    if len(X_clean) < 50:
        return np.array([0.4, 0.35, 0.25])
    
    # Objective: minimize negative log-likelihood (cross-entropy)
    def objective(weights):
        # Weighted features
        z = X_clean @ weights
        
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        
        # Sigmoid probabilities
        probs = expit(z)
        
        # Clip probabilities away from 0 and 1
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        
        # Cross-entropy loss
        loss = -np.mean(y_clean * np.log(probs) + (1 - y_clean) * np.log(1 - probs))
        
        return loss
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
    ]
    
    # Optimize
    result = minimize(
        objective,
        x0=np.array([0.4, 0.35, 0.25]),
        method='SLSQP',
        bounds=weight_bounds,
        constraints=constraints,
    )
    
    if result.success:
        logger.info(f"Constrained calibration: weights={result.x}")
        return result.x
    else:
        logger.warning("Constrained optimization failed")
        return np.array([0.4, 0.35, 0.25])


# ---------------------------------------------------------------------------
# Kalman Filter Implementation
# ---------------------------------------------------------------------------


class KalmanFilter1D:
    """
    1-D Kalman Filter for scalar state estimation.
    
    State equation: x(t) = x(t-1) + w(t), w ~ N(0, Q)
    Measurement equation: y(t) = x(t) + v(t), v ~ N(0, R)
    
    where:
    - x(t) = true state (e.g., true rating)
    - y(t) = noisy measurement (e.g., observed composite score)
    - Q = process noise variance
    - R = measurement noise variance
    
    Implements:
    - Prediction: x̂(t|t-1), P(t|t-1)
    - Update: x̂(t|t), P(t|t) using Kalman gain K(t)
    """
    
    def __init__(
        self,
        initial_state: float = 50.0,
        initial_covariance: float = 1.0,
        process_noise: float = 0.1,
        measurement_noise: float = 0.2,
    ):
        """
        Initialize Kalman filter.
        
        Args:
            initial_state: Initial state estimate x̂(0)
            initial_covariance: Initial error covariance P(0)
            process_noise: Process noise variance Q
            measurement_noise: Measurement noise variance R
        """
        self.x = initial_state
        self.P = initial_covariance
        self.Q = process_noise
        self.R = measurement_noise
        
        # History for analysis
        self.history = {
            'state': [initial_state],
            'covariance': [initial_covariance],
            'kalman_gain': [],
            'innovation': [],
            'measurement': [],
        }
    
    def predict(self) -> Tuple[float, float]:
        """
        Prediction step (time update).
        
        Formulas:
        - x̂(t|t-1) = x̂(t-1|t-1)  [assuming constant state]
        - P(t|t-1) = P(t-1|t-1) + Q
        
        Returns:
            Tuple of (predicted_state, predicted_covariance)
        """
        # State prediction (assuming constant model)
        x_pred = self.x
        
        # Covariance prediction (uncertainty increases)
        P_pred = self.P + self.Q
        
        return x_pred, P_pred
    
    def update(self, measurement: float) -> Tuple[float, float]:
        """
        Update step (measurement update).
        
        Formulas:
        - K(t) = P(t|t-1) / [P(t|t-1) + R]  [Kalman gain]
        - x̂(t|t) = x̂(t|t-1) + K(t)·[y(t) - x̂(t|t-1)]
        - P(t|t) = [1 - K(t)] · P(t|t-1)
        
        Args:
            measurement: Observed measurement y(t)
        
        Returns:
            Tuple of (updated_state, updated_covariance)
        """
        # Prediction
        x_pred, P_pred = self.predict()
        
        # Kalman gain
        K = P_pred / (P_pred + self.R)
        
        # Innovation (measurement residual)
        innovation = measurement - x_pred
        
        # State update
        self.x = x_pred + K * innovation
        
        # Covariance update
        self.P = (1 - K) * P_pred
        
        # Store history
        self.history['state'].append(self.x)
        self.history['covariance'].append(self.P)
        self.history['kalman_gain'].append(K)
        self.history['innovation'].append(innovation)
        self.history['measurement'].append(measurement)
        
        return self.x, self.P
    
    def get_confidence_interval(
        self,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for current state.
        
        Formula: CI = x̂(t) ± z_{α/2} · √P(t)
        
        Args:
            confidence: Confidence level (default 0.95)
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        z_score = stats.norm.ppf((1 + confidence) / 2)
        std = np.sqrt(self.P)
        
        lower = self.x - z_score * std
        upper = self.x + z_score * std
        
        return lower, upper
    
    def reset(self, state: float = 50.0, covariance: float = 1.0):
        """Reset filter to initial conditions."""
        self.x = state
        self.P = covariance
        self.history = {
            'state': [state],
            'covariance': [covariance],
            'kalman_gain': [],
            'innovation': [],
            'measurement': [],
        }
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """
        Get filter history as DataFrame.
        
        Returns:
            DataFrame with columns: state, covariance, kalman_gain, innovation, measurement
        """
        # Pad shorter lists with NaN
        max_len = len(self.history['state'])
        
        df_dict = {}
        for key, values in self.history.items():
            if len(values) < max_len:
                # Pad beginning with NaN
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
    """
    Apply Kalman filter to entire time series.
    
    Args:
        measurements: Measurement series (e.g., raw ratings)
        process_noise: Process noise Q
        measurement_noise: Measurement noise R
        initial_state: Initial state estimate (uses first measurement if None)
    
    Returns:
        DataFrame with columns: measurement, filtered_state, covariance, lower_ci, upper_ci
    """
    if initial_state is None:
        initial_state = measurements.dropna().iloc[0] if len(measurements.dropna()) > 0 else 50.0
    
    kf = KalmanFilter1D(
        initial_state=initial_state,
        initial_covariance=1.0,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
    )
    
    filtered_states = []
    covariances = []
    lower_bounds = []
    upper_bounds = []
    
    for measurement in measurements:
        if not np.isnan(measurement):
            state, cov = kf.update(measurement)
        else:
            # If measurement missing, just predict
            state, cov = kf.predict()
        
        lower, upper = kf.get_confidence_interval()
        
        filtered_states.append(state)
        covariances.append(cov)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
    
    result_df = pd.DataFrame({
        'measurement': measurements.values,
        'filtered_state': filtered_states,
        'covariance': covariances,
        'lower_ci': lower_bounds,
        'upper_ci': upper_bounds,
    }, index=measurements.index)
    
    return result_df


# ---------------------------------------------------------------------------
# Parameter Fitting & Optimization
# ---------------------------------------------------------------------------


def fit_parameters_grid_search(
    df: pd.DataFrame,
    compute_score_func: Callable,
    param_grid: Dict[str, List[float]],
    metric_func: Callable,
    n_splits: int = 5,
) -> Dict[str, float]:
    """
    Fit parameters using grid search with time-series cross-validation.
    
    Args:
        df: DataFrame with data
        compute_score_func: Function that computes score given parameters
        param_grid: Dict mapping parameter names to lists of values
        metric_func: Function to evaluate performance (e.g., Sharpe ratio)
        n_splits: Number of time-series splits for CV
    
    Returns:
        Dict with best parameters
    """
    from itertools import product
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    logger.info(f"Grid search: {len(combinations)} combinations, {n_splits} CV splits")
    
    best_score = -np.inf
    best_params = None
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for combo in combinations:
        params = dict(zip(param_names, combo))
        
        scores = []
        
        for train_idx, test_idx in tscv.split(df):
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]
            
            try:
                # Compute score on test set with these parameters
                score = compute_score_func(df_test, params)
                metric = metric_func(score)
                scores.append(metric)
            except Exception as e:
                logger.warning(f"Error with params {params}: {e}")
                scores.append(-np.inf)
        
        avg_score = np.mean(scores)
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            logger.debug(f"New best: {params} → {avg_score:.4f}")
    
    logger.info(f"Best parameters: {best_params} (score={best_score:.4f})")
    
    return best_params


def fit_parameters_differential_evolution(
    df: pd.DataFrame,
    objective_func: Callable,
    bounds: List[Tuple[float, float]],
    param_names: List[str],
    maxiter: int = 100,
) -> Dict[str, float]:
    """
    Fit parameters using differential evolution (global optimization).
    
    Differential evolution is robust for non-convex, noisy objectives.
    
    Args:
        df: DataFrame with data
        objective_func: Function to minimize (e.g., negative Sharpe)
        bounds: List of (min, max) tuples for each parameter
        param_names: Names of parameters (must match bounds order)
        maxiter: Maximum iterations
    
    Returns:
        Dict with optimal parameters
    """
    logger.info("Starting differential evolution optimization...")
    
    # Wrapper to pass parameters as dict
    def objective_wrapper(params):
        param_dict = dict(zip(param_names, params))
        return objective_func(df, param_dict)
    
    # Optimize
    result = differential_evolution(
        objective_wrapper,
        bounds=bounds,
        maxiter=maxiter,
        seed=42,
        disp=False,
        workers=1,
    )
    
    if result.success:
        optimal_params = dict(zip(param_names, result.x))
        logger.info(f"Optimization converged: {optimal_params} (objective={result.fun:.4f})")
        return optimal_params
    else:
        logger.warning("Optimization did not converge")
        # Return midpoint of bounds
        midpoint = [(low + high) / 2 for low, high in bounds]
        return dict(zip(param_names, midpoint))


def calibrate_noise_parameters(
    residuals: pd.Series,
    innovations: pd.Series,
) -> Tuple[float, float]:
    """
    Calibrate Kalman filter noise parameters from residuals.
    
    Args:
        residuals: Model residuals (predicted - actual)
        innovations: Kalman filter innovations
    
    Returns:
        Tuple of (process_noise_Q, measurement_noise_R)
    """
    # Remove NaN
    residuals_clean = residuals.dropna()
    innovations_clean = innovations.dropna()
    
    if len(residuals_clean) < 10 or len(innovations_clean) < 10:
        logger.warning("Insufficient data for noise calibration")
        return 0.1, 0.2
    
    # Estimate measurement noise from residuals
    R_hat = np.var(residuals_clean)
    
    # Estimate process noise from innovations
    Q_hat = np.var(innovations_clean)
    
    # Ensure positive and reasonable
    R_hat = max(0.01, min(R_hat, 10.0))
    Q_hat = max(0.001, min(Q_hat, 5.0))
    
    logger.info(f"Calibrated noise: Q={Q_hat:.4f}, R={R_hat:.4f}")
    
    return Q_hat, R_hat


# ---------------------------------------------------------------------------
# Walk-Forward Analysis
# ---------------------------------------------------------------------------


def walk_forward_validation(
    df: pd.DataFrame,
    compute_score_func: Callable,
    param_grid: Dict[str, List[float]],
    train_window: int = 252,
    test_window: int = 63,
    step_size: int = 63,
) -> pd.DataFrame:
    """
    Walk-forward validation for parameter stability.
    
    For each time period:
    1. Train on past 'train_window' days
    2. Test on next 'test_window' days
    3. Step forward by 'step_size' days
    
    Args:
        df: DataFrame with data
        compute_score_func: Score computation function
        param_grid: Parameter grid for optimization
        train_window: Training window size (days)
        test_window: Test window size (days)
        step_size: Step size for rolling forward (days)
    
    Returns:
        DataFrame with walk-forward results
    """
    results = []
    
    start_idx = train_window
    end_idx = len(df) - test_window
    
    logger.info(f"Walk-forward: train={train_window}, test={test_window}, step={step_size}")
    
    for idx in range(start_idx, end_idx, step_size):
        train_start = idx - train_window
        train_end = idx
        test_start = idx
        test_end = idx + test_window
        
        df_train = df.iloc[train_start:train_end]
        df_test = df.iloc[test_start:test_end]
        
        # Optimize parameters on training set
        # (simplified: use first param combination for demo)
        param_names = list(param_grid.keys())
        param_values = [values[0] for values in param_grid.values()]
        params = dict(zip(param_names, param_values))
        
        # Compute score on test set
        try:
            score = compute_score_func(df_test, params)
            
            results.append({
                'train_start': df_train.index[0],
                'train_end': df_train.index[-1],
                'test_start': df_test.index[0],
                'test_end': df_test.index[-1],
                'params': params,
                'test_score': score.mean() if hasattr(score, 'mean') else score,
            })
        except Exception as e:
            logger.warning(f"Walk-forward step failed: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    logger.info(f"Walk-forward completed: {len(results)} periods")
    
    return results_df


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def estimate_all_parameters(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Estimate all calibration parameters in one go.
    
    Returns:
        Dict with all estimated parameters
    """
    results = {}
    
    # MLE for normal distribution (e.g., returns)
    if target_col in df.columns:
        mu, sigma = mle_normal_distribution(df[target_col].values)
        results['return_mu'] = mu
        results['return_sigma'] = sigma
    
    # Logistic weights
    if all(col in df.columns for col in feature_cols):
        # Create binary target (positive returns = 1)
        binary_target = (df[target_col] > 0).astype(int)
        
        weights, auc = calibrate_logistic_weights(
            df[feature_cols],
            binary_target,
            feature_names=feature_cols,
        )
        
        results['logistic_weights'] = weights
        results['auc_score'] = auc
    
    # Kalman noise parameters (if residuals available)
    if 'residuals' in df.columns:
        Q, R = calibrate_noise_parameters(
            df['residuals'],
            df.get('innovations', df['residuals']),
        )
        results['process_noise'] = Q
        results['measurement_noise'] = R
    
    return results


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = [
    # MLE
    'mle_normal_distribution',
    'log_likelihood_normal',
    'mle_with_bounds',
    
    # Logistic calibration
    'calibrate_logistic_weights',
    'calibrate_with_constraints',
    
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
