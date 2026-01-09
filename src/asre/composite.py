"""
ASRE Composite Score & ASRE-Medallion Implementation

Implements the complete composite rating algorithms:

1. R_final(t) = 100 · (w^T·s(t)) / ||w||₂ + Δ_regime + ε_residual

2. R_ASRE(t) = Σ(w_i·S_i(t)·σ_i⁻¹) / Σ(σ_i⁻¹) + λ·(d²R/dt²)|_{t-Δt}

Components:
- Score vector: s(t) = [F(t), T(t), M(t)]
- Weight optimization: Markowitz mean-variance with Sharpe constraint
- Regime adjustment: Δ_regime from sector beta decay integral
- Kalman filter: Real-time error covariance reduction
- Risk parity: Inverse volatility weighting

Production-grade features:
- Integration with calibration.py (MLE, Kalman filter)
- Uses config objects from config.py
- Leverages score modules (fundamentals.py, technical.py, momentum.py)
- Numerical optimization (scipy.optimize)
- Multiple weighting schemes
- Comprehensive validation
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from .config import CompositeConfig, FundamentalsConfig, TechnicalConfig, MomentumConfig
from .momentum import compute_momentum_score
from .technical import compute_technical_score
from .fundamentals import compute_fundamental_score
from .indicators import rolling_volatility, log_returns
from .calibration import (
    KalmanFilter1D,
    apply_kalman_filter_to_series,
    calibrate_logistic_weights,
    calibrate_noise_parameters,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight Optimization (Markowitz with MLE Calibration)
# ---------------------------------------------------------------------------


def optimize_weights(
    scores: pd.DataFrame,
    returns: Optional[pd.Series] = None,
    lambda_risk: float = 2.0,
    use_mle: bool = True,
) -> np.ndarray:
    """
    Optimize component weights using Markowitz mean-variance with optional MLE calibration.
    
    Formula: w = arg min[Var(w^T·s) - λ·Sharpe(w^T·s)]
    
    If returns provided and use_mle=True, uses logistic MLE for weight initialization.
    
    Constraints:
    - Σw_i = 1 (weights sum to 1)
    - w_i ≥ 0 (no short-weighting)
    
    Args:
        scores: DataFrame with columns [f_score, t_score, m_score]
        returns: Optional forward returns for MLE calibration
        lambda_risk: Risk aversion parameter (1.5-2.5)
        use_mle: If True, use MLE for initial weight guess
    
    Returns:
        Optimal weight array [w_F, w_T, w_M]
    """
    # Extract score matrix
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
            # Create binary target (positive returns = 1)
            binary_target = (returns > 0).astype(int)
            
            # Align with scores
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
    
    # Objective function: minimize variance - λ·Sharpe
    def objective(w):
        portfolio_var = w @ cov_matrix @ w
        portfolio_return = w @ mean_returns
        portfolio_std = np.sqrt(portfolio_var)
        
        if portfolio_std < 1e-6:
            sharpe = 0
        else:
            sharpe = portfolio_return / portfolio_std
        
        # Minimize negative objective (maximize positive objective)
        return portfolio_var - lambda_risk * sharpe
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
    ]
    
    # Bounds: prevent weight collapse
    bounds = [(0.1, 0.8) for _ in range(3)]
    
    # Optimize
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
    """
    Adjust weights based on market regime and score correlation.
    
    Formula:
    - w_F = w_F_base · (1 + φ · IV_regime)
    - w_M = w_M_base · (1 - ψ · ρ_corr)
    
    Args:
        base_weights: Base weights [w_F, w_T, w_M]
        vix: VIX level (volatility index)
        correlation_mt: Correlation between M-score and T-score
        phi: VIX sensitivity (0.05-0.15)
        psi: Collinearity penalty (0.05-0.15)
    
    Returns:
        Adjusted weights [w_F, w_T, w_M]
    """
    w_F, w_T, w_M = base_weights
    
    # VIX regime adjustment (normalize VIX to [-1, 1])
    vix_neutral = 20.0
    iv_regime = (vix - vix_neutral) / vix_neutral
    iv_regime = np.clip(iv_regime, -1, 1)
    
    # Adjust fundamental weight (increase in high vol)
    w_F_adj = w_F * (1 + phi * iv_regime)
    
    # Adjust momentum weight (decrease if correlated with technical)
    w_M_adj = w_M * (1 - psi * correlation_mt)
    
    # Keep technical weight unchanged
    w_T_adj = w_T
    
    # Renormalize to sum to 1
    adjusted = np.array([w_F_adj, w_T_adj, w_M_adj])
    adjusted = adjusted / adjusted.sum()
    
    logger.debug(
        f"Regime-adjusted weights: F={adjusted[0]:.3f}, "
        f"T={adjusted[1]:.3f}, M={adjusted[2]:.3f} "
        f"(VIX={vix:.1f}, ρ_MT={correlation_mt:.3f})"
    )
    
    return adjusted


# ---------------------------------------------------------------------------
# Regime Adjustment Term
# ---------------------------------------------------------------------------


def compute_regime_adjustment(
    sector_beta: pd.Series,
    eta: float = 0.02,
    window: int = 60,
) -> pd.Series:
    """
    Compute regime adjustment from sector beta.
    
    Formula: Δ_regime = ∫_{-∞}^t e^(-η(t-τ)) · sign(β_sector(τ)) dτ
    
    Approximated as exponentially-weighted sum of signed sector beta.
    
    Args:
        sector_beta: Sector beta time series
        eta: Decay parameter (0.01-0.05)
        window: Integration window (default 60)
    
    Returns:
        Regime adjustment series (typically ±5-10 points)
    """
    # Sign of sector beta (momentum direction)
    beta_sign = np.sign(sector_beta)
    
    # Exponential decay weights
    decay_weights = np.exp(-eta * np.arange(window)[::-1])
    decay_weights = decay_weights / decay_weights.sum()
    
    # Rolling weighted sum
    def weighted_sum(values):
        if len(values) < window:
            return 0.0
        return np.dot(values, decay_weights)
    
    regime_adj = beta_sign.rolling(window=window).apply(weighted_sum, raw=True)
    
    # Scale to typical range ±5-10
    regime_adj = regime_adj * 5
    
    # Clip extreme values
    regime_adj = np.clip(regime_adj, -10, 10)
    
    return regime_adj


# ---------------------------------------------------------------------------
# R_final Computation (Enhanced with Calibration)
# ---------------------------------------------------------------------------


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
) -> pd.DataFrame:
    """
    Compute ASRE composite rating (R_final) with full integration.
    
    Formula: R_final(t) = 100 · (w^T·s(t)) / ||w||₂ + Δ_regime + ε_residual
    
    Enhancements:
    - Uses config objects for all modules
    - MLE-based weight initialization
    - Auto-calibrated Kalman filter noise
    - Comprehensive component tracking
    
    Args:
        df: DataFrame with all required data (OHLCV, fundamentals, market context)
        config: CompositeConfig (uses defaults if None)
        fundamentals_config: FundamentalsConfig for F-Score
        technical_config: TechnicalConfig for T-Score
        momentum_config: MomentumConfig for M-Score
        optimize_weights_flag: If True, optimize weights using Markowitz
        use_kalman: If True, apply Kalman filtering
        auto_calibrate_noise: Auto-calibrate Kalman noise from residuals
        return_components: If True, return all intermediate values
    
    Returns:
        DataFrame with added columns:
        - r_final: Composite ASRE rating [0, 100]
        - r_final_kalman: Kalman-filtered rating
        - confidence_lower: Lower confidence bound (95%)
        - confidence_upper: Upper confidence bound (95%)
    
    Raises:
        ValueError: If required columns missing
    """
    
    if config is None:
        config = CompositeConfig()
    
    result_df = df.copy()
    
    # Step 1: Compute all three component scores with their respective configs
    logger.info("Computing component scores...")
    
    # F-Score
    if 'f_score' not in result_df.columns:
        logger.debug("Computing F-Score...")
        result_df = compute_fundamental_score(
            result_df,
            config=fundamentals_config,
        )
    
    # T-Score
    if 't_score' not in result_df.columns:
        logger.debug("Computing T-Score...")
        result_df = compute_technical_score(
            result_df,
            config=technical_config,
        )
    
    # M-Score
    if 'm_score_adj' not in result_df.columns:
        logger.debug("Computing M-Score...")
        result_df = compute_momentum_score(
            result_df,
            config=momentum_config,
        )
        result_df['m_score'] = result_df['m_score_adj']  # Use adjusted version
    
    # Create score vector
    score_cols = ['f_score', 't_score', 'm_score']
    scores = result_df[score_cols].copy()
    
    # Step 2: Optimize or use base weights (with optional MLE)
    if optimize_weights_flag:
        logger.info("Optimizing component weights...")
        
        # Use returns if available for MLE
        returns = None
        if 'close' in result_df.columns:
            returns = result_df['close'].pct_change().shift(-1)  # Forward returns
        
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
    
    # Get VIX (or use neutral if missing)
    vix = result_df['vix'].iloc[-1] if 'vix' in result_df.columns else 20.0
    
    # Calculate correlation between M and T scores
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
    
    # Step 4: Compute weighted composite
    logger.debug("Computing weighted composite score...")
    
    # w^T · s(t)
    weighted_scores = scores.values @ adjusted_w
    
    # L2 normalization: ||w||₂
    l2_norm = np.linalg.norm(adjusted_w)
    
    # -------------------------------------------------------------------
    # Base composite (bounded to prevent saturation)  ✅ FIXED
    # -------------------------------------------------------------------
    # Normalize composite using rolling z-score to avoid saturation
    z = (weighted_scores - np.nanmean(weighted_scores)) / (np.nanstd(weighted_scores) + 1e-6)

    SCALE = 2.2
    composite_base = 50.0 + 40.0 * np.tanh(z / SCALE)

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

    # Clip to valid range [1, 99]
    r_final = np.clip(r_final, 1, 99)
    
    # Step 7: Apply Kalman filter with auto-calibrated noise
    if use_kalman:
        logger.debug("Applying Kalman filter...")
        
        # Auto-calibrate noise parameters if requested
        process_noise = 0.05
        measurement_noise = config.sigma_obs
        
        if auto_calibrate_noise and len(r_final.dropna()) > 100:
            try:
                # Use residuals from a simple moving average as proxy
                ma_50 = r_final.rolling(50).mean()
                residuals = r_final - ma_50
                
                # Estimate innovation variance
                innovations = r_final.diff()
                
                Q_hat, R_hat = calibrate_noise_parameters(
                    residuals.dropna(),
                    innovations.dropna(),
                )
                
                process_noise = Q_hat
                measurement_noise = R_hat
                logger.info(f"Auto-calibrated noise: Q={Q_hat:.4f}, R={R_hat:.4f}")
            except Exception as e:
                logger.warning(f"Noise calibration failed: {e}, using defaults")
        
        # Apply Kalman filter using calibration.py
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
    else:
        result_df['r_final'] = r_final
        result_df['r_final_kalman'] = r_final
    
    # Optional: return all components
    if return_components:
        result_df['composite_base'] = composite_base
        result_df['regime_adjustment'] = regime_adj
        result_df['weight_f'] = adjusted_w[0]
        result_df['weight_t'] = adjusted_w[1]
        result_df['weight_m'] = adjusted_w[2]
        result_df['l2_norm'] = l2_norm
        result_df['vix_regime'] = (vix - 20.0) / 20.0
        result_df['corr_mt'] = corr_mt
    
    logger.info(
        f"R_final computed: mean={r_final.mean():.2f}, "
        f"std={r_final.std():.2f}, "
        f"range=[{r_final.min():.2f}, {r_final.max():.2f}]"
    )
    
    return result_df


# ---------------------------------------------------------------------------
# ASRE-Medallion (Ultimate Algorithm with Risk Parity)
# ---------------------------------------------------------------------------


def compute_asre_medallion(
    df: pd.DataFrame,
    config: Optional[CompositeConfig] = None,
    fundamentals_config: Optional[FundamentalsConfig] = None,
    technical_config: Optional[TechnicalConfig] = None,
    momentum_config: Optional[MomentumConfig] = None,
    return_components: bool = False,
) -> pd.DataFrame:
    """
    Compute ASRE-Medallion rating (ultimate optimized algorithm).
    
    Formula: R_ASRE(t) = Σ(w_i·S_i(t)·σ_i⁻¹) / Σ(σ_i⁻¹) + λ·(d²R/dt²)|_{t-Δt}
    
    Uses risk-parity inverse volatility weighting + acceleration term.
    
    Args:
        df: DataFrame with component scores
        config: CompositeConfig
        fundamentals_config: FundamentalsConfig
        technical_config: TechnicalConfig
        momentum_config: MomentumConfig
        return_components: Return intermediate values
    
    Returns:
        DataFrame with added column:
        - r_asre: ASRE-Medallion rating [0, 100]
    """
    
    if config is None:
        config = CompositeConfig()
    
    result_df = df.copy()
    
    # Ensure component scores exist (with configs)
    if 'f_score' not in result_df.columns:
        result_df = compute_fundamental_score(result_df, config=fundamentals_config)
    if 't_score' not in result_df.columns:
        result_df = compute_technical_score(result_df, config=technical_config)
    if 'm_score_adj' not in result_df.columns:
        result_df = compute_momentum_score(result_df, config=momentum_config)
        result_df['m_score'] = result_df['m_score_adj']
    
    score_cols = ['f_score', 't_score', 'm_score']
    scores = result_df[score_cols]
    
    # Step 1: Calculate inverse volatility for each score (risk parity)
    logger.debug("Computing inverse volatility weights (risk parity)...")
    
    vol_window = 20
    sigma_f = scores['f_score'].rolling(window=vol_window).std()
    sigma_t = scores['t_score'].rolling(window=vol_window).std()
    sigma_m = scores['m_score'].rolling(window=vol_window).std()
    
    # Fill NaN with overall std
    sigma_f = sigma_f.fillna(scores['f_score'].std())
    sigma_t = sigma_t.fillna(scores['t_score'].std())
    sigma_m = sigma_m.fillna(scores['m_score'].std())
    
    # Inverse volatility (add epsilon to avoid division by zero)
    inv_vol_f = 1 / (sigma_f + 1e-6)
    inv_vol_t = 1 / (sigma_t + 1e-6)
    inv_vol_m = 1 / (sigma_m + 1e-6)
    
    # Step 2: Risk-parity weighted composite
    logger.debug("Computing risk-parity composite...")
    
    # Base weights from config
    w_f = config.w_f_base
    w_t = config.w_t_base
    w_m = config.w_m_base
    
    # Numerator: Σ(w_i·S_i·σ_i⁻¹)
    numerator = (
        w_f * scores['f_score'] * inv_vol_f +
        w_t * scores['t_score'] * inv_vol_t +
        w_m * scores['m_score'] * inv_vol_m
    )
    
    # Denominator: Σ(σ_i⁻¹)
    denominator = inv_vol_f + inv_vol_t + inv_vol_m
    
    # Normalize risk-parity composite into 0–100 range
    base_raw = numerator / (denominator + 1e-8)

    mu = base_raw.mean()
    sigma = base_raw.std() + 1e-6

    z = (base_raw - mu) / sigma

    # Map into bounded score
    r_asre_base = 50.0 + 40.0 * np.tanh(z / 2.0)
    
    # Step 3: Compute acceleration term (d²R/dt²)
    logger.debug("Computing rating acceleration term...")
    
    # Use r_final if available, otherwise use base composite
    if 'r_final' in result_df.columns:
        rating_series = result_df['r_final']
    else:
        rating_series = r_asre_base
    
    # First derivative (velocity)
    velocity = rating_series.diff()
    
    # Second derivative (acceleration)
    acceleration = velocity.diff()
    
    # Use lagged acceleration (5-10 days back)
    lag = 5
    lagged_acceleration = acceleration.shift(lag)
    
    # Scale factor (λ, typically 0.01-0.05)
    lambda_accel = 0.08
    
    # Acceleration term
    accel_term = lambda_accel * lagged_acceleration
    accel_term = accel_term.fillna(0.0)
    
    # Clip extreme acceleration values
    accel_term = np.clip(accel_term, -5, 5)
    
    # Step 4: Final ASRE-Medallion rating
    r_asre = r_asre_base + accel_term
    
    # Clip to valid range
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
    
    logger.info(
        f"R_ASRE computed: mean={r_asre.mean():.2f}, "
        f"std={r_asre.std():.2f}, "
        f"range=[{r_asre.min():.2f}, {r_asre.max():.2f}]"
    )
    
    return result_df


# ---------------------------------------------------------------------------
# Complete Pipeline (Fully Integrated)
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
    """
    Compute complete ASRE rating system with full integration.
    
    Computes:
    1. Component scores (F, T, M) with their configs
    2. R_final (Markowitz-optimized with MLE calibration)
    3. R_ASRE (Medallion formula with risk parity)
    
    All modules integrated:
    - config.py: All configuration objects
    - fundamentals.py: F-Score computation
    - technical.py: T-Score computation
    - momentum.py: M-Score computation
    - calibration.py: MLE weights, Kalman filtering, noise calibration
    - indicators.py: Technical indicators (used by score modules)
    
    Args:
        df: DataFrame with all required data
        config: CompositeConfig
        fundamentals_config: FundamentalsConfig
        technical_config: TechnicalConfig
        momentum_config: MomentumConfig
        medallion: If True, also compute ASRE-Medallion
        return_all_components: Return all intermediate values
    
    Returns:
        DataFrame with all ASRE ratings and components
    """
    
    logger.info("=" * 70)
    logger.info("ASRE Complete Rating System - Fully Integrated")
    logger.info("=" * 70)
    
    # Compute R_final (with all integrations)
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
    
    # Compute R_ASRE (Medallion)
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


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
    
    if rating.std() < 0.1:
        return False, f"Rating has no variance: std={rating.std():.4f}"
    
    return True, f"Rating valid: mean={rating.mean():.2f}, std={rating.std():.2f}"


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def get_asre_rating(
    df: pd.DataFrame,
    rating_type: str = 'medallion',
) -> pd.Series:
    """
    Get specific ASRE rating.
    
    Args:
        df: DataFrame with ASRE ratings
        rating_type: 'final', 'kalman', or 'medallion'
    
    Returns:
        Rating series
    """
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
]
