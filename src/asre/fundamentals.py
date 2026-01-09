"""
ASRE Fundamental Score (F-Score) Implementation

Implements the exact F-Score formula from ASRE-Medallion specification:

F(t) = 100 · Φ((μ_F - X_F) / σ_F) · (1 - e^(-α·D_F(t)))

Components:
- μ_F = (1/n) Σ L_j(PE, ROE, D/E) - Cross-sectional mean
- L_j = β₁·f₁(PE) + β₂·f₂(ROE) + β₃·f₃(D/E) + v^T·X - Logistic likelihood
- D_F(t) = ∫₀ᵗ ∇F_j(τ) dτ - Drift term (cumulative gradient)
- Φ(·) = Standard normal CDF
- σ_F = Rolling 252-day volatility
- α = Mean-reversion decay coefficient
- v = Eigenvector from PCA (covariance matrix weighting)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import FundamentalsConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature Transformation Functions
# ---------------------------------------------------------------------------

def f1_pe_transform(pe: pd.Series) -> pd.Series:
    """
    Transform P/E ratio: f₁(PE) = log(1 + PE)
    
    Compresses extreme values and handles negatives gracefully.
    """
    pe_clean = pe.clip(lower=0)
    return np.log1p(pe_clean)

def f2_roe_transform(roe: pd.Series, cap: float = 40.0) -> pd.Series:
    """
    Transform ROE: f₂(ROE) = min(ROE, 40%) / 40%
    
    Caps at 40% (sustainable level) and normalizes to [0, 1].
    """
    roe_capped = roe.clip(lower=0, upper=cap)
    return roe_capped / cap

def f3_de_transform(de: pd.Series) -> pd.Series:
    """
    Transform D/E: f₃(D/E) = 1 / (1 + D/E)
    
    Inverse relationship: lower debt → higher score.
    """
    de_clean = de.clip(lower=0)
    return 1.0 / (1.0 + de_clean)

def create_feature_matrix(
    pe: pd.Series,
    roe: pd.Series,
    de: pd.Series,
) -> pd.DataFrame:
    """Create feature matrix X = [f₁(PE), f₂(ROE), f₃(D/E)]"""
    return pd.DataFrame({
        'f1_pe': f1_pe_transform(pe),
        'f2_roe': f2_roe_transform(roe),
        'f3_de': f3_de_transform(de),
    })

# ---------------------------------------------------------------------------
# Logistic Likelihood Function L_j
# ---------------------------------------------------------------------------

def compute_logistic_likelihood(
    features: pd.DataFrame,
    beta_1: float,
    beta_2: float,
    beta_3: float,
    pca_eigenvector: Optional[np.ndarray] = None,
) -> pd.Series:
    """
    Compute logistic likelihood: L_j = β₁·f₁ + β₂·f₂ + β₃·f₃ + v^T·X
    
    Args:
        features: Transformed features [f1_pe, f2_roe, f3_de]
        beta_1, beta_2, beta_3: Feature weights
        pca_eigenvector: v from PCA (optional weighting)
    
    Returns:
        Likelihood series L_j
    """
    likelihood = (
        beta_1 * features['f1_pe'] +
        beta_2 * features['f2_roe'] +
        beta_3 * features['f3_de']
    )
    
    # Add PCA component v^T·X if available
    if pca_eigenvector is not None and len(pca_eigenvector) == 3:
        X = features[['f1_pe', 'f2_roe', 'f3_de']].values
        pca_term = X @ pca_eigenvector
        likelihood += pca_term
    
    return likelihood

# ---------------------------------------------------------------------------
# PCA Eigenvector Extraction
# ---------------------------------------------------------------------------

def extract_pca_eigenvector(features: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """
    Extract eigenvector v from PCA on covariance matrix.
    
    Returns:
        (eigenvector, explained_variance_ratio)
    """
    features_clean = features.dropna()
    
    if len(features_clean) < 3:
        logger.warning("Insufficient data for PCA")
        return np.zeros(3), 0.0
    
    # Check variance
    if features_clean.std().min() < 1e-10:
        logger.warning("Features have near-zero variance, skipping PCA")
        return np.zeros(3), 0.0
    
    # Standardize and fit PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_clean)
    
    pca = PCA(n_components=1)
    pca.fit(X_scaled)
    
    eigenvector = pca.components_[0]
    explained_var = pca.explained_variance_ratio_[0]
    
    logger.debug(f"PCA eigenvector: v=[{eigenvector[0]:.3f}, {eigenvector[1]:.3f}, {eigenvector[2]:.3f}], "
                 f"explained={explained_var:.2%}")
    
    return eigenvector, explained_var

# ---------------------------------------------------------------------------
# Cross-sectional Mean μ_F
# ---------------------------------------------------------------------------

def compute_cross_sectional_mean(likelihood: pd.Series) -> float:
    """
    Compute cross-sectional mean: μ_F = (1/n) Σ L_j
    
    This is the average likelihood across all observations.
    """
    return likelihood.mean()

# ---------------------------------------------------------------------------
# Rolling Volatility σ_F
# ---------------------------------------------------------------------------

def compute_rolling_volatility(
    likelihood: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    Compute rolling 252-day volatility: σ_F
    
    Rolling standard deviation of likelihood.
    """
    volatility = likelihood.rolling(window=window, min_periods=30).std()
    
    # Fill early NaN with overall std
    volatility = volatility.fillna(likelihood.std())
    
    # Ensure minimum volatility to avoid division by zero
    volatility = volatility.clip(lower=1e-6)
    
    return volatility

# ---------------------------------------------------------------------------
# Drift Term D_F(t) = ∫₀ᵗ ∇F_j(τ) dτ - FIXED
# ---------------------------------------------------------------------------
def compute_drift_term(
    likelihood: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    Compute drift term: D_F(t) = ∫₀ᵗ ∇F_j(τ) dτ
    
    Measures cumulative fundamental change, properly scaled.
    
    Args:
        likelihood: Logistic likelihood series
        window: Integration window (252 trading days = 1 year)
    
    Returns:
        Drift term series (scaled to [1, 10] for reasonable decay factors)
    """
    # Gradient (first difference)
    gradient = likelihood.diff()
    
    # Cumulative absolute sum over window (total activity)
    cumulative_change = gradient.abs().rolling(window=window, min_periods=10).sum()
    
    # Fill early NaN with median
    cumulative_change = cumulative_change.fillna(cumulative_change.median())
    
    # Scale to reasonable range [1, 10]
    # This ensures decay_factor = 1 - exp(-α·D_F) is meaningful
    # With α=0.02 and D_F in [1,10], decay factor ranges from ~0.02 to ~0.18
    
    min_drift = cumulative_change.min()
    max_drift = cumulative_change.max()
    
    if max_drift - min_drift < 1e-10:
        # No variance, return constant mid-range value
        return pd.Series(5.0, index=likelihood.index)
    
    # Normalize to [1, 10]
    drift_scaled = 1 + 9 * (cumulative_change - min_drift) / (max_drift - min_drift)
    
    return drift_scaled

# ---------------------------------------------------------------------------
# Main F-Score Computation
# ---------------------------------------------------------------------------

def compute_fundamental_score(
    df: pd.DataFrame,
    config: Optional[FundamentalsConfig] = None,
    universe_df: Optional[pd.DataFrame] = None,
    return_components: bool = False,
) -> pd.DataFrame:
    """
    Compute Fundamental Score (F-Score) using exact ASRE-Medallion formula:
    
    F(t) = 100 · Φ((μ_F - X_F) / σ_F) · (1 - e^(-α·D_F(t)))
    
    Args:
        df: DataFrame with columns [pe, roe, de]
        config: FundamentalsConfig with parameters (α, β₁, β₂, β₃)
        universe_df: Optional peer universe for cross-sectional μ_F
        return_components: Return intermediate calculations
    
    Returns:
        DataFrame with f_score column (0-100 scale)
    """
    if config is None:
        config = FundamentalsConfig()
    
    # Validate
    required = ['pe', 'roe', 'de']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    result_df = df.copy()
    
    # -----------------------------------------------------------------------
    # Step 1: Feature Transformation X = [f₁(PE), f₂(ROE), f₃(D/E)]
    # -----------------------------------------------------------------------
    logger.debug("Step 1: Transforming features...")
    features = create_feature_matrix(df['pe'], df['roe'], df['de'])
    
    # -----------------------------------------------------------------------
    # Step 2: Extract PCA Eigenvector v
    # -----------------------------------------------------------------------
    logger.debug("Step 2: Extracting PCA eigenvector...")
    pca_eigenvector, explained_var = extract_pca_eigenvector(features)
    
    # -----------------------------------------------------------------------
    # Step 3: Compute Logistic Likelihood L_j
    # -----------------------------------------------------------------------
    logger.debug("Step 3: Computing logistic likelihood L_j...")
    likelihood = compute_logistic_likelihood(
        features,
        beta_1=config.beta_1,
        beta_2=config.beta_2,
        beta_3=config.beta_3,
        pca_eigenvector=pca_eigenvector,
    )
    
    # -----------------------------------------------------------------------
    # Step 4: Compute Cross-sectional Mean μ_F
    # -----------------------------------------------------------------------
    logger.debug("Step 4: Computing cross-sectional mean μ_F...")
    if universe_df is not None:
        # Use universe for μ_F
        universe_features = create_feature_matrix(
            universe_df['pe'],
            universe_df['roe'],
            universe_df['de'],
        )
        universe_likelihood = compute_logistic_likelihood(
            universe_features,
            config.beta_1,
            config.beta_2,
            config.beta_3,
            pca_eigenvector,
        )
        mu_f = compute_cross_sectional_mean(universe_likelihood)
    else:
        # Use time-series mean
        mu_f = compute_cross_sectional_mean(likelihood)
    
    logger.debug(f"μ_F = {mu_f:.4f}")
    
    # -----------------------------------------------------------------------
    # Step 5: Compute Rolling Volatility σ_F
    # -----------------------------------------------------------------------
    logger.debug("Step 5: Computing rolling volatility σ_F...")
    sigma_f = compute_rolling_volatility(likelihood, window=config.window_252d)
    
    # -----------------------------------------------------------------------
    # Step 6: Compute Drift Term D_F(t)
    # -----------------------------------------------------------------------
    logger.debug(f"Step 6: Computing drift term D_F(t) with α={config.alpha}...")
    drift_term = compute_drift_term(likelihood, window=config.window_252d)
    
    # -----------------------------------------------------------------------
    # Step 7: Compute Z-Score: (μ_F - X_F) / σ_F
    # -----------------------------------------------------------------------
    logger.debug("Step 7: Computing z-score...")
    X_F = likelihood  # Current likelihood is X_F
    z_score = (mu_f - X_F) / sigma_f
    
    # Clip extreme z-scores
    z_score = z_score.clip(-5, 5)
    
    # -----------------------------------------------------------------------
    # Step 8: Apply Standard Normal CDF Φ(z)
    # -----------------------------------------------------------------------
    logger.debug("Step 8: Applying standard normal CDF Φ(z)...")
    phi_z = pd.Series(stats.norm.cdf(z_score), index=z_score.index)
    
    # -----------------------------------------------------------------------
    # Step 9: Compute Decay Factor (1 - e^(-α·D_F))  ✅ IMAGE-CORRECT
    # -----------------------------------------------------------------------
    logger.debug("Step 9: Computing decay factor...")

    decay_factor = 1.0 - np.exp(-config.alpha * drift_term)

    # Numerical safety
    decay_factor = decay_factor.clip(0.0, 1.0)
    
    # -----------------------------------------------------------------------
    # Step 10: Assemble F-Score
    # -----------------------------------------------------------------------
    logger.debug("Step 10: Assembling final F-Score...")
    f_score = 100 * phi_z * decay_factor
    f_score = f_score.clip(0, 100)
    
    result_df['f_score'] = f_score
    
    # Store intermediate components
    if return_components:
        result_df['f1_pe'] = features['f1_pe']
        result_df['f2_roe'] = features['f2_roe']
        result_df['f3_de'] = features['f3_de']
        result_df['likelihood_L_j'] = likelihood
        result_df['mu_f'] = mu_f
        result_df['sigma_f'] = sigma_f
        result_df['drift_D_f'] = drift_term
        result_df['z_score'] = z_score
        result_df['phi_z'] = phi_z
        result_df['decay_factor'] = decay_factor
        result_df['pca_explained_var'] = explained_var
    
    logger.info(
        f"F-Score computed: "
        f"mean={f_score.mean():.2f}, "
        f"std={f_score.std():.2f}, "
        f"range=[{f_score.min():.2f}, {f_score.max():.2f}]"
    )
    
    return result_df

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    'compute_fundamental_score',
    'f1_pe_transform',
    'f2_roe_transform',
    'f3_de_transform',
    'compute_logistic_likelihood',
    'extract_pca_eigenvector',
    'compute_cross_sectional_mean',
    'compute_rolling_volatility',
    'compute_drift_term',
]
