"""
ASRE Universal Fundamental Score (F-Score)
Statistical formula based on probabilistic modeling and mean-reversion dynamics.

Mathematical Foundation:
    F(t) = 100 · Φ((μ_F - X_F)/σ_F) · (1 - e^(-α·D_F(t)))

Where:
    - Φ(·) = Standard normal CDF (z-score transformation)
    - μ_F = Mean logistic likelihood across historical periods
    - σ_F = Rolling volatility of fundamental metrics
    - α = Mean-reversion decay parameter
    - D_F(t) = Cumulative drift (integral of gradient)

Author: ASRE Rating System
Version: 4.0 (Statistical)
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit  # Logistic sigmoid
from sklearn.decomposition import PCA

from asre.config import FundamentalsConfig

logger = logging.getLogger(__name__)


# ===========================================================================
# STATISTICAL PARAMETERS
# ===========================================================================

class FScoreParameters:
    """Statistical parameters for F-Score computation."""
    
    # Logistic likelihood weights (β coefficients)
    BETA_PE = 0.35      # Weight for P/E ratio
    BETA_ROE = 0.45     # Weight for ROE
    BETA_DE = 0.20      # Weight for Debt/Equity
    
    # Mean-reversion parameter
    ALPHA = 0.15        # Decay coefficient (higher = faster mean reversion)
    
    # Volatility window
    VOL_WINDOW = 252    # 1 year of trading days
    
    # Z-score bounds
    Z_LOWER = -3.0      # Lower bound for z-score
    Z_UPPER = 3.0       # Upper bound for z-score
    
    # PCA components
    N_COMPONENTS = 3    # Number of principal components


# ===========================================================================
# STOCK CLASSIFICATION SYSTEM (S/A/B/C/D Tiers) - CATEGORICAL APPROACH
# ===========================================================================

def classify_stock(
    roe: float,
    revenue_growth: float,
    pe: float,
    de: float,
    profit_margin: Optional[float] = None,
) -> Dict:
    """
    Classify stock into quality tier with base score.
    
    Tiers:
        S-Tier (90-95): Exceptional growth + profitability
        A-Tier (75-85): High-quality companies
        B-Tier (60-70): Solid fundamentals
        C-Tier (45-55): Fair value, limited growth
        D-Tier (30-40): Distressed/weak fundamentals
    
    Returns:
        Dict with category, tier, base_score, peg, description
    """
    peg = pe / revenue_growth if revenue_growth > 0.1 else 5.0
    
    # S-TIER: Exceptional Growth
    if roe > 80 and revenue_growth > 50 and peg < 1.0 and de < 0.5:
        return {
            'category': 'exceptional_growth',
            'tier': 'S',
            'base_score': 92,
            'peg': peg,
            'description': 'World-class: Exceptional growth + profitability + undervalued'
        }
    
    # A-TIER: High-Quality Growth
    if roe > 50 and revenue_growth > 30 and peg < 1.5 and de < 0.8:
        return {
            'category': 'high_quality_growth',
            'tier': 'A',
            'base_score': 82,
            'peg': peg,
            'description': 'High-quality growth: Strong across all metrics'
        }
    
    if roe > 30 and revenue_growth > 15 and peg < 2.5:
        return {
            'category': 'quality_growth',
            'tier': 'A',
            'base_score': 77,
            'peg': peg,
            'description': 'Quality growth: Balanced strength in growth + profitability'
        }
    
    if roe > 100 and de > 1.0:
        return {
            'category': 'cash_cow_leveraged',
            'tier': 'A',
            'base_score': 73,
            'peg': peg,
            'description': 'Cash cow: Exceptional ROE, modest growth, leveraged'
        }
    
    # B-TIER: Solid Quality
    if roe > 20 and revenue_growth > 10 and de < 1.5:
        return {
            'category': 'solid_growth',
            'tier': 'B',
            'base_score': 68,
            'peg': peg,
            'description': 'Solid growth: Above-average fundamentals'
        }
    
    if roe > 15 and de < 0.8:
        return {
            'category': 'balanced_quality',
            'tier': 'B',
            'base_score': 63,
            'peg': peg,
            'description': 'Balanced: Solid fundamentals, low risk'
        }
    
    # C-TIER: Fair Value
    if roe > 10 and pe < 20:
        return {
            'category': 'value',
            'tier': 'C',
            'base_score': 55,
            'peg': peg,
            'description': 'Value: Undervalued but limited growth'
        }
    
    if roe > 10 and revenue_growth > 5:
        return {
            'category': 'stable',
            'tier': 'C',
            'base_score': 52,
            'peg': peg,
            'description': 'Stable: Modest growth and profitability'
        }
    
    # D-TIER: Weak/Distressed
    if roe < 10 or revenue_growth < 0:
        return {
            'category': 'distressed',
            'tier': 'D',
            'base_score': 38,
            'peg': peg,
            'description': 'Distressed: Weak fundamentals, high risk'
        }
    
    # Default: Below Average
    return {
        'category': 'below_average',
        'tier': 'C',
        'base_score': 48,
        'peg': peg,
        'description': 'Below average: Mixed fundamentals'
    }


# ===========================================================================
# CATEGORICAL ADJUSTMENT FACTORS
# ===========================================================================

def compute_peg_multiplier(peg: float) -> float:
    """Compute PEG-based valuation multiplier."""
    if peg < 0.5:
        return 1.20
    elif peg < 1.0:
        return 1.15 - (peg - 0.5) * 0.30
    elif peg < 2.0:
        return 1.0 - (peg - 1.0) * 0.05
    elif peg < 3.0:
        return 0.95 - (peg - 2.0) * 0.10
    else:
        return 0.85


def compute_quality_momentum(df: pd.DataFrame, window: int = 60) -> float:
    """Compute quality trend (improving vs declining fundamentals)."""
    roe = df['roe']
    
    if len(roe) < window:
        return 1.0
    
    roe_recent = roe.iloc[-20:].mean()
    roe_past = roe.iloc[-window:-20].mean()
    roe_change = roe_recent - roe_past
    
    if roe_change > 10:
        return 1.08
    elif roe_change > 5:
        return 1.04
    elif roe_change < -10:
        return 0.92
    elif roe_change < -5:
        return 0.96
    else:
        return 1.0


def compute_financial_health_multiplier(de: float, current_ratio: Optional[float] = None) -> float:
    """Compute financial health multiplier based on leverage."""
    if de < 0.2:
        return 1.10
    elif de < 0.5:
        return 1.05
    elif de < 1.0:
        return 1.00
    elif de < 2.0:
        return 0.95
    else:
        return 0.90


def compute_margin_quality_bonus(profit_margin: Optional[float], operating_margin: Optional[float]) -> float:
    """Bonus for exceptional profitability margins."""
    if profit_margin is None:
        return 1.0
    
    if profit_margin > 40:
        return 1.10
    elif profit_margin > 30:
        return 1.05
    elif profit_margin > 20:
        return 1.02
    else:
        return 1.0


# ===========================================================================
# FEATURE TRANSFORMATION FUNCTIONS (STATISTICAL)
# ===========================================================================

def transform_pe(pe: float) -> float:
    """Transform P/E ratio to normalized score."""
    if pe <= 0:
        return 0.0
    return 1.0 / (1.0 + np.log1p(pe))


def transform_roe(roe: float) -> float:
    """Transform ROE to normalized score."""
    return np.tanh(roe / 50.0)


def transform_de(de: float) -> float:
    """Transform Debt/Equity to normalized score."""
    return np.exp(-de)


def transform_growth(growth: float) -> float:
    """Transform revenue growth to normalized score."""
    return np.tanh(growth / 30.0)


# ===========================================================================
# LOGISTIC LIKELIHOOD FUNCTION (STATISTICAL)
# ===========================================================================

def compute_logistic_likelihood(
    pe: float,
    roe: float,
    de: float,
    growth: Optional[float] = None,
    pca_weights: Optional[np.ndarray] = None,
    feature_vector: Optional[np.ndarray] = None,
) -> float:
    """
    Compute logistic likelihood function ℒⱼ.
    
    Formula:
        ℒⱼ = β₁·f₁(PE) + β₂·f₂(ROE) + β₃·f₃(D/E) + v^T·X
    """
    params = FScoreParameters()
    
    f1 = transform_pe(pe)
    f2 = transform_roe(roe)
    f3 = transform_de(de)
    
    likelihood = (
        params.BETA_PE * f1 +
        params.BETA_ROE * f2 +
        params.BETA_DE * f3
    )
    
    if growth is not None:
        f4 = transform_growth(growth)
        likelihood += 0.25 * f4
    
    if pca_weights is not None and feature_vector is not None:
        pca_component = np.dot(pca_weights, feature_vector)
        likelihood += 0.15 * pca_component
    
    return likelihood


# ===========================================================================
# PCA WEIGHTING (STATISTICAL)
# ===========================================================================

def compute_pca_weights(df: pd.DataFrame, n_components: int = 3) -> Tuple[np.ndarray, PCA]:
    """Compute PCA eigenvector weights for feature combination."""
    feature_cols = ['pe', 'roe', 'de']
    
    if 'revenue_growth_yoy' in df.columns:
        feature_cols.append('revenue_growth_yoy')
    
    X = df[feature_cols].dropna().values
    
    if len(X) < n_components:
        return np.ones(len(feature_cols)) / len(feature_cols), None
    
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_normalized = (X - X_mean) / X_std
    
    pca = PCA(n_components=min(n_components, X.shape[1]))
    pca.fit(X_normalized)
    
    eigenvector = pca.components_[0]
    
    return eigenvector, pca


# ===========================================================================
# DRIFT COMPUTATION (STATISTICAL)
# ===========================================================================

def compute_drift(likelihood_series: pd.Series, window: int = 60) -> pd.Series:
    """Compute cumulative drift D_F(t) = ∫₀ᵗ ∇F_j(τ) dτ"""
    gradient = likelihood_series.diff().fillna(0)
    drift = gradient.rolling(window=window, min_periods=1).sum()
    
    drift_std = drift.std()
    if drift_std > 0:
        drift = drift / drift_std
    
    return drift


# ===========================================================================
# CATEGORICAL F-SCORE (ORIGINAL PRODUCTION CODE)
# ===========================================================================

def compute_fundamental_score_universal(
    df: pd.DataFrame,
    config: Optional[FundamentalsConfig] = None,
    universe_df: Optional[pd.DataFrame] = None,
    return_components: bool = False,
) -> pd.DataFrame:
    """
    Universal Fundamental Score (F-Score) with absolute quality scoring.
    CATEGORICAL APPROACH (Production-ready)
    """
    required = ['pe', 'roe', 'de']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    result_df = df.copy()
    
    revenue_growth = df.get('revenue_growth_yoy', pd.Series(10.0, index=df.index))
    profit_margin = df.get('profit_margin', pd.Series(None, index=df.index))
    operating_margin = df.get('operating_margin', pd.Series(None, index=df.index))
    current_ratio = df.get('current_ratio', pd.Series(None, index=df.index))
    
    roe_latest = df['roe'].iloc[-1]
    growth_latest = revenue_growth.iloc[-1]
    pe_latest = df['pe'].iloc[-1]
    de_latest = df['de'].iloc[-1]
    profit_margin_latest = profit_margin.iloc[-1] if profit_margin.iloc[-1] is not None else None
    
    # Classify stock
    classification = classify_stock(
        roe=roe_latest,
        revenue_growth=growth_latest,
        pe=pe_latest,
        de=de_latest,
        profit_margin=profit_margin_latest
    )
    
    logger.info("="*70)
    logger.info("📊 STOCK CLASSIFICATION")
    logger.info("="*70)
    logger.info(f"Category:     {classification['category'].upper().replace('_', ' ')}")
    logger.info(f"Quality Tier: {classification['tier']}")
    logger.info(f"Base Score:   {classification['base_score']:.1f}/100")
    logger.info(f"PEG Ratio:    {classification['peg']:.2f}")
    logger.info(f"Description:  {classification['description']}")
    logger.info("="*70)
    
    # Compute multipliers
    peg_multiplier = compute_peg_multiplier(classification['peg'])
    quality_momentum = compute_quality_momentum(df, window=60)
    health_multiplier = compute_financial_health_multiplier(
        de_latest,
        current_ratio.iloc[-1] if len(current_ratio) > 0 else None
    )
    margin_bonus = compute_margin_quality_bonus(
        profit_margin_latest,
        operating_margin.iloc[-1] if len(operating_margin) > 0 else None
    )
    
    base_score = classification['base_score']
    
    f_score_latest = (
        base_score * 
        peg_multiplier * 
        quality_momentum * 
        health_multiplier * 
        margin_bonus
    )
    
    f_score_latest = np.clip(f_score_latest, 0, 95)
    
    n = len(df)
    time_decay = np.exp(-0.003 * np.arange(n-1, -1, -1))
    f_score_series = pd.Series(f_score_latest * time_decay, index=df.index)
    
    tier_floors = {'S': 85, 'A': 70, 'B': 55, 'C': 40, 'D': 25}
    floor = tier_floors.get(classification['tier'], 30)
    f_score_series = f_score_series.clip(lower=floor, upper=95)
    
    result_df['f_score'] = f_score_series
    
    logger.info("📈 F-SCORE COMPUTATION")
    logger.info(f"   Base Score:          {base_score:.1f}")
    logger.info(f"   PEG Multiplier:      {peg_multiplier:.3f}x (PEG={classification['peg']:.2f})")
    logger.info(f"   Quality Momentum:    {quality_momentum:.3f}x")
    logger.info(f"   Health Multiplier:   {health_multiplier:.3f}x (D/E={de_latest:.2f})")
    logger.info(f"   Margin Bonus:        {margin_bonus:.3f}x")
    logger.info(f"   ───────────────────────────────")
    logger.info(f"   Final F-Score:       {f_score_latest:.1f}%")
    logger.info(f"   Range (all periods): [{f_score_series.min():.1f}, {f_score_series.max():.1f}]")
    logger.info("="*70)
    
    if return_components:
        result_df['stock_category'] = classification['category']
        result_df['quality_tier'] = classification['tier']
        result_df['base_score'] = base_score
        result_df['peg_ratio'] = classification['peg']
        result_df['peg_multiplier'] = peg_multiplier
        result_df['quality_momentum'] = quality_momentum
        result_df['health_multiplier'] = health_multiplier
        result_df['margin_bonus'] = margin_bonus
    
    return result_df


# ===========================================================================
# STATISTICAL F-SCORE (IMAGE FORMULA)
# ===========================================================================

def compute_fundamental_score_statistical(
    df: pd.DataFrame,
    config: Optional[FundamentalsConfig] = None,
    universe_df: Optional[pd.DataFrame] = None,
    return_components: bool = False,
) -> pd.DataFrame:
    """
    Statistical Fundamental Score using image formula.
    F(t) = 100 · Φ((μ_F - X_F)/σ_F) · (1 - e^(-α·D_F(t)))
    """
    required = ['pe', 'roe', 'de']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    result_df = df.copy()
    params = FScoreParameters()
    
    pe = df['pe']
    roe = df['roe']
    de = df['de']
    growth = df.get('revenue_growth_yoy', None)
    
    logger.info("="*70)
    logger.info("📊 STATISTICAL F-SCORE COMPUTATION")
    logger.info("="*70)
    
    # PCA weights
    pca_weights, pca_model = compute_pca_weights(df, n_components=params.N_COMPONENTS)
    
    # Compute likelihood series
    likelihood_series = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        feature_vector = np.array([pe.iloc[i], roe.iloc[i], de.iloc[i]])
        
        if growth is not None:
            feature_vector = np.append(feature_vector, growth.iloc[i])
        
        likelihood = compute_logistic_likelihood(
            pe=pe.iloc[i],
            roe=roe.iloc[i],
            de=de.iloc[i],
            growth=growth.iloc[i] if growth is not None else None,
            pca_weights=pca_weights,
            feature_vector=feature_vector
        )
        
        likelihood_series.iloc[i] = likelihood
    
    # Statistical parameters
    mu_F = likelihood_series.rolling(window=params.VOL_WINDOW, min_periods=20).mean()
    sigma_F = likelihood_series.rolling(window=params.VOL_WINDOW, min_periods=20).std()
    sigma_F = sigma_F.clip(lower=0.1)
    
    X_F = likelihood_series
    drift = compute_drift(likelihood_series, window=60)
    
    # Z-score and CDF
    z_score = (mu_F - X_F) / sigma_F
    z_score = z_score.clip(lower=params.Z_LOWER, upper=params.Z_UPPER)
    cdf_value = stats.norm.cdf(z_score)
    
    # Decay factor
    decay_factor = 1.0 - np.exp(-params.ALPHA * np.abs(drift))
    decay_factor = decay_factor.clip(lower=0.3, upper=1.0)
    
    # Final F-Score
    f_score_series = 100.0 * cdf_value * decay_factor
    f_score_series = 100.0 - f_score_series  # Invert
    f_score_series = f_score_series.clip(lower=0, upper=95)
    
    result_df['f_score'] = f_score_series
    
    logger.info("="*70)
    logger.info("📈 F-SCORE RESULTS")
    logger.info("="*70)
    logger.info(f"   μ_F (Mean Likelihood):    {mu_F.iloc[-1]:.3f}")
    logger.info(f"   σ_F (Volatility):         {sigma_F.iloc[-1]:.3f}")
    logger.info(f"   Z-Score:                  {z_score.iloc[-1]:.3f}")
    logger.info(f"   Φ(z) [CDF]:               {cdf_value.iloc[-1]:.3f}")
    logger.info(f"   Final F-Score:            {f_score_series.iloc[-1]:.1f}%")
    logger.info("="*70)
    
    if return_components:
        result_df['likelihood'] = likelihood_series
        result_df['mu_F'] = mu_F
        result_df['sigma_F'] = sigma_F
        result_df['z_score'] = z_score
        result_df['cdf_value'] = cdf_value
        result_df['drift'] = drift
        result_df['decay_factor'] = decay_factor
    
    return result_df


# ===========================================================================
# EXPORTS
# ===========================================================================

__all__ = [
    # Production (categorical)
    'compute_fundamental_score_universal',
    
    # Research (statistical)
    'compute_fundamental_score_statistical',
    
    # Classification
    'classify_stock',
    
    # Multipliers
    'compute_peg_multiplier',
    'compute_quality_momentum',
    'compute_financial_health_multiplier',
    'compute_margin_quality_bonus',
    
    # Statistical utilities
    'compute_logistic_likelihood',
    'compute_pca_weights',
    'compute_drift',
]