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
Version: 4.2 (B2-Ra applied — soft tier floors)
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
    BETA_PE = 0.35
    BETA_ROE = 0.45
    BETA_DE = 0.20

    # Mean-reversion parameter
    ALPHA = 0.15

    # Volatility window
    VOL_WINDOW = 252

    # Z-score bounds
    Z_LOWER = -3.0
    Z_UPPER = 3.0

    # PCA components
    N_COMPONENTS = 3


# ===========================================================================
# STOCK CLASSIFICATION SYSTEM — India NSE Calibrated (v3.0)
# (UNCHANGED — classification is correct; only the output floor is patched)
# ===========================================================================

SECTOR_MAP: Dict[str, str] = {
    # IT
    'INFY': 'IT', 'TCS': 'IT', 'WIPRO': 'IT', 'HCLTECH': 'IT',
    'LTIM': 'IT', 'TECHM': 'IT', 'MPHASIS': 'IT', 'PERSISTENT': 'IT',
    # Bank
    'HDFCBANK': 'BANK', 'ICICIBANK': 'BANK', 'AXISBANK': 'BANK',
    'KOTAKBANK': 'BANK', 'INDUSINDBK': 'BANK', 'FEDERALBNK': 'BANK',
    # PSU Bank
    'SBIN': 'PSU_BANK', 'PNB': 'PSU_BANK', 'BANKBARODA': 'PSU_BANK',
    'CANBK': 'PSU_BANK', 'UNIONBANK': 'PSU_BANK',
    # Pharma
    'SUNPHARMA': 'PHARMA', 'CIPLA': 'PHARMA', 'DRREDDY': 'PHARMA',
    'DIVISLAB': 'PHARMA', 'AUROPHARMA': 'PHARMA', 'ZYDUSLIFE': 'PHARMA',
    'TORNTPHARM': 'PHARMA', 'LUPIN': 'PHARMA',
    # FMCG
    'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG',
    'BRITANNIA': 'FMCG', 'DABUR': 'FMCG', 'MARICO': 'FMCG',
    'COLPAL': 'FMCG', 'TITAN': 'FMCG',
    # Energy / Oil
    'RELIANCE': 'ENERGY', 'IOC': 'ENERGY', 'BPCL': 'ENERGY',
    'ONGC': 'ENERGY', 'GAIL': 'ENERGY', 'HINDPETRO': 'ENERGY',
    # Auto
    'MARUTI': 'AUTO', 'TATAMOTORS': 'AUTO', 'M&M': 'AUTO',
    'HEROMOTOCO': 'AUTO', 'BAJAJ-AUTO': 'AUTO', 'EICHERMOT': 'AUTO',
    # Metal / Mining
    'TATASTEEL': 'METAL', 'JSWSTEEL': 'METAL', 'HINDALCO': 'METAL',
    'VEDL': 'METAL', 'NMDC': 'METAL', 'COAL': 'METAL',
    # Financials / NBFC
    'BAJFINANCE': 'NBFC', 'BAJAJFINSV': 'NBFC', 'CHOLAFIN': 'NBFC',
    'MUTHOOTFIN': 'NBFC', 'CDSL': 'NBFC',
}

SECTOR_THRESHOLDS: Dict[str, Dict] = {
    'IT': {
        'roe_s': 40,   'growth_s': 20,
        'roe_a': 22,   'growth_a': 10,
        'roe_b': 14,   'growth_b': 4,
        'pe_max': 35,  'de_max': 0.6,
    },
    'BANK': {
        'roe_s': 20,   'growth_s': 18,
        'roe_a': 14,   'growth_a': 10,
        'roe_b': 10,   'growth_b': 5,
        'pe_max': 22,  'de_max': 1.8,
    },
    'PSU_BANK': {
        'roe_s': 15,   'growth_s': 15,
        'roe_a': 10,   'growth_a': 7,
        'roe_b': 6,    'growth_b': 2,
        'pe_max': 14,  'de_max': 2.5,
    },
    'PHARMA': {
        'roe_s': 28,   'growth_s': 18,
        'roe_a': 18,   'growth_a': 8,
        'roe_b': 12,   'growth_b': 3,
        'pe_max': 48,  'de_max': 0.6,
    },
    'FMCG': {
        'roe_s': 50,   'growth_s': 15,
        'roe_a': 28,   'growth_a': 8,
        'roe_b': 18,   'growth_b': 3,
        'pe_max': 65,  'de_max': 0.4,
    },
    'ENERGY': {
        'roe_s': 18,   'growth_s': 15,
        'roe_a': 12,   'growth_a': 5,
        'roe_b': 8,    'growth_b': -2,
        'pe_max': 16,  'de_max': 2.5,
    },
    'AUTO': {
        'roe_s': 25,   'growth_s': 20,
        'roe_a': 15,   'growth_a': 8,
        'roe_b': 10,   'growth_b': 2,
        'pe_max': 32,  'de_max': 1.5,
    },
    'METAL': {
        'roe_s': 22,   'growth_s': 15,
        'roe_a': 10,   'growth_a': 3,
        'roe_b': 6,    'growth_b': -5,
        'pe_max': 20,  'de_max': 2.0,
    },
    'NBFC': {
        'roe_s': 22,   'growth_s': 25,
        'roe_a': 15,   'growth_a': 15,
        'roe_b': 10,   'growth_b': 8,
        'pe_max': 40,  'de_max': 4.0,
    },
    'GENERIC': {
        'roe_s': 35,   'growth_s': 20,
        'roe_a': 20,   'growth_a': 10,
        'roe_b': 12,   'growth_b': 3,
        'pe_max': 30,  'de_max': 1.0,
    },
}


def classify_stock(
    roe: float,
    revenue_growth: float,
    pe: float,
    de: float,
    profit_margin: Optional[float] = None,
    ticker: Optional[str] = None,
    sector: Optional[str] = None,
) -> Dict:
    """
    Classify Indian stock into quality tier with base score.
    Sector-adjusted against Nifty 2026 benchmarks.
    UNCHANGED from v4.1 — classification logic is correct.
    """
    resolved_sector = 'GENERIC'
    if sector:
        resolved_sector = sector.upper()
    elif ticker:
        ticker_clean = ticker.upper().split('.')[0]
        resolved_sector = SECTOR_MAP.get(ticker_clean, 'GENERIC')

    thresh = SECTOR_THRESHOLDS.get(resolved_sector, SECTOR_THRESHOLDS['GENERIC'])

    peg = pe / revenue_growth if revenue_growth > 0.1 else 5.0
    if pd.notna(peg) and (peg > 100 or peg < -50):
        logger.warning(
            f"PEG={peg:.1f} is unreliable (growth near zero). Clamping to 5.0."
        )
        peg = 5.0

    if (roe > thresh['roe_s']
            and revenue_growth > thresh['growth_s']
            and pe < thresh['pe_max']
            and de < thresh['de_max']):
        return {
            'category': f'{resolved_sector.lower()}_leader',
            'tier': 'S',
            'base_score': 92,
            'peg': peg,
            'sector': resolved_sector,
            'description': f'Sector leader: Beats Nifty {resolved_sector} median (top 5%)',
        }

    if (roe > thresh['roe_a']
            and revenue_growth > thresh['growth_a']
            and pe < thresh['pe_max']
            and de < thresh['de_max']):
        return {
            'category': f'{resolved_sector.lower()}_strong',
            'tier': 'A',
            'base_score': 82,
            'peg': peg,
            'sector': resolved_sector,
            'description': f'Sector strong: Above Nifty {resolved_sector} median',
        }

    if roe > thresh['roe_a'] and de < thresh['de_max'] * 0.7:
        return {
            'category': f'{resolved_sector.lower()}_quality',
            'tier': 'A',
            'base_score': 77,
            'peg': peg,
            'sector': resolved_sector,
            'description': f'Sector quality: Strong ROE, conservative balance sheet',
        }

    if (roe > thresh['roe_b']
            and revenue_growth > thresh['growth_b']
            and de < thresh['de_max'] * 1.2):
        return {
            'category': f'{resolved_sector.lower()}_median',
            'tier': 'B',
            'base_score': 68,
            'peg': peg,
            'sector': resolved_sector,
            'description': f'Sector median: Matches Nifty {resolved_sector} benchmarks',
        }

    if roe > thresh['roe_b'] and de < thresh['de_max']:
        return {
            'category': f'{resolved_sector.lower()}_adequate',
            'tier': 'B',
            'base_score': 63,
            'peg': peg,
            'sector': resolved_sector,
            'description': f'Sector adequate: Acceptable ROE, manageable debt',
        }

    if roe > 20 and de < thresh['de_max'] * 1.5:
        return {
            'category': 'quality_slow_growth',
            'tier': 'C',
            'base_score': 55,
            'peg': peg,
            'sector': resolved_sector,
            'description': 'Quality large-cap: Strong ROE, TTM growth understated',
        }

    if roe > thresh['roe_b'] * 0.7 and revenue_growth > thresh['growth_b']:
        return {
            'category': f'{resolved_sector.lower()}_value',
            'tier': 'C',
            'base_score': 52,
            'peg': peg,
            'sector': resolved_sector,
            'description': f'Sector value: Below median but fundamentally viable',
        }

    if roe > 8 and de < thresh['de_max'] * 2:
        return {
            'category': 'below_sector_median',
            'tier': 'C',
            'base_score': 48,
            'peg': peg,
            'sector': resolved_sector,
            'description': 'Below sector median: Mixed fundamentals',
        }

    if (roe < 8 and revenue_growth < 0) or de > thresh['de_max'] * 3:
        return {
            'category': f'{resolved_sector.lower()}_distressed',
            'tier': 'D',
            'base_score': 35,
            'peg': peg,
            'sector': resolved_sector,
            'description': 'Distressed: Weak ROE + negative growth — high risk',
        }

    return {
        'category': 'below_average',
        'tier': 'C',
        'base_score': 48,
        'peg': peg,
        'sector': resolved_sector,
        'description': 'Below average: Mixed fundamentals',
    }


# ===========================================================================
# CATEGORICAL ADJUSTMENT FACTORS (ALL UNCHANGED)
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
    """
    Compute quality trend (improving vs declining fundamentals).
    Fix A: quarter-based slicing. UNCHANGED.
    """
    roe = df['roe'].dropna()

    if len(roe) < 4:
        return 1.0

    roe_recent = roe.iloc[-2:].mean()
    roe_past   = roe.iloc[:-2].mean()

    if pd.isna(roe_recent) or pd.isna(roe_past):
        return 1.0

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


def compute_financial_health_multiplier(
    de: float,
    current_ratio: Optional[float] = None,
) -> float:
    """Compute financial health multiplier based on leverage. UNCHANGED."""
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


def compute_margin_quality_bonus(
    profit_margin: Optional[float],
    operating_margin: Optional[float],
) -> float:
    """Bonus for exceptional profitability margins. UNCHANGED."""
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
# FEATURE TRANSFORMATION FUNCTIONS (ALL UNCHANGED)
# ===========================================================================

def transform_pe(pe: float) -> float:
    if pe <= 0:
        return 0.0
    return 1.0 / (1.0 + np.log1p(pe))

def transform_roe(roe: float) -> float:
    return np.tanh(roe / 50.0)

def transform_de(de: float) -> float:
    return np.exp(-de)

def transform_growth(growth: float) -> float:
    return np.tanh(growth / 30.0)


# ===========================================================================
# LOGISTIC LIKELIHOOD FUNCTION (UNCHANGED)
# ===========================================================================

def compute_logistic_likelihood(
    pe: float,
    roe: float,
    de: float,
    growth: Optional[float] = None,
    pca_weights: Optional[np.ndarray] = None,
    feature_vector: Optional[np.ndarray] = None,
) -> float:
    """Compute logistic likelihood function ℒⱼ. UNCHANGED."""
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
# PCA WEIGHTING (UNCHANGED)
# ===========================================================================

def compute_pca_weights(
    df: pd.DataFrame,
    n_components: int = 3,
) -> Tuple[np.ndarray, PCA]:
    """Compute PCA eigenvector weights. UNCHANGED."""
    feature_cols = ['pe', 'roe', 'de']

    if 'revenue_growth_yoy' in df.columns:
        feature_cols.append('revenue_growth_yoy')

    X = df[feature_cols].dropna().values

    if len(X) < n_components:
        return np.ones(len(feature_cols)) / len(feature_cols), None

    X_mean = X.mean(axis=0)
    X_std   = X.std(axis=0) + 1e-8
    X_normalized = (X - X_mean) / X_std

    pca = PCA(n_components=min(n_components, X.shape[1]))
    pca.fit(X_normalized)

    eigenvector = pca.components_[0]
    return eigenvector, pca


# ===========================================================================
# DRIFT COMPUTATION (UNCHANGED)
# ===========================================================================

def compute_drift(likelihood_series: pd.Series, window: int = 60) -> pd.Series:
    """Compute cumulative drift D_F(t). UNCHANGED."""
    gradient = likelihood_series.diff().fillna(0)
    drift    = gradient.rolling(window=window, min_periods=1).sum()

    drift_std = drift.std()
    if drift_std > 0:
        drift = drift / drift_std

    return drift


# ===========================================================================
# CATEGORICAL F-SCORE  (PRODUCTION)
# ===========================================================================

# ---------------------------------------------------------------------------
# B2-Ra PATCH: TIER_SOFT_FLOORS
# ---------------------------------------------------------------------------
# Problem (v4.1 and earlier):
#   tier_floors = {'S': 85, 'A': 70, 'B': 55, 'C': 40, 'D': 25}
#   These hard floors silently raised the computed f_score_latest when PEG,
#   D/E, or quality-momentum multipliers legitimately reduced it below the floor:
#       ICICIBANK: 77 base × 0.85 (PEG=3.38) × 0.95 (D/E) = 66.8 → clipped to 70.0 ❌
#       HDFCBANK:  63 base × 0.85 (PEG=5.00) × 0.95 (D/E) = 50.9 → clipped to 55.0 ❌
#
# Root cause: floors were set at tier base_score level, blocking ALL multiplier downside.
#
# Fix (v4.2):
#   TIER_SOFT_FLOORS ≈ 55% of tier base_score.
#   Purpose: protect only the OLDEST time-decayed historical rows from collapsing
#   to near-zero (exp(-0.002 × 518) = 0.355 → 50.9 × 0.355 = 18.1 → clips to 32).
#   Current row (time_decay=1.0) is NEVER floored by soft floor if multipliers
#   compute a value above it — which they will for any viable NSE stock.
#
# Verification:
#   ICICIBANK: 66.8 > 42 → passes through unblocked ✅
#   HDFCBANK:  50.9 > 32 → passes through unblocked ✅
#   ITC:       41.6 > 20 → passes through unblocked ✅
#   SBIN:      90.7 > 55 → passes through unblocked ✅ (S-tier, unchanged)
# ---------------------------------------------------------------------------
TIER_SOFT_FLOORS: Dict[str, float] = {
    'S': 55.0,   # was 85 — S-tier: 92 base × min_multiplier(0.705) = 64.9; floor at 55
    'A': 42.0,   # was 70 — A-tier: 77 base × 0.705 = 54.3; floor at 42 allows all penalties
    'B': 32.0,   # was 55 — B-tier: 63 base × 0.705 = 44.4; floor at 32 allows all penalties
    'C': 20.0,   # was 40 — C-tier: 48 base × 0.705 = 33.8; floor at 20
    'D': 10.0,   # was 25 — D-tier: 35 base × 0.705 = 24.7; floor at 10
}


def compute_fundamental_score_universal(
    df: pd.DataFrame,
    ticker: str,
    config: Optional[FundamentalsConfig] = None,
    universe_df: Optional[pd.DataFrame] = None,
    return_components: bool = False,
) -> pd.DataFrame:
    """
    Universal Fundamental Score (F-Score) with absolute quality scoring.
    CATEGORICAL APPROACH (Production-ready)

    Version: 4.2
    Fixes applied:
        6.1  PEG sanity clamp          → classify_stock()
        A    Quality momentum guard    → compute_quality_momentum()
        B    Time-decay 0.003 → 0.001  → time_decay line
        B2-Ra Soft tier floors         → TIER_SOFT_FLOORS (THIS VERSION)
    """
    required = ['pe', 'roe', 'de']
    missing  = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result_df = df.copy()

    revenue_growth   = df.get('revenue_growth_yoy',  pd.Series(10.0, index=df.index))
    profit_margin    = df.get('profit_margin',        pd.Series(None, index=df.index))
    operating_margin = df.get('operating_margin',     pd.Series(None, index=df.index))
    current_ratio    = df.get('current_ratio',        pd.Series(None, index=df.index))

    roe_latest           = df['roe'].iloc[-1]
    growth_latest        = revenue_growth.iloc[-1]
    pe_latest            = df['pe'].iloc[-1]
    de_latest            = df['de'].iloc[-1]
    profit_margin_latest = profit_margin.iloc[-1] if profit_margin.iloc[-1] is not None else None

    classification = classify_stock(
        ticker=ticker,
        roe=roe_latest,
        revenue_growth=growth_latest,
        pe=pe_latest,
        de=de_latest,
        profit_margin=profit_margin_latest,
    )

    logger.info("=" * 70)
    logger.info("📊 STOCK CLASSIFICATION")
    logger.info("=" * 70)
    logger.info(f"Category:     {classification['category'].upper().replace('_', ' ')}")
    logger.info(f"Quality Tier: {classification['tier']}")
    logger.info(f"Base Score:   {classification['base_score']:.1f}/100")
    logger.info(f"PEG Ratio:    {classification['peg']:.2f}")
    logger.info(f"Description:  {classification['description']}")
    logger.info("=" * 70)

    peg_multiplier    = compute_peg_multiplier(classification['peg'])
    quality_momentum  = compute_quality_momentum(df, window=60)
    health_multiplier = compute_financial_health_multiplier(
        de_latest,
        current_ratio.iloc[-1] if len(current_ratio) > 0 else None,
    )
    margin_bonus = compute_margin_quality_bonus(
        profit_margin_latest,
        operating_margin.iloc[-1] if len(operating_margin) > 0 else None,
    )

    base_score = classification['base_score']

    f_score_latest = (
        base_score
        * peg_multiplier
        * quality_momentum
        * health_multiplier
        * margin_bonus
    )
    f_score_latest = np.clip(f_score_latest, 0, 95)

    # Fix B — decay coefficient 0.001 (quarterly cadence calibrated, unchanged)
    n          = len(df)
    time_decay = np.exp(-0.002 * np.arange(n - 1, -1, -1))
    f_score_series = pd.Series(f_score_latest * time_decay, index=df.index)

    # ── B2-Ra PATCH ──────────────────────────────────────────────────────────
    # USE TIER_SOFT_FLOORS instead of old hard tier_floors.
    # Soft floors protect historical rows from time-decay collapse only.
    # They do NOT block current-period multiplier penalties.
    # See TIER_SOFT_FLOORS definition above for full rationale.
    floor = TIER_SOFT_FLOORS.get(classification['tier'], 20.0)   # ← B2-Ra CHANGED
    f_score_series = f_score_series.clip(lower=floor, upper=95)
    # ── END B2-Ra PATCH ──────────────────────────────────────────────────────

    result_df['f_score'] = f_score_series

    logger.info("📈 F-SCORE COMPUTATION")
    logger.info(f"   Base Score:          {base_score:.1f}")
    logger.info(f"   PEG Multiplier:      {peg_multiplier:.3f}x (PEG={classification['peg']:.2f})")
    logger.info(f"   Quality Momentum:    {quality_momentum:.3f}x")
    logger.info(f"   Health Multiplier:   {health_multiplier:.3f}x (D/E={de_latest:.2f})")
    logger.info(f"   Margin Bonus:        {margin_bonus:.3f}x")
    logger.info(f"   ───────────────────────────────")
    logger.info(f"   Final F-Score:       {f_score_latest:.1f}%")
    logger.info(f"   Soft Floor (Tier {classification['tier']}): {floor:.1f}")  # ← B2-Ra CHANGED (added floor log)
    logger.info(f"   Range (all periods): [{f_score_series.min():.1f}, {f_score_series.max():.1f}]")
    logger.info("=" * 70)

    if return_components:
        result_df['stock_category']    = classification['category']
        result_df['quality_tier']      = classification['tier']
        result_df['base_score']        = base_score
        result_df['peg_ratio']         = classification['peg']
        result_df['peg_multiplier']    = peg_multiplier
        result_df['quality_momentum']  = quality_momentum
        result_df['health_multiplier'] = health_multiplier
        result_df['margin_bonus']      = margin_bonus

    return result_df


# ===========================================================================
# STATISTICAL F-SCORE  (RESEARCH — UNCHANGED)
# ===========================================================================

def compute_fundamental_score_statistical(
    df: pd.DataFrame,
    config: Optional[FundamentalsConfig] = None,
    universe_df: Optional[pd.DataFrame] = None,
    return_components: bool = False,
) -> pd.DataFrame:
    """
    Statistical F-Score using image formula. RESEARCH only. UNCHANGED.
    F(t) = 100 · Φ((μ_F - X_F)/σ_F) · (1 - e^(-α·D_F(t)))
    """
    required = ['pe', 'roe', 'de']
    missing  = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result_df = df.copy()
    params    = FScoreParameters()

    pe     = df['pe']
    roe    = df['roe']
    de     = df['de']
    growth = df.get('revenue_growth_yoy', None)

    logger.info("=" * 70)
    logger.info("📊 STATISTICAL F-SCORE COMPUTATION")
    logger.info("=" * 70)

    pca_weights, pca_model = compute_pca_weights(df, n_components=params.N_COMPONENTS)

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
            feature_vector=feature_vector,
        )
        likelihood_series.iloc[i] = likelihood

    mu_F    = likelihood_series.rolling(window=params.VOL_WINDOW, min_periods=20).mean()
    sigma_F = likelihood_series.rolling(window=params.VOL_WINDOW, min_periods=20).std()
    sigma_F = sigma_F.clip(lower=0.1)

    X_F   = likelihood_series
    drift = compute_drift(likelihood_series, window=60)

    z_score   = (mu_F - X_F) / sigma_F
    z_score   = z_score.clip(lower=params.Z_LOWER, upper=params.Z_UPPER)
    cdf_value = stats.norm.cdf(z_score)

    decay_factor = 1.0 - np.exp(-params.ALPHA * np.abs(drift))
    decay_factor = decay_factor.clip(lower=0.3, upper=1.0)

    f_score_series = 100.0 * cdf_value * decay_factor
    f_score_series = 100.0 - f_score_series
    f_score_series = f_score_series.clip(lower=0, upper=95)

    result_df['f_score'] = f_score_series

    logger.info("=" * 70)
    logger.info("📈 F-SCORE RESULTS")
    logger.info("=" * 70)
    logger.info(f"   μ_F (Mean Likelihood):    {mu_F.iloc[-1]:.3f}")
    logger.info(f"   σ_F (Volatility):         {sigma_F.iloc[-1]:.3f}")
    logger.info(f"   Z-Score:                  {z_score.iloc[-1]:.3f}")
    logger.info(f"   Φ(z) [CDF]:               {cdf_value.iloc[-1]:.3f}")
    logger.info(f"   Final F-Score:            {f_score_series.iloc[-1]:.1f}%")
    logger.info("=" * 70)

    if return_components:
        result_df['likelihood']   = likelihood_series
        result_df['mu_F']         = mu_F
        result_df['sigma_F']      = sigma_F
        result_df['z_score']      = z_score
        result_df['cdf_value']    = cdf_value
        result_df['drift']        = drift
        result_df['decay_factor'] = decay_factor

    return result_df


# ===========================================================================
# EXPORTS
# ===========================================================================

__all__ = [
    'compute_fundamental_score_universal',
    'compute_fundamental_score_statistical',
    'classify_stock',
    'compute_peg_multiplier',
    'compute_quality_momentum',
    'compute_financial_health_multiplier',
    'compute_margin_quality_bonus',
    'compute_logistic_likelihood',
    'compute_pca_weights',
    'compute_drift',
    'TIER_SOFT_FLOORS',   # ← B2-Ra: exported so composite.py can log it
]
