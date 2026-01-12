"""
ASRE Universal Fundamental Score (F-Score)
Production formula with absolute quality scoring and categorical classification.

Handles:
- Exceptional growth stocks (NVDA, META)
- Quality growth (MSFT, GOOGL)
- Cash cows (AAPL, JNJ)
- Value stocks (KO, T)
- Distressed companies

Author: ASRE Rating System
Version: 3.0 (Universal)
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from asre.config import FundamentalsConfig

logger = logging.getLogger(__name__)


# ===========================================================================
# STOCK CLASSIFICATION SYSTEM (S/A/B/C/D Tiers)
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
    
    # -----------------------------------------------------------------------
    # S-TIER: Exceptional Growth (Top 1% companies)
    # -----------------------------------------------------------------------
    # Criteria: ROE > 80%, Growth > 50%, PEG < 1.0, Low Debt
    # Examples: NVDA, META, TSLA (peak), AMD (peak)
    if roe > 80 and revenue_growth > 50 and peg < 1.0 and de < 0.5:
        return {
            'category': 'exceptional_growth',
            'tier': 'S',
            'base_score': 92,
            'peg': peg,
            'description': 'World-class: Exceptional growth + profitability + undervalued'
        }
    
    # -----------------------------------------------------------------------
    # A-TIER: High-Quality Growth
    # -----------------------------------------------------------------------
    # Criteria: ROE > 50%, Growth > 30%, PEG < 1.5
    # Examples: NVDA (moderate), GOOGL (strong periods)
    if roe > 50 and revenue_growth > 30 and peg < 1.5 and de < 0.8:
        return {
            'category': 'high_quality_growth',
            'tier': 'A',
            'base_score': 82,
            'peg': peg,
            'description': 'High-quality growth: Strong across all metrics'
        }
    
    # Criteria: ROE > 30%, Growth > 15%, PEG < 2.0
    # Examples: MSFT, GOOGL, META (normal), V, MA
    if roe > 30 and revenue_growth > 15 and peg < 2.5:
        return {
            'category': 'quality_growth',
            'tier': 'A',
            'base_score': 77,
            'peg': peg,
            'description': 'Quality growth: Balanced strength in growth + profitability'
        }
    
    # Criteria: ROE > 100%, Growth < 15%, High Debt
    # Examples: AAPL (mature phase), mature tech with buybacks
    if roe > 100 and de > 1.0:
        return {
            'category': 'cash_cow_leveraged',
            'tier': 'A',
            'base_score': 73,
            'peg': peg,
            'description': 'Cash cow: Exceptional ROE, modest growth, leveraged'
        }
    
    # -----------------------------------------------------------------------
    # B-TIER: Solid Quality
    # -----------------------------------------------------------------------
    # Criteria: ROE > 20%, Growth > 10%
    # Examples: UNH, LLY, COST
    if roe > 20 and revenue_growth > 10 and de < 1.5:
        return {
            'category': 'solid_growth',
            'tier': 'B',
            'base_score': 68,
            'peg': peg,
            'description': 'Solid growth: Above-average fundamentals'
        }
    
    # Criteria: ROE > 15%, Low debt, Stable
    # Examples: JNJ, PG, WMT
    if roe > 15 and de < 0.8:
        return {
            'category': 'balanced_quality',
            'tier': 'B',
            'base_score': 63,
            'peg': peg,
            'description': 'Balanced: Solid fundamentals, low risk'
        }
    
    # -----------------------------------------------------------------------
    # C-TIER: Fair Value
    # -----------------------------------------------------------------------
    # Criteria: ROE > 10%, Cheap valuation
    # Examples: KO, PEP, T, VZ
    if roe > 10 and pe < 20:
        return {
            'category': 'value',
            'tier': 'C',
            'base_score': 55,
            'peg': peg,
            'description': 'Value: Undervalued but limited growth'
        }
    
    # Criteria: ROE > 10%, Growth > 5%
    # Examples: Mature industrials, utilities
    if roe > 10 and revenue_growth > 5:
        return {
            'category': 'stable',
            'tier': 'C',
            'base_score': 52,
            'peg': peg,
            'description': 'Stable: Modest growth and profitability'
        }
    
    # -----------------------------------------------------------------------
    # D-TIER: Weak/Distressed
    # -----------------------------------------------------------------------
    # Criteria: Low ROE or negative growth
    # Examples: Turnarounds, cyclicals in downturn
    if roe < 10 or revenue_growth < 0:
        return {
            'category': 'distressed',
            'tier': 'D',
            'base_score': 38,
            'peg': peg,
            'description': 'Distressed: Weak fundamentals, high risk'
        }
    
    # -----------------------------------------------------------------------
    # Default: Below Average
    # -----------------------------------------------------------------------
    return {
        'category': 'below_average',
        'tier': 'C',
        'base_score': 48,
        'peg': peg,
        'description': 'Below average: Mixed fundamentals'
    }


# ===========================================================================
# ADJUSTMENT FACTORS
# ===========================================================================

def compute_peg_multiplier(peg: float) -> float:
    """
    Compute PEG-based valuation multiplier.
    
    PEG Ratio Interpretation:
        < 0.5: Deeply undervalued (1.20x multiplier)
        0.5-1.0: Undervalued (1.00-1.15x)
        1.0-2.0: Fair value (0.95-1.00x)
        > 2.0: Overvalued (0.85-0.95x)
    """
    if peg < 0.5:
        return 1.20  # 20% boost
    elif peg < 1.0:
        # Linear from 1.15 at PEG=0.5 to 1.0 at PEG=1.0
        return 1.15 - (peg - 0.5) * 0.30
    elif peg < 2.0:
        # Linear from 1.0 at PEG=1.0 to 0.95 at PEG=2.0
        return 1.0 - (peg - 1.0) * 0.05
    elif peg < 3.0:
        # Linear from 0.95 at PEG=2.0 to 0.85 at PEG=3.0
        return 0.95 - (peg - 2.0) * 0.10
    else:
        return 0.85  # 15% penalty for very overvalued


def compute_quality_momentum(df: pd.DataFrame, window: int = 60) -> float:
    """
    Compute quality trend (improving vs declining fundamentals).
    
    Returns multiplier: 1.05 (improving) to 0.95 (declining)
    """
    roe = df['roe']
    
    if len(roe) < window:
        return 1.0
    
    # ROE trend over window
    roe_recent = roe.iloc[-20:].mean()
    roe_past = roe.iloc[-window:-20].mean()
    
    roe_change = roe_recent - roe_past
    
    if roe_change > 10:
        return 1.08  # Strong improvement
    elif roe_change > 5:
        return 1.04  # Moderate improvement
    elif roe_change < -10:
        return 0.92  # Strong decline
    elif roe_change < -5:
        return 0.96  # Moderate decline
    else:
        return 1.0  # Stable


def compute_financial_health_multiplier(de: float, current_ratio: Optional[float] = None) -> float:
    """
    Compute financial health multiplier based on leverage.
    
    Returns:
        1.05-1.10: Pristine balance sheet (D/E < 0.3)
        1.00: Normal (D/E 0.3-1.0)
        0.95-0.90: High leverage (D/E > 1.0)
    """
    if de < 0.2:
        return 1.10  # Fortress balance sheet
    elif de < 0.5:
        return 1.05  # Very strong
    elif de < 1.0:
        return 1.00  # Healthy
    elif de < 2.0:
        return 0.95  # Moderate leverage
    else:
        return 0.90  # High leverage risk


def compute_margin_quality_bonus(profit_margin: Optional[float], operating_margin: Optional[float]) -> float:
    """
    Bonus for exceptional profitability margins.
    
    Returns multiplier: 1.00-1.10
    """
    if profit_margin is None:
        return 1.0
    
    if profit_margin > 40:  # NVDA, META level
        return 1.10
    elif profit_margin > 30:  # AAPL, GOOGL level
        return 1.05
    elif profit_margin > 20:
        return 1.02
    else:
        return 1.0


# ===========================================================================
# MAIN UNIVERSAL F-SCORE FUNCTION
# ===========================================================================

def compute_fundamental_score_universal(
    df: pd.DataFrame,
    config: Optional[FundamentalsConfig] = None,           # ✅ ADD THIS
    universe_df: Optional[pd.DataFrame] = None,            # ✅ ADD THIS
    return_components: bool = False,
) -> pd.DataFrame:
    """
    Universal Fundamental Score (F-Score) with absolute quality scoring.
    
    ✅ Features:
    - Category-based scoring (S/A/B/C/D tiers)
    - PEG ratio adjustments
    - Quality momentum tracking
    - Financial health analysis
    - Handles all stock types correctly
    
    Args:
        df: DataFrame with columns [pe, roe, de, revenue_growth_yoy, profit_margin, ...]
        return_components: Return detailed breakdown
    
    Returns:
        DataFrame with f_score column (0-95 scale)
    """
    # -----------------------------------------------------------------------
    # Validate Input
    # -----------------------------------------------------------------------
    required = ['pe', 'roe', 'de']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    result_df = df.copy()
    
    # Extract metrics
    revenue_growth = df.get('revenue_growth_yoy', pd.Series(10.0, index=df.index))
    profit_margin = df.get('profit_margin', pd.Series(None, index=df.index))
    operating_margin = df.get('operating_margin', pd.Series(None, index=df.index))
    current_ratio = df.get('current_ratio', pd.Series(None, index=df.index))
    
    # Latest values for classification
    roe_latest = df['roe'].iloc[-1]
    growth_latest = revenue_growth.iloc[-1]
    pe_latest = df['pe'].iloc[-1]
    de_latest = df['de'].iloc[-1]
    profit_margin_latest = profit_margin.iloc[-1] if profit_margin.iloc[-1] is not None else None
    
    # -----------------------------------------------------------------------
    # Step 1: Classify Stock
    # -----------------------------------------------------------------------
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
    
    # -----------------------------------------------------------------------
    # Step 2: Compute Adjustment Multipliers
    # -----------------------------------------------------------------------
    
    # PEG valuation adjustment
    peg_multiplier = compute_peg_multiplier(classification['peg'])
    
    # Quality momentum (improving/declining)
    quality_momentum = compute_quality_momentum(df, window=60)
    
    # Financial health
    health_multiplier = compute_financial_health_multiplier(
        de_latest,
        current_ratio.iloc[-1] if len(current_ratio) > 0 else None
    )
    
    # Margin quality bonus
    margin_bonus = compute_margin_quality_bonus(
        profit_margin_latest,
        operating_margin.iloc[-1] if len(operating_margin) > 0 else None
    )
    
    # -----------------------------------------------------------------------
    # Step 3: Compute Final F-Score
    # -----------------------------------------------------------------------
    
    base_score = classification['base_score']
    
    # Apply all multipliers
    f_score_latest = (
        base_score * 
        peg_multiplier * 
        quality_momentum * 
        health_multiplier * 
        margin_bonus
    )
    
    # Clip to valid range
    f_score_latest = np.clip(f_score_latest, 0, 95)
    
    # Create time series with slight decay for historical values
    # (Recent data more reliable than old data)
    n = len(df)
    time_decay = np.exp(-0.003 * np.arange(n-1, -1, -1))  # 1.0 recent → 0.75 old
    f_score_series = pd.Series(f_score_latest * time_decay, index=df.index)
    
    # Apply floor based on tier
    tier_floors = {'S': 85, 'A': 70, 'B': 55, 'C': 40, 'D': 25}
    floor = tier_floors.get(classification['tier'], 30)
    f_score_series = f_score_series.clip(lower=floor, upper=95)
    
    result_df['f_score'] = f_score_series
    
    # -----------------------------------------------------------------------
    # Step 4: Logging
    # -----------------------------------------------------------------------
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
    
    # -----------------------------------------------------------------------
    # Step 5: Store Components (Optional)
    # -----------------------------------------------------------------------
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
# EXPORTS
# ===========================================================================

__all__ = [
    'compute_fundamental_score_universal',
    'classify_stock',
    'compute_peg_multiplier',
    'compute_quality_momentum',
    'compute_financial_health_multiplier',
]
