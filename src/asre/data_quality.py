"""
Data Quality Assessment Module
Detects incomplete/unreliable fundamental data for Indian stocks
"""

import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def assess_fundamental_data_quality(
    fundamental_df: pd.DataFrame,
    ticker: str,
    required_quarters: int = 6,
) -> Dict:
    """
    Comprehensive data quality assessment for fundamentals.
    
    Returns:
        Dict with:
            - overall_score: 0.0-1.0 quality score
            - completeness_score: Quarter availability
            - revenue_quality: Revenue data quality
            - metric_availability: Key metric presence
            - recency_score: Data freshness
            - issues: List of detected issues
            - warnings: List of warnings
    """
    
    if fundamental_df is None or len(fundamental_df) == 0:
        return {
            'overall_score': 0.0,
            'completeness_score': 0.0,
            'revenue_quality': 0.0,
            'metric_availability': 0.0,
            'recency_score': 0.0,
            'issues': ['No fundamental data available'],
            'warnings': [f'Zero quarters fetched for {ticker}'],
        }
    
    issues = []
    warnings = []
    
    # ========================================================================
    # CHECK 1: Quarter Completeness
    # ========================================================================
    
    quarters_present = len(fundamental_df)
    quarter_completeness = min(1.0, quarters_present / required_quarters)
    
    if quarters_present < required_quarters:
        issues.append(f"Only {quarters_present}/{required_quarters} quarters available")
    
    if quarters_present < 4:
        warnings.append(f"Insufficient historical data ({quarters_present} quarters)")
    
    # ========================================================================
    # CHECK 2: Revenue Data Quality
    # ========================================================================
    
    revenue_cols = ['revenue', 'total_revenue', 'totalRevenue', 'Total Revenue']
    revenue_col = None
    
    for col in revenue_cols:
        if col in fundamental_df.columns:
            revenue_col = col
            break
    
    if revenue_col is None:
        revenue_quality = 0.0
        issues.append("No revenue column found")
    else:
        revenue_data = fundamental_df[revenue_col]
        
        # Check positive revenue
        positive_revenue_count = (revenue_data > 0).sum()
        revenue_quality = positive_revenue_count / len(revenue_data)
        
        if revenue_quality < 0.8:
            issues.append(
                f"Revenue data incomplete: {positive_revenue_count}/{len(revenue_data)} quarters"
            )
        
        # Check for anomalies (sudden 10x changes)
        if len(revenue_data) > 1:
            revenue_changes = revenue_data.pct_change().abs()
            extreme_changes = (revenue_changes > 2.0).sum()  # >200% change
            
            if extreme_changes > 0:
                warnings.append(
                    f"Detected {extreme_changes} extreme revenue changes (possible data error)"
                )
    
    # ========================================================================
    # CHECK 3: Key Metrics Availability
    # ========================================================================
    
    required_metrics = ['roe', 'pe', 'de', 'revenue_growth_yoy']
    optional_metrics = ['profit_margin', 'operating_margin', 'current_ratio']
    
    available_required = sum(1 for metric in required_metrics if metric in fundamental_df.columns)
    available_optional = sum(1 for metric in optional_metrics if metric in fundamental_df.columns)
    
    metric_availability = available_required / len(required_metrics)
    
    missing_required = [m for m in required_metrics if m not in fundamental_df.columns]
    if missing_required:
        issues.append(f"Missing required metrics: {', '.join(missing_required)}")
    
    if available_optional < 2:
        warnings.append(f"Limited optional metrics available ({available_optional}/{len(optional_metrics)})")
    
    # ========================================================================
    # CHECK 4: Data Recency
    # ========================================================================
    
    date_cols = ['date', 'announcement_date', 'quarter_end_date', 'asOfDate']
    date_col = None
    
    for col in date_cols:
        if col in fundamental_df.columns:
            date_col = col
            break
    
    if date_col is None:
        recency_score = 0.7  # Unknown, assume reasonable
        warnings.append("No date column found, cannot verify data freshness")
    else:
        try:
            latest_date = pd.to_datetime(fundamental_df[date_col].max())
            days_since_update = (pd.Timestamp.now() - latest_date).days
            
            if days_since_update < 90:
                recency_score = 1.0
            elif days_since_update < 120:
                recency_score = 0.9
            elif days_since_update < 180:
                recency_score = 0.7
                warnings.append(f"Data is {days_since_update} days old")
            else:
                recency_score = max(0.3, 1.0 - (days_since_update - 180) / 365)
                issues.append(f"Stale data: {days_since_update} days old")
                
        except Exception as e:
            recency_score = 0.7
            warnings.append(f"Could not parse dates: {str(e)}")
    
    # ========================================================================
    # CHECK 5: India-Specific Issues (Yahoo Finance Known Problems)
    # ========================================================================
    
    is_indian = ticker.endswith('.NS') or ticker.endswith('.BO')
    
    if is_indian:
        # Known Yahoo Finance issues for Indian stocks
        if quarters_present < 6:
            warnings.append(
                "Yahoo Finance has limited Indian fundamental coverage - "
                "consider using NSE data source"
            )
        
        # Check for consolidated vs standalone mismatch
        if 'financial_type' in fundamental_df.columns:
            financial_types = fundamental_df['financial_type'].unique()
            if len(financial_types) > 1:
                warnings.append(
                    f"Mixed financial statement types detected: {financial_types} "
                    "(consolidated vs standalone)"
                )
    
    # ========================================================================
    # COMPUTE OVERALL SCORE
    # ========================================================================
    
    overall_score = (
        quarter_completeness * 0.35 +
        revenue_quality * 0.30 +
        metric_availability * 0.20 +
        recency_score * 0.15
    )
    
    overall_score = np.clip(overall_score, 0.0, 1.0)
    
    # ========================================================================
    # RETURN ASSESSMENT
    # ========================================================================
    
    return {
        'overall_score': overall_score,
        'completeness_score': quarter_completeness,
        'revenue_quality': revenue_quality,
        'metric_availability': metric_availability,
        'recency_score': recency_score,
        'quarters_available': quarters_present,
        'quarters_required': required_quarters,
        'issues': issues,
        'warnings': warnings,
        'recommendation': _get_recommendation(overall_score, is_indian),
    }


def _get_recommendation(score: float, is_indian: bool) -> str:
    """Get actionable recommendation based on data quality score."""
    
    if score >= 0.9:
        return "✅ Excellent data quality - safe to use for production"
    elif score >= 0.8:
        return "✅ Good data quality - usable with standard safeguards"
    elif score >= 0.6:
        return "⚠️ Moderate data quality - use with caution, apply conservative ratings"
    elif score >= 0.4:
        if is_indian:
            return "❌ Poor data quality - recommend switching to NSE data source"
        else:
            return "❌ Poor data quality - verify ticker symbol and data source"
    else:
        return "❌ Critical data issues - DO NOT USE for production trading"
