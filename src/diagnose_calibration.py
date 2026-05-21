"""
ASRE Calibration Diagnostics
=============================
Run this to understand WHY your calibration is failing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
import seaborn as sns

def diagnose_calibration_issues(scores_df, returns_series):
    """
    Comprehensive diagnostics for calibration failure.
    
    Args:
        scores_df: DataFrame with columns ['f_score', 't_score', 'm_score']
        returns_series: Series of next-day returns (aligned with scores_df)
    """
    
    print("="*70)
    print("ASRE CALIBRATION DIAGNOSTICS")
    print("="*70)
    
    # ===== 1. FEATURE-TARGET CORRELATION ANALYSIS =====
    print("\n1. FEATURE-TARGET CORRELATION")
    print("-" * 70)
    
    binary_target = (returns_series > 0).astype(int)
    
    correlations = {}
    for col in ['f_score', 't_score', 'm_score']:
        # Pearson (linear)
        pearson_r, pearson_p = pearsonr(
            scores_df[col].dropna(), 
            returns_series.reindex(scores_df[col].dropna().index).dropna()
        )
        
        # Spearman (monotonic)
        spearman_r, spearman_p = spearmanr(
            scores_df[col].dropna(), 
            returns_series.reindex(scores_df[col].dropna().index).dropna()
        )
        
        # Point-biserial (binary target)
        valid_idx = ~(scores_df[col].isna() | binary_target.isna())
        try:
            auc = roc_auc_score(
                binary_target[valid_idx], 
                scores_df[col][valid_idx]
            )
        except:
            auc = 0.5
        
        correlations[col] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'binary_auc': auc
        }
        
        print(f"\n{col.upper()}:")
        print(f"  Pearson r:  {pearson_r:+.4f} (p={pearson_p:.4f}) {'***' if pearson_p < 0.001 else '**' if pearson_p < 0.01 else '*' if pearson_p < 0.05 else 'n.s.'}")
        print(f"  Spearman ρ: {spearman_r:+.4f} (p={spearman_p:.4f}) {'***' if spearman_p < 0.001 else '**' if spearman_p < 0.01 else '*' if spearman_p < 0.05 else 'n.s.'}")
        print(f"  Binary AUC: {auc:.4f} {'(usable)' if auc > 0.55 else '(weak)' if auc > 0.52 else '(random)'}")
    
    # ===== 2. SCORE DISTRIBUTION ANALYSIS =====
    print("\n\n2. SCORE DISTRIBUTION ANALYSIS")
    print("-" * 70)
    
    for col in ['f_score', 't_score', 'm_score']:
        vals = scores_df[col].dropna()
        print(f"\n{col.upper()}:")
        print(f"  Mean:   {vals.mean():.2f}")
        print(f"  Std:    {vals.std():.2f}")
        print(f"  Min:    {vals.min():.2f}")
        print(f"  Max:    {vals.max():.2f}")
        print(f"  Range:  {vals.max() - vals.min():.2f}")
        print(f"  CV:     {vals.std() / vals.mean():.4f} (coef. of variation)")
        
        # Check for clustering/saturation
        q25, q50, q75 = vals.quantile([0.25, 0.50, 0.75])
        iqr = q75 - q25
        print(f"  IQR:    {iqr:.2f} (25%={q25:.2f}, 75%={q75:.2f})")
        
        if iqr < 5:
            print(f"  ⚠️  WARNING: Very narrow IQR suggests score saturation/clustering")
    
    # ===== 3. TEMPORAL STABILITY =====
    print("\n\n3. TEMPORAL STABILITY (Rolling Correlation)")
    print("-" * 70)
    
    window = 60  # 60-day rolling window
    for col in ['f_score', 't_score', 'm_score']:
        rolling_corr = []
        for i in range(window, len(scores_df)):
            window_scores = scores_df[col].iloc[i-window:i]
            window_returns = returns_series.iloc[i-window:i]
            valid = ~(window_scores.isna() | window_returns.isna())
            
            if valid.sum() > 30:
                corr, _ = spearmanr(window_scores[valid], window_returns[valid])
                rolling_corr.append(corr)
            else:
                rolling_corr.append(np.nan)
        
        rolling_corr = pd.Series(rolling_corr)
        print(f"\n{col.upper()} (60-day rolling Spearman):")
        print(f"  Mean correlation:  {rolling_corr.mean():+.4f}")
        print(f"  Std correlation:   {rolling_corr.std():.4f}")
        print(f"  % positive:        {(rolling_corr > 0).sum() / len(rolling_corr) * 100:.1f}%")
        print(f"  % above 0.1:       {(rolling_corr > 0.1).sum() / len(rolling_corr) * 100:.1f}%")
        
        if rolling_corr.std() > 0.15:
            print(f"  ⚠️  WARNING: High correlation variance suggests regime instability")
    
    # ===== 4. COMPOSITE SCORE ANALYSIS =====
    print("\n\n4. COMPOSITE SCORE ANALYSIS (Default Weights)")
    print("-" * 70)
    
    composite = (
        0.4 * scores_df['f_score'] + 
        0.3 * scores_df['t_score'] + 
        0.3 * scores_df['m_score']
    )
    
    valid_idx = ~(composite.isna() | returns_series.isna())
    
    # Correlation
    pearson_r, pearson_p = pearsonr(composite[valid_idx], returns_series[valid_idx])
    spearman_r, spearman_p = spearmanr(composite[valid_idx], returns_series[valid_idx])
    
    # Binary AUC
    binary_target_valid = binary_target[valid_idx]
    try:
        comp_auc = roc_auc_score(binary_target_valid, composite[valid_idx])
    except:
        comp_auc = 0.5
    
    print(f"Composite Score (0.4F + 0.3T + 0.3M):")
    print(f"  Pearson r:  {pearson_r:+.4f} (p={pearson_p:.4f})")
    print(f"  Spearman ρ: {spearman_r:+.4f} (p={spearman_p:.4f})")
    print(f"  Binary AUC: {comp_auc:.4f}")
    print(f"  Mean:       {composite.mean():.2f}")
    print(f"  Std:        {composite.std():.2f}")
    print(f"  Variance:   {composite.var():.2f}")
    
    if composite.var() < 8.0:
        print(f"  ⚠️  CRITICAL: Composite variance < 8.0 (your threshold)")
        print(f"     This explains why logistic fits are skipped!")
    
    # ===== 5. RETURN DISTRIBUTION =====
    print("\n\n5. RETURN DISTRIBUTION ANALYSIS")
    print("-" * 70)
    
    returns_clean = returns_series.dropna()
    print(f"Next-day returns:")
    print(f"  Mean:      {returns_clean.mean()*100:+.4f}%")
    print(f"  Std:       {returns_clean.std()*100:.4f}%")
    print(f"  Sharpe:    {returns_clean.mean() / returns_clean.std():.4f}")
    print(f"  % positive: {(returns_clean > 0).sum() / len(returns_clean) * 100:.1f}%")
    print(f"  Skewness:  {returns_clean.skew():.4f}")
    print(f"  Kurtosis:  {returns_clean.kurt():.4f}")
    
    if abs((returns_clean > 0).sum() / len(returns_clean) - 0.5) < 0.02:
        print(f"  ⚠️  WARNING: Returns are nearly 50/50 — very hard to predict!")
    
    # ===== 6. RECOMMENDATIONS =====
    print("\n\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    issues_found = []
    
    # Check individual feature AUCs
    weak_features = [col for col, stats in correlations.items() 
                     if stats['binary_auc'] < 0.55]
    if len(weak_features) == 3:
        issues_found.append("ALL_FEATURES_WEAK")
        print("\n❌ CRITICAL: All three features have AUC < 0.55")
        print("   → Your F/T/M scores have no predictive power for binary returns")
        print("   → Logistic calibration CANNOT fix this — you need better features")
    
    # Check composite variance
    if composite.var() < 8.0:
        issues_found.append("COMPOSITE_FLAT")
        print("\n❌ CRITICAL: Composite variance below threshold")
        print("   → Even with variation in individual scores, composite is flat")
        print("   → This triggers your B2-Rb guard correctly")
    
    # Check if features are redundant
    f_t_corr = scores_df[['f_score', 't_score']].corr().iloc[0, 1]
    f_m_corr = scores_df[['f_score', 'm_score']].corr().iloc[0, 1]
    t_m_corr = scores_df[['t_score', 'm_score']].corr().iloc[0, 1]
    
    if min(f_t_corr, f_m_corr, t_m_corr) > 0.7:
        issues_found.append("HIGH_MULTICOLLINEARITY")
        print(f"\n⚠️  WARNING: High feature correlation")
        print(f"   F-T: {f_t_corr:.3f}, F-M: {f_m_corr:.3f}, T-M: {t_m_corr:.3f}")
        print("   → Features are redundant — adds noise to weight optimization")
    
    if not issues_found:
        print("\n✓ No critical issues found — calibration should work")
        print("  However, AUC 0.52-0.60 suggests weak signal overall")
    
    return correlations, composite


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    # You would load your actual data here
    # scores_df = pd.read_csv('your_scores.csv')
    # returns_series = pd.read_csv('your_returns.csv')['next_day_return']
    
    print("Run this script with your actual scores_df and returns_series")
    print("\nExample:")
    print("  correlations, composite = diagnose_calibration_issues(scores_df, returns_series)")
