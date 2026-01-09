"""Enhanced diagnostic script to debug F-Score and T-Score issues."""

from asre.data_loader import load_stock_data
from asre.fundamentals import compute_fundamental_score
from asre.technical import compute_technical_score
from asre.momentum import compute_momentum_score
import pandas as pd
import numpy as np

# Load data
print("Loading AAPL data...")
df = load_stock_data('AAPL', '2025-01-01', '2026-01-07')

print(f"\nData shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# Check fundamental inputs - FULL TIME SERIES
print("="*80)
print("FUNDAMENTAL TIME SERIES ANALYSIS")
print("="*80)
print(f"\nPE Ratio:")
print(f"  Mean: {df['pe'].mean():.2f}")
print(f"  Std:  {df['pe'].std():.4f}")
print(f"  Min:  {df['pe'].min():.2f}")
print(f"  Max:  {df['pe'].max():.2f}")
print(f"  Unique values: {df['pe'].nunique()}")
print(f"  First 5: {df['pe'].head().values}")
print(f"  Last 5:  {df['pe'].tail().values}")

print(f"\nROE:")
print(f"  Mean: {df['roe'].mean():.2f}%")
print(f"  Std:  {df['roe'].std():.4f}%")
print(f"  Min:  {df['roe'].min():.2f}%")
print(f"  Max:  {df['roe'].max():.2f}%")
print(f"  Unique values: {df['roe'].nunique()}")
print(f"  First 5: {df['roe'].head().values}")
print(f"  Last 5:  {df['roe'].tail().values}")

print(f"\nDebt/Equity:")
print(f"  Mean: {df['de'].mean():.4f}")
print(f"  Std:  {df['de'].std():.4f}")
print(f"  Min:  {df['de'].min():.4f}")
print(f"  Max:  {df['de'].max():.4f}")
print(f"  Unique values: {df['de'].nunique()}")
print(f"  First 5: {df['de'].head().values}")
print(f"  Last 5:  {df['de'].tail().values}")

# Check technical inputs
print("\n" + "="*80)
print("TECHNICAL TIME SERIES ANALYSIS")
print("="*80)
print(f"\nPrice (Close):")
print(f"  Mean: ${df['close'].mean():.2f}")
print(f"  Std:  ${df['close'].std():.2f}")
print(f"  Min:  ${df['close'].min():.2f}")
print(f"  Max:  ${df['close'].max():.2f}")
print(f"  Returns std: {df['close'].pct_change().std():.4f}")

print(f"\nVolume:")
print(f"  Mean: {df['volume'].mean():,.0f}")
print(f"  Std:  {df['volume'].std():,.0f}")

# Compute individual scores with detailed output
print("\n" + "="*80)
print("COMPUTING INDIVIDUAL SCORES WITH DIAGNOSTICS")
print("="*80)

# Test fundamental score WITH COMPONENTS
print("\n1. Computing F-Score with detailed components...")
print("-" * 60)
try:
    df_with_f = compute_fundamental_score(df.copy(), return_components=True)
    
    print(f"   F-Score statistics:")
    print(f"     Mean:  {df_with_f['f_score'].mean():.4f}")
    print(f"     Std:   {df_with_f['f_score'].std():.4f}")
    print(f"     Min:   {df_with_f['f_score'].min():.4f}")
    print(f"     Max:   {df_with_f['f_score'].max():.4f}")
    print(f"     Range: [{df_with_f['f_score'].min():.2f}, {df_with_f['f_score'].max():.2f}]")
    
    # Check if all values are the same
    if df_with_f['f_score'].nunique() == 1:
        print(f"   ⚠️  WARNING: All F-Scores are identical! Value: {df_with_f['f_score'].iloc[0]:.4f}")
        print(f"   This indicates zero variance in fundamental scoring.")
    elif df_with_f['f_score'].std() < 1.0:
        print(f"   ⚠️  WARNING: F-Score has very low variance (std={df_with_f['f_score'].std():.4f})")
    else:
        print(f"   ✅ F-Score has good variance!")
    
    # Check intermediate components
    print(f"\n   Intermediate Components:")
    component_cols = [
        'f1_pe', 'f2_roe', 'f3_de', 'likelihood_L_j', 
        'mu_f', 'sigma_f', 'drift_D_f', 'z_score', 
        'phi_z', 'decay_factor'
    ]
    
    for col in component_cols:
        if col in df_with_f.columns:
            vals = df_with_f[col].dropna()
            if len(vals) > 0:
                if col == 'mu_f':
                    # Scalar value
                    print(f"   {col}: {df_with_f[col].iloc[0]:.4f} (constant)")
                else:
                    print(f"   {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, range=[{vals.min():.2f}, {vals.max():.2f}]")
    
    print(f"\n   Sample F-Scores (first 10):")
    print(f"   {df_with_f['f_score'].head(10).values}")
    
    print(f"\n   Sample F-Scores (last 10):")
    print(f"   {df_with_f['f_score'].tail(10).values}")
    
    # Export for detailed inspection
    print(f"\n   Exporting detailed F-Score data to 'f_score_debug.csv'...")
    debug_cols = ['date', 'pe', 'roe', 'de', 'f1_pe', 'f2_roe', 'f3_de', 
                  'likelihood_L_j', 'drift_D_f', 'z_score', 'phi_z', 
                  'decay_factor', 'f_score']
    available_cols = [c for c in debug_cols if c in df_with_f.columns]
    df_with_f[available_cols].to_csv('f_score_debug.csv', index=False)
    print(f"   ✅ Exported to f_score_debug.csv")
    
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test technical score
print("\n2. Computing T-Score...")
print("-" * 60)
try:
    df_with_t = compute_technical_score(df.copy())
    
    print(f"   T-Score statistics:")
    print(f"     Mean:  {df_with_t['t_score'].mean():.4f}")
    print(f"     Std:   {df_with_t['t_score'].std():.4f}")
    print(f"     Min:   {df_with_t['t_score'].min():.4f}")
    print(f"     Max:   {df_with_t['t_score'].max():.4f}")
    print(f"     Range: [{df_with_t['t_score'].min():.2f}, {df_with_t['t_score'].max():.2f}]")
    
    # Check if all values are the same or maxed out
    if df_with_t['t_score'].nunique() == 1:
        print(f"   ⚠️  WARNING: All T-Scores are identical! Value: {df_with_t['t_score'].iloc[0]:.4f}")
    elif df_with_t['t_score'].mean() > 95:
        print(f"   ⚠️  WARNING: T-Scores are saturated near maximum (mean={df_with_t['t_score'].mean():.2f})")
    elif df_with_t['t_score'].std() < 1.0:
        print(f"   ⚠️  WARNING: T-Score has very low variance (std={df_with_t['t_score'].std():.4f})")
    else:
        print(f"   ✅ T-Score has good variance!")
    
    # Check intermediate indicators
    print(f"\n   Technical Indicators:")
    indicator_cols = ['rsi', 'bb_position', 'macd', 'macd_signal', 'adx', 'atr', 't_raw']
    for col in indicator_cols:
        if col in df_with_t.columns:
            vals = df_with_t[col].dropna()
            if len(vals) > 0:
                print(f"   {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, range=[{vals.min():.2f}, {vals.max():.2f}]")
    
    print(f"\n   Sample T-Scores (first 10):")
    print(f"   {df_with_t['t_score'].head(10).values}")
    
    print(f"\n   Sample T-Scores (last 10):")
    print(f"   {df_with_t['t_score'].tail(10).values}")
        
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test momentum score
print("\n3. Computing M-Score...")
print("-" * 60)
try:
    df_with_m = compute_momentum_score(df.copy())
    
    print(f"   M-Score statistics:")
    print(f"     Mean:  {df_with_m['m_score'].mean():.4f}")
    print(f"     Std:   {df_with_m['m_score'].std():.4f}")
    print(f"     Min:   {df_with_m['m_score'].min():.4f}")
    print(f"     Max:   {df_with_m['m_score'].max():.4f}")
    print(f"     Range: [{df_with_m['m_score'].min():.2f}, {df_with_m['m_score'].max():.2f}]")
    
    if df_with_m['m_score'].std() > 10:
        print(f"   ✅ M-Score has excellent variance!")
    
    print(f"\n   Sample M-Scores (first 10):")
    print(f"   {df_with_m['m_score'].head(10).values}")
    
    print(f"\n   Sample M-Scores (last 10):")
    print(f"   {df_with_m['m_score'].tail(10).values}")
    
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
print("\n📊 Summary:")
print("1. Check 'f_score_debug.csv' for detailed F-Score component breakdown")
print("2. Look for components with zero variance (they break the formula)")
print("3. Verify that:")
print("   - Likelihood (L_j) has variance")
print("   - Drift term (D_F) is not all zeros")
print("   - Decay factor is not all zeros")
print("   - Phi(z) values vary between 0 and 1")
