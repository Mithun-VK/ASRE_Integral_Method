"""
Quick diagnostic to check dip quality column names
"""

import sys
from datetime import datetime, timedelta
sys.path.insert(0, 'D:/asre-project/src')

from asre.data_loader import load_stock_data
from asre.composite import compute_complete_asre

print("Loading TSLA data...")
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
df = load_stock_data('TSLA', start_date, end_date)

print("Computing ASRE with dip quality...")
df = compute_complete_asre(df, medallion=True, return_all_components=True)

print("\n" + "=" * 70)
print("ALL COLUMNS IN DATAFRAME:")
print("=" * 70)
dip_cols = [col for col in df.columns if 'dip' in col.lower()]
print(f"\nDip-related columns ({len(dip_cols)}):")
for col in sorted(dip_cols):
    print(f"  - {col}")

print("\n" + "=" * 70)
print("LOOKING FOR:")
print("=" * 70)
needed = ['dip_dip_quality_score', 'dip_dip_stage', 'dip_quality_score', 'dip_stage']
for col in needed:
    exists = "✅" if col in df.columns else "❌"
    print(f"{exists} {col}")
