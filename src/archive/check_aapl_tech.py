from asre.data_loader import load_stock_data
from asre.technical import compute_technical_score

df = load_stock_data('AAPL', '2023-01-01', '2026-01-09')
df = compute_technical_score(df, return_components=True)

print('\n=== Available Columns ===')
print(df.columns.tolist())

print('\n=== Last Row (All Values) ===')
latest = df.iloc[-1]
for col in df.columns:
    if 'score' in col.lower() or 'rsi' in col.lower() or 'macd' in col.lower() or 'ma_' in col.lower():
        print(f'{col}: {latest[col]}')

print(f'\n=== T-Score Breakdown (Last 5 Days) ===')
display_cols = ['date', 'close', 't_score']
print(df[display_cols].tail(5).to_string(index=False))

print(f'\n=== Latest T-Score ===')
print(f'T-Score: {latest.t_score:.1f}/100')
print(f'Close: ')
