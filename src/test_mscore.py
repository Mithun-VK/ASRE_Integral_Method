# test_pure_momentum.py
"""
Test M-Score STANDALONE without F-Score or T-Score
This isolates momentum strategy performance
"""
import yfinance as yf
from asre.momentum import compute_momentum_score, momentum_signal
from asre.config import MomentumConfig

def pure_momentum_strategy(ticker, config):
    """Trade using ONLY M-Score (ignore fundamentals)"""
    
    # Get data
    df = yf.download(ticker, period="2y", progress=False)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    
    # Calculate M-Score only
    result = compute_momentum_score(df, config, return_components=True)
    
    # Generate signals based ONLY on M-Score
    if config.use_enhancements:
        signals = momentum_signal(
            m_score=result['m_score_adj'],
            use_enhancements=True,
            trend_strength=result['trend_strength'],
            trend_maturity=result['trend_maturity'],
            prices=df['close'],
            config=config
        )
    else:
        signals = momentum_signal(result['m_score_adj'], threshold_long=70, threshold_short=30)
    
    # Backtest
    result['signal'] = signals
    result['returns'] = result['close'].pct_change()
    result['strategy_returns'] = result['signal'].shift(1) * result['returns']
    
    total_return = (1 + result['strategy_returns'].fillna(0)).prod() - 1
    buy_hold = (result['close'].iloc[-1] / result['close'].iloc[0]) - 1
    trades = (signals.diff() != 0).sum()
    
    return {
        'strategy_return': total_return * 100,
        'buy_hold': buy_hold * 100,
        'alpha': (total_return - buy_hold) * 100,
        'trades': trades,
        'm_score_current': result['m_score_adj'].iloc[-1]
    }

# Test both configurations
stocks = ['AAPL', 'TSLA', 'NVDA', 'JPM']

print("="*80)
print("PURE MOMENTUM STRATEGY (M-Score Only, No Fundamentals)")
print("="*80)

for ticker in stocks:
    print(f"\n{'='*80}")
    print(f"{ticker}")
    print(f"{'='*80}")
    
    # Original
    config_orig = MomentumConfig.original()
    orig = pure_momentum_strategy(ticker, config_orig)
    
    # Enhanced
    config_enh = MomentumConfig.balanced()
    enh = pure_momentum_strategy(ticker, config_enh)
    
    print(f"\nOriginal:  {orig['strategy_return']:>6.1f}% return, {orig['trades']:>2} trades, M-Score={orig['m_score_current']:.0f}")
    print(f"Enhanced:  {enh['strategy_return']:>6.1f}% return, {enh['trades']:>2} trades, M-Score={enh['m_score_current']:.0f}")
    print(f"Change:    {enh['alpha'] - orig['alpha']:>+6.1f}% alpha, {enh['trades'] - orig['trades']:>+2} trades")
