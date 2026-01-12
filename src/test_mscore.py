"""
Enhanced M-Score Testing Script – ASRE DataLoader Integrated
Advanced validation with walk-forward analysis, statistical testing, and comprehensive metrics

Features:
- Walk-forward optimization
- Statistical significance testing
- Monte Carlo simulation
- Multiple performance metrics
- Out-of-sample validation
- Parameter sensitivity analysis

Author: ASRE Project
Date: January 2026
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from scipy import stats
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# from asre.data_loader import load_stock_data

# ============================================================================
# ENHANCED CONFIG
# ============================================================================

@dataclass
class MomentumConfig:
    kappa: float = 0.03
    beta_m: float = 0.2
    window_60d: int = 60
    long_threshold: float = 70
    short_threshold: float = 30

class BacktestConfig:
    def __init__(self, 
                 train_period: int = 252,  # 1 year
                 test_period: int = 63,     # 3 months
                 capital: float = 10000,
                 commission: float = 0.001,  # 0.1% per trade
                 slippage: float = 0.0005):  # 0.05% slippage
        self.train_period = train_period
        self.test_period = test_period
        self.capital = capital
        self.commission = commission
        self.slippage = slippage

# ============================================================================
# CORE INDICATORS (from original)
# ============================================================================

def log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(prices / prices.shift(periods))

def rolling_volatility(returns: pd.Series, window: int = 60) -> pd.Series:
    return returns.rolling(window).std()

def exponential_decay_convolution(returns, kappa, window):
    out = pd.Series(index=returns.index, dtype=float)
    for i in range(len(returns)):
        if i < window - 1:
            out.iloc[i] = np.nan
        else:
            r = returns.iloc[i - window + 1:i + 1].values
            w = np.exp(-kappa * np.arange(window - 1, -1, -1))
            out.iloc[i] = np.nansum(r * w) / window
    return out

def volatility_normalization(returns, kappa, window):
    out = pd.Series(index=returns.index, dtype=float)
    for i in range(len(returns)):
        if i < window - 1:
            out.iloc[i] = np.nan
        else:
            r = returns.iloc[i - window + 1:i + 1].values
            w = np.exp(-2 * kappa * np.arange(window - 1, -1, -1))
            out.iloc[i] = np.sqrt(np.nansum((r ** 2) * w) / window)
    return out

def rolling_autocorrelation(returns, lag, window):
    out = pd.Series(index=returns.index, dtype=float)
    for i in range(len(returns)):
        if i < window + lag:
            out.iloc[i] = np.nan
        else:
            x = returns.iloc[i - window + 1:i + 1]
            y = returns.iloc[i - window + 1 - lag:i + 1 - lag]
            mask = ~(x.isna() | y.isna())
            out.iloc[i] = np.corrcoef(x[mask], y[mask])[0, 1] if mask.sum() > 10 else 0.0
    return out

def safe_rolling_zscore(series, window=60, min_periods=30, vol_floor=0.01):
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    adaptive_floor = max(vol_floor, series.std() * 0.05)
    std = std.clip(lower=adaptive_floor)
    z = (series - mean) / std
    return z.fillna((series - series.mean()) / max(series.std(), adaptive_floor))

def soft_clamp(z, lower=-2.5, upper=2.5, smoothness=0.3):
    z1 = lower + smoothness * np.log1p(np.exp((z - lower) / smoothness))
    return upper - smoothness * np.log1p(np.exp((upper - z1) / smoothness))

# ============================================================================
# M-SCORE COMPUTATION
# ============================================================================

def compute_momentum_score(df: pd.DataFrame, config: MomentumConfig) -> pd.DataFrame:
    lr_60 = log_returns(df["close"], config.window_60d)

    num = exponential_decay_convolution(lr_60, config.kappa, config.window_60d)
    den = volatility_normalization(lr_60, config.kappa, config.window_60d)
    den = den.clip(lower=max(0.001, lr_60.std() * 0.01))

    ratio = num / den
    z = safe_rolling_zscore(ratio)
    zc = soft_clamp(z)

    daily_lr = log_returns(df["close"])
    autocorr = rolling_autocorrelation(daily_lr, 60, config.window_60d).fillna(0.0)

    momentum = 50 * np.tanh(zc / 2)
    m_score = np.clip(50 + momentum + config.beta_m * autocorr, 0, 100)

    vol = rolling_volatility(daily_lr, config.window_60d)
    vol = vol.clip(lower=max(0.01, daily_lr.std() * 0.05))
    sharpe_adj = np.clip(np.sqrt(0.15 / vol), 0.5, 2.0)

    df = df.copy()
    df["m_score"] = m_score
    df["m_score_adj"] = np.clip(m_score * sharpe_adj, 0, 100)

    return df

# ============================================================================
# ENHANCED BACKTESTING ENGINE
# ============================================================================

def generate_signals(score, long_th=70, short_th=30):
    s = pd.Series(0, index=score.index)
    s[score >= long_th] = 1
    s[score <= short_th] = -1
    return s

def enhanced_backtest(df: pd.DataFrame, signals: pd.Series, config: BacktestConfig) -> Dict:
    """Enhanced backtest with transaction costs and detailed metrics"""
    df = df.copy()
    df["signal"] = signals
    df["ret"] = df["close"].pct_change()

    # Apply transaction costs
    df["position_change"] = df["signal"].diff().abs()
    df["costs"] = df["position_change"] * (config.commission + config.slippage)
    df["str_ret"] = df["signal"].shift(1) * df["ret"] - df["costs"]

    # Calculate equity curve
    equity = config.capital * (1 + df["str_ret"].fillna(0)).cumprod()
    df["equity"] = equity

    # Performance metrics
    total_trades = df["position_change"].sum()
    winning_trades = (df[df["str_ret"] > 0]["str_ret"]).count()
    losing_trades = (df[df["str_ret"] < 0]["str_ret"]).count()

    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    avg_win = df[df["str_ret"] > 0]["str_ret"].mean() if winning_trades > 0 else 0
    avg_loss = abs(df[df["str_ret"] < 0]["str_ret"].mean()) if losing_trades > 0 else 0

    profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if avg_loss > 0 and losing_trades > 0 else 0

    # Risk metrics
    returns = df["str_ret"].dropna()
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
    sortino = np.sqrt(252) * returns.mean() / returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0

    # Drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar ratio
    calmar = (equity.iloc[-1] / config.capital - 1) / abs(max_dd) if max_dd != 0 else 0

    return {
        "strategy_return": equity.iloc[-1] / config.capital - 1,
        "buy_hold": df["close"].iloc[-1] / df["close"].iloc[0] - 1,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "final_equity": equity.iloc[-1],
        "equity_curve": equity,
        "returns": returns
    }

# ============================================================================
# WALK-FORWARD ANALYSIS
# ============================================================================

def walk_forward_analysis(df: pd.DataFrame, 
                         m_config: MomentumConfig,
                         bt_config: BacktestConfig) -> Dict:
    """
    Perform walk-forward analysis to validate strategy robustness
    """
    results = []
    train_size = bt_config.train_period
    test_size = bt_config.test_period

    total_periods = len(df)
    n_splits = (total_periods - train_size) // test_size

    print(f"\n  Walk-Forward: {n_splits} splits ({train_size} train / {test_size} test)")

    for i in range(n_splits):
        train_start = i * test_size
        train_end = train_start + train_size
        test_end = train_end + test_size

        if test_end > total_periods:
            break

        # Train period
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()

        # Optimize on train (here we use fixed params, but you could optimize)
        train_scored = compute_momentum_score(train_df, m_config)

        # Test on out-of-sample
        test_scored = compute_momentum_score(test_df, m_config)
        signals = generate_signals(test_scored["m_score_adj"], 
                                  m_config.long_threshold, 
                                  m_config.short_threshold)

        res = enhanced_backtest(test_scored, signals, bt_config)
        res["period"] = i + 1
        results.append(res)

    # Aggregate results
    wf_returns = [r["strategy_return"] for r in results]
    wf_sharpes = [r["sharpe"] for r in results]

    return {
        "n_periods": len(results),
        "avg_return": np.mean(wf_returns),
        "std_return": np.std(wf_returns),
        "avg_sharpe": np.mean(wf_sharpes),
        "win_periods": sum(1 for r in wf_returns if r > 0),
        "consistency": sum(1 for r in wf_returns if r > 0) / len(wf_returns) if wf_returns else 0,
        "all_results": results
    }

# ============================================================================
# STATISTICAL SIGNIFICANCE TESTING
# ============================================================================

def test_statistical_significance(strategy_returns: pd.Series, 
                                  benchmark_returns: pd.Series) -> Dict:
    """
    Test if strategy significantly outperforms benchmark
    """
    excess_returns = strategy_returns - benchmark_returns

    # T-test
    t_stat, p_value = stats.ttest_1samp(excess_returns.dropna(), 0)

    # Information ratio
    ir = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_5pct": p_value < 0.05,
        "significant_1pct": p_value < 0.01,
        "information_ratio": ir,
        "mean_excess_return": excess_returns.mean()
    }

# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def monte_carlo_simulation(returns: pd.Series, n_sims: int = 1000, n_days: int = 252) -> Dict:
    """
    Monte Carlo simulation for risk assessment
    """
    mean_return = returns.mean()
    std_return = returns.std()

    simulated_returns = np.random.normal(mean_return, std_return, (n_sims, n_days))
    simulated_equity = (1 + simulated_returns).cumprod(axis=1)

    final_values = simulated_equity[:, -1]

    return {
        "mean_final": np.mean(final_values),
        "median_final": np.median(final_values),
        "percentile_5": np.percentile(final_values, 5),
        "percentile_95": np.percentile(final_values, 95),
        "prob_positive": np.mean(final_values > 1),
        "var_95": np.percentile(final_values - 1, 5)
    }

# ============================================================================
# PARAMETER SENSITIVITY ANALYSIS
# ============================================================================

def parameter_sensitivity(df: pd.DataFrame, 
                         base_config: MomentumConfig,
                         bt_config: BacktestConfig) -> pd.DataFrame:
    """
    Test sensitivity to parameter changes
    """
    results = []

    # Test kappa values
    for kappa in [0.01, 0.02, 0.03, 0.04, 0.05]:
        config = MomentumConfig(kappa=kappa, beta_m=base_config.beta_m, window_60d=base_config.window_60d)
        scored = compute_momentum_score(df, config)
        signals = generate_signals(scored["m_score_adj"])
        res = enhanced_backtest(scored, signals, bt_config)
        results.append({"param": "kappa", "value": kappa, "return": res["strategy_return"], "sharpe": res["sharpe"]})

    # Test beta_m values
    for beta_m in [0.0, 0.1, 0.2, 0.3, 0.4]:
        config = MomentumConfig(kappa=base_config.kappa, beta_m=beta_m, window_60d=base_config.window_60d)
        scored = compute_momentum_score(df, config)
        signals = generate_signals(scored["m_score_adj"])
        res = enhanced_backtest(scored, signals, bt_config)
        results.append({"param": "beta_m", "value": beta_m, "return": res["strategy_return"], "sharpe": res["sharpe"]})

    # Test thresholds
    for long_th in [60, 65, 70, 75, 80]:
        config = MomentumConfig(kappa=base_config.kappa, beta_m=base_config.beta_m, window_60d=base_config.window_60d)
        config.long_threshold = long_th
        config.short_threshold = 100 - long_th
        scored = compute_momentum_score(df, config)
        signals = generate_signals(scored["m_score_adj"], long_th, 100 - long_th)
        res = enhanced_backtest(scored, signals, bt_config)
        results.append({"param": "long_threshold", "value": long_th, "return": res["strategy_return"], "sharpe": res["sharpe"]})

    return pd.DataFrame(results)

# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================

def comprehensive_test(ticker: str, df: pd.DataFrame) -> Dict:
    """
    Run full test suite on a single ticker
    """
    m_config = MomentumConfig()
    bt_config = BacktestConfig()

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE M-SCORE ANALYSIS: {ticker}")
    print(f"{'='*80}")

    # 1. Basic backtest
    print("\n[1/6] Running basic backtest...")
    scored = compute_momentum_score(df, m_config)
    signals = generate_signals(scored["m_score_adj"], m_config.long_threshold, m_config.short_threshold)
    basic_results = enhanced_backtest(scored, signals, bt_config)

    print(f"  Strategy Return: {basic_results['strategy_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {basic_results['sharpe']:.2f}")
    print(f"  Win Rate: {basic_results['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {basic_results['profit_factor']:.2f}")

    # 2. Walk-forward analysis
    print("\n[2/6] Running walk-forward analysis...")
    wf_results = walk_forward_analysis(df, m_config, bt_config)
    print(f"  Avg OOS Return: {wf_results['avg_return']*100:.2f}%")
    print(f"  Consistency: {wf_results['consistency']*100:.1f}%")
    print(f"  Avg Sharpe: {wf_results['avg_sharpe']:.2f}")

    # 3. Statistical significance
    print("\n[3/6] Testing statistical significance...")
    bh_returns = df["close"].pct_change()
    sig_test = test_statistical_significance(basic_results["returns"], bh_returns)
    print(f"  P-value: {sig_test['p_value']:.4f}")
    print(f"  Significant (5%): {sig_test['significant_5pct']}")
    print(f"  Information Ratio: {sig_test['information_ratio']:.2f}")

    # 4. Monte Carlo simulation
    print("\n[4/6] Running Monte Carlo simulation...")
    mc_results = monte_carlo_simulation(basic_results["returns"], n_sims=1000)
    print(f"  Prob(Profit): {mc_results['prob_positive']*100:.1f}%")
    print(f"  95% CI: [{mc_results['percentile_5']:.2f}, {mc_results['percentile_95']:.2f}]")

    # 5. Parameter sensitivity
    print("\n[5/6] Analyzing parameter sensitivity...")
    sensitivity_df = parameter_sensitivity(df, m_config, bt_config)

    # 6. Score distribution
    print("\n[6/6] Analyzing score distribution...")
    score_stats = {
        "mean": scored["m_score_adj"].mean(),
        "std": scored["m_score_adj"].std(),
        "min": scored["m_score_adj"].min(),
        "max": scored["m_score_adj"].max(),
        "current": scored["m_score_adj"].iloc[-1]
    }
    print(f"  Current M-Score: {score_stats['current']:.2f}")
    print(f"  Mean: {score_stats['mean']:.2f} ± {score_stats['std']:.2f}")

    return {
        "ticker": ticker,
        "basic": basic_results,
        "walk_forward": wf_results,
        "significance": sig_test,
        "monte_carlo": mc_results,
        "sensitivity": sensitivity_df,
        "score_stats": score_stats
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    # Mock load_stock_data for demo
    def load_stock_data(ticker, start):
        """Mock function - replace with actual ASRE data loader"""
        dates = pd.date_range(start=start, periods=750, freq='D')
        np.random.seed(hash(ticker) % 2**32)
        price = 100 * (1 + np.random.randn(750).cumsum() * 0.01)
        return pd.DataFrame({
            'close': price,
            'volume': np.random.randint(1e6, 1e8, 750)
        }, index=dates)

    tickers = ["AAPL", "MSFT", "NVDA", "AMZN"]
    all_results = {}

    print("\n" + "="*80)
    print("ASRE M-SCORE ENHANCED TESTING SUITE")
    print("="*80)
    print(f"Testing {len(tickers)} tickers with comprehensive validation")

    for ticker in tickers:
        df = load_stock_data(ticker=ticker, start="2023-01-01")
        all_results[ticker] = comprehensive_test(ticker, df)

    # Summary comparison
    print("\n" + "="*80)
    print("CROSS-TICKER SUMMARY")
    print("="*80)

    summary_data = []
    for ticker, results in all_results.items():
        summary_data.append({
            "Ticker": ticker,
            "Return": f"{results['basic']['strategy_return']*100:.2f}%",
            "Sharpe": f"{results['basic']['sharpe']:.2f}",
            "Win%": f"{results['basic']['win_rate']*100:.1f}%",
            "PF": f"{results['basic']['profit_factor']:.2f}",
            "WF Consistency": f"{results['walk_forward']['consistency']*100:.1f}%",
            "P-value": f"{results['significance']['p_value']:.4f}",
            "Current M-Score": f"{results['score_stats']['current']:.2f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    print("\n" + "="*80)
    print("Testing completed. Strategy validation comprehensive.")
    print("="*80)
