"""
ASRE Backtesting Framework V3 (Institutional Grade)

CHANGELOG V3:
1. Execution Timing: Signal at Close (T) -> Trade at Open (T+1).
2. Slippage Model: Volatility-based slippage (High Vol = High Slippage).
3. Price Data: Uses 'adj_close' for signals, 'open' for execution.

Author: ASRE Project
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List

logger = logging.getLogger(__name__)

# Re-export metrics for compatibility
from asre.backtest import (
    sharpe_ratio, sortino_ratio, calmar_ratio, information_ratio,
    max_drawdown, cagr, win_rate, profit_factor,
    value_at_risk, conditional_var, generate_backtest_report, print_backtest_report
)

def compute_institutional_returns(
    df: pd.DataFrame,
    signal_col: str = 'signal',
    open_col: str = 'open',
    close_col: str = 'close',
    vol_col: str = 'vix',  # Optional, defaults to rolling std if missing
    base_slippage: float = 0.0005,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """
    Compute returns with REALISTIC Institutional constraints.
    """
    result = df.copy()

    # 1. Execution Logic: Shift Signal to Next Day
    # Signal generated at T Close applies to T+1 Open
    result['target_position'] = result[signal_col].shift(1).fillna(0)

    # 2. Calculate Trade Returns
    result['exec_price'] = result[open_col]
    result['price_return'] = result['exec_price'].pct_change()
    result['strategy_return'] = result['target_position'] * result['price_return']

    # 3. Dynamic Slippage Model
    if vol_col in result.columns:
        vol_multiplier = result[vol_col] / 20.0
    else:
        vol_multiplier = result[close_col].pct_change().rolling(20).std() * 100
        vol_multiplier = vol_multiplier.fillna(1.0)

    result['dynamic_slippage'] = base_slippage * vol_multiplier

    # 4. Transaction Costs
    trades = result['target_position'].diff().abs().fillna(0)

    # FIX: Rename to match `generate_backtest_report` expectation
    result['transaction_cost_incurred'] = trades * (transaction_cost + result['dynamic_slippage'])

    # 5. Net Returns
    result['net_return'] = result['strategy_return'] - result['transaction_cost_incurred']

    # 6. Cumulative Metrics
    result['cumulative_return'] = (1 + result['net_return']).cumprod()

    # Drawdown Calculation (Manual to ensure column exists)
    cumulative = result['cumulative_return']
    running_max = cumulative.cummax()
    result['drawdown'] = (cumulative - running_max) / running_max

    return result