"""
Backtest Routes
===============
Investment-focused ASRE backtesting API endpoints.

Directly imports from existing backtest scripts:
- run_investing_backtest.py
- investing_10k.py

Features:
- Single stock backtesting with tiered allocation
- Multiple strategy comparison
- Performance metrics (CAGR, Sharpe, drawdown)
- Detailed trade logs
- Risk management (stop loss, trailing stop)
- Tax simulation (LTCG/STCG)

Author: ASRE Team
Version: 2.0.0
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Path

# Ticker path-param validation (matches stocks.py): uppercase symbol, optional
# .NS/.BO suffix.
_TICKER_PATTERN = r"^[A-Z][A-Z0-9]*(\.[A-Z]{1,3})?$"
from pydantic import BaseModel, Field
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import existing backtest modules
from asre.data.fundamental_fetcher import FundamentalFetcher
from asre.data_loader import DataLoader
from asre.composite import compute_complete_asre
from api.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class StrategyParams(BaseModel):
    """Strategy parameters for backtesting"""
    overweight_threshold: float = Field(75.0, description="ASRE threshold for 120% allocation")
    full_threshold: float = Field(60.0, description="ASRE threshold for 100% allocation")
    reduce_threshold: float = Field(45.0, description="ASRE threshold for 50% allocation")
    exit_threshold: float = Field(30.0, description="ASRE threshold for 25% allocation")
    rebalance_period: str = Field("W", description="Rebalancing frequency: W (weekly), M (monthly), Q (quarterly)")


class BacktestRequest(BaseModel):
    """Request model for running a backtest"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., NVDA, AAPL)")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    initial_capital: float = Field(10000.0, description="Initial investment capital")
    strategy_params: Optional[StrategyParams] = Field(None, description="Strategy parameters")
    rating_col: str = Field("r_asre", description="Rating column to use: r_asre or r_final")


class TradeLog(BaseModel):
    """Individual trade entry"""
    Date: str
    Action: str
    Reason: str
    Price: str
    Size: str
    PnL: str


class BacktestMetrics(BaseModel):
    """Performance metrics"""
    initial_capital: float
    final_value: float
    profit: float
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    avg_drawdown: float
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_tax_rate: float
    annual_turnover: float
    time_in_market: float
    years: float


class BacktestResponse(BaseModel):
    """Response model for backtest results"""
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float
    strategy_params: StrategyParams
    metrics: BacktestMetrics
    trade_log: List[TradeLog]
    message: str


class CompareStrategiesRequest(BaseModel):
    """Request for comparing multiple strategies"""
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    strategies: List[StrategyParams]


# ============================================================================
# IMPORT EXISTING BACKTEST LOGIC
# ============================================================================

class InvestmentSignalEngine:
    """
    Generates tiered allocation signals based on ASRE ratings.
    (From run_investing_backtest.py)
    """
    
    def __init__(
        self,
        overweight_threshold: float = 75.0,
        full_threshold: float = 60.0,
        reduce_threshold: float = 45.0,
        exit_threshold: float = 30.0,
        rebalance_period: str = 'W'
    ):
        self.overweight = overweight_threshold
        self.full = full_threshold
        self.reduce = reduce_threshold
        self.exit = exit_threshold
        self.rebalance_period = rebalance_period
        
    def generate_signals(self, df: pd.DataFrame, rating_col: str = 'r_asre') -> pd.DataFrame:
        """Generate tiered allocation signals"""
        df = df.copy()
        
        resample_rule = self.rebalance_period.replace('M', 'ME')
        df_resampled = df.resample(resample_rule).last()
        
        def get_allocation(rating):
            if pd.isna(rating):
                return 0.0
            elif rating >= self.overweight:
                return 1.20
            elif rating >= self.full:
                return 1.00
            elif rating >= self.reduce:
                return 0.50
            elif rating >= self.exit:
                return 0.25
            else:
                return 0.00
        
        df_resampled['target_allocation'] = df_resampled[rating_col].apply(get_allocation)
        df_resampled = df_resampled.sort_index()
        
        temp_alloc = df_resampled[['target_allocation']].reset_index()
        temp_alloc.columns = ['rebalance_date', 'target_allocation']
        
        df_reset = df.reset_index()
        df_merged = pd.merge_asof(
            df_reset,
            temp_alloc,
            left_on='date',
            right_on='rebalance_date',
            direction='backward'
        )
        df_merged.index = df.index
        
        df['allocation'] = df_merged['target_allocation'].fillna(method='ffill').fillna(0.0)
        df['prev_allocation'] = df['allocation'].shift(1).fillna(0.0)
        df['is_rebalance_date'] = df['allocation'] != df['prev_allocation']
        
        if df['allocation'].iloc[0] > 0:
            df.iloc[0, df.columns.get_loc('is_rebalance_date')] = True
            
        return df


def compute_investment_returns(
    df: pd.DataFrame,
    allocation_col: str = 'allocation',
    price_col: str = 'close',
    transaction_cost: float = 0.0005,
    stop_loss_pct: float = 0.25,
    trailing_stop_pct: float = 0.15,
    ltcg_tax: float = 0.10,
    stcg_tax: float = 0.15,
) -> tuple:
    """
    Compute investment returns with risk management.
    (From run_investing_backtest.py)
    """
    import numpy as np
    
    result_df = df.copy()
    
    # Initialize tracking columns
    result_df['position'] = 0.0
    result_df['entry_price'] = np.nan
    result_df['entry_date'] = pd.NaT
    result_df['position_high'] = np.nan
    result_df['exit_reason'] = ''
    result_df['trade_pnl'] = 0.0
    result_df['days_held'] = 0
    result_df['tax_applied'] = 0.0
    
    trade_log = []
    
    # Position state
    current_position = 0.0
    entry_price = np.nan
    entry_date = None
    position_high = np.nan
    
    for idx, row in result_df.iterrows():
        target_allocation = row[allocation_col]
        current_price = row[price_col]
        
        # Update position high
        if pd.isna(position_high) and current_position > 0:
            position_high = current_price
        elif current_position > 0:
            position_high = max(position_high, current_price)
        
        # Check risk exits
        exit_triggered = False
        exit_reason = ''
        
        if current_position > 0:
            # Stop loss
            sl_price = entry_price * (1 - stop_loss_pct)
            if current_price <= sl_price:
                exit_triggered = True
                exit_reason = 'Stop_Loss'
                
            # Trailing stop
            elif not pd.isna(position_high):
                trailing_price = position_high * (1 - trailing_stop_pct)
                if current_price <= trailing_price:
                    exit_triggered = True
                    exit_reason = 'Trailing_Stop'
        
        # Execute risk exit
        if exit_triggered:
            trade_return = (current_price - entry_price) / entry_price
            days_held = (idx - entry_date).days if entry_date else 0
            tax_rate = stcg_tax if days_held < 365 else ltcg_tax
            net_pnl = trade_return * (1 - tax_rate) if trade_return > 0 else trade_return
            
            result_df.at[idx, 'trade_pnl'] = net_pnl * current_position
            result_df.at[idx, 'exit_reason'] = exit_reason
            result_df.at[idx, 'days_held'] = days_held
            result_df.at[idx, 'tax_applied'] = tax_rate if trade_return > 0 else 0.0
            
            trade_log.append({
                'Date': idx.strftime('%Y-%m-%d'),
                'Action': 'SELL (Risk Exit)',
                'Reason': exit_reason,
                'Price': f"{current_price:.2f}",
                'Size': f"{current_position * 100:.0f}%",
                'PnL': f"{net_pnl * 100:.2f}%"
            })
            
            current_position = 0.0
            entry_price = np.nan
            entry_date = None
            position_high = np.nan
        
        # Check rebalancing
        if row.get('is_rebalance_date', False) or exit_triggered:
            position_change = target_allocation - current_position
            
            if abs(position_change) > 0.01:
                # BUY
                if position_change > 0:
                    if current_position == 0:
                        entry_price = current_price
                        entry_date = idx
                        position_high = current_price
                        
                    trade_log.append({
                        'Date': idx.strftime('%Y-%m-%d'),
                        'Action': f"BUY (+{position_change * 100:.0f}%)",
                        'Reason': 'Entry/Increase',
                        'Price': f"{current_price:.2f}",
                        'Size': f"{target_allocation * 100:.0f}%",
                        'PnL': '-'
                    })
                
                # SELL
                elif position_change < 0:
                    if target_allocation == 0:
                        # Full exit
                        trade_return = (current_price - entry_price) / entry_price if not pd.isna(entry_price) else 0
                        days_held = (idx - entry_date).days if entry_date else 0
                        tax_rate = stcg_tax if days_held < 365 else ltcg_tax
                        net_pnl = trade_return * (1 - tax_rate) if trade_return > 0 else trade_return
                        
                        result_df.at[idx, 'trade_pnl'] = net_pnl * current_position
                        result_df.at[idx, 'exit_reason'] = 'Signal_Exit'
                        result_df.at[idx, 'days_held'] = days_held
                        result_df.at[idx, 'tax_applied'] = tax_rate if trade_return > 0 else 0.0
                        
                        trade_log.append({
                            'Date': idx.strftime('%Y-%m-%d'),
                            'Action': 'SELL (Full Exit)',
                            'Reason': 'Signal Exit',
                            'Price': f"{current_price:.2f}",
                            'Size': '0%',
                            'PnL': f"{net_pnl * 100:.2f}%"
                        })
                        
                        entry_price = np.nan
                        entry_date = None
                        position_high = np.nan
                    else:
                        # Partial reduction
                        trade_log.append({
                            'Date': idx.strftime('%Y-%m-%d'),
                            'Action': f"SELL ({position_change * 100:.0f}%)",
                            'Reason': 'Reduce Exposure',
                            'Price': f"{current_price:.2f}",
                            'Size': f"{target_allocation * 100:.0f}%",
                            'PnL': '-'
                        })
                
                current_position = target_allocation
        
        # Update tracking
        result_df.at[idx, 'position'] = current_position
        result_df.at[idx, 'entry_price'] = entry_price
        result_df.at[idx, 'entry_date'] = entry_date
        result_df.at[idx, 'position_high'] = position_high
    
    # Calculate returns
    result_df['price_return'] = result_df[price_col].pct_change()
    result_df['strategy_return'] = result_df['position'].shift(1) * result_df['price_return']
    result_df['position_change'] = result_df['position'].diff().abs()
    result_df['transaction_cost_incurred'] = result_df['position_change'] * transaction_cost
    result_df['net_return'] = result_df['strategy_return'] - result_df['transaction_cost_incurred']
    result_df['cumulative_return'] = (1 + result_df['net_return']).cumprod()
    
    # Drawdown
    cummax = result_df['cumulative_return'].cummax()
    result_df['drawdown'] = (result_df['cumulative_return'] - cummax) / cummax
    
    return result_df, trade_log


def generate_investment_report(df: pd.DataFrame, initial_capital: float = 10000.0) -> dict:
    """
    Generate comprehensive performance metrics.
    (From investing_10k.py)
    """
    import numpy as np
    
    # Return metrics
    total_return = (df['cumulative_return'].iloc[-1] - 1) * 100
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = ((df['cumulative_return'].iloc[-1]) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # Risk metrics
    volatility = df['net_return'].std() * np.sqrt(252) * 100
    sharpe = (cagr / volatility) if volatility > 0 else 0
    max_dd = df['drawdown'].min() * 100
    avg_dd = df[df['drawdown'] < 0]['drawdown'].mean() * 100 if (df['drawdown'] < 0).any() else 0
    
    # Trade metrics
    trades = df[df['trade_pnl'] != 0]
    num_trades = len(trades)
    wins = trades[trades['trade_pnl'] > 0]
    losses = trades[trades['trade_pnl'] < 0]
    
    win_rate = (len(wins) / num_trades * 100) if num_trades > 0 else 0
    avg_win = wins['trade_pnl'].mean() * 100 if len(wins) > 0 else 0
    avg_loss = losses['trade_pnl'].mean() * 100 if len(losses) > 0 else 0
    profit_factor = (abs(wins['trade_pnl'].sum()) / losses['trade_pnl'].sum()) if len(losses) > 0 and losses['trade_pnl'].sum() != 0 else 0
    
    # Tax & turnover
    total_tax = trades['tax_applied'].sum() if 'tax_applied' in trades.columns else 0
    total_position_change = df['position_change'].sum()
    annual_turnover = (total_position_change / years * 100) if years > 0 else 0
    time_in_market = (df['position'] > 0).sum() / len(df) * 100
    
    # Capital metrics
    final_value = initial_capital * df['cumulative_return'].iloc[-1]
    profit = final_value - initial_capital
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'profit': profit,
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'avg_drawdown': avg_dd,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_tax_rate': total_tax,
        'annual_turnover': annual_turnover,
        'time_in_market': time_in_market,
        'years': years
    }


def fetch_all_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch fundamentals, price data, and compute ASRE ratings.
    (From run_investing_backtest.py)
    """
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")

    # NSE tickers must carry the .NS suffix for Yahoo Finance / the cache.
    yf_ticker = ticker if '.' in ticker else f"{ticker}.NS"

    # Fetch fundamentals
    fetcher = FundamentalFetcher(cache_dir=str(settings.FUNDAMENTALS_CACHE_DIR))
    try:
        df_fundamentals, _ = fetcher.fetch_quarterly_fundamentals(yf_ticker, start_date, end_date)
        logger.info(f"✓ Fetched {len(df_fundamentals)} quarters")
    except Exception as e:
        logger.error(f"Error fetching fundamentals: {e}")
        df_fundamentals = None

    # Load price data
    loader = DataLoader()
    try:
        df = loader.load_stock_data(yf_ticker, start_date, end_date, quarterly_fundamentals=df_fundamentals)
        logger.info(f"✓ Loaded {len(df)} days of data")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    # Compute ASRE ratings
    try:
        df_complete = compute_complete_asre(df, yf_ticker, medallion=True, return_all_components=True)
        logger.info("✓ ASRE ratings computed")
        
        df_complete['date'] = pd.to_datetime(df_complete['date'])
        df_complete = df_complete.set_index('date')
        
    except Exception as e:
        logger.error(f"Error computing ASRE: {e}")
        raise
        
    return df_complete


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post("/run", response_model=BacktestResponse)
def run_backtest(request: BacktestRequest):
    """
    Run investment backtest for a single stock with tiered allocation strategy.
    
    **Features:**
    - Tiered allocation based on ASRE ratings
    - Risk management (stop loss 25%, trailing stop 15%)
    - Tax simulation (LTCG 10%, STCG 15%)
    - Transaction costs (5 bps)
    - Detailed trade log
    
    **Example:**
    ```json
    {
      "ticker": "NVDA",
      "start_date": "2023-01-01",
      "end_date": "2026-01-24",
      "initial_capital": 10000.0,
      "strategy_params": {
        "overweight_threshold": 75.0,
        "full_threshold": 60.0,
        "reduce_threshold": 45.0,
        "exit_threshold": 30.0,
        "rebalance_period": "W"
      }
    }
    ```
    """
    try:
        logger.info(f"Running backtest for {request.ticker}")
        
        # Use default strategy params if not provided
        if request.strategy_params is None:
            request.strategy_params = StrategyParams()
        
        params = request.strategy_params.dict()
        
        # Fetch data
        df = fetch_all_data(request.ticker, request.start_date, request.end_date)
        
        # Generate signals
        engine = InvestmentSignalEngine(**params)
        df_signals = engine.generate_signals(df, rating_col=request.rating_col)
        
        # Compute returns
        df_results, trade_log = compute_investment_returns(
            df_signals,
            allocation_col='allocation',
            price_col='close'
        )
        
        # Generate report
        report = generate_investment_report(df_results, request.initial_capital)
        
        logger.info(f"Backtest complete: {report['total_return']:.2f}% return")
        
        return BacktestResponse(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            strategy_params=request.strategy_params,
            metrics=BacktestMetrics(**report),
            trade_log=[TradeLog(**trade) for trade in trade_log],
            message=f"Backtest completed successfully. Total return: {report['total_return']:.2f}%"
        )
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@router.post("/compare")
def compare_strategies(request: CompareStrategiesRequest):
    """
    Compare multiple backtest strategies side-by-side.
    
    **Example:**
    ```json
    {
      "ticker": "NVDA",
      "start_date": "2023-01-01",
      "end_date": "2026-01-24",
      "initial_capital": 10000.0,
      "strategies": [
        {
          "overweight_threshold": 75.0,
          "full_threshold": 60.0,
          "reduce_threshold": 45.0,
          "exit_threshold": 30.0,
          "rebalance_period": "W"
        },
        {
          "overweight_threshold": 80.0,
          "full_threshold": 65.0,
          "reduce_threshold": 50.0,
          "exit_threshold": 35.0,
          "rebalance_period": "M"
        }
      ]
    }
    ```
    """
    try:
        results = []
        
        # Fetch data once
        df = fetch_all_data(request.ticker, request.start_date, request.end_date)
        
        for idx, strategy in enumerate(request.strategies):
            params = strategy.dict()
            
            # Generate signals
            engine = InvestmentSignalEngine(**params)
            df_signals = engine.generate_signals(df, rating_col='r_asre')
            
            # Compute returns
            df_results, trade_log = compute_investment_returns(df_signals)
            
            # Generate report
            report = generate_investment_report(df_results, request.initial_capital)
            
            results.append({
                "strategy_id": f"Strategy {idx + 1}",
                "params": params,
                "metrics": report
            })
        
        # Find best strategy
        best_strategy = max(results, key=lambda x: x['metrics']['sharpe'])
        
        return {
            "ticker": request.ticker,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "results": results,
            "best_strategy": best_strategy['strategy_id'],
            "message": f"Compared {len(results)} strategies. Best: {best_strategy['strategy_id']} (Sharpe: {best_strategy['metrics']['sharpe']:.3f})"
        }
        
    except Exception as e:
        logger.error(f"Strategy comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.get("/quick/{ticker}")
def quick_backtest(
    ticker: str = Path(..., pattern=_TICKER_PATTERN, description="Stock ticker symbol"),
    years: int = Query(3, description="Number of years to backtest"),
    capital: float = Query(10000.0, description="Initial capital")
):
    """
    Quick backtest with default parameters for the last N years.
    
    **Example:** `/api/backtest/quick/NVDA?years=3&capital=10000`
    """
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        
        request = BacktestRequest(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_capital=capital
        )
        
        return run_backtest(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick backtest failed: {str(e)}")
