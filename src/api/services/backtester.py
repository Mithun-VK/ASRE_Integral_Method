"""
Backtester Service
Compares user's actual trades against ASRE recommendations.

Features:
- CSV parsing (multiple broker formats)
- ASRE strategy simulation
- Performance comparison
- Metrics calculation (returns, Sharpe, win rate, drawdown)
- PDF report generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, BinaryIO
from datetime import datetime, timedelta
from io import StringIO
import logging

from api.services.asre_service import ASREService
from api.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# CSV Format Mappings (broker-specific)
CSV_FORMATS = {
    "zerodha": {
        "date": "trade_date",
        "ticker": "symbol",
        "action": "trade_type",
        "quantity": "quantity",
        "price": "price",
        "buy_value": "BUY",
        "sell_value": "SELL"
    },
    "groww": {
        "date": "Date",
        "ticker": "Stock",
        "action": "Type",
        "quantity": "Qty",
        "price": "Price",
        "buy_value": "Buy",
        "sell_value": "Sell"
    },
    "standard": {
        "date": "date",
        "ticker": "ticker",
        "action": "action",
        "quantity": "quantity",
        "price": "price",
        "buy_value": "buy",
        "sell_value": "sell"
    }
}

# Risk-free rate for Sharpe ratio (annualized)
RISK_FREE_RATE = 0.04  # 4%


# ============================================================================
# CSV PARSER
# ============================================================================

class TradeCSVParser:
    """Parse trade CSV files from various broker formats"""
    
    @staticmethod
    def detect_format(df: pd.DataFrame) -> str:
        """Auto-detect CSV format based on column names"""
        columns = [col.lower() for col in df.columns]
        
        # Check for Zerodha format
        if 'trade_date' in columns and 'symbol' in columns:
            return "zerodha"
        
        # Check for Groww format
        if 'date' in columns and 'stock' in columns and 'type' in columns:
            return "groww"
        
        # Default to standard format
        return "standard"
    
    @staticmethod
    def parse_csv(
        file_content: str,
        format_hint: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Parse CSV file into standardized trade format.
        
        Args:
            file_content: CSV file content as string
            format_hint: Optional format hint ('zerodha', 'groww', 'standard')
            
        Returns:
            Standardized DataFrame with columns: date, ticker, action, quantity, price
        """
        try:
            # Read CSV
            df = pd.read_csv(StringIO(file_content))
            
            if df.empty:
                raise ValueError("CSV file is empty")
            
            # Auto-detect format if not provided
            if format_hint is None or format_hint not in CSV_FORMATS:
                format_hint = TradeCSVParser.detect_format(df)
            
            logger.info(f"Detected CSV format: {format_hint}")
            
            # Get format mapping
            fmt = CSV_FORMATS[format_hint]
            
            # Map columns to standard format
            required_cols = ['date', 'ticker', 'action', 'quantity', 'price']
            mapped_df = pd.DataFrame()
            
            for col in required_cols:
                source_col = fmt[col]
                if source_col not in df.columns:
                    raise ValueError(f"Missing required column: {source_col}")
                mapped_df[col] = df[source_col]
            
            # Normalize action values (buy/sell)
            buy_val = fmt['buy_value'].lower()
            sell_val = fmt['sell_value'].lower()
            
            mapped_df['action'] = mapped_df['action'].str.lower()
            mapped_df['action'] = mapped_df['action'].replace({buy_val: 'buy', sell_val: 'sell'})
            
            # Convert data types
            mapped_df['date'] = pd.to_datetime(mapped_df['date'])
            mapped_df['ticker'] = mapped_df['ticker'].str.upper().str.strip()
            mapped_df['quantity'] = pd.to_numeric(mapped_df['quantity'])
            mapped_df['price'] = pd.to_numeric(mapped_df['price'])
            
            # Sort by date
            mapped_df = mapped_df.sort_values('date').reset_index(drop=True)
            
            # Validate actions
            if not mapped_df['action'].isin(['buy', 'sell']).all():
                raise ValueError("Invalid action values. Must be 'buy' or 'sell'")
            
            logger.info(f"Parsed {len(mapped_df)} trades from CSV")
            return mapped_df
            
        except Exception as e:
            logger.error(f"Failed to parse CSV: {e}")
            raise ValueError(f"CSV parsing error: {str(e)}")


# ============================================================================
# ASRE STRATEGY SIMULATOR
# ============================================================================

class ASREStrategySimulator:
    """Simulate ASRE-based trading strategy"""
    
    def __init__(self):
        self.asre_service = ASREService
    
    def simulate(
        self,
        user_trades: pd.DataFrame,
        initial_capital: float = 100000.0
    ) -> pd.DataFrame:
        """
        Simulate ASRE strategy on same stocks and dates.
        
        Args:
            user_trades: User's actual trades
            initial_capital: Starting capital
            
        Returns:
            DataFrame with ASRE-recommended trades
        """
        logger.info(f"Simulating ASRE strategy with ${initial_capital:,.0f} capital")
        
        # Get unique tickers and date range
        tickers = user_trades['ticker'].unique().tolist()
        start_date = user_trades['date'].min()
        end_date = user_trades['date'].max()
        
        logger.info(f"Analyzing {len(tickers)} stocks from {start_date} to {end_date}")
        
        # Generate ASRE signals for each ticker
        asre_trades = []
        
        for ticker in tickers:
            try:
                # Get user's trades for this ticker
                user_ticker_trades = user_trades[user_trades['ticker'] == ticker]
                
                # Generate ASRE signals at user's trade dates
                for _, user_trade in user_ticker_trades.iterrows():
                    trade_date = user_trade['date']
                    
                    # Get ASRE rating for that date
                    # Note: In production, you'd want historical ratings
                    # For now, we'll use current rating as proxy
                    rating = self.asre_service.get_stock_rating(ticker)
                    
                    # Determine ASRE action
                    asre_action = self._get_asre_action(rating['rfinal'])
                    
                    if asre_action:
                        asre_trades.append({
                            'date': trade_date,
                            'ticker': ticker,
                            'action': asre_action,
                            'price': user_trade['price'],  # Use same price
                            'quantity': user_trade['quantity'],  # Use same quantity
                            'rfinal': rating['rfinal'],
                            'signal': rating['signal']
                        })
                
            except Exception as e:
                logger.warning(f"Failed to get ASRE rating for {ticker}: {e}")
                continue
        
        if not asre_trades:
            logger.warning("No ASRE trades generated")
            return pd.DataFrame()
        
        asre_df = pd.DataFrame(asre_trades)
        asre_df = asre_df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Generated {len(asre_df)} ASRE-recommended trades")
        return asre_df
    
    @staticmethod
    def _get_asre_action(rfinal: float) -> Optional[str]:
        """Convert ASRE rating to trade action"""
        if rfinal >= 80:
            return "buy"
        elif rfinal >= 65:
            return "buy"
        elif rfinal < 40:
            return "sell"
        else:
            return None  # Hold - no action


# ============================================================================
# PERFORMANCE METRICS CALCULATOR
# ============================================================================

class PerformanceCalculator:
    """Calculate trading performance metrics"""
    
    @staticmethod
    def calculate_metrics(
        trades: pd.DataFrame,
        initial_capital: float = 100000.0
    ) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            trades: DataFrame with trades
            initial_capital: Starting capital
            
        Returns:
            Dictionary with performance metrics
        """
        if trades.empty:
            return {
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'final_capital': initial_capital
            }
        
        # Calculate position-level returns
        positions = PerformanceCalculator._calculate_positions(trades)
        
        if positions.empty:
            return {
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': len(trades),
                'winning_trades': 0,
                'losing_trades': 0,
                'final_capital': initial_capital
            }
        
        # Total return
        total_pnl = positions['pnl'].sum()
        total_return_pct = (total_pnl / initial_capital) * 100
        
        # Win rate
        winning_trades = len(positions[positions['pnl'] > 0])
        losing_trades = len(positions[positions['pnl'] < 0])
        total_closed_trades = len(positions)
        win_rate = (winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0
        
        # Average return per trade
        avg_return = positions['return_pct'].mean()
        
        # Sharpe ratio
        sharpe = PerformanceCalculator._calculate_sharpe(positions['return_pct'])
        
        # Max drawdown
        max_dd = PerformanceCalculator._calculate_max_drawdown(positions, initial_capital)
        
        return {
            'total_return': round(total_pnl, 2),
            'total_return_pct': round(total_return_pct, 2),
            'win_rate': round(win_rate, 2),
            'avg_return': round(avg_return, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_dd, 2),
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'final_capital': round(initial_capital + total_pnl, 2)
        }
    
    @staticmethod
    def _calculate_positions(trades: pd.DataFrame) -> pd.DataFrame:
        """Calculate closed positions from trades"""
        positions = []
        holdings = {}  # {ticker: {'quantity': x, 'avg_price': y}}
        
        for _, trade in trades.iterrows():
            ticker = trade['ticker']
            action = trade['action']
            quantity = trade['quantity']
            price = trade['price']
            
            if action == 'buy':
                # Add to position
                if ticker not in holdings:
                    holdings[ticker] = {'quantity': 0, 'avg_price': 0}
                
                old_qty = holdings[ticker]['quantity']
                old_price = holdings[ticker]['avg_price']
                
                new_qty = old_qty + quantity
                new_avg_price = ((old_qty * old_price) + (quantity * price)) / new_qty
                
                holdings[ticker] = {
                    'quantity': new_qty,
                    'avg_price': new_avg_price
                }
            
            elif action == 'sell':
                # Close position
                if ticker in holdings and holdings[ticker]['quantity'] > 0:
                    buy_price = holdings[ticker]['avg_price']
                    sell_price = price
                    
                    pnl = (sell_price - buy_price) * quantity
                    return_pct = ((sell_price - buy_price) / buy_price) * 100
                    
                    positions.append({
                        'ticker': ticker,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'quantity': quantity,
                        'pnl': pnl,
                        'return_pct': return_pct
                    })
                    
                    # Update holdings
                    holdings[ticker]['quantity'] -= quantity
                    if holdings[ticker]['quantity'] <= 0:
                        del holdings[ticker]
        
        return pd.DataFrame(positions) if positions else pd.DataFrame()
    
    @staticmethod
    def _calculate_sharpe(returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (RISK_FREE_RATE * 100 / 252)  # Daily risk-free
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        return sharpe
    
    @staticmethod
    def _calculate_max_drawdown(positions: pd.DataFrame, initial_capital: float) -> float:
        """Calculate maximum drawdown"""
        if positions.empty:
            return 0.0
        
        # Calculate cumulative returns
        cumulative_pnl = positions['pnl'].cumsum()
        cumulative_capital = initial_capital + cumulative_pnl
        
        # Calculate running maximum
        running_max = cumulative_capital.cummax()
        
        # Calculate drawdown
        drawdown = (cumulative_capital - running_max) / running_max * 100
        
        return abs(drawdown.min())


# ============================================================================
# BACKTESTER SERVICE
# ============================================================================

class BacktesterService:
    """
    Main backtester service.
    
    Orchestrates CSV parsing, ASRE simulation, and comparison.
    """
    
    def __init__(self):
        self.parser = TradeCSVParser()
        self.simulator = ASREStrategySimulator()
        self.calculator = PerformanceCalculator()
    
    def run_backtest(
        self,
        csv_content: str,
        initial_capital: float = 100000.0,
        format_hint: Optional[str] = None
    ) -> Dict:
        """
        Run complete backtest analysis.
        
        Args:
            csv_content: CSV file content
            initial_capital: Starting capital
            format_hint: CSV format hint
            
        Returns:
            Complete backtest results
        """
        logger.info("Starting backtest analysis...")
        
        # Step 1: Parse user trades
        user_trades = self.parser.parse_csv(csv_content, format_hint)
        
        # Step 2: Simulate ASRE strategy
        asre_trades = self.simulator.simulate(user_trades, initial_capital)
        
        # Step 3: Calculate metrics
        user_metrics = self.calculator.calculate_metrics(user_trades, initial_capital)
        asre_metrics = self.calculator.calculate_metrics(asre_trades, initial_capital)
        
        # Step 4: Compare results
        comparison = self._compare_strategies(user_metrics, asre_metrics)
        
        return {
            'user_trades': user_trades.to_dict('records'),
            'asre_trades': asre_trades.to_dict('records'),
            'user_metrics': user_metrics,
            'asre_metrics': asre_metrics,
            'comparison': comparison,
            'initial_capital': initial_capital,
            'timestamp': datetime.now().isoformat()
        }
    
    def _compare_strategies(
        self,
        user_metrics: Dict,
        asre_metrics: Dict
    ) -> Dict:
        """Compare user vs ASRE performance"""
        
        return_diff = asre_metrics['total_return_pct'] - user_metrics['total_return_pct']
        sharpe_diff = asre_metrics['sharpe_ratio'] - user_metrics['sharpe_ratio']
        winrate_diff = asre_metrics['win_rate'] - user_metrics['win_rate']
        
        # Determine winner
        if return_diff > 5:
            winner = "ASRE"
            message = f"ASRE outperformed by {return_diff:.1f}%"
        elif return_diff < -5:
            winner = "USER"
            message = f"You outperformed ASRE by {abs(return_diff):.1f}%!"
        else:
            winner = "TIE"
            message = "Similar performance"
        
        return {
            'winner': winner,
            'message': message,
            'return_difference': round(return_diff, 2),
            'sharpe_difference': round(sharpe_diff, 2),
            'winrate_difference': round(winrate_diff, 2),
            'metrics_summary': {
                'better_return': asre_metrics['total_return_pct'] > user_metrics['total_return_pct'],
                'better_sharpe': asre_metrics['sharpe_ratio'] > user_metrics['sharpe_ratio'],
                'better_winrate': asre_metrics['win_rate'] > user_metrics['win_rate'],
            }
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def parse_trade_csv(file_content: str, format_hint: Optional[str] = None) -> pd.DataFrame:
    """Quick CSV parsing"""
    parser = TradeCSVParser()
    return parser.parse_csv(file_content, format_hint)


def run_backtest(
    csv_content: str,
    initial_capital: float = 100000.0,
    format_hint: Optional[str] = None
) -> Dict:
    """Quick backtest execution"""
    service = BacktesterService()
    return service.run_backtest(csv_content, initial_capital, format_hint)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'BacktesterService',
    'TradeCSVParser',
    'ASREStrategySimulator',
    'PerformanceCalculator',
    'parse_trade_csv',
    'run_backtest',
]
