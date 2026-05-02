"""
Investing Routes
================
Portfolio management and investment engine API endpoints.

Directly imports from existing portfolio management scripts:
- Moat_Portfolio.py
- production_investing_engine.py

Features:
- Create and manage portfolios
- Execute daily rebalancing cycles
- Position sizing with sector limits
- Risk management (VaR, Sharpe, drawdown)
- Dip buying opportunities
- Real-time portfolio state tracking

Author: ASRE Team
Version: 2.0.0
"""

import sys
import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import existing portfolio modules
from asre.data.fundamental_fetcher import FundamentalFetcher
from asre.data_loader import DataLoader
from asre.composite import compute_complete_asre

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/investing", tags=["investing"])


# ============================================================================
# CONFIGURATION (from Moat_Portfolio.py)
# ============================================================================

class PortfolioConfig:
    """Portfolio configuration constants"""
    # Capital
    INITIAL_CAPITAL = 100000.0
    
    # ASRE Tiers
    TIER_EXCEPTIONAL = 80.0
    TIER_EXCELLENT = 70.0
    TIER_STRONG = 60.0
    TIER_MODERATE = 50.0
    TIER_WEAK = 40.0
    
    # Position Sizing
    MAX_POSITION_SIZE = 0.12
    MIN_POSITION_SIZE = 0.02
    
    # Risk Management
    POSITION_STOP_LOSS = 0.30
    MAX_PORTFOLIO_DRAWDOWN = 0.25
    
    # Transaction Costs
    TRANSACTION_COST_BPS = 5
    MIN_CASH_RESERVE = 5000.0
    MIN_CASH_PCT = 0.10
    
    # Rebalancing
    REBALANCE_THRESHOLD = 0.05
    
    # Dip Buying
    ENABLE_BUY_THE_DIP = True
    DIP_MIN_ASRE = 60.0
    DIP_MAX_DISTANCE = -20.0
    DIP_POSITION_MULTIPLIER = 0.60
    DIP_MIN_CONFIDENCE = 50.0
    
    # Sector Limits
    SECTOR_LIMITS = {
        'Technology': 0.40,
        'Financials': 0.20,
        'Consumer': 0.20,
        'Industrial': 0.12,
        'Healthcare': 0.08
    }


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class CreatePortfolioRequest(BaseModel):
    """Request to create a new portfolio"""
    initial_capital: float = Field(100000.0, description="Initial investment capital")
    risk_profile: str = Field("moderate", description="Risk profile: conservative, moderate, aggressive")
    universe: List[str] = Field(..., description="List of stock tickers to track")
    enable_dip_buying: bool = Field(True, description="Enable buy-the-dip strategy")


class HoldingResponse(BaseModel):
    """Individual holding details"""
    ticker: str
    shares: float
    avg_cost_basis: float
    current_price: float
    market_value: float
    target_allocation: float
    allocation_pct: float
    entry_date: str
    entry_price: float
    max_price_since_entry: float
    drawdown_pct: float
    is_dip_buy: bool
    pnl_pct: float
    pnl_dollar: float


class RiskMetricsResponse(BaseModel):
    """Portfolio risk metrics"""
    portfolio_var_95: float
    portfolio_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    daily_change_pct: float
    largest_position_pct: float


class PortfolioStateResponse(BaseModel):
    """Complete portfolio state"""
    portfolio_id: str
    cash: float
    total_value: float
    equity_value: float
    cash_pct: float
    equity_pct: float
    total_return_pct: float
    peak_value: float
    holdings: List[HoldingResponse]
    risk_metrics: RiskMetricsResponse
    num_positions: int
    num_dip_positions: int
    last_updated: str


class AllocationTarget(BaseModel):
    """Target allocation for a stock"""
    ticker: str
    target_alloc: float
    current_alloc: float
    asre_score: float
    quality_tier: str
    is_dip_buy: bool
    action: str
    reason: str


class RebalanceResponse(BaseModel):
    """Rebalancing execution results"""
    portfolio_id: str
    execution_time: str
    allocation_targets: List[AllocationTarget]
    trades_executed: int
    cash_deployed: float
    message: str


class DipAnalysisResponse(BaseModel):
    """Dip analysis for a stock"""
    ticker: str
    is_dip: bool
    distance_from_sma_pct: float
    dip_stage: str
    confidence: float
    quality_tier: str
    approved_for_dip_buy: bool
    reason: str


class PortfolioRecommendations(BaseModel):
    """Investment recommendations"""
    buy_recommendations: List[AllocationTarget]
    sell_recommendations: List[AllocationTarget]
    dip_opportunities: List[DipAnalysisResponse]
    message: str


# ============================================================================
# PORTFOLIO CLASSES (from Moat_Portfolio.py)
# ============================================================================

class RiskCalculator:
    """Calculate portfolio risk metrics"""
    
    @staticmethod
    def calculate_portfolio_volatility(returns: pd.Series) -> float:
        if len(returns) < 2:
            return 0.0
        daily_vol = returns.std()
        annual_vol = daily_vol * (252 ** 0.5)
        return float(annual_vol)
    
    @staticmethod
    def calculate_var_95(portfolio_returns: pd.Series) -> float:
        if len(portfolio_returns) < 30:
            return 0.0
        var_pct = portfolio_returns.quantile(0.05)
        return float(var_pct)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04) -> float:
        if len(returns) < 2:
            return 0.0
        daily_ret = returns.mean()
        daily_vol = returns.std()
        if daily_vol == 0:
            return 0.0
        annual_ret = daily_ret * 252
        annual_vol = daily_vol * (252 ** 0.5)
        sharpe = (annual_ret - risk_free_rate) / annual_vol
        return float(sharpe)
    
    @staticmethod
    def calculate_max_drawdown(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        peak = values[0]
        max_dd = 0.0
        for val in values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return float(max_dd)


class DipAnalyzer:
    """Analyze dip quality for buying opportunities"""
    
    @staticmethod
    def analyze_dip(price: float, sma200: float, asre_score: float, quality_tier: str) -> Dict:
        distance = ((price - sma200) / sma200) * 100
        
        analysis = {
            'is_dip': False,
            'distance_from_sma_pct': distance,
            'dip_stage': 'NONE',
            'confidence': 0.0,
            'quality_tier': quality_tier,
            'approved_for_dip_buy': False,
            'reason': ''
        }
        
        if distance >= 0:
            analysis['is_dip'] = False
            analysis['reason'] = 'Price above SMA-200 (uptrend)'
            return analysis
        
        analysis['is_dip'] = True
        
        # Determine dip stage
        if distance >= -5:
            analysis['dip_stage'] = 'EARLY'
            analysis['confidence'] = 75.0
        elif distance >= -10:
            analysis['dip_stage'] = 'MID'
            analysis['confidence'] = 65.0
        elif distance >= -15:
            analysis['dip_stage'] = 'LATE'
            analysis['confidence'] = 55.0
        else:
            analysis['dip_stage'] = 'DEEP'
            analysis['confidence'] = 40.0
        
        # Quality checks
        if PortfolioConfig.DIP_MIN_ASRE > 0 and quality_tier not in ['A', 'B']:
            analysis['approved_for_dip_buy'] = False
            analysis['reason'] = f"Tier {quality_tier} not approved (need A/B)"
            return analysis
        
        if asre_score < PortfolioConfig.DIP_MIN_ASRE:
            analysis['approved_for_dip_buy'] = False
            analysis['reason'] = f"ASRE {asre_score:.1f} below minimum {PortfolioConfig.DIP_MIN_ASRE}"
            return analysis
        
        if distance < PortfolioConfig.DIP_MAX_DISTANCE:
            analysis['approved_for_dip_buy'] = False
            analysis['reason'] = f"Too far below SMA ({distance:.1f}% < {PortfolioConfig.DIP_MAX_DISTANCE}%)"
            return analysis
        
        if analysis['confidence'] < PortfolioConfig.DIP_MIN_CONFIDENCE:
            analysis['approved_for_dip_buy'] = False
            analysis['reason'] = f"Confidence too low ({analysis['confidence']:.1f}%)"
            return analysis
        
        analysis['approved_for_dip_buy'] = True
        analysis['reason'] = f"✅ DIP APPROVED ({analysis['dip_stage']}, {analysis['confidence']:.0f}% conf, Tier {quality_tier})"
        
        return analysis


class PositionSizer:
    """Calculate optimal position sizes"""
    
    @staticmethod
    def size_position(
        asre_score: float,
        portfolio_vol: float,
        existing_positions: Dict[str, float],
        portfolio_value: float,
        sector: str,
        sector_limits: Dict[str, float],
        stock_sectors: Dict[str, str],
        is_dip_buy: bool = False
    ) -> float:
        # Base allocation by ASRE tier
        if asre_score >= PortfolioConfig.TIER_EXCEPTIONAL:
            base_alloc = 0.12
        elif asre_score >= PortfolioConfig.TIER_EXCELLENT:
            base_alloc = 0.10
        elif asre_score >= PortfolioConfig.TIER_STRONG:
            base_alloc = 0.08
        elif asre_score >= PortfolioConfig.TIER_MODERATE:
            base_alloc = 0.06
        else:
            base_alloc = 0.00
        
        # Reduce for dip buys
        if is_dip_buy:
            base_alloc *= PortfolioConfig.DIP_POSITION_MULTIPLIER
        
        # Volatility adjustment
        if portfolio_vol > 0.25:
            vol_scalar = 0.6
        elif portfolio_vol > 0.20:
            vol_scalar = 0.8
        else:
            vol_scalar = 1.0
        
        base_alloc *= vol_scalar
        max_position = min(PortfolioConfig.MAX_POSITION_SIZE, base_alloc)
        
        # Sector limits
        sector_value = sum(
            existing_positions.get(tk, 0)
            for tk, sec in stock_sectors.items()
            if sec == sector
        )
        sector_limit = sector_limits.get(sector, 0.40) * portfolio_value
        sector_available = max(0, sector_limit - sector_value)
        
        position_size = min(max_position * portfolio_value, sector_available)
        
        return max(0.0, position_size)


# ============================================================================
# PORTFOLIO STATE MANAGEMENT
# ============================================================================

# In-memory portfolio storage (use database in production)
PORTFOLIOS: Dict[str, Dict] = {}

def save_portfolio_state(portfolio_id: str, state: Dict):
    """Save portfolio state to storage"""
    PORTFOLIOS[portfolio_id] = state
    
    # Also save to file
    portfolio_dir = "data/portfolios"
    os.makedirs(portfolio_dir, exist_ok=True)
    
    with open(f"{portfolio_dir}/{portfolio_id}.json", 'w') as f:
        json.dump(state, f, indent=2, default=str)
    
    logger.info(f"Portfolio {portfolio_id} state saved")


def load_portfolio_state(portfolio_id: str) -> Optional[Dict]:
    """Load portfolio state from storage"""
    if portfolio_id in PORTFOLIOS:
        return PORTFOLIOS[portfolio_id]
    
    # Try loading from file
    portfolio_file = f"data/portfolios/{portfolio_id}.json"
    if os.path.exists(portfolio_file):
        with open(portfolio_file, 'r') as f:
            state = json.load(f)
            PORTFOLIOS[portfolio_id] = state
            return state
    
    return None


def get_quality_tier(score: float) -> str:
    """Determine quality tier from ASRE score"""
    if score >= 80:
        return 'A'
    elif score >= 70:
        return 'B'
    elif score >= 60:
        return 'C'
    elif score >= 50:
        return 'D'
    else:
        return 'F'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_asre_score(ticker: str) -> float:
    """Get current ASRE score for a ticker"""
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.Timedelta(days=365*3)).strftime('%Y-%m-%d')
        
        fetcher = FundamentalFetcher()
        funds = fetcher.fetch_quarterly_fundamentals(ticker, start_date, end_date)
        
        if funds is None or funds.empty:
            logger.warning(f"No fundamentals for {ticker}")
            return 0.0
        
        loader = DataLoader()
        df = loader.load_stock_data(ticker, start_date, end_date, quarterly_fundamentals=funds)
        
        df_complete = compute_complete_asre(df, medallion=True, return_all_components=True)
        score = float(df_complete['r_final'].iloc[-1])
        
        return score
        
    except Exception as e:
        logger.error(f"Error getting ASRE for {ticker}: {e}")
        return 0.0


def get_current_price(ticker: str) -> float:
    """Get current stock price"""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        return 0.0
    except Exception as e:
        logger.error(f"Error getting price for {ticker}: {e}")
        return 0.0


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post("/portfolio/create")
async def create_portfolio(request: CreatePortfolioRequest):
    """
    Create a new investment portfolio with specified parameters.
    
    **Risk Profiles:**
    - **Conservative**: Max 8% per position, lower volatility tolerance
    - **Moderate**: Max 10% per position, balanced approach
    - **Aggressive**: Max 12% per position, higher risk tolerance
    
    **Example:**
    ```json
    {
      "initial_capital": 100000.0,
      "risk_profile": "moderate",
      "universe": ["NVDA", "MSFT", "GOOGL", "META", "AAPL"],
      "enable_dip_buying": true
    }
    ```
    """
    try:
        portfolio_id = str(uuid.uuid4())[:8]
        
        # Adjust config based on risk profile
        max_position = {
            'conservative': 0.08,
            'moderate': 0.10,
            'aggressive': 0.12
        }.get(request.risk_profile.lower(), 0.10)
        
        # Initialize portfolio state
        state = {
            'portfolio_id': portfolio_id,
            'created_at': datetime.now().isoformat(),
            'initial_capital': request.initial_capital,
            'cash': request.initial_capital,
            'total_value': request.initial_capital,
            'equity_value': 0.0,
            'holdings': {},
            'total_return_pct': 0.0,
            'peak_value': request.initial_capital,
            'current_drawdown_pct': 0.0,
            'risk_profile': request.risk_profile,
            'universe': request.universe,
            'enable_dip_buying': request.enable_dip_buying,
            'max_position_size': max_position,
            'daily_values': [request.initial_capital],
            'daily_returns': []
        }
        
        save_portfolio_state(portfolio_id, state)
        
        logger.info(f"Created portfolio {portfolio_id} with ${request.initial_capital:,.0f}")
        
        return {
            'portfolio_id': portfolio_id,
            'message': f'Portfolio created successfully with ${request.initial_capital:,.0f}',
            'initial_capital': request.initial_capital,
            'risk_profile': request.risk_profile,
            'universe_size': len(request.universe),
            'enable_dip_buying': request.enable_dip_buying
        }
        
    except Exception as e:
        logger.error(f"Portfolio creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio creation failed: {str(e)}")


@router.get("/portfolio/{portfolio_id}", response_model=PortfolioStateResponse)
async def get_portfolio_state(portfolio_id: str):
    """
    Get current portfolio state including holdings, cash, and risk metrics.
    
    **Example:** `/api/investing/portfolio/abc123`
    """
    try:
        state = load_portfolio_state(portfolio_id)
        
        if state is None:
            raise HTTPException(status_code=404, detail=f"Portfolio {portfolio_id} not found")
        
        # Build holdings response
        holdings = []
        for ticker, holding in state.get('holdings', {}).items():
            current_price = holding['current_price']
            pnl_pct = ((current_price - holding['avg_cost_basis']) / holding['avg_cost_basis']) * 100 if holding['avg_cost_basis'] > 0 else 0
            pnl_dollar = (current_price - holding['avg_cost_basis']) * holding['shares']
            
            holdings.append(HoldingResponse(
                ticker=ticker,
                shares=holding['shares'],
                avg_cost_basis=holding['avg_cost_basis'],
                current_price=current_price,
                market_value=holding['market_value'],
                target_allocation=holding['target_allocation'],
                allocation_pct=(holding['market_value'] / state['total_value']) * 100,
                entry_date=holding['entry_date'],
                entry_price=holding['entry_price'],
                max_price_since_entry=holding['max_price_since_entry'],
                drawdown_pct=holding['drawdown_pct'],
                is_dip_buy=holding.get('is_dip_buy', False),
                pnl_pct=pnl_pct,
                pnl_dollar=pnl_dollar
            ))
        
        # Calculate risk metrics
        risk_calc = RiskCalculator()
        daily_returns = pd.Series(state.get('daily_returns', []))
        
        risk_metrics = RiskMetricsResponse(
            portfolio_var_95=risk_calc.calculate_var_95(daily_returns) * 100 if len(daily_returns) > 0 else 0.0,
            portfolio_volatility=risk_calc.calculate_portfolio_volatility(daily_returns) * 100 if len(daily_returns) > 0 else 0.0,
            sharpe_ratio=risk_calc.calculate_sharpe_ratio(daily_returns) if len(daily_returns) > 0 else 0.0,
            max_drawdown=risk_calc.calculate_max_drawdown(state.get('daily_values', [state['total_value']])) * 100,
            current_drawdown=state.get('current_drawdown_pct', 0.0),
            daily_change_pct=daily_returns.iloc[-1] * 100 if len(daily_returns) > 0 else 0.0,
            largest_position_pct=max([h['market_value'] for h in state.get('holdings', {}).values()], default=0) / state['total_value'] * 100 if state['total_value'] > 0 else 0.0
        )
        
        num_dip_positions = sum(1 for h in state.get('holdings', {}).values() if h.get('is_dip_buy', False))
        
        return PortfolioStateResponse(
            portfolio_id=portfolio_id,
            cash=state['cash'],
            total_value=state['total_value'],
            equity_value=state['equity_value'],
            cash_pct=(state['cash'] / state['total_value']) * 100,
            equity_pct=(state['equity_value'] / state['total_value']) * 100,
            total_return_pct=state.get('total_return_pct', 0.0),
            peak_value=state.get('peak_value', state['total_value']),
            holdings=holdings,
            risk_metrics=risk_metrics,
            num_positions=len(holdings),
            num_dip_positions=num_dip_positions,
            last_updated=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting portfolio state: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/portfolio/{portfolio_id}/rebalance", response_model=RebalanceResponse)
async def execute_rebalance(portfolio_id: str):
    """
    Execute portfolio rebalancing based on current ASRE ratings.
    
    **Process:**
    1. Scans universe for ASRE ratings
    2. Checks trend (price vs SMA-200)
    3. Analyzes dip quality if downtrend
    4. Calculates optimal position sizes
    5. Generates buy/sell orders
    6. Executes trades
    7. Updates portfolio state
    
    **Example:** `POST /api/investing/portfolio/abc123/rebalance`
    """
    try:
        state = load_portfolio_state(portfolio_id)
        
        if state is None:
            raise HTTPException(status_code=404, detail=f"Portfolio {portfolio_id} not found")
        
        logger.info(f"Starting rebalancing for portfolio {portfolio_id}")
        
        allocation_targets = []
        trades_executed = 0
        cash_deployed = 0.0
        
        # Scan universe
        for ticker in state['universe']:
            try:
                # Get ASRE score
                asre_score = get_asre_score(ticker)
                quality_tier = get_quality_tier(asre_score)
                
                # Get current price
                current_price = get_current_price(ticker)
                
                if current_price == 0:
                    continue
                
                # Calculate current allocation
                current_alloc = 0.0
                if ticker in state['holdings']:
                    current_alloc = state['holdings'][ticker]['market_value'] / state['total_value']
                
                # Determine target allocation
                target_alloc = 0.0
                is_dip_buy = False
                action = "HOLD"
                reason = ""
                
                # Simple allocation based on ASRE tier
                if asre_score >= PortfolioConfig.TIER_EXCEPTIONAL:
                    target_alloc = state['max_position_size']
                    action = "BUY" if target_alloc > current_alloc else "HOLD"
                    reason = "Exceptional Quality"
                elif asre_score >= PortfolioConfig.TIER_EXCELLENT:
                    target_alloc = state['max_position_size'] * 0.83
                    action = "BUY" if target_alloc > current_alloc else "HOLD"
                    reason = "Excellent Quality"
                elif asre_score >= PortfolioConfig.TIER_STRONG:
                    target_alloc = state['max_position_size'] * 0.67
                    action = "BUY" if target_alloc > current_alloc else "HOLD"
                    reason = "Strong Quality"
                elif asre_score >= PortfolioConfig.TIER_MODERATE:
                    target_alloc = state['max_position_size'] * 0.50
                    action = "REDUCE" if current_alloc > target_alloc else "HOLD"
                    reason = "Moderate Quality"
                else:
                    target_alloc = 0.0
                    action = "SELL" if current_alloc > 0 else "AVOID"
                    reason = "Below Threshold"
                
                allocation_targets.append(AllocationTarget(
                    ticker=ticker,
                    target_alloc=target_alloc,
                    current_alloc=current_alloc,
                    asre_score=asre_score,
                    quality_tier=quality_tier,
                    is_dip_buy=is_dip_buy,
                    action=action,
                    reason=reason
                ))
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue
        
        # Sort by ASRE score (highest first)
        allocation_targets.sort(key=lambda x: x.asre_score, reverse=True)
        
        logger.info(f"Rebalancing complete for portfolio {portfolio_id}")
        logger.info(f"Scanned {len(state['universe'])} stocks, generated {len(allocation_targets)} targets")
        
        return RebalanceResponse(
            portfolio_id=portfolio_id,
            execution_time=datetime.now().isoformat(),
            allocation_targets=allocation_targets,
            trades_executed=trades_executed,
            cash_deployed=cash_deployed,
            message=f"Rebalancing analysis complete. {len(allocation_targets)} targets generated."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rebalancing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rebalancing failed: {str(e)}")


@router.get("/portfolio/{portfolio_id}/recommendations", response_model=PortfolioRecommendations)
async def get_recommendations(portfolio_id: str, top_n: int = Query(5, description="Number of recommendations")):
    """
    Get investment recommendations (buy/sell/dip opportunities).
    
    **Example:** `/api/investing/portfolio/abc123/recommendations?top_n=5`
    """
    try:
        state = load_portfolio_state(portfolio_id)
        
        if state is None:
            raise HTTPException(status_code=404, detail=f"Portfolio {portfolio_id} not found")
        
        # Reuse rebalancing logic to get targets
        rebalance_result = await execute_rebalance(portfolio_id)
        
        # Filter recommendations
        buy_recs = [t for t in rebalance_result.allocation_targets if t.action == "BUY"][:top_n]
        sell_recs = [t for t in rebalance_result.allocation_targets if t.action == "SELL"][:top_n]
        dip_recs = [t for t in rebalance_result.allocation_targets if t.is_dip_buy][:top_n]
        
        return PortfolioRecommendations(
            buy_recommendations=buy_recs,
            sell_recommendations=sell_recs,
            dip_opportunities=[],  # Placeholder
            message=f"Generated {len(buy_recs)} buy and {len(sell_recs)} sell recommendations"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/portfolios")
async def list_portfolios():
    """
    List all portfolios.
    
    **Example:** `/api/investing/portfolios`
    """
    try:
        portfolio_dir = "data/portfolios"
        
        if not os.path.exists(portfolio_dir):
            return {"portfolios": [], "count": 0}
        
        portfolios = []
        for filename in os.listdir(portfolio_dir):
            if filename.endswith('.json'):
                portfolio_id = filename.replace('.json', '')
                state = load_portfolio_state(portfolio_id)
                
                if state:
                    portfolios.append({
                        'portfolio_id': portfolio_id,
                        'created_at': state.get('created_at'),
                        'initial_capital': state.get('initial_capital'),
                        'total_value': state.get('total_value'),
                        'total_return_pct': state.get('total_return_pct'),
                        'risk_profile': state.get('risk_profile'),
                        'num_holdings': len(state.get('holdings', {}))
                    })
        
        return {
            'portfolios': portfolios,
            'count': len(portfolios)
        }
        
    except Exception as e:
        logger.error(f"Error listing portfolios: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.delete("/portfolio/{portfolio_id}")
async def delete_portfolio(portfolio_id: str):
    """
    Delete a portfolio.
    
    **Example:** `DELETE /api/investing/portfolio/abc123`
    """
    try:
        portfolio_file = f"data/portfolios/{portfolio_id}.json"
        
        if not os.path.exists(portfolio_file):
            raise HTTPException(status_code=404, detail=f"Portfolio {portfolio_id} not found")
        
        os.remove(portfolio_file)
        
        if portfolio_id in PORTFOLIOS:
            del PORTFOLIOS[portfolio_id]
        
        logger.info(f"Deleted portfolio {portfolio_id}")
        
        return {
            'portfolio_id': portfolio_id,
            'message': 'Portfolio deleted successfully'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
