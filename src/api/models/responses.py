"""
API Response Models
Pydantic models for all API responses.

These models define the structure and validation for data returned by API endpoints.
Used for automatic API documentation and response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class SignalType(str, Enum):
    """Stock signal types"""
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    CAUTION = "CAUTION"
    STRONG_SELL = "STRONG SELL"


class QualityTier(str, Enum):
    """Stock quality tiers"""
    S = "S"  # Exceptional
    A = "A"  # High quality
    B = "B"  # Good
    C = "C"  # Stable
    D = "D"  # Distressed


class RiskLevel(str, Enum):
    """Portfolio risk levels"""
    SAFE = "safe"
    MODERATE = "moderate"
    HIGH_RISK = "high_risk"


# ============================================================================
# STOCK RATING RESPONSES
# ============================================================================

class StockRatingResponse(BaseModel):
    """
    Complete ASRE stock rating response.
    
    Used by: GET /api/stocks/{ticker}
    """
    ticker: str = Field(..., example="NVDA", description="Stock ticker symbol")
    
    # Core ASRE Scores
    rfinal: float = Field(..., ge=0, le=100, example=89.04, description="Final ASRE rating (0-100)")
    rasre: float = Field(..., ge=0, le=100, example=90.0, description="Risk-adjusted ASRE rating")
    fscore: float = Field(..., ge=0, le=100, example=95.0, description="Fundamental score (0-100)")
    tscore: float = Field(..., ge=0, le=100, example=5.0, description="Technical score (0-100)")
    mscore: float = Field(..., ge=0, le=100, example=30.0, description="Momentum score (0-100)")
    
    # Signal & Category
    signal: str = Field(..., example="STRONG BUY", description="Buy/Sell/Hold signal")
    category: str = Field(..., example="EXCEPTIONAL GROWTH", description="Stock category")
    
    # Dip Analysis
    dip_quality: Optional[float] = Field(None, ge=0, le=100, example=100.0, description="Dip quality score")
    dip_stage: Optional[str] = Field(None, example="EARLY", description="Dip stage (EARLY/MID/LATE)")
    
    # Context Flags
    context: str = Field(default="", example="🎯 DIP", description="Context flags (DIP/TRAP/PUMP)")
    
    # Metadata
    timestamp: str = Field(..., example="2026-01-23T23:30:00", description="Rating timestamp")
    peg_ratio: Optional[float] = Field(None, example=0.73, description="Price/Earnings to Growth ratio")
    quality_tier: Optional[str] = Field(None, example="S", description="Quality tier (S/A/B/C/D)")
    close_price: Optional[float] = Field(None, example=142.50, description="Current close price")

    # Audit Proof — computed by the ASRE engine, not the frontend
    run_id: Optional[str] = Field(None, example="ASRE-NVDA-20260419-001", description="Formatted ASRE run identifier")
    score_hash: Optional[str] = Field(None, example="a3f1b2c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2", description="SHA-256 of the score DataFrame — tamper-proof audit token")

    # Optional fields
    ai_explanation: Optional[str] = Field(None, description="AI-generated explanation")
    momentum_trap: Optional[Dict[str, Any]] = Field(None, description="Momentum trap analysis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "NVDA",
                "rfinal": 89.04,
                "rasre": 90.0,
                "fscore": 95.0,
                "tscore": 5.0,
                "mscore": 30.0,
                "signal": "STRONG BUY",
                "category": "EXCEPTIONAL GROWTH",
                "dip_quality": 100.0,
                "dip_stage": "EARLY",
                "context": "🎯 DIP",
                "timestamp": "2026-01-23T23:30:00",
                "peg_ratio": 0.73,
                "quality_tier": "S",
                "close_price": 142.50
            }
        }


class StockComparison(BaseModel):
    """
    Simplified stock comparison response.
    
    Used by: POST /api/stocks/compare
    """
    rank: int = Field(..., ge=1, example=1, description="Ranking position")
    ticker: str = Field(..., example="NVDA", description="Stock ticker")
    rfinal: float = Field(..., ge=0, le=100, example=89.04, description="ASRE rating")
    signal: str = Field(..., example="STRONG BUY", description="Signal")
    fscore: Optional[float] = Field(None, ge=0, le=100, example=95.0, description="F-Score")
    category: Optional[str] = Field(None, example="EXCEPTIONAL GROWTH", description="Category")
    
    class Config:
        json_schema_extra = {
            "example": {
                "rank": 1,
                "ticker": "NVDA",
                "rfinal": 89.04,
                "signal": "STRONG BUY",
                "fscore": 95.0,
                "category": "EXCEPTIONAL GROWTH"
            }
        }


class SupportedStocksResponse(BaseModel):
    """Response for supported stocks list"""
    tickers: List[str] = Field(..., description="List of supported ticker symbols")
    count: int = Field(..., description="Total number of supported stocks")
    market: str = Field(..., example="US", description="Market region")
    last_updated: str = Field(..., example="2026-01-24", description="Last update date")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tickers": ["NVDA", "MSFT", "GOOGL", "META", "AAPL"],
                "count": 13,
                "market": "US",
                "last_updated": "2026-01-24"
            }
        }


class CompareStocksResponse(BaseModel):
    """Response for stock comparison"""
    stocks: List[Dict[str, Any]] = Field(..., description="List of compared stocks with rankings")
    count: int = Field(..., description="Number of stocks compared")
    sort_by: str = Field(..., example="rating", description="Sort criterion used")
    comparison_explanation: Optional[str] = Field(None, description="AI-generated comparison explanation")
    top_pick: Optional[Dict[str, Any]] = Field(None, description="Top-ranked stock")
    
    class Config:
        json_schema_extra = {
            "example": {
                "stocks": [
                    {"rank": 1, "ticker": "NVDA", "rfinal": 89.04, "signal": "STRONG BUY"},
                    {"rank": 2, "ticker": "META", "rfinal": 82.5, "signal": "STRONG BUY"}
                ],
                "count": 2,
                "sort_by": "rating",
                "comparison_explanation": "NVDA ranks highest with superior fundamentals...",
                "top_pick": {"ticker": "NVDA", "rfinal": 89.04, "signal": "STRONG BUY"}
            }
        }


class MomentumTrapResponse(BaseModel):
    """Response for momentum trap analysis"""
    ticker: str = Field(..., example="PUMP", description="Stock ticker symbol")
    is_trap: bool = Field(..., example=True, description="Whether stock is in momentum trap")
    severity_level: str = Field(..., example="HIGH", description="Trap severity (NONE/MILD/MODERATE/HIGH/CRITICAL)")
    severity_score: float = Field(..., ge=0, le=100, example=75.0, description="Trap severity score")
    warning: str = Field(..., example="⚠️ HIGH RISK: Stock is overbought", description="Warning message")
    explanation: str = Field(..., description="Detailed explanation of trap condition")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    visual_indicator: str = Field(..., example="🟠", description="Visual severity indicator")
    color_code: str = Field(..., example="#FF6B00", description="Color code for UI")
    ai_explanation: Optional[str] = Field(None, description="AI-generated explanation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "PUMP",
                "is_trap": True,
                "severity_level": "HIGH",
                "severity_score": 75.0,
                "warning": "⚠️ HIGH RISK: PUMP is significantly overbought",
                "explanation": "Technical score (85%) much higher than fundamental score (30%)",
                "recommendations": ["Avoid new positions", "Consider selling if profitable"],
                "risk_factors": ["Weak fundamentals", "Overbought conditions"],
                "visual_indicator": "🟠",
                "color_code": "#FF6B00"
            }
        }


# ============================================================================
# AI EXPLANATION RESPONSES
# ============================================================================

class AIExplanation(BaseModel):
    """
    AI-generated stock explanation response.
    
    Used by: POST /api/ai/explain, POST /api/ai/chat
    """
    ticker: Optional[str] = Field(None, example="NVDA", description="Stock ticker")
    explanation: str = Field(..., example="NVIDIA shows exceptional fundamentals...", description="AI explanation text")
    confidence: Optional[float] = Field(None, ge=0, le=1, example=0.95, description="Explanation confidence")
    sources: Optional[List[str]] = Field(None, example=["F-Score: 95%", "ROE: 45%"], description="Data sources used")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "NVDA",
                "explanation": "NVIDIA Corporation (NVDA) receives a STRONG BUY rating with an ASRE score of 89.04...",
                "confidence": 0.95,
                "sources": ["F-Score: 95%", "ROE: 45%", "Revenue Growth: 24%"],
                "timestamp": "2026-01-23T23:30:00"
            }
        }


class ChatResponse(BaseModel):
    """
    Conversational AI chat response.
    
    Used by: POST /api/ai/chat
    """
    message: str = Field(..., example="NVDA is a strong buy based on...", description="AI response message")
    context: Optional[Dict[str, Any]] = Field(None, description="Conversation context")
    suggestions: Optional[List[str]] = Field(None, example=["Ask about META", "Compare NVDA vs MSFT"], description="Follow-up suggestions")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# PORTFOLIO ANALYSIS RESPONSES
# ============================================================================

class RiskBreakdown(BaseModel):
    """Risk category breakdown"""
    count: int = Field(..., ge=0, example=3, description="Number of stocks in category")
    value: float = Field(..., ge=0, example=2500000, description="Total value in category")
    percentage: float = Field(..., ge=0, le=100, example=50.0, description="Percentage of portfolio")


class HighRiskStock(BaseModel):
    """High-risk stock details"""
    ticker: str = Field(..., example="META")
    value: float = Field(..., example=300000)
    shares: int = Field(..., example=450)
    rating: float = Field(..., ge=0, le=100, example=41.0)
    potential_loss: float = Field(..., example=45000, description="Estimated loss if not sold")
    action: str = Field(..., example="SELL", description="Recommended action")
    urgency: str = Field(..., example="HIGH", description="Action urgency")


class PortfolioAnalysis(BaseModel):
    """
    Complete portfolio analysis response.
    
    Used by: POST /api/portfolio/analyze
    """
    overall_score: float = Field(..., ge=0, le=100, example=72.5, description="Weighted portfolio health score")
    health_level: str = Field(..., example="GOOD", description="Health level (EXCELLENT/GOOD/FAIR/POOR/CRITICAL)")
    total_value: float = Field(..., ge=0, example=5000000, description="Total portfolio value")
    currency: str = Field(..., example="USD", description="Currency")
    holdings_count: int = Field(..., example=8, description="Number of holdings")
    
    risk_breakdown: Dict[str, RiskBreakdown] = Field(
        ...,
        description="Portfolio risk distribution"
    )
    
    high_risk_stocks: List[HighRiskStock] = Field(
        default_factory=list,
        description="List of high-risk stocks requiring attention"
    )
    
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable portfolio recommendations"
    )
    
    rebalancing: Optional[Dict[str, Any]] = Field(None, description="Rebalancing suggestions")
    portfolio_composition: Optional[Dict[str, Any]] = Field(None, description="Portfolio composition summary")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# BACKTEST RESPONSES
# ============================================================================

class TradeMetrics(BaseModel):
    """Trading metrics"""
    total_trades: int = Field(..., example=25)
    winning_trades: int = Field(..., example=15)
    losing_trades: int = Field(..., example=10)
    win_rate: float = Field(..., ge=0, le=100, example=60.0, description="Win rate percentage")
    avg_return: float = Field(..., example=12.5, description="Average return percentage")
    sharpe_ratio: float = Field(..., example=1.5, description="Sharpe ratio")
    max_drawdown: float = Field(..., example=-15.5, description="Maximum drawdown percentage")
    total_return: float = Field(..., example=35000, description="Total profit/loss")
    total_return_pct: float = Field(..., example=35.0, description="Total return percentage")
    final_capital: float = Field(..., example=135000, description="Final capital")


class BacktestResult(BaseModel):
    """
    Backtest comparison result.
    
    Used by: POST /api/backtest/upload
    """
    user_trades: List[Dict[str, Any]] = Field(..., description="User's actual trades")
    asre_trades: List[Dict[str, Any]] = Field(..., description="ASRE-recommended trades")
    
    user_metrics: TradeMetrics = Field(..., description="User's trading metrics")
    asre_metrics: TradeMetrics = Field(..., description="ASRE strategy metrics")
    
    comparison: Dict[str, Any] = Field(..., description="Comparison summary")
    
    initial_capital: float = Field(..., example=100000, description="Starting capital")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# ALERT RESPONSES
# ============================================================================

class AlertSimulation(BaseModel):
    """
    Market crash alert simulation.
    
    Used by: POST /api/alerts/simulate
    """
    alert_type: str = Field(..., example="MARKET_CRASH", description="Alert type")
    message: str = Field(..., example="Market crash detected!", description="Alert message")
    affected_stocks: List[Dict[str, Any]] = Field(
        ...,
        description="List of affected stocks with crash data"
    )
    severity: str = Field(..., example="MODERATE", description="Crash severity")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# ERROR RESPONSES
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response"""
    detail: str = Field(..., example="Stock 'XYZ' is not supported", description="Error message")
    timestamp: Optional[str] = Field(None, description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Stock 'XYZ' is not supported. Supported stocks: NVDA, MSFT, GOOGL...",
                "timestamp": "2026-01-24T00:30:00"
            }
        }


# ============================================================================
# HEALTH CHECK RESPONSES
# ============================================================================

class HealthCheck(BaseModel):
    """API health check response"""
    status: str = Field(..., example="healthy")
    asre_available: bool = Field(..., example=True)
    cache_size: int = Field(..., example=5)
    supported_stocks_count: int = Field(..., example=13)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Stock responses
    "StockRatingResponse",
    "StockComparison",
    "SupportedStocksResponse",
    "CompareStocksResponse",
    "MomentumTrapResponse",
    
    # AI responses
    "AIExplanation",
    "ChatResponse",
    
    # Portfolio responses
    "PortfolioAnalysis",
    "RiskBreakdown",
    "HighRiskStock",
    
    # Backtest responses
    "BacktestResult",
    "TradeMetrics",
    
    # Alert responses
    "AlertSimulation",
    
    # General responses
    "ErrorResponse",
    "HealthCheck",
    
    # Enums
    "SignalType",
    "QualityTier",
    "RiskLevel",
]
