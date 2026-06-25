"""
API Request Models
Pydantic models for all API request payloads.

These models define the structure and validation for data sent to API endpoints.
Used for automatic request validation and API documentation.
"""

import re
from pydantic import BaseModel, Field, validator, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class AlertType(str, Enum):
    """Alert notification types"""
    RATING_DROP = "rating_drop"
    RATING_RISE = "rating_rise"
    MARKET_CRASH = "market_crash"
    DIP_OPPORTUNITY = "dip_opportunity"
    MOMENTUM_TRAP = "momentum_trap"


class TradeAction(str, Enum):
    """Trade action types for backtest CSV"""
    BUY = "BUY"
    SELL = "SELL"


# ============================================================================
# STOCK COMPARISON REQUEST
# ============================================================================

class CompareStocksRequest(BaseModel):
    """
    Request to compare multiple stocks.
    
    Used by: POST /api/stocks/compare
    """
    tickers: List[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        example=["NVDA", "MSFT", "GOOGL"],
        description="List of stock tickers to compare (2-10 stocks)"
    )
    
    include_details: bool = Field(
        default=False,
        description="Include detailed F/T/M scores in response"
    )

    sort_by: str = Field(
        default="rating",
        description="Sort/ranking criterion: 'rating', 'fscore', or 'momentum'"
    )

    @field_validator('sort_by')
    @classmethod
    def validate_sort_by(cls, v):
        """Validate the sort criterion, defaulting unknown values to 'rating'."""
        allowed = {"rating", "fscore", "momentum"}
        v = (v or "rating").lower().strip()
        return v if v in allowed else "rating"

    @field_validator('tickers')
    @classmethod
    def validate_tickers(cls, v):
        """Validate ticker format and uniqueness"""
        if not v:
            raise ValueError("At least 2 tickers required")
        
        # Convert to uppercase and remove duplicates
        tickers = [ticker.upper().strip() for ticker in v]
        unique_tickers = list(dict.fromkeys(tickers))  # Preserve order
        
        if len(unique_tickers) < 2:
            raise ValueError("At least 2 unique tickers required")
        
        if len(unique_tickers) > 10:
            raise ValueError("Maximum 10 tickers allowed")
        
        # Validate ticker format (alphanumeric, 1-5 chars)
        for ticker in unique_tickers:
            if not re.match(r"^[A-Z][A-Z0-9]*(\.[A-Z]{1,3})?$", ticker) or len(ticker) < 1 or len(ticker) > 20:
                raise ValueError(f"Invalid ticker format: {ticker}")
        
        return unique_tickers
    
    class Config:
        json_schema_extra = {
            "example": {
                "tickers": ["NVDA", "MSFT", "GOOGL"],
                "include_details": True
            }
        }


# ============================================================================
# PORTFOLIO ANALYSIS REQUEST
# ============================================================================

class Holding(BaseModel):
    """Individual stock holding"""
    ticker: str = Field(..., example="NVDA", description="Stock ticker symbol")
    shares: int = Field(..., gt=0, example=100, description="Number of shares owned")
    value: float = Field(..., gt=0, example=14250.00, description="Current market value")
    avg_buy_price: Optional[float] = Field(None, example=120.50, description="Average purchase price")
    
    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v):
        """Validate ticker format"""
        ticker = v.upper().strip()
        if not re.match(r"^[A-Z][A-Z0-9]*(\.[A-Z]{1,3})?$", ticker) or len(ticker) < 1 or len(ticker) > 20:
            raise ValueError(f"Invalid ticker format: {ticker}")
        return ticker
    
    @field_validator('value')
    @classmethod
    def validate_value(cls, v):
        """Ensure value is positive"""
        if v <= 0:
            raise ValueError("Holding value must be positive")
        return round(v, 2)


class PortfolioRequest(BaseModel):
    """
    Request to analyze portfolio health.
    
    Used by: POST /api/portfolio/analyze
    """
    holdings: List[Holding] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of stock holdings (1-50 stocks)"
    )
    
    currency: str = Field(
        default="USD",
        example="USD",
        description="Portfolio currency (USD, INR, etc.)"
    )
    
    risk_tolerance: Optional[str] = Field(
        default="moderate",
        example="moderate",
        description="Risk tolerance: conservative, moderate, aggressive"
    )
    
    @field_validator('holdings')
    @classmethod
    def validate_holdings(cls, v):
        """Validate portfolio holdings"""
        if not v:
            raise ValueError("At least 1 holding required")
        
        if len(v) > 50:
            raise ValueError("Maximum 50 holdings allowed")
        
        # Check for duplicate tickers
        tickers = [h.ticker for h in v]
        if len(tickers) != len(set(tickers)):
            raise ValueError("Duplicate tickers found in portfolio")
        
        # Validate total portfolio value
        total_value = sum(h.value for h in v)
        if total_value <= 0:
            raise ValueError("Total portfolio value must be positive")
        
        return v
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value"""
        return sum(h.value for h in self.holdings)
    
    class Config:
        json_schema_extra = {
            "example": {
                "holdings": [
                    {"ticker": "NVDA", "shares": 100, "value": 14250.00, "avg_buy_price": 120.50},
                    {"ticker": "MSFT", "shares": 50, "value": 21500.00, "avg_buy_price": 400.00},
                    {"ticker": "GOOGL", "shares": 75, "value": 10500.00, "avg_buy_price": 135.00}
                ],
                "currency": "USD",
                "risk_tolerance": "moderate"
            }
        }


# ============================================================================
# BACKTEST UPLOAD REQUEST
# ============================================================================

class Trade(BaseModel):
    """Individual trade record"""
    date: str = Field(..., example="2024-01-15", description="Trade date (YYYY-MM-DD)")
    ticker: str = Field(..., example="NVDA", description="Stock ticker")
    action: TradeAction = Field(..., example="BUY", description="BUY or SELL")
    price: float = Field(..., gt=0, example=142.50, description="Trade price")
    quantity: int = Field(..., gt=0, example=100, description="Number of shares")
    
    @field_validator('date')
    @classmethod
    def validate_date(cls, v):
        """Validate date format"""
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use YYYY-MM-DD")
    
    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v):
        """Validate ticker format"""
        ticker = v.upper().strip()
        if not re.match(r"^[A-Z][A-Z0-9]*(\.[A-Z]{1,3})?$", ticker) or len(ticker) < 1 or len(ticker) > 20:
            raise ValueError(f"Invalid ticker format: {ticker}")
        return ticker


class BacktestUploadRequest(BaseModel):
    """
    Request to upload trading history for backtest comparison.
    
    Used by: POST /api/backtest/upload
    """
    trades: List[Trade] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of historical trades (1-1000 trades)"
    )
    
    start_date: Optional[str] = Field(
        None,
        example="2023-01-01",
        description="Backtest start date (YYYY-MM-DD)"
    )
    
    end_date: Optional[str] = Field(
        None,
        example="2024-01-01",
        description="Backtest end date (YYYY-MM-DD)"
    )
    
    initial_capital: float = Field(
        default=100000.0,
        gt=0,
        example=100000.0,
        description="Initial capital amount"
    )
    
    generate_pdf: bool = Field(
        default=True,
        description="Generate PDF report"
    )
    
    @field_validator('trades')
    @classmethod
    def validate_trades(cls, v):
        """Validate trade list"""
        if not v:
            raise ValueError("At least 1 trade required")
        
        if len(v) > 1000:
            raise ValueError("Maximum 1000 trades allowed")
        
        # Sort trades by date
        try:
            sorted_trades = sorted(v, key=lambda t: datetime.strptime(t.date, "%Y-%m-%d"))
            return sorted_trades
        except Exception as e:
            raise ValueError(f"Error sorting trades: {e}")
    
    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v):
        """Validate date format if provided"""
        if v is None:
            return v
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use YYYY-MM-DD")
    
    class Config:
        json_schema_extra = {
            "example": {
                "trades": [
                    {"date": "2023-01-15", "ticker": "NVDA", "action": "BUY", "price": 142.50, "quantity": 100},
                    {"date": "2023-06-20", "ticker": "NVDA", "action": "SELL", "price": 195.30, "quantity": 100}
                ],
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "initial_capital": 100000.0,
                "generate_pdf": True
            }
        }


# ============================================================================
# AI CHAT REQUEST
# ============================================================================

class AIChatRequest(BaseModel):
    """
    Request for AI chat/explanation.
    
    Used by: POST /api/ai/chat, POST /api/ai/explain
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        example="Why is NVDA rated 89? Should I buy now?",
        description="User question or prompt (1-500 characters)"
    )
    
    ticker: Optional[str] = Field(
        None,
        example="NVDA",
        description="Stock ticker for context (optional)"
    )
    
    context: Optional[Dict[str, Any]] = Field(
        None,
        example={"previous_question": "Tell me about NVDA"},
        description="Conversation context (for multi-turn chat)"
    )
    
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        example=0.7,
        description="AI creativity level (0.0-1.0)"
    )
    
    max_tokens: Optional[int] = Field(
        default=500,
        ge=50,
        le=2000,
        example=500,
        description="Maximum response length (50-2000 tokens)"
    )
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        """Validate and clean message"""
        message = v.strip()
        if not message:
            raise ValueError("Message cannot be empty")
        if len(message) > 500:
            raise ValueError("Message too long (max 500 characters)")
        return message
    
    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v):
        """Validate ticker format if provided"""
        if v is None:
            return v
        ticker = v.upper().strip()
        if not re.match(r"^[A-Z][A-Z0-9]*(\.[A-Z]{1,3})?$", ticker) or len(ticker) < 1 or len(ticker) > 20:
            raise ValueError(f"Invalid ticker format: {ticker}")
        return ticker
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Why is NVDA rated 89? Should I buy now?",
                "ticker": "NVDA",
                "context": {"previous_rating": 89.04},
                "temperature": 0.7,
                "max_tokens": 500
            }
        }


# ============================================================================
# ALERT REQUEST
# ============================================================================

class AlertRequest(BaseModel):
    """
    Request to register or simulate an alert.
    
    Used by: POST /api/alerts/simulate, POST /api/alerts/register
    """
    alert_type: AlertType = Field(
        ...,
        example="market_crash",
        description="Type of alert to trigger"
    )
    
    tickers: Optional[List[str]] = Field(
        None,
        example=["NVDA", "MSFT"],
        description="Tickers to monitor (optional, defaults to all)"
    )
    
    threshold: Optional[float] = Field(
        None,
        ge=-100,
        le=100,
        example=-20.0,
        description="Alert threshold (e.g., -20 for 20% drop)"
    )
    
    phone_number: Optional[str] = Field(
        None,
        example="+1234567890",
        description="Phone number for WhatsApp alerts (optional)"
    )
    
    simulate_only: bool = Field(
        default=True,
        description="Simulate alert without sending (demo mode)"
    )
    
    @field_validator('tickers')
    @classmethod
    def validate_tickers(cls, v):
        """Validate ticker list if provided"""
        if v is None:
            return v
        
        tickers = [ticker.upper().strip() for ticker in v]
        unique_tickers = list(dict.fromkeys(tickers))
        
        if len(unique_tickers) > 20:
            raise ValueError("Maximum 20 tickers allowed for alerts")
        
        for ticker in unique_tickers:
            if not re.match(r"^[A-Z][A-Z0-9]*(\.[A-Z]{1,3})?$", ticker) or len(ticker) < 1 or len(ticker) > 20:
                raise ValueError(f"Invalid ticker format: {ticker}")
        
        return unique_tickers
    
    @field_validator('phone_number')
    @classmethod
    def validate_phone(cls, v):
        """Validate phone number format if provided"""
        if v is None:
            return v
        
        # Remove spaces and dashes
        phone = v.replace(" ", "").replace("-", "")
        
        # Check if it starts with + and contains only digits after
        if not phone.startswith("+") or not phone[1:].isdigit():
            raise ValueError("Phone number must be in format: +1234567890")
        
        if len(phone) < 10 or len(phone) > 15:
            raise ValueError("Phone number must be 10-15 digits")
        
        return phone
    
    class Config:
        json_schema_extra = {
            "example": {
                "alert_type": "market_crash",
                "tickers": ["NVDA", "MSFT", "GOOGL"],
                "threshold": -20.0,
                "phone_number": "+1234567890",
                "simulate_only": True
            }
        }

class PortfolioHolding(BaseModel):
    """Single portfolio holding"""
    ticker: str = Field(
        ...,
        min_length=1,
        max_length=20,
        example="NVDA",
        description="Stock ticker symbol"
    )
    shares: float = Field(
        ...,
        gt=0,
        example=100,
        description="Number of shares held"
    )
    value: float = Field(
        ...,
        gt=0,
        example=14250.00,
        description="Current market value"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "NVDA",
                "shares": 100,
                "value": 14250.00
            }
        }


class PortfolioAnalysisRequest(BaseModel):
    """Request for portfolio analysis"""
    holdings: List[PortfolioHolding] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="List of portfolio holdings"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "holdings": [
                    {"ticker": "NVDA", "shares": 100, "value": 14250.00},
                    {"ticker": "MSFT", "shares": 50, "value": 21500.00},
                    {"ticker": "GOOGL", "shares": 75, "value": 10500.00}
                ]
            }
        }

# ============================================================================
# BULK RATING REQUEST (Optional)
# ============================================================================

class BulkRatingRequest(BaseModel):
    """
    Request to get ratings for multiple stocks in one call.
    
    Used by: POST /api/stocks/bulk (optional endpoint)
    """
    tickers: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        example=["NVDA", "MSFT", "GOOGL"],
        description="List of tickers (1-20 stocks)"
    )
    
    force_refresh: bool = Field(
        default=False,
        description="Force refresh data (ignore cache)"
    )
    
    @field_validator('tickers')
    @classmethod
    def validate_tickers(cls, v):
        """Validate ticker list"""
        tickers = [ticker.upper().strip() for ticker in v]
        unique_tickers = list(dict.fromkeys(tickers))
        
        if len(unique_tickers) > 20:
            raise ValueError("Maximum 20 tickers allowed")
        
        for ticker in unique_tickers:
            if not re.match(r"^[A-Z][A-Z0-9]*(\.[A-Z]{1,3})?$", ticker) or len(ticker) < 1 or len(ticker) > 20:
                raise ValueError(f"Invalid ticker format: {ticker}")
        
        return unique_tickers


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Stock requests
    "CompareStocksRequest",
    "BulkRatingRequest",
    
    # Portfolio requests
    "PortfolioRequest",
    "Holding",
    "PortfolioHolding",
    "PortfolioAnalysisRequest",
    
    # Backtest requests
    "BacktestUploadRequest",
    "Trade",
    
    # AI requests
    "AIChatRequest",
    
    # Alert requests
    "AlertRequest",
    
    # Enums
    "AlertType",
    "TradeAction",
]
