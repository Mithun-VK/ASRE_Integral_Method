"""
AI Chat Routes
==============
AI-powered ASRE explanations using Groq LLM.

Directly imports from existing ai_explainer.py:
- Natural language rating explanations
- Momentum trap warnings
- Conversational Q&A
- Stock comparisons

Uses Groq AI for intelligent, context-aware responses.

Author: ASRE Team
Version: 2.0.0
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any  # ✅ ADDED Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import existing AI explainer service
try:
    from api.services.ai_explainer import (
        AIExplainer,
        get_explainer,
        explain_rating,
        explain_trap
    )
    AI_EXPLAINER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"AI Explainer not available: {e}")
    AI_EXPLAINER_AVAILABLE = False

# Import ASRE service for ratings
try:
    from api.services.asre_service import ASREService
    ASRE_SERVICE_AVAILABLE = True
except ImportError:
    ASRE_SERVICE_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai-chat", tags=["ai-chat"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ExplainRatingRequest(BaseModel):
    """Request to explain ASRE rating"""
    ticker: str = Field(..., description="Stock ticker")
    include_trap_analysis: bool = Field(True, description="Include momentum trap analysis")


class TrapAnalysisDetail(BaseModel):
    """Momentum trap analysis details"""
    is_trap: bool
    severity: str
    severity_score: float
    warning: str
    explanation: str
    recommendations: List[str]
    risk_factors: List[str]
    visual_indicator: str
    color_code: str


class ExplainRatingResponse(BaseModel):
    """AI explanation of ASRE rating"""
    ticker: str
    explanation: str
    rating: float
    signal: str
    category: str
    scores: Dict[str, float]
    trap_analysis: Optional[TrapAnalysisDetail]
    timestamp: str
    ai_generated: bool


class ExplainTrapRequest(BaseModel):
    """Request for momentum trap analysis"""
    ticker: str = Field(..., description="Stock ticker")


class ExplainTrapResponse(BaseModel):
    """Momentum trap explanation"""
    ticker: str
    is_trap: bool
    severity: str
    severity_score: Optional[float]
    warning: str
    explanation: str
    recommendations: List[str]
    risk_factors: Optional[List[str]]
    visual_indicator: Optional[str]
    color_code: Optional[str]


class ChatRequest(BaseModel):
    """Chat with AI about stocks"""
    message: str = Field(..., description="Your question")
    ticker: Optional[str] = Field(None, description="Stock ticker for context")
    context: Optional[Dict[str, Any]] = Field(None, description="Previous conversation context")  # ✅ FIXED


class ChatResponse(BaseModel):
    """AI chat response"""
    message: str
    ticker: Optional[str]
    context: Optional[Dict[str, Any]]  # ✅ FIXED
    timestamp: str
    suggestions: List[str]


class CompareStocksRequest(BaseModel):
    """Compare multiple stocks"""
    tickers: List[str] = Field(..., min_items=2, max_items=10, description="List of tickers to compare")
    focus: str = Field("rating", description="Focus area: rating, fundamentals, risk")


class TopPickDetail(BaseModel):  # ✅ NEW MODEL
    """Top pick details"""
    ticker: str
    rating: float
    signal: str


class CompareStocksResponse(BaseModel):
    """Stock comparison results"""
    explanation: str
    top_pick: TopPickDetail  # ✅ FIXED - Use proper model instead of Dict
    comparison: List[str]
    timestamp: str


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post("/explain", response_model=ExplainRatingResponse)
def explain_stock_rating(request: ExplainRatingRequest):
    """
    Get AI-powered explanation of ASRE rating in plain English.
    
    **Uses Groq LLM** for natural language explanations optimized for retail investors.
    
    **Features:**
    - Simple, jargon-free language
    - Actionable insights
    - Risk warnings
    - Optional momentum trap analysis
    
    **Example:**
    ```json
    {
      "ticker": "NVDA",
      "include_trap_analysis": true
    }
    ```
    """
    try:
        if not AI_EXPLAINER_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="AI Explainer service not available. Check if ai_explainer.py is accessible."
            )
        
        logger.info(f"Generating AI explanation for {request.ticker}")
        
        # Get ASRE rating data
        if ASRE_SERVICE_AVAILABLE:
            asre_service = ASREService()
            asre_data = asre_service.get_stock_rating(request.ticker)
        else:
            # Mock data for testing (replace with actual service call)
            asre_data = {
                'rfinal': 85.5,
                'fscore': 82.0,
                'tscore': 88.0,
                'mscore': 86.0,
                'signal': 'STRONG_BUY',
                'category': 'HIGH_QUALITY',
                'dip_quality': None,
                'dip_stage': None
            }
        
        # Get AI explainer
        explainer = get_explainer()
        
        # Generate explanation
        result = explainer.explain_stock_rating(
            ticker=request.ticker,
            asre_data=asre_data,
            include_trap_analysis=request.include_trap_analysis
        )
        
        # Format trap analysis
        trap_detail = None
        if result.get('trap_analysis'):
            trap = result['trap_analysis']
            trap_detail = TrapAnalysisDetail(
                is_trap=trap.get('is_trap', False),
                severity=trap.get('severity_level', 'NONE'),
                severity_score=trap.get('severity_score', 0.0),
                warning=trap.get('warning', ''),
                explanation=trap.get('explanation', ''),
                recommendations=trap.get('recommendations', []),
                risk_factors=trap.get('risk_factors', []),
                visual_indicator=trap.get('visual_indicator', ''),
                color_code=trap.get('color_code', '')
            )
        
        return ExplainRatingResponse(
            ticker=result['ticker'],
            explanation=result['explanation'],
            rating=result['rating'],
            signal=result['signal'],
            category=result['category'],
            scores=result['scores'],
            trap_analysis=trap_detail,
            timestamp=result['timestamp'],
            ai_generated=result['ai_generated']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI explanation failed for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.post("/explain-trap", response_model=ExplainTrapResponse)
def explain_momentum_trap(request: ExplainTrapRequest):
    """
    Get AI explanation of momentum trap risk.
    
    **Momentum Trap** = Stock looks great technically but has weak fundamentals.
    
    **Severity Levels:**
    - **CRITICAL**: Very high risk (F-Score < 40, T-Score > 70)
    - **HIGH**: High risk
    - **MODERATE**: Medium risk
    - **LOW**: Low risk
    - **NONE**: No trap detected
    
    **Example:**
    ```json
    {
      "ticker": "NVDA"
    }
    ```
    """
    try:
        if not AI_EXPLAINER_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="AI Explainer service not available"
            )
        
        logger.info(f"Analyzing momentum trap for {request.ticker}")
        
        # Get ASRE scores
        if ASRE_SERVICE_AVAILABLE:
            asre_service = ASREService()
            asre_data = asre_service.get_stock_rating(request.ticker)
            fscore = asre_data.get('fscore', 0)
            tscore = asre_data.get('tscore', 0)
            mscore = asre_data.get('mscore', 0)
        else:
            # Mock data
            fscore = 45.0  # Weak fundamentals
            tscore = 88.0  # Strong technicals
            mscore = 82.0  # Strong momentum
        
        # Get AI explainer
        explainer = get_explainer()
        
        # Generate trap explanation
        result = explainer.explain_momentum_trap(
            ticker=request.ticker,
            fscore=fscore,
            tscore=tscore,
            mscore=mscore
        )
        
        return ExplainTrapResponse(
            ticker=result['ticker'],
            is_trap=result['is_trap'],
            severity=result['severity'],
            severity_score=result.get('severity_score'),
            warning=result['warning'],
            explanation=result['explanation'],
            recommendations=result['recommendations'],
            risk_factors=result.get('risk_factors'),
            visual_indicator=result.get('visual_indicator'),
            color_code=result.get('color_code')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trap analysis failed for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
def chat_with_ai(request: ChatRequest):
    """
    Ask AI questions about stocks and ASRE ratings.
    
    **Conversational AI** powered by Groq LLM.
    
    **Example Questions:**
    - "What does the F-score measure?"
    - "Is NVDA a good buy right now?"
    - "What's a momentum trap?"
    - "Should I invest in tech stocks?"
    
    **Example:**
    ```json
    {
      "message": "What does the F-score measure?",
      "ticker": "NVDA",
      "context": {
        "previous_rating": 85.5,
        "previous_signal": "STRONG_BUY"
      }
    }
    ```
    """
    try:
        if not AI_EXPLAINER_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="AI Explainer service not available"
            )
        
        logger.info(f"Processing chat: {request.message[:50]}...")
        
        # Get AI explainer
        explainer = get_explainer()
        
        # Generate response
        result = explainer.chat_with_ai(
            message=request.message,
            context=request.context,
            ticker=request.ticker
        )
        
        return ChatResponse(
            message=result['message'],
            ticker=result.get('ticker'),
            context=result.get('context'),
            timestamp=result['timestamp'],
            suggestions=result.get('suggestions', [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.post("/compare", response_model=CompareStocksResponse)
def compare_stocks(request: CompareStocksRequest):
    """
    Compare multiple stocks with AI insights.
    
    **Focus Areas:**
    - **rating**: Overall ASRE rating comparison
    - **fundamentals**: F-Score comparison (business quality)
    - **risk**: Momentum trap and risk analysis
    
    **Example:**
    ```json
    {
      "tickers": ["NVDA", "AMD", "INTC"],
      "focus": "rating"
    }
    ```
    """
    try:
        if not AI_EXPLAINER_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="AI Explainer service not available"
            )
        
        logger.info(f"Comparing {len(request.tickers)} stocks: {', '.join(request.tickers)}")
        
        # Get ratings for all stocks
        stock_list = []
        
        if ASRE_SERVICE_AVAILABLE:
            asre_service = ASREService()
            for ticker in request.tickers:
                try:
                    asre_data = asre_service.get_stock_rating(ticker)
                    stock_list.append({
                        'ticker': ticker,
                        'rfinal': asre_data.get('rfinal', 0),
                        'fscore': asre_data.get('fscore', 0),
                        'tscore': asre_data.get('tscore', 0),
                        'mscore': asre_data.get('mscore', 0),
                        'signal': asre_data.get('signal', 'UNKNOWN')
                    })
                except Exception as e:
                    logger.error(f"Error getting rating for {ticker}: {e}")
        else:
            # Mock data
            mock_scores = {
                'NVDA': {'rfinal': 85.5, 'fscore': 82, 'tscore': 88, 'mscore': 86, 'signal': 'STRONG_BUY'},
                'AMD': {'rfinal': 78.2, 'fscore': 75, 'tscore': 80, 'mscore': 79, 'signal': 'BUY'},
                'INTC': {'rfinal': 62.1, 'fscore': 65, 'tscore': 60, 'mscore': 61, 'signal': 'HOLD'}
            }
            stock_list = [
                {**{'ticker': ticker}, **mock_scores.get(ticker, {'rfinal': 70, 'fscore': 70, 'tscore': 70, 'mscore': 70, 'signal': 'HOLD'})}
                for ticker in request.tickers
            ]
        
        # Need at least 2 successfully-rated stocks to compare
        if len(stock_list) < 2:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Comparison needs at least 2 stocks that could be rated; "
                    f"only {len(stock_list)} of {len(request.tickers)} succeeded. "
                    "Check the tickers are supported and data is available."
                ),
            )

        # Sort by rating
        stock_list.sort(key=lambda x: x['rfinal'], reverse=True)

        # Get AI explainer
        explainer = get_explainer()

        # Generate comparison
        result = explainer.explain_comparison(
            stock_list=stock_list,
            focus=request.focus
        )

        # ✅ FIXED - Properly format top_pick
        top_pick_data = result['top_pick']
        top_pick = TopPickDetail(
            ticker=top_pick_data['ticker'],
            rating=top_pick_data['rating'],
            signal=top_pick_data['signal']
        )
        
        return CompareStocksResponse(
            explanation=result['explanation'],
            top_pick=top_pick,  # ✅ Use proper model
            comparison=result['comparison'],
            timestamp=result['timestamp']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.get("/quick-explain/{ticker}")
def quick_explain(ticker: str):
    """
    Quick AI explanation with default settings.
    
    **Example:** `/api/ai-chat/quick-explain/NVDA`
    """
    try:
        request = ExplainRatingRequest(ticker=ticker, include_trap_analysis=True)
        return explain_stock_rating(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick explain failed: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Check if AI services are available.
    
    **Returns:**
    - AI Explainer status
    - Groq API status
    - ASRE Service status
    """
    try:
        status = {
            'ai_explainer_available': AI_EXPLAINER_AVAILABLE,
            'asre_service_available': ASRE_SERVICE_AVAILABLE,
            'groq_configured': False
        }
        
        if AI_EXPLAINER_AVAILABLE:
            explainer = get_explainer()
            status['groq_configured'] = explainer.ai_client.client is not None
        
        return {
            'status': 'healthy' if AI_EXPLAINER_AVAILABLE else 'degraded',
            'services': status,
            'message': 'AI chat services operational' if AI_EXPLAINER_AVAILABLE else 'AI services unavailable'
        }
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'message': 'Health check failed'
        }
