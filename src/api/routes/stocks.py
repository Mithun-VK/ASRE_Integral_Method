"""
Stock Rating Routes
API endpoints for stock ratings and comparisons.

Endpoints:
- GET /api/stocks - List supported stocks
- GET /api/stocks/{ticker} - Get ASRE rating for a stock
- POST /api/stocks/compare - Compare multiple stocks
- GET /api/stocks/{ticker}/history - Get rating history (cached)
- GET /api/stocks/momentum-trap/{ticker} - Check momentum trap
"""

from fastapi import APIRouter, HTTPException, Query, Path
import logging

from api.services.asre_service import ASREService
from api.services.momentum_trap import analyze_trap_risk
from api.services.ai_explainer import AIExplainer
from api.models.responses import (
    StockRatingResponse,
    CompareStocksResponse,
    MomentumTrapResponse,
    ErrorResponse,
    SupportedStocksResponse
)
from api.models.requests import CompareStocksRequest
from api.config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/stocks", tags=["stocks"])


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get(
    "",
    response_model=SupportedStocksResponse,
    summary="List supported stocks",
    description="Get list of all supported stock tickers"
)
async def list_supported_stocks():
    """
    Get list of all stocks supported by ASRE.
    
    Returns a list of ticker symbols that can be analyzed.
    """
    return {
        "tickers": settings.SUPPORTED_STOCKS,
        "count": len(settings.SUPPORTED_STOCKS),
        "market": "IN",
        "exchange": "NSE",
        "last_updated": "2026-04-19"
    }


# NOTE: static single-segment routes (e.g. /health) MUST be declared before
# the parametrized "/{ticker}" route below, otherwise FastAPI matches them as
# a ticker and rejects them with a 422 pattern-mismatch.
@router.get(
    "/health",
    summary="Service health check",
    description="Check if stock rating service is operational"
)
async def health_check():
    """
    Health check endpoint.

    **Returns:**
    Service health status including:
    - ASRE availability
    - Cache status
    - Supported stocks count
    """
    health = ASREService.health_check()
    return {
        **health,
        "status": "healthy" if health['asre_available'] else "degraded",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }


@router.get(
    "/{ticker}",
    response_model=StockRatingResponse,
    summary="Get stock rating",
    description="Get comprehensive ASRE rating for a single stock",
    responses={
        200: {"description": "Successfully retrieved rating"},
        404: {"model": ErrorResponse, "description": "Stock not found"},
        400: {"model": ErrorResponse, "description": "Invalid ticker format"}
    }
)
def get_stock_rating(
    ticker: str = Path(
        ...,
        description="Stock ticker symbol (e.g., NVDA, MSFT)",
        min_length=1,
        max_length=20,
        regex="^[A-Z][A-Z0-9]*(\.[A-Z]{1,3})?$"
    ),
    force_refresh: bool = Query(
        False,
        description="Force refresh data (bypass cache)"
    ),
    include_explanation: bool = Query(
        False,
        description="Include AI-generated explanation"
    ),
    include_trap_analysis: bool = Query(
        True,
        description="Include momentum trap analysis"
    )
):
    """
    Get comprehensive ASRE rating for a stock.
    
    **Parameters:**
    - **ticker**: Stock ticker symbol (uppercase, 1-5 characters)
    - **force_refresh**: Set to `true` to bypass cache and fetch fresh data
    - **include_explanation**: Set to `true` to get AI explanation of the rating
    - **include_trap_analysis**: Set to `true` to include momentum trap risk
    
    **Returns:**
    Complete ASRE rating including:
    - Overall rating (0-100)
    - Buy/Sell signal
    - Component scores (F-Score, T-Score, M-Score)
    - Category classification
    - Dip analysis
    - Optional AI explanation
    - Optional momentum trap warning
    
    **Example:**
    ```
    GET /api/stocks/NVDA?force_refresh=true&include_explanation=true
    ```
    """
    ticker = ticker.upper().strip()

    # ── Guard: reject unsupported tickers immediately (before expensive ASRE call) ──
    if ticker not in settings.SUPPORTED_STOCKS:
        supported_str = ", ".join(settings.SUPPORTED_STOCKS)
        raise HTTPException(
            status_code=404,
            detail=(
                f"Ticker '{ticker}' is not supported. "
                f"Supported stocks: {supported_str}"
            )
        )

    logger.info(f"Fetching rating for {ticker} (force_refresh={force_refresh})")

    try:
        # Get ASRE rating
        rating_data = ASREService.get_stock_rating(
            ticker,
            force_refresh=force_refresh
        )
        
        # Add AI explanation if requested
        if include_explanation:
            try:
                explainer = AIExplainer()
                explanation = explainer.explain_stock_rating(
                    ticker,
                    rating_data,
                    include_trap_analysis=False  # We'll add it separately
                )
                rating_data['ai_explanation'] = explanation['explanation']
            except Exception as e:
                logger.warning(f"Failed to generate explanation: {e}")
                rating_data['ai_explanation'] = None
        
        # Add momentum trap analysis if requested
        if include_trap_analysis:
            try:
                trap_data = analyze_trap_risk(
                    ticker,
                    rating_data['fscore'],
                    rating_data['tscore'],
                    rating_data['mscore'],
                    rating_data['rfinal']
                )
                rating_data['momentum_trap'] = trap_data
            except Exception as e:
                logger.warning(f"Failed to analyze trap: {e}")
                rating_data['momentum_trap'] = None
        
        return rating_data
        
    except ValueError as e:
        # Invalid ticker or not supported
        raise HTTPException(
            status_code=404 if "not supported" in str(e) else 400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to get rating for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process {ticker}: {str(e)}"
        )


@router.post(
    "/compare",
    response_model=CompareStocksResponse,
    summary="Compare multiple stocks",
    description="Compare and rank multiple stocks by ASRE rating"
)
def compare_stocks(
    request: CompareStocksRequest,
    include_explanations: bool = Query(
        False,
        description="Include AI comparison explanation"
    )
):
    """
    Compare multiple stocks and rank by ASRE rating.
    
    **Request Body:**
    ```json
    {
      "tickers": ["NVDA", "MSFT", "GOOGL"],
      "sort_by": "rating"
    }
    ```
    
    **Parameters:**
    - **tickers**: List of 2-10 stock tickers to compare
    - **sort_by**: Sort criterion (rating, fscore, momentum)
    - **include_explanations**: Get AI explanation of comparison
    
    **Returns:**
    Ranked list of stocks with:
    - Full ASRE ratings
    - Rank position
    - Comparison metrics
    - Optional AI explanation
    
    **Example:**
    ```
    POST /api/stocks/compare
    {
      "tickers": ["NVDA", "MSFT", "GOOGL", "META"],
      "sort_by": "rating"
    }
    ```
    """
    tickers = [t.upper().strip() for t in request.tickers]
    
    # Validation
    if len(tickers) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 stocks required for comparison"
        )
    
    if len(tickers) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 stocks allowed per comparison"
        )
    
    logger.info(f"Comparing {len(tickers)} stocks: {', '.join(tickers)}")
    
    try:
        # Get ratings and compare
        results = ASREService.compare_stocks(tickers)
        
        # Sort by requested criterion
        if request.sort_by == "fscore":
            results.sort(key=lambda x: x.get('fscore', 0), reverse=True)
        elif request.sort_by == "momentum":
            results.sort(key=lambda x: x.get('mscore', 0), reverse=True)
        # Default is already sorted by rating
        
        # Update ranks after sorting
        for i, result in enumerate(results, 1):
            result['rank'] = i
        
        # Add AI comparison explanation if requested
        comparison_explanation = None
        if include_explanations and len(results) >= 2:
            try:
                explainer = AIExplainer()
                explanation_data = explainer.explain_comparison(
                    results,
                    focus=request.sort_by
                )
                comparison_explanation = explanation_data['explanation']
            except Exception as e:
                logger.warning(f"Failed to generate comparison: {e}")
        
        return {
            "stocks": results,
            "count": len(results),
            "sort_by": request.sort_by,
            "comparison_explanation": comparison_explanation,
            "top_pick": results[0] if results else None
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to compare stocks: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )


@router.get(
    "/{ticker}/history",
    summary="Get rating history",
    description="Get historical ratings (cached data)"
)
def get_rating_history(
    ticker: str = Path(
        ...,
        description="Stock ticker symbol",
        regex="^[A-Z][A-Z0-9]*(\.[A-Z]{1,3})?$"
    ),
    days: int = Query(
        30,
        ge=7,
        le=90,
        description="Number of days of history (7-90)"
    )
):
    """
    Get historical ASRE ratings (cached).
    
    **Note:** This endpoint returns cached historical data if available.
    In production, this would query a time-series database.
    
    **Parameters:**
    - **ticker**: Stock ticker symbol
    - **days**: Number of days of history (7-90, default 30)
    
    **Returns:**
    Historical rating data points.
    
    **Status:** Currently returns placeholder data. In production,
    this would fetch from a time-series database.
    """
    ticker = ticker.upper().strip()
    
    logger.info(f"Fetching {days} days history for {ticker}")
    
    # Validate ticker
    if ticker not in settings.SUPPORTED_STOCKS:
        raise HTTPException(
            status_code=404,
            detail=f"Stock '{ticker}' not supported"
        )
    
    # Get current rating
    try:
        current_rating = ASREService.get_stock_rating(ticker)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch history: {str(e)}"
        )
    
    # Generate placeholder historical data
    # In production, this would query a database
    import random
    from datetime import datetime, timedelta
    
    history = []
    base_rating = current_rating['rfinal']
    
    for i in range(days, 0, -1):
        date = datetime.now() - timedelta(days=i)
        # Simulate some variance
        variance = random.uniform(-5, 5)
        historical_rating = max(0, min(100, base_rating + variance))
        
        history.append({
            "date": date.strftime("%Y-%m-%d"),
            "rfinal": round(historical_rating, 2),
            "signal": ASREService._determine_signal(historical_rating)
        })
    
    # Add current rating
    history.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "rfinal": current_rating['rfinal'],
        "signal": current_rating['signal']
    })
    
    return {
        "ticker": ticker,
        "history": history,
        "days": days,
        "data_points": len(history),
        "note": "Historical data is simulated. In production, this would fetch real historical ratings."
    }


@router.get(
    "/momentum-trap/{ticker}",
    response_model=MomentumTrapResponse,
    summary="Check momentum trap",
    description="Analyze if stock is in a momentum trap"
)
def check_momentum_trap(
    ticker: str = Path(
        ...,
        description="Stock ticker symbol",
        regex="^[A-Z][A-Z0-9]*(\.[A-Z]{1,3})?$"
    ),
    include_explanation: bool = Query(
        True,
        description="Include AI explanation"
    )
):
    """
    Analyze if a stock is in a momentum trap.
    
    A momentum trap occurs when:
    - Technical score is high (price momentum)
    - But fundamental score is low (weak business)
    - Risk of sharp correction
    
    **Parameters:**
    - **ticker**: Stock ticker symbol
    - **include_explanation**: Include AI explanation
    
    **Returns:**
    - Trap detection result
    - Severity level
    - Risk factors
    - Recommendations
    - Optional AI explanation
    
    **Example:**
    ```
    GET /api/stocks/momentum-trap/NVDA
    ```
    """
    ticker = ticker.upper().strip()
    
    logger.info(f"Checking momentum trap for {ticker}")
    
    try:
        # Get current rating
        rating = ASREService.get_stock_rating(ticker)
        
        # Analyze trap
        trap_data = analyze_trap_risk(
            ticker,
            rating['fscore'],
            rating['tscore'],
            rating['mscore'],
            rating['rfinal']
        )
        
        # Add AI explanation if requested
        if include_explanation and trap_data['is_trap']:
            try:
                explainer = AIExplainer()
                explanation = explainer.explain_momentum_trap(
                    ticker,
                    rating['fscore'],
                    rating['tscore'],
                    rating['mscore']
                )
                trap_data['ai_explanation'] = explanation['explanation']
            except Exception as e:
                logger.warning(f"Failed to generate trap explanation: {e}")
                trap_data['ai_explanation'] = None
        
        return trap_data
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to check trap for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Trap analysis failed: {str(e)}"
        )


@router.delete(
    "/cache/{ticker}",
    summary="Clear cache",
    description="Clear cached rating for a stock"
)
async def clear_stock_cache(
    ticker: str = Path(
        ...,
        description="Stock ticker symbol or 'all' to clear entire cache",
        regex="^[A-Z][A-Z0-9]*(\.[A-Z]{1,3})?$|^all$"
    )
):
    """
    Clear cached rating data.
    
    **Parameters:**
    - **ticker**: Stock ticker to clear, or 'all' for entire cache
    
    **Returns:**
    Cache clear confirmation
    
    **Example:**
    ```
    DELETE /api/stocks/cache/NVDA
    DELETE /api/stocks/cache/all
    ```
    """
    if ticker.lower() == "all":
        ASREService.clear_cache()
        logger.info("Cleared entire cache")
        return {
            "success": True,
            "message": "Entire cache cleared",
            "ticker": "all"
        }
    else:
        ticker = ticker.upper()
        ASREService.clear_cache(ticker)
        logger.info(f"Cleared cache for {ticker}")
        return {
            "success": True,
            "message": f"Cache cleared for {ticker}",
            "ticker": ticker
        }


@router.get(
    "/cache/stats",
    summary="Get cache statistics",
    description="Get cache usage statistics"
)
async def get_cache_stats():
    """
    Get cache statistics.
    
    **Returns:**
    - Cache size
    - Cached tickers
    - Oldest/newest entries
    
    **Example:**
    ```
    GET /api/stocks/cache/stats
    ```
    """
    stats = ASREService.get_cache_stats()
    return stats


# ============================================================================
# HEALTH CHECK
# ============================================================================

