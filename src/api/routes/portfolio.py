"""
Portfolio Analysis Routes
API endpoints for portfolio health analysis and recommendations.

Endpoints:
- POST /api/portfolio/analyze - Analyze portfolio health
- POST /api/portfolio/optimize - Get optimization suggestions
- GET /api/portfolio/risk-breakdown - Detailed risk analysis
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging

from api.services.portfolio_analyzer import PortfolioAnalyzer, analyze_portfolio
from api.models.responses import PortfolioAnalysis, ErrorResponse
from api.models.requests import PortfolioAnalysisRequest
from api.config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "/analyze",
    response_model=PortfolioAnalysis,
    summary="Analyze portfolio health",
    description="Get comprehensive portfolio analysis with health score and recommendations",
    responses={
        200: {"description": "Successfully analyzed portfolio"},
        400: {"model": ErrorResponse, "description": "Invalid portfolio data"},
        422: {"model": ErrorResponse, "description": "Validation error"}
    }
)
async def analyze_portfolio_endpoint(
    request: PortfolioAnalysisRequest,
    currency: str = Query(
        "USD",
        description="Portfolio currency (USD, INR, EUR, etc.)",
        regex="^[A-Z]{3}$"
    ),
    include_rebalancing: bool = Query(
        True,
        description="Include rebalancing suggestions"
    )
):
    """
    Analyze portfolio health and get recommendations.
    
    **Request Body:**
    ```json
    {
      "holdings": [
        {
          "ticker": "NVDA",
          "shares": 100,
          "value": 14250.00
        },
        {
          "ticker": "MSFT",
          "shares": 50,
          "value": 21500.00
        }
      ]
    }
    ```
    
    **Parameters:**
    - **holdings**: List of portfolio holdings with ticker, shares, and current value
    - **currency**: Portfolio currency (default: USD)
    - **include_rebalancing**: Include portfolio rebalancing suggestions
    
    **Returns:**
    Complete portfolio analysis including:
    - Overall health score (0-100)
    - Health level (EXCELLENT/GOOD/FAIR/POOR/CRITICAL)
    - Risk breakdown by category
    - High-risk stocks requiring attention
    - Actionable recommendations
    - Optional rebalancing suggestions
    
    **Example:**
    ```
    POST /api/portfolio/analyze?currency=USD&include_rebalancing=true
    ```
    """
    logger.info(f"Analyzing portfolio with {len(request.holdings)} holdings")
    
    # Validation
    if not request.holdings:
        raise HTTPException(
            status_code=400,
            detail="Portfolio must contain at least one holding"
        )
    
    if len(request.holdings) > 50:
        raise HTTPException(
            status_code=400,
            detail="Portfolio cannot exceed 50 holdings"
        )
    
    # Convert holdings to dict format
    holdings = []
    total_value = 0
    
    for holding in request.holdings:
        ticker = holding.ticker.upper().strip()
        
        # Validate ticker is supported
        if ticker not in settings.SUPPORTED_STOCKS:
            logger.warning(f"Ticker {ticker} not supported, will skip in analysis")
            # Don't fail entire request, just warn
        
        # Validate values
        if holding.shares <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid shares for {ticker}: must be positive"
            )
        
        if holding.value <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value for {ticker}: must be positive"
            )
        
        holdings.append({
            "ticker": ticker,
            "shares": holding.shares,
            "value": holding.value
        })
        
        total_value += holding.value
    
    # Validate total portfolio value
    if total_value <= 0:
        raise HTTPException(
            status_code=400,
            detail="Total portfolio value must be positive"
        )
    
    logger.info(f"Portfolio total value: {currency} {total_value:,.2f}")
    
    # Analyze portfolio
    try:
        analyzer = PortfolioAnalyzer()
        
        result = analyzer.analyze_portfolio(
            holdings=holdings,
            currency=currency
        )
        
        # Optionally remove rebalancing if not requested
        if not include_rebalancing:
            result.pop('rebalancing', None)
        
        # Limit recommendations to top 5 for clarity
        if 'recommendations' in result and len(result['recommendations']) > 5:
            result['recommendations'] = result['recommendations'][:5]
        
        logger.info(
            f"Portfolio analysis complete: Score={result['overall_score']:.1f}, "
            f"Health={result['health_level']}, "
            f"High-risk stocks={len(result.get('high_risk_stocks', []))}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Portfolio analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post(
    "/optimize",
    summary="Get portfolio optimization suggestions",
    description="Get detailed suggestions for improving portfolio allocation"
)
async def optimize_portfolio(
    request: PortfolioAnalysisRequest,
    target_risk: str = Query(
        "moderate",
        description="Target risk level (conservative, moderate, aggressive)",
        regex="^(conservative|moderate|aggressive)$"
    )
):
    """
    Get portfolio optimization suggestions based on target risk level.
    
    **Request Body:** Same as `/analyze` endpoint
    
    **Parameters:**
    - **holdings**: Current portfolio holdings
    - **target_risk**: Desired risk level
      - `conservative`: 70% safe, 25% moderate, 5% high-risk
      - `moderate`: 60% safe, 30% moderate, 10% high-risk (default)
      - `aggressive`: 40% safe, 40% moderate, 20% high-risk
    
    **Returns:**
    Optimization recommendations including:
    - Current vs target allocation
    - Stocks to add/reduce
    - Expected improvement in risk-adjusted returns
    
    **Example:**
    ```
    POST /api/portfolio/optimize?target_risk=moderate
    ```
    """
    logger.info(f"Optimizing portfolio for {target_risk} risk profile")
    
    # Validation (same as analyze)
    if not request.holdings or len(request.holdings) > 50:
        raise HTTPException(
            status_code=400,
            detail="Invalid portfolio size"
        )
    
    # Convert holdings
    holdings = [
        {
            "ticker": h.ticker.upper().strip(),
            "shares": h.shares,
            "value": h.value
        }
        for h in request.holdings
    ]
    
    total_value = sum(h['value'] for h in holdings)
    
    if total_value <= 0:
        raise HTTPException(
            status_code=400,
            detail="Total portfolio value must be positive"
        )
    
    try:
        analyzer = PortfolioAnalyzer()
        
        # Get current analysis
        current = analyzer.analyze_portfolio(holdings, "USD")
        
        # Get rebalancing suggestions
        rebalancing = analyzer.suggest_rebalancing(
            analyzer._enrich_holdings_with_ratings(holdings),
            total_value
        )
        
        # Define target allocations based on risk profile
        target_allocations = {
            "conservative": {
                "safe": 70.0,
                "moderate": 25.0,
                "high_risk": 5.0
            },
            "moderate": {
                "safe": 60.0,
                "moderate": 30.0,
                "high_risk": 10.0
            },
            "aggressive": {
                "safe": 40.0,
                "moderate": 40.0,
                "high_risk": 20.0
            }
        }
        
        target = target_allocations[target_risk]
        
        # Calculate optimization actions
        optimization_actions = []
        
        current_alloc = {
            k: v['percentage'] 
            for k, v in current['risk_breakdown'].items()
        }
        
        for category, target_pct in target.items():
            current_pct = current_alloc.get(category, 0)
            diff_pct = target_pct - current_pct
            diff_value = (diff_pct / 100) * total_value
            
            if abs(diff_pct) > 2:  # 2% threshold
                if diff_value > 0:
                    optimization_actions.append({
                        "action": "increase",
                        "category": category,
                        "current_pct": round(current_pct, 1),
                        "target_pct": target_pct,
                        "amount": round(abs(diff_value), 2),
                        "description": f"Add ${abs(diff_value):,.0f} to {category} stocks"
                    })
                else:
                    optimization_actions.append({
                        "action": "decrease",
                        "category": category,
                        "current_pct": round(current_pct, 1),
                        "target_pct": target_pct,
                        "amount": round(abs(diff_value), 2),
                        "description": f"Reduce {category} by ${abs(diff_value):,.0f}"
                    })
        
        return {
            "target_risk_profile": target_risk,
            "target_allocation": target,
            "current_allocation": current_alloc,
            "optimization_actions": optimization_actions,
            "stocks_to_add": rebalancing.get('best_to_add', [])[:3],
            "stocks_to_reduce": rebalancing.get('to_reduce', [])[:3],
            "expected_improvement": {
                "score_increase": "5-10 points estimated",
                "risk_reduction": "Lower volatility",
                "better_diversification": True
            },
            "total_value": total_value,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )


@router.post(
    "/risk-breakdown",
    summary="Get detailed risk breakdown",
    description="Get detailed analysis of portfolio risk distribution"
)
async def get_risk_breakdown(
    request: PortfolioAnalysisRequest
):
    """
    Get detailed risk breakdown of portfolio.
    
    **Request Body:** Same as `/analyze` endpoint
    
    **Returns:**
    Detailed risk analysis including:
    - Risk distribution by category
    - High-risk stock details
    - Risk-adjusted metrics
    - Diversification score
    
    **Example:**
    ```
    POST /api/portfolio/risk-breakdown
    ```
    """
    logger.info("Calculating detailed risk breakdown")
    
    # Validation
    if not request.holdings:
        raise HTTPException(
            status_code=400,
            detail="Portfolio cannot be empty"
        )
    
    # Convert holdings
    holdings = [
        {
            "ticker": h.ticker.upper().strip(),
            "shares": h.shares,
            "value": h.value
        }
        for h in request.holdings
    ]
    
    total_value = sum(h['value'] for h in holdings)
    
    try:
        analyzer = PortfolioAnalyzer()
        enriched = analyzer._enrich_holdings_with_ratings(holdings)
        
        # Calculate risk breakdown
        risk_breakdown = analyzer.calculate_risk_breakdown(enriched, total_value)
        
        # Identify high-risk stocks
        high_risk = analyzer.identify_high_risk_stocks(enriched)
        
        # Calculate diversification metrics
        concentration = max(h['value'] for h in holdings) / total_value * 100
        
        diversification_score = 100 - concentration
        if len(holdings) < 5:
            diversification_score *= 0.7
        elif len(holdings) > 15:
            diversification_score *= 0.9
        
        return {
            "risk_breakdown": risk_breakdown,
            "high_risk_stocks": high_risk,
            "diversification": {
                "score": round(diversification_score, 1),
                "holdings_count": len(holdings),
                "concentration_risk": round(concentration, 1),
                "status": "Good" if diversification_score > 70 else "Needs improvement"
            },
            "recommendations": [
                f"Portfolio has {concentration:.1f}% concentrated in top position",
                f"Consider adding 2-3 more stocks" if len(holdings) < 5 else "Good diversification",
                f"High-risk exposure: {risk_breakdown['high_risk']['percentage']:.1f}%"
            ],
            "total_value": total_value,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Risk breakdown failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Risk analysis failed: {str(e)}"
        )


@router.get(
    "/health",
    summary="Portfolio service health check"
)
async def portfolio_health_check():
    """
    Check if portfolio analysis service is operational.
    
    **Returns:**
    Service health status
    """
    try:
        # Quick test
        analyzer = PortfolioAnalyzer()
        test_holding = [{"ticker": "NVDA", "shares": 1, "value": 100}]
        
        # This should work if service is healthy
        result = analyzer.analyze_portfolio(test_holding, "USD")
        
        return {
            "status": "healthy",
            "service": "portfolio_analyzer",
            "test_passed": True,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Portfolio service unhealthy: {e}")
        return {
            "status": "unhealthy",
            "service": "portfolio_analyzer",
            "error": str(e),
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
