"""
Community Routes (Discovery Edition)
====================================
Stock discovery and trending analysis without social features.

Features:
- Trending stocks (most queried)
- Top-rated stocks by ASRE score
- Most-searched tickers
- Sector activity analysis
- Popular watchlists
- Market statistics

Author: ASRE Team
Version: 2.0.0
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import Counter, defaultdict
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import ASRE service
try:
    from api.services.asre_service import ASREService
    ASRE_SERVICE_AVAILABLE = True
except ImportError:
    ASRE_SERVICE_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/community", tags=["community"])


# ============================================================================
# IN-MEMORY ANALYTICS STORAGE
# ============================================================================

class DiscoveryAnalytics:
    """Track stock queries and ratings for discovery features."""
    
    def __init__(self):
        # Query tracking
        self._query_log: List[Dict] = []
        self._ticker_queries: Counter = Counter()
        self._sector_queries: Counter = Counter()
        
        # Rating cache (ticker -> {rating, timestamp})
        self._rating_cache: Dict[str, Dict] = {}
        
        # Watchlist tracking (user_id -> [tickers])
        self._watchlists: Dict[str, List[str]] = {}
        
        # Popular stocks by time period
        self._trending_window_hours = 24
        
    def log_query(self, ticker: str, sector: Optional[str] = None):
        """Log a stock query."""
        self._query_log.append({
            'ticker': ticker,
            'sector': sector,
            'timestamp': datetime.now()
        })
        self._ticker_queries[ticker] += 1
        
        if sector:
            self._sector_queries[sector] += 1
        
        # Keep only last 10,000 queries
        if len(self._query_log) > 10000:
            self._query_log = self._query_log[-10000:]
    
    def update_rating(self, ticker: str, rating: float):
        """Update rating cache."""
        self._rating_cache[ticker] = {
            'rating': rating,
            'timestamp': datetime.now()
        }
    
    def get_trending(self, hours: int = 24, limit: int = 10) -> List[Dict]:
        """Get trending stocks in last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent_queries = [
            q for q in self._query_log
            if q['timestamp'] > cutoff
        ]
        
        ticker_counts = Counter(q['ticker'] for q in recent_queries)
        
        trending = [
            {
                'ticker': ticker,
                'query_count': count,
                'rating': self._rating_cache.get(ticker, {}).get('rating')
            }
            for ticker, count in ticker_counts.most_common(limit)
        ]
        
        return trending
    
    def get_top_rated(self, limit: int = 10) -> List[Dict]:
        """Get highest-rated stocks."""
        rated_stocks = [
            {
                'ticker': ticker,
                'rating': data['rating'],
                'last_updated': data['timestamp'].isoformat()
            }
            for ticker, data in self._rating_cache.items()
            if data.get('rating') is not None
        ]
        
        # Sort by rating descending
        rated_stocks.sort(key=lambda x: x['rating'], reverse=True)
        
        return rated_stocks[:limit]
    
    def get_most_searched(self, limit: int = 10) -> List[Dict]:
        """Get most-searched stocks (all time)."""
        return [
            {
                'ticker': ticker,
                'total_queries': count,
                'rating': self._rating_cache.get(ticker, {}).get('rating')
            }
            for ticker, count in self._ticker_queries.most_common(limit)
        ]
    
    def get_sector_activity(self) -> Dict[str, int]:
        """Get query counts by sector."""
        return dict(self._sector_queries.most_common())
    
    def add_to_watchlist(self, user_id: str, ticker: str):
        """Add ticker to user watchlist."""
        if user_id not in self._watchlists:
            self._watchlists[user_id] = []
        
        if ticker not in self._watchlists[user_id]:
            self._watchlists[user_id].append(ticker)
    
    def get_watchlist_counts(self, limit: int = 20) -> List[Dict]:
        """Get most-watched stocks."""
        all_tickers = []
        for watchlist in self._watchlists.values():
            all_tickers.extend(watchlist)
        
        ticker_counts = Counter(all_tickers)
        
        return [
            {
                'ticker': ticker,
                'watchers': count,
                'rating': self._rating_cache.get(ticker, {}).get('rating')
            }
            for ticker, count in ticker_counts.most_common(limit)
        ]
    
    def get_stats(self) -> Dict:
        """Get overall statistics."""
        return {
            'total_queries': len(self._query_log),
            'unique_tickers': len(self._ticker_queries),
            'rated_stocks': len(self._rating_cache),
            'active_watchlists': len(self._watchlists),
            'total_sectors_tracked': len(self._sector_queries)
        }


# Global analytics instance
_analytics = DiscoveryAnalytics()


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class TrendingStock(BaseModel):
    """Trending stock entry"""
    ticker: str
    query_count: int
    rating: Optional[float]
    trend_direction: str = "neutral"


class TopRatedStock(BaseModel):
    """Top-rated stock entry"""
    ticker: str
    rating: float
    last_updated: str
    quality_tier: str


class MostSearchedStock(BaseModel):
    """Most-searched stock entry"""
    ticker: str
    total_queries: int
    rating: Optional[float]
    popularity_rank: int


class SectorActivity(BaseModel):
    """Sector activity summary"""
    sector: str
    query_count: int
    percentage: float


class WatchlistStock(BaseModel):
    """Most-watched stock"""
    ticker: str
    watchers: int
    rating: Optional[float]


class CommunityStats(BaseModel):
    """Overall community statistics"""
    total_queries: int
    unique_tickers: int
    rated_stocks: int
    active_watchlists: int
    total_sectors_tracked: int
    timestamp: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_quality_tier(rating: float) -> str:
    """Get quality tier from rating."""
    if rating >= 85:
        return "S-Tier"
    elif rating >= 80:
        return "A-Tier"
    elif rating >= 70:
        return "B-Tier"
    elif rating >= 60:
        return "C-Tier"
    elif rating >= 50:
        return "D-Tier"
    else:
        return "F-Tier"


def get_trend_direction(current_count: int, historical_avg: float) -> str:
    """Determine trend direction."""
    if current_count > historical_avg * 1.5:
        return "surging"
    elif current_count > historical_avg * 1.2:
        return "rising"
    elif current_count < historical_avg * 0.8:
        return "declining"
    else:
        return "stable"


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/trending", response_model=List[TrendingStock])
async def get_trending_stocks(
    hours: int = Query(24, description="Time window in hours", ge=1, le=168),
    limit: int = Query(10, description="Number of results", ge=1, le=50)
):
    """
    Get trending stocks based on recent query activity.
    
    **Trending = Most queried in last N hours**
    
    **Use Cases:**
    - Discover what other investors are researching
    - Identify emerging market interest
    - Track popularity shifts
    
    **Example:** `/api/community/trending?hours=24&limit=10`
    """
    try:
        logger.info(f"Getting trending stocks (last {hours}h)")
        
        trending_data = _analytics.get_trending(hours=hours, limit=limit)
        
        # Calculate historical average for trend direction
        total_queries = sum(item['query_count'] for item in trending_data)
        avg_queries = total_queries / len(trending_data) if trending_data else 0
        
        results = []
        for item in trending_data:
            trend_dir = get_trend_direction(item['query_count'], avg_queries)
            
            results.append(TrendingStock(
                ticker=item['ticker'],
                query_count=item['query_count'],
                rating=item.get('rating'),
                trend_direction=trend_dir
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to get trending stocks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@router.get("/top-rated", response_model=List[TopRatedStock])
async def get_top_rated_stocks(
    limit: int = Query(10, description="Number of results", ge=1, le=50),
    min_rating: float = Query(0, description="Minimum rating filter", ge=0, le=100)
):
    """
    Get highest-rated stocks by ASRE score.
    
    **Sorted by rating (highest first)**
    
    **Use Cases:**
    - Find best investment opportunities
    - Quality screening
    - Top picks discovery
    
    **Example:** `/api/community/top-rated?limit=20&min_rating=70`
    """
    try:
        logger.info(f"Getting top-rated stocks (min: {min_rating})")
        
        top_rated = _analytics.get_top_rated(limit=limit * 2)  # Get extra for filtering
        
        # Filter by minimum rating
        filtered = [
            stock for stock in top_rated
            if stock['rating'] >= min_rating
        ][:limit]
        
        results = []
        for stock in filtered:
            results.append(TopRatedStock(
                ticker=stock['ticker'],
                rating=stock['rating'],
                last_updated=stock['last_updated'],
                quality_tier=get_quality_tier(stock['rating'])
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to get top-rated stocks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@router.get("/most-searched", response_model=List[MostSearchedStock])
async def get_most_searched_stocks(
    limit: int = Query(10, description="Number of results", ge=1, le=50)
):
    """
    Get most-searched stocks (all-time).
    
    **Based on cumulative query count**
    
    **Use Cases:**
    - Identify most popular stocks
    - Track sustained interest
    - Discover community favorites
    
    **Example:** `/api/community/most-searched?limit=15`
    """
    try:
        logger.info("Getting most-searched stocks")
        
        most_searched = _analytics.get_most_searched(limit=limit)
        
        results = []
        for rank, stock in enumerate(most_searched, 1):
            results.append(MostSearchedStock(
                ticker=stock['ticker'],
                total_queries=stock['total_queries'],
                rating=stock.get('rating'),
                popularity_rank=rank
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to get most-searched stocks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@router.get("/sector-activity", response_model=List[SectorActivity])
async def get_sector_activity():
    """
    Get sector activity analysis.
    
    **Shows query distribution by sector**
    
    **Use Cases:**
    - Identify hot sectors
    - Sector rotation tracking
    - Market focus analysis
    
    **Example:** `/api/community/sector-activity`
    """
    try:
        logger.info("Getting sector activity")
        
        sector_data = _analytics.get_sector_activity()
        
        total_queries = sum(sector_data.values())
        
        results = []
        for sector, count in sorted(sector_data.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_queries * 100) if total_queries > 0 else 0
            
            results.append(SectorActivity(
                sector=sector,
                query_count=count,
                percentage=round(percentage, 2)
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to get sector activity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@router.get("/watchlist-counts", response_model=List[WatchlistStock])
async def get_watchlist_counts(
    limit: int = Query(20, description="Number of results", ge=1, le=50)
):
    """
    Get most-watched stocks.
    
    **Based on watchlist additions**
    
    **Use Cases:**
    - See what others are tracking
    - Discover popular holdings
    - Community interest gauge
    
    **Example:** `/api/community/watchlist-counts?limit=20`
    """
    try:
        logger.info("Getting watchlist counts")
        
        watchlist_data = _analytics.get_watchlist_counts(limit=limit)
        
        results = []
        for stock in watchlist_data:
            results.append(WatchlistStock(
                ticker=stock['ticker'],
                watchers=stock['watchers'],
                rating=stock.get('rating')
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to get watchlist counts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@router.get("/stats", response_model=CommunityStats)
async def get_community_stats():
    """
    Get overall community statistics.
    
    **Returns:**
    - Total queries processed
    - Unique tickers tracked
    - Rated stocks count
    - Active watchlists
    - Sectors tracked
    
    **Example:** `/api/community/stats`
    """
    try:
        logger.info("Getting community stats")
        
        stats = _analytics.get_stats()
        
        return CommunityStats(
            total_queries=stats['total_queries'],
            unique_tickers=stats['unique_tickers'],
            rated_stocks=stats['rated_stocks'],
            active_watchlists=stats['active_watchlists'],
            total_sectors_tracked=stats['total_sectors_tracked'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@router.post("/track-query/{ticker}")
async def track_query(
    ticker: str,
    sector: Optional[str] = Query(None, description="Stock sector")
):
    """
    Track a stock query (internal use for analytics).
    
    **Called automatically when stocks are queried via other endpoints.**
    
    **Example:** `POST /api/community/track-query/NVDA?sector=Technology`
    """
    try:
        ticker = ticker.upper()
        _analytics.log_query(ticker, sector)
        
        return {
            'ticker': ticker,
            'tracked': True,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to track query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@router.post("/update-rating/{ticker}")
async def update_rating(
    ticker: str,
    rating: float = Query(..., description="ASRE rating", ge=0, le=100)
):
    """
    Update rating cache (internal use for analytics).
    
    **Called automatically when ratings are computed.**
    
    **Example:** `POST /api/community/update-rating/NVDA?rating=85.5`
    """
    try:
        ticker = ticker.upper()
        _analytics.update_rating(ticker, rating)
        
        return {
            'ticker': ticker,
            'rating': rating,
            'updated': True,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to update rating: {e}")
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@router.post("/watchlist/add")
async def add_to_watchlist(
    user_id: str = Query(..., description="User identifier"),
    ticker: str = Query(..., description="Stock ticker")
):
    """
    Add stock to user watchlist.
    
    **Example:** `POST /api/community/watchlist/add?user_id=user123&ticker=NVDA`
    """
    try:
        ticker = ticker.upper()
        _analytics.add_to_watchlist(user_id, ticker)
        
        return {
            'user_id': user_id,
            'ticker': ticker,
            'added': True,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to add to watchlist: {e}")
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Check community service health.
    """
    try:
        stats = _analytics.get_stats()
        
        return {
            'status': 'healthy',
            'analytics_enabled': True,
            'tracked_queries': stats['total_queries'],
            'message': 'Community discovery services operational'
        }
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'message': 'Health check failed'
        }
