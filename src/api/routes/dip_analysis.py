"""
Dip Analysis Routes
===================
Dip buying opportunity detection and analysis API endpoints.

Directly imports from existing scripts:
- production_investing_engine.py
- Moat_Portfolio.py

Features:
- Analyze individual stock dip quality
- Scan universe for dip opportunities
- Historical dip pattern analysis
- Entry timing optimization
- Confidence scoring with 4 dip stages

Author: ASRE Team
Version: 2.0.0
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import existing modules
from asre.data.fundamental_fetcher import FundamentalFetcher
from asre.data_loader import DataLoader
from asre.composite import compute_complete_asre

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dip-analysis", tags=["dip-analysis"])


# ============================================================================
# CONFIGURATION
# ============================================================================

class DipConfig:
    """Dip buying configuration constants"""
    # Dip Detection
    DIP_MIN_ASRE = 60.0
    DIP_MAX_DISTANCE = -20.0
    DIP_POSITION_MULTIPLIER = 0.60
    DIP_MIN_CONFIDENCE = 50.0
    DIP_REQUIRE_A_OR_B_TIER = True
    
    # SMA Period
    SMA_PERIOD = 200
    
    # Quality Tiers
    TIER_S = 85.0  # S-Tier
    TIER_A = 80.0
    TIER_B = 70.0
    TIER_C = 60.0
    TIER_D = 50.0


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class DipAnalysisResponse(BaseModel):
    """Dip analysis result for a single stock"""
    ticker: str
    current_price: float
    sma_200: float
    distance_from_sma_pct: float
    is_dip: bool
    dip_stage: str
    confidence: float
    asre_score: float
    quality_tier: str
    approved_for_dip_buy: bool
    reason: str
    entry_timing_score: float
    recommended_allocation_pct: float
    stop_loss_price: float
    target_price: float


class DipScanRequest(BaseModel):
    """Request to scan universe for dip opportunities"""
    universe: List[str] = Field(..., description="List of stock tickers to scan")
    min_asre: float = Field(60.0, description="Minimum ASRE score")
    quality_tiers: List[str] = Field(["A", "B"], description="Allowed quality tiers")
    max_distance_from_sma: float = Field(-20.0, description="Max distance below SMA-200 (%)")
    min_confidence: float = Field(50.0, description="Minimum confidence score")


class DipOpportunity(BaseModel):
    """Dip opportunity summary"""
    ticker: str
    asre_score: float
    quality_tier: str
    distance_from_sma_pct: float
    dip_stage: str
    confidence: float
    entry_timing_score: float
    recommended_allocation_pct: float
    reason: str


class DipScanResponse(BaseModel):
    """Scan results"""
    scan_time: str
    universe_size: int
    opportunities_found: int
    dip_opportunities: List[DipOpportunity]
    filters_applied: Dict
    message: str


class HistoricalDipPattern(BaseModel):
    """Historical dip pattern analysis"""
    dip_date: str
    entry_price: float
    bottom_price: float
    recovery_price: float
    days_to_bottom: int
    days_to_recovery: int
    max_drawdown_pct: float
    recovery_return_pct: float
    dip_stage: str


class HistoricalDipResponse(BaseModel):
    """Historical dip analysis"""
    ticker: str
    analysis_period: str
    total_dips_found: int
    successful_recoveries: int
    avg_recovery_days: float
    avg_recovery_return_pct: float
    dip_patterns: List[HistoricalDipPattern]
    message: str


class EntryTimingResponse(BaseModel):
    """Entry timing optimization"""
    ticker: str
    current_stage: str
    optimal_entry_stage: str
    current_timing_score: float
    wait_recommendation: str
    estimated_better_entry_pct: float
    reasoning: str


# ============================================================================
# DIP ANALYZER (from Moat_Portfolio.py)
# ============================================================================

class DipAnalyzer:
    """Analyze dip quality and generate buy signals"""
    
    @staticmethod
    def analyze_dip(
        price: float,
        sma200: float,
        asre_score: float,
        quality_tier: str
    ) -> Dict:
        """
        Analyze dip quality with 4-stage classification.
        
        Stages:
        - EARLY: 0-5% below SMA (75% confidence)
        - MID: 5-10% below SMA (65% confidence)
        - LATE: 10-15% below SMA (55% confidence)
        - DEEP: >15% below SMA (40% confidence)
        """
        distance = ((price - sma200) / sma200) * 100
        
        analysis = {
            'distance_from_sma_pct': distance,
            'is_dip': False,
            'dip_stage': 'NONE',
            'confidence': 0.0,
            'quality_tier': quality_tier,
            'approved_for_dip_buy': False,
            'reason': ''
        }
        
        # Check if in dip
        if distance >= 0:
            analysis['reason'] = 'Price above SMA-200 (uptrend)'
            return analysis
        
        analysis['is_dip'] = True
        
        # Determine dip stage and confidence
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
        if DipConfig.DIP_REQUIRE_A_OR_B_TIER and quality_tier not in ['S', 'A', 'B']:
            analysis['approved_for_dip_buy'] = False
            analysis['reason'] = f"Tier {quality_tier} not approved (need S/A/B)"
            return analysis
        
        if asre_score < DipConfig.DIP_MIN_ASRE:
            analysis['approved_for_dip_buy'] = False
            analysis['reason'] = f"ASRE {asre_score:.1f} below minimum {DipConfig.DIP_MIN_ASRE}"
            return analysis
        
        if distance < DipConfig.DIP_MAX_DISTANCE:
            analysis['approved_for_dip_buy'] = False
            analysis['reason'] = f"Too far below SMA ({distance:.1f}% < {DipConfig.DIP_MAX_DISTANCE}%)"
            return analysis
        
        if analysis['confidence'] < DipConfig.DIP_MIN_CONFIDENCE:
            analysis['approved_for_dip_buy'] = False
            analysis['reason'] = f"Confidence too low ({analysis['confidence']:.1f}%)"
            return analysis
        
        # APPROVED
        analysis['approved_for_dip_buy'] = True
        analysis['reason'] = f"✅ DIP APPROVED ({analysis['dip_stage']}, {analysis['confidence']:.0f}% conf, Tier {quality_tier})"
        
        return analysis
    
    @staticmethod
    def calculate_entry_timing_score(
        dip_stage: str,
        quality_tier: str,
        asre_score: float,
        distance_from_sma_pct: float
    ) -> float:
        """
        Calculate entry timing score (0-100).
        Higher score = better entry timing.
        
        Factors:
        - Dip stage (MID is optimal)
        - Quality tier
        - Distance from SMA
        """
        score = 0.0
        
        # Stage scoring (MID is optimal)
        stage_scores = {
            'EARLY': 60.0,   # Good but might go lower
            'MID': 85.0,     # Optimal entry
            'LATE': 70.0,    # Still good
            'DEEP': 50.0,    # Risky, might be value trap
            'NONE': 0.0
        }
        score += stage_scores.get(dip_stage, 0.0) * 0.4
        
        # Quality tier bonus
        tier_scores = {
            'S': 100.0,
            'A': 90.0,
            'B': 75.0,
            'C': 60.0,
            'D': 40.0,
            'F': 20.0
        }
        score += tier_scores.get(quality_tier, 0.0) * 0.3
        
        # Distance scoring (sweet spot: -8% to -12%)
        distance_score = 0.0
        abs_distance = abs(distance_from_sma_pct)
        if 8 <= abs_distance <= 12:
            distance_score = 100.0
        elif 5 <= abs_distance < 8:
            distance_score = 80.0
        elif 12 < abs_distance <= 15:
            distance_score = 70.0
        elif abs_distance > 15:
            distance_score = 50.0
        else:
            distance_score = 60.0
        
        score += distance_score * 0.3
        
        return min(100.0, score)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_quality_tier(score: float) -> str:
    """Determine quality tier from ASRE score"""
    if score >= DipConfig.TIER_S:
        return 'S'
    elif score >= DipConfig.TIER_A:
        return 'A'
    elif score >= DipConfig.TIER_B:
        return 'B'
    elif score >= DipConfig.TIER_C:
        return 'C'
    elif score >= DipConfig.TIER_D:
        return 'D'
    else:
        return 'F'


def fetch_stock_data(ticker: str, days: int = 730) -> pd.DataFrame:
    """Fetch stock data with ASRE ratings"""
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Fetch fundamentals
        fetcher = FundamentalFetcher()
        df_fundamentals = fetcher.fetch_quarterly_fundamentals(ticker, start_date, end_date)
        
        # Load price data
        loader = DataLoader()
        df = loader.load_stock_data(
            ticker,
            start_date,
            end_date,
            quarterly_fundamentals=df_fundamentals if df_fundamentals is not None else None
        )
        
        # Compute ASRE
        df_complete = compute_complete_asre(df, medallion=True, return_all_components=True)
        df_complete['date'] = pd.to_datetime(df_complete['date'])
        df_complete = df_complete.set_index('date')
        
        # Calculate SMA-200
        df_complete['sma_200'] = df_complete['close'].rolling(window=DipConfig.SMA_PERIOD).mean()
        
        return df_complete
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        raise


def calculate_stop_loss_and_target(current_price: float, distance_from_sma_pct: float) -> tuple:
    """Calculate stop loss and target prices"""
    # Stop loss: 15% below current price
    stop_loss = current_price * 0.85
    
    # Target: SMA-200 level + buffer
    target_multiplier = 1.0 + (abs(distance_from_sma_pct) / 100) + 0.05
    target = current_price * target_multiplier
    
    return stop_loss, target


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/{ticker}", response_model=DipAnalysisResponse)
async def analyze_dip(ticker: str):
    """
    Analyze dip quality for a single stock.
    
    **Returns:**
    - Dip stage (EARLY/MID/LATE/DEEP)
    - Confidence score (0-100)
    - Approval for dip buying
    - Entry timing score
    - Recommended allocation
    
    **Example:** `/api/dip-analysis/NVDA`
    """
    try:
        logger.info(f"Analyzing dip for {ticker}")
        
        # Fetch data
        df = fetch_stock_data(ticker)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        # Get latest values
        latest = df.iloc[-1]
        current_price = float(latest['close'])
        sma_200 = float(latest['sma_200'])
        asre_score = float(latest['r_final'])
        quality_tier = get_quality_tier(asre_score)
        
        # Analyze dip
        analyzer = DipAnalyzer()
        dip_analysis = analyzer.analyze_dip(
            current_price,
            sma_200,
            asre_score,
            quality_tier
        )
        
        # Calculate entry timing score
        entry_timing_score = analyzer.calculate_entry_timing_score(
            dip_analysis['dip_stage'],
            quality_tier,
            asre_score,
            dip_analysis['distance_from_sma_pct']
        )
        
        # Calculate recommended allocation
        if dip_analysis['approved_for_dip_buy']:
            if quality_tier == 'S':
                base_alloc = 12.0
            elif quality_tier == 'A':
                base_alloc = 10.0
            elif quality_tier == 'B':
                base_alloc = 8.0
            else:
                base_alloc = 6.0
            
            recommended_allocation = base_alloc * DipConfig.DIP_POSITION_MULTIPLIER
        else:
            recommended_allocation = 0.0
        
        # Calculate stop loss and target
        stop_loss, target = calculate_stop_loss_and_target(
            current_price,
            dip_analysis['distance_from_sma_pct']
        )
        
        return DipAnalysisResponse(
            ticker=ticker,
            current_price=current_price,
            sma_200=sma_200,
            distance_from_sma_pct=dip_analysis['distance_from_sma_pct'],
            is_dip=dip_analysis['is_dip'],
            dip_stage=dip_analysis['dip_stage'],
            confidence=dip_analysis['confidence'],
            asre_score=asre_score,
            quality_tier=quality_tier,
            approved_for_dip_buy=dip_analysis['approved_for_dip_buy'],
            reason=dip_analysis['reason'],
            entry_timing_score=entry_timing_score,
            recommended_allocation_pct=recommended_allocation,
            stop_loss_price=stop_loss,
            target_price=target
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dip analysis failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/scan", response_model=DipScanResponse)
async def scan_universe_for_dips(request: DipScanRequest):
    """
    Scan multiple stocks for dip buying opportunities.
    
    **Filters:**
    - Minimum ASRE score
    - Quality tiers (S/A/B/C/D)
    - Max distance from SMA-200
    - Minimum confidence
    
    **Example:**
    ```json
    {
      "universe": ["NVDA", "MSFT", "GOOGL", "META", "AAPL", "TSLA", "AMD"],
      "min_asre": 60.0,
      "quality_tiers": ["A", "B"],
      "max_distance_from_sma": -20.0,
      "min_confidence": 50.0
    }
    ```
    """
    try:
        logger.info(f"Scanning {len(request.universe)} stocks for dip opportunities")
        
        opportunities = []
        analyzer = DipAnalyzer()
        
        for ticker in request.universe:
            try:
                # Fetch data
                df = fetch_stock_data(ticker, days=365)
                
                if df.empty:
                    logger.warning(f"No data for {ticker}, skipping")
                    continue
                
                # Get latest values
                latest = df.iloc[-1]
                current_price = float(latest['close'])
                sma_200 = float(latest['sma_200'])
                asre_score = float(latest['r_final'])
                quality_tier = get_quality_tier(asre_score)
                
                # Apply filters
                if asre_score < request.min_asre:
                    continue
                
                if quality_tier not in request.quality_tiers:
                    continue
                
                # Analyze dip
                dip_analysis = analyzer.analyze_dip(
                    current_price,
                    sma_200,
                    asre_score,
                    quality_tier
                )
                
                # Check if approved
                if not dip_analysis['approved_for_dip_buy']:
                    continue
                
                # Check confidence
                if dip_analysis['confidence'] < request.min_confidence:
                    continue
                
                # Check distance
                if dip_analysis['distance_from_sma_pct'] < request.max_distance_from_sma:
                    continue
                
                # Calculate metrics
                entry_timing_score = analyzer.calculate_entry_timing_score(
                    dip_analysis['dip_stage'],
                    quality_tier,
                    asre_score,
                    dip_analysis['distance_from_sma_pct']
                )
                
                # Calculate allocation
                if quality_tier == 'S':
                    base_alloc = 12.0
                elif quality_tier == 'A':
                    base_alloc = 10.0
                elif quality_tier == 'B':
                    base_alloc = 8.0
                else:
                    base_alloc = 6.0
                
                recommended_allocation = base_alloc * DipConfig.DIP_POSITION_MULTIPLIER
                
                opportunities.append(DipOpportunity(
                    ticker=ticker,
                    asre_score=asre_score,
                    quality_tier=quality_tier,
                    distance_from_sma_pct=dip_analysis['distance_from_sma_pct'],
                    dip_stage=dip_analysis['dip_stage'],
                    confidence=dip_analysis['confidence'],
                    entry_timing_score=entry_timing_score,
                    recommended_allocation_pct=recommended_allocation,
                    reason=dip_analysis['reason']
                ))
                
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                continue
        
        # Sort by entry timing score (best first)
        opportunities.sort(key=lambda x: x.entry_timing_score, reverse=True)
        
        logger.info(f"Found {len(opportunities)} dip opportunities")
        
        return DipScanResponse(
            scan_time=datetime.now().isoformat(),
            universe_size=len(request.universe),
            opportunities_found=len(opportunities),
            dip_opportunities=opportunities,
            filters_applied={
                'min_asre': request.min_asre,
                'quality_tiers': request.quality_tiers,
                'max_distance_from_sma': request.max_distance_from_sma,
                'min_confidence': request.min_confidence
            },
            message=f"Found {len(opportunities)} dip opportunities out of {len(request.universe)} stocks scanned"
        )
        
    except Exception as e:
        logger.error(f"Dip scan failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")


@router.get("/{ticker}/historical", response_model=HistoricalDipResponse)
async def get_historical_dips(
    ticker: str,
    lookback_days: int = Query(730, description="Days to look back for dip patterns")
):
    """
    Analyze historical dip patterns and recovery times.
    
    **Returns:**
    - Past dip occurrences
    - Recovery times
    - Success rate
    - Average returns after dip entry
    
    **Example:** `/api/dip-analysis/NVDA/historical?lookback_days=730`
    """
    try:
        logger.info(f"Analyzing historical dips for {ticker}")
        
        # Fetch data
        df = fetch_stock_data(ticker, days=lookback_days)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        # Calculate distance from SMA
        df['distance_from_sma'] = ((df['close'] - df['sma_200']) / df['sma_200']) * 100
        
        # Find dip periods (below SMA-200)
        df['in_dip'] = df['distance_from_sma'] < 0
        
        # Identify dip starts
        df['dip_start'] = (df['in_dip']) & (~df['in_dip'].shift(1).fillna(False))
        
        dip_patterns = []
        dip_dates = df[df['dip_start']].index
        
        for dip_date in dip_dates:
            try:
                # Find dip period
                dip_df = df[df.index >= dip_date]
                entry_price = float(dip_df.iloc[0]['close'])
                
                # Find bottom
                dip_period = dip_df[dip_df['in_dip']]
                if dip_period.empty:
                    continue
                
                bottom_idx = dip_period['close'].idxmin()
                bottom_price = float(dip_period.loc[bottom_idx, 'close'])
                days_to_bottom = (bottom_idx - dip_date).days
                
                # Find recovery (back above SMA)
                recovery_df = dip_df[dip_df.index > bottom_idx]
                recovery_df = recovery_df[~recovery_df['in_dip']]
                
                if recovery_df.empty:
                    continue
                
                recovery_date = recovery_df.index[0]
                recovery_price = float(recovery_df.iloc[0]['close'])
                days_to_recovery = (recovery_date - dip_date).days
                
                # Calculate metrics
                max_drawdown_pct = ((bottom_price - entry_price) / entry_price) * 100
                recovery_return_pct = ((recovery_price - entry_price) / entry_price) * 100
                
                # Determine dip stage
                distance = float(dip_period.loc[bottom_idx, 'distance_from_sma'])
                if distance >= -5:
                    dip_stage = 'EARLY'
                elif distance >= -10:
                    dip_stage = 'MID'
                elif distance >= -15:
                    dip_stage = 'LATE'
                else:
                    dip_stage = 'DEEP'
                
                dip_patterns.append(HistoricalDipPattern(
                    dip_date=dip_date.strftime('%Y-%m-%d'),
                    entry_price=entry_price,
                    bottom_price=bottom_price,
                    recovery_price=recovery_price,
                    days_to_bottom=days_to_bottom,
                    days_to_recovery=days_to_recovery,
                    max_drawdown_pct=max_drawdown_pct,
                    recovery_return_pct=recovery_return_pct,
                    dip_stage=dip_stage
                ))
                
            except Exception as e:
                logger.error(f"Error processing dip at {dip_date}: {e}")
                continue
        
        # Calculate statistics
        successful_recoveries = sum(1 for p in dip_patterns if p.recovery_return_pct > 0)
        avg_recovery_days = np.mean([p.days_to_recovery for p in dip_patterns]) if dip_patterns else 0
        avg_recovery_return = np.mean([p.recovery_return_pct for p in dip_patterns]) if dip_patterns else 0
        
        return HistoricalDipResponse(
            ticker=ticker,
            analysis_period=f"{lookback_days} days",
            total_dips_found=len(dip_patterns),
            successful_recoveries=successful_recoveries,
            avg_recovery_days=float(avg_recovery_days),
            avg_recovery_return_pct=float(avg_recovery_return),
            dip_patterns=dip_patterns[-10:],  # Last 10 dips
            message=f"Analyzed {len(dip_patterns)} historical dips. Success rate: {(successful_recoveries/len(dip_patterns)*100):.1f}%" if dip_patterns else "No dips found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Historical analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/{ticker}/entry-timing", response_model=EntryTimingResponse)
async def get_entry_timing(ticker: str):
    """
    Optimize entry timing for dip buying.
    
    **Provides:**
    - Current dip stage
    - Optimal entry stage recommendation
    - Wait vs buy now suggestion
    - Expected price improvement if waiting
    
    **Example:** `/api/dip-analysis/NVDA/entry-timing`
    """
    try:
        # Get current dip analysis
        dip_analysis = await analyze_dip(ticker)
        
        # Determine optimal entry
        optimal_stage = "MID"  # MID stage is optimal
        current_stage = dip_analysis.dip_stage
        
        # Wait recommendation
        if current_stage == "EARLY":
            wait_recommendation = "⏳ WAIT - Price likely to dip further"
            estimated_improvement = 5.0
            reasoning = "Early stage dips typically drop another 3-7% before stabilizing"
        elif current_stage == "MID":
            wait_recommendation = "✅ BUY NOW - Optimal entry zone"
            estimated_improvement = 0.0
            reasoning = "Mid-stage offers best risk/reward balance"
        elif current_stage == "LATE":
            wait_recommendation = "✅ BUY NOW - Good entry, but consider partial position"
            estimated_improvement = 0.0
            reasoning = "Late stage is still acceptable, recovery may start soon"
        elif current_stage == "DEEP":
            wait_recommendation = "⚠️ CAUTION - Potential value trap, verify fundamentals"
            estimated_improvement = 0.0
            reasoning = "Deep dips may indicate structural issues. Ensure F-score remains strong."
        else:
            wait_recommendation = "❌ NO DIP - Wait for pullback"
            estimated_improvement = 0.0
            reasoning = "Stock is in uptrend, not a dip buying opportunity"
        
        return EntryTimingResponse(
            ticker=ticker,
            current_stage=current_stage,
            optimal_entry_stage=optimal_stage,
            current_timing_score=dip_analysis.entry_timing_score,
            wait_recommendation=wait_recommendation,
            estimated_better_entry_pct=estimated_improvement,
            reasoning=reasoning
        )
        
    except Exception as e:
        logger.error(f"Entry timing analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/quick-scan")
async def quick_scan(
    tickers: str = Query(..., description="Comma-separated list of tickers (e.g., NVDA,MSFT,GOOGL)")
):
    """
    Quick dip scan with default filters.
    
    **Example:** `/api/dip-analysis/quick-scan?tickers=NVDA,MSFT,GOOGL,META,AAPL`
    """
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        
        request = DipScanRequest(
            universe=ticker_list,
            min_asre=60.0,
            quality_tiers=['S', 'A', 'B'],
            max_distance_from_sma=-20.0,
            min_confidence=50.0
        )
        
        return await scan_universe_for_dips(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick scan failed: {str(e)}")
