"""
ASRE Service Layer
Wraps the existing ASRE algorithm for API consumption.

This service provides a clean interface to the ASRE rating engine,
with caching, batch processing, and error handling.
"""

import sys
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import logging
from datetime import datetime, timedelta
from threading import Lock
import traceback

# Add parent directories to path for ASRE imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
logger = logging.getLogger(__name__)

# Import ASRE modules
try:
    from asre.composite import compute_complete_asre
    from asre.data.fundamental_fetcher import FundamentalFetcher
    from asre.data_loader_indian import load_stock_data
    ASRE_AVAILABLE = True
    logger.info("[OK] ASRE modules imported successfully")
except ImportError as e:
    logger.warning(f"ASRE modules not available: {e}")
    ASRE_AVAILABLE = False

# Import HashLedger for audit trail (same ledger as CLI)
try:
    from asre.compliance.hash_ledger import HashLedger
    LEDGER_AVAILABLE = True
    logger.info("[OK] HashLedger imported — API runs will be recorded in ledger")
except ImportError as exc:
    logger.warning(f"HashLedger not available — audit ledger will NOT be updated: {exc}")
    LEDGER_AVAILABLE = False

from api.config import settings


# ============================================================================
# CACHE MANAGER
# ============================================================================

class RatingCache:
    """Thread-safe in-memory cache for stock ratings"""
    
    def __init__(self, expiry_hours: int = 24):
        self._cache: Dict[str, Dict] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._lock = Lock()
        self.expiry_hours = expiry_hours
    
    def get(self, ticker: str) -> Optional[Dict]:
        """Get cached rating if not expired"""
        with self._lock:
            ticker = ticker.upper()
            
            if ticker not in self._cache:
                return None
            
            # Check expiry
            if ticker in self._timestamps:
                age = datetime.now() - self._timestamps[ticker]
                if age > timedelta(hours=self.expiry_hours):
                    logger.debug(f"Cache expired for {ticker} (age: {age})")
                    del self._cache[ticker]
                    del self._timestamps[ticker]
                    return None
            
            logger.debug(f"Cache hit for {ticker}")
            return self._cache[ticker]
    
    def set(self, ticker: str, data: Dict):
        """Store rating in cache"""
        with self._lock:
            ticker = ticker.upper()
            self._cache[ticker] = data
            self._timestamps[ticker] = datetime.now()
            logger.debug(f"Cached rating for {ticker}")
    
    def clear(self, ticker: Optional[str] = None):
        """Clear cache for specific ticker or all"""
        with self._lock:
            if ticker:
                ticker = ticker.upper()
                self._cache.pop(ticker, None)
                self._timestamps.pop(ticker, None)
                logger.info(f"Cleared cache for {ticker}")
            else:
                self._cache.clear()
                self._timestamps.clear()
                logger.info("Cleared entire cache")
    
    def stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            return {
                "size": len(self._cache),
                "tickers": list(self._cache.keys()),
                "oldest_entry": min(self._timestamps.values()) if self._timestamps else None,
                "newest_entry": max(self._timestamps.values()) if self._timestamps else None,
            }


# ============================================================================
# ASRE SERVICE
# ============================================================================

class ASREService:
    """
    ASRE Algorithm Service Wrapper
    
    Provides high-level interface to ASRE rating engine with:
    - Automatic caching
    - Batch processing
    - Error handling
    - Performance monitoring
    """
    
    # Class-level cache (shared across all instances)
    _cache = RatingCache(expiry_hours=settings.CACHE_EXPIRY_HOURS)
    
    @classmethod
    def get_stock_rating(
        cls, 
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> Dict:
        """
        Get comprehensive ASRE rating for a single stock.
        """

        ticker = ticker.upper().strip()

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=1095)).strftime("%Y-%m-%d")

        if not force_refresh:
            cached_data = cls._cache.get(ticker)
            if cached_data:
                logger.info(f"[CACHE HIT] Returning cached data for {ticker}")
                return cached_data

        if not ASRE_AVAILABLE:
            raise RuntimeError("ASRE core modules not available")

        logger.info(f"[PROCESSING] Fetching ASRE rating for {ticker}...")
        start_time = datetime.now()

        # ── Append .NS suffix for Yahoo Finance / ASRE pipeline ──────────────
        # Config stores clean names (RELIANCE), ASRE engine needs RELIANCE.NS
        yf_ticker = ticker + '.NS'

        try:
        # ----------------------------
        # STEP 1: FUNDAMENTALS
        # ----------------------------
            fetcher = FundamentalFetcher(cache_dir=str(settings.FUNDAMENTALS_CACHE_DIR))
            quarterly_result = fetcher.fetch_quarterly_fundamentals(
                yf_ticker,
                start_date=start_date,
                end_date=end_date
            )

            quarterly_fundamentals = (
                quarterly_result[0]
                if isinstance(quarterly_result, tuple)
                else quarterly_result
            )

            if quarterly_fundamentals is None or getattr(quarterly_fundamentals, "empty", False):
                raise RuntimeError(f"No fundamentals data for {yf_ticker}")

            logger.info(f"[FUNDAMENTALS] {yf_ticker}: {len(quarterly_fundamentals)} quarters fetched")

        # ----------------------------
        # STEP 2: PRICE DATA
        # ----------------------------
            df = load_stock_data(
                yf_ticker,
                start=start_date,
                end=end_date,
                quarterly_fundamentals=quarterly_fundamentals
            )

            if isinstance(df, str):
                raise RuntimeError(f"DataLoader error for {yf_ticker}: {df}")

            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Expected DataFrame, got {type(df)}")

            if df.empty:
                raise RuntimeError(f"Empty DataFrame for {yf_ticker}")

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            logger.info(f"[DATA] {yf_ticker}: {len(df)} rows loaded")

        # ----------------------------
        # STEP 3: ASRE CORE
        # ----------------------------
            from asre.composite import compute_asre_rating, compute_asre_medallion

            result_df = compute_asre_rating(
                df=df,
                ticker=yf_ticker,
                return_components=True
            )

            if not isinstance(result_df, pd.DataFrame):
                raise TypeError(f"ASRE rating returned {type(result_df)}")

            result_df = compute_asre_medallion(
                df=result_df,
                ticker=yf_ticker,
                return_components=True
            )

            if result_df is None or result_df.empty:
                raise RuntimeError(f"ASRE computation returned empty result for {yf_ticker}")

            # ----------------------------
            # STEP 4: BUILD RESPONSE
            # ----------------------------
            latest_row = result_df.iloc[-1]

            # Generate the run UUID now — shared between ledger entry and API response
            run_uuid = str(uuid.uuid4())

            # Pass the clean ticker (no .NS) so the response stays consistent
            rating_data = cls._build_rating_response(
                ticker, latest_row, result_df, run_uuid=run_uuid
            )

            cls._cache.set(ticker, rating_data)

            # ── Write to the SAME hash ledger as the CLI ──────────────────────
            # This makes every API lookup auditable alongside CLI runs.
            if LEDGER_AVAILABLE:
                try:
                    entry_hash = HashLedger.append(
                        run_id     = run_uuid,
                        score_hash = rating_data.get('score_hash', ''),
                        mode       = 'api',
                        tickers    = [yf_ticker],   # use .NS form (consistent with CLI)
                        pdf_paths  = [],
                    )
                    logger.info(
                        "[LEDGER] API run recorded — run=%s hash=%s ticker=%s",
                        run_uuid[:8], entry_hash[:16], yf_ticker,
                    )
                except Exception as ledger_exc:
                    # Never let ledger failure crash the API response
                    logger.warning("[LEDGER] append failed — %s", ledger_exc)

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"[SUCCESS] {ticker}: R_Final={rating_data['rfinal']:.1f}, "
                f"Signal={rating_data['signal']}, Time={elapsed:.2f}s"
            )

            return rating_data

        except Exception as e:
            logger.error(f"[FATAL] ASRE pipeline failed for {ticker}: {e}")
            logger.debug(traceback.format_exc())

        # HARD FAIL — DO NOT MASK WITH MOCK DATA
            raise RuntimeError(f"ASRE computation failed for {ticker}") from e
        
    @classmethod
    def get_multiple_stocks(
        cls, 
        tickers: List[str],
        continue_on_error: bool = True
    ) -> List[Dict]:
        """Get ratings for multiple stocks"""
        results = []
        errors = []
        
        logger.info(f"[BATCH] Processing {len(tickers)} stocks...")
        
        for ticker in tickers:
            try:
                rating = cls.get_stock_rating(ticker)
                results.append(rating)
            except Exception as e:
                error_entry = {
                    'ticker': ticker.upper(),
                    'error': str(e),
                    'rfinal': None,
                    'signal': 'ERROR',
                    'timestamp': datetime.now().isoformat()
                }
                
                if continue_on_error:
                    logger.warning(f"[SKIP] {ticker}: {e}")
                    results.append(error_entry)
                    errors.append(ticker)
                else:
                    raise
        
        if errors:
            logger.warning(f"[BATCH] Completed with {len(errors)} errors: {', '.join(errors)}")
        else:
            logger.info(f"[BATCH] Successfully processed all {len(tickers)} stocks")
        
        return results
    
    @classmethod
    def compare_stocks(cls, tickers: List[str]) -> List[Dict]:
        """Compare multiple stocks and rank by ASRE rating"""
        logger.info(f"[COMPARE] Comparing {len(tickers)} stocks...")
        
        # Get ratings for all stocks
        results = cls.get_multiple_stocks(tickers, continue_on_error=True)
        
        # Filter out errors
        valid_results = [r for r in results if r.get('rfinal') is not None]
        error_count = len(results) - len(valid_results)
        
        if error_count > 0:
            logger.warning(f"[COMPARE] {error_count} stocks failed, comparing {len(valid_results)}")
        
        # Sort by rating (descending)
        sorted_results = sorted(
            valid_results, 
            key=lambda x: x.get('rfinal', 0), 
            reverse=True
        )
        
        # Add rank
        for i, result in enumerate(sorted_results, 1):
            result['rank'] = i
        
        if len(sorted_results) >= 3:
            logger.info(
                f"[COMPARE] Top 3: "
                f"1. {sorted_results[0]['ticker']} ({sorted_results[0]['rfinal']:.1f}), "
                f"2. {sorted_results[1]['ticker']} ({sorted_results[1]['rfinal']:.1f}), "
                f"3. {sorted_results[2]['ticker']} ({sorted_results[2]['rfinal']:.1f})"
            )
        
        return sorted_results
    
    @classmethod
    def _build_rating_response(
        cls,
        ticker: str,
        latest_row: pd.Series,
        full_df: pd.DataFrame,
        run_uuid: Optional[str] = None,   # Real UUID4 shared with ledger entry
    ) -> Dict:
        """Build standardized rating response from ASRE output."""

        # Extract core scores (with fallbacks)
        rfinal = float(latest_row.get('r_final', 0))
        rasre = float(latest_row.get('r_asre', rfinal))
        fscore = float(latest_row.get('f_score', 0))
        tscore = float(latest_row.get('t_score', 0))
        mscore = float(latest_row.get('mscore_adj', latest_row.get('m_score', 0)))
        
        # Determine signal
        signal = cls._determine_signal(rfinal)
        
        # Extract category
        category = str(latest_row.get('category', 'UNKNOWN')).upper()
        
        # Extract dip analysis
        dip_quality = latest_row.get('dip_quality_score')
        if dip_quality is not None:
            dip_quality = float(dip_quality)
        
        dip_stage = latest_row.get('dip_stage')
        if dip_stage is not None:
            dip_stage = str(dip_stage).upper()
        
        # Extract context flags
        context = cls._get_context_flags(latest_row)
        
        # Extract additional metadata
        peg_ratio = latest_row.get('peg')
        if peg_ratio is not None:
            peg_ratio = float(peg_ratio)
        
        quality_tier = latest_row.get('quality_tier')
        if quality_tier is not None:
            quality_tier = str(quality_tier).upper()
        
        close_price = latest_row.get('close')
        if close_price is not None:
            close_price = float(close_price)
        
        # Compute score hash
        score_hash = cls._compute_score_hash(full_df)

        # run_id: use the real UUID from the ledger entry when available,
        # fall back to display-format for cache replays.
        run_id = run_uuid if run_uuid else cls._build_run_id(ticker)

        # Build response
        return {
            'ticker': ticker,
            'rfinal': round(rfinal, 2),
            'rasre': round(rasre, 2),
            'fscore': round(fscore, 2),
            'tscore': round(tscore, 2),
            'mscore': round(mscore, 2),
            'signal': signal,
            'category': category,
            'dip_quality': round(dip_quality, 2) if dip_quality else None,
            'dip_stage': dip_stage,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'peg_ratio': round(peg_ratio, 2) if peg_ratio else None,
            'quality_tier': quality_tier,
            'close_price': round(close_price, 2) if close_price else None,
            'run_id': run_id,
            'score_hash': score_hash,
            '_raw_scores': {
                'f_score': fscore,
                't_score': tscore,
                'm_score': mscore,
            },
            '_data_points': len(full_df),
        }
    
    @classmethod
    def _build_run_id(cls, ticker: str) -> str:
        """
        Fallback display-format run ID (used only for cache replays where
        the original UUID is not available). Live runs use the real UUID4.
        Format: ASRE-{TICKER}-{YYYYMMDD}-{SEQ3}
        """
        date_str = datetime.now().strftime("%Y%m%d")
        seq = str(uuid.uuid4()).replace('-', '')[:3].upper()
        seq_num = str(int(seq, 16) % 999 + 1).zfill(3)
        short_ticker = ticker[:6].upper()
        return f"ASRE-{short_ticker}-{date_str}-{seq_num}"

    @staticmethod
    def _compute_score_hash(full_df: pd.DataFrame) -> Optional[str]:
        """
        Compute SHA-256 of the score DataFrame — identical logic to cli.py line 1664.
        Hash is over: f_score, t_score, m_score, r_final, r_asre columns as CSV.
        Returns None if the columns are unavailable.
        """
        try:
            score_cols = [c for c in ('f_score', 't_score', 'm_score', 'r_final', 'r_asre')
                          if c in full_df.columns]
            if not score_cols:
                return None
            hash_csv = full_df[score_cols].to_csv().encode('utf-8')
            return hashlib.sha256(hash_csv).hexdigest()
        except Exception as exc:
            logger.debug("score_hash computation failed — %s", exc)
            return None

    @classmethod
    def _get_mock_rating(cls, ticker: str) -> Dict:
        """Return mock rating data for Indian NSE stocks when ASRE unavailable."""

        # Realistic mock values for Indian NSE blue-chips
        mock_data = {
            # IT
            "TCS":        (78.5, 82.0, 52.0, 48.0, "IT SERVICES"),
            "INFY":       (72.3, 76.0, 45.0, 55.0, "IT SERVICES"),
            "WIPRO":      (61.8, 65.0, 38.0, 42.0, "IT SERVICES"),
            "HCLTECH":    (74.1, 78.0, 50.0, 60.0, "IT SERVICES"),
            # Banking
            "HDFCBANK":   (81.2, 85.0, 68.0, 62.0, "PRIVATE BANK"),
            "ICICIBANK":  (79.6, 83.0, 65.0, 70.0, "PRIVATE BANK"),
            "SBIN":       (65.4, 70.0, 55.0, 58.0, "PSU BANK"),
            "KOTAKBANK":  (76.8, 80.0, 60.0, 55.0, "PRIVATE BANK"),
            "AXISBANK":   (68.9, 72.0, 52.0, 60.0, "PRIVATE BANK"),
            "BAJFINANCE": (83.4, 87.0, 72.0, 68.0, "NBFC"),
            # Energy / Conglomerate
            "RELIANCE":   (84.7, 88.0, 75.0, 65.0, "CONGLOMERATE"),
            "ONGC":       (58.2, 62.0, 42.0, 38.0, "PSU ENERGY"),
            # FMCG
            "HINDUNILVR": (77.3, 80.0, 58.0, 52.0, "FMCG"),
            "ITC":        (70.1, 74.0, 55.0, 60.0, "FMCG"),
            # Auto
            "MARUTI":     (73.6, 77.0, 62.0, 55.0, "AUTO"),
            "TATAMOTORS": (66.8, 70.0, 58.0, 65.0, "AUTO"),
            # Pharma
            "SUNPHARMA":  (71.9, 76.0, 52.0, 48.0, "PHARMA"),
            "DRREDDY":    (69.4, 73.0, 48.0, 45.0, "PHARMA"),
        }

        if ticker in mock_data:
            rfinal, fscore, tscore, mscore, category = mock_data[ticker]
        else:
            rfinal, fscore, tscore, mscore, category = (50.0, 55.0, 35.0, 40.0, "NSE STOCK")

        return {
            "ticker": ticker,
            "rfinal": round(rfinal, 2),
            "rasre": round(rfinal, 2),
            "fscore": round(fscore, 2),
            "tscore": round(tscore, 2),
            "mscore": round(mscore, 2),
            "signal": cls._determine_signal(rfinal),
            "category": category,
            "dip_quality": 65.0,
            "dip_stage": "MID",
            "context": "DIP" if rfinal > 70 else "",
            "timestamp": datetime.now().isoformat(),
            "peg_ratio": None,
            "quality_tier": "A" if fscore > 80 else "B",
            "close_price": None,
            "_raw_scores": {"f_score": fscore, "t_score": tscore, "m_score": mscore},
            "_data_points": 100,
        }
    
    @staticmethod
    def _determine_signal(rfinal: float) -> str:
        """Convert ASRE rating to buy/sell signal"""
        if rfinal >= 80:
            return "STRONG BUY"
        elif rfinal >= 65:
            return "BUY"
        elif rfinal >= 50:
            return "HOLD"
        elif rfinal >= 35:
            return "CAUTION"
        else:
            return "STRONG SELL"
    
    @staticmethod
    def _get_context_flags(row: pd.Series) -> str:
        """Extract context flags from ASRE row"""
        flags = []
        
        if row.get('is_buy_dip', False):
            flags.append("DIP")
        
        if row.get('is_momentum_trap', False):
            flags.append("TRAP")
        
        if row.get('is_pump_risk', False):
            flags.append("PUMP")
        
        quality_tier = row.get('quality_tier', '')
        if quality_tier == 'S':
            flags.append("EXCEPTIONAL")
        elif quality_tier == 'A':
            flags.append("HIGH QUALITY")
        
        return " ".join(flags) if flags else ""
    
    @classmethod
    def clear_cache(cls, ticker: Optional[str] = None):
        """Clear rating cache"""
        cls._cache.clear(ticker)
    
    @classmethod
    def get_cache_stats(cls) -> Dict:
        """Get cache statistics"""
        return cls._cache.stats()
    
    @classmethod
    def health_check(cls) -> Dict:
        """Check service health"""
        return {
            'asre_available': ASRE_AVAILABLE,
            'cache_size': len(cls._cache._cache),
            'supported_stocks_count': len(settings.SUPPORTED_STOCKS),
            'cache_expiry_hours': cls._cache.expiry_hours,
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_stock_rating(ticker: str, force_refresh: bool = False) -> Dict:
    """Convenience function for ASREService.get_stock_rating()"""
    return ASREService.get_stock_rating(ticker, force_refresh=force_refresh)


def get_multiple_stocks(tickers: List[str]) -> List[Dict]:
    """Convenience function for ASREService.get_multiple_stocks()"""
    return ASREService.get_multiple_stocks(tickers)


def compare_stocks(tickers: List[str]) -> List[Dict]:
    """Convenience function for ASREService.compare_stocks()"""
    return ASREService.compare_stocks(tickers)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ASREService',
    'RatingCache',
    'get_stock_rating',
    'get_multiple_stocks',
    'compare_stocks',
]
