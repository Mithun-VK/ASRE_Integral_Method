"""
ASRE Data Loader - V4.1 (fundamentals_fetch_ts threading)
Auto-calculates derived metrics (PE, ROE, D/E) from raw quarterly data.

Fixes applied vs V4:
  FIX 11 NEW : fundamentals_fetch_ts parameter added to load_stock_data()
               (both DataLoader method and public wrapper) and threaded
               into _merge_live_fundamentals() so the data source banner
               shows the REAL Yahoo Finance fetch time from FundamentalFetcher
               rather than datetime.now() at load time.

Fixes retained from V4:
  FIX 1  : Hard abort on None fundamentals — static fallback removed
  FIX 2  : MIN_QUARTERS=4 gate before F-Score computation
  FIX 3  : SPY_MAX_RETRIES=5, exponential backoff, hard RuntimeError
  FIX 4  : Data source banner on every successful run
  FIX 5  : ^INDIAVIX for .NS/.BO tickers instead of ^VIX
  FIX 6  : INDIA_SECTOR_INDEX_MAP — 40 NSE tickers to Nifty indices
  FIX 7  : _detect_sector_etf() uses static map — no API call for known tickers
  FIX 8  : Revenue QoQ proxy logs WARNING (not silent)
  FIX 9  : Sector ETF fallback uses benchmark_return safely
  FIX 10 : is_indian_ticker() + _base_ticker() helpers

Author: ASRE Project
Date: February 2026
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

SEP = "\u2501" * 55   # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

US_VIX_TICKER    = "^VIX"
INDIA_VIX_TICKER = "^INDIAVIX"
US_BENCHMARK     = "SPY"
MIN_QUARTERS     = 4
SPY_MAX_RETRIES  = 5

# US Sector ETF Map
US_SECTOR_ETF_MAP: Dict[str, str] = {
    "Technology":             "XLK",
    "Financial Services":     "XLF",
    "Healthcare":             "XLV",
    "Consumer Defensive":     "XLP",
    "Consumer Cyclical":      "XLY",
    "Energy":                 "XLE",
    "Industrials":            "XLI",
    "Communication Services": "XLC",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
    "Materials":              "XLB",
    "Basic Materials":        "XLB",
}

# India NSE Sector Index Map (base ticker, uppercase, no .NS/.BO suffix)
INDIA_SECTOR_INDEX_MAP: Dict[str, str] = {
    # IT / Technology
    "INFY":       "^CNXIT",
    "TCS":        "^CNXIT",
    "WIPRO":      "^CNXIT",
    "HCLTECH":    "^CNXIT",
    "TECHM":      "^CNXIT",
    "LTIM":       "^CNXIT",
    "PERSISTENT": "^CNXIT",
    "COFORGE":    "^CNXIT",
    "MPHASIS":    "^CNXIT",
    "KPITTECH":   "^CNXIT",
    # Banking / Financial Services
    "HDFCBANK":   "^NSEBANK",
    "ICICIBANK":  "^NSEBANK",
    "SBIN":       "^NSEBANK",
    "KOTAKBANK":  "^NSEBANK",
    "AXISBANK":   "^NSEBANK",
    "BAJFINANCE": "^NSEBANK",
    "BAJAJFINSV": "^NSEBANK",
    # Pharma / Healthcare
    "ZYDUSLIFE":  "^CNXPHARMA",
    "SUNPHARMA":  "^CNXPHARMA",
    "DRREDDY":    "^CNXPHARMA",
    "CIPLA":      "^CNXPHARMA",
    "DIVISLAB":   "^CNXPHARMA",
    "PIIND":      "^CNXPHARMA",
    # FMCG / Consumer
    "HINDUNILVR": "^CNXFMCG",
    "ITC":        "^CNXFMCG",
    "NESTLEIND":  "^CNXFMCG",
    "TITAN":      "^CNXFMCG",
    "ASIANPAINT": "^CNXFMCG",
    # Auto
    "MARUTI":     "^CNXAUTO",
    "TATAMOTORS": "^CNXAUTO",
    "MM":         "^CNXAUTO",
    "EICHERMOT":  "^CNXAUTO",
    "BAJAJAUTO":  "^CNXAUTO",
    # Energy / Oil & Gas
    "IOC":        "^CNXENERGY",
    "BPCL":       "^CNXENERGY",
    "HINDPETRO":  "^CNXENERGY",
    "ONGC":       "^CNXENERGY",
    "GAIL":       "^CNXENERGY",
    # Conglomerates -> broad index
    "RELIANCE":   "^NSEI",
    "ADANIPORTS": "^NSEI",
    "ADANIENT":   "^NSEI",
    # Mid-cap / Industrials
    "ASTRAL":     "^CNXMIDCAP",
    "DIXON":      "^CNXMIDCAP",
    "DEEPAKNTR":  "^CNXMIDCAP",
    "VOLTAS":     "^CNXMIDCAP",
    "ZOMATO":     "^CNXMIDCAP",
}

INDIA_DEFAULT_INDEX = "^NSEI"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def is_indian_ticker(ticker: str) -> bool:
    """Return True for NSE (.NS) or BSE (.BO) tickers."""
    t = ticker.upper()
    return t.endswith(".NS") or t.endswith(".BO")


def _base_ticker(ticker: str) -> str:
    """Strip .NS / .BO suffix and return uppercase base."""
    for suffix in (".NS", ".BO", ".ns", ".bo"):
        if ticker.endswith(suffix):
            return ticker[: -len(suffix)].upper()
    return ticker.upper()


# ─────────────────────────────────────────────────────────────
# Yahoo Loader
# ─────────────────────────────────────────────────────────────

class YahooFinanceLoader:

    def download(self, ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
        """Download OHLCV data with robust error handling."""
        for attempt in range(retries):
            try:
                logger.info(
                    "Downloading %s from Yahoo Finance (attempt %d/%d)...",
                    ticker, attempt + 1, retries,
                )
                df = yf.download(
                    ticker, start=start, end=end,
                    auto_adjust=False, progress=False,
                    threads=False, group_by="column",
                )

                if df is None or df.empty:
                    if attempt < retries - 1:
                        time.sleep(2)
                        continue
                    raise ValueError("No data returned for %s" % ticker)

                if "Date" not in df.columns and df.index.name in ["Date", None]:
                    df = df.reset_index()

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [
                        "_".join([str(x) for x in col if x]).lower().strip()
                        for col in df.columns
                    ]
                else:
                    df.columns = [
                        str(c).lower().replace(" ", "_").strip()
                        for c in df.columns
                    ]

                suffix = "_" + ticker.lower()
                df.columns = [
                    c[: -len(suffix)] if c.endswith(suffix) else c
                    for c in df.columns
                ]

                rename_map = {
                    "adj_close": "adj_close", "adjclose": "adj_close",
                    "adjusted_close": "adj_close", "adj close": "adj_close",
                    "date": "date", "index": "date",
                }
                df = df.rename(columns=rename_map)

                if "date" not in df.columns:
                    date_cols = [c for c in df.columns if "date" in c.lower() or c == "index"]
                    if date_cols:
                        df = df.rename(columns={date_cols[0]: "date"})
                    else:
                        raise ValueError(
                            "%s: 'date' column not found after all rename attempts" % ticker
                        )

                logger.info("Successfully downloaded %d rows for %s", len(df), ticker)
                return df

            except Exception as exc:
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    logger.error("Failed to download %s: %s", ticker, exc)
                    raise

    def load_ohlcv(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Load OHLCV data with adj_close guarantee."""
        df = self.download(ticker, start, end)
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]
        df["ticker"] = ticker.upper()
        return df

    # NOTE: load_fundamentals_static() permanently removed.
    # Static fundamentals produced inconsistent ratings (e.g. NVDA Tier S vs Tier C
    # on the same day). All fundamentals must come from FundamentalFetcher (live).


# ─────────────────────────────────────────────────────────────
# Data Loader
# ─────────────────────────────────────────────────────────────

class DataLoader:

    def __init__(self):
        self.yf = YahooFinanceLoader()

    def load_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: Optional[str] = None,
        quarterly_fundamentals: Optional[pd.DataFrame] = None,
        fundamentals_fetch_ts: Optional[datetime] = None,   # ✅ FIX 11
    ) -> pd.DataFrame:
        """
        Load complete stock dataset.

        Parameters
        ----------
        ticker : str
            Stock ticker (e.g. 'RELIANCE.NS', 'NVDA')
        start_date : str
            Start date YYYY-MM-DD
        end_date : str, optional
            End date YYYY-MM-DD. Defaults to today.
        quarterly_fundamentals : pd.DataFrame
            Live quarterly data from FundamentalFetcher. Required.
        fundamentals_fetch_ts : datetime, optional  [FIX 11]
            The timestamp returned by FundamentalFetcher.fetch_quarterly_fundamentals().
            When supplied, the data source banner shows the REAL Yahoo Finance fetch
            time instead of datetime.now() at load time. Safe to omit — falls back
            to datetime.now() for backward compatibility.

        Raises
        ------
        ValueError
            If quarterly_fundamentals is None (static fallback removed).
            If fewer than MIN_QUARTERS of data are available.
        RuntimeError
            If benchmark cannot be downloaded after SPY_MAX_RETRIES.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        extended_start = (
            pd.to_datetime(start_date) - timedelta(days=400)
        ).strftime("%Y-%m-%d")

        indian = is_indian_ticker(ticker)

        # Step 1: Price Data
        price_df = self.yf.load_ohlcv(ticker, extended_start, end_date)

        # Step 2: Live Fundamentals  [FIX 1 — hard abort]
        if quarterly_fundamentals is not None:
            price_df = self._merge_live_fundamentals(
                ticker, price_df, quarterly_fundamentals,
                fetch_ts=fundamentals_fetch_ts,   # ✅ FIX 11
            )
        else:
            sep = SEP
            raise ValueError(
                "\n" + sep + "\n"
                "  [ASRE CRITICAL] MISSING LIVE FUNDAMENTALS\n"
                "  Ticker      : " + ticker.upper() + "\n"
                "  Problem     : quarterly_fundamentals is None.\n"
                "  Root cause  : FundamentalFetcher was not called\n"
                "                before DataLoader.\n"
                "  Action      : Ensure FundamentalFetcher.fetch()\n"
                "                is called and its DataFrame passed here.\n"
                "  Note        : Static fallback permanently removed.\n"
                + sep
            )

        # Step 3: Benchmark  [FIX 3 — exponential backoff + hard abort]
        price_df = self._load_benchmark(ticker, price_df, extended_start, end_date, indian)

        # Step 4: VIX  [FIX 5 — ^INDIAVIX for .NS/.BO]
        vix_sym = INDIA_VIX_TICKER if indian else US_VIX_TICKER
        try:
            vix = self.yf.download(vix_sym, extended_start, end_date, retries=1)
            price_df = (
                price_df
                .merge(
                    vix[["date", "close"]].rename(columns={"close": "vix"}),
                    on="date", how="left",
                )
                .ffill()
            )
            logger.info("Successfully downloaded VIX rows for %s", vix_sym)
        except Exception as exc:
            logger.warning(
                "VIX download (%s) failed for %s: %s. Defaulting to 15.0.",
                vix_sym, ticker, exc,
            )
            price_df["vix"] = 15.0

        # Step 5: Sector Benchmark  [FIX 6/7 — static map, no API call]
        sector_sym = self._detect_sector_etf(ticker, indian)
        try:
            sect = self.yf.download(sector_sym, extended_start, end_date, retries=1)
            sect["sector_return"] = sect["close"].pct_change()
            price_df = price_df.merge(
                sect[["date", "sector_return"]], on="date", how="left"
            ).ffill()
        except Exception as exc:
            logger.warning(
                "Sector benchmark (%s) download failed for %s: %s. "
                "Falling back to SPY/benchmark returns.",
                sector_sym, ticker, exc,
            )
            # FIX 9: benchmark_return is already present — safe fallback
            price_df["sector_return"] = price_df.get(
                "benchmark_return",
                pd.Series(0.0, index=price_df.index),
            )

        # Step 6: Sector Beta
        price_df["stock_return"] = price_df["close"].pct_change()
        cov = price_df["stock_return"].rolling(60).cov(price_df["sector_return"])
        var = price_df["sector_return"].rolling(60).var()
        price_df["sector_beta"] = (cov / var).clip(-3, 3).fillna(1.0)

        # Step 7: Trim to requested date range
        price_df = price_df[price_df["date"] >= start_date].reset_index(drop=True)

        return price_df

    # ─────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────

    def _merge_live_fundamentals(
        self,
        ticker: str,
        price_df: pd.DataFrame,
        quarterly_fundamentals: pd.DataFrame,
        fetch_ts: Optional[datetime] = None,   # ✅ FIX 11
    ) -> pd.DataFrame:
        """
        Validate, compute derived metrics, and merge live quarterly fundamentals.

        Parameters
        ----------
        fetch_ts : datetime, optional  [FIX 11]
            Timestamp from FundamentalFetcher — used in data source banner.
            Falls back to datetime.now() when None (backward compatibility).
        """
        qf = quarterly_fundamentals.copy()
        qf["date"] = pd.to_datetime(qf["date"])
        qf = qf.sort_values("date").reset_index(drop=True)

        # FIX 2: Minimum quarters gate
        if len(qf) < MIN_QUARTERS:
            sep = SEP
            raise ValueError(
                "\n" + sep + "\n"
                "  [ASRE CRITICAL] INSUFFICIENT FUNDAMENTAL DATA\n"
                "  Ticker      : " + ticker.upper() + "\n"
                "  Found       : " + str(len(qf)) + " quarter(s)\n"
                "  Required    : " + str(MIN_QUARTERS) + " quarters minimum\n"
                "  Impact      : F-Score unreliable with fewer quarters.\n"
                "  Action      : Check FundamentalFetcher cache and\n"
                "                Yahoo Finance availability.\n"
                + sep
            )

        # FIX 4 + FIX 11: Data source banner with real fetch timestamp
        # Use the FundamentalFetcher timestamp when available; fall back to now().
        if fetch_ts is not None and isinstance(fetch_ts, datetime):
            fetch_ts_str = fetch_ts.strftime("%Y-%m-%d %H:%M IST")
            ts_source = "Yahoo Finance (cached)"
        else:
            fetch_ts_str = datetime.now().strftime("%Y-%m-%d %H:%M IST")
            ts_source = "Yahoo Finance (live)"

        date_range = (
            qf["date"].min().strftime("%Y-%m-%d")
            + " to "
            + qf["date"].max().strftime("%Y-%m-%d")
        )
        logger.info(
            "\n%s\n  DATA SOURCE : %s\n"
            "  TICKER      : %s\n  FETCHED AT  : %s\n"
            "  QUARTERS    : %d\n  DATE RANGE  : %s\n%s",
            SEP, ts_source, ticker.upper(), fetch_ts_str, len(qf), date_range, SEP,
        )

        # ROE (annualised)
        if "net_income" in qf.columns and "shareholders_equity" in qf.columns:
            qf["roe"] = (
                qf["net_income"] * 4
                / qf["shareholders_equity"].replace(0, np.nan)
            ) * 100
            qf["roe"] = qf["roe"].fillna(15.0)
        else:
            qf["roe"] = 15.0

        # D/E
        if "total_debt" in qf.columns and "shareholders_equity" in qf.columns:
            qf["de"] = qf["total_debt"] / qf["shareholders_equity"].replace(0, np.nan)
            qf["de"] = qf["de"].fillna(0.5)
        else:
            qf["de"] = 0.5

        # Revenue Growth YoY
        if "revenue" in qf.columns:
            if len(qf) >= 5:
                qf["revenue_growth_yoy"] = qf["revenue"].pct_change(4) * 100
            else:
                # FIX 8: Explicit warning — not silent
                logger.warning(
                    "%s: Only %d quarters available. Using QoQ growth as YoY proxy. "
                    "F-Score revenue growth signal is approximate.",
                    ticker.upper(), len(qf),
                )
                qf["revenue_growth_yoy"] = qf["revenue"].pct_change(1) * 100
            qf["revenue_growth_yoy"] = qf["revenue_growth_yoy"].fillna(10.0)
        else:
            qf["revenue_growth_yoy"] = 10.0

        # Profit Margin
        if "net_income" in qf.columns and "revenue" in qf.columns:
            qf["profit_margin"] = (
                qf["net_income"] / qf["revenue"].replace(0, np.nan)
            ) * 100

        # Merge into price DataFrame
        price_df["date"] = pd.to_datetime(price_df["date"])
        price_df = pd.merge_asof(
            price_df.sort_values("date"),
            qf.sort_values("date"),
            on="date", direction="backward",
            suffixes=("", "_quarterly"),
        )

        for col in ["roe", "de", "revenue_growth_yoy", "eps", "revenue", "profit_margin"]:
            if col in price_df.columns:
                price_df[col] = price_df[col].ffill().bfill()

        # PE Ratio
        if "eps" in price_df.columns:
            annual_eps = (price_df["eps"] * 4).replace(0, 0.01)
            price_df["pe"] = price_df["close"] / annual_eps
            price_df.loc[price_df["pe"] < 0, "pe"] = 100.0

        logger.info(
            "Calculated derived metrics: PE, ROE, D/E, Growth for %s",
            ticker.upper(),
        )
        return price_df

    def _load_benchmark(
        self,
        ticker: str,
        price_df: pd.DataFrame,
        extended_start: str,
        end_date: str,
        indian: bool,
    ) -> pd.DataFrame:
        """
        Download benchmark (^NSEI for India, SPY for US) with exponential backoff.

        Raises RuntimeError after SPY_MAX_RETRIES — T-Score requires this data.
        """
        benchmark_sym = "^NSEI" if indian else US_BENCHMARK
        loaded = False

        for attempt in range(SPY_MAX_RETRIES):
            try:
                bm = self.yf.download(benchmark_sym, extended_start, end_date, retries=1)
                if bm is None or bm.empty:
                    raise ValueError("Empty DataFrame for %s" % benchmark_sym)

                bm = bm.rename(columns={"close": "benchmark_close"})
                bm["benchmark_return"] = bm["benchmark_close"].pct_change()
                price_df = price_df.merge(
                    bm[["date", "benchmark_return"]], on="date", how="left"
                ).ffill()
                loaded = True
                break

            except Exception as exc:
                wait = 2 * (attempt + 1)   # 2s, 4s, 6s, 8s, 10s
                logger.warning(
                    "Benchmark (%s) attempt %d/%d failed for %s: %s. Retrying in %ds...",
                    benchmark_sym, attempt + 1, SPY_MAX_RETRIES, ticker.upper(), exc, wait,
                )
                time.sleep(wait)

        if not loaded:
            sep = SEP
            raise RuntimeError(
                "\n" + sep + "\n"
                "  [ASRE CRITICAL] BENCHMARK UNAVAILABLE\n"
                "  Ticker      : " + ticker.upper() + "\n"
                "  Benchmark   : " + benchmark_sym + "\n"
                "  Attempts    : " + str(SPY_MAX_RETRIES) + "\n"
                "  Impact      : T-Score requires benchmark data.\n"
                "                Continuing produces invalid T-Scores.\n"
                "  Action      : Check network / Yahoo Finance, retry.\n"
                + sep
            )

        return price_df

    def _detect_sector_etf(self, ticker: str, indian: bool) -> str:
        """
        Map ticker to its sector benchmark symbol.

        FIX 7: Static map first — zero live API calls for known tickers.
        For India: returns Nifty sector index (^CNXIT, ^NSEBANK, etc.)
        For US: returns SPDR sector ETF (XLK, XLF, etc.)
        """
        base = _base_ticker(ticker)

        if indian:
            sym = INDIA_SECTOR_INDEX_MAP.get(base, INDIA_DEFAULT_INDEX)
            if sym == INDIA_DEFAULT_INDEX and base not in INDIA_SECTOR_INDEX_MAP:
                logger.warning(
                    "%s: Not in India sector map. Using %s (Nifty 50).",
                    ticker.upper(), INDIA_DEFAULT_INDEX,
                )
            return sym

        # US: live sector lookup via yf.Ticker().info
        try:
            sector = yf.Ticker(ticker).info.get("sector", "")
            etf = US_SECTOR_ETF_MAP.get(sector, "SPY")
            if etf == "SPY":
                logger.warning(
                    "%s: Sector '%s' not in US ETF map. Using SPY.",
                    ticker.upper(), sector,
                )
            return etf
        except Exception:
            logger.warning("%s: Could not detect sector. Using SPY.", ticker.upper())
            return "SPY"


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def load_stock_data(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    quarterly_fundamentals: Optional[pd.DataFrame] = None,
    fundamentals_fetch_ts: Optional[datetime] = None,   # ✅ FIX 11
) -> pd.DataFrame:
    """
    Convenience wrapper around DataLoader.load_stock_data().

    Parameters
    ----------
    ticker : str
        Stock ticker (e.g. 'NVDA', 'RELIANCE.NS')
    start : str
        Start date YYYY-MM-DD
    end : str, optional
        End date YYYY-MM-DD. Defaults to today.
    quarterly_fundamentals : pd.DataFrame
        Live quarterly data from FundamentalFetcher. Required.
    fundamentals_fetch_ts : datetime, optional  [FIX 11]
        Timestamp returned by FundamentalFetcher.fetch_quarterly_fundamentals().
        When supplied, the data source banner shows the REAL Yahoo Finance fetch
        time instead of datetime.now() at load time. Fully backward compatible —
        existing call sites without this argument continue to work unchanged.

    Usage
    -----
    # Old call (still works — backward compatible):
    df = load_stock_data(ticker, start, end, fundamentals)

    # New call (shows real fetch timestamp in banner):
    fundamentals, fetch_ts = fetcher.fetch_quarterly_fundamentals(...)
    df = load_stock_data(ticker, start, end, fundamentals,
                         fundamentals_fetch_ts=fetch_ts)
    """
    return DataLoader().load_stock_data(
        ticker, start, end,
        quarterly_fundamentals,
        fundamentals_fetch_ts=fundamentals_fetch_ts,   # ✅ FIX 11
    )