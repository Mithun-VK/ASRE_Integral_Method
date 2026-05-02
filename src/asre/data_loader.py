"""
ASRE Data Loader - V3 (FIXED)
Auto-calculates derived metrics (PE, ROE, D/E) from raw quarterly data.

✅ FIX 1: Static fundamentals fallback REMOVED — hard abort if no live data
✅ FIX 2: Minimum quarters gate (4 quarters minimum) before F-Score computation
✅ FIX 3: SPY benchmark failure now aborts scan instead of silently using 0.0
✅ FIX 4: Data source banner logged on every successful run

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


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

VIX_TICKER = "^VIX"
US_BENCHMARK = "SPY"
MIN_QUARTERS = 4          # Minimum live quarters required for F-Score
SPY_MAX_RETRIES = 3       # Hard retry limit before aborting on SPY failure

SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Communication Services": "XLC",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}


# ─────────────────────────────────────────────────────────────
# Yahoo Loader
# ─────────────────────────────────────────────────────────────

class YahooFinanceLoader:

    def download(
        self,
        ticker: str,
        start: str,
        end: str,
        retries: int = 3,
    ) -> pd.DataFrame:
        """Download OHLCV data with robust error handling."""
        for attempt in range(retries):
            try:
                logger.info(
                    f"Downloading {ticker} from Yahoo Finance "
                    f"(attempt {attempt + 1}/{retries})..."
                )

                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    group_by="column",
                )

                if df is None or df.empty:
                    if attempt < retries - 1:
                        time.sleep(2)
                        continue
                    raise ValueError(f"No data returned for {ticker}")

                if "Date" not in df.columns and df.index.name in ["Date", None]:
                    df = df.reset_index()

                # Flatten MultiIndex columns
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

                # Strip ticker suffix from column names
                suffix = f"_{ticker.lower()}"
                df.columns = [
                    c.replace(suffix, "") if c.endswith(suffix) else c
                    for c in df.columns
                ]

                rename_map = {
                    "adj_close": "adj_close",
                    "adjclose": "adj_close",
                    "adjusted_close": "adj_close",
                    "adj close": "adj_close",
                    "date": "date",
                    "index": "date",
                }
                df = df.rename(columns=rename_map)

                if "date" not in df.columns:
                    date_cols = [
                        c for c in df.columns
                        if "date" in c.lower() or c == "index"
                    ]
                    if date_cols:
                        df = df.rename(columns={date_cols[0]: "date"})
                    else:
                        raise ValueError(
                            f"{ticker}: 'date' column not found after "
                            f"all rename attempts"
                        )

                logger.info(
                    f"✅ Successfully downloaded {len(df)} rows for {ticker}"
                )
                return df

            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    logger.error(f"Failed to download {ticker}: {e}")
                    raise

    def load_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Load OHLCV data with adj_close guarantee."""
        df = self.download(ticker, start, end)
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]
        df["ticker"] = ticker.upper()
        return df

    # ── REMOVED: load_fundamentals_static() ───────────────────
    # Static fundamentals produced inconsistent ratings for the
    # same stock on the same day (e.g., NVDA Tier S vs Tier C).
    # This method has been permanently removed. All fundamental
    # data must come through FundamentalFetcher (live quarterly).
    # If you need to debug a single stock's info snapshot, use
    # yf.Ticker(ticker).info directly in a notebook — never in
    # production ASRE scoring.
    # ──────────────────────────────────────────────────────────


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
    ) -> pd.DataFrame:
        """
        Load complete stock dataset.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        start_date : str
            Start date in YYYY-MM-DD format.
        end_date : str, optional
            End date. Defaults to today.
        quarterly_fundamentals : pd.DataFrame, optional
            Live quarterly fundamentals from FundamentalFetcher.
            THIS IS NOT OPTIONAL IN PRACTICE — passing None will
            raise a ValueError and abort the scan. The parameter
            is kept Optional only for type-signature compatibility.

        Raises
        ------
        ValueError
            If quarterly_fundamentals is None (static fallback removed).
            If fewer than MIN_QUARTERS of data are available.
        RuntimeError
            If SPY benchmark cannot be downloaded after SPY_MAX_RETRIES.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        extended_start = (
            pd.to_datetime(start_date) - timedelta(days=400)
        ).strftime("%Y-%m-%d")

        # ── Step 1: Load Price Data ────────────────────────────
        price_df = self.yf.load_ohlcv(ticker, extended_start, end_date)

        # ── Step 2: Merge Live Fundamentals ───────────────────
        if quarterly_fundamentals is not None:
            price_df = self._merge_live_fundamentals(
                ticker, price_df, quarterly_fundamentals
            )
        else:
            # ✅ FIX 1: Hard abort — no silent static fallback
            raise ValueError(
                f"\n"
                f"{'━' * 55}\n"
                f"  [ASRE CRITICAL] MISSING LIVE FUNDAMENTALS\n"
                f"  Ticker      : {ticker.upper()}\n"
                f"  Problem     : quarterly_fundamentals is None.\n"
                f"  Root cause  : The caller did not run FundamentalFetcher\n"
                f"                before calling DataLoader.\n"
                f"  Action      : Ensure FundamentalFetcher.fetch() is called\n"
                f"                and its DataFrame is passed here.\n"
                f"  Note        : Static fallback has been permanently\n"
                f"                removed. ASRE requires live data.\n"
                f"{'━' * 55}"
            )

        # ── Step 3: SPY Benchmark ──────────────────────────────
        # ✅ FIX 3: Hard abort after SPY_MAX_RETRIES failures
        price_df = self._load_spy_benchmark(
            ticker, price_df, extended_start, end_date
        )

        # ── Step 4: VIX ───────────────────────────────────────
        try:
            vix = self.yf.download(
                VIX_TICKER, extended_start, end_date, retries=1
            )
            price_df = (
                price_df
                .merge(
                    vix[["date", "close"]].rename(columns={"close": "vix"}),
                    on="date",
                    how="left",
                )
                .ffill()
            )
        except Exception as e:
            logger.warning(
                f"VIX download failed for {ticker}: {e}. "
                f"Defaulting to 15.0 — volatility context unavailable."
            )
            price_df["vix"] = 15.0

        # ── Step 5: Sector ETF ────────────────────────────────
        etf = self._detect_sector_etf(ticker)
        try:
            sect = self.yf.download(etf, extended_start, end_date, retries=1)
            sect["sector_return"] = sect["close"].pct_change()
            price_df = price_df.merge(
                sect[["date", "sector_return"]], on="date", how="left"
            ).ffill()
        except Exception as e:
            logger.warning(
                f"Sector ETF ({etf}) download failed for {ticker}: {e}. "
                f"Falling back to benchmark returns for sector beta."
            )
            price_df["sector_return"] = price_df["benchmark_return"]

        # ── Step 6: Sector Beta ───────────────────────────────
        price_df["stock_return"] = price_df["close"].pct_change()
        cov = price_df["stock_return"].rolling(60).cov(
            price_df["sector_return"]
        )
        var = price_df["sector_return"].rolling(60).var()
        price_df["sector_beta"] = (cov / var).clip(-3, 3).fillna(1.0)

        # ── Step 7: Trim to requested date range ──────────────
        price_df = price_df[
            price_df["date"] >= start_date
        ].reset_index(drop=True)

        return price_df

    # ─────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────

    def _merge_live_fundamentals(
        self,
        ticker: str,
        price_df: pd.DataFrame,
        quarterly_fundamentals: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate, compute derived metrics, and merge live quarterly
        fundamentals into the price DataFrame.

        Raises
        ------
        ValueError
            If fewer than MIN_QUARTERS rows are present.
        """
        qf = quarterly_fundamentals.copy()
        qf["date"] = pd.to_datetime(qf["date"])
        qf = qf.sort_values("date").reset_index(drop=True)

        # ✅ FIX 2: Minimum quarters gate
        if len(qf) < MIN_QUARTERS:
            raise ValueError(
                f"\n"
                f"{'━' * 55}\n"
                f"  [ASRE CRITICAL] INSUFFICIENT FUNDAMENTAL DATA\n"
                f"  Ticker      : {ticker.upper()}\n"
                f"  Found       : {len(qf)} quarter(s)\n"
                f"  Required    : {MIN_QUARTERS} quarters minimum\n"
                f"  Impact      : F-Score would be unreliable with\n"
                f"                fewer than {MIN_QUARTERS} quarters.\n"
                f"  Action      : Check FundamentalFetcher cache and\n"
                f"                Yahoo Finance availability for this\n"
                f"                ticker before retrying.\n"
                f"{'━' * 55}"
            )

        # ✅ FIX 4: Data source banner on every successful run
        fetch_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M IST")
        date_range = (
            f"{qf['date'].min().strftime('%Y-%m-%d')} → "
            f"{qf['date'].max().strftime('%Y-%m-%d')}"
        )
        logger.info(
            f"\n"
            f"{'━' * 55}\n"
            f"  DATA SOURCE : Live (Yahoo Finance Quarterly)\n"
            f"  TICKER      : {ticker.upper()}\n"
            f"  FETCHED AT  : {fetch_timestamp}\n"
            f"  QUARTERS    : {len(qf)}\n"
            f"  DATE RANGE  : {date_range}\n"
            f"{'━' * 55}"
        )

        # ── Derived Metric Calculations ────────────────────────

        # ROE — Annualized: (Net Income × 4) / Shareholders Equity
        if "net_income" in qf.columns and "shareholders_equity" in qf.columns:
            qf["roe"] = (
                qf["net_income"] * 4
                / qf["shareholders_equity"].replace(0, np.nan)
            ) * 100
            qf["roe"] = qf["roe"].fillna(15.0)

        # Debt-to-Equity
        if "total_debt" in qf.columns and "shareholders_equity" in qf.columns:
            qf["de"] = qf["total_debt"] / qf["shareholders_equity"].replace(
                0, np.nan
            )
            qf["de"] = qf["de"].fillna(0.5)

        # Revenue Growth YoY — requires at least 5 quarters for pct_change(4)
        if "revenue" in qf.columns:
            if len(qf) >= 5:
                qf["revenue_growth_yoy"] = qf["revenue"].pct_change(4) * 100
            else:
                # Not enough quarters for true YoY — use QoQ as proxy
                # and log a warning so the analyst knows
                logger.warning(
                    f"{ticker.upper()}: Only {len(qf)} quarters available. "
                    f"Using QoQ revenue growth as YoY proxy. "
                    f"F-Score revenue growth signal is approximate."
                )
                qf["revenue_growth_yoy"] = qf["revenue"].pct_change(1) * 100
            qf["revenue_growth_yoy"] = qf["revenue_growth_yoy"].fillna(10.0)

        # Profit Margin
        if "net_income" in qf.columns and "revenue" in qf.columns:
            qf["profit_margin"] = (
                qf["net_income"] / qf["revenue"].replace(0, np.nan)
            ) * 100

        # ── Merge into Price DataFrame ─────────────────────────
        price_df["date"] = pd.to_datetime(price_df["date"])
        price_df = pd.merge_asof(
            price_df.sort_values("date"),
            qf.sort_values("date"),
            on="date",
            direction="backward",
            suffixes=("", "_quarterly"),
        )

        # Forward fill fundamental columns
        cols_to_fill = [
            "roe", "de", "revenue_growth_yoy",
            "eps", "revenue", "profit_margin",
        ]
        for col in cols_to_fill:
            if col in price_df.columns:
                price_df[col] = price_df[col].ffill().bfill()

        # PE Ratio — requires price data, computed post-merge
        if "eps" in price_df.columns:
            annual_eps = (price_df["eps"] * 4).replace(0, 0.01)
            price_df["pe"] = price_df["close"] / annual_eps
            # Negative earnings get a penalty PE of 100
            price_df.loc[price_df["pe"] < 0, "pe"] = 100.0

        logger.info(
            f"✅ Calculated derived metrics: PE, ROE, D/E, Growth "
            f"for {ticker.upper()}"
        )

        return price_df

    def _load_spy_benchmark(
        self,
        ticker: str,
        price_df: pd.DataFrame,
        extended_start: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Download SPY benchmark returns and merge into price_df.

        ✅ FIX 3: Retries with backoff. Hard abort after
        SPY_MAX_RETRIES failures — T-Score requires this data.

        Raises
        ------
        RuntimeError
            If SPY cannot be downloaded after SPY_MAX_RETRIES attempts.
        """
        spy_loaded = False

        for attempt in range(SPY_MAX_RETRIES):
            try:
                spy = self.yf.download(
                    US_BENCHMARK,
                    extended_start,
                    end_date,
                    retries=1,
                )
                if spy is None or spy.empty:
                    raise ValueError("Empty DataFrame returned for SPY")

                spy = spy.rename(columns={"close": "benchmark_close"})
                spy["benchmark_return"] = spy["benchmark_close"].pct_change()
                price_df = price_df.merge(
                    spy[["date", "benchmark_return"]],
                    on="date",
                    how="left",
                ).ffill()

                spy_loaded = True
                break

            except Exception as e:
                wait = 2 * (attempt + 1)
                logger.warning(
                    f"SPY download attempt {attempt + 1}/{SPY_MAX_RETRIES} "
                    f"failed for {ticker.upper()}: {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        if not spy_loaded:
            raise RuntimeError(
                f"\n"
                f"{'━' * 55}\n"
                f"  [ASRE CRITICAL] SPY BENCHMARK UNAVAILABLE\n"
                f"  Ticker      : {ticker.upper()}\n"
                f"  Attempts    : {SPY_MAX_RETRIES}\n"
                f"  Impact      : T-Score and relative momentum\n"
                f"                calculations require SPY data.\n"
                f"                Continuing would produce invalid\n"
                f"                T-Scores for this stock.\n"
                f"  Action      : Check network connectivity and\n"
                f"                Yahoo Finance status, then retry.\n"
                f"{'━' * 55}"
            )

        return price_df

    def _detect_sector_etf(self, ticker: str) -> str:
        """Map ticker to its sector ETF for beta calculation."""
        try:
            sector = yf.Ticker(ticker).info.get("sector", "")
            etf = SECTOR_ETF_MAP.get(sector, "SPY")
            if etf == "SPY":
                logger.warning(
                    f"{ticker.upper()}: Sector '{sector}' not in ETF map. "
                    f"Using SPY as sector proxy."
                )
            return etf
        except Exception:
            logger.warning(
                f"{ticker.upper()}: Could not detect sector. "
                f"Using SPY as sector proxy."
            )
            return "SPY"


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def load_stock_data(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    quarterly_fundamentals: Optional[pd.DataFrame] = None,
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
        Live quarterly data from FundamentalFetcher.
        Required — None will raise ValueError.
    """
    return DataLoader().load_stock_data(
        ticker, start, end, quarterly_fundamentals
    )
