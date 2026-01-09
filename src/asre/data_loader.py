"""
ASRE Data Loader - Real Yahoo Finance Integration
Production version for accurate rating calculation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Market Proxies
# ------------------------------------------------------------

VIX_TICKER = "^VIX"
US_BENCHMARK = "SPY"

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

# ------------------------------------------------------------
# Yahoo Loader
# ------------------------------------------------------------

class YahooFinanceLoader:

    def download(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        logger.info(f"Downloading {ticker} from Yahoo Finance...")

        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="column"
        )

        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        df = df.reset_index()

    # --------------------------------------------------------
    # ✅ Flatten MultiIndex columns safely
    # --------------------------------------------------------
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join([str(x) for x in col if x]).lower()
                for col in df.columns
            ]
        else:
            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

        # --------------------------------------------------------
        # ✅ Remove ticker suffix: close_aapl → close
        # --------------------------------------------------------
        suffix = f"_{ticker.lower()}"
        df.columns = [
            c.replace(suffix, "") if c.endswith(suffix) else c
            for c in df.columns
        ]

        # --------------------------------------------------------
        # ✅ Standardize column aliases
        # --------------------------------------------------------
        rename_map = {
            "adj_close": "adj_close",
            "adjclose": "adj_close",
            "adjusted_close": "adj_close",
        }

        df = df.rename(columns=rename_map)

        # --------------------------------------------------------
        # ✅ Validate required columns
        # --------------------------------------------------------
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"{ticker} missing columns: {sorted(missing)}")

        return df

    def load_ohlcv(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        df = self.download(ticker, start, end)

        required = ["date", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{ticker} missing columns: {missing}")

        df["adj_close"] = df.get("adj_close", df["close"])
        df["ticker"] = ticker.upper()

        logger.info(f"Loaded {len(df)} rows for {ticker}")
        return df

    def load_fundamentals(self, ticker: str) -> Dict[str, float]:
        logger.info(f"Loading fundamentals for {ticker}")
        stock = yf.Ticker(ticker)
        info = stock.fast_info or {}
        finfo = stock.info or {}

        fundamentals = {
            "pe": finfo.get("trailingPE") or finfo.get("forwardPE"),
            "roe": finfo.get("returnOnEquity"),
            "de": finfo.get("debtToEquity"),
            "pb": finfo.get("priceToBook"),
            "dividend_yield": finfo.get("dividendYield"),
            "market_cap": finfo.get("marketCap"),
            "shares_outstanding": finfo.get("sharesOutstanding"),
            "book_value": finfo.get("bookValue"),
            "profit_margin": finfo.get("profitMargins"),
            "revenue_growth": finfo.get("revenueGrowth"),
        }

        return fundamentals

# ------------------------------------------------------------
# Data Loader
# ------------------------------------------------------------

class DataLoader:

    def __init__(self):
        self.yf = YahooFinanceLoader()

    def load_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: Optional[str] = None,
        region: str = "US",
    ) -> pd.DataFrame:

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Extend window for indicators
        extended_start = (
            pd.to_datetime(start_date) - timedelta(days=400)
        ).strftime("%Y-%m-%d")

        logger.info(f"Loading data for {ticker}")

        price_df = self.yf.load_ohlcv(ticker, extended_start, end_date)

        # --------------------------------------------------
        # Fundamentals (static but forward filled)
        # --------------------------------------------------

        fundamentals = self.yf.load_fundamentals(ticker)

        for k, v in fundamentals.items():
            price_df[k] = v

        # --------------------------------------------------
        # Market Context
        # --------------------------------------------------

        vix_df = self.yf.download(VIX_TICKER, extended_start, end_date)[["date", "close"]]
        vix_df.rename(columns={"close": "vix"}, inplace=True)

        benchmark_df = self.yf.download(
            US_BENCHMARK, extended_start, end_date
        )[["date", "close"]]
        benchmark_df.rename(columns={"close": "benchmark_close"}, inplace=True)
        benchmark_df["benchmark_return"] = benchmark_df["benchmark_close"].pct_change()

        # Sector ETF detection
        sector_ticker = self._detect_sector_etf(ticker)
        sector_df = self.yf.download(sector_ticker, extended_start, end_date)[["date", "close"]]
        sector_df.rename(columns={"close": "sector_close"}, inplace=True)
        sector_df["sector_return"] = sector_df["sector_close"].pct_change()

        # --------------------------------------------------
        # Merge
        # --------------------------------------------------

        df = price_df.merge(vix_df, on="date", how="left")
        df = df.merge(sector_df[["date", "sector_return"]], on="date", how="left")
        df = df.merge(
            benchmark_df[["date", "benchmark_return"]],
            on="date",
            how="left",
        )

        # --------------------------------------------------
        # Sector Beta
        # --------------------------------------------------

        df["stock_return"] = df["close"].pct_change()

        rolling_cov = df["stock_return"].rolling(60).cov(df["sector_return"])
        rolling_var = df["sector_return"].rolling(60).var()
        df["sector_beta"] = (rolling_cov / rolling_var).clip(-3, 3)

        df.drop(columns=["stock_return"], inplace=True)

        # --------------------------------------------------
        # Cleanup
        # --------------------------------------------------

        df = df.ffill().bfill()
        df = df[df["date"] >= start_date].reset_index(drop=True)

        self._validate(df)

        logger.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _detect_sector_etf(self, ticker: str) -> str:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector")
            return SECTOR_ETF_MAP.get(sector, "SPY")
        except Exception:
            return "SPY"

    def _validate(self, df: pd.DataFrame):

        required = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_close",
            "pe",
            "roe",
            "de",
            "pb",
            "dividend_yield",
            "vix",
            "sector_return",
            "benchmark_return",
            "sector_beta",
        ]

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Dataset validation passed")


# ------------------------------------------------------------
# Convenience API
# ------------------------------------------------------------

def load_stock_data(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    loader = DataLoader()
    return loader.load_stock_data(ticker, start_date, end_date)


__all__ = ["DataLoader", "load_stock_data"]
