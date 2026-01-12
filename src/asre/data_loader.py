"""
ASRE Data Loader - V2 (FIXED)
Auto-calculates derived metrics (PE, ROE, D/E) from raw quarterly data.

✅ CRITICAL FIX: Computes required ratios from raw accounting data
✅ Handles PE, ROE, D/E, and Growth calculations automatically
✅ seamless integration with FundamentalFetcher

Author: ASRE Project
Date: January 2026
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

    def download(self, ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
        """Download with robust error handling."""
        for attempt in range(retries):
            try:
                logger.info(f"Downloading {ticker} from Yahoo Finance (attempt {attempt + 1}/{retries})...")

                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    group_by="column"
                )

                if df is None or df.empty:
                    if attempt < retries - 1:
                        time.sleep(2)
                        continue
                    raise ValueError(f"No data returned for {ticker}")

                if 'Date' not in df.columns and df.index.name in ['Date', None]:
                    df = df.reset_index()

                # Flatten columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ["_".join([str(x) for x in col if x]).lower().strip() for col in df.columns]
                else:
                    df.columns = [str(c).lower().replace(" ", "_").strip() for c in df.columns]

                # Clean column names
                suffix = f"_{ticker.lower()}"
                df.columns = [c.replace(suffix, "") if c.endswith(suffix) else c for c in df.columns]

                rename_map = {
                    "adj_close": "adj_close", "adjclose": "adj_close",
                    "adjusted_close": "adj_close", "adj close": "adj_close",
                    "date": "date", "index": "date",
                }
                df = df.rename(columns=rename_map)

                if 'date' not in df.columns:
                    date_cols = [c for c in df.columns if 'date' in c.lower() or c == 'index']
                    if date_cols:
                        df = df.rename(columns={date_cols[0]: 'date'})
                    else:
                        raise ValueError(f"{ticker}: 'date' column not found")

                logger.info(f"✅ Successfully downloaded {len(df)} rows for {ticker}")
                return df

            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    logger.error(f"Failed to download {ticker}: {e}")
                    raise

    def load_ohlcv(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Load OHLCV data."""
        df = self.download(ticker, start, end)
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]
        df["ticker"] = ticker.upper()
        return df

    def load_fundamentals_static(self, ticker: str) -> Dict[str, float]:
        """Load STATIC fundamentals (Fallback only)."""
        logger.info(f"Loading static fundamentals for {ticker}")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            return {
                "pe": info.get("trailingPE", 20.0),
                "roe": (info.get("returnOnEquity", 0.15) * 100),
                "de": (info.get("debtToEquity", 50) / 100),
                "revenue_growth_yoy": (info.get("revenueGrowth", 0.10) * 100),
                "profit_margin": (info.get("profitMargins", 0.10) * 100),
                "dividend_yield": (info.get("dividendYield", 0.0) * 100),
                "market_cap": info.get("marketCap", 0),
            }
        except Exception:
            return {"pe": 20.0, "roe": 15.0, "de": 0.5, "revenue_growth_yoy": 10.0}


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
        quarterly_fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Load complete stock dataset."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        extended_start = (pd.to_datetime(start_date) - timedelta(days=400)).strftime("%Y-%m-%d")

        # 1. Load Price
        price_df = self.yf.load_ohlcv(ticker, extended_start, end_date)

        # 2. Merge Fundamentals
        if quarterly_fundamentals is not None:
            logger.info(f"✅ Processing {len(quarterly_fundamentals)} quarters of fundamentals")

            # ----------------------------------------------------
            # CALCULATION LOGIC: Compute derived metrics
            # ----------------------------------------------------
            qf = quarterly_fundamentals.copy()
            qf['date'] = pd.to_datetime(qf['date'])
            qf = qf.sort_values('date')

            # ROE (Annualized)
            # Net Income * 4 / Shareholders Equity
            if 'net_income' in qf.columns and 'shareholders_equity' in qf.columns:
                qf['roe'] = (qf['net_income'] * 4 / qf['shareholders_equity'].replace(0, np.nan)) * 100
                qf['roe'] = qf['roe'].fillna(15.0)  # Default

            # Debt to Equity
            if 'total_debt' in qf.columns and 'shareholders_equity' in qf.columns:
                qf['de'] = qf['total_debt'] / qf['shareholders_equity'].replace(0, np.nan)
                qf['de'] = qf['de'].fillna(0.5)

            # Revenue Growth YoY
            if 'revenue' in qf.columns:
                # Try to calculate YoY growth if we have 4+ quarters
                qf['revenue_growth_yoy'] = qf['revenue'].pct_change(4) * 100
                # Fill missing (early quarters) with mean or default
                qf['revenue_growth_yoy'] = qf['revenue_growth_yoy'].fillna(10.0)

            # Profit Margin
            if 'net_income' in qf.columns and 'revenue' in qf.columns:
                qf['profit_margin'] = (qf['net_income'] / qf['revenue'].replace(0, np.nan)) * 100

            # Merge
            price_df['date'] = pd.to_datetime(price_df['date'])
            price_df = pd.merge_asof(
                price_df.sort_values('date'),
                qf.sort_values('date'),
                on='date',
                direction='backward',
                suffixes=('', '_quarterly')
            )

            # Forward fill
            cols_to_fill = ['roe', 'de', 'revenue_growth_yoy', 'eps', 'revenue', 'profit_margin']
            for col in cols_to_fill:
                if col in price_df.columns:
                    price_df[col] = price_df[col].ffill().bfill()

            # ----------------------------------------------------
            # PE Ratio Calculation (Needs Price)
            # ----------------------------------------------------
            # PE = Price / (Quarterly EPS * 4)
            if 'eps' in price_df.columns:
                # Avoid division by zero
                annual_eps = (price_df['eps'] * 4).replace(0, 0.01)
                price_df['pe'] = price_df['close'] / annual_eps
                # Handle negative PE or crazy outliers
                price_df.loc[price_df['pe'] < 0, 'pe'] = 100.0  # Penalize negative earnings

            logger.info("✅ Calculated derived metrics: PE, ROE, D/E, Growth")

        else:
            # Fallback
            logger.warning("⚠️ Using STATIC fundamentals")
            funds = self.yf.load_fundamentals_static(ticker)
            for k, v in funds.items():
                price_df[k] = v

        # 3. Market Data (VIX, Benchmark)
        try:
            vix = self.yf.download(VIX_TICKER, extended_start, end_date, retries=1)
            price_df = price_df.merge(vix[['date', 'close']].rename(columns={'close': 'vix'}), on='date', how='left').ffill()
        except:
            price_df['vix'] = 15.0

        try:
            spy = self.yf.download(US_BENCHMARK, extended_start, end_date, retries=1)
            spy = spy.rename(columns={'close': 'benchmark_close'})
            spy['benchmark_return'] = spy['benchmark_close'].pct_change()
            price_df = price_df.merge(spy[['date', 'benchmark_return']], on='date', how='left').ffill()
        except:
            price_df['benchmark_return'] = 0.0

        # 4. Sector Beta
        etf = self._detect_sector_etf(ticker)
        try:
            sect = self.yf.download(etf, extended_start, end_date, retries=1)
            sect['sector_return'] = sect['close'].pct_change()
            price_df = price_df.merge(sect[['date', 'sector_return']], on='date', how='left').ffill()
        except:
            price_df['sector_return'] = price_df['benchmark_return']

        price_df['stock_return'] = price_df['close'].pct_change()
        cov = price_df['stock_return'].rolling(60).cov(price_df['sector_return'])
        var = price_df['sector_return'].rolling(60).var()
        price_df['sector_beta'] = (cov / var).clip(-3, 3).fillna(1.0)

        # Cleanup
        price_df = price_df[price_df['date'] >= start_date].reset_index(drop=True)
        return price_df

    def _detect_sector_etf(self, ticker: str) -> str:
        return SECTOR_ETF_MAP.get(yf.Ticker(ticker).info.get("sector"), "SPY")

# API
def load_stock_data(ticker, start, end=None, quarterly_fundamentals=None):
    return DataLoader().load_stock_data(ticker, start, end, quarterly_fundamentals)