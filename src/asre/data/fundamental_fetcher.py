"""
Fundamental Data Fetcher - Yahoo Finance Integration (PRODUCTION v2.2)

Changes vs v2.1:
  HIST-1 : fetch_quarterly_fundamentals() now fetches MAX available history
           from Yahoo Finance (ignores start_date/end_date during live fetch)
           — all quarters are cached, filter applied on read.
  HIST-2 : yfinance fetches all 4 statement sources without date restriction:
           quarterly_income_stmt, quarterly_balance_sheet, quarterly_cashflow
           — Yahoo returns up to ~20 quarters automatically.
  HIST-3 : Cache stores FULL history; date filter applied at return time only.
  HIST-4 : Cache schema bumped to v2.2 to invalidate old 6-quarter caches.
  HIST-5 : _validate_no_lookahead_gap() warns when requested start_date
           predates available fundamental history.

Author: ASRE Project
Date:   April 2026
Version: 2.2.0 (Production)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)

SEP            = "\u2501" * 55
CACHE_SCHEMA_V = "2.2"   # Bumped: forces re-fetch of old 6-quarter caches

# ── ROE extraction constants ────────────────────────────────
_ROE_EQUITY_KEYS = [
    "Stockholders Equity",
    "StockholdersEquity",
    "Total Equity Gross Minority Interest",
    "Common Stock Equity",
    "CommonStockEquity",
    "Shareholders Equity",
    "ShareholdersEquity",
]
_ROE_NI_KEYS = [
    "Net Income",
    "NetIncome",
    "Net Income Common Stockholders",
    "NetIncomeCommonStockholders",
]
_ROE_MIN_PCT = -500.0
_ROE_MAX_PCT =  500.0


class FundamentalFetcherError(Exception):
    pass


class FundamentalFetcher:
    """
    Fetches quarterly fundamental data with earnings announcement dates.

    v2.2 key change: fetches FULL available history from Yahoo Finance
    (up to ~20 quarters / 5 years) regardless of start_date.
    The full history is cached; date filtering happens only at return time.
    This ensures the backtest engine always has historical fundamentals
    available for walk-forward periods.
    """

    def __init__(
        self,
        cache_dir: str = "data/cache/fundamentals",
        cache_ttl_days: int = 7,
    ):
        self.cache_dir      = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_days = cache_ttl_days
        self._cache_ttl_hours = cache_ttl_days * 24

        logger.info("✅ FundamentalFetcher v2.2 initialized (Yahoo Finance)")
        logger.info("   Cache directory : %s", self.cache_dir)
        logger.info("   Cache TTL       : %d days", cache_ttl_days)
        logger.info("   Schema version  : %s", CACHE_SCHEMA_V)

    # ──────────────────────────────────────────────────────────
    # Cache helpers
    # ──────────────────────────────────────────────────────────

    def _get_cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}_yfinance_fundamentals.parquet"

    def _get_cache_meta_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}_yfinance_fundamentals.meta"

    def _read_cache_meta(self, ticker: str) -> Tuple[Optional[datetime], Optional[str]]:
        meta_path  = self._get_cache_meta_path(ticker)
        cache_path = self._get_cache_path(ticker)

        if meta_path.exists():
            try:
                parts = meta_path.read_text().strip().split("|")
                ts    = datetime.fromisoformat(parts[0].strip())
                sv    = parts[1].strip() if len(parts) > 1 else None
                return ts, sv
            except Exception:
                pass

        if cache_path.exists():
            return datetime.fromtimestamp(cache_path.stat().st_mtime), None

        return None, None

    def _write_cache_meta(self, ticker: str, ts: datetime):
        try:
            self._get_cache_meta_path(ticker).write_text(
                f"{ts.isoformat()}|{CACHE_SCHEMA_V}"
            )
        except Exception as exc:
            logger.warning("Could not write cache meta for %s: %s", ticker, exc)

    def _check_staleness(self, ticker: str, fetch_timestamp: datetime):
        cache_age = datetime.now() - fetch_timestamp
        hours_old = cache_age.total_seconds() / 3600
        fetch_str = fetch_timestamp.strftime("%Y-%m-%d %H:%M IST")

        if hours_old > self._cache_ttl_hours:
            lines = [
                SEP,
                "  [ASRE CRITICAL] STALE DATA - CACHE TTL EXCEEDED",
                f"  Ticker      : {ticker.upper()}",
                f"  Cache age   : {hours_old/24:.1f} days",
                f"  TTL         : {self.cache_ttl_days} days",
                f"  Fetched at  : {fetch_str}",
                "  Action      : Delete cache files and retry.",
                f"  Command     : del data\\cache\\fundamentals\\{ticker}_yfinance_fundamentals.*",
                SEP,
            ]
            raise RuntimeError("\n".join(lines))

        if hours_old > 24:
            logger.warning(
                "STALE CACHE: %s fundamentals are %.0fh old "
                "(fetched %s). Live re-fetch recommended before scoring.",
                ticker.upper(), hours_old, fetch_str,
            )

    def _is_cache_valid(self, cache_path: Path) -> bool:
        return cache_path.exists()

    def _load_cache(self, ticker: str) -> Optional[Tuple[pd.DataFrame, datetime]]:
        """
        Load full-history DataFrame from cache.
        Returns (DataFrame, fetch_timestamp) or None.
        Auto-invalidates on schema version mismatch.
        """
        cache_path = self._get_cache_path(ticker)
        if not self._is_cache_valid(cache_path):
            return None

        try:
            fetch_ts, schema_v = self._read_cache_meta(ticker)
            if fetch_ts is None:
                fetch_ts = datetime.fromtimestamp(cache_path.stat().st_mtime)

            if schema_v != CACHE_SCHEMA_V:
                logger.warning(
                    "Cache schema mismatch for %s (cached=%s current=%s) — "
                    "forcing re-fetch.",
                    ticker, schema_v, CACHE_SCHEMA_V,
                )
                return None

            df = pd.read_parquet(cache_path)

            # Physical column guards
            for required_col in ["roe", "date", "announced_date"]:
                if required_col not in df.columns:
                    logger.warning(
                        "Cache for %s missing '%s' column — forcing re-fetch.",
                        ticker, required_col,
                    )
                    return None

            logger.info("✅ Loaded %s from cache (%d quarters total, schema %s)",
                        ticker, len(df), schema_v)
            return df, fetch_ts

        except Exception as exc:
            logger.warning("Cache read error for %s: %s", ticker, exc)
            return None

    def _save_cache(self, ticker: str, df: pd.DataFrame, fetch_ts: datetime):
        try:
            df.to_parquet(self._get_cache_path(ticker), index=False)
            self._write_cache_meta(ticker, fetch_ts)
            logger.info("✅ Cached %s full history (%d quarters, schema %s)",
                        ticker, len(df), CACHE_SCHEMA_V)
        except Exception as exc:
            logger.warning("Cache write error for %s: %s", ticker, exc)

    # ──────────────────────────────────────────────────────────
    # ROE Extraction (3-layer, Indian .NS hardened)
    # ──────────────────────────────────────────────────────────

    def fetch_roe_layered(
        self,
        ticker: str,
        stock: Optional[yf.Ticker] = None,
    ) -> Tuple[Optional[float], str]:
        """
        3-layer ROE extraction strategy hardened for Indian (.NS) tickers.
        Unchanged from v2.1.
        """
        if stock is None:
            stock = yf.Ticker(ticker)

        # Layer 1: yf.info pre-computed TTM ROE
        try:
            info    = stock.info
            roe_raw = info.get("returnOnEquity")
            if roe_raw is not None and pd.notna(roe_raw) and float(roe_raw) != 0.0:
                roe_pct = round(float(roe_raw) * 100, 2)
                if _ROE_MIN_PCT <= roe_pct <= _ROE_MAX_PCT:
                    logger.info("   [ROE-L1] %s: %.2f%% (source: yf.info TTM)", ticker, roe_pct)
                    return roe_pct, "L1_info"
                else:
                    logger.warning(
                        "[ROE-L1] %s: yf.info ROE=%.2f%% outside sanity bounds — skipping.",
                        ticker, roe_pct,
                    )
        except Exception as exc:
            logger.debug("[ROE-L1] %s: yf.info failed (%s)", ticker, exc)

        # Layer 2: Quarterly statements
        try:
            roe_pct, source = self._compute_roe_from_statements(
                ticker=ticker,
                income=stock.quarterly_income_stmt.T,
                balance=stock.quarterly_balance_sheet.T,
                label="L2_quarterly",
            )
            if roe_pct is not None:
                return roe_pct, source
        except Exception as exc:
            logger.debug("[ROE-L2] %s: Quarterly statements failed (%s)", ticker, exc)

        # Layer 3: Annual statements
        try:
            roe_pct, source = self._compute_roe_from_statements(
                ticker=ticker,
                income=stock.income_stmt.T,
                balance=stock.balance_sheet.T,
                label="L3_annual",
            )
            if roe_pct is not None:
                return roe_pct, source
        except Exception as exc:
            logger.debug("[ROE-L3] %s: Annual statements failed (%s)", ticker, exc)

        logger.warning("[ROE] %s: All 3 layers failed — ROE will be NaN.", ticker)
        return None, "FAILED"

    def _compute_roe_from_statements(
        self,
        ticker: str,
        income: pd.DataFrame,
        balance: pd.DataFrame,
        label: str,
    ) -> Tuple[Optional[float], str]:
        if income.empty or balance.empty:
            return None, label

        ni_col = next((c for c in _ROE_NI_KEYS if c in income.columns), None)
        eq_col = next((c for c in _ROE_EQUITY_KEYS if c in balance.columns), None)

        if ni_col is None or eq_col is None:
            return None, label

        ni_series = income[ni_col].dropna()
        ni_series = ni_series[ni_series != 0.0]
        eq_series = balance[eq_col].dropna()
        eq_series = eq_series[eq_series != 0.0]

        if ni_series.empty or eq_series.empty:
            return None, label

        ttm_quarters = ni_series.iloc[:4]
        if len(ttm_quarters) < 2:
            return None, label

        net_income = float(ttm_quarters.sum())
        avg_equity = (
            (float(eq_series.iloc[0]) + float(eq_series.iloc[1])) / 2
            if len(eq_series) >= 2
            else float(eq_series.iloc[0])
        )

        if avg_equity == 0.0:
            return None, label

        roe_pct = round((net_income / avg_equity) * 100, 2)

        if not (_ROE_MIN_PCT <= roe_pct <= _ROE_MAX_PCT):
            return None, label

        logger.info(
            "   [ROE-%s] %s: %.2f%% (NI=%.2fM, avg_eq=%.2fM)",
            label, ticker, roe_pct, net_income / 1e6, avg_equity / 1e6,
        )
        return roe_pct, label

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def fetch_quarterly_fundamentals(
        self,
        ticker:        str,
        start_date:    str,
        end_date:      str,
        use_cache:     bool = True,
        force_refresh: bool = False,
    ) -> Tuple[pd.DataFrame, datetime]:
        """
        Fetch ALL available quarterly fundamental history from Yahoo Finance.

        HIST-1/2/3: Full history is fetched and cached; start_date/end_date
        filter is applied ONLY at return time. This means a 2020-01-01
        start_date will correctly receive all quarters Yahoo has available
        (typically 16–20 quarters / 4–5 years for NSE tickers).

        Returns
        -------
        (DataFrame, fetch_timestamp)
            DataFrame filtered to [start_date, end_date].
            Full history is cached for future calls.
        """
        logger.info("\n📊 Fetching fundamentals for %s", ticker)
        logger.info("   Requested: %s to %s", start_date, end_date)
        logger.info("   Source: Yahoo Finance (FREE!)")

        # ── Try cache (stores full history) ───────────────────
        if use_cache and not force_refresh:
            result = self._load_cache(ticker)
            if result is not None:
                full_df, fetch_ts = result
                self._check_staleness(ticker, fetch_ts)
                full_df["date"] = pd.to_datetime(full_df["date"])

                # HIST-3: filter full cached history to requested window
                filtered = self._filter_by_date(full_df, start_date, end_date)

                logger.info(
                    "✅ Cache hit: %d total quarters, %d in [%s → %s]",
                    len(full_df), len(filtered), start_date, end_date,
                )
                self._warn_if_history_gap(ticker, full_df, start_date)
                return filtered, fetch_ts

        # ── Live fetch: NO date restriction — get everything ──
        logger.info("   Live fetch: requesting MAX history from Yahoo Finance...")
        fetch_ts = datetime.now()

        try:
            stock = yf.Ticker(ticker)

            logger.info("   Loading quarterly income statement (all available)...")
            income = stock.quarterly_income_stmt.T    # up to ~20 quarters

            logger.info("   Loading quarterly balance sheet (all available)...")
            balance = stock.quarterly_balance_sheet.T

            logger.info("   Loading quarterly cash flow (all available)...")
            cashflow = stock.quarterly_cashflow.T

            logger.info("   Loading earnings dates...")
            try:
                earnings_dates = stock.earnings_dates
            except Exception:
                earnings_dates = None
                logger.info("   Will estimate announcement dates (+40 days)")

            logger.info("   Extracting ROE (3-layer strategy)...")
            roe_pct, roe_source = self.fetch_roe_layered(ticker, stock=stock)

            # Build FULL history DataFrame (no date filter yet)
            full_df = self._build_fundamentals_df(
                income, balance, cashflow, earnings_dates, roe_pct, roe_source
            )
            full_df = self._validate_and_clean(full_df, ticker)

            logger.info(
                "✅ Fetched %d total quarters for %s [%s → %s] (ROE=%.2f%% [%s])",
                len(full_df), ticker,
                full_df["date"].min().date() if not full_df.empty else "N/A",
                full_df["date"].max().date() if not full_df.empty else "N/A",
                roe_pct if roe_pct is not None else 0.0,
                roe_source,
            )

            # Cache the FULL history
            if use_cache:
                self._save_cache(ticker, full_df, fetch_ts)

            # Warn if requested start_date predates available history
            self._warn_if_history_gap(ticker, full_df, start_date)

            # Filter to requested window
            filtered = self._filter_by_date(full_df, start_date, end_date)

            logger.info(
                "   Returning %d quarters in [%s → %s]",
                len(filtered), start_date, end_date,
            )

            if filtered.empty:
                logger.warning(
                    "⚠️  No quarters in [%s → %s]. "
                    "Available history: %s to %s. "
                    "The backtest will use all available quarters for scoring.",
                    start_date, end_date,
                    full_df["date"].min().date() if not full_df.empty else "N/A",
                    full_df["date"].max().date() if not full_df.empty else "N/A",
                )
                # Return full history so engine has something to work with
                return full_df, fetch_ts

            return filtered, fetch_ts

        except FundamentalFetcherError:
            raise
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", ticker, exc)
            import traceback
            traceback.print_exc()
            raise FundamentalFetcherError(f"Failed to fetch {ticker}: {exc}")

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _filter_by_date(
        self, df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Filter DataFrame to [start_date, end_date] inclusive."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        start_dt = pd.to_datetime(start_date)
        end_dt   = pd.to_datetime(end_date)
        mask = (df["date"] >= start_dt) & (df["date"] <= end_dt)
        return df[mask].reset_index(drop=True)

    def _warn_if_history_gap(
        self, ticker: str, full_df: pd.DataFrame, start_date: str
    ):
        """
        HIST-5: Warn if requested start_date predates available fundamental history.
        This is expected for most tickers — Yahoo Finance caps at ~5 years.
        The engine handles this gracefully by skipping periods with no PIT data.
        """
        if full_df.empty:
            return
        earliest = full_df["date"].min()
        requested_start = pd.to_datetime(start_date)
        gap_days = (earliest - requested_start).days

        if gap_days > 90:
            logger.info(
                "ℹ️  History gap for %s: requested start=%s, "
                "earliest available=%s (gap=%d days / %.1f years). "
                "Yahoo Finance max history is ~5 years. "
                "Backtest periods before %s will be skipped (expected).",
                ticker,
                requested_start.date(),
                earliest.date(),
                gap_days,
                gap_days / 365.25,
                (earliest + pd.Timedelta(days=40)).date(),
            )

    def _safe_extract_column(
        self,
        df: pd.DataFrame,
        column_names: List[str],
        default_value: float = 0.0,
    ) -> np.ndarray:
        for col_name in column_names:
            if col_name in df.columns:
                return df[col_name].values
        return np.full(len(df), default_value, dtype=float)

    def _build_fundamentals_df(
        self,
        income: pd.DataFrame,
        balance: pd.DataFrame,
        cashflow: pd.DataFrame,
        earnings_dates: Optional[pd.DataFrame],
        roe_pct: Optional[float],
        roe_source: str,
    ) -> pd.DataFrame:
        if income.empty:
            raise FundamentalFetcherError("No income statement data available")

        logger.info(
            "   Building fundamentals DataFrame (%d quarters)...", len(income)
        )

        df = pd.DataFrame({
            "date": pd.to_datetime(income.index),
            "revenue": self._safe_extract_column(
                income, ["Total Revenue", "TotalRevenue", "Operating Revenue"]
            ),
            "net_income": self._safe_extract_column(
                income, ["Net Income", "NetIncome", "Net Income Common Stockholders"]
            ),
            "eps": self._safe_extract_column(
                income, ["Diluted EPS", "DilutedEPS", "Basic EPS"]
            ),
            "gross_profit": self._safe_extract_column(
                income, ["Gross Profit", "GrossProfit"]
            ),
            "operating_income": self._safe_extract_column(
                income, ["Operating Income", "OperatingIncome"]
            ),
            "ebitda": self._safe_extract_column(
                income, ["EBITDA", "Normalized EBITDA"]
            ),
        })

        # Balance sheet — align by date index if lengths differ
        if not balance.empty:
            bal_aligned = balance.reindex(income.index)
            df["total_assets"]        = self._safe_extract_column(bal_aligned, ["Total Assets", "TotalAssets"])
            df["total_debt"]          = self._safe_extract_column(bal_aligned, ["Total Debt", "TotalDebt", "Long Term Debt"])
            df["shareholders_equity"] = self._safe_extract_column(bal_aligned, _ROE_EQUITY_KEYS)
        else:
            df["total_assets"]        = 0.0
            df["total_debt"]          = 0.0
            df["shareholders_equity"] = 0.0

        # Cash flow — align by date index
        if not cashflow.empty:
            cf_aligned = cashflow.reindex(income.index)
            df["free_cash_flow"]      = self._safe_extract_column(cf_aligned, ["Free Cash Flow", "FreeCashFlow"])
            df["operating_cash_flow"] = self._safe_extract_column(
                cf_aligned, ["Operating Cash Flow", "OperatingCashFlow",
                             "Total Cash From Operating Activities"]
            )
        else:
            df["free_cash_flow"]      = 0.0
            df["operating_cash_flow"] = 0.0

        df["roe"]        = roe_pct if roe_pct is not None else np.nan
        df["roe_source"] = roe_source

        df["announced_date"] = self._map_earnings_dates(df["date"], earnings_dates)
        df = df.sort_values("date", ascending=True).reset_index(drop=True)
        return df

    def _map_earnings_dates(
        self,
        quarter_dates: pd.Series,
        earnings_dates: Optional[pd.DataFrame],
    ) -> pd.Series:
        announced = []
        for qdate in quarter_dates:
            if earnings_dates is not None and not earnings_dates.empty:
                try:
                    qts    = pd.Timestamp(qdate)
                    future = earnings_dates[earnings_dates.index > qts]
                    if not future.empty:
                        ann       = future.index[0]
                        days_diff = (ann - qts).days
                        announced.append(
                            ann if 0 <= days_diff <= 90
                            else qts + pd.Timedelta(days=40)
                        )
                    else:
                        announced.append(pd.Timestamp(qdate) + pd.Timedelta(days=40))
                except Exception:
                    announced.append(pd.Timestamp(qdate) + pd.Timedelta(days=40))
            else:
                announced.append(pd.Timestamp(qdate) + pd.Timedelta(days=40))
        return pd.Series(announced, index=quarter_dates.index)

    def _validate_and_clean(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        logger.info("   Validating data...")

        required_cols = ["date", "announced_date", "revenue", "net_income", "eps", "roe"]
        missing_cols  = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise FundamentalFetcherError(f"Missing required columns: {missing_cols}")

        if df["date"].isnull().any():
            raise FundamentalFetcherError("Missing quarter end dates")

        df["date"]           = pd.to_datetime(df["date"])
        df["announced_date"] = pd.to_datetime(df["announced_date"])

        invalid = df["announced_date"] < df["date"]
        if invalid.any():
            logger.warning("Fixing %d invalid announcement dates", invalid.sum())
            df.loc[invalid, "announced_date"] = (
                df.loc[invalid, "date"] + pd.Timedelta(days=40)
            )

        initial_len = len(df)
        df = df.drop_duplicates(subset=["date"], keep="first")
        if len(df) < initial_len:
            logger.warning("Removed %d duplicate quarters", initial_len - len(df))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(["roe"])
        df[numeric_cols] = df[numeric_cols].fillna(0)

        roe_val    = df["roe"].iloc[0] if not df.empty else np.nan
        roe_src    = df["roe_source"].iloc[0] if "roe_source" in df.columns else "N/A"
        roe_status = (
            f"{roe_val:.2f}% [{roe_src}]"
            if pd.notna(roe_val)
            else f"UNAVAILABLE [{roe_src}]"
        )
        logger.info("✅ Validation complete: %d quarters, %d columns", len(df), len(df.columns))
        logger.info(
            "   Date range: %s → %s",
            df["date"].min().date(), df["date"].max().date(),
        )
        logger.info("   ROE: %s", roe_status)
        return df

    # ──────────────────────────────────────────────────────────
    # Convenience methods
    # ──────────────────────────────────────────────────────────

    def get_latest_quarter(self, ticker: str) -> pd.Series:
        today       = datetime.now().strftime("%Y-%m-%d")
        five_yrs_ago = (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d")
        df, _       = self.fetch_quarterly_fundamentals(ticker, five_yrs_ago, today)
        if df.empty:
            raise FundamentalFetcherError(f"No recent data for {ticker}")
        return df.iloc[-1]

    def get_roe(self, ticker: str) -> Optional[float]:
        try:
            roe_pct, source = self.fetch_roe_layered(ticker)
            logger.info("get_roe(%s) → %.2f%% [%s]",
                        ticker, roe_pct if roe_pct is not None else 0.0, source)
            return roe_pct
        except Exception as exc:
            logger.error("get_roe(%s) failed: %s", ticker, exc)
            return None

    def get_historical_series(
        self,
        ticker:     str,
        metric:     str,
        start_date: str,
        end_date:   str,
    ) -> pd.Series:
        df, _ = self.fetch_quarterly_fundamentals(ticker, start_date, end_date)
        if metric not in df.columns:
            raise FundamentalFetcherError(
                f"Metric '{metric}' not found. Available: {', '.join(df.columns)}"
            )
        return df.set_index("date")[metric]


# ──────────────────────────────────────────────────────────────────────────────
# Manual test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    TICKERS = ["INFY.NS", "TCS.NS", "RELIANCE.NS", "HDFCBANK.NS"]

    print("=" * 80)
    print("FUNDAMENTAL FETCHER v2.2 - HISTORY TEST")
    print("=" * 80)

    fetcher = FundamentalFetcher()

    for ticker in TICKERS:
        try:
            df, fetch_ts = fetcher.fetch_quarterly_fundamentals(
                ticker        = ticker,
                start_date    = "2020-01-01",
                end_date      = "2026-04-01",
                force_refresh = True,
            )
            print(f"\n{ticker}: {len(df)} quarters fetched")
            print(f"   Range: {df['date'].min().date()} → {df['date'].max().date()}")
            print(f"   Columns: {list(df.columns)}")
            print(df[['date', 'announced_date', 'revenue', 'net_income', 'roe']].to_string(index=False))
        except Exception as exc:
            print(f"\n{ticker}: ERROR — {exc}")

    print("\n✅ FundamentalFetcher v2.2 ready!")
    print("=" * 80)