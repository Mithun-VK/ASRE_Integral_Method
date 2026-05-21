#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            ASRE DAILY PORTFOLIO MANAGER — Production Edition v2.0           ║
║                        Stateful Decision Engine                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

WORKFLOW:
---------
1. Load previous portfolio state (positions, ledger, score history)
2. Compute fresh ASRE scores for entire stock universe
3. Apply threshold-based decision rules:
   • BUY   : R_ASRE ≥ buy_threshold, quality tier acceptable, cash available
   • HOLD  : Currently held, R_ASRE ≥ hold_threshold
   • SELL  : Currently held, R_ASRE < sell_threshold OR adverse conditions
   • WATCH : R_ASRE ≥ watch_threshold, not actionable yet
   • SKIP  : Data unavailable or very low scores

4. Update portfolio ledger with today's actions
5. Save state and export action report

USAGE:
------
# First run (initialize portfolio)
ASRE_SKIP_REG=1 python asre_portfolio_daily.py --init --capital 100000

# Daily run (compute scores + generate actions)
ASRE_SKIP_REG=1 python asre_portfolio_daily.py

# Production with SEBI registration
ASRE_IA_REG_NO=INA000012345 python asre_portfolio_daily.py --mode ia

# Custom thresholds
python asre_portfolio_daily.py --buy-threshold 70 --hold-threshold 55 --sell-threshold 45

# Dry run (no state changes)
python asre_portfolio_daily.py --dry-run

OUTPUTS:
--------
• portfolio_state.json       — Current positions and metadata
• portfolio_ledger.csv        — All transactions (buys/sells)
• score_history.csv           — Time-series of all R_ASRE scores
• daily_actions_YYYYMMDD.txt  — Today's BUY/HOLD/SELL recommendations
• daily_report_YYYYMMDD.json  — Structured action report
• audit_log.jsonl             — Compliance audit trail (append-only)

Author: ASRE Project Team
License: Proprietary - SEBI Regulated Use Only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import math

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import traceback
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# Auto-install dependencies
import importlib.util
import subprocess


def _ensure_pkg(pkg: str, import_as: str | None = None):
    mod = import_as or pkg
    if importlib.util.find_spec(mod) is None:
        print(f"📦 Installing {pkg}...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)


for _p, _m in [
    ("yfinance", "yfinance"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("rich", "rich"),
]:
    _ensure_pkg(_p, _m)

import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP (Fixed for multiprocessing compatibility)
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("asre.portfolio")

# ══════════════════════════════════════════════════════════════════════════════
# ASRE ENGINE IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

try:
    from asre.composite import compute_complete_asre
    from asre.config import CompositeConfig, FundamentalsConfig, MomentumConfig, TechnicalConfig
    from asre.data_loader_indian import DataLoader
    from asre.data_quality import assess_fundamental_data_quality
    from asre.role_gate import RoleGate, RoleGateError

    ENGINE = "package"
except ImportError:
    try:
        _here = Path(__file__).parent
        sys.path.insert(0, str(_here))
        from asre.composite import compute_complete_asre
        from asre.config import CompositeConfig, FundamentalsConfig, MomentumConfig, TechnicalConfig
        from asre.data_loader_indian import DataLoader
        from asre.data_quality import assess_fundamental_data_quality
        from asre.role_gate import RoleGate, RoleGateError

        ENGINE = "relative"
    except ImportError as e:
        console.print(f"[bold red]✗ Cannot import ASRE modules: {e}[/bold red]")
        sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# STOCK UNIVERSE
# ══════════════════════════════════════════════════════════════════════════════

STOCK_UNIVERSE: List[Tuple[str, str]] = [
    ("Eternal Ltd.", "ETERNAL.NS"),
    ("Muthoot Finance Ltd.", "MUTHOOTFIN.NS"),
    ("Multi Commodity Exchange", "MCX.NS"),
    ("CG Power & Industrial", "CGPOWER.NS"),
    ("Waaree Energies Ltd.", "WAAREEENER.NS"),
    ("Shriram Finance Ltd.", "SHRIRAMFIN.NS"),
    ("PTC Industries Ltd.", "PTCIL.NS"),
    ("Bharat Electronics Ltd.", "BEL.NS"),
    ("Apar Industries Ltd.", "APARINDS.NS"),
    ("Billionbrains Garage", "UNIMECH.NS"),
    ("Premier Energies Ltd.", "PREMIERENE.NS"),
    ("Onesource Specialty", "ONESOURCE.NS"),
    ("Ather Energy Ltd.", "ATHER.NS"),
    ("Paytm", "PAYTM.NS"),
    ("Amber Enterprises", "AMBER.NS"),
    ("TVS Motor Company", "TVSMOTOR.NS"),
    ("Motherson International", "MOTHERSON.NS"),
    ("Gujarat Fluorochemicals", "FLUOROCHEM.NS"),
    ("Zen Technologies", "ZENTEC.NS"),
    ("Suzlon Energy", "SUZLON.NS"),
    ("Bajaj Finance", "BAJFINANCE.NS"),
    ("Prestige Estates", "PRESTIGE.NS"),
    ("Bharat Dynamics", "BDL.NS"),
    ("Jain Resource", "JAINRES.NS"),
    ("V2 Retail", "V2RETAIL.NS"),
    ("Ola Electric", "OLAELEC.NS"),
    ("GE Vernova T&D", "GEVERNOVA.NS"),
    ("Religare Enterprises", "RELIGARE.NS"),
    ("State Bank of India", "SBIN.NS"),
    ("Indusind Bank", "INDUSINDBK.NS"),
]

NO_DATA_TICKERS = {"ATHER.NS", "GEVERNOVA.NS", "JAINRES.NS", "ONESOURCE.NS"}

# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class StockScore:
    """Daily ASRE score for a single stock."""

    ticker: str
    name: str
    date: str
    r_asre: float = np.nan
    r_final: float = np.nan
    r_medallion: float = np.nan
    f_score: float = np.nan
    t_score: float = np.nan
    m_score: float = np.nan
    quality_tier: str = "N/A"
    stock_category: str = "N/A"
    dip_stage: str = "N/A"
    market_context: str = "N/A"
    signal: str = "N/A"
    current_price: float = np.nan
    data_quality_score: float = np.nan
    error: Optional[str] = None


@dataclass
class Position:
    """Active portfolio position."""

    ticker: str
    name: str
    quantity: int
    entry_price: float
    entry_date: str
    current_price: float = np.nan
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    days_held: int = 0
    last_r_asre: float = np.nan


@dataclass
class Transaction:
    """Buy/Sell transaction record."""

    date: str
    action: str  # BUY or SELL
    ticker: str
    name: str
    quantity: int
    price: float
    value: float
    reason: str
    r_asre: float = np.nan

    # --- Reporting snapshot fields ---
    entry_price: float = np.nan
    days_held: int = 0
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0

@dataclass
class PortfolioState:
    """Complete portfolio state."""

    capital: float
    cash: float
    invested: float
    positions: Dict[str, Position] = field(default_factory=dict)
    last_updated: str = ""
    total_value: float = 0.0
    total_pnl: float = 0.0
    # ── NAV tracking ──────────────────────────────────────────────────────────
    cumulative_realized_pnl: float = 0.0   # total P&L locked in from all sells
    nav: float = 100.0                      # current NAV per unit (₹/unit)
    nav_base_capital: float = 0.0           # initial corpus used as NAV baseline
    units_outstanding: float = 1000.0      # fixed denominator for NAV calculation


@dataclass
class DailyAction:
    """Recommended action for today."""

    ticker: str
    name: str
    action: str  # BUY / HOLD / SELL / WATCH / SKIP
    r_asre: float
    current_price: float
    reason: str
    priority: int = 0
    suggested_allocation: int = 0
    explanation: str = ""
    # Rotation audit trail (populated only for rotation-driven BUY/SELL pairs)
    rotation_sell_of: str = ""   # BUY action: ticker this buy replaced
    rotation_buy_of: str = ""    # SELL action: ticker this sell made room for
    capital_haircut_used: float = 0.0  # SELL action: haircut fraction applied


# ══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL FETCHER (Simplified for daily use)
# ══════════════════════════════════════════════════════════════════════════════


class FundamentalFetcher:
    """Lightweight fundamental fetcher with caching."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=12)  # Refresh twice daily

    def fetch(self, ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[datetime]]:
        """Fetch quarterly fundamentals with 12-hour cache."""
        cache_path = self.cache_dir / f"{ticker.replace('.', '_')}_fund.parquet"

        # Check cache
        if cache_path.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if age < self.cache_ttl:
                try:
                    df = pd.read_parquet(cache_path)
                    ts = datetime.fromtimestamp(cache_path.stat().st_mtime)
                    return df, ts
                except:
                    pass

        # Fetch fresh
        try:
            t = yf.Ticker(ticker)
            qf = t.quarterly_financials
            if qf is None or qf.empty:
                return None, None

            qf = qf.T.reset_index().rename(columns={"index": "date"})
            qf.columns = [c.lower().replace(" ", "").replace("-", "") for c in qf.columns]
            qf["date"] = pd.to_datetime(qf["date"])

            # Normalize key columns
            for old, new in [
                (["totalrevenue", "revenue", "netsales"], "revenue"),
                (["netincome", "netincomeloss"], "netincome"),
                (["dilutedeps", "basiceps", "eps"], "eps"),
            ]:
                for o in old:
                    if o in qf.columns:
                        qf.rename(columns={o: new}, inplace=True)
                        break

            qf.to_parquet(cache_path, index=False)
            return qf, datetime.now()

        except Exception as e:
            logger.warning(f"[{ticker}] Fundamental fetch failed: {e}")
            return None, None


# ══════════════════════════════════════════════════════════════════════════════
# STOCK SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_stock_score_daily(
    ticker: str,
    name: str,
    date: str,
    loader: DataLoader,
    fetcher: FundamentalFetcher,
    configs: Dict[str, Any],
    lookback_days: int = 730,
) -> StockScore:
    """
    Compute ASRE score for a single stock (daily workflow).
    Uses the same logic as CLI compare command.
    """
    result = StockScore(ticker=ticker, name=name, date=date)

    if ticker in NO_DATA_TICKERS:
        result.r_asre = 50.0
        result.error = "Insufficient historical data"
        result.signal = "⚠️  No data"
        return result

    try:
        fundamentals, fetch_ts = fetcher.fetch(ticker)
        if fundamentals is None or len(fundamentals) < 4:
            raise ValueError(
                f"Insufficient fundamentals: {len(fundamentals) if fundamentals is not None else 0} quarters"
            )

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        df = loader.load_stock_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            quarterly_fundamentals=fundamentals,
            fundamentals_fetch_ts=fetch_ts,
        )

        if df is None or len(df) < 60:
            raise ValueError(f"Insufficient price data: {len(df) if df is not None else 0} rows")

        asre_df = compute_complete_asre(
            df=df,
            ticker=ticker,
            config=configs["composite"],
            fundamentals_config=configs["fundamentals"],
            technical_config=configs["technical"],
            momentum_config=configs["momentum"],
            medallion=True,
            return_all_components=True,
        )

        if asre_df is None or asre_df.empty:
            raise ValueError("ASRE output is empty")

        colmap = {
            "f_score": ["f_score", "fscore"],
            "t_score": ["t_score", "tscore"],
            "m_score": ["m_score", "mscore"],
            "r_final": ["r_final", "rfinal"],
            "r_asre": ["r_asre", "rasre"],
            "quality_tier": ["quality_tier", "qualitytier"],
            "stock_category": ["stock_category", "stockcategory"],
            "market_context": ["market_context", "marketcontext"],
            "dip_stage": ["dip_stage", "dip_dip_stage", "medallion_dip_dip_stage"],
        }

        def resolve_col(*candidates):
            for col in candidates:
                if col in asre_df.columns:
                    return col
            return None

        resolved = {key: resolve_col(*cands) for key, cands in colmap.items()}

        score_cols = [resolved[k] for k in ["r_asre", "r_final", "f_score", "t_score", "m_score"] if resolved[k]]
        if not score_cols:
            raise ValueError(
                f"ASRE output missing required score columns. Found: {list(asre_df.columns)}"
            )

        valid_score_rows = asre_df.dropna(subset=score_cols, how="all")
        if valid_score_rows.empty:
            raise ValueError(
                f"ASRE output has no valid scored rows for {ticker}. Last rows:\n"
                f"{asre_df[score_cols].tail(5).to_string()}"
            )

        latest_row = valid_score_rows.iloc[-1]

        def safe_value(row, col_name, default=np.nan):
            if not col_name or col_name not in row.index:
                return default
            value = row[col_name]
            return default if pd.isna(value) else value

        def safe_round(value, digits=1, default=np.nan):
            if value is None or pd.isna(value):
                return default
            return round(float(value), digits)

        result.f_score = safe_round(safe_value(latest_row, resolved["f_score"]))
        result.t_score = safe_round(safe_value(latest_row, resolved["t_score"]))
        result.m_score = safe_round(safe_value(latest_row, resolved["m_score"]))
        result.r_final = safe_round(safe_value(latest_row, resolved["r_final"]))
        result.r_medallion = safe_round(
            safe_value(latest_row, resolved["r_asre"], safe_value(latest_row, resolved["r_final"]))
        )
        result.r_asre = result.r_medallion if not np.isnan(result.r_medallion) else result.r_final

        result.quality_tier = str(safe_value(latest_row, resolved["quality_tier"], "N/A"))
        result.stock_category = str(safe_value(latest_row, resolved["stock_category"], "N/A"))
        result.market_context = str(safe_value(latest_row, resolved["market_context"], "N/A"))

        raw_dip_stage = str(safe_value(latest_row, resolved["dip_stage"], "")).upper()
        mc = result.market_context.upper()

        if raw_dip_stage and raw_dip_stage != "NAN":
            result.dip_stage = raw_dip_stage
        elif "HIGH QUALITY DIP" in mc or "EARLY" in mc:
            result.dip_stage = "EARLY"
        elif "GOOD DIP" in mc or "MID" in mc:
            result.dip_stage = "MID"
        elif "LATE" in mc:
            result.dip_stage = "LATE"
        elif "RECOVERY" in mc:
            result.dip_stage = "RECOVERY"
        else:
            result.dip_stage = "BALANCED"

        result.current_price = round(float(df["close"].dropna().iloc[-1]), 2)

        quality = assess_fundamental_data_quality(fundamentals, ticker)
        result.data_quality_score = quality.get("overall_score", np.nan)

        if np.isnan(result.r_asre):
            raise ValueError(
                f"Latest valid ASRE row still resolved to NaN for {ticker}. "
                f"Selected row:\n{latest_row.to_string()}"
            )

        score = result.r_asre
        if score >= 75:
            result.signal = "🚀 Strong"
        elif score >= 65:
            result.signal = "📈 Good"
        elif score >= 55:
            result.signal = "➡️  Neutral"
        elif score >= 45:
            result.signal = "⚠️  Weak"
        else:
            result.signal = "📉 Poor"

        logger.info(
            f"[{ticker:<18}] R_ASRE={result.r_asre:>5.1f} "
            f"F={result.f_score:>5.1f} T={result.t_score:>5.1f} "
            f"M={result.m_score:>5.1f} Price=₹{result.current_price}"
        )

    except Exception as e:
        result.error = str(e)[:300]
        result.r_asre = 50.0
        result.signal = "❌ Error"
        logger.error(f"[{ticker}] Scoring failed: {e}")

    return result

# ══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════


class StateManager:
    """Manages persistent portfolio state across daily runs."""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = state_dir / "portfolio_state.json"
        self.ledger_file = state_dir / "portfolio_ledger.csv"
        self.score_history_file = state_dir / "score_history.csv"
        self.audit_file = state_dir / "audit_log.jsonl"

    def load_state(self, default_capital: float = 100000.0) -> PortfolioState:
        """Load portfolio state or initialize new."""
        if not self.state_file.exists():
            logger.info("No existing state found, initializing new portfolio")
            return PortfolioState(capital=default_capital, cash=default_capital, invested=0.0)

        try:
            with open(self.state_file, encoding="utf-8") as f:
                data = json.load(f)

            positions = {
                ticker: Position(**pos_data) for ticker, pos_data in data.get("positions", {}).items()
            }

            state = PortfolioState(
                capital=data["capital"],
                cash=data["cash"],
                invested=data["invested"],
                positions=positions,
                last_updated=data.get("last_updated", ""),
                total_value=data.get("total_value", 0.0),
                total_pnl=data.get("total_pnl", 0.0),
                cumulative_realized_pnl=data.get("cumulative_realized_pnl", 0.0),
                nav=data.get("nav", data["capital"] / data.get("units_outstanding", 1000.0)),
                nav_base_capital=data.get("nav_base_capital", data["capital"]),
                units_outstanding=data.get("units_outstanding", 1000.0),
            )

            logger.info(
                f"Loaded state: {len(positions)} positions, "
                f"Cash=₹{state.cash:,.0f}, Invested=₹{state.invested:,.0f}"
            )
            return state

        except Exception as e:
            logger.error(f"Failed to load state: {e}, initializing new")
            return PortfolioState(capital=default_capital, cash=default_capital, invested=0.0)

    def save_state(self, state: PortfolioState, date: str):
        """Save current portfolio state."""
        state.last_updated = date

        data = {
            "capital": state.capital,
            "cash": state.cash,
            "invested": state.invested,
            "total_value": state.total_value,
            "total_pnl": state.total_pnl,
            "cumulative_realized_pnl": state.cumulative_realized_pnl,
            "nav": state.nav,
            "nav_base_capital": state.nav_base_capital,
            "units_outstanding": state.units_outstanding,

            "last_updated": state.last_updated,
            "positions": {ticker: asdict(pos) for ticker, pos in state.positions.items()},
        }

        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"State saved: {len(state.positions)} positions")

    def append_transaction(self, txn: Transaction):
        """Append transaction to ledger."""
        df = pd.DataFrame([asdict(txn)])

        if self.ledger_file.exists():
            df.to_csv(self.ledger_file, mode="a", header=False, index=False)
        else:
            df.to_csv(self.ledger_file, index=False)

    def append_scores(self, scores: List[StockScore]):
        """Append daily scores to history."""
        df = pd.DataFrame([asdict(s) for s in scores])

        if self.score_history_file.exists():
            df.to_csv(self.score_history_file, mode="a", header=False, index=False)
        else:
            df.to_csv(self.score_history_file, index=False)

    def append_audit(self, entry: Dict[str, Any]):
        """Append audit log entry."""
        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# DECISION ENGINE
# ══════════════════════════════════════════════════════════════════════════════


class DecisionEngine:
    """
    Stateful, rule-governed portfolio management engine.

    Behaves like a disciplined SEBI-style portfolio manager:
    deterministic, auditable, F/T/M-aware, and replacement-conscious.
    All decisions are explainable and free of randomness or black-box inference.
    """

    # Quality tier ordering for rank comparison (higher index = better)
    _TIER_ORDER: List[str] = ["D", "C", "C+", "B-", "B", "B+", "A", "A+"]

    # Dip stages considered adverse for hold/sell evaluation
    _ADVERSE_DIP_STAGES: set = {"LATE", "DIVERGENCE", "DISTRIBUTION"}

    # Dip stages considered favourable for BUY timing
    _FAVOURABLE_DIP_STAGES: set = {"EARLY", "MID", "RECOVERY", "ACCUMULATION"}

    def __init__(
        self,
        buy_threshold: float = 70.0,
        hold_threshold: float = 55.0,
        sell_threshold: float = 45.0,
        watch_threshold: float = 60.0,
        max_position_pct: float = 0.15,
        min_position_pct: float = 0.02,
        min_quality_tier: str = "B",
        min_data_quality_score: float = 0.35,
        min_f_score_buy: float = 45.0,
        min_t_score_buy: float = 40.0,
        min_m_score_buy: float = 35.0,
        min_hold_score_buffer: float = 5.0,
        score_drop_sell_threshold: float = 15.0,
        weak_t_score_threshold: float = 35.0,
        weak_m_score_threshold: float = 30.0,
        max_positions: int = 15,
        replacement_gap: float = 8.0,
        allow_replacement: bool = True,
        max_replacements: int = 2,
        freed_capital_haircut: float = 0.02,
    ):
        self.buy_threshold = buy_threshold
        self.hold_threshold = hold_threshold
        self.sell_threshold = sell_threshold
        self.watch_threshold = watch_threshold
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.min_quality_tier = min_quality_tier
        self.min_data_quality_score = min_data_quality_score
        self.min_f_score_buy = min_f_score_buy
        self.min_t_score_buy = min_t_score_buy
        self.min_m_score_buy = min_m_score_buy
        self.min_hold_score_buffer = min_hold_score_buffer
        self.score_drop_sell_threshold = score_drop_sell_threshold
        self.weak_t_score_threshold = weak_t_score_threshold
        self.weak_m_score_threshold = weak_m_score_threshold
        self.max_positions = max_positions
        self.replacement_gap = replacement_gap
        self.allow_replacement = allow_replacement
        self.max_replacements = max_replacements
        # Conservative haircut on freed capital to absorb intraday gap-down risk.
        # E.g. 0.02 means we only count 98% of the estimated sale proceeds.
        self.freed_capital_haircut = freed_capital_haircut

    # ── Internal Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _safe_float(val: Any, default: float = np.nan) -> float:
        """Return float or default if NaN/None."""
        try:
            v = float(val)
            return default if np.isnan(v) else v
        except (TypeError, ValueError):
            return default

    def _quality_rank(self, tier: str) -> int:
        """Return numeric rank of a quality tier (higher = better). Unknown = -1."""
        t = (tier or "").strip().upper()
        return self._TIER_ORDER.index(t) if t in self._TIER_ORDER else -1

    def _tier_is_acceptable(self, tier: str) -> bool:
        """True if the quality tier meets the configured minimum."""
        if not tier or tier.strip().upper() in ("N/A", "NAN", ""):
            return True  # treat unknown as acceptable (data gap, not a red flag)
        return self._quality_rank(tier) >= self._quality_rank(self.min_quality_tier)

    def _conviction_score(self, score: StockScore) -> float:
        """
        Compute a normalized conviction score [0–100] blending R_ASRE,
        F/T/M components, quality tier bonus, and a dip-stage modifier.
        Used only for ranking and priority — never for raw thresholds.
        """
        r = self._safe_float(score.r_asre, 50.0)
        f = self._safe_float(score.f_score, 50.0)
        t = self._safe_float(score.t_score, 50.0)
        m = self._safe_float(score.m_score, 50.0)

        # Weighted blend: R_ASRE dominates, F/T/M add nuance
        blend = 0.50 * r + 0.20 * f + 0.18 * t + 0.12 * m

        # Quality tier bonus/penalty  (+3 for A+, +2 for A, -3 for C/D, etc.)
        tier_bonus = max(-4.0, min(4.0, (self._quality_rank(score.quality_tier) - 4) * 1.0))

        # Dip-stage modifier
        dip = (score.dip_stage or "").upper()
        dip_mod = 2.0 if dip in self._FAVOURABLE_DIP_STAGES else (-2.0 if dip in self._ADVERSE_DIP_STAGES else 0.0)

        return min(100.0, max(0.0, blend + tier_bonus + dip_mod))

    def _compute_priority(self, action: str, conviction: float) -> int:
        """
        Return a sort-friendly priority integer.
        Lower number = execute first (BUY high-conviction first, SELL urgent first).
        """
        if action == "BUY":
            # Highest conviction buys get priority 1; lowest get ~30
            return max(1, int(30 - (conviction - 50) * 0.6))
        if action == "SELL":
            # Urgency: lower conviction held name → more urgent sell
            return max(1, int(conviction * 0.3))
        if action == "WATCH":
            return max(50, int(100 - conviction))
        return 99  # HOLD / SKIP

    def _suggest_allocation(self, score: StockScore, state: PortfolioState, conviction: float) -> float:
        """
        Size the position by conviction band, capped at max_position_pct of capital
        and available cash.
        """
        max_alloc = state.capital * self.max_position_pct
        if conviction >= 85:
            pct = self.max_position_pct          # Full size
        elif conviction >= 75:
            pct = self.max_position_pct * 0.75   # Moderate
        elif conviction >= 65:
            pct = self.max_position_pct * 0.55   # Conservative
        else:
            pct = self.max_position_pct * 0.40   # Starter

        alloc = min(state.cash, state.capital * pct, max_alloc)
        return round(alloc, 2)

    def _build_reason(self, action: str, score: StockScore, **kwargs: Any) -> str:
        """Build a concise, human-readable reason string for audit/compliance."""
        r = self._safe_float(score.r_asre)
        f = self._safe_float(score.f_score)
        t = self._safe_float(score.t_score)
        m = self._safe_float(score.m_score)
        tier = score.quality_tier or "N/A"
        dip = score.dip_stage or "N/A"

        ftm_str = f"F={f:.0f}/T={t:.0f}/M={m:.0f}"

        if action == "BUY":
            extra = kwargs.get("extra", "")
            return (
                f"BUY: R_ASRE={r:.1f}; {ftm_str}; Tier {tier}; {dip} dip"
                + (f"; {extra}" if extra else "")
            )
        if action == "HOLD":
            drop = kwargs.get("drop")
            drop_str = f"; drop from prior={drop:.1f}pts" if drop is not None else ""
            return f"HOLD: score stable at {r:.1f}{drop_str}; {ftm_str}; technicals intact"
        if action == "SELL":
            reason_tag = kwargs.get("tag", "threshold breach")
            return f"SELL: {reason_tag}; R_ASRE={r:.1f}; {ftm_str}"
        if action == "WATCH":
            block = kwargs.get("block", "not yet actionable")
            return f"WATCH: R_ASRE={r:.1f}; {block}"
        if action == "SKIP":
            block = kwargs.get("block", "ineligible")
            return f"SKIP: {block}"
        return f"{action}: R_ASRE={r:.1f}"

    # ── Eligibility Gates ──────────────────────────────────────────────────────

    def _hard_skip(self, score: StockScore) -> Optional[str]:
        """
        Return a skip reason string if the stock must be hard-rejected,
        else return None.
        """
        if score.error:
            return f"error: {score.error[:80]}"
        r = self._safe_float(score.r_asre)
        if np.isnan(r):
            return "R_ASRE is NaN"
        p = self._safe_float(score.current_price)
        if np.isnan(p) or p <= 0:
            return "price unavailable or zero"
        dq = self._safe_float(score.data_quality_score)
        if not np.isnan(dq) and dq < self.min_data_quality_score:
            return f"poor data quality ({dq:.2f} < {self.min_data_quality_score})"
        # All three component scores missing = unreliable
        f = self._safe_float(score.f_score)
        t = self._safe_float(score.t_score)
        m = self._safe_float(score.m_score)
        if np.isnan(f) and np.isnan(t) and np.isnan(m):
            return "all component scores missing"
        return None

    def _is_buy_eligible(self, score: StockScore) -> Tuple[bool, str]:
        """
        Check all buy-eligibility gates.
        Returns (eligible, block_reason).
        """
        r = self._safe_float(score.r_asre)
        if r < self.buy_threshold:
            return False, f"R_ASRE {r:.1f} below buy threshold {self.buy_threshold}"

        f = self._safe_float(score.f_score)
        if not np.isnan(f) and f < self.min_f_score_buy:
            return False, f"F-score too weak ({f:.1f} < {self.min_f_score_buy})"

        t = self._safe_float(score.t_score)
        m = self._safe_float(score.m_score)
        # Both T and M weak simultaneously: bad timing
        if (not np.isnan(t) and t < self.min_t_score_buy and
                not np.isnan(m) and m < self.min_m_score_buy):
            return False, f"timing & momentum both weak (T={t:.1f}, M={m:.1f})"

        if not self._tier_is_acceptable(score.quality_tier):
            return False, f"quality tier {score.quality_tier} below minimum {self.min_quality_tier}"

        dip = (score.dip_stage or "").upper()
        if dip in self._ADVERSE_DIP_STAGES:
            return False, f"adverse dip stage ({dip})"

        return True, ""

    def _is_hold_healthy(self, score: StockScore, pos: "Position") -> Tuple[bool, str]:
        """
        Check if a held position should continue to be held.
        Returns (healthy, sell_reason).
        """
        r = self._safe_float(score.r_asre)

        # Hard sell: score below sell floor
        if r < self.sell_threshold:
            return False, f"R_ASRE {r:.1f} dropped below sell floor {self.sell_threshold}"

        # Score deterioration check
        prior = self._safe_float(pos.last_r_asre)
        if not np.isnan(prior):
            drop = prior - r
            if drop >= self.score_drop_sell_threshold:
                return False, f"R_ASRE dropped {drop:.1f} pts from prior review ({prior:.1f} → {r:.1f})"

        # Both T and M severely weak while held
        t = self._safe_float(score.t_score)
        m = self._safe_float(score.m_score)
        if (not np.isnan(t) and t < self.weak_t_score_threshold and
                not np.isnan(m) and m < self.weak_m_score_threshold):
            return False, f"timing & momentum severely weak (T={t:.1f}, M={m:.1f})"

        # Adverse dip stage combined with weak score
        dip = (score.dip_stage or "").upper()
        if dip in self._ADVERSE_DIP_STAGES and r < (self.hold_threshold + self.min_hold_score_buffer):
            return False, f"adverse dip stage ({dip}) with marginal score {r:.1f}"

        # Data quality breakdown for held positions
        dq = self._safe_float(score.data_quality_score)
        if not np.isnan(dq) and dq < (self.min_data_quality_score * 0.7):
            return False, f"data quality severely degraded ({dq:.2f})"

        return True, ""

    def _should_force_sell(self, score: StockScore, pos: "Position") -> Tuple[bool, str]:
        """Alias wrapper — returns (force_sell, reason)."""
        healthy, reason = self._is_hold_healthy(score, pos)
        return (not healthy), reason

    # ── Public API ────────────────────────────────────────────────────────────

    def decide(
        self, score: StockScore, state: PortfolioState, max_positions: int = 15
    ) -> DailyAction:
        """
        Decide BUY/HOLD/SELL/WATCH/SKIP for a single stock.

        Order of evaluation:
        1. Hard SKIP gates (error, NaN, price, data quality)
        2. If held → HOLD/SELL evaluation using prior state + component scores
        3. If not held → BUY eligibility → capacity/cash → BUY/WATCH
        4. WATCH if score is between watch_threshold and buy_threshold
        5. SKIP as default fallback
        """
        ticker = score.ticker
        currently_held = ticker in state.positions

        # ── 1. Hard rejection ──────────────────────────────────────────────
        skip_reason = self._hard_skip(score)
        if skip_reason:
            return DailyAction(
                ticker=ticker,
                name=score.name,
                action="SKIP",
                r_asre=self._safe_float(score.r_asre, np.nan),
                current_price=self._safe_float(score.current_price, np.nan),
                reason=self._build_reason("SKIP", score, block=skip_reason),
                priority=99,
            )

        r = self._safe_float(score.r_asre)
        conviction = self._conviction_score(score)

        # ── 2. Held position evaluation ────────────────────────────────────
        if currently_held:
            pos = state.positions[ticker]
            prior = self._safe_float(pos.last_r_asre)
            drop = (prior - r) if not np.isnan(prior) else None

            force_sell, sell_tag = self._should_force_sell(score, pos)
            if force_sell:
                return DailyAction(
                    ticker=ticker,
                    name=score.name,
                    action="SELL",
                    r_asre=r,
                    current_price=score.current_price,
                    reason=self._build_reason("SELL", score, tag=sell_tag),
                    priority=self._compute_priority("SELL", conviction),
                )

            # HOLD: position is healthy
            return DailyAction(
                ticker=ticker,
                name=score.name,
                action="HOLD",
                r_asre=r,
                current_price=score.current_price,
                reason=self._build_reason("HOLD", score, drop=drop),
                priority=self._compute_priority("HOLD", conviction),
            )

        # ── 3. Not held — BUY eligibility ────────────────────────────────
        eligible, block = self._is_buy_eligible(score)

        if eligible:
            # Capacity check
            n_positions = len(state.positions)
            cap = max_positions if max_positions else self.max_positions
            if n_positions >= cap:
                return DailyAction(
                    ticker=ticker,
                    name=score.name,
                    action="WATCH",
                    r_asre=r,
                    current_price=score.current_price,
                    reason=self._build_reason("WATCH", score, block=f"buy-eligible but portfolio full ({n_positions}/{cap})"),
                    priority=self._compute_priority("WATCH", conviction),
                )

            # Cash check
            min_alloc = state.capital * self.min_position_pct
            if state.cash < min_alloc:
                return DailyAction(
                    ticker=ticker,
                    name=score.name,
                    action="WATCH",
                    r_asre=r,
                    current_price=score.current_price,
                    reason=self._build_reason("WATCH", score, block=f"buy-eligible but insufficient cash (₹{state.cash:,.0f})"),
                    priority=self._compute_priority("WATCH", conviction),
                )

            # BUY
            alloc = self._suggest_allocation(score, state, conviction)
            extra = "high conviction" if conviction >= 85 else ("moderate conviction" if conviction >= 75 else "standard conviction")
            return DailyAction(
                ticker=ticker,
                name=score.name,
                action="BUY",
                r_asre=r,
                current_price=score.current_price,
                reason=self._build_reason("BUY", score, extra=extra),
                priority=self._compute_priority("BUY", conviction),
                suggested_allocation=alloc,
            )

        # ── 4. WATCH — good but blocked ───────────────────────────────────
        if r >= self.watch_threshold:
            return DailyAction(
                ticker=ticker,
                name=score.name,
                action="WATCH",
                r_asre=r,
                current_price=score.current_price,
                reason=self._build_reason("WATCH", score, block=block or f"below buy threshold ({self.buy_threshold})"),
                priority=self._compute_priority("WATCH", conviction),
            )

        # ── 5. Default SKIP ───────────────────────────────────────────────
        return DailyAction(
            ticker=ticker,
            name=score.name,
            action="SKIP",
            r_asre=r,
            current_price=score.current_price,
            reason=self._build_reason("SKIP", score, block=f"score {r:.1f} below watch threshold {self.watch_threshold}"),
            priority=99,
        )

    def rank_candidates(
        self, scores: List[StockScore], state: PortfolioState
    ) -> List[Tuple[float, StockScore]]:
        """
        Rank buy-candidate stocks by conviction score (descending).
        Returns [(conviction, StockScore), ...] for stocks that pass hard gates
        and buy-eligibility, excluding currently held names.
        Useful for replacement logic in the orchestration layer.
        """
        results: List[Tuple[float, StockScore]] = []
        for s in scores:
            if s.ticker in state.positions:
                continue
            if self._hard_skip(s):
                continue
            eligible, _ = self._is_buy_eligible(s)
            if eligible:
                results.append((self._conviction_score(s), s))
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def weakest_holdings(
        self, state: PortfolioState, scores_dict: Dict[str, "StockScore"]
    ) -> List[Tuple[float, str]]:
        """
        Return held tickers sorted by conviction ascending (weakest first).
        Useful for replacement-aware portfolio rotation decisions.
        Returns [(conviction, ticker), ...].
        """
        result: List[Tuple[float, str]] = []
        for ticker, pos in state.positions.items():
            sc = scores_dict.get(ticker)
            if sc is None:
                result.append((0.0, ticker))
            else:
                result.append((self._conviction_score(sc), ticker))
        result.sort(key=lambda x: x[0])
        return result


# ══════════════════════════════════════════════════════════════════════════════
# REPLACEMENT PASS
# ══════════════════════════════════════════════════════════════════════════════


def _estimate_freed_capital(
    ticker: str,
    state: PortfolioState,
    scores_dict: Dict[str, StockScore],
    haircut: float,
) -> float:
    """
    Conservative estimate of net sale proceeds for a held position.

    Uses today's scored price when available; falls back to entry price.
    Applies a fractional haircut (e.g. 2%) to account for intraday gap-down
    risk, bid-ask slippage, and brokerage costs.
    """
    pos = state.positions[ticker]
    raw_price = (
        scores_dict[ticker].current_price
        if ticker in scores_dict and not np.isnan(scores_dict[ticker].current_price)
        else pos.entry_price
    )
    gross = pos.quantity * raw_price
    return gross * (1.0 - haircut)


def _run_replacement_pass(
    actions: List[DailyAction],
    scores_dict: Dict[str, StockScore],
    state: PortfolioState,
    engine: DecisionEngine,
    max_positions: int,
    today: str,
) -> List[str]:
    """
    Multi-candidate, capital-optimal portfolio rotation pass.

    Runs after initial decide() calls and mutates ``actions`` in-place.

    Algorithm (greedy, deterministic):
    1. Rank all buy-eligible non-held candidates by conviction (descending).
    2. Rank all currently-held names by conviction (ascending = weakest first).
    3. Greedily match best-available candidate against weakest-available holding:
       - Skip the pair if gap < replacement_gap.
       - Skip the holding if its current action is not HOLD (already SELL'd).
       - Skip the candidate if already BUY.
       - Deduct haircutted freed capital from a running pool; add to available cash.
       - Size the incoming BUY from the running capital pool.
       - Stop when the pool is exhausted, no pairs remain, or max_replacements hit.
    4. At most engine.max_replacements swaps per daily run.

    Capital haircut (engine.freed_capital_haircut, default 2%) is applied to
    every freed position to absorb intraday gap-down / slippage risk.

    Returns a list of human-readable log lines for console + audit.
    """
    if not engine.allow_replacement:
        return ["Replacement pass disabled (allow_replacement=False)"]

    log: List[str] = []

    # ── Build a mutable lookup: ticker → DailyAction ───────────────────────
    action_map: Dict[str, DailyAction] = {a.ticker: a for a in actions}

    # ── Rank candidates (not held, pass hard gates + buy-eligibility) ──────
    candidates = engine.rank_candidates(scores_dict.values(), state)  # type: ignore[arg-type]
    if not candidates:
        log.append("Replacement pass: no buy-eligible candidates found")
        return log

    # ── Rank held names by conviction ascending (weakest first) ───────────
    weakest_list = engine.weakest_holdings(state, scores_dict)
    if not weakest_list:
        log.append("Replacement pass: no held positions to evaluate")
        return log

    log.append(
        f"Replacement pass: {len(candidates)} candidate(s), "
        f"{len(weakest_list)} holding(s); "
        f"max_replacements={engine.max_replacements}, "
        f"gap_required={engine.replacement_gap}, "
        f"haircut={engine.freed_capital_haircut*100:.0f}%"
    )

    # ── Greedy matching loop ───────────────────────────────────────────────
    # running_cash: simulated cash available for incoming buys across all swaps
    running_cash: float = state.cash
    n_replaced: int = 0
    # Track which holdings/candidates are already committed in this pass
    committed_sells: set = set()
    committed_buys: set = set()

    # Pre-compute which tickers were already SELL or BUY from the base Phase 4
    # pass so that max_replacements counts only true rotation swaps.
    base_sells: set = {a.ticker for a in actions if a.action == "SELL"}
    base_buys: set = {a.ticker for a in actions if a.action == "BUY"}

    for cand_conviction, cand_score in candidates:
        if n_replaced >= engine.max_replacements:
            log.append(f"  → max_replacements={engine.max_replacements} reached, stopping")
            break

        if cand_score.ticker in committed_buys:
            continue

        # Skip if candidate already has a BUY action from Phase 4 base pass
        # (does not count toward max_replacements — it's not a rotation swap)
        cand_action = action_map.get(cand_score.ticker)
        if cand_action is not None and cand_score.ticker in base_buys:
            committed_buys.add(cand_score.ticker)
            continue

        # Find the weakest eligible holding for this candidate.
        # Skip holdings already flagged SELL by the base Phase 4 pass —
        # those exits are already happening and must not count as rotation swaps.
        matched_holding: Optional[Tuple[float, str]] = None
        for hold_conv, hold_ticker in weakest_list:
            if hold_ticker in committed_sells:
                continue
            if hold_ticker in base_sells:
                # Already a natural SELL — skip without consuming the slot
                continue
            hold_action = action_map.get(hold_ticker)
            # Guard: must be a HOLD decision (not already flagged for SELL)
            if hold_action is None or hold_action.action != "HOLD":
                continue
            gap = cand_conviction - hold_conv
            if gap < engine.replacement_gap:
                # weakest_list is ascending; all subsequent will also fail
                break
            matched_holding = (hold_conv, hold_ticker)
            break  # take the weakest match; move to next candidate in outer loop

        if matched_holding is None:
            # No holding is weak enough relative to this candidate
            log.append(
                f"  → {cand_score.ticker} (conv={cand_conviction:.1f}): "
                f"no holding weak enough to replace"
            )
            continue

        hold_conv, hold_ticker = matched_holding
        gap = cand_conviction - hold_conv

        # ── Haircutted freed capital for this swap ─────────────────────────
        freed = _estimate_freed_capital(
            hold_ticker, state, scores_dict, engine.freed_capital_haircut
        )
        running_cash += freed

        # ── Size the incoming BUY against the running pool ─────────────────
        simulated_state = PortfolioState(
            capital=state.capital,
            cash=running_cash,
            invested=state.invested,
            positions=state.positions,
        )
        new_alloc = engine._suggest_allocation(cand_score, simulated_state, cand_conviction)

        # Capital budget guard: don't over-allocate the pool
        min_alloc = state.capital * engine.min_position_pct
        if new_alloc < min_alloc:
            log.append(
                f"  → {cand_score.ticker}: allocation ₹{new_alloc:,.0f} below "
                f"min position size ₹{min_alloc:,.0f}, skipping swap"
            )
            running_cash -= freed  # roll back freed capital
            continue

        running_cash -= new_alloc  # deduct the committed buy amount

        # ── Flip holding HOLD → SELL ───────────────────────────────────────
        hold_action = action_map[hold_ticker]
        hold_action.action = "SELL"
        hold_action.reason = (
            f"SELL (rotation #{n_replaced+1}): replaced by {cand_score.ticker}; "
            f"conviction gap={gap:.1f}pts; haircutted proceeds=₹{freed:,.0f}"
        )
        hold_action.priority = 1
        hold_action.rotation_buy_of = cand_score.ticker
        hold_action.capital_haircut_used = engine.freed_capital_haircut
        committed_sells.add(hold_ticker)

        # ── Flip candidate WATCH → BUY (or inject) ────────────────────────
        if cand_action is not None:
            cand_action.action = "BUY"
            cand_action.reason = (
                f"BUY (rotation #{n_replaced+1}): replaces {hold_ticker}; "
                f"gap={gap:.1f}pts; "
                + engine._build_reason("BUY", cand_score, extra="rotation")
            )
            cand_action.suggested_allocation = new_alloc
            cand_action.priority = engine._compute_priority("BUY", cand_conviction)
            cand_action.rotation_sell_of = hold_ticker
        else:
            injected = DailyAction(
                ticker=cand_score.ticker,
                name=cand_score.name,
                action="BUY",
                r_asre=engine._safe_float(cand_score.r_asre),
                current_price=engine._safe_float(cand_score.current_price),
                reason=(
                    f"BUY (rotation #{n_replaced+1}): replaces {hold_ticker}; "
                    f"gap={gap:.1f}pts"
                ),
                priority=engine._compute_priority("BUY", cand_conviction),
                suggested_allocation=new_alloc,
                rotation_sell_of=hold_ticker,
            )
            actions.append(injected)
            action_map[cand_score.ticker] = injected

        committed_buys.add(cand_score.ticker)
        n_replaced += 1

        log.append(
            f"  → SWAP #{n_replaced}: SELL {hold_ticker} (conv={hold_conv:.1f}) "
            f"| BUY {cand_score.ticker} (conv={cand_conviction:.1f}) "
            f"alloc=₹{new_alloc:,.0f} freed=₹{freed:,.0f} pool=₹{running_cash:,.0f}"
        )

    if n_replaced == 0:
        log.append("  → No replacements triggered (gap or capital constraints not met)")
    else:
        log.append(f"  → Replacement pass complete: {n_replaced} swap(s) executed")

    return log


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def execute_buy(
    state: PortfolioState,
    action: DailyAction,
    date: str,
    state_mgr: StateManager,
) -> Optional[Transaction]:
    """Execute a BUY using whole shares only. Returns None if < 1 share affordable."""
    quantity, amount = calculate_whole_quantity(action.suggested_allocation, action.current_price)

    # If suggested allocation can't buy 1 share, try with all available cash
    if quantity <= 0:
        quantity, amount = calculate_whole_quantity(state.cash, action.current_price)

    if quantity <= 0 or amount <= 0:
        logger.warning(
            f"BUY skipped [{action.ticker}]: ₹{action.suggested_allocation:,.2f} "
            f"cannot buy 1 share @ ₹{action.current_price:.2f}"
        )
        return None

    # Clamp to actual available cash (safety guard)
    if amount > state.cash:
        quantity = int(math.floor(state.cash / action.current_price))
        amount = round(quantity * action.current_price, 2)

    if quantity <= 0:
        return None

    # Update action for report/audit
    action.suggested_quantity = quantity
    action.suggested_allocation = amount

    pos = Position(
        ticker=action.ticker,
        name=action.name,
        quantity=quantity,
        entry_price=action.current_price,
        entry_date=date,
        current_price=action.current_price,
        current_value=amount,
        last_r_asre=action.r_asre,
    )
    state.positions[action.ticker] = pos
    state.cash = round(state.cash - amount, 2)
    state.invested = round(state.invested + amount, 2)

    txn = Transaction(
        date=date, action="BUY", ticker=action.ticker, name=action.name,
        quantity=quantity, price=action.current_price, value=amount,
        reason=action.reason, r_asre=action.r_asre,
    )
    state_mgr.append_transaction(txn)
    logger.info(f"BUY  {action.ticker}: {quantity} shares @ ₹{action.current_price:.2f} = ₹{amount:,.2f}")
    return txn

def execute_sell(
    state: PortfolioState,
    action: DailyAction,
    date: str,
    state_mgr: StateManager,
) -> Transaction:
    """Execute a full SELL of all whole shares held."""
    pos = state.positions[action.ticker]

    sale_value = round(pos.quantity * action.current_price, 2)
    cost_basis = round(pos.quantity * pos.entry_price, 2)
    realized_pnl = round(sale_value - cost_basis, 2)
    realized_pnl_pct = round((realized_pnl / cost_basis) * 100, 2) if cost_basis > 0 else 0.0

    txn = Transaction(
        date=date,
        action="SELL",
        ticker=action.ticker,
        name=action.name,
        quantity=int(pos.quantity),
        price=action.current_price,
        value=sale_value,
        reason=f"{action.reason} | P&L: ₹{realized_pnl:+,.2f}",
        r_asre=action.r_asre,
        entry_price=pos.entry_price,
        days_held=pos.days_held,
        realized_pnl=realized_pnl,
        realized_pnl_pct=realized_pnl_pct,
    )

    state.cash = round(state.cash + sale_value, 2)
    state.invested = round(max(0.0, state.invested - cost_basis), 2)
    state.cumulative_realized_pnl = round(state.cumulative_realized_pnl + realized_pnl, 2)

    del state.positions[action.ticker]

    state_mgr.append_transaction(txn)
    logger.info(
        f"SELL {action.ticker}: {pos.quantity} shares @ ₹{action.current_price:.2f} "
        f"= ₹{sale_value:,.2f} (P&L: ₹{realized_pnl:+,.2f})"
    )
    return txn

def update_positions(state: PortfolioState, scores: Dict[str, StockScore], date: str):
    for ticker, pos in state.positions.items():
        if ticker in scores:
            score = scores[ticker]
            pos.current_price = score.current_price
            pos.current_value = round(pos.quantity * pos.current_price, 2)
            pos.unrealized_pnl = round(pos.current_value - (pos.quantity * pos.entry_price), 2)
            pos.unrealized_pnl_pct = round(
                (pos.unrealized_pnl / (pos.quantity * pos.entry_price)) * 100, 2
            ) if (pos.quantity * pos.entry_price) > 0 else 0.0
            pos.last_r_asre = score.r_asre
            entry_dt = datetime.strptime(pos.entry_date, "%Y-%m-%d")
            current_dt = datetime.strptime(date, "%Y-%m-%d")
            pos.days_held = (current_dt - entry_dt).days

    state.total_value = round(state.cash + sum(p.current_value for p in state.positions.values()), 2)
    state.total_pnl = round(state.total_value - state.capital, 2)
    state.nav = compute_nav(state)   # ← NEW: recalculate NAV after every position refresh

def compute_nav(state: PortfolioState) -> float:
    """NAV per unit = total portfolio value ÷ units outstanding."""
    if state.units_outstanding <= 0:
        return 0.0
    return round(state.total_value / state.units_outstanding, 4)


def calculate_whole_quantity(target_amount: float, price: float) -> Tuple[int, float]:
    """
    Return the largest whole-share count affordable within target_amount,
    and the exact cash outlay for those shares.
    Never returns fractional shares.
    """
    if price <= 0 or target_amount <= 0:
        return 0, 0.0
    qty = int(math.floor(target_amount / price))
    if qty <= 0:
        return 0, 0.0
    return qty, round(qty * price, 2)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════


def main(args: argparse.Namespace):
    """Daily portfolio manager orchestration."""

    RUN_ID = str(uuid.uuid4())
    TODAY = datetime.now().strftime("%Y-%m-%d")
    TODAY_COMPACT = datetime.now().strftime("%Y%m%d")

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]ASRE DAILY PORTFOLIO MANAGER[/bold cyan]\n"
            f"[dim]Production Edition v2.0[/dim]\n\n"
            f"Date:    {TODAY}\n"
            f"Run ID:  {RUN_ID}\n"
            f"Mode:    {args.mode.upper()}\n"
            f"Dry Run: {args.dry_run}",
            border_style="cyan",
        )
    )

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1: SEBI VALIDATION
    # ──────────────────────────────────────────────────────────────────────────

    console.print("\n[bold]PHASE 1:[/bold] SEBI Role Gate Validation")

    try:
        gate = RoleGate(mode=args.mode, strict=True)
        approval = gate.validate()
        console.print(f"  ✓ Authorized: {approval}", style="green")
    except RoleGateError as e:
        console.print(f"  ✗ SEBI authorization failed: {e}", style="bold red")
        console.print("  [yellow]For dev/testing: ASRE_SKIP_REG=1[/yellow]")
        sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────
# PHASE 2: LOAD STATE
# ──────────────────────────────────────────────────────────────────────────

    console.print("\n[bold]PHASE 2:[/bold] Loading Portfolio State")

    state_dir = Path(args.state_dir)
    state_mgr = StateManager(state_dir)

    if args.init:
        console.print(f"  [yellow]Initializing new portfolio with ₹{args.capital:,}[/yellow]")

        initial_units = 1000.0
        initial_nav = round(args.capital / initial_units, 4)

        state = PortfolioState(
            capital=args.capital,
            cash=args.capital,
            invested=0.0,
            total_value=args.capital,
            total_pnl=0.0,
            cumulative_realized_pnl=0.0,
            nav_base_capital=args.capital,
            units_outstanding=initial_units,
            nav=initial_nav,
        )

        state_mgr.save_state(state, TODAY)

        console.print(
            f"  ✓ Portfolio initialized | NAV: {state.nav:.4f} | Units: {state.units_outstanding:,.2f}",
            style="green",
        )
        return

    state = state_mgr.load_state(args.capital)

# Backward-compatibility patch for older state files that don't yet contain NAV fields
    if not hasattr(state, "nav_base_capital") or state.nav_base_capital <= 0:
        state.nav_base_capital = state.capital

    if not hasattr(state, "units_outstanding") or state.units_outstanding <= 0:
        state.units_outstanding = 1000.0

    if not hasattr(state, "cumulative_realized_pnl"):
        state.cumulative_realized_pnl = 0.0

# Rebuild total_value if old state files don't have it populated correctly
    if not hasattr(state, "total_value") or state.total_value <= 0:
        state.total_value = round(
            state.cash + sum(getattr(pos, "current_value", 0.0) for pos in state.positions.values()),
            2,
        )

    state.nav = round(state.total_value / state.units_outstanding, 4)

    console.print(
        f"  ✓ Loaded: {len(state.positions)} positions | "
        f"Cash: ₹{state.cash:,.0f} | Invested: ₹{state.invested:,.0f} | "
        f"NAV: {state.nav:.4f}",
        style="green",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 3: COMPUTE SCORES
    # ──────────────────────────────────────────────────────────────────────────

    console.print(f"\n[bold]PHASE 3:[/bold] Computing ASRE Scores ({len(STOCK_UNIVERSE)} stocks)")

    configs = {
        "composite": CompositeConfig(),
        "fundamentals": FundamentalsConfig(),
        "technical": TechnicalConfig(),
        "momentum": MomentumConfig.balanced(),
    }

    cache_dir = Path.home() / ".asrecache"
    loader = DataLoader()
    fetcher = FundamentalFetcher(cache_dir)

    scores: List[StockScore] = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn()
    ) as progress:
        task = progress.add_task("[cyan]Scoring stocks...", total=len(STOCK_UNIVERSE))

        for name, ticker in STOCK_UNIVERSE:
            score = compute_stock_score_daily(
                ticker=ticker,
                name=name,
                date=TODAY,
                loader=loader,
                fetcher=fetcher,
                configs=configs,
                lookback_days=args.lookback_days,
            )
            scores.append(score)
            progress.advance(task)

    scores_dict = {s.ticker: s for s in scores}

    console.print(f"  ✓ Scored {len(scores)} stocks", style="green")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 4: GENERATE ACTIONS
    # ──────────────────────────────────────────────────────────────────────────

    console.print("\n[bold]PHASE 4:[/bold] Generating Daily Actions")

    engine = DecisionEngine(
        buy_threshold=args.buy_threshold,
        hold_threshold=args.hold_threshold,
        sell_threshold=args.sell_threshold,
        watch_threshold=args.watch_threshold,
        max_position_pct=args.max_position_pct,
        max_positions=args.max_positions,
    )

    actions: List[DailyAction] = []
    for score in scores:
        action = engine.decide(score, state, max_positions=args.max_positions)
        actions.append(action)

    # Count by action type
    action_counts = defaultdict(int)
    for a in actions:
        action_counts[a.action] += 1

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 4.5: REPLACEMENT PASS (conviction-based portfolio rotation)
    # ──────────────────────────────────────────────────────────────────────────

    console.print("\n[bold]PHASE 4.5:[/bold] Replacement Pass (conviction rotation)")

    replacement_log = _run_replacement_pass(
        actions=actions,
        scores_dict=scores_dict,
        state=state,
        engine=engine,
        max_positions=args.max_positions,
        today=TODAY,
    )

    for line in replacement_log:
        console.print(f"  {line}", style="dim cyan")

    # Recount after replacement mutations
    action_counts = defaultdict(int)
    for a in actions:
        action_counts[a.action] += 1

    # --- Explainability via Groq ---
    explainable_actions = [a for a in actions if a.action in ("BUY", "SELL", "HOLD")]
    if explainable_actions:
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                
                with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn()) as prog:
                    task = prog.add_task("[magenta]Generating AI Explanations...", total=len(explainable_actions))
                    for a in explainable_actions:
                        score_obj = scores_dict[a.ticker]
                        prompt = (
                            f"You are the ASRE AI Analyst. Explain in one short, punchy sentence why we are issuing a {a.action} signal for {a.ticker} ({a.name}).\n"
                            f"Metrics: Action Reason: '{a.reason}', ASRE Score: {score_obj.r_asre:.1f}/100, "
                            f"F-Score: {score_obj.f_score:.1f}, T-Score: {score_obj.t_score:.1f}, M-Score: {score_obj.m_score:.1f}, "
                            f"Quality Tier: {score_obj.quality_tier}, Dip Stage: {score_obj.dip_stage}.\n"
                            f"Keep it under 25 words. Be direct. Do not include intro or outro."
                        )
                        try:
                            response = client.chat.completions.create(
                                messages=[{"role": "user", "content": prompt}],
                                model="llama3-8b-8192",
                                temperature=0.3,
                                max_tokens=60
                            )
                            a.explanation = response.choices[0].message.content.strip()
                        except Exception as e:
                            a.explanation = f"LLM Error: {e}"
                        prog.advance(task)
            except ImportError:
                logger.warning("groq library not installed, skipping AI explanations.")
        else:
            logger.info("GROQ_API_KEY not set, skipping AI explanations.")

    console.print(
        f"  ✓ Actions: BUY={action_counts['BUY']} SELL={action_counts['SELL']} "
        f"HOLD={action_counts['HOLD']} WATCH={action_counts['WATCH']} SKIP={action_counts['SKIP']}",
        style="green",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 5: EXECUTE ACTIONS (if not dry run)
    # ──────────────────────────────────────────────────────────────────────────

    console.print("\n[bold]PHASE 5:[/bold] Executing Actions")

    transactions: List[Transaction] = []

    if args.dry_run:
        console.print("  [yellow]DRY RUN — No actual trades executed[/yellow]")
    else:
        # Execute SELL actions first
        for action in sorted(
            [a for a in actions if a.action == "SELL"], key=lambda x: x.priority
        ):
            txn = execute_sell(state, action, TODAY, state_mgr)
            transactions.append(txn)

        # Execute BUY actions
        for action in sorted(
            [a for a in actions if a.action == "BUY"], key=lambda x: x.priority
        ):
            if state.cash >= action.suggested_allocation:
                txn = execute_buy(state, action, TODAY, state_mgr)
                transactions.append(txn)
            else:
                logger.warning(f"Insufficient cash for {action.ticker}, skipping")

        console.print(
            f"  ✓ Executed {len(transactions)} transactions "
            f"({len([t for t in transactions if t.action == 'BUY'])} buys, "
            f"{len([t for t in transactions if t.action == 'SELL'])} sells)",
            style="green",
        )

    # Update position values
    update_positions(state, scores_dict, TODAY)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 6: SAVE STATE & OUTPUTS
    # ──────────────────────────────────────────────────────────────────────────

    console.print("\n[bold]PHASE 6:[/bold] Saving Results")

    if not args.dry_run:
        state_mgr.save_state(state, TODAY)
        state_mgr.append_scores(scores)
        console.print(f"  ✓ State saved to {state_dir}", style="green")

    # Export daily report
    report_txt = state_dir / f"daily_actions_{TODAY_COMPACT}.txt"
    report_json = state_dir / f"daily_report_{TODAY_COMPACT}.json"

    # Text report
    report_lines = [
        "=" * 80,
        f"ASRE DAILY ACTION REPORT — {TODAY}",
        "=" * 80,
        f"Run ID: {RUN_ID}",
        f"Mode: {args.mode.upper()}",
        f"Dry Run: {args.dry_run}",
        "",
        "─" * 80,
        "PORTFOLIO SUMMARY",
        "─" * 80,
        f"Total Value:     ₹{state.total_value:,.0f}",
        f"Cash:            ₹{state.cash:,.0f}",
        f"Invested:        ₹{state.invested:,.0f}",
        f"Total P&L:       ₹{state.total_pnl:+,.0f} ({(state.total_pnl/state.capital)*100:+.2f}%)",
        f"Active Positions: {len(state.positions)}",
        "",
    ]

    # BUY recommendations
    buy_actions = sorted([a for a in actions if a.action == "BUY"], key=lambda x: x.priority)
    if buy_actions:
        report_lines += [
            "─" * 80,
            "🟢 BUY RECOMMENDATIONS (Sorted by Priority)",
            "─" * 80,
        ]
        for a in buy_actions:
            report_lines.append(
                f"  {a.ticker:<18} R_ASRE={a.r_asre:>5.1f}  Price=₹{a.current_price:>8.2f}  "
                f"Allocation=₹{a.suggested_allocation:>10,.0f}"
            )
            report_lines.append(f"    Reason: {a.reason}")
            if a.explanation:
                report_lines.append(f"    🤖 AI Note: {a.explanation}")
            report_lines.append("")
            
    sell_txn_map = {
        t.ticker: t
        for t in transactions
        if t.action == "SELL"
    }
    # SELL recommendations
    sell_actions = sorted([a for a in actions if a.action == "SELL"], key=lambda x: x.priority)
    if sell_actions:
        report_lines += [
            "─" * 80,
            "🔴 SELL RECOMMENDATIONS (Urgent)",
            "─" * 80,
        ]
        for a in sell_actions:
            sell_txn = sell_txn_map.get(a.ticker)

            if sell_txn:
                report_lines.append(
                    f"  {a.ticker:<18} R_ASRE={a.r_asre:>5.1f}  Price=₹{a.current_price:>8.2f}  "
                    f"Held {sell_txn.days_held} days  P&L: ₹{sell_txn.realized_pnl:+,.0f} "
                    f"({sell_txn.realized_pnl_pct:+.1f}%)"
                )
            else:
                report_lines.append(
                    f"  {a.ticker:<18} R_ASRE={a.r_asre:>5.1f}  Price=₹{a.current_price:>8.2f}"
                )

            report_lines.append(f"    Reason: {a.reason}")
            if a.explanation:
                report_lines.append(f"    🤖 AI Note: {a.explanation}")
            report_lines.append("")

    # HOLD positions
    hold_actions = [a for a in actions if a.action == "HOLD"]
    if hold_actions:
        report_lines += [
            "─" * 80,
            "🟡 CURRENT HOLDINGS (HOLD)",
            "─" * 80,
        ]
        for a in hold_actions:
            pos = state.positions.get(a.ticker)
            if pos:
                report_lines.append(
                    f"  {a.ticker:<18} R_ASRE={a.r_asre:>5.1f}  Price=₹{a.current_price:>8.2f}  "
                    f"Value=₹{pos.current_value:>10,.0f}  P&L: ₹{pos.unrealized_pnl:+,.0f} "
                    f"({pos.unrealized_pnl_pct:+.1f}%)"
                )
            if a.explanation:
                report_lines.append(f"    🤖 AI Note: {a.explanation}")

    # WATCH list
    watch_actions = sorted(
        [a for a in actions if a.action == "WATCH"], key=lambda x: x.r_asre, reverse=True
    )[:10]
    if watch_actions:
        report_lines += [
            "",
            "─" * 80,
            "👁️  WATCH LIST (Top 10)",
            "─" * 80,
        ]
        for a in watch_actions:
            report_lines.append(
                f"  {a.ticker:<18} R_ASRE={a.r_asre:>5.1f}  Price=₹{a.current_price:>8.2f}"
            )
            report_lines.append(f"    {a.reason}")

    report_lines += [
        "",
        "=" * 80,
        "DISCLAIMER: Algorithmic output only. Not investment advice.",
        "For SEBI-registered IA/RA use only.",
        "=" * 80,
    ]

    report_txt.write_text("\n".join(report_lines), encoding="utf-8")
    console.print(f"  ✓ Text report: {report_txt}", style="green")

    # JSON report
    rotation_swaps = [
        {
            "sell": a.ticker,
            "buy": a.rotation_buy_of,
            "haircutted_proceeds": round(
                (state.positions[a.ticker].quantity * scores_dict[a.ticker].current_price
                 if a.ticker in state.positions and a.ticker in scores_dict
                 else 0.0) * (1.0 - a.capital_haircut_used), 2
            ) if a.capital_haircut_used else None,
            "haircut_pct": round(a.capital_haircut_used * 100, 1) if a.capital_haircut_used else None,
            "sell_reason": a.reason,
        }
        for a in actions
        if a.action == "SELL" and a.rotation_buy_of
    ]

    report_data = {
        "run_id": RUN_ID,
        "date": TODAY,
        "mode": args.mode.upper(),
        "dry_run": args.dry_run,
        "portfolio": {
            "total_value": state.total_value,
            "cash": state.cash,
            "invested": state.invested,
            "total_pnl": state.total_pnl,
            "total_pnl_pct": (state.total_pnl / state.capital) * 100,
            "active_positions": len(state.positions),
        },
        "actions": {
            "buy": [asdict(a) for a in buy_actions],
            "sell": [asdict(a) for a in sell_actions],
            "hold": [asdict(a) for a in hold_actions],
            "watch": [asdict(a) for a in watch_actions[:10]],
        },
        "rotation_swaps": rotation_swaps,
        "transactions": [asdict(t) for t in transactions],
        "score_hash": hashlib.sha256(
            json.dumps([(s.ticker, s.r_asre) for s in scores], default=str).encode()
        ).hexdigest()[:32],
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)

    console.print(f"  ✓ JSON report: {report_json}", style="green")

    # Audit log
    state_mgr.append_audit(report_data)

    # ──────────────────────────────────────────────────────────────────────────
    # DISPLAY SUMMARY
    # ──────────────────────────────────────────────────────────────────────────

    console.print("\n")

    # Portfolio table
    table = Table(title="Current Portfolio", show_header=True, header_style="bold cyan")
    table.add_column("Ticker", style="cyan", width=18)
    table.add_column("Qty", justify="right", width=10)
    table.add_column("Entry", justify="right", width=10)
    table.add_column("Current", justify="right", width=10)
    table.add_column("Value", justify="right", width=12)
    table.add_column("P&L", justify="right", width=12)
    table.add_column("Days", justify="right", width=6)
    table.add_column("R_ASRE", justify="right", width=7)

    for pos in sorted(state.positions.values(), key=lambda p: p.unrealized_pnl_pct, reverse=True):
        pnl_style = "green" if pos.unrealized_pnl >= 0 else "red"
        table.add_row(
            pos.ticker,
            f"{pos.quantity:.2f}",
            f"₹{pos.entry_price:.2f}",
            f"₹{pos.current_price:.2f}",
            f"₹{pos.current_value:,.0f}",
            f"[{pnl_style}]₹{pos.unrealized_pnl:+,.0f} ({pos.unrealized_pnl_pct:+.1f}%)[/{pnl_style}]",
            str(pos.days_held),
            f"{pos.last_r_asre:.1f}",
        )

    console.print(table)

    # Summary panel
    pnl_style = "green" if state.total_pnl >= 0 else "red"
    console.print()
    console.print(
        Panel.fit(
            f"[bold]Portfolio Value:[/bold] ₹{state.total_value:,.0f}\n"
            f"[bold]P&L:[/bold] [{pnl_style}]₹{state.total_pnl:+,.0f} ({(state.total_pnl/state.capital)*100:+.2f}%)[/{pnl_style}]\n"
            f"\n"
            f"[bold]Actions Today:[/bold]\n"
            f"  🟢 BUY:  {action_counts['BUY']}\n"
            f"  🔴 SELL: {action_counts['SELL']}\n"
            f"  🟡 HOLD: {action_counts['HOLD']}\n"
            f"  👁️  WATCH: {action_counts['WATCH']}",
            border_style="green",
            title="Daily Summary",
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(
        description="ASRE Daily Portfolio Manager — Production Edition v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Operational
    p.add_argument("--init", action="store_true", help="Initialize new portfolio")
    p.add_argument("--dry-run", action="store_true", help="Simulate without executing trades")
    p.add_argument("--mode", default="ia", choices=["ia", "ra"], help="SEBI mode (default: ia)")
    p.add_argument("--state-dir", default="portfolio_state", help="State directory (default: portfolio_state/)")

    # Portfolio parameters
    p.add_argument("--capital", type=float, default=100000, help="Initial capital (default: 100000)")
    p.add_argument("--max-positions", type=int, default=15, help="Max concurrent positions (default: 15)")
    p.add_argument("--max-position-pct", type=float, default=0.15, help="Max position size (default: 0.15 = 15%%)")

    # Thresholds
    p.add_argument("--buy-threshold", type=float, default=70, help="R_ASRE buy threshold (default: 70)")
    p.add_argument("--hold-threshold", type=float, default=55, help="R_ASRE hold threshold (default: 55)")
    p.add_argument("--sell-threshold", type=float, default=45, help="R_ASRE sell threshold (default: 45)")
    p.add_argument("--watch-threshold", type=float, default=60, help="R_ASRE watch threshold (default: 60)")

    # Analysis
    p.add_argument("--lookback-days", type=int, default=730, help="Price history lookback (default: 730)")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        args = parse_args()
        main(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]✗ Fatal error: {e}[/bold red]")
        logger.error(f"Fatal exception:\n{traceback.format_exc()}")
        sys.exit(1)