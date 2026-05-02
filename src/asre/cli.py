"""
ASRE Command-Line Interface (v3.1 - PDF Generator V4.2 Integration)

v3.1 Changes (on top of v3.0):

  PDF-INT-1: _build_pdf_analysis_data() now accepts two new optional params:
             fundamentals_df (pd.DataFrame) and fundamentals_fetch_ts (datetime).
             When present it computes all five derived ratios inline:
               P/E (trailing 4-quarter EPS), ROE, D/E,
               Revenue Growth YoY, Profit Margin.
             It also populates the six raw quarterly fields that V4.2
             renders in the "Additional Data" sub-table on Page 3:
               _eps, _revenue, _net_income, _free_cash_flow,
               _total_debt, _shareholders_equity,
               _announced_date, _quarter_end_date.
             The data_sources['fundamentals']['timestamp'] entry now carries
             the true Yahoo Finance fetch_timestamp from FundamentalFetcher,
             not datetime.now().

  PDF-INT-2: generate_stock_pdf() signature extended with two optional params:
             fundamentals_df and fundamentals_fetch_ts. Both default to None
             so all existing callers continue to work without change.
             They are forwarded straight to _build_pdf_analysis_data().

  PDF-INT-3: In command_compute() and command_compare(), _process() now stores
             fundamentals_df and fundamentals_fetch_ts in each all_results
             entry alongside "df" and "latest". generate_stock_pdf() receives
             them so PDF Page 3 always shows live ratios.

  PDF-INT-4: All imports from report_generator updated to reference
             ASREReportGenerator (V4.2). fetch_fundamentals_for_report is
             imported and re-exported for callers who need the standalone helper.

  Backward compatibility: all existing v3.0 call sites unchanged.
  The two new params are keyword-only with None defaults throughout.

v3.0 Changes (retained):
  FIX 5.1-5.8: Full compliance architecture — see v3.0 docstring.

v2.x Changes (retained):
  FIX 1.1-4.3: IA-safe labels, disclosure, status line, tier injection, etc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from time import sleep

import pandas as pd
import numpy as np

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.prompt import Prompt, Confirm
    from rich.markdown import Markdown
    from rich.columns import Columns
    from rich.syntax import Syntax
    from rich import box
    from rich.tree import Tree
    from rich.align import Align
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("⚠️  Warning: 'rich' library not installed. Install with: pip install rich")
    print("   Using fallback plain text output.\n")

from .config import (
    MomentumConfig,
    TechnicalConfig,
    FundamentalsConfig,
    CompositeConfig,
    BacktestConfig,
)
from .data_loader_indian import load_stock_data
from .composite import compute_complete_asre, validate_asre_rating
from .backtest import Backtester
from .data.fundamental_fetcher import FundamentalFetcher
from .role_gate import RoleGate, RoleGateError, clear_role_lock

# FIX 5.1/5.2/5.3/5.4 — compliance architecture
from .compliance.compliance_filter import ComplianceFilter, OutputMode
from .compliance.status_line        import StatusLineRenderer
from .compliance.disclosure         import DisclosureBlock
from .compliance.hash_ledger        import HashLedger

# FIX 5.8 — theoretical weight prior for audit metadata
from .theory.factor_registry import THEORETICAL_WEIGHT_PRIOR

# PDF-INT-4 — V4.2 generator + standalone ratio helper
from .report_generator import ASREReportGenerator, fetch_fundamentals_for_report

# ---------------------------------------------------------------------------
# Constants & Theme
# ---------------------------------------------------------------------------

APP_VERSION = "3.1.0"

THEME_COLORS = {
    'primary':   'cyan',
    'secondary': 'blue',
    'success':   'green',
    'warning':   'yellow',
    'danger':    'red',
    'info':      'magenta',
    'muted':     'bright_black',
}

SIGNAL_EMOJI = {
    'priority_review': '\U0001f680',
    'positive_outlook': '\U0001f4c8',
    'neutral':          '\u2696\ufe0f',
    'flag_for_review':  '\u26a0\ufe0f',
    'watch':            '\U0001f4c9',
    'urgent_review':    '\U0001f53b',
}

MIN_PRICE_ROWS = 200
SMA_PERIOD     = 200

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FIX 4.1 — Tier interception
# ---------------------------------------------------------------------------

import re as _re
import threading as _threading

_tier_local = _threading.local()

_TIER_PATTERN = _re.compile(
    r'Quality\s+Tier\s*[:\-]\s*([ABCD])', _re.IGNORECASE
)


class TierCapturingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self._captured: Optional[str] = None

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        m   = _TIER_PATTERN.search(msg)
        if m:
            self._captured = m.group(1).upper()
            logger.debug("TierCapturingHandler: captured tier=%s", self._captured)

    def captured(self) -> Optional[str]:
        return self._captured

    def attach(self) -> None:
        self._captured = None
        logging.getLogger().addHandler(self)

    def detach(self) -> None:
        logging.getLogger().removeHandler(self)


def _inject_tier_column(df: pd.DataFrame,
                         captured_tier: Optional[str]) -> pd.DataFrame:
    existing = df.get('quality_tier') if 'quality_tier' in df.columns else None
    already_written = (
        existing is not None
        and not existing.isnull().all()
        and str(existing.iloc[-1]).strip() not in ('', 'nan', 'None', 'C')
    )
    if already_written:
        return df

    if captured_tier:
        df = df.copy()
        df['quality_tier'] = captured_tier
        logger.debug("_inject_tier_column: stamped quality_tier=%s into df", captured_tier)

    return df

if HAS_RICH:
    console = Console()
else:
    console = None


# ---------------------------------------------------------------------------
# FIX 2.3 — Decision Log (audit trail)
# ---------------------------------------------------------------------------

@dataclass
class DecisionLog:
    run_id:       str = field(default_factory=lambda: str(uuid.uuid4()))
    version:      str = APP_VERSION
    mode:         str = "ia"
    command:      str = ""
    tickers:      List[str] = field(default_factory=list)
    start_date:   str = ""
    end_date:     str = ""
    ia_notes:     str = ""
    role_status:  str = "pending"
    role_detail:  str = ""
    started_at:   str = field(
        default_factory=lambda: datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    )
    completed_at: str = ""
    exit_status:  str = "pending"
    error_detail: str = ""
    pdf_exports:  List[str] = field(default_factory=list)
    disclosures_shown: List[str] = field(default_factory=list)
    disclosure_text: str = ""
    score_hash:   str = "pending"
    weight_prior: Dict[str, float] = field(
        default_factory=lambda: dict(THEORETICAL_WEIGHT_PRIOR)
    )

    def approve_role(self, detail: str = ""):
        self.role_status = "approved"
        self.role_detail = detail

    def reject_role(self, detail: str = ""):
        self.role_status = "rejected"
        self.role_detail = detail

    def mark_complete(self, status: str = "success", error: str = ""):
        self.completed_at = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
        self.exit_status  = status
        self.error_detail = error

    def add_disclosure(self, label: str):
        if label not in self.disclosures_shown:
            self.disclosures_shown.append(label)

    def add_pdf(self, path: str):
        self.pdf_exports.append(path)

    def flush(self):
        audit_dir = Path.home() / ".asre" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        log_file = audit_dir / f"{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
        try:
            record = asdict(self)
            with open(log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            logger.debug("Audit record written → %s", log_file)
        except Exception as exc:
            logger.warning("Could not write audit log: %s", exc)


# ---------------------------------------------------------------------------
# FIX 2.2 — Role gate enforcement
# ---------------------------------------------------------------------------

def enforce_role(mode: str, decision_log: DecisionLog) -> None:
    try:
        gate   = RoleGate(mode=mode)
        detail = gate.validate()
        decision_log.approve_role(detail)
        logger.info("[role_gate] Mode '%s' validated — %s", mode, detail)
    except RoleGateError as exc:
        decision_log.reject_role(str(exc))
        decision_log.mark_complete(status="aborted", error=str(exc))
        decision_log.flush()
        print_error(
            f"[ROLE GATE] Access denied for mode '{mode}': {exc}\n"
            f"Register as a SEBI {mode.upper()} or contact your compliance officer."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# FIX 2.4 — Mode / run-id banner
# ---------------------------------------------------------------------------

def print_run_banner(disclosure: "DisclosureBlock"):
    if HAS_RICH and console:
        console.print(Panel(
            disclosure.render_rich_banner(),
            title="[bold white]🔐 Compliance Session[/bold white]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 2),
        ))
        console.print()
    else:
        print(disclosure.render_plain_banner())


# ---------------------------------------------------------------------------
# Setup & Utilities
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    if HAS_RICH:
        logging.basicConfig(
            level=level, format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)]
        )
    else:
        logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def print_header(title: str, subtitle: Optional[str] = None):
    if HAS_RICH and console:
        header_text = f"[bold cyan]ASRE[/bold cyan] [dim]|[/dim] {title}"
        content = f"{header_text}\n[dim]{subtitle}[/dim]" if subtitle else header_text
        console.print(Panel(Align.center(content), border_style="cyan", box=box.DOUBLE, padding=(1, 2)))
        console.print()
    else:
        print(f"\n{'='*70}\nASRE | {title}")
        if subtitle:
            print(subtitle)
        print(f"{'='*70}\n")


def print_success(message: str):
    if HAS_RICH and console:
        console.print(f"✓ [green]{message}[/green]")
    else:
        print(f"✓ {message}")


def print_error(message: str):
    if HAS_RICH and console:
        console.print(f"✗ [red]{message}[/red]")
    else:
        print(f"✗ {message}")


def print_info(message: str):
    if HAS_RICH and console:
        console.print(f"ℹ [cyan]{message}[/cyan]")
    else:
        print(f"ℹ {message}")


def safe_date_format(date_val: Any, default: str = "N/A") -> str:
    if pd.isna(date_val):
        return default
    if isinstance(date_val, (pd.Timestamp, datetime)):
        return date_val.strftime("%Y-%m-%d")
    elif isinstance(date_val, str):
        try:
            return pd.to_datetime(date_val).strftime("%Y-%m-%d")
        except Exception:
            return date_val
    return str(date_val)


# ---------------------------------------------------------------------------
# FIX 1.1 — IA-safe signal labels
# ---------------------------------------------------------------------------

def get_signal_interpretation(rating: float) -> Dict[str, str]:
    if rating is None or (isinstance(rating, float) and pd.isna(rating)):
        return {
            'signal':         'N/A',
            'emoji':          '❓',
            'color':          'dim',
            'interpretation': 'Insufficient data to compute signal.',
        }
    if rating >= 75:
        return {
            'signal':         'PRIORITY REVIEW',
            'emoji':          SIGNAL_EMOJI['priority_review'],
            'color':          'bold green',
            'interpretation': 'Strong multi-factor picture. Candidate for IA priority review.',
        }
    elif rating >= 60:
        return {
            'signal':         'POSITIVE OUTLOOK',
            'emoji':          SIGNAL_EMOJI['positive_outlook'],
            'color':          'green',
            'interpretation': 'Positive momentum and fundamentals. Merits IA attention.',
        }
    elif rating >= 45:
        return {
            'signal':         'NEUTRAL',
            'emoji':          SIGNAL_EMOJI['neutral'],
            'color':          'yellow',
            'interpretation': 'Neutral outlook. Suitable for ongoing IA monitoring.',
        }
    elif rating >= 35:
        return {
            'signal':         'FLAG FOR REVIEW',
            'emoji':          SIGNAL_EMOJI['flag_for_review'],
            'color':          'orange1',
            'interpretation': 'Weakening signals. Flag for IA review and client discussion.',
        }
    elif rating >= 25:
        return {
            'signal':         'WATCH',
            'emoji':          SIGNAL_EMOJI['watch'],
            'color':          'red',
            'interpretation': 'Deteriorating outlook. IA watchlist candidate.',
        }
    else:
        return {
            'signal':         'URGENT REVIEW',
            'emoji':          SIGNAL_EMOJI['urgent_review'],
            'color':          'bold red',
            'interpretation': 'Significant deterioration. Immediate IA review per client mandate.',
        }


# ---------------------------------------------------------------------------
# FIX 1.3 — Status line shim
# ---------------------------------------------------------------------------

def generate_status_line(f: float, t: float, m: float,
                          r_final: float, tier: str,
                          r_asre: Optional[float] = None) -> str:
    """DEPRECATED in v3.0. Delegates to StatusLineRenderer."""
    return StatusLineRenderer.render(
        f=f, t=t, m=m, r_final=r_final, tier=tier, r_asre=r_asre,
    )


# ==============================================================================
# DipAnalysis / DipAnalyzer — unchanged from v3.0
# ==============================================================================

@dataclass
class DipAnalysis:
    is_dip:               bool  = False
    dip_stage:            str   = "NONE"
    dip_quality_score:    float = 0.0
    confidence:           float = 0.0
    distance_from_sma_pct: float = 0.0
    quality_tier:         str   = ""
    approved_for_dip_buy: bool  = False
    entry_signal:         str   = ""
    reason:               str   = ""


class DipAnalyzer:
    """
    Unified Dip Quality Score (Strategy C).
    Score formula: raw = (base_conf + T_bonus + F_adj + M_adj + Div_bonus) x tier_mult
    """

    @staticmethod
    def analyze_dip(
        price:        float,
        sma_200:      float,
        r_final:      float,
        r_asre:       float,
        quality_tier: str,
        f_score:      float,
        t_score:      float,
        m_score:      float,
    ) -> DipAnalysis:
        analysis = DipAnalysis()
        distance = ((price - sma_200) / sma_200) * 100
        analysis.distance_from_sma_pct = distance
        analysis.quality_tier = quality_tier

        if distance >= 0:
            analysis.is_dip    = False
            analysis.dip_stage = "NONE"
            analysis.reason    = ComplianceFilter(mode=OutputMode.IA).apply(
                f"Price above long-term average (+{distance:.1f}%) -- no deviation"
            )
            return analysis

        if distance < -25:
            _cf_sb = ComplianceFilter(mode=OutputMode.IA)
            analysis.is_dip               = True
            analysis.dip_stage            = "STRUCTURAL BREAK"
            analysis.approved_for_dip_buy = False
            analysis.entry_signal         = _cf_sb.apply("❌ STRUCTURAL BREAK")
            analysis.reason               = _cf_sb.apply(
                f"Too deep ({distance:.1f}%) -- possible structural breakdown. "
                "Avoid until recovery pattern confirmed."
            )
            return analysis

        analysis.is_dip = True

        if   distance > -5:  analysis.dip_stage = "EARLY"; base_conf = 80
        elif distance > -10: analysis.dip_stage = "MID";   base_conf = 70
        elif distance > -15: analysis.dip_stage = "LATE";  base_conf = 55
        else:                analysis.dip_stage = "DEEP";  base_conf = 40

        if   t_score <= 10: t_bonus = +20
        elif t_score <= 20: t_bonus = +10
        elif t_score <= 35: t_bonus =   0
        else:               t_bonus = -15

        tier_mult = {'A': 1.15, 'B': 1.05, 'C': 0.90, 'D': 0.70}.get(quality_tier, 0.85)

        if   f_score >= 65: f_adj = +15
        elif f_score >= 55: f_adj =  +8
        elif f_score >= 45: f_adj =   0
        elif f_score >= 35: f_adj = -10
        else:               f_adj = -20

        if   m_score >= 50: m_adj = +10
        elif m_score >= 40: m_adj =  +5
        elif m_score >= 30: m_adj =   0
        else:               m_adj = -15

        gap = abs(r_final - r_asre)
        if   gap >= 30 and r_asre > r_final: divergence_bonus = +8
        elif gap >= 20 and r_asre > r_final: divergence_bonus = +4
        else:                                divergence_bonus =  0

        raw   = (base_conf + t_bonus + f_adj + m_adj + divergence_bonus) * tier_mult
        score = max(0.0, min(100.0, raw))

        cap_notes: List[str] = []
        if t_score > 15:
            score = min(score, 65.0)
            cap_notes.append(f"T={t_score:.0f}>15 -> max GOOD")

        if quality_tier in ['C', 'D'] and analysis.dip_stage in ['EARLY', 'MID'] and t_score > 10:
            score = min(score, 65.0)
            cap_notes.append(f"Tier {quality_tier}+{analysis.dip_stage} -> max GOOD")

        analysis.dip_quality_score = round(score, 1)
        analysis.confidence        = round(base_conf + t_bonus, 1)
        cap_str = f" [{', '.join(cap_notes)}]" if cap_notes else ""

        if score >= 75:
            analysis.approved_for_dip_buy = True
            analysis.entry_signal         = "🎯 HIGH QUALITY DIP"
            analysis.reason = (
                f"✅ DIP APPROVED: {analysis.dip_stage} stage, "
                f"score={score:.0f}/100, conf={analysis.confidence:.0f}%, "
                f"Tier {quality_tier}"
            )
        elif score >= 60:
            analysis.approved_for_dip_buy = True
            analysis.entry_signal         = "📈 GOOD DIP"
            analysis.reason = (
                f"✅ DIP APPROVED: {analysis.dip_stage} stage, "
                f"score={score:.0f}/100 (GOOD), Tier {quality_tier}{cap_str}"
            )
        elif score >= 45:
            analysis.approved_for_dip_buy = False
            analysis.entry_signal         = "⚖️ MARGINAL"
            analysis.reason = f"⚠️ MARGINAL: score={score:.0f}/100 -- wait for M or T improvement{cap_str}"
        else:
            analysis.approved_for_dip_buy = False
            analysis.entry_signal         = "❌ POOR DIP"
            analysis.reason = f"❌ REJECTED: score={score:.0f}/100 -- insufficient quality{cap_str}"

        _cf = ComplianceFilter(mode=OutputMode.IA)
        analysis.entry_signal = _cf.apply(analysis.entry_signal)
        analysis.reason       = _cf.apply(analysis.reason)

        return analysis


# ==============================================================================
# SMA-200 helpers — unchanged from v3.0
# ==============================================================================

def _compute_sma200_from_df(df: pd.DataFrame) -> Optional[Tuple[float, float]]:
    price_cols = ['close', 'Close', 'Adj Close', 'adj_close', 'adjusted_close', 'price', 'Price']
    sma_cols   = ['sma_200', 'sma200', 'SMA_200', 'SMA200', 'technical_sma_200',
                  'tech_sma_200', 'sma_200_day', '200_sma', 'ma_200', 'ma200']

    price_col = next((c for c in price_cols if c in df.columns), None)
    if price_col is None:
        return None

    price_series = df[price_col].dropna()
    if len(price_series) == 0:
        return None

    latest_price = float(price_series.iloc[-1])

    sma_col = next((c for c in sma_cols if c in df.columns), None)
    if sma_col is not None:
        sma_series = df[sma_col].dropna()
        if len(sma_series) > 0 and pd.notna(sma_series.iloc[-1]):
            return (latest_price, float(sma_series.iloc[-1]))

    if len(price_series) >= SMA_PERIOD:
        sma_200 = float(price_series.rolling(SMA_PERIOD).mean().iloc[-1])
        if pd.notna(sma_200) and sma_200 > 0:
            return (latest_price, sma_200)

    if len(price_series) >= 50:
        sma_period_used = len(price_series)
        sma_approx = float(price_series.rolling(sma_period_used).mean().iloc[-1])
        if pd.notna(sma_approx) and sma_approx > 0:
            logger.debug(
                "SMA-200 not available -- using SMA-%d as proxy (%d rows available)",
                sma_period_used, len(price_series)
            )
            return (latest_price, sma_approx)

    return None


def run_dip_analyzer(latest: pd.Series,
                     df: Optional[pd.DataFrame] = None) -> Optional[DipAnalysis]:
    r_final      = float(latest.get('r_final',      50) or 50)
    r_asre       = float(latest.get('r_asre',        50) or 50)
    quality_tier = str(latest.get('quality_tier',  'C') or 'C')
    f_score      = float(latest.get('f_score',       50) or 50)
    t_score      = float(latest.get('t_score',       50) or 50)
    m_score      = float(latest.get('m_score',       50) or 50)

    if any(not pd.notna(v) for v in [f_score, t_score, m_score]):
        return None

    price_sma: Optional[Tuple[float, float]] = None

    sma_cols   = ['sma_200', 'sma200', 'SMA_200', 'SMA200', 'technical_sma_200',
                  'tech_sma_200', 'sma_200_day', '200_sma', 'ma_200', 'ma200']
    price_cols = ['close', 'Close', 'Adj Close', 'adj_close', 'adjusted_close', 'price', 'Price']

    price_val = next(
        (latest.get(c) for c in price_cols
         if latest.get(c) is not None and pd.notna(latest.get(c))), None)
    sma_val = next(
        (latest.get(c) for c in sma_cols
         if latest.get(c) is not None and pd.notna(latest.get(c))), None)

    if price_val is not None and sma_val is not None and float(sma_val) > 0:
        price_sma = (float(price_val), float(sma_val))

    if price_sma is None and df is not None and len(df) >= 50:
        price_sma = _compute_sma200_from_df(df)

    if price_sma is None:
        return None

    price, sma_200 = price_sma
    if sma_200 <= 0:
        return None

    return DipAnalyzer.analyze_dip(
        price=price, sma_200=sma_200,
        r_final=r_final, r_asre=r_asre,
        quality_tier=quality_tier,
        f_score=f_score, t_score=t_score, m_score=m_score,
    )


# ==============================================================================
# FIX 1.1 — detect_scenario
# ==============================================================================

def detect_scenario(f_score: float, t_score: float, m_score: float) -> Dict[str, Any]:
    scores   = [x for x in [f_score, t_score, m_score] if pd.notna(x)]
    variance = np.std(scores) if len(scores) >= 2 else 0.0
    f = f_score if pd.notna(f_score) else 50.0
    t = t_score if pd.notna(t_score) else 50.0
    m = m_score if pd.notna(m_score) else 50.0

    if f >= 55 and t <= 20:
        return {'scenario': 'DIP OPPORTUNITY',  'emoji': '🎯', 'color': 'bold green',
                'description': f'Quality fundamentals ({f:.0f}%) at oversold levels (T={t:.0f}%). IA review warranted.'}
    if f <= 40 and t >= 80:
        return {'scenario': 'DIVERGENCE ALERT', 'emoji': '⚠️', 'color': 'bold red',
                'description': f'Weak fundamentals ({f:.0f}%) with overbought price (T={t:.0f}%). Flag for IA review.'}
    if t <= 15 and m >= 40:
        return {'scenario': 'OVERSOLD RECOVERY','emoji': '🔄', 'color': 'cyan',
                'description': (f'Price near floor (T={t:.0f}%) -- momentum recovering (M={m:.0f}%). '
                                'R_ASRE favours entry; R_Final checks conviction.')}
    if t <= 15 and m < 40:
        return {'scenario': 'LATE-STAGE DIP',   'emoji': '⛔', 'color': 'bold red',
                'description': (f'Deeply oversold (T={t:.0f}%) -- no momentum catalyst (M={m:.0f}%). '
                                'Await M recovery before IA action.')}
    if t >= 80 and m < 50:
        return {'scenario': 'OVERBOUGHT FADE',  'emoji': '📉', 'color': 'orange1',
                'description': (f'Technically extended (T={t:.0f}%) -- momentum fading (M={m:.0f}%). '
                                'Monitor for IA risk-management action.')}
    if variance < 15:
        return {'scenario': 'BALANCED',         'emoji': '⚖️', 'color': 'cyan',
                'description': f'Components aligned (F={f:.0f}%, T={t:.0f}%, M={m:.0f}%)'}
    return {'scenario': 'DIVERGENT',            'emoji': '📊', 'color': 'yellow',
            'description': f'High divergence -- sigma={variance:.1f} (F={f:.0f}%, T={t:.0f}%, M={m:.0f}%)'}


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> Dict:
    if config_path is None:
        return {}
    config_file = Path(config_path)
    if not config_file.exists():
        print_error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        print_success(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        print_error(f"Failed to load configuration: {e}")
        sys.exit(1)


def create_configs_from_dict(config_dict: Dict) -> Dict:
    configs = {}
    if 'momentum'     in config_dict: configs['momentum']     = MomentumConfig(**config_dict['momentum'])
    if 'technical'    in config_dict: configs['technical']    = TechnicalConfig(**config_dict['technical'])
    if 'fundamentals' in config_dict: configs['fundamentals'] = FundamentalsConfig(**config_dict['fundamentals'])
    if 'composite'    in config_dict: configs['composite']    = CompositeConfig(**config_dict['composite'])
    if 'backtest'     in config_dict: configs['backtest']     = BacktestConfig(**config_dict['backtest'])
    return configs


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def fetch_fundamentals(
    ticker:     str,
    start_date: str,
    end_date:   str,
) -> Tuple[Optional[pd.DataFrame], Optional[datetime]]:
    try:
        fetcher = FundamentalFetcher()
        end_dt  = pd.to_datetime(end_date)
        fundamental_start = (end_dt - pd.DateOffset(months=18)).strftime("%Y-%m-%d")

        fundamentals, fetch_ts = fetcher.fetch_quarterly_fundamentals(
            ticker, start_date=fundamental_start, end_date=end_date)

        if fundamentals is None or fundamentals.empty:
            logger.error("%s: FundamentalFetcher returned empty data.", ticker.upper())
            return None, None

        if len(fundamentals) < 4:
            logger.error(
                "%s: Only %d quarters available -- minimum 4 required. Skipping.",
                ticker.upper(), len(fundamentals))
            return None, None

        logger.info("%s: Fetched %d quarters of live fundamentals.",
                    ticker.upper(), len(fundamentals))
        return fundamentals, fetch_ts

    except RuntimeError as exc:
        logger.error("%s: Stale data abort -- %s", ticker.upper(), exc)
        return None, None
    except Exception as exc:
        logger.error("%s: FundamentalFetcher failed -- %s.", ticker.upper(), exc)
        return None, None


def load_price_data_guarded(
    ticker:               str,
    start_str:            str,
    end_str:              str,
    fundamentals:         pd.DataFrame,
    fundamentals_fetch_ts: Optional[datetime] = None,
) -> pd.DataFrame:
    df = load_stock_data(
        ticker, start_str, end_str,
        quarterly_fundamentals=fundamentals,
        fundamentals_fetch_ts=fundamentals_fetch_ts,
    )
    if len(df) < MIN_PRICE_ROWS:
        extended_start = (pd.to_datetime(end_str) - timedelta(days=365)).strftime("%Y-%m-%d")
        logger.warning(
            "%s: Only %d rows from %s. Auto-extending to %s.",
            ticker.upper(), len(df), start_str, extended_start,
        )
        if HAS_RICH and console:
            console.print(
                f"  [yellow]⚠ Short window ({len(df)} rows) — "
                f"auto-extending to {extended_start} for reliable scores.[/yellow]"
            )
        df = load_stock_data(
            ticker, extended_start, end_str,
            quarterly_fundamentals=fundamentals,
            fundamentals_fetch_ts=fundamentals_fetch_ts,
        )
        logger.info(
            "%s: Extended to %d rows (%s to %s).",
            ticker.upper(), len(df), extended_start, end_str,
        )

    if len(df) < MIN_PRICE_ROWS:
        raise ValueError(
            f"{ticker.upper()}: Only {len(df)} rows available after auto-extension. "
            f"Minimum {MIN_PRICE_ROWS} rows required for reliable M/T scores. "
            f"Use --start with an earlier date (e.g. 2024-01-01)."
        )

    return df


# ==============================================================================
# PDF-INT-1 — _compute_ratios_from_fundamentals_df
# ==============================================================================

def _compute_ratios_from_fundamentals_df(
    fundamentals_df:  pd.DataFrame,
    fetch_ts:         Optional[datetime],
    price:            Optional[float] = None,
    ticker:           Optional[str]   = None,
) -> Dict:
    """
    Compute the five derived ratios + six raw quarterly fields that
    pdf_generator V4.2 renders on Page 3 of the single-stock report.

    Strategy (three-layer):
      1. If FundamentalFetcher already stored pre-computed ratio columns
         (roe, de, pe, etc.) in the returned df, use them directly.
      2. Otherwise derive from raw columns, trying multiple Yahoo Finance
         column name aliases (yfinance naming is inconsistent across
         tickers and package versions).
      3. Fall back gracefully — any missing ratio is omitted from result
         so the PDF shows "N/A" rather than crashing.

    Returns a dict whose keys are the exact ones _build_pdf_analysis_data
    merges into analysis_data:
        pe, roe, de, revenue_growth, profit_margin
        _eps, _revenue, _net_income, _free_cash_flow,
        _total_debt, _shareholders_equity,
        _announced_date, _quarter_end_date
        data_sources  (with 'fundamentals' sub-dict)
    """
    result: Dict = {}

    if fundamentals_df is None or fundamentals_df.empty:
        logger.warning("[v3.1] _compute_ratios: fundamentals_df is empty — "
                       "PDF Page 3 will show 'Not available'")
        result['data_sources'] = {
            'fundamentals': {
                'source':    'Yahoo Finance (Quarterly) via FundamentalFetcher v2.0',
                'timestamp': (fetch_ts.strftime('%Y-%m-%d %H:%M IST')
                              if fetch_ts else datetime.now().strftime('%Y-%m-%d %H:%M IST'))
                              + ' (no data)',
            }
        }
        return result

    # Sort ascending so iloc[-1] is always the most recent quarter
    df = fundamentals_df.copy()
    if 'date' in df.columns:
        df = df.sort_values('date', ascending=True).reset_index(drop=True)

    latest = df.iloc[-1]
    prev   = df.iloc[-5] if len(df) >= 5 else None   # same quarter prior year
    cols   = set(df.columns.str.lower())  # for fast membership tests

    # Log available columns once at DEBUG level to aid future diagnostics
    logger.debug("[v3.1] FF df columns: %s", sorted(df.columns.tolist()))

    # ── Column alias resolvers ───────────────────────────────
    def _col(row: pd.Series, *aliases) -> Optional[float]:
        """Return the first non-null, non-zero value from the alias list (single row)."""
        for alias in aliases:
            val = row.get(alias)
            if val is not None and pd.notna(val):
                try:
                    fval = float(val)
                    if fval != 0.0:
                        return fval
                except (TypeError, ValueError):
                    pass
        return None

    def _col_df(frame: pd.DataFrame, *aliases) -> Optional[pd.Series]:
        """Return the first matching Series from the alias list."""
        for alias in aliases:
            if alias in frame.columns:
                return frame[alias]
        return None

    def _col_scan(frame: pd.DataFrame, *aliases) -> Optional[float]:
        """
        Scan the full DataFrame for the most recent non-null, non-zero value.

        Balance sheet columns (shareholders_equity, total_debt) are often NaN
        in the latest income-statement quarter because yfinance mixes income
        statement and balance sheet data on different reporting schedules.
        This function looks backwards through all rows so we always use the
        most recently reported value even if it's one or two quarters old.
        """
        for alias in aliases:
            if alias not in frame.columns:
                continue
            series = frame[alias].dropna()
            # Filter zeros separately so we don't discard legitimate NaN-drops
            series = series[series != 0.0]
            if len(series) > 0:
                try:
                    return float(series.iloc[-1])
                except (TypeError, ValueError):
                    pass
        return None

    # Staleness flag
    hours_old  = (datetime.now() - fetch_ts).total_seconds() / 3600 if fetch_ts else 0
    stale_note = ' [STALE >24h — re-fetch recommended]' if hours_old > 24 else ''
    ts_str     = (fetch_ts.strftime('%Y-%m-%d %H:%M IST') if fetch_ts
                  else datetime.now().strftime('%Y-%m-%d %H:%M IST')) + stale_note

    # ── Layer 1: Use FF pre-computed ratio columns if present ─
    # FundamentalFetcher logs "Calculated derived metrics: PE, ROE, D/E, Growth"
    # meaning it may store them as lowercase columns in the returned df.
    _precomp_roe = _col(latest,
        'roe', 'return_on_equity', 'return_on_equity_pct',
        'roe_pct', 'annualized_roe')
    if _precomp_roe is not None:
        result['roe'] = round(_precomp_roe, 2)

    _precomp_de = _col(latest,
        'de', 'd_e', 'debt_equity', 'debt_to_equity',
        'debt_equity_ratio', 'leverage')
    if _precomp_de is not None:
        result['de'] = round(_precomp_de, 2)

    _precomp_pe = _col(latest,
        'pe', 'p_e', 'pe_ratio', 'price_earnings',
        'trailing_pe', 'pe_ttm')
    if _precomp_pe is not None and price is not None:
        # Accept pre-computed PE only if plausible (0 < PE < 500).
        # Values outside this range are almost certainly data artefacts.
        if 0.0 < _precomp_pe < 500:
            result['pe'] = round(_precomp_pe, 2)

    _precomp_rev_growth = _col(latest,
        'revenue_growth', 'revenue_growth_yoy', 'rev_growth',
        'revenue_growth_pct', 'yoy_revenue_growth')
    if _precomp_rev_growth is not None:
        result['revenue_growth'] = round(_precomp_rev_growth, 2)

    _precomp_margin = _col(latest,
        'profit_margin', 'net_margin', 'net_profit_margin',
        'profit_margin_pct', 'margin')
    if _precomp_margin is not None:
        result['profit_margin'] = round(_precomp_margin, 2)

    # ── Layer 2: Derive from raw columns (with alias fallbacks) ─

    # Equity + Debt — use _col_scan (full-df backward scan) because yfinance
    # often puts balance-sheet fields as NaN in the latest income-statement row.
    # _col_scan returns the most recently reported non-null value across all rows.
    equity_val = _col_scan(df,
        'shareholders_equity', 'stockholders_equity',
        'total_stockholder_equity', 'total_equity',
        'common_stock_equity', 'equity',
        'stockholders_equity_total',
        'total_equity_gross_minority_interest')

    debt_val = _col_scan(df,
        'total_debt', 'long_term_debt',
        'total_long_term_debt',
        'long_term_debt_and_capital_lease_obligation',
        'current_debt', 'short_long_term_debt_total')

    # Trailing P/E from EPS (if not already set from pre-computed)
    if 'pe' not in result and price is not None and len(df) >= 4:
        try:
            eps_series = _col_df(df, 'eps', 'basic_eps', 'diluted_eps', 'earnings_per_share')
            if eps_series is not None:
                trailing_eps = float(eps_series.tail(4).sum())
                if trailing_eps > 0:
                    raw_pe = price / trailing_eps
                    # Sanity clamp: PE > 500 is almost certainly a data artefact
                    # (e.g. one near-zero EPS quarter distorting the TTM sum).
                    # Cap at 500 and log a warning so it's visible but not fatal.
                    if raw_pe > 500:
                        logger.warning(
                            "[v3.1] Computed PE=%.1f is unreliable "
                            "(trailing_eps=%.4f near zero). Clamping to 500.",
                            raw_pe, trailing_eps,
                        )
                        raw_pe = 500.0
                    result['pe'] = round(raw_pe, 2)
        except Exception as exc:
            logger.debug("[v3.1] P/E derivation skipped: %s", exc)

    # ROE (if not already set from pre-computed)
    if 'roe' not in result and equity_val is not None:
        try:
            ni_series = _col_df(df, 'net_income', 'net_income_common_stockholders',
                                'net_income_including_noncontrolling_interests',
                                'net_income_from_continuing_operations')
            if ni_series is not None:
                annual_ni = float(ni_series.tail(4).sum())
                result['roe'] = round((annual_ni / equity_val) * 100, 2)
        except Exception as exc:
            logger.debug("[v3.1] ROE derivation skipped: %s", exc)

    # D/E (if not already set from pre-computed)
    if 'de' not in result and equity_val is not None and debt_val is not None:
        try:
            result['de'] = round(debt_val / equity_val, 2)
        except Exception as exc:
            logger.debug("[v3.1] D/E derivation skipped: %s", exc)

    # Revenue Growth YoY (if not already set)
    if 'revenue_growth' not in result:
        try:
            rev_series = _col_df(df, 'revenue', 'total_revenue',
                                 'revenues', 'net_revenues', 'operating_revenue')
            if rev_series is not None and prev is not None:
                prev_rev_series = _col_df(df.iloc[[-5]] if len(df) >= 5 else df.iloc[[-1]],
                                          'revenue', 'total_revenue',
                                          'revenues', 'net_revenues', 'operating_revenue')
                if prev_rev_series is not None:
                    rev_now  = float(rev_series.iloc[-1])
                    rev_prev = float(prev_rev_series.iloc[-1])
                    if rev_prev > 0:
                        result['revenue_growth'] = round(
                            ((rev_now - rev_prev) / rev_prev) * 100, 2
                        )
        except Exception as exc:
            logger.debug("[v3.1] Revenue growth derivation skipped: %s", exc)

    # Profit Margin (if not already set)
    if 'profit_margin' not in result:
        try:
            rev_val = _col(latest, 'revenue', 'total_revenue',
                           'revenues', 'net_revenues', 'operating_revenue')
            ni_val  = _col(latest, 'net_income', 'net_income_common_stockholders',
                           'net_income_including_noncontrolling_interests')
            if rev_val and ni_val:
                result['profit_margin'] = round((ni_val / rev_val) * 100, 2)
        except Exception as exc:
            logger.debug("[v3.1] Profit margin derivation skipped: %s", exc)

    # ── Raw quarterly fields (Additional Data table on Page 3) ─
    try:
        result['_eps'] = float(_col(latest,
            'eps', 'basic_eps', 'diluted_eps', 'earnings_per_share') or 0)
        result['_revenue'] = float(_col(latest,
            'revenue', 'total_revenue', 'revenues', 'net_revenues') or 0)
        result['_net_income'] = float(_col(latest,
            'net_income', 'net_income_common_stockholders') or 0)
        result['_free_cash_flow'] = float(_col(latest,
            'free_cash_flow', 'free_cash_flow_to_firm', 'fcf') or 0)
        result['_total_debt'] = float(debt_val or 0)
        result['_shareholders_equity'] = float(equity_val or 0)
        # Note: equity_val and debt_val already use _col_scan (most recent
        # non-null from full df), so these raw fields are also best-available.
        result['_announced_date']    = str(latest.get('announced_date', 'N/A'))
        result['_quarter_end_date']  = str(latest.get('date', 'N/A'))
    except Exception as exc:
        logger.debug("[v3.1] Raw fields extraction failed: %s", exc)

    # ── Layer 3: yf.Ticker().info fallback for ROE / D/E / PE ───
    # FF computes ROE and D/E via yf.info internally but does NOT store them
    # back in the quarterly DataFrame.  For tickers like IOC.NS the quarterly
    # df has all-zero balance sheet columns (yfinance income-statement endpoint
    # doesn't carry balance-sheet data for PSU/energy names).
    # Trigger when: ROE or D/E are missing, OR our derived PE is absurd (>200).
    _needs_yf_info = ('roe' not in result or 'de' not in result
                      or result.get('pe', 0) > 200)
    if _needs_yf_info and ticker:
        try:
            import yfinance as _yf
            _info = _yf.Ticker(ticker).info or {}
            # ── PE ──────────────────────────────────────────────
            # Replace our derived PE if it's unreliable (>200) or missing.
            _yf_pe = _info.get('trailingPE') or _info.get('forwardPE')
            if _yf_pe is not None and pd.notna(_yf_pe):
                _yf_pe_f = float(_yf_pe)
                if 0 < _yf_pe_f < 200:
                    if 'pe' not in result or result.get('pe', 0) > 200:
                        result['pe'] = round(_yf_pe_f, 2)
                        logger.info("[v3.1] PE from yf.info: %.2f", result['pe'])

            # ── ROE ─────────────────────────────────────────────
            _yf_roe = _info.get('returnOnEquity')
            if 'roe' not in result and _yf_roe is not None and pd.notna(_yf_roe):
                # yf.info returnOnEquity is a decimal fraction (0.1262 = 12.62%)
                result['roe'] = round(float(_yf_roe) * 100, 2)
                logger.info("[v3.1] ROE from yf.info: %.2f%%", result['roe'])

            # ── D/E ─────────────────────────────────────────────
            _yf_de = _info.get('debtToEquity')
            if 'de' not in result and _yf_de is not None and pd.notna(_yf_de):
                # yf.info debtToEquity: some versions return percent (74.0),
                # others return the ratio (0.74). Normalise: if > 10 → divide by 100.
                _de_raw = float(_yf_de)
                result['de'] = round(_de_raw / 100 if _de_raw > 10 else _de_raw, 2)
                logger.info("[v3.1] D/E from yf.info: %.2f", result['de'])

            # ── Equity raw field (book value × shares) ──────────
            _yf_bv     = _info.get('bookValue')
            _yf_shares = (_info.get('sharesOutstanding')
                          or _info.get('impliedSharesOutstanding'))
            if result.get('_shareholders_equity', 0) == 0 and _yf_bv and _yf_shares:
                result['_shareholders_equity'] = round(
                    float(_yf_bv) * float(_yf_shares), 0)

        except Exception as _yf_exc:
            logger.debug("[v3.1] yf.info fallback failed: %s", _yf_exc)

    # ── data_sources entry ───────────────────────────────────
    result['data_sources'] = {
        'fundamentals': {
            'source':    'Yahoo Finance (Quarterly) via FundamentalFetcher v2.0',
            'timestamp': ts_str,
        }
    }

    # Summarise what was resolved vs. missing (INFO so it always appears)
    resolved  = [k for k in ('pe','roe','de','revenue_growth','profit_margin') if k in result]
    missing   = [k for k in ('pe','roe','de','revenue_growth','profit_margin') if k not in result]
    logger.info(
        "[v3.1] PDF ratios — %s | Quarter: %s | Announced: %s%s",
        "  ".join(f"{k}={result[k]:.2f}" for k in resolved),
        result.get('_quarter_end_date', '?'),
        result.get('_announced_date',   '?'),
        f"  | MISSING: {missing}" if missing else "",
    )
    if missing:
        logger.info(
            "[v3.1] Missing ratios diagnostics — equity_col=%s  debt_col=%s  "
            "Available FF columns: %s",
            equity_val, debt_val,
            [c for c in df.columns if any(
                kw in c.lower() for kw in
                ('equity','debt','share','roe','de','net_income','revenue')
            )],
        )

    return result


# ==============================================================================
# FIX 1.5 — PDF export helpers (PDF-INT-2 updates generate_stock_pdf)
# ==============================================================================

def _build_pdf_analysis_data(
    ticker:              str,
    df:                  pd.DataFrame,
    latest:              pd.Series,
    config_dict:         Dict,
    fundamentals_df:     Optional[pd.DataFrame] = None,   # PDF-INT-1
    fundamentals_fetch_ts: Optional[datetime]   = None,   # PDF-INT-1
) -> Dict:
    """
    Build the analysis_data dict consumed by ASREReportGenerator.

    PDF-INT-1: When fundamentals_df is supplied (the already-fetched quarterly
    DataFrame from FundamentalFetcher), derived ratios (P/E, ROE, D/E,
    Revenue Growth, Profit Margin) and raw quarterly figures are computed
    inline and merged into analysis_data.  The data_sources['fundamentals']
    timestamp reflects the true Yahoo Finance fetch time, not now().

    All existing fields are unchanged.
    """
    df_flat  = df.reset_index()
    date_col = next(
        (c for c in df_flat.columns
         if c.lower() in ('date', 'index', 'datetime')), None
    )

    if date_col and date_col in df_flat.columns:
        latest_date_raw = df_flat[date_col].iloc[-1]
        try:
            latest_date_str = pd.to_datetime(latest_date_raw).strftime('%Y-%m-%d')
        except Exception:
            latest_date_str = str(latest_date_raw)
    elif isinstance(latest.name, (pd.Timestamp, datetime)):
        latest_date_str = latest.name.strftime('%Y-%m-%d')
    else:
        latest_date_str = datetime.now().strftime('%Y-%m-%d')

    fund_cfg = config_dict.get('fundamentals', {})

    tier_val = str(
        latest.get('quality_tier')
        or latest.get('tier')
        or latest.get('stock_tier')
        or latest.get('classification_tier')
        or latest.get('asre_tier')
        or fund_cfg.get('tier', 'C')
    )

    mctx = str(latest.get('market_context', '') or '')
    if 'DIP' in mctx.upper():
        category_val = 'DIP'
    elif 'PUMP' in mctx.upper() or 'DIVERGENCE' in mctx.upper():
        category_val = 'DIVERGENCE'
    else:
        category_val = fund_cfg.get('category', 'STABLE')

    # ── Base analysis_data ───────────────────────────────────────────────
    analysis_data: Dict = {
        'date':     latest_date_str,
        'f_score':  float(latest.get('f_score',  50) or 50),
        't_score':  float(latest.get('t_score',  50) or 50),
        'm_score':  float(latest.get('m_score',  50) or 50),
        'r_final':  float(latest.get('r_final',  50) or 50),
        'r_asre':   float(latest.get('r_asre',   50) or 50),
        'signal':   str(latest.get('signal', get_signal_interpretation(
                        float(latest.get('r_asre', 50) or 50))['signal'])),
        'tier':     tier_val,
        'category': category_val,
        'f_weight': float(latest.get('weight_f', latest.get('f_weight', 0.40)) or 0.40),
        't_weight': float(latest.get('weight_t', latest.get('t_weight', 0.30)) or 0.30),
        'm_weight': float(latest.get('weight_m', latest.get('m_weight', 0.30)) or 0.30),
        # Placeholder timestamps — overwritten below if fundamentals_df supplied
        'data_sources': {
            'fundamentals': {
                'source':    'Yahoo Finance (Quarterly) via FundamentalFetcher v2.0',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M IST'),
            },
            'prices': {
                'source':    'Yahoo Finance (Daily)',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M IST'),
            },
            'benchmark': {
                'source':    'Yahoo Finance (^NSEI / sector)',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M IST'),
            },
        },
    }

    # ── Current price ────────────────────────────────────────────────────
    price_cols = ['close', 'Close', 'Adj Close', 'adj_close', 'price', 'Price']
    for pc in price_cols:
        if pc in latest and pd.notna(latest[pc]):
            analysis_data['price'] = float(latest[pc])
            break

    # ── PDF-INT-1: Merge derived ratios from fundamentals_df ─────────────
    if fundamentals_df is not None and not fundamentals_df.empty:
        ratios = _compute_ratios_from_fundamentals_df(
            fundamentals_df  = fundamentals_df,
            fetch_ts         = fundamentals_fetch_ts,
            price            = analysis_data.get('price'),
            ticker           = ticker,
        )
        # Merge ratio fields (pe, roe, de, revenue_growth, profit_margin)
        # and raw quarterly fields (_eps, _revenue, etc.)
        _ratio_keys = {'pe', 'roe', 'de', 'revenue_growth', 'profit_margin'}
        _raw_keys   = {
            '_eps', '_revenue', '_net_income', '_free_cash_flow',
            '_total_debt', '_shareholders_equity',
            '_announced_date', '_quarter_end_date',
        }
        for key, val in ratios.items():
            if key in _ratio_keys or key in _raw_keys:
                analysis_data[key] = val
        # Overwrite data_sources['fundamentals'] with FF timestamp
        if 'data_sources' in ratios:
            analysis_data['data_sources'].update(ratios['data_sources'])
    else:
        logger.warning(
            "[v3.1] %s: fundamentals_df not supplied to _build_pdf_analysis_data — "
            "Page 3 fundamental metrics will show 'Not available'. "
            "Pass fundamentals_df to generate_stock_pdf() to fix this.",
            ticker,
        )

    # ── Price history ────────────────────────────────────────────────────
    close_col = next((c for c in price_cols if c in df_flat.columns), None)
    sma_col   = next((c for c in ['sma_200', 'sma200', 'SMA_200', 'SMA200']
                      if c in df_flat.columns), None)
    if date_col and close_col:
        ph = df_flat[[date_col, close_col]].tail(120).copy()
        ph = ph.rename(columns={date_col: 'date', close_col: 'close'})
        if sma_col:
            ph['sma_200'] = df_flat[sma_col].tail(120).values
        analysis_data['price_history'] = ph

    # ── Dip Quality ──────────────────────────────────────────────────────
    dip = run_dip_analyzer(latest, df=df)
    if dip is not None and dip.is_dip and dip.dip_stage not in ('NONE', 'STRUCTURAL BREAK'):
        analysis_data['market_context'] = {
            'stage':       dip.dip_stage,
            'sma200_dist': dip.distance_from_sma_pct,
            'score':       dip.dip_quality_score,
            'confidence':  dip.confidence,
        }
        _cf_pdf = ComplianceFilter(mode=OutputMode.IA)
        analysis_data['dip_quality'] = {
            'score':       dip.dip_quality_score,
            'stage':       _cf_pdf.apply(dip.dip_stage),
            'assessment':  _cf_pdf.apply(dip.entry_signal),
            'position_size': _cf_pdf.apply(
                "100% of planned position (HIGH QUALITY)" if dip.dip_quality_score >= 75
                else "50-75% of planned position (GOOD)"  if dip.dip_quality_score >= 60
                else "25% max -- wait for confirmation"
            ),
        }

    return analysis_data


def generate_stock_pdf(
    ticker:               str,
    df:                   pd.DataFrame,
    latest:               pd.Series,
    output_dir:           str,
    ia_notes:             Optional[str]       = None,
    config:               Optional[Dict]      = None,
    fundamentals_df:      Optional[pd.DataFrame] = None,  # PDF-INT-2
    fundamentals_fetch_ts: Optional[datetime] = None,     # PDF-INT-2
) -> Path:
    """
    Generate a single-stock PDF using ASREReportGenerator V4.2.

    PDF-INT-2: Pass fundamentals_df and fundamentals_fetch_ts (obtained from
    fetch_fundamentals() earlier in the pipeline) to populate Page 3 with
    live ratios sourced from FundamentalFetcher.  Both params default to None
    for backward compatibility with existing call sites.
    """
    config = config or {}
    analysis_data = _build_pdf_analysis_data(
        ticker               = ticker,
        df                   = df,
        latest               = latest,
        config_dict          = config,
        fundamentals_df      = fundamentals_df,       # PDF-INT-2
        fundamentals_fetch_ts= fundamentals_fetch_ts, # PDF-INT-2
    )

    safe_ticker = ticker.replace('.', '_')
    date_tag    = datetime.now().strftime('%Y%m%d')
    output_path = Path(output_dir) / f"{safe_ticker}_{date_tag}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = ASREReportGenerator()
    return generator.generate_single_stock_report(
        ticker=ticker,
        analysis_data=analysis_data,
        output_path=str(output_path),
        include_charts=True,
        ia_notes=ia_notes,
    )


def generate_portfolio_pdf(
    results:        List[Dict],
    output_dir:     str,
    portfolio_name: Optional[str] = None,
    ia_notes:       Optional[str] = None,
    config:         Optional[Dict] = None,
) -> Path:
    config = config or {}
    stocks = []

    for r in results:
        ticker = r['ticker']
        latest = r['latest']
        signal_val = latest.get('signal')
        if not signal_val or (isinstance(signal_val, float) and pd.isna(signal_val)):
            r_asre_val = float(latest.get('r_asre', 50) or 50)
            signal_val = get_signal_interpretation(r_asre_val)['signal']

        stocks.append({
            'ticker':  ticker,
            'signal':  str(signal_val),
            'r_final': float(latest.get('r_final', 50) or 50),
            'r_asre':  float(latest.get('r_asre',  50) or 50),
            'f_score': float(latest.get('f_score', 50) or 50),
            't_score': float(latest.get('t_score', 50) or 50),
            'm_score': float(latest.get('m_score', 50) or 50),
            'tier':    str(latest.get('quality_tier', 'C') or 'C'),
            **({'market_context': {
                    'stage':       latest.get('market_stage', 'N/A'),
                    'sma200_dist': float(latest.get('sma200_distance', 0) or 0) * 100,
                    'score':       float(latest.get('market_score', 0) or 0),
                    'confidence':  float(latest.get('market_confidence', 0) or 0) * 100,
                }}
               if 'market_stage' in latest else {}),
        })

    name        = portfolio_name or "ASRE Portfolio"
    date_tag    = datetime.now().strftime('%Y%m%d')
    safe_name   = name.replace(' ', '_')
    output_path = Path(output_dir) / f"{safe_name}_{date_tag}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = ASREReportGenerator()
    return generator.generate_portfolio_summary(
        stocks=stocks,
        output_path=str(output_path),
        portfolio_name=name,
        ia_notes=ia_notes,
    )


def generate_compare_pdf(
    comparison_df:  pd.DataFrame,
    per_stock_dfs:  Dict[str, pd.DataFrame],
    start_date:     str,
    end_date:       str,
    ia_notes:       Optional[str] = None,
    output_path:    Optional[str] = None,
    output_dir:     str = "reports",
    config:         Optional[Dict] = None,
) -> Path:
    config = config or {}
    stocks = []

    for _, row in comparison_df.iterrows():
        ticker     = row['ticker']
        r_asre_val = float(row.get('r_asre', 50) or 50)
        signal_val = get_signal_interpretation(r_asre_val)['signal']

        entry: Dict = {
            'ticker':  ticker,
            'signal':  signal_val,
            'r_final': float(row.get('r_final', 50) or 50),
            'r_asre':  r_asre_val,
            'f_score': float(row.get('f_score', 50) or 50),
            't_score': float(row.get('t_score', 50) or 50),
            'm_score': float(row.get('m_score', 50) or 50),
            'tier':    'C',
        }

        if ticker in per_stock_dfs:
            full_df = per_stock_dfs[ticker]
            latest  = full_df.iloc[-1]
            entry['tier'] = str(latest.get('quality_tier', 'C') or 'C')

            dip = run_dip_analyzer(latest, df=full_df)
            if dip is not None and dip.is_dip and dip.dip_stage not in ('NONE', 'STRUCTURAL BREAK'):
                entry['market_context'] = {
                    'stage':       dip.dip_stage,
                    'sma200_dist': dip.distance_from_sma_pct,
                    'score':       dip.dip_quality_score,
                    'confidence':  dip.confidence,
                }

        stocks.append(entry)

    if output_path:
        out = Path(output_path)
    else:
        date_tag = datetime.now().strftime('%Y%m%d')
        out = Path(output_dir) / f"comparison_{date_tag}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    portfolio_name = (
        f"ASRE Comparison — {len(stocks)} Stocks  "
        f"[{start_date} → {end_date}]"
    )

    generator = ASREReportGenerator()
    return generator.generate_portfolio_summary(
        stocks=stocks,
        output_path=str(out),
        portfolio_name=portfolio_name,
        ia_notes=ia_notes,
    )


# ---------------------------------------------------------------------------
# FIX 2.4 — Audit-enriched ia_notes builder
# ---------------------------------------------------------------------------

def _build_ia_notes(args_ia_notes: Optional[str],
                    run_id: str,
                    mode: str,
                    version: str) -> str:
    header = (
        f"[ASRE Audit] Run: {run_id} | Mode: {mode.upper()} | "
        f"Version: {version} | "
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )
    if args_ia_notes:
        return f"{header}\n\n{args_ia_notes}"
    return header


# ---------------------------------------------------------------------------
# FIX 5.7 — Morning scan summary display  (unchanged from v3.0)
# ---------------------------------------------------------------------------

def display_scan_summary(
    all_results: List[Dict],
    cf:          "ComplianceFilter",
    decision_log: Optional["DecisionLog"] = None,
):
    if not all_results:
        print_error("No results to display in scan summary.")
        return

    if decision_log:
        decision_log.add_disclosure("SEBI_AI_DISCLOSURE_SCAN")

    if HAS_RICH and console:
        table = Table(
            title=cf.apply_panel_title("Morning Scan — ASRE Research Summary"),
            title_style="bold cyan",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
        )
        table.add_column("Ticker",       style="cyan",      no_wrap=True, width=14)
        table.add_column("F",            style="green",     justify="right", width=5)
        table.add_column("T",            style="blue",      justify="right", width=5)
        table.add_column("M",            style="magenta",   justify="right", width=5)
        table.add_column("R_Final",      style="yellow",    justify="right", width=8)
        table.add_column("R_ASRE",       style="bold cyan", justify="right", width=8)
        table.add_column("Signal",       justify="center",  width=22)
        table.add_column("Research Note",style="dim",       width=40)

        for r in sorted(all_results, key=lambda x: x["latest"].get("r_asre", 0), reverse=True):
            ticker = r["ticker"]
            latest = r["latest"]
            f_val  = float(latest.get("f_score", 50) or 50)
            t_val  = float(latest.get("t_score", 50) or 50)
            m_val  = float(latest.get("m_score", 50) or 50)
            rf_val = float(latest.get("r_final", 50) or 50)
            ra_val = float(latest.get("r_asre",  50) or 50)
            tier   = str(latest.get("quality_tier", "C") or "C")

            sig  = get_signal_interpretation(ra_val)
            note = cf.apply(StatusLineRenderer.render_short(
                f=f_val, t=t_val, m=m_val, r_final=rf_val, tier=tier))
            if len(note) > 38:
                note = note[:35] + "..."

            ra_color = ("bold green" if ra_val >= 70
                        else "yellow" if ra_val >= 50 else "bold red")

            table.add_row(
                ticker,
                f"{f_val:.0f}", f"{t_val:.0f}", f"{m_val:.0f}",
                f"[yellow]{rf_val:.1f}[/yellow]",
                f"[{ra_color}]{ra_val:.1f}[/{ra_color}]",
                f"[{sig['color']}]{cf.apply(sig['signal'])}[/{sig['color']}]",
                note,
            )

        console.print()
        console.print(table)
        console.print()
        console.print(
            f"[dim]Scan complete — {len(all_results)} ticker(s). "
            "All scores are research outputs only. "
            "Not investment advice. SEBI Circular Dec 2024.[/dim]"
        )
        console.print()

    else:
        header = f"{'Ticker':<14} {'F':>5} {'T':>5} {'M':>5} {'R_Final':>8} {'R_ASRE':>8}  Signal"
        print(f"\n{'='*80}\nMorning Scan — ASRE Research Summary\n{'='*80}")
        print(header)
        print("-" * 80)
        for r in sorted(all_results, key=lambda x: x["latest"].get("r_asre", 0), reverse=True):
            latest = r["latest"]
            f_val  = float(latest.get("f_score", 50) or 50)
            t_val  = float(latest.get("t_score", 50) or 50)
            m_val  = float(latest.get("m_score", 50) or 50)
            rf_val = float(latest.get("r_final", 50) or 50)
            ra_val = float(latest.get("r_asre",  50) or 50)
            sig    = get_signal_interpretation(ra_val)
            print(f"{r['ticker']:<14} {f_val:>5.0f} {t_val:>5.0f} {m_val:>5.0f} "
                  f"{rf_val:>8.1f} {ra_val:>8.1f}  {cf.apply(sig['signal'])}")
        print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Command: compute  (PDF-INT-3 — store fundamentals_df in all_results)
# ---------------------------------------------------------------------------

def command_compute(args, decision_log: DecisionLog, disclosure: "DisclosureBlock"):
    setup_logging(args.verbose)
    # --clear-cache: wipe fundamentals cache so next fetch is always live
    if getattr(args, 'clear_cache', False):
        import shutil
        _cache_dir = Path('data') / 'cache' / 'fundamentals'
        if _cache_dir.exists():
            shutil.rmtree(_cache_dir)
            _cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("--clear-cache: fundamentals cache wiped at %s", _cache_dir)
        else:
            logger.info("--clear-cache: cache dir not found, nothing to wipe.")

    print_header("Multi-Stock Analysis", f"Processing {len(args.tickers)} tickers")

    cf = ComplianceFilter(mode=OutputMode.IA if decision_log.mode == "ia" else OutputMode.RA)

    decision_log.command = "compute"
    decision_log.tickers = [t.upper() for t in args.tickers]

    config_dict = load_config(args.config)
    configs     = create_configs_from_dict(config_dict)

    if args.date:
        end_date   = pd.to_datetime(args.date)
        start_date = end_date - timedelta(days=365)
    else:
        start_date = pd.to_datetime(args.start) if args.start else datetime.now() - timedelta(days=365)
        end_date   = pd.to_datetime(args.end)   if args.end   else datetime.now()

    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")
    decision_log.start_date = start_str
    decision_log.end_date   = end_str

    enriched_ia_notes = _build_ia_notes(
        getattr(args, 'pdf_ia_notes', None),
        run_id=decision_log.run_id,
        mode=decision_log.mode,
        version=decision_log.version,
    )
    decision_log.ia_notes = enriched_ia_notes

    all_results = []

    def _process(ticker):
        fundamentals, fetch_ts = fetch_fundamentals(ticker, start_str, end_str)
        if fundamentals is None:
            print_error(f"Skipped {ticker}: could not fetch fundamentals")
            decision_log.add_disclosure(f"SKIPPED:{ticker}:fundamentals_unavailable")
            return

        df = load_price_data_guarded(
            ticker, start_str, end_str,
            fundamentals,
            fundamentals_fetch_ts=fetch_ts,
        )

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        tier_handler = TierCapturingHandler()
        tier_handler.attach()
        try:
            df = compute_complete_asre(
                df,
                ticker=ticker,
                config=configs.get("composite"),
                fundamentals_config=configs.get("fundamentals"),
                technical_config=configs.get("technical"),
                momentum_config=configs.get("momentum"),
                medallion=True,
                return_all_components=True,
            )
        finally:
            tier_handler.detach()

        df = _inject_tier_column(df, tier_handler.captured())

        if "date" in df.columns:
            df = df.set_index("date")

        try:
            import hashlib as _hashlib
            _score_cols = [c for c in ("f_score", "t_score", "m_score", "r_final", "r_asre")
                           if c in df.columns]
            if _score_cols:
                _hash_csv = df[_score_cols].to_csv().encode("utf-8")
                decision_log.score_hash = _hashlib.sha256(_hash_csv).hexdigest()
        except Exception as _he:
            logger.debug("score_hash computation failed — %s", _he)

        # PDF-INT-3 — store fundamentals_df and fetch_ts alongside df/latest
        all_results.append({
            "ticker":           ticker.upper(),
            "df":               df,
            "latest":           df.iloc[-1],
            "fundamentals_df":  fundamentals,    # ← new in v3.1
            "fetch_ts":         fetch_ts,        # ← new in v3.1
        })

    # ── execution ────────────────────────────────────────────
    if HAS_RICH and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Analyzing stocks...", total=len(args.tickers))
            for ticker in args.tickers:
                progress.update(task, description=f"[cyan]Processing {ticker.upper()}...")
                try:
                    _process(ticker)
                except Exception as e:
                    print_error(f"Failed {ticker}: {e}")
                    if args.verbose:
                        import traceback; traceback.print_exc()
                progress.advance(task)
    else:
        for ticker in args.tickers:
            try:
                _process(ticker)
            except Exception as e:
                print_error(f"Failed {ticker}: {e}")

    print_success(f"Completed {len(all_results)}/{len(args.tickers)} stocks!")

    if getattr(args, 'scan', False):
        display_scan_summary(all_results, cf=cf, decision_log=decision_log)
    else:
        for result in all_results:
            ticker     = result["ticker"]
            df_display = result["df"].tail(args.last) if not args.date else result["df"].tail(1)
            display_ratings_dual_score(
                df_display, ticker, args.output_format,
                full_df=result["df"],
                decision_log=decision_log,
                cf=cf,
            )
            print()

    # ── PDF export ────────────────────────────────────────────
    if args.export_pdf:
        os.makedirs(args.pdf_output_dir, exist_ok=True)
        pdf_errors = 0

        for r in all_results:
            try:
                # PDF-INT-3 — pass fundamentals_df + fetch_ts to generate_stock_pdf
                out = generate_stock_pdf(
                    ticker               = r["ticker"],
                    df                   = r["df"],
                    latest               = r["latest"],
                    output_dir           = args.pdf_output_dir,
                    ia_notes             = enriched_ia_notes,
                    config               = config_dict,
                    fundamentals_df      = r.get("fundamentals_df"),   # ← new in v3.1
                    fundamentals_fetch_ts= r.get("fetch_ts"),          # ← new in v3.1
                )
                decision_log.add_pdf(str(out))
                print_success(f"PDF: {out.name}")
            except Exception as e:
                print_error(f"PDF failed for {r['ticker']}: {e}")
                pdf_errors += 1
                if args.verbose:
                    import traceback; traceback.print_exc()

        if args.pdf_portfolio and len(all_results) > 1:
            try:
                out = generate_portfolio_pdf(
                    results        = all_results,
                    output_dir     = args.pdf_output_dir,
                    portfolio_name = args.pdf_portfolio_name,
                    ia_notes       = enriched_ia_notes,
                    config         = config_dict,
                )
                decision_log.add_pdf(str(out))
                print_success(f"Portfolio PDF: {out.name}")
            except Exception as e:
                print_error(f"Portfolio PDF failed: {e}")
                pdf_errors += 1
                if args.verbose:
                    import traceback; traceback.print_exc()

        if pdf_errors == 0:
            print_success(f"All PDF report(s) generated in {args.pdf_output_dir}/")
        else:
            print_error(f"{pdf_errors} PDF(s) failed — check logs above.")

    if args.output:
        combined_df = pd.concat(
            [r["df"].assign(ticker=r["ticker"]) for r in all_results]
        )
        export_results(combined_df.reset_index(), args.output, args.output_format)
        print_success(f"Exported {len(all_results)} stocks to {args.output}")


# ---------------------------------------------------------------------------
# Command: compare  (PDF-INT-3 — store fundamentals_df in full_dfs)
# ---------------------------------------------------------------------------

def command_compare(args, decision_log: DecisionLog, disclosure: "DisclosureBlock"):
    setup_logging(args.verbose)
        # --clear-cache: wipe fundamentals cache so next fetch is always live
    if getattr(args, 'clear_cache', False):
        import shutil
        _cache_dir = Path('data') / 'cache' / 'fundamentals'
        if _cache_dir.exists():
            shutil.rmtree(_cache_dir)
            _cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("--clear-cache: fundamentals cache wiped at %s", _cache_dir)
        else:
            logger.info("--clear-cache: cache dir not found, nothing to wipe.")
    print_header("Stock Comparison", f"Comparing {len(args.tickers)} stocks")

    cf = ComplianceFilter(
        mode=OutputMode.RA if decision_log.mode == "ra" else OutputMode.IA
    )

    decision_log.command = "compare"
    decision_log.tickers = [t.upper() for t in args.tickers]

    config_dict = load_config(args.config)
    configs     = create_configs_from_dict(config_dict)

    end_date   = pd.to_datetime(args.date) if args.date else datetime.now()
    start_date = end_date - timedelta(days=365)
    start_str  = start_date.strftime('%Y-%m-%d')
    end_str    = end_date.strftime('%Y-%m-%d')
    decision_log.start_date = start_str
    decision_log.end_date   = end_str

    enriched_ia_notes = _build_ia_notes(
        getattr(args, 'pdf_ia_notes', None),
        run_id=decision_log.run_id,
        mode=decision_log.mode,
        version=decision_log.version,
    )
    decision_log.ia_notes = enriched_ia_notes

    results   = []
    full_dfs  = {}

    def _process(ticker):
        fundamentals, fetch_ts = fetch_fundamentals(ticker, start_str, end_str)
        if fundamentals is None:
            logger.error(f"Skipped {ticker}: could not fetch fundamentals")
            return

        df = load_price_data_guarded(
            ticker, start_str, end_str,
            fundamentals,
            fundamentals_fetch_ts=fetch_ts,
        )

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

        tier_handler = TierCapturingHandler()
        tier_handler.attach()
        try:
            df = compute_complete_asre(
                df,
                ticker=ticker,
                config=configs.get('composite'),
                fundamentals_config=configs.get('fundamentals'),
                technical_config=configs.get('technical'),
                momentum_config=configs.get('momentum'),
                medallion=True,
                return_all_components=True,
            )
        finally:
            tier_handler.detach()

        df = _inject_tier_column(df, tier_handler.captured())

        try:
            import hashlib as _hashlib
            _score_cols = [c for c in ("f_score", "t_score", "m_score", "r_final", "r_asre")
                           if c in df.columns]
            if _score_cols:
                _hash_csv = df[_score_cols].to_csv().encode("utf-8")
                _ticker_hash = _hashlib.sha256(_hash_csv).hexdigest()
                if decision_log.score_hash == "pending":
                    decision_log.score_hash = _ticker_hash
                else:
                    combined_input = (decision_log.score_hash + _ticker_hash).encode('utf-8')
                    decision_log.score_hash = hashlib.sha256(combined_input).hexdigest()
        except Exception as _he:
            logger.debug("score_hash computation failed — %s", _he)

        latest   = df.iloc[-1]
        dip      = run_dip_analyzer(latest, df=df)
        ticker_u = ticker.upper()

        # PDF-INT-3 — store scored df AND fundamentals_df keyed by ticker
        full_dfs[ticker_u] = {
            'df':              df,
            'fundamentals_df': fundamentals,   # ← new in v3.1
            'fetch_ts':        fetch_ts,       # ← new in v3.1
        }

        results.append({
            'ticker':      ticker_u,
            'date':        latest.name,
            'f_score':     latest.get('f_score',  0),
            't_score':     latest.get('t_score',  0),
            'm_score':     latest.get('m_score',  0),
            'r_final':     latest.get('r_final',  0),
            'r_asre':      latest.get('r_asre',   0),
            'dip_quality': dip.dip_quality_score   if dip else None,
            'dip_stage':   dip.dip_stage            if dip else None,
            'dip_signal':  dip.entry_signal         if dip else None,
            'is_buy_dip':  dip.approved_for_dip_buy if dip else False,
        })

    # ── execution ─────────────────────────────────────────────
    if HAS_RICH and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Processing stocks...", total=len(args.tickers))
            for ticker in args.tickers:
                progress.update(task, description=f"[cyan]Processing {ticker.upper()}...")
                try:
                    _process(ticker)
                except Exception as e:
                    logger.error(f"Failed to process {ticker}: {e}")
                    if args.verbose:
                        import traceback; traceback.print_exc()
                progress.advance(task)
    else:
        for ticker in args.tickers:
            print(f"Processing {ticker.upper()}...")
            try:
                _process(ticker)
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")

    if not results:
        print_error("No results to display")
        return

    print_success(f"Analyzed {len(results)} stocks")

    results_df = (
        pd.DataFrame(results)
        .sort_values('r_asre', ascending=False)
        .reset_index(drop=True)
    )

    display_comparison_dual_score(results_df, decision_log=decision_log, cf=cf)

    if args.export_pdf:
        output_dir = Path(args.pdf_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        today_tag  = datetime.now().strftime("%Y%m%d")
        pdf_path   = output_dir / f"comparison_{today_tag}.pdf"

        # PDF-INT-3 — build per_stock_dfs with just the scored df (for comparison PDF)
        # and pass fundamentals for the primary per-stock PDFs
        per_stock_scored_dfs = {
            t: v['df'] for t, v in full_dfs.items()
        }

        try:
            out = generate_compare_pdf(
                comparison_df  = results_df,
                per_stock_dfs  = per_stock_scored_dfs,
                start_date     = start_str,
                end_date       = end_str,
                ia_notes       = enriched_ia_notes,
                output_path    = str(pdf_path),
                config         = config_dict,
            )
            decision_log.add_pdf(str(out))
            print_success(f"Comparison PDF generated: {out.name}")
        except Exception as e:
            print_error(f"Comparison PDF failed: {e}")
            if args.verbose:
                import traceback; traceback.print_exc()

    if args.output:
        results_df.to_csv(args.output, index=False)
        print_success(f"Exported to {args.output}")


# ---------------------------------------------------------------------------
# Command: backtest  (unchanged from v3.0)
# ---------------------------------------------------------------------------

def command_backtest(args, decision_log: DecisionLog, disclosure: "DisclosureBlock"):
    setup_logging(args.verbose)
    print_header("Strategy Backtest", f"Testing ASRE strategy on {args.ticker.upper()}")
    decision_log.command = "backtest"
    decision_log.tickers = [args.ticker.upper()]

    config_dict = load_config(args.config)
    configs     = create_configs_from_dict(config_dict)

    start_date = pd.to_datetime(args.start) if args.start else datetime.now() - timedelta(days=365 * 2)
    end_date   = pd.to_datetime(args.end)   if args.end   else datetime.now()
    start_str  = start_date.strftime("%Y-%m-%d")
    end_str    = end_date.strftime("%Y-%m-%d")
    decision_log.start_date = start_str
    decision_log.end_date   = end_str

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Canonical cleaner. Handles ALL cases that occur in practice:
          1. Mixed-case column names from yfinance  ('Close', 'close', 'Adj Close')
          2. Duplicate column names after lowercasing   ← THE ROOT CAUSE
          3. Date as a column vs already in index
          4. Timezone-aware DatetimeIndex
          5. Duplicate index rows
        Order matters — do NOT reorder these steps.
        """
        df = df.copy()
        # Step 1 — strip THEN lowercase (order is critical)
        df.columns = df.columns.str.strip().str.lower()
        # Step 2 — promote 'date' column to index if present
        if 'date' in df.columns:
            df = df.set_index('date')
        # Step 3 — ensure DatetimeIndex, strip tz
        df.index = pd.to_datetime(df.index)
        if df.index.tzinfo is not None:
            df.index = df.index.tz_localize(None)
        # Step 4 — deduplicate COLUMNS (before any df['close'] access)
        if df.columns.duplicated().any():
            dupes = df.columns[df.columns.duplicated(keep=False)].unique().tolist()
            logger.warning("_clean_df: dropping duplicate columns %s (keeping last)", dupes)
            df = df.loc[:, ~df.columns.duplicated(keep='last')]
        # Step 5 — deduplicate index rows
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='last')]
        return df.sort_index()

    def _build_bt_config() -> BacktestConfig:
        cfg = configs.get('backtest')
        if cfg is None:
            cfg = BacktestConfig(
                threshold_long=args.threshold_long,
                threshold_short=args.threshold_short,
            )
        return cfg

    def _run_backtest(df: pd.DataFrame) -> "Backtester":
        # _clean_df runs AFTER compute_complete_asre because ASRE
        # re-injects 'close', 'r_asre', 'signal', 'position' columns.
        df = _clean_df(df)

        # Hard assertion — fail loudly with exact column names
        dupes = df.columns[df.columns.duplicated()].tolist()
        if dupes:
            raise RuntimeError(
                f"Duplicate columns remain after _clean_df: {dupes}. "
                "This is a data pipeline bug — check load_stock_data and "
                "compute_complete_asre column outputs."
            )

        if 'r_asre' not in df.columns:
            available = [c for c in df.columns if 'asre' in c or 'final' in c or 'score' in c]
            raise ValueError(
                f"Column 'r_asre' not found after compute_complete_asre. "
                f"Available score columns: {available}"
            )

        bt = Backtester(df, rating_col='r_asre', config=_build_bt_config())
        bt.run(
            signal_type=args.signal_type,
            transaction_cost=args.transaction_cost,
            slippage=args.slippage,
            max_position=args.max_position,
        )
        return bt

    # ── Main pipeline ─────────────────────────────────────────────────────────
    bt = None

    if HAS_RICH and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Loading historical data...", total=100)
            try:
                progress.update(task, description="[cyan]Fetching fundamentals...")
                fundamentals, fetch_ts = fetch_fundamentals(args.ticker, start_str, end_str)
                if fundamentals is None:
                    print_error(f"Cannot run backtest for {args.ticker}: live fundamentals unavailable.")
                    sys.exit(1)
                progress.update(task, completed=20)

                progress.update(task, description="[cyan]Loading historical data...")
                df = load_price_data_guarded(
                    args.ticker, start_str, end_str,
                    fundamentals, fundamentals_fetch_ts=fetch_ts,
                )
                progress.update(task, completed=40)

                progress.update(task, description="[cyan]Computing ASRE ratings...")
                df = compute_complete_asre(
                    df,
                    ticker=args.ticker,
                    config=configs.get('composite'),
                    fundamentals_config=configs.get('fundamentals'),
                    technical_config=configs.get('technical'),
                    momentum_config=configs.get('momentum'),
                    medallion=True,
                    return_all_components=True,          # ← REQUIRED
                )
                progress.update(task, completed=70)

                progress.update(task, description="[cyan]Running backtest simulation...")
                bt = _run_backtest(df)
                progress.update(task, completed=100)

            except SystemExit:
                raise
            except Exception as e:
                print_error(f"Backtest failed: {e}")
                import traceback; traceback.print_exc()  # always print — no verbose gate
                sys.exit(1)

    else:
        # ── Plain-text branch ─────────────────────────────────────────────────
        try:
            print("Fetching fundamentals...")
            fundamentals, fetch_ts = fetch_fundamentals(args.ticker, start_str, end_str)
            if fundamentals is None:
                print(f"✗ Cannot run backtest: no fundamentals for {args.ticker}")
                sys.exit(1)

            print("Loading data...")
            df = load_price_data_guarded(
                args.ticker, start_str, end_str,
                fundamentals, fundamentals_fetch_ts=fetch_ts,
            )

            print("Computing ratings...")
            df = compute_complete_asre(
                df,
                ticker=args.ticker,
                config=configs.get('composite'),
                fundamentals_config=configs.get('fundamentals'),
                technical_config=configs.get('technical'),
                momentum_config=configs.get('momentum'),
                medallion=True,
                return_all_components=True,              # ← REQUIRED (was missing)
            )

            print("Running backtest...")
            bt = _run_backtest(df)

        except SystemExit:
            raise
        except Exception as e:
            print(f"✗ Backtest failed: {e}")
            import traceback; traceback.print_exc()      # always print — no verbose gate
            sys.exit(1)

    # ── Results ───────────────────────────────────────────────────────────────
    if bt is None or bt.results_df is None or bt.results_df.empty:
        print_error("Backtest produced no results. Check input data and date range.")
        sys.exit(1)

    report = bt.get_report()
    display_backtest_elegant(report, args.ticker.upper())

    if args.output:
        export_backtest(bt.results_df, report, args.output, args.output_format)
        print_success(f"Exported to {args.output}")

    try:
        entry_hash = HashLedger.append(
            run_id=decision_log.run_id,
            pdf_paths=decision_log.pdf_exports,
            score_hash=decision_log.score_hash,
            mode=decision_log.mode,
            tickers=decision_log.tickers,
            version=APP_VERSION,
        )
        logger.info("Ledger entry: %s", entry_hash[:16])
    except Exception as exc:
        logger.warning("HashLedger.append failed: %s", exc)

    try:
        clear_role_lock()
    except Exception as exc:
        logger.debug("clear_role_lock failed: %s", exc)

#---------------------------------------------------------------------------
# display_ratings_dual_score  (unchanged from v3.0)
# ---------------------------------------------------------------------------

def display_ratings_dual_score(
    df:           pd.DataFrame,
    ticker:       str,
    format_type:  str = "table",
    full_df:      Optional[pd.DataFrame] = None,
    decision_log: Optional[DecisionLog]  = None,
    cf:           Optional["ComplianceFilter"] = None,
):
    cf = cf or ComplianceFilter(mode=OutputMode.IA)

    # ── FIX: normalise before any access ────────────────────────────────────
    df = df.copy()
    df.columns = df.columns.str.lower()
    if 'date' in df.columns:
        df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='last')].sort_index()

    # ── Score extraction ─────────────────────────────────────────────────────
    latest       = df.iloc[-1]
    latest_f     = latest.get('f_score', np.nan)
    latest_t     = latest.get('t_score', np.nan)
    latest_m     = latest.get('m_score', np.nan)
    r_final_val  = latest.get('r_final', np.nan)
    r_asre_val   = latest.get('r_asre',  np.nan)

    r_final_display = float(r_final_val) if pd.notna(r_final_val) else 50.0
    r_asre_display  = float(r_asre_val)  if pd.notna(r_asre_val)  else 50.0
    latest_r_asre   = r_asre_display

    signal_final = get_signal_interpretation(r_final_display)
    signal_asre  = get_signal_interpretation(r_asre_display)
    score_gap    = abs(r_final_display - r_asre_display)

    has_divergence = score_gap >= 25
    dip_result     = run_dip_analyzer(latest, df=full_df if full_df is not None else df)

    tier = str(
        latest.get('quality_tier') or latest.get('tier') or
        latest.get('stock_tier')   or latest.get('medallion_tier') or
        latest.get('classification_tier') or latest.get('asre_tier') or 'C'
    )
    status_line = cf.apply(StatusLineRenderer.render(
        f=latest_f   if pd.notna(latest_f)   else 50.0,
        t=latest_t   if pd.notna(latest_t)   else 50.0,
        m=latest_m   if pd.notna(latest_m)   else 50.0,
        r_final=r_final_display,
        tier=tier,
        r_asre=r_asre_display,
    ))

    if decision_log:
        decision_log.add_disclosure("SEBI_AI_DISCLOSURE")

    if HAS_RICH and console:
        table = Table(
            title=f"📊 ASRE Ratings Timeline - {ticker}",
            title_style="bold cyan", box=box.ROUNDED,
            show_header=True, header_style="bold magenta", border_style="cyan",
        )
        table.add_column("Date",    style="cyan",        no_wrap=True, justify="center")
        table.add_column("F %",     style="green",       justify="right", width=6)
        table.add_column("T %",     style="blue",        justify="right", width=6)
        table.add_column("M %",     style="magenta",     justify="right", width=6)
        table.add_column("R_Final", style="bold yellow", justify="right", width=8)
        table.add_column("R_ASRE",  style="bold cyan",   justify="right", width=8)
        table.add_column("Signal",  justify="center",    width=18)

        for idx, row in df.tail(10).iterrows():
            date_str = safe_date_format(idx)
            row_rf   = row.get('r_final', np.nan)
            row_ra   = row.get('r_asre',  np.nan)
            row_f    = row.get('f_score', np.nan)
            row_t    = row.get('t_score', np.nan)
            row_m    = row.get('m_score', np.nan)
            row_sig  = get_signal_interpretation(row_ra if pd.notna(row_ra) else None)
            asre_col = ("bold green" if pd.notna(row_ra) and row_ra >= 70
                        else "yellow" if pd.notna(row_ra) and row_ra >= 50 else "bold red")
            table.add_row(
                date_str,
                f"[green]{row_f:.0f}%[/green]"           if pd.notna(row_f)  else "[dim]--[/dim]",
                f"[blue]{row_t:.0f}%[/blue]"             if pd.notna(row_t)  else "[dim]--[/dim]",
                f"[magenta]{row_m:.0f}%[/magenta]"       if pd.notna(row_m)  else "[dim]--[/dim]",
                f"[yellow]{row_rf:.1f}[/yellow]"         if pd.notna(row_rf) else "[dim]nan[/dim]",
                f"[{asre_col}]{row_ra:.1f}[/{asre_col}]" if pd.notna(row_ra) else "[dim]nan[/dim]",
                f"{row_sig['emoji']} [{row_sig['color']}]{row_sig['signal']}[/{row_sig['color']}]",
            )

        console.print()
        console.print(table)
        console.print()

        def _fmt(v, unit="%"):
            return f"{v:.0f}{unit}" if pd.notna(v) else f"--{unit}"

        score_panels = [
            Panel(Align.center(
                f"[bold green]{_fmt(latest_f)}[/bold green]\n"
                f"[dim]({_fmt(latest_f, '')} / 100)[/dim]\nFundamentals"),
                border_style="green", box=box.ROUNDED),
            Panel(Align.center(
                f"[bold blue]{_fmt(latest_t)}[/bold blue]\n"
                f"[dim]({_fmt(latest_t, '')} / 100)[/dim]\nTechnical"),
                border_style="blue", box=box.ROUNDED),
            Panel(Align.center(
                f"[bold magenta]{_fmt(latest_m)}[/bold magenta]\n"
                f"[dim]({_fmt(latest_m, '')} / 100)[/dim]\nMomentum"),
                border_style="magenta", box=box.ROUNDED),
        ]
        console.print(Columns(score_panels, equal=True, expand=True))
        console.print()

        latest_date  = safe_date_format(latest.name)
        r_final_str  = f"{r_final_display:.1f}/100"
        r_asre_str   = f"{r_asre_display:.1f}/100"
        border_color = signal_asre["color"].replace("bold ", "")

        panel_lines = [
            f"[bold yellow]R_Final:[/bold yellow]  {r_final_str}   "
            f"[dim]-> {signal_final['emoji']} [{signal_final['color']}]{signal_final['signal']}[/{signal_final['color']}][/dim]",
            f"[dim]Composite weighted score[/dim]", "",
            f"[{signal_asre['color']}]R_ASRE:[/{signal_asre['color']}]   {r_asre_str}   "
            f"-> {signal_asre['emoji']} [{signal_asre['color']}]{signal_asre['signal']}[/{signal_asre['color']}]",
            f"[dim]Risk-parity Medallion score[/dim]", "",
            f"[dim]{signal_asre['interpretation']}[/dim]",
        ]
        if has_divergence:
            panel_lines += [
                "",
                f"[bold yellow]⚠ SCORE DIVERGENCE  ({score_gap:.0f} pts)[/bold yellow]",
                "[dim]R_Final reflects fundamental quality.[/dim]",
                "[dim]R_ASRE reflects entry timing / momentum.[/dim]",
                "[dim]Use R_ASRE for timing analysis, R_Final for conviction.[/dim]",
            ]

        console.print(Panel(
            Align.center("\n".join(panel_lines)),
            title=f"[bold cyan]{ticker}[/bold cyan] Assessment",
            subtitle=f"[dim]{latest_date}[/dim]",
            border_style=border_color, box=box.DOUBLE, padding=(1, 2),
        ))
        console.print()

        console.print(Panel(
            f"[bold white]{status_line}[/bold white]",
            title="📋 IA Research Summary",
            border_style="white",
            box=box.SIMPLE,
        ))
        console.print()

        # ── Market Context panel ─────────────────────────────────────────────
        market_context_display = ""
        context_color = "cyan"

        if dip_result is not None and dip_result.dip_stage == "NONE":
            scenario = detect_scenario(
                latest_f if pd.notna(latest_f) else 50,
                latest_t if pd.notna(latest_t) else 50,
                latest_m if pd.notna(latest_m) else 50,
            )
            market_context_display = (
                f"{scenario['emoji']} [{scenario['color']}]{scenario['scenario']}"
                f"[/{scenario['color']}]\n[dim]{scenario['description']}[/dim]\n"
                f"[dim](Price {dip_result.distance_from_sma_pct:+.1f}% vs long-term price average -- in uptrend)[/dim]"
            )
            context_color = scenario["color"].replace("bold ", "")

        elif dip_result is not None and dip_result.is_dip:
            dq   = dip_result.dip_quality_score
            dist = dip_result.distance_from_sma_pct
            dq_color = ("bold green"  if dq >= 75
                        else "bold yellow" if dq >= 60
                        else "yellow"      if dq >= 45
                        else "bold red")
            market_context_display = (
                f"[{dq_color}]{dip_result.entry_signal}[/{dq_color}]\n"
                f"[dim]Stage: {dip_result.dip_stage}  |  "
                f"SMA-200 dist: {dist:.1f}%  |  "
                f"Score: {dq:.0f}/100  |  "
                f"Conf: {dip_result.confidence:.0f}%[/dim]\n"
                f"[dim]{dip_result.reason}[/dim]"
            )
            context_color = dq_color.replace("bold ", "")

        elif 'market_context' in latest and pd.notna(latest['market_context']):
            market_context = str(latest['market_context'])
            dip_score      = latest.get('dip_dip_quality_score', 0) or 0
            if   'HIGH QUALITY' in market_context or dip_score >= 80: context_color = 'green'
            elif 'LATE-STAGE'   in market_context or dip_score <= 30: context_color = 'red'
            elif 'GOOD'         in market_context or dip_score >= 60: context_color = 'yellow'
            elif 'DIVERGENCE'   in market_context:                    context_color = 'red'
            else:                                                      context_color = 'cyan'
            market_context_display = market_context

        else:
            scenario = detect_scenario(
                latest_f if pd.notna(latest_f) else 50,
                latest_t if pd.notna(latest_t) else 50,
                latest_m if pd.notna(latest_m) else 50,
            )
            market_context_display = (
                f"{scenario['emoji']} [{scenario['color']}]{scenario['scenario']}"
                f"[/{scenario['color']}]\n{scenario['description']}"
            )
            context_color = scenario["color"].replace("bold ", "")

        _pc_summary = cf.apply(market_context_display) if market_context_display else ""
        if _pc_summary and _pc_summary.strip():
            console.print(Panel(
                _pc_summary,
                title=cf.apply_panel_title("📌 Market Context"),
                border_style=context_color, box=box.ROUNDED,
            ))
        elif market_context_display:
            _dist_str = (
                f"Price {dip_result.distance_from_sma_pct:+.1f}% vs SMA-200"
                if dip_result is not None
                else "SMA-200 distance data available"
            )
            console.print(Panel(
                f"[dim]{_dist_str}. Review F / T / M scores above for research context.[/dim]",
                title=cf.apply_panel_title("📌 Market Context"),
                border_style="dim", box=box.ROUNDED,
            ))
        else:
            console.print(Panel(
                "[dim]Price condition unavailable — insufficient price history[/dim]",
                title=cf.apply_panel_title("📌 Market Context"),
                border_style="dim", box=box.ROUNDED,
            ))
        console.print()

        # ── Dip Quality panel ────────────────────────────────────────────────
        if dip_result is not None and dip_result.is_dip and \
                dip_result.dip_stage not in ["NONE", "STRUCTURAL BREAK"]:
            dq = dip_result.dip_quality_score
            dip_color = ("bold green"  if dq >= 75
                         else "bold yellow" if dq >= 60
                         else "yellow"      if dq >= 45
                         else "bold red")
            sizing = (
                "100% of planned position (HIGH QUALITY)"  if dq >= 75
                else "50-75% of planned position (GOOD)"   if dq >= 60
                else "25% max -- wait for confirmation"     if dq >= 45
                else "No position -- insufficient quality (POOR)"
            )
            dip_factual = (
                f"[{dip_color}]Score: {dq:.0f}/100[/{dip_color}]   "
                f"Stage: {dip_result.dip_stage}   "
                f"Confidence: {dip_result.confidence:.0f}%\n"
                f"SMA-200 distance: {dip_result.distance_from_sma_pct:.1f}%   "
                f"Tier: {dip_result.quality_tier}"
            )
            _sizing_filtered = cf.apply(f"[dim]Position sizing guidance: {sizing}[/dim]")
            dip_content_final = (
                dip_factual + "\n" + _sizing_filtered
                if _sizing_filtered and _sizing_filtered.strip()
                else dip_factual
            )
            console.print(Panel(
                dip_content_final,
                title=cf.apply_panel_title("🎯 Dip Quality Analysis (Strategy C)"),
                border_style=dip_color.replace("bold ", ""), box=box.ROUNDED,
            ))
            console.print()

        elif dip_result is not None and dip_result.dip_stage == "STRUCTURAL BREAK":
            console.print(Panel(
                f"[bold red]{dip_result.reason}[/bold red]",
                title="🚨 Structural Breakdown Warning",
                border_style="red", box=box.ROUNDED,
            ))
            console.print()

        elif dip_result is not None and dip_result.dip_stage == "NONE":
            dist = dip_result.distance_from_sma_pct
            console.print(Panel(
                f"[dim]Price above SMA-200 ({dist:+.1f}%) — no dip active. "
                "Monitor for pullback opportunity.[/dim]",
                title=cf.apply_panel_title("🎯 Dip Quality Analysis (Strategy C)"),
                border_style="dim", box=box.ROUNDED,
            ))
            console.print()

        elif dip_result is None and 'dip_dip_quality_score' in latest \
                and pd.notna(latest.get('dip_dip_quality_score')):
            dip_quality     = latest['dip_dip_quality_score']
            dip_stage_str   = latest.get('dip_dip_stage', 'N/A')
            entry_timing    = latest.get('dip_entry_timing_score',  0) or 0
            expected_upside = latest.get('dip_expected_upside',     0) or 0
            risk_reward     = latest.get('dip_risk_reward_ratio',   0) or 0
            confidence      = latest.get('dip_confidence',          0) or 0
            dip_color       = ("bold green"  if dip_quality >= 80
                               else "bold yellow" if dip_quality >= 60
                               else "yellow"      if dip_quality >= 40
                               else "bold red")
            console.print(Panel(
                f"[{dip_color}]Overall Score:[/{dip_color}] "
                f"[{dip_color}]{dip_quality:.0f}/100[/{dip_color}]\n"
                f"Stage: {dip_stage_str} | Entry Timing: {entry_timing:.0f}/100\n"
                f"Expected Upside: {expected_upside:.1f}% | R/R: {risk_reward:.2f}\n"
                f"Confidence: {confidence:.0f}%",
                title="🎯 Dip Quality Metrics (legacy)",
                border_style=dip_color.replace("bold ", ""), box=box.ROUNDED,
            ))
            console.print()

        else:
            console.print(Panel(
                "[dim]Price condition unavailable — insufficient price history[/dim]",
                title=cf.apply_panel_title("🎯 Dip Quality Analysis (Strategy C)"),
                border_style="dim", box=box.ROUNDED,
            ))
            console.print()

        # ── Prediction Confidence panel ──────────────────────────────────────
        if 'confidence_lower' in latest and 'confidence_upper' in latest:
            ci_lower = latest.get('confidence_lower', latest_r_asre)
            ci_upper = latest.get('confidence_upper', latest_r_asre)
            ci_width = (ci_upper - ci_lower) if (pd.notna(ci_upper) and pd.notna(ci_lower)) else 0
            denom    = max(r_asre_display, 1)
            conf_pct = max(0, 100 - (ci_width / denom * 100))
            ci_l_str = f"{ci_lower:.1f}" if pd.notna(ci_lower) else "--"
            ci_u_str = f"{ci_upper:.1f}" if pd.notna(ci_upper) else "--"
            console.print(Panel(
                f"Range: [{ci_l_str}, {ci_u_str}]\nConfidence: {conf_pct:.0f}%",
                title="📈 Prediction Confidence",
                border_style="cyan", box=box.ROUNDED,
            ))
            console.print()

    else:
        # ── Plain text fallback ──────────────────────────────────────────────
        print(f"\n{'='*80}\nASRE Ratings - {ticker}\n{'='*80}\n")
        print(df[['f_score', 't_score', 'm_score', 'r_final', 'r_asre']].tail().round(1).to_string())
        print(f"\n{'='*80}\n")
        print(f"R_Final:  {r_final_display:.1f}/100  ->  {signal_final['signal']}")
        print(f"R_ASRE:   {r_asre_display:.1f}/100   ->  {signal_asre['signal']}")
        if has_divergence:
            print(f"\n⚠ SCORE DIVERGENCE ({score_gap:.0f} pts):")
            print("  R_Final = fundamental quality | R_ASRE = entry timing / momentum")
        f_p = f"{latest_f:.0f}%" if pd.notna(latest_f) else "--"
        t_p = f"{latest_t:.0f}%" if pd.notna(latest_t) else "--"
        m_p = f"{latest_m:.0f}%" if pd.notna(latest_m) else "--"
        print(f"F: {f_p} | T: {t_p} | M: {m_p}")
        print(f"\nIA Research Summary:\n{status_line}\n")
        if dip_result is not None and dip_result.is_dip:
            print("Dip Analysis (Strategy C):")
            print(f"  {dip_result.entry_signal}")
            print(f"  Score: {dip_result.dip_quality_score:.0f}/100 | "
                  f"Stage: {dip_result.dip_stage} | "
                  f"Dist: {dip_result.distance_from_sma_pct:.1f}%")
            print(f"  {dip_result.reason}")
        else:
            scenario = detect_scenario(
                latest_f if pd.notna(latest_f) else 50,
                latest_t if pd.notna(latest_t) else 50,
                latest_m if pd.notna(latest_m) else 50,
            )
            print(f"\n{scenario['scenario']}: {scenario['description']}")
        print(f"\n{signal_asre['interpretation']}\n")

# ---------------------------------------------------------------------------
# Comparison display  (unchanged from v3.0)
# ---------------------------------------------------------------------------

def display_comparison_dual_score(
    results_df:   pd.DataFrame,
    decision_log: Optional[DecisionLog] = None,
    cf:           Optional["ComplianceFilter"] = None,
):
    if cf is None:
        cf = ComplianceFilter(mode=OutputMode.IA)

    show_dip = (cf.mode != OutputMode.RA)

    if decision_log:
        decision_log.add_disclosure("SEBI_AI_DISCLOSURE")

    if HAS_RICH and console:
        table = Table(
            title=cf.apply_panel_title("🏆 Stock Rankings"),
            title_style="bold cyan", box=box.ROUNDED,
            show_header=True, header_style="bold magenta",
        )
        table.add_column("Rank",    justify="center", style="bold")
        table.add_column("Ticker",  style="cyan",     no_wrap=True)
        table.add_column("F",       style="green",    justify="right")
        table.add_column("T",       style="blue",     justify="right")
        table.add_column("M",       style="magenta",  justify="right")
        table.add_column("R_Final", style="yellow",   justify="right")
        table.add_column("R_ASRE",  style="bold cyan", justify="right")
        table.add_column("Signal",  justify="center")
        if show_dip:
            table.add_column("Dip Context", style="dim")

        for rank, (_, row) in enumerate(results_df.iterrows(), 1):
            rfinal      = row.get('r_final',     0) or 0
            rasre       = row.get('r_asre',       0) or 0
            fscore      = row.get('f_score',      0) or 0
            tscore      = row.get('t_score',      0) or 0
            mscore      = row.get('m_score',      0) or 0
            dip_quality = row.get('dip_quality',  None)
            dip_stage   = row.get('dip_stage',    None)
            signal_info = get_signal_interpretation(rasre)

            medal = ""; rank_style = "white"
            if rank == 1:   medal = "🥇"; rank_style = "bold yellow"
            elif rank == 2: medal = "🥈"; rank_style = "bold white"
            elif rank == 3: medal = "🥉"; rank_style = "bold orange1"

            ctx = ""
            if show_dip:
                if dip_stage and dip_stage not in [None, 'NONE', 'STRUCTURAL BREAK']:
                    if pd.notna(dip_quality):
                        if   dip_quality >= 75: ctx = f"🎯 {dip_stage}"
                        elif dip_quality >= 60: ctx = f"📈 {dip_stage}"
                        elif dip_quality >= 45: ctx = f"⚖️ {dip_stage}"
                        else:                   ctx = f"❌ {dip_stage}"
                elif dip_stage == 'STRUCTURAL BREAK':  ctx = '🚨 BREAK'
                elif row.get('is_buy_dip', False):      ctx = '🎯 DIP'
                if abs(rfinal - rasre) >= 25:           ctx += " ⚡DIV"

            sig_label = cf.apply(signal_info['signal'])
            row_data = [
                f"[{rank_style}]{rank}[/{rank_style}]",
                row["ticker"],
                f"{fscore:.0f}", f"{tscore:.0f}", f"{mscore:.0f}",
                f"[yellow]{rfinal:.1f}[/yellow]",
                f"[{signal_info['color']}]{rasre:.1f}[/{signal_info['color']}]",
                f"{signal_info['emoji']} [{signal_info['color']}]{sig_label}[/{signal_info['color']}]",
            ]
            if show_dip:
                row_data.append(f"{medal} {cf.apply(ctx)}")
            table.add_row(*row_data)

        console.print()
        console.print(table)
        console.print()
    else:
        print("\n" + results_df.to_string(index=False) + "\n")


# ---------------------------------------------------------------------------
# Backtest display  (unchanged from v3.0)
# ---------------------------------------------------------------------------

def display_backtest_elegant(report: Dict, ticker: str):
    if HAS_RICH and console:
        table = Table(
            title="📈 Performance Metrics",
            title_style="bold cyan", box=box.ROUNDED,
            show_header=True, header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan",   no_wrap=True)
        table.add_column("Value",  style="yellow", justify="right")
        table.add_column("",       style="dim")

        table.add_row("Total Return",  f"{report.get('total_return', 0):.2%}", "📊")
        table.add_row("CAGR",          f"[bold]{report.get('cagr', 0):.2%}[/bold]", "📈")
        table.add_row("Volatility",    f"{report.get('volatility', 0):.2%}", "📉")
        table.add_row("", "", "")
        sharpe = report.get('sharpe_ratio', 0)
        sc = "green" if sharpe >= 1.0 else "yellow"
        table.add_row("Sharpe Ratio",  f"[{sc}]{sharpe:.3f}[/{sc}]", "⚡")
        table.add_row("Sortino Ratio", f"{report.get('sortino_ratio', 0):.3f}", "✨")
        table.add_row("Calmar Ratio",  f"{report.get('calmar_ratio', 0):.3f}", "🎯")
        table.add_row("", "", "")
        table.add_row("Max Drawdown",  f"[red]{report.get('max_drawdown', 0):.2%}[/red]", "📉")
        table.add_row("VaR (95%)",     f"{report.get('var_95', 0):.2%}", "⚠️")
        table.add_row("", "", "")
        table.add_row("Win Rate",      f"{report.get('win_rate', 0):.2%}", "🎲")
        table.add_row("Profit Factor", f"{report.get('profit_factor', 0):.3f}", "💰")
        table.add_row("Trades",        f"{report.get('num_trades', 0):.0f}", "🔄")
        console.print()
        console.print(table)
        console.print()

        if   sharpe >= 2.0: rating, rc, interp = "EXCELLENT 🌟", "bold green", "Outstanding risk-adjusted returns."
        elif sharpe >= 1.0: rating, rc, interp = "GOOD ✅",       "green",      "Solid risk-adjusted returns."
        elif sharpe >= 0.5: rating, rc, interp = "FAIR ⚖️",       "yellow",     "Moderate risk-adjusted returns."
        else:               rating, rc, interp = "POOR ❌",        "red",        "Weak risk-adjusted returns."

        console.print(Panel(
            Align.center(
                f"[{rc}]{rating}[/{rc}]\n\n[dim]{interp}[/dim]\n\n"
                f"Sharpe: [{rc}]{sharpe:.3f}[/{rc}] | "
                f"Max DD: [red]{report.get('max_drawdown', 0):.1%}[/red] | "
                f"Win Rate: {report.get('win_rate', 0):.0%}"
            ),
            title=f"[bold cyan]{ticker}[/bold cyan] Strategy Assessment",
            border_style=rc.replace("bold ", ""), box=box.DOUBLE, padding=(1, 2),
        ))
        console.print()
    else:
        print(f"\n{'='*80}\nBacktest Results - {ticker}\n{'='*80}")
        print(f"Total Return:  {report.get('total_return', 0):.2%}")
        print(f"CAGR:          {report.get('cagr', 0):.2%}")
        print(f"Sharpe Ratio:  {report.get('sharpe_ratio', 0):.3f}")
        print(f"Max Drawdown:  {report.get('max_drawdown', 0):.2%}")
        print(f"Win Rate:      {report.get('win_rate', 0):.2%}\n{'='*80}\n")


# ---------------------------------------------------------------------------
# Export utilities  (unchanged from v3.0)
# ---------------------------------------------------------------------------

def export_results(df: pd.DataFrame, output_path: str, format_type: str = "csv"):
    output_file = Path(output_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    if 'date' in df.columns:
        df['date'] = df['date'].astype(str)
    if format_type == 'json':
        df.to_json(output_file, orient='records', date_format='iso', indent=2)
    else:
        df.to_csv(output_file, index=False)


def export_backtest(results_df: pd.DataFrame, report: Dict,
                    output_path: str, format_type: str = "csv"):
    output_file = Path(output_path)
    if format_type == 'json':
        output = {
            'report':       report,
            'equity_curve': results_df['cumulative_return'].to_dict(),
            'returns':      results_df['net_return'].to_dict(),
        }
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2, default=str)
    else:
        results_df.to_csv(output_file)


# ---------------------------------------------------------------------------
# Argument parser  (unchanged from v3.0)
# ---------------------------------------------------------------------------

def create_parser():
    parser = argparse.ArgumentParser(
        prog="asre",
        description=(
            "ASRE - Advanced Stock Rating Engine "
            "(v3.1 – PDF V4.2 Integration)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  asre --mode ia compute RELIANCE.NS
  asre --mode ra compute RELIANCE.NS INFY.NS --start 2024-01-01
  asre compute RELIANCE.NS INFY.NS --export-pdf --pdf-portfolio
  asre backtest INFY.NS --start 2023-01-01
  asre compare RELIANCE.NS INFY.NS TCS.NS --export-pdf
        """,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--config",  type=str,  help="Configuration file (JSON)")
    parser.add_argument("--version", action="version", version=f"ASRE {APP_VERSION}")
    parser.add_argument(
        "--mode", choices=["ia", "ra"], default="ia",
        help=(
            "Regulatory mode: 'ia' = Investment Adviser (SEBI IA), "
            "'ra' = Research Analyst (SEBI RA). "
            "Controls disclosure language and role gate checks. "
            "(default: ia)"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    cp = subparsers.add_parser("compute", help="Compute ASRE rating")
    cp.add_argument("tickers", nargs="+", type=str)
    cp.add_argument("--date", type=str)
    cp.add_argument("--start", type=str)
    cp.add_argument("--end", type=str)
    cp.add_argument("--last", type=int, default=10)
    cp.add_argument("-o", "--output", type=str)
    cp.add_argument("--output-format", choices=["table", "csv", "json"], default="table")
    cp.add_argument("-v", "--verbose", action="store_true")
    cp.add_argument("--export-pdf",         action="store_true")
    cp.add_argument("--pdf-output-dir",     type=str, default="reports")
    cp.add_argument("--pdf-portfolio",      action="store_true")
    cp.add_argument("--pdf-portfolio-name", type=str)
    cp.add_argument("--pdf-ia-notes",       type=str)
    cp.add_argument("--scan",               action="store_true",
                    help="Morning scan mode: compact summary table only")
    cp.add_argument(
        "--clear-cache",
        action="store_true",
        default=False,
        dest="clear_cache",
        help="Wipe fundamentals cache before computing (forces a fresh Yahoo Finance fetch).",
    )
    bp = subparsers.add_parser("backtest", help="Run strategy backtest")
    bp.add_argument("ticker", type=str)
    bp.add_argument("--start", type=str)
    bp.add_argument("--end", type=str)
    bp.add_argument("--signal-type", choices=["threshold", "quantile", "regime"],
                    default="threshold")
    bp.add_argument("--threshold-long",   type=float, default=70.0)
    bp.add_argument("--threshold-short",  type=float, default=30.0)
    bp.add_argument("--transaction-cost", type=float, default=0.001)
    bp.add_argument("--slippage",         type=float, default=0.0005)
    bp.add_argument("--max-position",     type=float, default=1.0)
    bp.add_argument("-o", "--output",     type=str)
    bp.add_argument("--output-format",    choices=["csv", "json"], default="csv")
    bp.add_argument("-v", "--verbose",    action="store_true")

    comp = subparsers.add_parser("compare", help="Compare multiple stocks")
    comp.add_argument("tickers", nargs="+")
    comp.add_argument("--date", type=str)
    comp.add_argument("-o", "--output", type=str)
    comp.add_argument("-v", "--verbose", action="store_true")
    comp.add_argument("--export-pdf",     action="store_true")
    comp.add_argument("--pdf-output-dir", type=str, default="reports")
    comp.add_argument("--pdf-ia-notes",   type=str)

    return parser


def main():
    parser = create_parser()
    args   = parser.parse_args()

    if args.command is None:
        if HAS_RICH and console:
            print_header("Advanced Stock Rating Engine",
                         f"v{APP_VERSION} - PDF V4.2 Integration")
            console.print("[dim]Use --help to see available commands[/dim]\n")
        parser.print_help()
        sys.exit(0)

    decision_log         = DecisionLog()
    decision_log.mode    = args.mode
    decision_log.version = APP_VERSION

    disclosure = DisclosureBlock(
        run_id  = decision_log.run_id,
        mode    = args.mode,
        version = APP_VERSION,
    )
    decision_log.disclosure_text = disclosure.render_audit()

    print_run_banner(disclosure)
    enforce_role(mode=args.mode, decision_log=decision_log)

    exit_status = "success"
    error_msg   = ""
    try:
        if   args.command == 'compute':
            command_compute(args,  decision_log, disclosure)
        elif args.command == 'backtest':
            command_backtest(args, decision_log, disclosure)
        elif args.command == 'compare':
            command_compare(args,  decision_log, disclosure)
        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print_info("\nOperation cancelled by user")
        exit_status = "aborted"
        error_msg   = "KeyboardInterrupt"
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if hasattr(args, "verbose") and args.verbose:
            import traceback; traceback.print_exc()
        exit_status = "error"
        error_msg   = str(e)

    finally:
        disclosure.update_score_hash(decision_log.score_hash)
        decision_log.disclosure_text = disclosure.render_audit()

        if HAS_RICH and console:
            console.print()
            console.out(disclosure.render_text())
            console.print()
        else:
            print(disclosure.render_text())

        decision_log.mark_complete(status=exit_status, error=error_msg)
        decision_log.flush()

        try:
            entry_hash = HashLedger.append(
                run_id     = decision_log.run_id,
                pdf_paths  = decision_log.pdf_exports,
                score_hash = decision_log.score_hash,
                mode       = decision_log.mode,
                tickers    = decision_log.tickers,
                version    = APP_VERSION,
            )
            logger.info("Ledger entry: %s", entry_hash[:16])
        except Exception as exc:
            logger.warning("HashLedger.append() failed — %s", exc)

        try:
            clear_role_lock()
        except Exception as exc:
            logger.debug("clear_role_lock() failed — %s", exc)

        if exit_status not in ("success",):
            sys.exit(0 if exit_status == "aborted" else 1)


if __name__ == '__main__':
    main()


__all__ = [
    'main', 'create_parser',
    'DecisionLog',
    'DipAnalysis', 'DipAnalyzer',
    'detect_scenario', 'run_dip_analyzer',
    'generate_status_line', 'get_signal_interpretation',
    'display_scan_summary',
    'generate_stock_pdf', 'generate_portfolio_pdf', 'generate_compare_pdf',
    # PDF-INT-4 — re-export V4.2 helpers for external callers
    'fetch_fundamentals_for_report',
    '_compute_ratios_from_fundamentals_df',
]