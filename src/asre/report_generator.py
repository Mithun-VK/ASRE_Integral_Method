"""
ASRE PDF Report Generator — V4.2  (FundamentalFetcher Integration)

Based on V4.1 (Enhanced UX + IA/RIA Notes).
All existing components preserved exactly.

What's new in V4.2:
    [INT] Integrated with FundamentalFetcher v2.0 (fundamental_fetcher.py)
          - New helper: fetch_fundamentals_for_report(ticker, fetcher, ...) → Dict
            Calls fetcher.fetch_quarterly_fundamentals(), computes derived ratios
            (P/E, ROE, D/E, Revenue Growth YoY, Profit Margin) and returns a
            dict ready to merge into analysis_data for generate_single_stock_report().
          - fetch_timestamp from FundamentalFetcher is stored in analysis_data
            ['data_sources']['fundamentals']['timestamp'] and rendered in the
            PDF Data Sources table exactly as Yahoo Finance returned it.
          - Graceful degradation: if FundamentalFetcher is unavailable or the
            fetch fails, a clear warning is logged and report generation continues
            with whatever data was already in analysis_data.

    [INT] export_stock_report() now accepts an optional `fetcher` parameter
          (a FundamentalFetcher instance). When supplied it calls
          fetch_fundamentals_for_report() automatically — no change needed to
          calling code that does not pass a fetcher.

    [INT] Staleness warnings from FundamentalFetcher (>24 h) are surfaced in the
          PDF Data Sources table via a "(STALE — re-fetch recommended)" suffix.
          Hard TTL RuntimeErrors are re-raised so the CLI sees them immediately.

All V4.1 compliance features (SEBI Dec 2024) fully retained.
API signature unchanged from V4.1 for callers that do not use the fetcher.

Author: ASRE Project
Version: 4.2
Date: March 2026
"""

from __future__ import annotations

import hashlib
import io
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)
from reportlab.platypus.flowables import Flowable

logger = logging.getLogger(__name__)


# ================================================================
# DESIGN SYSTEM  — unchanged from V4.0 / V4.1
# ================================================================

class DS:
    WHITE         = colors.HexColor("#FFFFFF")
    NEAR_WHITE    = colors.HexColor("#F8F9FA")
    SURFACE       = colors.HexColor("#F0F4F8")
    BORDER        = colors.HexColor("#DEE5ED")
    MUTED         = colors.HexColor("#8A9AB0")
    BODY_TEXT     = colors.HexColor("#2D3748")
    HEADING       = colors.HexColor("#1A202C")
    INK           = colors.HexColor("#0F1723")
    BRAND         = colors.HexColor("#1B4F72")
    BRAND_LIGHT   = colors.HexColor("#2E86C1")

    SIG_GREEN     = colors.HexColor("#1E8449")
    SIG_GREEN_BG  = colors.HexColor("#EAFAF1")
    SIG_GREEN_BD  = colors.HexColor("#A9DFBF")
    SIG_AMBER     = colors.HexColor("#B7770D")
    SIG_AMBER_BG  = colors.HexColor("#FEFCE8")
    SIG_AMBER_BD  = colors.HexColor("#F9E79F")
    SIG_RED       = colors.HexColor("#922B21")
    SIG_RED_BG    = colors.HexColor("#FDEDEC")
    SIG_RED_BD    = colors.HexColor("#F1948A")
    SIG_GRAY      = colors.HexColor("#566573")
    SIG_GRAY_BG   = colors.HexColor("#F2F3F4")
    SIG_GRAY_BD   = colors.HexColor("#BFC9CA")

    SCORE_HIGH    = colors.HexColor("#1E8449")
    SCORE_MID     = colors.HexColor("#2E86C1")
    SCORE_LOW     = colors.HexColor("#B7770D")
    SCORE_POOR    = colors.HexColor("#922B21")

    CHART_LINE1   = "#2E86C1"
    CHART_LINE2   = "#B7770D"
    CHART_LINE3   = "#1E8449"
    CHART_BG      = "#FFFFFF"
    CHART_GRID    = "#E8EDF2"
    CHART_AXIS    = "#8A9AB0"
    CHART_TEXT    = "#2D3748"

    T_DISPLAY = 32; T_H1 = 18; T_H2 = 14; T_H3 = 11
    T_BODY = 9.5;   T_SMALL = 8;  T_MICRO = 7

    SP_XS = 4; SP_S = 8; SP_M = 14; SP_L = 22; SP_XL = 36

    PAGE_W, PAGE_H = A4
    MARGIN_H  = 24 * mm
    MARGIN_V  = 20 * mm
    CONTENT_W = PAGE_W - 2 * MARGIN_H

    @staticmethod
    def hex(s: str) -> colors.HexColor:
        return colors.HexColor(s)

    @staticmethod
    def signal_style(signal: str) -> Tuple:
        key = signal.upper().strip()
        if any(k in key for k in ("PRIORITY", "POSITIVE")):
            return DS.SIG_GREEN, DS.SIG_GREEN_BG, DS.SIG_GREEN_BD
        if any(k in key for k in ("NEUTRAL", "FLAG")):
            return DS.SIG_AMBER, DS.SIG_AMBER_BG, DS.SIG_AMBER_BD
        if any(k in key for k in ("WATCH", "URGENT")):
            return DS.SIG_RED, DS.SIG_RED_BG, DS.SIG_RED_BD
        return DS.SIG_GRAY, DS.SIG_GRAY_BG, DS.SIG_GRAY_BD

    @staticmethod
    def score_colour(score: float) -> colors.HexColor:
        if score >= 70: return DS.SCORE_HIGH
        if score >= 50: return DS.SCORE_MID
        if score >= 35: return DS.SCORE_LOW
        return DS.SCORE_POOR

    @staticmethod
    def score_label(score: float) -> str:
        if score >= 70: return "Strong"
        if score >= 50: return "Moderate"
        if score >= 35: return "Weak"
        return "Poor"


# ================================================================
# CONSTANTS
# ================================================================

ASRE_VERSION = "4.2.0"
TOOL_NAME    = "ASRE Report Generator v4.2"

SEBI_DISCLOSURE = (
    "AI TOOL DISCLOSURE (SEBI Circular: SEBI/HO/MIRSD/MIRSD-PoD-1/P/CIR/2024/160, Dec 2024): "
    "This report is generated by ASRE, an AI-assisted quantitative research system. "
    "Scores and signals are for internal IA/RA research use only and do NOT constitute "
    "investment advice, a recommendation to buy or sell any security, or a guarantee of "
    "future returns. The system does not hold a SEBI registration as an Investment Adviser "
    "for AI tools. All investment decisions are the sole responsibility of the "
    "SEBI-registered Investment Adviser or Research Analyst reviewing this output. "
    "Data sourced from Yahoo Finance (public APIs) and may contain errors or delays. "
    "Past scores do not predict future performance."
)


# ================================================================
# FUNDAMENTAL FETCHER INTEGRATION  [NEW in V4.2]
# ================================================================

def fetch_fundamentals_for_report(
    ticker:      str,
    fetcher,                        # FundamentalFetcher instance
    price:       Optional[float] = None,
    start_date:  Optional[str]   = None,
    end_date:    Optional[str]   = None,
    use_cache:   bool            = True,
    force_refresh: bool          = False,
) -> Dict:
    """
    Fetch quarterly fundamentals via FundamentalFetcher and return a dict
    ready to be merged into analysis_data for generate_single_stock_report().

    The returned dict contains these keys (all optional in analysis_data):
        pe, roe, de, revenue_growth, profit_margin
        data_sources['fundamentals']  — with Yahoo Finance fetch_timestamp

    Parameters
    ----------
    ticker : str
        e.g. "RELIANCE.NS"
    fetcher : FundamentalFetcher
        Instance of FundamentalFetcher from fundamental_fetcher.py.
        Passed as Any to avoid a hard import dependency.
    price : float, optional
        Current price, used to compute trailing P/E ratio.
        If None, P/E is omitted.
    start_date : str, optional
        YYYY-MM-DD. Defaults to 2 years ago.
    end_date : str, optional
        YYYY-MM-DD. Defaults to today.
    use_cache : bool
        Passed through to FundamentalFetcher.
    force_refresh : bool
        Passed through to FundamentalFetcher.

    Returns
    -------
    dict
        Keys: pe, roe, de, revenue_growth, profit_margin,
              data_sources (dict with 'fundamentals' sub-dict),
              _fundamental_df (raw DataFrame, for advanced callers),
              _fetch_timestamp (datetime, for audit trail).

    Notes
    -----
    - Staleness warnings (>24 h) from FundamentalFetcher are logged and
      appended to the data_sources timestamp string so they appear in the PDF.
    - Hard RuntimeErrors (cache TTL exceeded) are re-raised immediately.
    - If any ratio calculation fails it is silently omitted (None stays out of
      analysis_data so the PDF shows "N/A" gracefully).
    """
    today        = datetime.now().strftime("%Y-%m-%d")
    # Use a generous 5-year window so FF's internal date filter never silently
    # excludes rows. We pick the latest quarters ourselves from the full df.
    five_yrs_ago = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")

    # ── Call FundamentalFetcher ────────────────────────────────
    # Re-raise RuntimeError (TTL exceeded) so CLI sees it immediately.
    df, fetch_ts = fetcher.fetch_quarterly_fundamentals(
        ticker=ticker,
        start_date=five_yrs_ago,
        end_date=today,
        use_cache=use_cache,
        force_refresh=force_refresh,
    )

    # Defensive empty check — if 5-yr window returns nothing, retry with
    # a far-back start so we catch any unusual ticker date ranges.
    _is_empty = df is None or (hasattr(df, 'empty') and df.empty)
    if _is_empty:
        logger.warning(
            "[V4.2] Empty df for %s with 5-yr window — retrying from 2000-01-01", ticker
        )
        try:
            df, fetch_ts = fetcher.fetch_quarterly_fundamentals(
                ticker=ticker,
                start_date="2000-01-01",
                end_date=today,
                use_cache=use_cache,
                force_refresh=True,
            )
            _is_empty = df is None or (hasattr(df, 'empty') and df.empty)
        except Exception as retry_exc:
            logger.warning("[V4.2] Extended retry also failed for %s: %s", ticker, retry_exc)

    result: Dict = {
        "_fundamental_df":  df,
        "_fetch_timestamp": fetch_ts,
    }

    if _is_empty:
        logger.warning("[V4.2] FundamentalFetcher returned empty DataFrame for %s "
                       "— fundamental metrics will not appear in PDF", ticker)
        result["data_sources"] = {
            "fundamentals": {
                "source":    "Yahoo Finance (Quarterly) via FundamentalFetcher v2.0",
                "timestamp": (fetch_ts.strftime("%Y-%m-%d %H:%M IST")
                              if fetch_ts else "N/A") + " (no data returned)",
            }
        }
        return result

    # ── Use latest quarter for ratios ─────────────────────────
    # Sort ascending so iloc[-1] is always the most recent quarter
    if 'date' in df.columns:
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
    latest = df.iloc[-1]
    prev   = df.iloc[-5] if len(df) >= 5 else None   # same quarter prior year

    logger.info(
        "[V4.2] %s: %d quarters loaded. Latest quarter end: %s  "
        "Announced: %s  Revenue: %.0f  Net income: %.0f",
        ticker, len(df),
        latest.get("date", "?"), latest.get("announced_date", "?"),
        float(latest.get("revenue", 0) or 0),
        float(latest.get("net_income", 0) or 0),
    )

    # ── Staleness flag for PDF ─────────────────────────────────
    hours_old  = (datetime.now() - fetch_ts).total_seconds() / 3600
    stale_note = " [STALE >24h — re-fetch recommended]" if hours_old > 24 else ""
    ts_str     = fetch_ts.strftime("%Y-%m-%d %H:%M IST") + stale_note

    # ── Derived ratios ─────────────────────────────────────────

    # Trailing P/E  = price / (annualised EPS from last 4 quarters)
    if price is not None and len(df) >= 4:
        try:
            trailing_eps = float(df.tail(4)["eps"].sum())
            if trailing_eps > 0:
                result["pe"] = round(price / trailing_eps, 2)
        except Exception as exc:
            logger.debug("P/E calculation skipped: %s", exc)

    # ── Helpers for multi-alias column resolution ──────────────
    def _ff_col(row: pd.Series, *aliases) -> Optional[float]:
        """Return first non-null, non-zero value from alias list (single row)."""
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

    def _ff_scan(frame: pd.DataFrame, *aliases) -> Optional[float]:
        """
        Scan all rows backwards for the most recent non-null, non-zero value.
        Needed because yfinance income-statement quarterly data often stores
        balance-sheet columns (shareholders_equity, total_debt) as zero or NaN
        in the most recent row even when they exist in earlier rows.
        """
        for alias in aliases:
            if alias not in frame.columns:
                continue
            series = frame[alias].dropna()
            series = series[series != 0.0]
            if len(series) > 0:
                try:
                    return float(series.iloc[-1])
                except (TypeError, ValueError):
                    pass
        return None

    # Layer 1 — FF pre-computed ratio columns
    # FundamentalFetcher logs "Calculated derived metrics: PE, ROE, D/E, Growth"
    # meaning it may already store these as lowercase columns in the returned df.
    _l1_roe = _ff_col(latest,
        'roe', 'return_on_equity', 'return_on_equity_pct', 'roe_pct')
    if _l1_roe is not None:
        result["roe"] = round(_l1_roe, 2)

    _l1_de = _ff_col(latest,
        'de', 'd_e', 'debt_equity', 'debt_to_equity', 'debt_equity_ratio')
    if _l1_de is not None:
        result["de"] = round(_l1_de, 2)

    # Layer 2 — yf.Ticker(ticker).info (covers most NSE tickers including PSU/energy)
    # FF computes ROE and D/E internally via yf.info but doesn't always store them
    # back in the quarterly DataFrame (IOC.NS, ONGC.NS etc. are known cases).
    _yf_info: Dict = {}
    try:
        import yfinance as _yf
        _yf_info = _yf.Ticker(ticker).info or {}
    except Exception as _yfi_exc:
        logger.debug("[V4.2] yf.info fetch failed for %s: %s", ticker, _yfi_exc)

    if "roe" not in result:
        _l2_roe = _yf_info.get("returnOnEquity")
        if _l2_roe is not None and pd.notna(_l2_roe):
            result["roe"] = round(float(_l2_roe) * 100, 2)   # stored as fraction
            logger.info("[V4.2] ROE from yf.info: %.2f%%", result["roe"])

    if "de" not in result:
        _l2_de = _yf_info.get("debtToEquity")
        if _l2_de is not None and pd.notna(_l2_de):
            _de_raw = float(_l2_de)
            # yf.info debtToEquity is sometimes percent-form (74.0 = 0.74)
            result["de"] = round(_de_raw / 100 if _de_raw > 10 else _de_raw, 2)
            logger.info("[V4.2] D/E from yf.info: %.2f", result["de"])

    # Also grab yf.info P/E here — used below to replace unreliable TTM EPS PE
    _yf_pe = _yf_info.get("trailingPE") or _yf_info.get("forwardPE")

    # Layer 3 — derive from raw DataFrame columns (multi-alias backward scan)
    # equity_val uses _ff_scan (not just latest row) because balance-sheet
    # columns are often NaN in the most recent income-statement quarter row.
    _equity_val = _ff_scan(df,
        "shareholders_equity", "stockholders_equity",
        "total_stockholder_equity", "total_equity",
        "common_stock_equity", "equity",
        "stockholders_equity_total",
        "total_equity_gross_minority_interest")

    _debt_val = _ff_scan(df,
        "total_debt", "long_term_debt",
        "total_long_term_debt",
        "long_term_debt_and_capital_lease_obligation",
        "current_debt", "short_long_term_debt_total")

    _ni_series = next(
        (df[c] for c in ("net_income", "net_income_common_stockholders",
                          "net_income_including_noncontrolling_interests")
         if c in df.columns), None)

    if "roe" not in result and _equity_val is not None and _ni_series is not None:
        try:
            annual_ni = float(_ni_series.tail(4).sum())
            result["roe"] = round((annual_ni / _equity_val) * 100, 2)
        except Exception as exc:
            logger.debug("[V4.2] ROE layer-3 derivation skipped: %s", exc)

    if "de" not in result and _equity_val is not None and _debt_val is not None:
        try:
            result["de"] = round(_debt_val / _equity_val, 2)
        except Exception as exc:
            logger.debug("[V4.2] D/E layer-3 derivation skipped: %s", exc)

    # Layer 4 — graceful N/A: flag missing ratios so PDF footnote can fire
    _missing_ratios = [k for k in ("pe", "roe", "de") if k not in result]
    if _missing_ratios:
        result["_missing_ratio_flag"] = True
        logger.info(
            "[V4.2] %s: ratios still missing after 4-layer chain — %s. "
            "PDF will show 'N/A' with footnote.",
            ticker, _missing_ratios,
        )

    # ── P/E — apply yf.info override when derived value is unreliable ──
    # A near-zero TTM EPS quarter can inflate the calculated PE to absurd values.
    # Replace with yf.info trailingPE when our value exceeds 200x.
    if _yf_pe is not None and pd.notna(_yf_pe):
        _yf_pe_f = float(_yf_pe)
        if 0 < _yf_pe_f < 200:
            if "pe" not in result or result.get("pe", 0) > 200:
                result["pe"] = round(_yf_pe_f, 2)
                logger.info("[V4.2] PE from yf.info: %.2f", result["pe"])

    # Revenue Growth YoY  = (latest_revenue / same_quarter_prev_year) - 1
    try:
        if prev is not None:
            rev_now  = float(latest.get("revenue", 0) or 0)
            rev_prev = float(prev.get("revenue",  0) or 0)
            if rev_prev > 0:
                result["revenue_growth"] = round(
                    ((rev_now - rev_prev) / rev_prev) * 100, 2
                )
    except Exception as exc:
        logger.debug("Revenue growth calculation skipped: %s", exc)

    # Profit Margin  = net_income / revenue  (latest quarter)
    try:
        rev = float(latest.get("revenue",    0) or 0)
        ni  = float(latest.get("net_income", 0) or 0)
        if rev > 0:
            result["profit_margin"] = round((ni / rev) * 100, 2)
    except Exception as exc:
        logger.debug("Profit margin calculation skipped: %s", exc)

    # ── Supplementary raw fields (Additional Data table on Page 3) ──
    # _equity_val and _debt_val use _ff_scan so they reflect the most-recently
    # reported balance-sheet value even when the latest income-statement row is NaN.
    try:
        result["_eps"]               = float(_ff_col(latest, "eps", "basic_eps",
                                                      "diluted_eps") or 0)
        result["_revenue"]           = float(_ff_col(latest, "revenue",
                                                      "total_revenue", "revenues") or 0)
        result["_net_income"]        = float(_ff_col(latest, "net_income",
                                                      "net_income_common_stockholders") or 0)
        result["_free_cash_flow"]    = float(_ff_col(latest, "free_cash_flow",
                                                      "free_cash_flow_to_firm", "fcf") or 0)
        result["_total_debt"]        = float(_debt_val or 0)
        result["_shareholders_equity"] = float(_equity_val or 0)
        # Fill equity from yf.info book value × shares if still zero
        if result["_shareholders_equity"] == 0:
            _bv     = _yf_info.get("bookValue")
            _shares = (_yf_info.get("sharesOutstanding")
                       or _yf_info.get("impliedSharesOutstanding"))
            if _bv and _shares:
                result["_shareholders_equity"] = round(float(_bv) * float(_shares), 0)
        result["_announced_date"]    = str(latest.get("announced_date", "N/A"))
        result["_quarter_end_date"]  = str(latest.get("date", "N/A"))
    except Exception as _raw_exc:
        logger.debug("[V4.2] Raw fields extraction failed: %s", _raw_exc)

    # ── data_sources entry (shown in PDF Data Sources table) ───
    result["data_sources"] = {
        "fundamentals": {
            "source":    "Yahoo Finance (Quarterly) via FundamentalFetcher v2.0",
            "timestamp": ts_str,
        }
    }

    logger.info(
        "[V4.2] Fundamentals for %s: PE=%.1f, ROE=%.1f%%, DE=%.2f, "
        "RevGrowth=%.1f%%, Margin=%.1f%%",
        ticker,
        result.get("pe", float("nan")),
        result.get("roe", float("nan")),
        result.get("de", float("nan")),
        result.get("revenue_growth", float("nan")),
        result.get("profit_margin", float("nan")),
    )

    return result


# ================================================================
# FLOWABLES  — preserved exactly from V4.1
# ================================================================

class ThinRule(Flowable):
    def __init__(self, width=None, colour=None, thickness=0.5):
        super().__init__()
        self.width     = width or DS.CONTENT_W
        self.colour    = colour or DS.BORDER
        self.thickness = thickness
        self.height    = 1

    def draw(self):
        c = self.canv
        c.setStrokeColor(self.colour)
        c.setLineWidth(self.thickness)
        c.line(0, 0, self.width, 0)


class SectionLabel(Flowable):
    def __init__(self, title: str, width=None):
        super().__init__()
        self.title = title.upper()
        self.width = width or DS.CONTENT_W
        self.height = 22

    def draw(self):
        c = self.canv
        c.setFillColor(DS.BRAND)
        c.rect(0, 4, 3, 14, fill=1, stroke=0)
        c.setFillColor(DS.BRAND)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(10, 8, self.title)
        c.setStrokeColor(DS.BORDER)
        c.setLineWidth(0.5)
        text_w = len(self.title) * 5.5 + 15
        c.line(text_w, 11, self.width, 11)


class SignalPill(Flowable):
    def __init__(self, signal: str, score: float, label: str = "", width=None):
        super().__init__()
        self.signal = signal
        self.score  = score
        self.label  = label
        self.width  = width or DS.CONTENT_W
        self.height = 56

    def draw(self):
        c = self.canv
        txt_col, bg_col, bd_col = DS.signal_style(self.signal)
        c.setFillColor(bg_col)
        c.setStrokeColor(bd_col)
        c.setLineWidth(1)
        c.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=1)
        c.setFillColor(txt_col)
        c.rect(0, 0, 4, self.height, fill=1, stroke=0)
        c.setFillColor(DS.HEADING)
        c.setFont("Helvetica-Bold", 22)
        c.drawString(16, 28, f"{self.score:.1f}")
        c.setFillColor(DS.MUTED)
        c.setFont("Helvetica", 9)
        c.drawString(16, 18, "/ 100")
        c.setFillColor(txt_col)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(72, 34, self.signal)
        if self.label:
            c.setFillColor(DS.MUTED)
            c.setFont("Helvetica", 8)
            c.drawString(72, 20, self.label)
        c.setFillColor(txt_col)
        c.setFont("Helvetica-Bold", 8)
        c.drawRightString(self.width - 10, 34, DS.score_label(self.score))


class ComponentBar(Flowable):
    def __init__(self, label: str, score: float, weight: float, width=None):
        super().__init__()
        self.label  = label
        self.score  = score
        self.weight = weight
        self.width  = width or DS.CONTENT_W
        self.height = 32

    def draw(self):
        c      = self.canv
        col    = DS.score_colour(self.score)
        bar_x  = 110
        bar_w  = self.width - bar_x - 70
        fill_w = bar_w * (self.score / 100)
        c.setFillColor(DS.HEADING)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(0, 12, self.label)
        c.setFillColor(DS.SURFACE)
        c.setStrokeColor(DS.BORDER)
        c.setLineWidth(0.5)
        c.roundRect(bar_x, 9, bar_w, 11, 3, fill=1, stroke=1)
        if fill_w > 4:
            c.setFillColor(col)
            c.roundRect(bar_x, 9, fill_w, 11, 3, fill=1, stroke=0)
        c.setFillColor(DS.HEADING)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(bar_x + bar_w + 8, 9, f"{self.score:.0f}")
        c.setFillColor(DS.MUTED)
        c.setFont("Helvetica", 8)
        c.drawString(bar_x + bar_w + 26, 9, "/ 100")
        c.setFillColor(DS.MUTED)
        c.setFont("Helvetica", 7)
        c.drawRightString(self.width, 9, f"Weight {self.weight:.0%}")
        c.setFillColor(DS.score_colour(self.score))
        c.setFont("Helvetica", 7)
        c.drawString(bar_x, 2, DS.score_label(self.score))


class DivergenceNote(Flowable):
    def __init__(self, r_final: float, r_asre: float, width=None):
        super().__init__()
        self.r_final = r_final
        self.r_asre  = r_asre
        self.width   = width or DS.CONTENT_W
        self.height  = 46

    def draw(self):
        c      = self.canv
        gap    = self.r_final - self.r_asre
        bd_col = DS.SIG_AMBER_BD if gap > 0 else DS.SIG_GREEN_BD
        bg_col = DS.SIG_AMBER_BG if gap > 0 else DS.SIG_GREEN_BG
        tx_col = DS.SIG_AMBER    if gap > 0 else DS.SIG_GREEN
        c.setFillColor(bg_col)
        c.setStrokeColor(bd_col)
        c.setLineWidth(1)
        c.roundRect(0, 0, self.width, self.height, 3, fill=1, stroke=1)
        c.setFillColor(tx_col)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(12, 28, "Score Divergence")
        direction = "above" if gap > 0 else "below"
        msg = (
            f"R_Final ({self.r_final:.0f}) is {abs(gap):.0f} pts {direction} "
            f"R_ASRE ({self.r_asre:.0f}). "
        )
        msg += ("Composite more constructive than risk-parity — review before acting."
                if gap > 0 else
                "Momentum/technicals recovering ahead of composite — monitor for confirmation.")
        c.setFillColor(DS.BODY_TEXT)
        c.setFont("Helvetica", 8)
        if len(msg) > 110:
            c.drawString(12, 18, msg[:110])
            c.drawString(12,  8, msg[110:])
        else:
            c.drawString(12, 14, msg)


class DipQualityBar(Flowable):
    TIERS = [
        ("HIGH QUALITY", 75, DS.SCORE_HIGH),
        ("GOOD",         60, DS.SCORE_MID),
        ("MARGINAL",     45, DS.SCORE_LOW),
        ("POOR",          0, DS.SCORE_POOR),
    ]

    def __init__(self, score: float, width=None):
        super().__init__()
        self.score  = score
        self.width  = width or DS.CONTENT_W
        self.height = 36

    def draw(self):
        c      = self.canv
        step_w = (self.width - 6) / 4
        for i, (label, threshold, col) in enumerate(self.TIERS):
            x = i * (step_w + 2)
            active = (
                (self.score >= 75 and label == "HIGH QUALITY") or
                (60 <= self.score < 75 and label == "GOOD") or
                (45 <= self.score < 60 and label == "MARGINAL") or
                (self.score < 45 and label == "POOR")
            )
            c.setStrokeColor(col if active else DS.BORDER)
            c.setLineWidth(1.2 if active else 0.5)
            c.setFillColor(col if active else DS.NEAR_WHITE)
            c.roundRect(x, 10, step_w, 20, 3, fill=1, stroke=1)
            c.setFillColor(DS.WHITE if active else DS.MUTED)
            c.setFont("Helvetica-Bold" if active else "Helvetica", 6.5)
            c.drawCentredString(x + step_w / 2, 19, label)
            c.setFillColor(DS.MUTED)
            c.setFont("Helvetica", 6)
            c.drawCentredString(x + step_w / 2, 4, f">= {threshold}")


class IARIANotesBlock(Flowable):
    """
    Structured IA / RIA analyst notes form — preserved from V4.1.
    Navy header bar, amber border, cream background.
    Sections: Reviewer Details | Signal Agreement | Analysis Notes |
              Action Taken | Follow-up | Authorisation.
    """
    _PAD = 10; _LH = 15; _NL = 5

    def __init__(self, notes_text: str = "", width=None):
        super().__init__()
        self.notes_text = notes_text or ""
        self.width      = width or DS.CONTENT_W
        self.height = (
            self._PAD + 20 + self._PAD
            + 10 + self._LH + self._PAD
            + 10 + self._LH + self._PAD
            + 10 + self._LH * max(self._NL, len(self.notes_text.splitlines()) + 1)
            + self._PAD
            + 10 + self._LH + self._PAD
            + 10 + self._LH + self._PAD
            + 10 + self._LH + 8 + self._PAD
        )

    def _sec_label(self, c, x, y, text):
        c.setFillColor(DS.BRAND)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(x, y, text)

    def _underline_field(self, c, x, y, label, field_w, value=""):
        c.setFillColor(DS.MUTED)
        c.setFont("Helvetica", 6.5)
        c.drawString(x, y + self._LH - 3, label)
        c.setStrokeColor(DS.BORDER)
        c.setLineWidth(0.7)
        c.line(x, y, x + field_w, y)
        if value:
            c.setFillColor(DS.BODY_TEXT)
            c.setFont("Helvetica-Oblique", 7.5)
            c.drawString(x + 2, y + 2, value[:max(int(field_w / 4.5), 1)])

    def _checkbox(self, c, x, y, label, size=7):
        c.setStrokeColor(DS.BORDER)
        c.setLineWidth(0.7)
        c.setFillColor(DS.WHITE)
        c.rect(x, y + 1, size, size, fill=1, stroke=1)
        c.setFillColor(DS.BODY_TEXT)
        c.setFont("Helvetica", 7)
        c.drawString(x + size + 3, y + 2, label)

    def draw(self):
        c  = self.canv
        W  = self.width
        P  = self._PAD
        LH = self._LH

        # Card
        c.setFillColor(DS.SIG_AMBER_BG)
        c.setStrokeColor(DS.SIG_AMBER_BD)
        c.setLineWidth(1.2)
        c.roundRect(0, 0, W, self.height, 5, fill=1, stroke=1)
        c.setFillColor(DS.SIG_AMBER)
        c.rect(0, 0, 5, self.height, fill=1, stroke=0)

        # Header
        hdr_y = self.height - 20
        c.setFillColor(DS.BRAND)
        c.roundRect(0, hdr_y, W, 20, 5, fill=1, stroke=0)
        c.rect(0, hdr_y, W, 10, fill=1, stroke=0)
        c.setFillColor(DS.WHITE)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(14, hdr_y + 6, "IA / RIA ANALYST NOTES")
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor("#90B8D8"))
        c.drawRightString(W - P, hdr_y + 6, "Complete before filing or client communication")

        y = hdr_y - P

        # 1 — Reviewer Details
        y -= 10; self._sec_label(c, 14, y, "REVIEWER DETAILS"); y -= LH
        iw = W - 14 - P
        fw_d, fw_n, fw_r = iw*0.22, iw*0.45, iw*0.28
        self._underline_field(c, 14,                 y, "Date",          fw_d)
        self._underline_field(c, 14+fw_d+P,          y, "Analyst Name",  fw_n)
        self._underline_field(c, 14+fw_d+P+fw_n+P,  y, "SEBI Reg No",   fw_r)
        y -= P

        # 2 — Signal Agreement
        y -= 10; self._sec_label(c, 14, y, "SIGNAL AGREEMENT"); y -= LH
        cb_w = (W - 14 - P) / 4
        for idx, opt in enumerate(["Agree with signal", "Partially agree",
                                    "Override (see notes)", "Defer — need more data"]):
            self._checkbox(c, 14 + idx*cb_w, y, opt)
        y -= P

        # 3 — Analysis Notes
        y -= 10; self._sec_label(c, 14, y, "ANALYSIS NOTES")
        nl   = max(self._NL, len(self.notes_text.splitlines()) + 1)
        txts = self.notes_text.splitlines() if self.notes_text else []
        for i in range(nl):
            ly = y - (i+1)*LH
            c.setStrokeColor(DS.BORDER); c.setLineWidth(0.5)
            c.line(14, ly, W-P, ly)
            if i < len(txts):
                c.setFillColor(DS.BODY_TEXT)
                c.setFont("Helvetica-Oblique", 8)
                c.drawString(16, ly+3, txts[i][:max(int((W-P-14)/4.5), 1)])
        y -= nl*LH; y -= P

        # 4 — Action Taken
        y -= 10; self._sec_label(c, 14, y, "ACTION TAKEN"); y -= LH
        act_w = (W - 14 - P) / 4
        for idx, act in enumerate(["No action — monitor", "Client briefed",
                                    "Position adjusted", "Escalated to compliance"]):
            self._checkbox(c, 14 + idx*act_w, y, act)
        y -= P

        # 5 — Follow-up
        y -= 10; self._sec_label(c, 14, y, "FOLLOW-UP"); y -= LH
        half = iw*0.42
        self._underline_field(c, 14, y, "Follow-up Date", half)
        ci_x = 14 + half + P*2
        c.setFillColor(DS.MUTED); c.setFont("Helvetica", 6.5)
        c.drawString(ci_x, y+LH-3, "Client Informed?")
        self._checkbox(c, ci_x,     y, "Yes")
        self._checkbox(c, ci_x+36,  y, "No")
        y -= P

        # 6 — Authorisation
        y -= 10; self._sec_label(c, 14, y, "AUTHORISATION"); y -= LH+4
        sig_w = W*0.46
        c.setStrokeColor(DS.MUTED); c.setLineWidth(0.8)
        c.line(14, y, 14+sig_w, y)
        c.setFillColor(DS.MUTED); c.setFont("Helvetica", 6.5)
        c.drawString(14, y-9, "Signature of SEBI-Registered IA / RA")
        stx = 14+sig_w+P; stw = W-stx-P
        c.setStrokeColor(DS.BORDER); c.setLineWidth(0.6)
        c.setFillColor(DS.WHITE)
        c.roundRect(stx, y-4, stw, LH+6, 3, fill=1, stroke=1)
        c.setFillColor(DS.MUTED); c.setFont("Helvetica", 6.5)
        c.drawCentredString(stx+stw/2, y+2, "Official Stamp / Seal (if applicable)")


# ================================================================
# CHART FACTORIES  — unchanged from V4.1
# ================================================================

def _clean_style():
    plt.rcParams.update({
        'figure.facecolor': DS.CHART_BG,  'axes.facecolor':  DS.CHART_BG,
        'axes.edgecolor':   DS.CHART_GRID, 'axes.labelcolor': DS.CHART_AXIS,
        'xtick.color':      DS.CHART_AXIS, 'ytick.color':     DS.CHART_AXIS,
        'grid.color':       DS.CHART_GRID, 'grid.linewidth':  0.7,
        'text.color':       DS.CHART_TEXT, 'font.family':     'sans-serif',
        'font.size':        8,
        'axes.spines.top':  False,         'axes.spines.right': False,
    })


def _chart_buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def create_score_overview_chart(f, t, m, r_final, r_asre) -> io.BytesIO:
    _clean_style()
    fig, (ax_bars, ax_ring) = plt.subplots(
        1, 2, figsize=(7.6, 2.6),
        gridspec_kw={'width_ratios': [3, 1.2]}, facecolor=DS.CHART_BG,
    )
    labels  = ["M  Momentum", "T  Technical", "F  Fundamentals", "R_Final", "R_ASRE"]
    values  = [m, t, f, r_final, r_asre]
    colours = [("#1E8449" if v >= 70 else "#2E86C1" if v >= 50
                else "#B7770D" if v >= 35 else "#C0392B") for v in values]
    ax_bars.barh(list(range(5)), [100]*5, color=DS.CHART_GRID, height=0.55,
                 zorder=0, edgecolor='white', linewidth=0.5)
    bars = ax_bars.barh(list(range(5)), values, color=colours, height=0.55,
                        zorder=1, edgecolor='white', linewidth=0.5)
    ax_bars.set_yticks(list(range(5)))
    ax_bars.set_yticklabels(labels, fontsize=8)
    ax_bars.set_xlim(0, 110)
    ax_bars.set_xticks([0, 35, 50, 70, 100])
    ax_bars.set_xticklabels(['0','35','50','70','100'], fontsize=7)
    ax_bars.set_xlabel("Score / 100", fontsize=7.5, color=DS.CHART_AXIS)
    ax_bars.grid(axis='x', alpha=0.6, linewidth=0.6)
    ax_bars.spines['left'].set_visible(False)
    ax_bars.spines['bottom'].set_color(DS.CHART_GRID)
    for xref in [35, 50, 70]:
        ax_bars.axvline(xref, color=DS.CHART_AXIS, linewidth=0.5,
                        linestyle='--', alpha=0.4)
    for bar, val in zip(bars, values):
        ax_bars.text(val+1.5, bar.get_y()+bar.get_height()/2, f"{val:.0f}",
                     va='center', fontsize=7.5, color=DS.CHART_TEXT, fontweight='bold')
    ax_bars.set_title("Score Summary", fontsize=9, color=DS.CHART_TEXT,
                      fontweight='bold', pad=8, loc='left')
    r_col = ("#1E8449" if r_asre >= 70 else "#2E86C1" if r_asre >= 50
             else "#B7770D" if r_asre >= 35 else "#C0392B")
    f_col = ("#1E8449" if r_final >= 70 else "#2E86C1" if r_final >= 50
             else "#B7770D" if r_final >= 35 else "#C0392B")
    ax_ring.pie([r_asre, 100-r_asre], colors=[r_col, DS.CHART_GRID],
                startangle=90, counterclock=False,
                wedgeprops=dict(width=0.32, edgecolor='white', linewidth=1.5))
    ax_ring.text(0,  0.15, f"{r_asre:.0f}", ha='center', va='center',
                 fontsize=15, fontweight='bold', color=r_col)
    ax_ring.text(0, -0.22, "R_ASRE", ha='center', va='center',
                 fontsize=7, color=DS.CHART_AXIS)
    ax_ring.text(0, -0.42, f"R_Final  {r_final:.0f}", ha='center', va='center',
                 fontsize=7, color=f_col, fontweight='bold')
    ax_ring.set_title("Composite", fontsize=8.5, color=DS.CHART_TEXT,
                      fontweight='bold', pad=8)
    plt.tight_layout(pad=0.8)
    return _chart_buf(fig)


def create_price_chart(dates, prices, sma200=None, ticker="") -> io.BytesIO:
    _clean_style()
    fig, ax = plt.subplots(figsize=(7.6, 2.8), facecolor=DS.CHART_BG)
    dates_dt = pd.to_datetime(dates)
    ax.plot(dates_dt, prices, color=DS.CHART_LINE1, linewidth=1.6, label='Price', zorder=3)
    if sma200 is not None and not sma200.dropna().empty:
        ax.plot(dates_dt, sma200, color=DS.CHART_LINE2, linewidth=1.2,
                linestyle='--', alpha=0.85, label='SMA-200', zorder=2)
    ax.fill_between(dates_dt, prices, prices.min()*0.998,
                    color=DS.CHART_LINE1, alpha=0.06, zorder=1)
    ax.grid(True, alpha=0.5, linewidth=0.6)
    ax.set_ylabel('Price (INR)', fontsize=8, color=DS.CHART_AXIS)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9, edgecolor=DS.CHART_GRID)
    ax.set_title(f'{ticker}  --  Price History (Last 120 Days)',
                 fontsize=9, color=DS.CHART_TEXT, pad=8, loc='left', fontweight='bold')
    fig.autofmt_xdate(rotation=30, ha='right')
    plt.tight_layout(pad=0.8)
    return _chart_buf(fig)


def create_trend_sparkline(rows: List[Dict]) -> io.BytesIO:
    _clean_style()
    fig, ax = plt.subplots(figsize=(7.6, 1.9), facecolor=DS.CHART_BG)
    dates  = [str(r['date'])[:10] for r in rows]
    rasres = [r['r_asre']  for r in rows]
    finals = [r['r_final'] for r in rows]
    x = range(len(rows))
    ax.axhspan(70, 100, alpha=0.04, color=DS.CHART_LINE3, zorder=0)
    ax.axhspan(50, 70,  alpha=0.03, color=DS.CHART_LINE1, zorder=0)
    ax.fill_between(x, rasres, alpha=0.08, color=DS.CHART_LINE1)
    ax.plot(x, rasres, color=DS.CHART_LINE1, linewidth=1.8,
            label='R_ASRE', marker='o', markersize=3.5, zorder=3)
    ax.plot(x, finals,  color=DS.CHART_LINE2, linewidth=1.2,
            linestyle='--', label='R_Final', alpha=0.8, zorder=2)
    for ref, lbl, col in [(70,"70","#1E8449"),(50,"50","#2E86C1"),(35,"35","#B7770D")]:
        ax.axhline(ref, color=col, linewidth=0.6, linestyle=':', alpha=0.5)
        ax.text(len(rows)-0.3, ref+1, lbl, fontsize=6, color=col, va='bottom')
    ax.set_ylim(0, 100)
    ax.set_xticks(list(x))
    ax.set_xticklabels(dates, rotation=30, ha='right', fontsize=6.5)
    ax.set_yticks([35, 50, 70])
    ax.set_yticklabels(['35','50','70'], fontsize=7)
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9,
              edgecolor=DS.CHART_GRID, ncol=2)
    ax.set_title("Score Trend", fontsize=8.5, color=DS.CHART_TEXT,
                 pad=6, loc='left', fontweight='bold')
    plt.tight_layout(pad=0.5)
    return _chart_buf(fig)


# ================================================================
# HELPERS  — unchanged from V4.1
# ================================================================

def _plain_english_summary(ticker, signal, r_final, r_asre,
                            f_score, t_score, m_score, tier,
                            market_context=None) -> str:
    divergence = r_final - r_asre
    div_note   = ""
    if abs(divergence) >= 10:
        direction = "above" if divergence > 0 else "below"
        div_note  = (
            f" R_Final ({r_final:.0f}) is {abs(divergence):.0f} pts {direction} "
            f"R_ASRE ({r_asre:.0f}) -- the IA should review this divergence before acting."
        )
    f_tone = "strong" if f_score >= 60 else "moderate" if f_score >= 45 else "weak"
    t_tone = ("a strong uptrend" if t_score >= 60 else
              "a neutral trend"  if t_score >= 40 else "a downtrend")
    m_tone = ("positive momentum" if m_score >= 60 else
              "fading momentum"   if m_score >= 30 else "weak momentum")
    mc_note = ""
    if market_context:
        stage = market_context.get("stage", "")
        dist  = market_context.get("sma200_dist", 0)
        if "STRUCTURAL" in stage.upper():
            mc_note = f" Price is {abs(dist):.1f}% below SMA-200, indicating a structural breakdown."
        elif "DEEP" in stage.upper():
            mc_note = f" Price is {abs(dist):.1f}% below SMA-200 -- deep dip territory, elevated risk."
        elif "EARLY" in stage.upper():
            mc_note = f" Price is {abs(dist):.1f}% below SMA-200 -- early dip stage."
    return (
        f"{ticker} shows {f_tone} fundamentals (F = {f_score:.0f}/100), "
        f"{t_tone} (T = {t_score:.0f}/100), and {m_tone} (M = {m_score:.0f}/100). "
        f"The composite R_Final score is {r_final:.1f}/100 and the risk-parity "
        f"R_ASRE is {r_asre:.1f}/100. System signal: {signal} (Tier {tier})."
        f"{div_note}{mc_note}"
    )


def _report_hash(ticker: str, data: Dict) -> str:
    payload = (
        f"{ticker}|{data.get('date')}|{data.get('r_final')}|"
        f"{data.get('r_asre')}|{data.get('signal')}|{data.get('tier')}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16].upper()


# ================================================================
# PARAGRAPH STYLES  — unchanged from V4.1
# ================================================================

def _build_styles() -> dict:
    base = getSampleStyleSheet()
    S    = {}

    def add(name, **kw):
        S[name] = ParagraphStyle(name, parent=base['Normal'], **kw)

    add('cover_ticker', fontSize=DS.T_DISPLAY, textColor=DS.INK,
        fontName='Helvetica-Bold', alignment=TA_LEFT, spaceBefore=0, spaceAfter=6, leading=36)
    add('cover_sub',  fontSize=13, textColor=DS.MUTED, fontName='Helvetica',
        alignment=TA_LEFT, spaceAfter=0)
    add('cover_meta', fontSize=9,  textColor=DS.MUTED, fontName='Helvetica',
        alignment=TA_LEFT, spaceAfter=4)
    add('body',       fontSize=DS.T_BODY, textColor=DS.BODY_TEXT, fontName='Helvetica',
        alignment=TA_JUSTIFY, leading=15, spaceAfter=DS.SP_S)
    add('body_bold',  fontSize=DS.T_BODY, textColor=DS.HEADING,
        fontName='Helvetica-Bold', spaceAfter=DS.SP_XS)
    add('caption',    fontSize=DS.T_SMALL, textColor=DS.MUTED,
        fontName='Helvetica', alignment=TA_LEFT, spaceAfter=2)
    add('th',         fontSize=DS.T_SMALL, textColor=DS.HEADING,
        fontName='Helvetica-Bold', alignment=TA_LEFT, leading=12)
    add('td',         fontSize=DS.T_BODY,  textColor=DS.BODY_TEXT,
        fontName='Helvetica', alignment=TA_LEFT, leading=13)
    add('td_center',  fontSize=DS.T_BODY,  textColor=DS.BODY_TEXT,
        fontName='Helvetica', alignment=TA_CENTER, leading=13)
    add('td_num',     fontSize=DS.T_BODY,  textColor=DS.HEADING,
        fontName='Helvetica-Bold', alignment=TA_RIGHT, leading=13)
    add('disclosure', fontSize=DS.T_MICRO, textColor=DS.MUTED,
        fontName='Helvetica', alignment=TA_JUSTIFY, leading=10.5, spaceAfter=2)
    add('ia_log',     fontSize=DS.T_BODY,  textColor=DS.BODY_TEXT,
        fontName='Helvetica-Oblique', leading=15, spaceAfter=4,
        leftIndent=6, rightIndent=6)
    add('port_title', fontSize=22, textColor=DS.INK,
        fontName='Helvetica-Bold', alignment=TA_LEFT, spaceBefore=4, spaceAfter=4)
    add('port_meta',  fontSize=9,  textColor=DS.MUTED,
        fontName='Helvetica', alignment=TA_LEFT)
    return S


# ================================================================
# PAGE CHROME  — unchanged from V4.1
# ================================================================

def _page_callbacks(ticker: str, report_hash: str, report_date: str):
    PW = DS.PAGE_W; PH = DS.PAGE_H; MH = DS.MARGIN_H

    def _footer(c, doc):
        c.setStrokeColor(DS.SIG_AMBER_BD); c.setLineWidth(0.8)
        c.line(0, 22, PW, 22)
        c.setFont('Helvetica', 5.5); c.setFillColor(DS.MUTED)
        c.drawString(MH, 10,
            f"{TOOL_NAME}  |  v{ASRE_VERSION}  |  "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M IST')}  |  SHA: {report_hash}")
        c.setFont('Helvetica-Bold', 5.5); c.setFillColor(DS.SIG_AMBER)
        c.drawRightString(PW-MH, 10,
            "IA/RA INTERNAL RESEARCH ONLY -- NOT INVESTMENT ADVICE -- SEBI DEC 2024")
        c.setFont('Helvetica', 6); c.setFillColor(DS.MUTED)
        c.drawCentredString(PW/2, 10, f"Page {doc.page}")

    def on_cover(c, doc):
        c.saveState(); _footer(c, doc); c.restoreState()

    def on_inner(c, doc):
        c.saveState()
        c.setFillColor(DS.BRAND); c.rect(0, PH-20, PW, 20, fill=1, stroke=0)
        c.setFillColor(DS.WHITE); c.setFont('Helvetica-Bold', 8.5)
        c.drawString(MH, PH-13, f"ASRE  |  {ticker}")
        c.setFont('Helvetica', 8)
        c.setFillColor(colors.HexColor("#90B8D8"))
        c.drawRightString(PW-MH, PH-13, f"{report_date}  |  Internal Research Only")
        _footer(c, doc); c.restoreState()

    return on_cover, on_inner


def _draw_cover_bg(ticker, tier, report_date, signal, r_asre, r_final):
    PW = DS.PAGE_W; PH = DS.PAGE_H; MH = DS.MARGIN_H
    txt_col, bg_col, bd_col = DS.signal_style(signal)

    def draw(c, doc):
        c.saveState()
        c.setFillColor(DS.WHITE); c.rect(0, 0, PW, PH, fill=1, stroke=0)
        c.setFillColor(DS.BRAND); c.rect(0, PH-52, PW, 52, fill=1, stroke=0)
        c.setFont('Helvetica-Bold', 13); c.setFillColor(DS.WHITE)
        c.drawString(MH, PH-28, "ASRE")
        c.setFont('Helvetica', 9)
        c.setFillColor(colors.HexColor("#90B8D8"))
        c.drawString(MH+48, PH-27, "Advanced Stock Rating Engine")
        c.drawRightString(PW-MH, PH-27, report_date)
        c.setFont('Helvetica-Bold', 7); c.setFillColor(DS.SIG_AMBER)
        c.drawRightString(PW-MH, PH-42, "FOR SEBI-REGISTERED IA/RA INTERNAL USE ONLY")
        c.setFont('Helvetica-Bold', 48); c.setFillColor(DS.INK)
        c.drawString(MH, PH*0.66, ticker)
        c.setStrokeColor(DS.BRAND); c.setLineWidth(2)
        c.line(MH, PH*0.64, MH+len(ticker)*28, PH*0.64)
        c.setFont('Helvetica', 12); c.setFillColor(DS.MUTED)
        c.drawString(MH, PH*0.60, f"Quality Tier {tier}  \u00b7  {report_date}")
        cx=MH; cy=PH*0.45; cw=220; ch=50
        c.setFillColor(bg_col); c.setStrokeColor(bd_col); c.setLineWidth(1)
        c.roundRect(cx, cy, cw, ch, 4, fill=1, stroke=1)
        c.setFillColor(txt_col); c.rect(cx, cy, 4, ch, fill=1, stroke=0)
        c.setFont('Helvetica-Bold', 13); c.drawString(cx+14, cy+28, signal)
        c.setFont('Helvetica', 8.5); c.setFillColor(DS.MUTED)
        c.drawString(cx+14, cy+12, "System Research Signal")
        sx=MH+240; sy=PH*0.45
        r_col = DS.score_colour(r_asre)
        c.setFont('Helvetica-Bold', 36); c.setFillColor(r_col)
        c.drawString(sx, sy+14, f"{r_asre:.1f}")
        c.setFont('Helvetica', 9); c.setFillColor(DS.MUTED)
        c.drawString(sx, sy+4, "R_ASRE  / 100")
        rf_col = DS.score_colour(r_final)
        c.setFont('Helvetica-Bold', 14); c.setFillColor(rf_col)
        c.drawString(sx+100, sy+20, f"{r_final:.1f}")
        c.setFont('Helvetica', 8); c.setFillColor(DS.MUTED)
        c.drawString(sx+100, sy+8, "R_Final")
        c.setStrokeColor(DS.BORDER); c.setLineWidth(0.5)
        c.line(MH, PH*0.42, PW-MH, PH*0.42)
        c.setFont('Helvetica', 9); c.setFillColor(DS.MUTED)
        c.drawString(MH, PH*0.39, "This document is an AI-generated quantitative research output.")
        c.drawString(MH, PH*0.37, "Review by a SEBI-registered IA or RA is required before any action.")
        c.setStrokeColor(DS.SIG_AMBER_BD); c.setLineWidth(0.8)
        c.line(0, 22, PW, 22)
        c.setFont('Helvetica-Bold', 6); c.setFillColor(DS.SIG_AMBER)
        c.drawRightString(PW-MH, 10, "IA/RA INTERNAL RESEARCH ONLY -- NOT INVESTMENT ADVICE -- SEBI DEC 2024")
        c.setFont('Helvetica', 6); c.setFillColor(DS.MUTED)
        c.drawCentredString(PW/2, 10, f"Page {doc.page}")
        c.restoreState()

    return draw


# ================================================================
# TABLE BUILDER  — unchanged from V4.1
# ================================================================

def _clean_table(rows, col_widths, extra_styles=None) -> Table:
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    base = [
        ('BACKGROUND',    (0,0),(-1,0),  DS.BRAND),
        ('TEXTCOLOR',     (0,0),(-1,0),  DS.WHITE),
        ('FONTNAME',      (0,0),(-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,0),  8),
        ('TOPPADDING',    (0,0),(-1,0),  7),
        ('BOTTOMPADDING', (0,0),(-1,0),  7),
        ('LEFTPADDING',   (0,0),(-1,0),  8),
        ('BACKGROUND',    (0,1),(-1,-1), DS.WHITE),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [DS.WHITE, DS.NEAR_WHITE]),
        ('FONTNAME',      (0,1),(-1,-1), 'Helvetica'),
        ('FONTSIZE',      (0,1),(-1,-1), 9),
        ('TOPPADDING',    (0,1),(-1,-1), 5),
        ('BOTTOMPADDING', (0,1),(-1,-1), 5),
        ('LEFTPADDING',   (0,0),(-1,-1), 8),
        ('RIGHTPADDING',  (0,0),(-1,-1), 8),
        ('TEXTCOLOR',     (0,1),(-1,-1), DS.BODY_TEXT),
        ('ALIGN',         (0,0),(-1,-1), 'LEFT'),
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ('BOX',           (0,0),(-1,-1), 0.5, DS.BORDER),
        ('LINEBELOW',     (0,0),(-1,0),  0.5, DS.BORDER),
        ('INNERGRID',     (0,1),(-1,-1), 0.3, DS.BORDER),
    ]
    if extra_styles:
        base.extend(extra_styles)
    t.setStyle(TableStyle(base))
    return t


# ================================================================
# MAIN GENERATOR
# ================================================================

class ASREReportGenerator:
    """
    V4.2 — V4.1 base + FundamentalFetcher integration.
    All V4.1 pages and components unchanged.
    """

    def __init__(self):
        self.S  = _build_styles()
        self.CW = DS.CONTENT_W

    def _sp(self, pts=DS.SP_M): return Spacer(1, pts)
    def _p(self, text, style='body'): return Paragraph(text, self.S[style])
    def _rule(self, thick=0.5): return ThinRule(self.CW, DS.BORDER, thick)
    def _section(self, title):
        return [self._sp(DS.SP_L), SectionLabel(title, self.CW), self._sp(DS.SP_S)]

    # ----------------------------------------------------------
    # SINGLE STOCK REPORT
    # ----------------------------------------------------------

    def generate_single_stock_report(
        self,
        ticker:              str,
        analysis_data:       Dict,
        output_path:         str,
        include_charts:      bool           = True,
        ia_notes:            Optional[str]  = None,
        suitability_context: Optional[Dict] = None,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        required = ['date','f_score','t_score','m_score',
                    'r_final','r_asre','signal','tier','category']
        missing = [f for f in required if f not in analysis_data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        rh       = _report_hash(ticker, analysis_data)
        rdate    = analysis_data['date']
        signal   = analysis_data['signal']
        r_asre   = float(analysis_data['r_asre'])
        r_final  = float(analysis_data['r_final'])
        f_score  = float(analysis_data['f_score'])
        t_score  = float(analysis_data['t_score'])
        m_score  = float(analysis_data['m_score'])
        tier     = analysis_data['tier']
        f_weight = float(analysis_data.get('f_weight', 0.40))
        t_weight = float(analysis_data.get('t_weight', 0.30))
        m_weight = float(analysis_data.get('m_weight', 0.30))

        cover_cb           = _draw_cover_bg(ticker, tier, rdate, signal, r_asre, r_final)
        on_cover, on_inner = _page_callbacks(ticker, rh, rdate)

        doc = SimpleDocTemplate(
            str(output_path), pagesize=A4,
            leftMargin=DS.MARGIN_H, rightMargin=DS.MARGIN_H,
            topMargin=DS.MARGIN_V+20, bottomMargin=DS.MARGIN_V+12,
            title=f"ASRE Analysis -- {ticker}", author=TOOL_NAME,
        )
        story = []

        # PAGE 1 — Cover
        story.append(Spacer(1, DS.PAGE_H - DS.MARGIN_V*2 - 80))
        story.append(PageBreak())

        # PAGE 2 — Executive Summary
        story += self._section("Executive Summary")
        story.append(self._p(_plain_english_summary(
            ticker, signal, r_final, r_asre, f_score, t_score, m_score, tier,
            analysis_data.get('market_context'))))
        story.append(self._sp(DS.SP_M))

        asre_sig  = signal
        final_sig = (signal if abs(r_final-r_asre) < 10
                     else ("NEUTRAL" if r_final >= 45 else "FLAG FOR REVIEW"))
        pill_tbl = Table(
            [[SignalPill(asre_sig,  r_asre,  "R_ASRE -- Risk-Parity Medallion", self.CW*0.485),
              Spacer(self.CW*0.03, 1),
              SignalPill(final_sig, r_final, "R_Final -- Composite Weighted",   self.CW*0.485)]],
            colWidths=[self.CW*0.485, self.CW*0.03, self.CW*0.485],
        )
        pill_tbl.setStyle(TableStyle([
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),
            ('RIGHTPADDING',(0,0),(-1,-1),0),('TOPPADDING',(0,0),(-1,-1),0),
            ('BOTTOMPADDING',(0,0),(-1,-1),0),
        ]))
        story.append(pill_tbl)
        story.append(self._sp(DS.SP_M))

        if abs(r_final - r_asre) >= 10:
            story.append(DivergenceNote(r_final, r_asre, self.CW))
            story.append(self._sp(DS.SP_S))

        if include_charts:
            buf = create_score_overview_chart(f_score, t_score, m_score, r_final, r_asre)
            story.append(Image(buf, width=self.CW, height=self.CW*0.34))
            story.append(self._sp(DS.SP_S))
            story.append(self._p(
                "Chart: scores below 35 are Poor (red), 35-49 Weak (amber), "
                "50-69 Moderate (blue), 70+ Strong (green). "
                "Donut shows R_ASRE; R_Final noted below it.", 'caption'))

        # Key metrics
        story += self._section("Key Metrics")
        sym = "INR" if ticker.upper().endswith((".NS",".BO")) else "USD"
        km  = [[self._p("Metric",'th'), self._p("Value",'th'),
                self._p("Metric",'th'), self._p("Value",'th')],
               [self._p("R_ASRE Score",'td'), self._p(f"{r_asre:.1f} / 100",'td'),
                self._p("R_Final Score",'td'), self._p(f"{r_final:.1f} / 100",'td')],
               [self._p("Quality Tier",'td'), self._p(tier,'td'),
                self._p("Category",'td'), self._p(analysis_data['category'],'td')]]
        if 'price' in analysis_data:
            km.append([self._p("Current Price",'td'),
                       self._p(f"{sym} {analysis_data['price']:,.2f}",'td'),
                       self._p("Analysis Date",'td'), self._p(rdate,'td')])
        # ── B4: R_ASRE saturation cap detection ─────────────────────────────
        # When Medallion ceiling (90.0) is reached for 3+ consecutive days, an IA
        # reading "90.0" might misinterpret it as a live computation. A prominent
        # footnote below the Key Metrics table clarifies it is a model cap.
        _timeline        = analysis_data.get('timeline_rows', [])
        _rasre_cap_days  = sum(1 for r in _timeline if float(r.get('r_asre', 0)) >= 90.0)
        _rasre_capped    = _rasre_cap_days >= 3
        story.append(_clean_table(km, [self.CW*0.27, self.CW*0.23, self.CW*0.27, self.CW*0.23]))
        if _rasre_capped:
            story.append(self._sp(DS.SP_XS))
            story.append(self._p(
                f"\u2691 R_ASRE cap note: score has been at or above 90.0 for "
                f"{_rasre_cap_days} of the last {len(_timeline)} trading days. "
                "The Medallion ceiling of 90.0 is a hard model cap \u2014 this value "
                "reflects maximum conviction, not a live incremental computation. "
                "Verify fundamentals independently before acting on this signal.",
                'caption'))
        story.append(PageBreak())

        # PAGE 3 — Component Breakdown
        story += self._section("Component Score Breakdown")
        story.append(self._p(
            "Each component contributes to the overall R_Final score with the "
            "weighting shown. Use these to identify where to focus your review.", 'body'))
        story.append(self._sp(DS.SP_S))
        # ── B5: T-score rapid recovery detection ─────────────────────────────
        # A >30 pt T-score change in 5 days is a technical recovery (price
        # momentum shift), not a fundamental change. Annotate so IAs don't
        # confuse a technical bounce with a fundamental upgrade.
        _t_scores         = [float(r.get('t_score', t_score))
                             for r in analysis_data.get('timeline_rows', [])]
        _t_jump_detected  = False
        _t_change         = 0.0   # initialised so all downstream references are safe
        _t_jump_direction = ""
        if len(_t_scores) >= 5:
            _t_window         = _t_scores[-5:]
            _t_change         = _t_window[-1] - _t_window[0]
            _t_jump_detected  = abs(_t_change) > 30
            _t_jump_direction = "recovery" if _t_change > 0 else "decline"

        for lbl, val, wt in [("F   Fundamentals", f_score, f_weight),
                              ("T   Technical",    t_score, t_weight),
                              ("M   Momentum",     m_score, m_weight)]:
            story.append(ComponentBar(lbl, val, wt, self.CW))
            story.append(self._sp(DS.SP_XS))
            # Inject T-score jump note immediately after the T bar
            if lbl.startswith("T") and _t_jump_detected:
                # _t_jump_detected is only True when len(_t_scores) >= 5,
                # so _t_change, _t_scores[-5], _t_scores[-1] are always safe here.
                story.append(self._p(
                    f"⚑ Technical score {_t_jump_direction} detected: "
                    f"T-score moved {abs(_t_change):.0f} pts in 5 days "
                    f"({_t_scores[-5]:.0f} \u2192 {_t_scores[-1]:.0f}). "
                    "This reflects a price momentum shift, not a fundamental change. "
                    "Review price action before attributing to earnings quality.",
                    'caption'))
                story.append(self._sp(DS.SP_XS))
        story.append(self._sp(DS.SP_M))

        # Fundamental Metrics — now populated from FundamentalFetcher when available
        story += self._section("Fundamental Metrics")

        # [V4.2] Show quarter end date + announcement date if fetched via FF
        fund_header_note = ""
        if analysis_data.get("_quarter_end_date"):
            fund_header_note = (
                f"Data from Yahoo Finance via FundamentalFetcher v2.0. "
                f"Quarter end: {analysis_data['_quarter_end_date']}. "
                f"Announced: {analysis_data.get('_announced_date', 'N/A')}."
            )
            story.append(self._p(fund_header_note, 'caption'))
            story.append(self._sp(DS.SP_XS))

        fund_items = [('pe',             'P/E Ratio (Trailing)',    '{:.2f}x'),
                      ('roe',            'Return on Equity',         '{:.1f}%'),
                      ('de',             'Debt / Equity Ratio',      '{:.2f}'),
                      ('revenue_growth', 'Revenue Growth (YoY)',     '{:.1f}%'),
                      ('profit_margin',  'Profit Margin',            '{:.1f}%')]
        fund_rows = [[self._p("Metric",'th'), self._p("Value",'th')]]
        for key, label, fmt in fund_items:
            val = analysis_data.get(key)
            if val is not None:
                try:    formatted = fmt.format(float(val))
                except: formatted = "N/A"
                fund_rows.append([self._p(label,'td'), self._p(formatted,'td')])
        if len(fund_rows) == 1:
            fund_rows.append([self._p("Fundamental data",'td'),
                               self._p("Not available for this ticker",'td')])
        story.append(_clean_table(fund_rows, [self.CW*0.55, self.CW*0.45]))

        # [V4.2] Extra raw fundamentals sub-table if fetched from FundamentalFetcher
        raw_keys = [
            ("_eps",               "EPS (Latest Quarter)",   "INR {:.2f}"),
            ("_revenue",           "Revenue (Latest Qtr)",   "INR {:,.0f}"),
            ("_net_income",        "Net Income (Latest Qtr)","INR {:,.0f}"),
            ("_free_cash_flow",    "Free Cash Flow (Qtr)",   "INR {:,.0f}"),
            ("_total_debt",        "Total Debt",             "INR {:,.0f}"),
            ("_shareholders_equity","Shareholders' Equity",  "INR {:,.0f}"),
        ]
        raw_rows = [[self._p("Additional Data (Yahoo Finance)",'th'),
                     self._p("Value",'th')]]
        for key, label, fmt in raw_keys:
            val = analysis_data.get(key)
            if val is not None and float(val) != 0.0:
                try:    formatted = fmt.format(float(val))
                except: formatted = "N/A"
                raw_rows.append([self._p(label,'td'), self._p(formatted,'td')])
        if len(raw_rows) > 1:
            story.append(self._sp(DS.SP_S))
            story.append(_clean_table(raw_rows, [self.CW*0.55, self.CW*0.45]))

        story.append(PageBreak())

        # PAGE 4 — Market Context + Dip Analysis + Charts
        mc = analysis_data.get('market_context')
        if mc:
            story += self._section("Market Context")
            mc_rows = [[self._p("Field",'th'), self._p("Value",'th')]]
            for label, key, fmt in [
                ("Market Stage",      "stage",      "{}"),
                ("SMA-200 Distance",  "sma200_dist","{:.1f}%"),
                ("Context Score",     "score",      "{:.0f} / 100"),
                ("Signal Confidence", "confidence", "{:.0f}%"),
            ]:
                val = mc.get(key)
                if val is not None:
                    mc_rows.append([self._p(label,'td'),
                                    self._p(fmt.format(val),'td')])
            story.append(_clean_table(mc_rows, [self.CW*0.5, self.CW*0.5]))
            story.append(self._sp(DS.SP_M))

        dq = analysis_data.get('dip_quality')
        if dq:
            story += self._section("Dip Quality Analysis -- Strategy C")
            dq_score = float(dq.get('score', 0) or 0)
            story.append(DipQualityBar(dq_score, self.CW))
            story.append(self._sp(DS.SP_S))
            dq_rows = [[self._p("Field",'th'), self._p("Value",'th')]]
            for label, key, fmt in [
                ("Quality Score",   "score",        "{:.0f} / 100"),
                ("Stage",           "stage",        "{}"),
                ("Assessment",      "assessment",   "{}"),
                ("Position Sizing", "position_size","{}"),
            ]:
                val = dq.get(key)
                if val is not None:
                    dq_rows.append([self._p(label,'td'),
                                    self._p(fmt.format(val),'td')])
            sc = DS.score_colour(dq_score)
            story.append(_clean_table(dq_rows, [self.CW*0.5, self.CW*0.5],
                [('TEXTCOLOR',(1,1),(1,1),sc),('FONTNAME',(1,1),(1,1),'Helvetica-Bold')]))
            story.append(self._sp(DS.SP_M))

        if include_charts and 'price_history' in analysis_data:
            story += self._section("Price History")
            ph = analysis_data['price_history']
            try:
                sma_s = (ph.get('sma_200') if isinstance(ph, dict)
                         else (ph['sma_200'] if 'sma_200' in ph.columns else None))
                price_buf = create_price_chart(
                    dates=pd.to_datetime(ph['date']), prices=ph['close'],
                    sma200=sma_s, ticker=ticker)
                story.append(Image(price_buf, width=self.CW, height=self.CW*0.38))
                story.append(self._p(
                    "Blue line = closing price. Amber dashed = SMA-200. "
                    "Price below SMA-200 indicates bearish medium-term trend.", 'caption'))
            except Exception as exc:
                logger.warning("Price chart error: %s", exc)
            story.append(self._sp(DS.SP_S))

        timeline = analysis_data.get('timeline_rows')
        if include_charts and timeline and len(timeline) >= 3:
            story += self._section("Score Trend")
            try:
                trend_buf = create_trend_sparkline(timeline)
                story.append(Image(trend_buf, width=self.CW, height=self.CW*0.26))
                story.append(self._p(
                    "Blue = R_ASRE (risk-parity). Amber dashed = R_Final (composite). "
                    "Dotted reference lines at 35 / 50 / 70.", 'caption'))
            except Exception as exc:
                logger.warning("Trend chart error: %s", exc)

        story.append(PageBreak())

        # PAGE 5 — IA / RIA NOTES
        story += self._section("IA / RIA Analyst Notes")
        story.append(self._p(
            "This section must be completed by the reviewing SEBI-registered "
            "Investment Adviser or Research Analyst before filing or client "
            "communication. It forms part of the audit trail required under "
            "SEBI Circular Dec 2024. All mandatory fields must be completed.", 'body'))
        story.append(self._sp(DS.SP_S))
        story.append(IARIANotesBlock(notes_text=ia_notes or "", width=self.CW))
        story.append(self._sp(DS.SP_M))

        if suitability_context:
            story += self._section("Suitability Context")
            suit_rows = [[self._p("Field",'th'), self._p("Value",'th')]]
            for k, v in suitability_context.items():
                suit_rows.append([self._p(k.replace('_',' ').title(),'td'),
                                   self._p(str(v),'td')])
            story.append(_clean_table(suit_rows, [self.CW*0.45, self.CW*0.55]))
            story.append(self._sp(DS.SP_M))

        story.append(PageBreak())

        # PAGE 6 — Data Sources + Disclosure
        ds_dict = analysis_data.get('data_sources', {})
        if ds_dict:
            story += self._section("Data Sources")
            src_rows = [[self._p("Type",'th'),
                         self._p("Source",'th'),
                         self._p("Timestamp / Notes",'th')]]
            for dtype, info in ds_dict.items():
                src_rows.append([
                    self._p(dtype.title(),'td'),
                    self._p(info.get('source','Yahoo Finance'),'td'),
                    self._p(info.get('timestamp','N/A'),'td'),
                ])
            story.append(_clean_table(src_rows,
                [self.CW*0.18, self.CW*0.44, self.CW*0.38]))
            story.append(self._sp(DS.SP_L))

        story += self._section("Regulatory Disclosure")
        disc_tbl = Table([[Paragraph(SEBI_DISCLOSURE, self.S['disclosure'])]],
                         colWidths=[self.CW])
        disc_tbl.setStyle(TableStyle([
            ('BOX',          (0,0),(-1,-1), 0.8, DS.SIG_AMBER_BD),
            ('LINEAFTER',    (0,0),(0,-1),  3,   DS.SIG_AMBER),
            ('BACKGROUND',   (0,0),(-1,-1), DS.SIG_AMBER_BG),
            ('TOPPADDING',   (0,0),(-1,-1), 12),
            ('BOTTOMPADDING',(0,0),(-1,-1), 12),
            ('LEFTPADDING',  (0,0),(-1,-1), 14),
            ('RIGHTPADDING', (0,0),(-1,-1), 14),
        ]))
        story.append(disc_tbl)

        def first_page(c, doc): cover_cb(c, doc)
        doc.build(story, onFirstPage=first_page, onLaterPages=on_inner)
        sz = output_path.stat().st_size / 1024
        logger.info("PDF ready: %s  (%.1f KB)", output_path, sz)
        return output_path

    # ----------------------------------------------------------
    # PORTFOLIO SUMMARY  — unchanged from V4.1
    # ----------------------------------------------------------

    def generate_portfolio_summary(
        self,
        stocks:         List[Dict],
        output_path:    str,
        portfolio_name: str           = "Portfolio Analysis",
        ia_notes:       Optional[str] = None,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_date = datetime.now().strftime('%Y-%m-%d')
        batch_hash  = hashlib.sha256(
            "|".join(s['ticker'] for s in stocks).encode()
        ).hexdigest()[:16].upper()

        _, on_inner = _page_callbacks(portfolio_name, batch_hash, report_date)

        doc = SimpleDocTemplate(
            str(output_path), pagesize=A4,
            leftMargin=DS.MARGIN_H, rightMargin=DS.MARGIN_H,
            topMargin=DS.MARGIN_V+20, bottomMargin=DS.MARGIN_V+12,
            title=f"ASRE -- {portfolio_name}", author=TOOL_NAME,
        )
        story = []

        hdr_tbl = Table(
            [[self._p("ASRE  |  Portfolio Report",'caption')],
             [self._p(portfolio_name,'port_title')],
             [self._p(f"{report_date}  \u00b7  {len(stocks)} stocks  \u00b7  "
                      f"SHA {batch_hash}",'port_meta')]],
            colWidths=[DS.CONTENT_W],
        )
        hdr_tbl.setStyle(TableStyle([
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
            ('LEFTPADDING',(0,0),(-1,-1),14),('RIGHTPADDING',(0,0),(-1,-1),14),
            ('BOX',(0,0),(-1,-1),0.5,DS.BORDER),('LINEAFTER',(0,0),(0,-1),3,DS.BRAND),
            ('BACKGROUND',(0,0),(-1,-1),DS.NEAR_WHITE),
        ]))
        story.append(hdr_tbl)
        story.append(self._sp(DS.SP_M))

        story += self._section("Portfolio Rankings")
        rank_rows = [[self._p("#",'th'), self._p("Ticker",'th'),
                      self._p("Signal",'th'), self._p("R_ASRE",'th'),
                      self._p("R_Final",'th'), self._p("F",'th'),
                      self._p("T",'th'), self._p("M",'th'),
                      self._p("Tier",'th')]]
        extra_styles = []
        for i, s in enumerate(stocks, 1):
            rasre  = float(s.get('r_asre', 0) or 0)
            rfinal = float(s.get('r_final',0) or 0)
            sig    = s.get('signal','N/A')
            txt_col, _, _ = DS.signal_style(sig)
            rank_rows.append([
                self._p(str(i),'td'),
                self._p(f"<b>{s['ticker']}</b>",'td'),
                Paragraph(f'<font color="{txt_col.hexval()}"><b>{sig}</b></font>',
                           self.S['td']),
                self._p(f"{rasre:.1f}",'td'), self._p(f"{rfinal:.1f}",'td'),
                self._p(f"{s.get('f_score',0):.0f}",'td_center'),
                self._p(f"{s.get('t_score',0):.0f}",'td_center'),
                self._p(f"{s.get('m_score',0):.0f}",'td_center'),
                self._p(s.get('tier','C'),'td'),
            ])
            extra_styles.append(('TEXTCOLOR',(3,i),(3,i),DS.score_colour(rasre)))

        cw9 = [self.CW*w for w in [0.05,0.14,0.24,0.09,0.09,0.07,0.07,0.07,0.08]]
        story.append(_clean_table(rank_rows, cw9, extra_styles))
        story.append(self._sp(DS.SP_L))

        story += self._section("IA / RIA Analyst Notes")
        story.append(IARIANotesBlock(notes_text=ia_notes or "", width=DS.CONTENT_W))
        story.append(self._sp(DS.SP_M))
        story.append(PageBreak())

        for i, stock in enumerate(stocks):
            ticker = stock['ticker']
            sig    = stock.get('signal','N/A')
            rasre  = float(stock.get('r_asre', 50) or 50)
            rfinal = float(stock.get('r_final',50) or 50)
            story += self._section(ticker)
            story.append(self._p(_plain_english_summary(
                ticker=ticker, signal=sig, r_final=rfinal, r_asre=rasre,
                f_score=float(stock.get('f_score',50) or 50),
                t_score=float(stock.get('t_score',50) or 50),
                m_score=float(stock.get('m_score',50) or 50),
                tier=stock.get('tier','C'),
                market_context=stock.get('market_context'))))
            story.append(self._sp(DS.SP_S))
            txt_col, _, _ = DS.signal_style(sig)
            score_rows = [
                [self._p("Metric",'th'), self._p("Value",'th')],
                [self._p("Signal",'td'),
                 Paragraph(f'<font color="{txt_col.hexval()}"><b>{sig}</b></font>',
                            self.S['td'])],
                [self._p("R_ASRE",'td'),  self._p(f"{rasre:.1f} / 100",'td')],
                [self._p("R_Final",'td'), self._p(f"{rfinal:.1f} / 100",'td')],
                [self._p("F / T / M",'td'),
                 self._p(f"{stock.get('f_score',0):.0f}  /  "
                         f"{stock.get('t_score',0):.0f}  /  "
                         f"{stock.get('m_score',0):.0f}",'td')],
                [self._p("Tier",'td'), self._p(stock.get('tier','C'),'td')],
            ]
            story.append(_clean_table(score_rows, [self.CW*0.45, self.CW*0.55],
                [('TEXTCOLOR',(1,2),(1,2),DS.score_colour(rasre)),
                 ('FONTNAME', (1,2),(1,2),'Helvetica-Bold')]))
            if i < len(stocks) - 1:
                story.append(self._sp(DS.SP_XL))
                story.append(self._rule())
                story.append(self._sp(DS.SP_M))

        story.append(PageBreak())
        story += self._section("Regulatory Disclosure")
        disc_tbl = Table([[Paragraph(SEBI_DISCLOSURE, self.S['disclosure'])]],
                         colWidths=[DS.CONTENT_W])
        disc_tbl.setStyle(TableStyle([
            ('BOX',(0,0),(-1,-1),0.8,DS.SIG_AMBER_BD),
            ('LINEAFTER',(0,0),(0,-1),3,DS.SIG_AMBER),
            ('BACKGROUND',(0,0),(-1,-1),DS.SIG_AMBER_BG),
            ('TOPPADDING',(0,0),(-1,-1),12),('BOTTOMPADDING',(0,0),(-1,-1),12),
            ('LEFTPADDING',(0,0),(-1,-1),14),('RIGHTPADDING',(0,0),(-1,-1),14),
        ]))
        story.append(disc_tbl)
        doc.build(story, onFirstPage=on_inner, onLaterPages=on_inner)
        logger.info("Portfolio PDF ready: %s", output_path)
        return output_path


# ================================================================
# CLI WRAPPER  — V4.2: optional `fetcher` parameter added
# ================================================================

def export_stock_report(
    ticker:              str,
    results_df:          pd.DataFrame,
    fundamentals:        Dict,
    output_dir:          str           = "reports",
    ia_notes:            Optional[str] = None,
    suitability_context: Optional[Dict] = None,
    fetcher                            = None,   # [V4.2] FundamentalFetcher instance
) -> Path:
    """
    Convenience wrapper — V4.0 / V4.1 API contract preserved.

    [V4.2 addition]
    Pass a FundamentalFetcher instance as `fetcher` and it will be used to
    pull live/cached fundamentals from Yahoo Finance automatically.
    Ratios (P/E, ROE, D/E, Revenue Growth, Profit Margin) and raw
    quarterly figures are merged into analysis_data before PDF generation.

    RuntimeError from FundamentalFetcher (cache TTL exceeded) is re-raised.
    All other fetcher errors are logged as warnings and the report continues.
    """
    latest   = results_df.iloc[-1]
    date_col = next((c for c in results_df.columns
                     if c.lower() in ('date','index')), None)
    date_val = latest[date_col] if date_col else latest.name
    try:    date_str = pd.to_datetime(date_val).strftime('%Y-%m-%d')
    except: date_str = datetime.now().strftime('%Y-%m-%d')

    ad: Dict = {
        'date':     date_str,
        'f_score':  float(latest.get('f_score', 50) or 50),
        't_score':  float(latest.get('t_score', 50) or 50),
        'm_score':  float(latest.get('m_score', 50) or 50),
        'r_final':  float(latest.get('r_final', 50) or 50),
        'r_asre':   float(latest.get('r_asre',  50) or 50),
        'signal':   str(latest.get('signal','NEUTRAL')),
        'tier':     str(fundamentals.get('tier',
                        latest.get('quality_tier','C')) or 'C'),
        'category': str(fundamentals.get('category','STABLE')),
        'f_weight': float(latest.get('f_weight',0.40) or 0.40),
        't_weight': float(latest.get('t_weight',0.30) or 0.30),
        'm_weight': float(latest.get('m_weight',0.30) or 0.30),
        'data_sources': {
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

    # Price from results_df
    for pc in ('close','Close','price','Price'):
        if pc in latest.index and pd.notna(latest[pc]):
            ad['price'] = float(latest[pc]); break

    # ── [V4.2] FundamentalFetcher integration ─────────────────
    if fetcher is not None:
        try:
            fund_data = fetch_fundamentals_for_report(
                ticker      = ticker,
                fetcher     = fetcher,
                price       = ad.get('price'),
                # Always use today as end_date so FundamentalFetcher's internal
                # date filter never excludes the most recent quarters.
                # date_str is the *scoring* date and may predate the latest
                # quarter-end date that Yahoo Finance returns.
                end_date    = datetime.now().strftime('%Y-%m-%d'),
                start_date  = None,   # defaults to 2 yrs ago inside helper
            )
            # Merge ratios and named raw fields into ad.
            # Explicitly whitelist keys to avoid polluting ad with large
            # objects (_fundamental_df, _fetch_timestamp).
            _public_keys = {
                'pe', 'roe', 'de', 'revenue_growth', 'profit_margin',
            }
            _raw_keys = {
                '_eps', '_revenue', '_net_income', '_free_cash_flow',
                '_total_debt', '_shareholders_equity',
                '_announced_date', '_quarter_end_date',
            }
            for key, val in fund_data.items():
                if key in _public_keys or key in _raw_keys:
                    ad[key] = val
            # Merge data_sources separately — preserves existing price/benchmark entries
            if 'data_sources' in fund_data:
                ad['data_sources'].update(fund_data['data_sources'])

        except RuntimeError:
            # Hard TTL abort — caller must clear the cache
            raise
        except Exception as exc:
            logger.warning(
                "[V4.2] FundamentalFetcher failed for %s — "
                "falling back to fundamentals dict. Error: %s", ticker, exc
            )
            # Fall back to manually provided fundamentals dict (V4.1 behaviour)
            for k in ('pe','roe','de'):
                if k in fundamentals and fundamentals[k]:
                    ad[k] = fundamentals[k]
            ad['data_sources']['fundamentals'] = {
                'source':    'Yahoo Finance (Quarterly) — manual fallback',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M IST'),
            }
    else:
        # No fetcher — use whatever is in the fundamentals dict (V4.1 behaviour)
        for k in ('pe','roe','de'):
            if k in fundamentals and fundamentals[k]:
                ad[k] = fundamentals[k]
        ad['data_sources']['fundamentals'] = {
            'source':    'Yahoo Finance (Quarterly)',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M IST'),
        }

    # Price history
    price_col = next((c for c in ('close','Close') if c in results_df.columns), None)
    if price_col:
        ph = results_df.tail(120).copy()
        rn = {price_col: 'close'}
        if date_col: rn[date_col] = 'date'
        ph = ph.rename(columns=rn)
        sma_c = next((c for c in ('sma_200','sma200','SMA_200')
                      if c in results_df.columns), None)
        if sma_c:
            ph['sma_200'] = results_df[sma_c].tail(120).values
        if 'date' in ph.columns:
            ad['price_history'] = ph

    if 'market_stage' in latest:
        ad['market_context'] = {
            'stage':      latest.get('market_stage','N/A'),
            'sma200_dist':float(latest.get('sma200_distance',0) or 0)*100,
            'score':      float(latest.get('market_score',0) or 0),
            'confidence': float(latest.get('market_confidence',0) or 0)*100,
        }
    if 'dip_score' in latest:
        ad['dip_quality'] = {
            'score':        float(latest.get('dip_score',0) or 0),
            'stage':        str(latest.get('dip_stage','N/A')),
            'assessment':   str(latest.get('dip_assessment','N/A')),
            'position_size':str(latest.get('position_sizing','N/A')),
        }

    rows = []
    for _, row in results_df.tail(10).iterrows():
        rows.append({'date':    row.get(date_col,row.name) if date_col else row.name,
                     'r_asre':  float(row.get('r_asre', 50) or 50),
                     'r_final': float(row.get('r_final',50) or 50),
                     't_score': float(row.get('t_score', 50) or 50)})
    if len(rows) >= 3:
        ad['timeline_rows'] = rows

    safe_ticker = ticker.replace('.','_')
    output_path = Path(output_dir) / f"{safe_ticker}_{date_str.replace('-','')}.pdf"

    return ASREReportGenerator().generate_single_stock_report(
        ticker=ticker, analysis_data=ad, output_path=str(output_path),
        include_charts=True, ia_notes=ia_notes,
        suitability_context=suitability_context,
    )


# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'ASREReportGenerator',
    'export_stock_report',
    'fetch_fundamentals_for_report',   # [V4.2] new
    '_plain_english_summary',
    '_report_hash',
    'SEBI_DISCLOSURE',
    'DS',
    'IARIANotesBlock',
]