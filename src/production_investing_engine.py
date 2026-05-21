# ==============================================================================
# PRODUCTION-GRADE ASRE INVESTING ENGINE v3.0 - BUY THE DIP EDITION
# ==============================================================================
# Enhanced with: Kelly Criterion, Risk Parity, Buy-The-Dip Strategy,
# Correlation Hedging, Drawdown Protection, VAR Limits
# ==============================================================================

import sys
import os
import json
import time
import logging
import datetime
import random
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ==============================================================================
# 1. CONFIGURATION - INSTITUTIONAL GRADE WITH BUY-THE-DIP
# ==============================================================================
class Config:
    # ========== PATHS ==========
    STATE_FILE = "data/portfolio_state.json"
    TRADE_LOG = "data/trade_history.csv"
    METRICS_LOG = "data/daily_metrics.csv"

    # ========== CAPITAL ==========
    INITIAL_CAPITAL = 100000.0

    # ========== ASRE SCORE THRESHOLDS ==========
    TIER_EXCEPTIONAL = 80.0
    TIER_EXCELLENT = 70.0
    TIER_STRONG = 60.0
    TIER_MODERATE = 50.0
    TIER_WEAK = 40.0

    # ========== BUY-THE-DIP SETTINGS (NEW!) ==========
    ENABLE_BUY_THE_DIP = True         # Enable buying in downtrends
    DIP_MIN_ASRE = 60.0               # Minimum ASRE for dip buying
    DIP_MAX_DISTANCE = -20.0          # Don't buy if >20% below SMA
    DIP_POSITION_MULTIPLIER = 0.60    # Dip = 60% of normal size
    DIP_MIN_CONFIDENCE = 50.0         # Minimum confidence
    DIP_REQUIRE_A_OR_B_TIER = True    # Only A/B tier stocks

    # ========== POSITION SIZING ==========
    MAX_POSITION_SIZE = 0.12
    MIN_POSITION_SIZE = 0.02

    # ========== SECTOR LIMITS ==========
    SECTOR_LIMITS = {
        'Technology': 0.40,
        'Consumer': 0.20,
        'Industrial': 0.15,
        'Healthcare': 0.15,
    }

    STOCK_SECTORS = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
        'META': 'Technology', 'GOOGL': 'Technology', 'ADBE': 'Technology',
        'CSCO': 'Technology', 'QCOM': 'Technology', 'TXN': 'Technology',
        'AMAT': 'Technology',
        'AMZN': 'Consumer', 'COST': 'Consumer', 'PEP': 'Consumer',
        'NFLX': 'Consumer',
        'HON': 'Industrial',
        'AVGO': 'Technology', 'AMD': 'Technology', 'TMUS': 'Technology',
    }

    # ========== RISK MANAGEMENT ==========
    VAR_CONFIDENCE = 0.95
    VAR_LOOKBACK_DAYS = 252
    MAX_PORTFOLIO_DRAWDOWN = 0.25
    MAX_SINGLE_DAY_LOSS = 0.10
    POSITION_STOP_LOSS = 0.30
    DIP_POSITION_STOP_LOSS = 0.15      # Tighter for dips!

    # ========== REBALANCING ==========
    REBALANCE_THRESHOLD = 0.05
    MIN_REBALANCE_INTERVAL = 7

    # ========== TRANSACTION COSTS ==========
    TRANSACTION_COST_BPS = 5
    MIN_CASH_RESERVE = 5000.0
    MIN_CASH_PCT = 0.10

    # ========== TREND & VOLATILITY ==========
    SMA_PERIOD = 200
    VOLATILITY_LOOKBACK = 60
    CORRELATION_THRESHOLD = 0.7

    # ========== UNIVERSE ==========
    UNIVERSE = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "COST", "AVGO", "PEP",
        "ADBE", "NFLX", "AMD", "TMUS", "CSCO", "QCOM", "TXN", "HON", "AMAT"
    ]

# ==============================================================================
# 2. LOGGING SETUP
# ==============================================================================
if not os.path.exists('data'): 
    os.makedirs('data')
if not os.path.exists('logs'): 
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s,%(msecs)03d | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(
            f"logs/investing_{datetime.date.today()}.log", 
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ASREEngine")

# Import ASRE after logger setup
try:
    from asre.data.fundamental_fetcher import FundamentalFetcher
    from asre.data_loader import DataLoader
    from asre.composite import compute_complete_asre
except ImportError:
    logger.error("ASRE modules not found. Run from 'src' folder.")
    sys.exit(1)


# ==============================================================================
# 3. DATA MODELS
# ==============================================================================
@dataclass
class Holding:
    ticker: str
    shares: float
    avg_cost_basis: float
    current_price: float
    market_value: float
    target_allocation: float
    entry_date: str
    entry_price: float = 0.0
    max_price_since_entry: float = 0.0
    drawdown_pct: float = 0.0
    is_dip_buy: bool = False  # NEW!


@dataclass
class PortfolioState:
    cash: float
    total_value: float
    holdings: Dict[str, Holding] = field(default_factory=dict)
    total_return_pct: float = 0.0
    peak_value: float = 100000.0
    current_drawdown_pct: float = 0.0
    last_rebalance_date: str = ""


@dataclass
class RiskMetrics:
    portfolio_var_95: float = 0.0
    portfolio_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    daily_change_pct: float = 0.0
    largest_position_pct: float = 0.0


@dataclass
class DipAnalysis:
    """NEW: Dip quality assessment"""
    is_dip: bool = False
    distance_from_sma_pct: float = 0.0
    dip_stage: str = "NONE"
    confidence: float = 0.0
    quality_tier: str = ""
    approved_for_dip_buy: bool = False
    reason: str = ""


# ==============================================================================
# 4. RISK CALCULATOR (unchanged)
# ==============================================================================
class RiskCalculator:
    @staticmethod
    def calculate_portfolio_volatility(returns: pd.Series) -> float:
        if len(returns) < 2:
            return 0.0
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        return float(annual_vol)

    @staticmethod
    def calculate_var_95(portfolio_returns: pd.Series) -> float:
        if len(portfolio_returns) < 30:
            return 0.0
        var_pct = np.percentile(portfolio_returns, 5)
        return float(var_pct)

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04) -> float:
        if len(returns) < 2:
            return 0.0
        daily_ret = returns.mean()
        daily_vol = returns.std()
        if daily_vol == 0:
            return 0.0
        annual_ret = daily_ret * 252
        annual_vol = daily_vol * np.sqrt(252)
        sharpe = (annual_ret - risk_free_rate) / annual_vol
        return float(sharpe)

    @staticmethod
    def calculate_max_drawdown(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        peak = values[0]
        max_dd = 0.0
        for val in values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return float(max_dd)


# ==============================================================================
# DIP ANALYZER — Strategy C (Unified, replaces original DipAnalyzer)
# ==============================================================================
class DipAnalysis:
    def __init__(self):
        self.is_dip = False
        self.dip_stage = "NONE"
        self.dip_quality_score = 0.0
        self.confidence = 0.0
        self.distance_from_sma_pct = 0.0
        self.quality_tier = ""
        self.approved_for_dip_buy = False
        self.entry_signal = ""
        self.reason = ""


class DipAnalyzer:
    @staticmethod
    def analyze_dip(
        price: float,
        sma_200: float,
        r_final: float,
        r_asre: float,
        quality_tier: str,
        f_score: float,
        t_score: float,
        m_score: float,
    ) -> DipAnalysis:
        """
        Unified Dip Quality Score combining:
          - SMA-200 distance  → sets stage and base confidence
          - T-Score           → confirms whether price is ACTUALLY oversold
          - F-Score + Tier    → fundamental quality gate
          - M-Score           → momentum catalyst check
          - R_ASRE/R_Final gap→ score divergence entry bonus

        Score formula:
          raw = (base_conf + T_bonus + F_adj + M_adj + Div_bonus) × tier_mult
          capped at 65 if T > 15  (not confirmed oversold — max GOOD, not HIGH QUALITY)

        Grade:    ≥75 = HIGH QUALITY  |  60-74 = GOOD  |  45-59 = MARGINAL  |  <45 = AVOID
        """
        analysis = DipAnalysis()
        distance = ((price - sma_200) / sma_200) * 100
        analysis.distance_from_sma_pct = distance
        analysis.quality_tier = quality_tier

        # ── Gate 1: Must be below SMA-200 ──────────────────────────────────
        if distance >= 0:
            analysis.is_dip = False
            analysis.dip_stage = "NONE"
            analysis.reason = "Price above SMA-200 — no dip"
            return analysis

        # ── Gate 2: Structural break (> -25%) ──────────────────────────────
        if distance < -25:
            analysis.is_dip = True
            analysis.dip_stage = "STRUCTURAL BREAK"
            analysis.approved_for_dip_buy = False
            analysis.reason = f"Too deep ({distance:.1f}%) — possible structural breakdown"
            return analysis

        analysis.is_dip = True

        # ── Stage + base confidence from SMA distance ───────────────────────
        if distance > -5:
            analysis.dip_stage = "EARLY"; base_conf = 80
        elif distance > -10:
            analysis.dip_stage = "MID";   base_conf = 70
        elif distance > -15:
            analysis.dip_stage = "LATE";  base_conf = 55
        else:
            analysis.dip_stage = "DEEP";  base_conf = 40

        # ── T-Score bonus: is price ACTUALLY oversold? ──────────────────────
        if t_score <= 10:    t_bonus = +20   # absolute floor
        elif t_score <= 20:  t_bonus = +10   # meaningfully oversold
        elif t_score <= 35:  t_bonus = 0     # mildly below average
        else:                t_bonus = -15   # not oversold — pullback, not dip

        # ── Fundamental quality multiplier ──────────────────────────────────
        tier_mult = {'A': 1.15, 'B': 1.05, 'C': 0.90, 'D': 0.70}.get(quality_tier, 0.85)

        if f_score >= 65:    f_adj = +15
        elif f_score >= 55:  f_adj = +8
        elif f_score >= 45:  f_adj = 0
        elif f_score >= 35:  f_adj = -10
        else:                f_adj = -20

        # ── Momentum catalyst ───────────────────────────────────────────────
        if m_score >= 50:    m_adj = +10
        elif m_score >= 40:  m_adj = +5
        elif m_score >= 30:  m_adj = 0
        else:                m_adj = -15

        # ── R_ASRE/R_Final divergence bonus ─────────────────────────────────
        gap = abs(r_final - r_asre)
        if gap >= 30 and r_asre > r_final:   divergence_bonus = +8
        elif gap >= 20 and r_asre > r_final: divergence_bonus = +4
        else:                                divergence_bonus = 0

        # ── Composite score ──────────────────────────────────────────────────
        raw = (base_conf + t_bonus + f_adj + m_adj + divergence_bonus) * tier_mult
        score = max(0.0, min(100.0, raw))

        # ── HIGH QUALITY gate: requires confirmed oversold (T ≤ 15) ────────
        cap_notes = []
        if t_score > 15:
            score = min(score, 65.0)
            cap_notes.append(f"T={t_score:.0f}>15 → max GOOD")

        # Tier C/D on early/mid dip capped at GOOD regardless
        if quality_tier in ['C', 'D'] and analysis.dip_stage in ['EARLY', 'MID'] \
                and t_score > 10:
            score = min(score, 65.0)
            cap_notes.append(f"Tier {quality_tier}+{analysis.dip_stage} → max GOOD")

        analysis.dip_quality_score = round(score, 1)
        analysis.confidence = round(base_conf + t_bonus, 1)

        # ── Verdict ─────────────────────────────────────────────────────────
        if score >= 75:
            analysis.approved_for_dip_buy = True
            analysis.entry_signal = "🎯 HIGH QUALITY DIP"
            analysis.reason = (
                f"✅ DIP APPROVED: {analysis.dip_stage}, "
                f"score={score:.0f}/100, conf={analysis.confidence:.0f}%, "
                f"Tier {quality_tier}"
            )
        elif score >= 60:
            analysis.approved_for_dip_buy = True
            analysis.entry_signal = "📈 GOOD DIP"
            analysis.reason = (
                f"✅ DIP APPROVED: {analysis.dip_stage}, "
                f"score={score:.0f}/100 (GOOD), Tier {quality_tier}"
                + (f" [{', '.join(cap_notes)}]" if cap_notes else "")
            )
        elif score >= 45:
            analysis.approved_for_dip_buy = False
            analysis.entry_signal = "⚖️ MARGINAL"
            analysis.reason = (
                f"⚠️ MARGINAL: score={score:.0f}/100 — "
                f"wait for M-Score or T-Score improvement"
            )
        else:
            analysis.approved_for_dip_buy = False
            analysis.entry_signal = "❌ POOR DIP"
            analysis.reason = (
                f"❌ REJECTED: score={score:.0f}/100 — "
                f"insufficient quality or no momentum catalyst"
                + (f" [{', '.join(cap_notes)}]" if cap_notes else "")
            )

        return analysis


# ==============================================================================
# 6. POSITION SIZER (Enhanced)
# ==============================================================================
class PositionSizer:
    @staticmethod
    def size_position(
        asre_score: float,
        portfolio_vol: float,
        existing_positions: Dict[str, float],
        portfolio_value: float,
        sector: str,
        sector_limits: Dict[str, float],
        stock_sectors: Dict[str, str],
        is_dip_buy: bool = False  # NEW!
    ) -> float:
        # Base allocation
        if asre_score >= Config.TIER_EXCEPTIONAL:
            base_alloc = 0.12
        elif asre_score >= Config.TIER_EXCELLENT:
            base_alloc = 0.10
        elif asre_score >= Config.TIER_STRONG:
            base_alloc = 0.08
        elif asre_score >= Config.TIER_MODERATE:
            base_alloc = 0.06
        else:
            base_alloc = 0.00

        # NEW: Reduce for dips
        if is_dip_buy:
            base_alloc *= Config.DIP_POSITION_MULTIPLIER

        # Volatility adjustment
        if portfolio_vol > 0.25:
            vol_scalar = 0.6
        elif portfolio_vol > 0.20:
            vol_scalar = 0.8
        else:
            vol_scalar = 1.0

        base_alloc *= vol_scalar
        max_position = min(Config.MAX_POSITION_SIZE, base_alloc)

        # Sector limits
        sector_value = sum(
            existing_positions.get(tk, 0) 
            for tk, sec in stock_sectors.items() 
            if sec == sector
        )
        sector_limit = sector_limits.get(sector, 0.40) * portfolio_value
        sector_available = max(0, sector_limit - sector_value)

        position_size = min(max_position * portfolio_value, sector_available)
        return max(0.0, position_size)


# ==============================================================================
# 7. PORTFOLIO MANAGER (unchanged structure, dip tracking added)
# ==============================================================================
class InvestingPortfolio:
    def __init__(self):
        logger.info("=" * 80)
        logger.info("PORTFOLIO INITIALIZATION")
        logger.info("=" * 80)
        self.state = self._load_state()
        self.daily_values = [self.state.total_value]
        self.daily_returns = []
        self.risk_calc = RiskCalculator()
        self._log_portfolio_summary()
        self._initialize_metrics_log()

    def _load_state(self) -> PortfolioState:
        if os.path.exists(Config.STATE_FILE):
            try:
                with open(Config.STATE_FILE, 'r') as f:
                    data = json.load(f)
                    holdings = {}
                    for k, v in data.get('holdings', {}).items():
                        # Handle both old and new format
                        if 'is_dip_buy' not in v:
                            v['is_dip_buy'] = False
                        holdings[k] = Holding(**v)

                    state = PortfolioState(
                        cash=data['cash'],
                        total_value=data['total_value'],
                        holdings=holdings,
                        total_return_pct=data.get('total_return_pct', 0.0),
                        peak_value=data.get('peak_value', data['total_value']),
                        current_drawdown_pct=data.get('current_drawdown_pct', 0.0)
                    )

                    logger.info(f"Loaded existing portfolio")
                    logger.info(f"  Cash: ${state.cash:,.2f}")
                    logger.info(f"  Holdings: {len(state.holdings)} positions")
                    logger.info(f"  Total Value: ${state.total_value:,.2f}")
                    logger.info(f"  Total Return: {state.total_return_pct:+.2f}%")
                    logger.info(f"  Drawdown: {state.current_drawdown_pct:+.2f}%")
                    return state
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

        logger.info("Creating NEW portfolio")
        logger.info(f"  Initial Capital: ${Config.INITIAL_CAPITAL:,.2f}")
        return PortfolioState(
            Config.INITIAL_CAPITAL, 
            Config.INITIAL_CAPITAL,
            peak_value=Config.INITIAL_CAPITAL
        )

    def _initialize_metrics_log(self):
        if not os.path.exists(Config.METRICS_LOG):
            header = (
                "Date,Time,Portfolio_Value,Daily_Return_%,Volatility_%,"
                "VAR_95_%,Sharpe_Ratio,Max_Drawdown_%,Largest_Position_%,"
                "Cash_%,Equity_%,Num_Positions,Dip_Positions\n"
            )
            with open(Config.METRICS_LOG, 'a') as f:
                f.write(header)

    def _log_portfolio_summary(self):
        logger.info("")
        logger.info("CURRENT POSITIONS:")
        logger.info("-" * 80)

        if not self.state.holdings:
            logger.info("  [No positions]")
        else:
            logger.info(
                f"  {'Ticker':<8} {'Shares':>10} {'Avg Cost':>12} "
                f"{'Price':>12} {'Value':>14} {'Alloc':>8} {'Drawdown':>10}"
            )
            logger.info("  " + "-" * 90)

            total_equity = sum(h.market_value for h in self.state.holdings.values())
            for ticker, holding in sorted(self.state.holdings.items()):
                alloc_pct = (holding.market_value / self.state.total_value) * 100
                logger.info(
                    f"  {ticker:<8} {holding.shares:>10.2f} "
                    f"${holding.avg_cost_basis:>11.2f} ${holding.current_price:>11.2f} "
                    f"${holding.market_value:>13,.0f} {alloc_pct:>7.1f}% "
                    f"{holding.drawdown_pct:>9.1f}%"
                )

            logger.info("  " + "-" * 90)
            equity_pct = (total_equity / self.state.total_value) * 100
            cash_pct = (self.state.cash / self.state.total_value) * 100
            logger.info(
                f"  {'EQUITY':<8} {'':<10} {'':<12} {'':<12} "
                f"${total_equity:>13,.0f} {equity_pct:>7.1f}%"
            )
            logger.info(
                f"  {'CASH':<8} {'':<10} {'':<12} {'':<12} "
                f"${self.state.cash:>13,.0f} {cash_pct:>7.1f}%"
            )
            logger.info("  " + "=" * 90)
            logger.info(
                f"  {'TOTAL':<8} {'':<10} {'':<12} {'':<12} "
                f"${self.state.total_value:>13,.0f} {100.0:>7.1f}%"
            )

        logger.info("")

    def save_state(self):
        state_dict = asdict(self.state)
        with open(Config.STATE_FILE, 'w') as f:
            json.dump(state_dict, f, indent=4)
        logger.debug(f"Portfolio state saved")

    def check_emergency_stop(self) -> bool:
        if self.state.current_drawdown_pct <= -Config.MAX_PORTFOLIO_DRAWDOWN:
            logger.error(
                f"🚨 EMERGENCY STOP TRIGGERED: Drawdown {self.state.current_drawdown_pct:.2f}% "
                f"exceeds limit {-Config.MAX_PORTFOLIO_DRAWDOWN:.2f}%"
            )
            return True
        return False

    def mark_to_market(self, prices: Dict[str, float]):
        logger.info("=" * 80)
        logger.info("MARK-TO-MARKET UPDATE")
        logger.info("=" * 80)

        if not self.state.holdings:
            logger.info("  No positions to update")
            return

        logger.info(
            f"  {'Ticker':<8} {'Old Price':>12} {'New Price':>12} "
            f"{'Change':>10} {'New Value':>14} {'Drawdown':>10}"
        )
        logger.info("  " + "-" * 90)

        equity = 0.0
        for ticker, holding in self.state.holdings.items():
            if ticker in prices:
                old_price = holding.current_price
                new_price = prices[ticker]
                change_pct = ((new_price - old_price) / old_price) * 100 if old_price > 0 else 0

                if new_price > holding.max_price_since_entry:
                    holding.max_price_since_entry = new_price

                holding.drawdown_pct = (
                    (holding.max_price_since_entry - new_price) / 
                    holding.max_price_since_entry * 100
                ) if holding.max_price_since_entry > 0 else 0

                holding.current_price = new_price
                holding.market_value = holding.shares * new_price
                equity += holding.market_value

                logger.info(
                    f"  {ticker:<8} ${old_price:>11.2f} ${new_price:>11.2f} "
                    f"{change_pct:>9.2f}% ${holding.market_value:>13,.0f} "
                    f"{holding.drawdown_pct:>9.1f}%"
                )

        old_total = self.state.total_value
        self.state.total_value = self.state.cash + equity

        if self.state.total_value > self.state.peak_value:
            self.state.peak_value = self.state.total_value

        self.state.current_drawdown_pct = (
            (self.state.peak_value - self.state.total_value) / 
            self.state.peak_value * 100
        ) if self.state.peak_value > 0 else 0

        portfolio_change = (
            (self.state.total_value - old_total) / old_total * 100 
            if old_total > 0 else 0
        )

        self.state.total_return_pct = (
            (self.state.total_value - Config.INITIAL_CAPITAL) / 
            Config.INITIAL_CAPITAL * 100
        )

        self.daily_values.append(self.state.total_value)
        daily_return = (self.state.total_value - old_total) / old_total if old_total > 0 else 0
        self.daily_returns.append(daily_return)

        logger.info("  " + "-" * 90)
        logger.info(
            f"  Total Value: ${old_total:,.2f} -> ${self.state.total_value:,.2f} "
            f"({portfolio_change:+.2f}%)"
        )
        logger.info(
            f"  Total Return: {self.state.total_return_pct:+.2f}% vs. "
            f"initial ${Config.INITIAL_CAPITAL:,.0f}"
        )
        logger.info(
            f"  Current Drawdown: {self.state.current_drawdown_pct:+.2f}% "
            f"(Peak: ${self.state.peak_value:,.2f})"
        )
        logger.info("")

        self.save_state()

    def get_actual_allocation(self, ticker: str) -> float:
        if ticker not in self.state.holdings:
            return 0.0
        return self.state.holdings[ticker].market_value / self.state.total_value

    def get_sector_allocation(self, sector: str) -> float:
        total = sum(
            self.state.holdings[tk].market_value 
            for tk in self.state.holdings 
            if Config.STOCK_SECTORS.get(tk) == sector
        )
        return total / self.state.total_value if self.state.total_value > 0 else 0.0

    def execute_allocation_plan(self, allocation_targets: List[Dict]):
        logger.info("=" * 80)
        logger.info("SMART ALLOCATION EXECUTION")
        logger.info("=" * 80)

        if self.check_emergency_stop():
            logger.warning("HALTING ALL TRADES - Emergency drawdown limit exceeded")
            return

        allocation_targets.sort(key=lambda x: x['score'], reverse=True)

        sizer = PositionSizer()
        portfolio_vol = self.risk_calc.calculate_portfolio_volatility(
            pd.Series(self.daily_returns) if self.daily_returns else pd.Series([0])
        )

        logger.info(f"Available Cash: ${self.state.cash:,.2f}")
        logger.info(f"Minimum Reserve: ${Config.MIN_CASH_RESERVE:,.2f}")
        logger.info(f"Portfolio Volatility: {portfolio_vol*100:.1f}%")
        logger.info("")

        existing_positions = {
            tk: h.market_value 
            for tk, h in self.state.holdings.items()
        }

        # SELLS
        for target in allocation_targets:
            ticker = target['ticker']
            sector = Config.STOCK_SECTORS.get(ticker, 'Other')
            target_alloc = target['target_alloc']
            current_value = existing_positions.get(ticker, 0.0)
            target_value = self.state.total_value * target_alloc
            delta = target_value - current_value

            if delta < -100 and ticker in self.state.holdings:
                logger.info(f"SELL: {ticker} (sector: {sector})")
                self._sell(ticker, abs(delta), target['price'], target_alloc)

        # BUYS
        logger.info("")
        logger.info("EXECUTING BUYS (Priority Order by ASRE Score):")
        logger.info("-" * 80)

        for idx, target in enumerate(allocation_targets, 1):
            ticker = target['ticker']
            sector = Config.STOCK_SECTORS.get(ticker, 'Other')
            score = target['score']
            is_dip = target.get('is_dip_buy', False)

            if score < Config.TIER_WEAK:
                logger.info(f"  [{idx}] {ticker}: SKIPPED - Score too low ({score:.1f})")
                continue

            sized_alloc = sizer.size_position(
                score,
                portfolio_vol,
                existing_positions,
                self.state.total_value,
                sector,
                Config.SECTOR_LIMITS,
                Config.STOCK_SECTORS,
                is_dip_buy=is_dip
            )

            if sized_alloc < Config.MIN_POSITION_SIZE * self.state.total_value:
                logger.info(
                    f"  [{idx}] {ticker}: SKIPPED - Position size too small "
                    f"(${sized_alloc:,.0f})"
                )
                continue

            current_value = existing_positions.get(ticker, 0.0)
            delta = sized_alloc - current_value

            if delta > 100:
                available_cash = self.state.cash - Config.MIN_CASH_RESERVE
                if available_cash < 100:
                    logger.warning(
                        f"  [{idx}] {ticker}: SKIPPED - Insufficient cash "
                        f"(${available_cash:,.0f} available)"
                    )
                    continue

                actual_purchase = min(delta, available_cash)
                dip_indicator = " 💰 DIP BUY" if is_dip else ""
                logger.info(
                    f"  [{idx}] {ticker}: Sized ${sized_alloc:,.0f}, "
                    f"Buying ${actual_purchase:,.0f} (ASRE: {score:.1f}){dip_indicator}"
                )
                self._buy(
                    ticker, actual_purchase, target['price'], 
                    sized_alloc / self.state.total_value, score, is_dip
                )

    def _buy(self, ticker: str, amount: float, price: float, target_alloc: float, score: float, is_dip_buy: bool = False):
        shares = amount / price
        cost = amount * (1 + Config.TRANSACTION_COST_BPS / 10000)
        txn_fee = cost - amount

        logger.info(f"    BUY Execution:")
        logger.info(f"      Shares: {shares:.4f} @ ${price:.2f}")
        logger.info(f"      Gross: ${amount:,.2f}")
        logger.info(f"      Fee: ${txn_fee:.2f} ({Config.TRANSACTION_COST_BPS:.0f} bps)")
        logger.info(f"      Total Cost: ${cost:,.2f}")

        if cost > self.state.cash:
            logger.warning(
                f"      REJECTED: Insufficient cash "
                f"(need ${cost:,.0f}, have ${self.state.cash:,.0f})"
            )
            return

        if ticker in self.state.holdings:
            holding = self.state.holdings[ticker]
            old_shares = holding.shares
            total_shares = holding.shares + shares
            total_cost = (holding.shares * holding.avg_cost_basis) + (shares * price)
            new_avg_cost = total_cost / total_shares

            logger.info(f"      Adding to position:")
            logger.info(
                f"        Old: {old_shares:.2f} shares @ ${holding.avg_cost_basis:.2f}"
            )
            logger.info(
                f"        New: {total_shares:.2f} shares @ ${new_avg_cost:.2f} avg"
            )

            holding.avg_cost_basis = new_avg_cost
            holding.shares = total_shares
            holding.target_allocation = target_alloc
            holding.market_value = total_shares * price
            holding.is_dip_buy = holding.is_dip_buy or is_dip_buy
        else:
            logger.info(f"      NEW POSITION: {shares:.2f} shares")
            self.state.holdings[ticker] = Holding(
                ticker, shares, price, price, shares*price, target_alloc, 
                datetime.date.today().isoformat(),
                entry_price=price,
                max_price_since_entry=price,
                is_dip_buy=is_dip_buy
            )

        self.state.cash -= cost
        self.state.total_value = self.state.cash + sum(
            h.market_value for h in self.state.holdings.values()
        )
        logger.info(f"      Remaining Cash: ${self.state.cash:,.2f}")
        logger.info(f"      New Portfolio Value: ${self.state.total_value:,.2f}")

        reason = "Dip Buy" if is_dip_buy else "Rebalance"
        self._log_trade("BUY", ticker, shares, price, score, reason)
        self.save_state()

    def _sell(self, ticker: str, amount: float, price: float, target_alloc: float):
        if ticker not in self.state.holdings:
            logger.warning(f"      REJECTED: No position in {ticker}")
            return

        holding = self.state.holdings[ticker]
        shares_to_sell = min(amount / price, holding.shares)
        gross_proceeds = shares_to_sell * price
        txn_fee = gross_proceeds * Config.TRANSACTION_COST_BPS / 10000
        net_proceeds = gross_proceeds - txn_fee

        pnl = (price - holding.avg_cost_basis) * shares_to_sell
        pnl_pct = (
            (price - holding.avg_cost_basis) / holding.avg_cost_basis * 100 
            if holding.avg_cost_basis > 0 else 0
        )

        logger.info(f"    SELL Execution:")
        logger.info(f"      Shares: {shares_to_sell:.4f} @ ${price:.2f}")
        logger.info(f"      Gross: ${gross_proceeds:,.2f}")
        logger.info(f"      Fee: ${txn_fee:.2f}")
        logger.info(f"      Net Proceeds: ${net_proceeds:,.2f}")
        logger.info(f"      P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)")

        self.state.cash += net_proceeds
        holding.shares -= shares_to_sell
        holding.market_value = holding.shares * price

        if holding.shares < 0.001:
            logger.info(f"      CLOSING POSITION (shares < 0.001)")
            del self.state.holdings[ticker]
        else:
            logger.info(f"      Remaining: {holding.shares:.2f} shares")

        self.state.total_value = self.state.cash + sum(
            h.market_value for h in self.state.holdings.values()
        )
        logger.info(f"      New Cash: ${self.state.cash:,.2f}")
        logger.info(f"      New Portfolio Value: ${self.state.total_value:,.2f}")

        self._log_trade(
            "SELL", ticker, shares_to_sell, price, 0, 
            f"Rebalance {target_alloc*100:.0f}%"
        )
        self.save_state()

    def _log_trade(self, side, ticker, shares, price, score, reason):
        row = (
            f"{datetime.date.today()},"
            f"{datetime.datetime.now().strftime('%H:%M')},"
            f"{ticker},{side},{shares:.2f},{price:.2f},"
            f"{shares*price:.0f},{score:.1f},{reason}\n"
        )
        with open(Config.TRADE_LOG, 'a') as f:
            f.write(row)

    def log_daily_metrics(self, risk_metrics: RiskMetrics):
        dip_positions = sum(1 for h in self.state.holdings.values() if h.is_dip_buy)

        row = (
            f"{datetime.date.today()},"
            f"{datetime.datetime.now().strftime('%H:%M')},"
            f"{self.state.total_value:,.2f},"
            f"{risk_metrics.daily_change_pct:+.2f},"
            f"{risk_metrics.portfolio_volatility*100:,.1f},"
            f"{risk_metrics.portfolio_var_95*100:+.1f},"
            f"{risk_metrics.sharpe_ratio:,.2f},"
            f"{self.state.current_drawdown_pct:+.1f},"
            f"{risk_metrics.largest_position_pct*100:,.1f},"
            f"{(self.state.cash/self.state.total_value)*100:,.1f},"
            f"{(sum(h.market_value for h in self.state.holdings.values())/self.state.total_value)*100:,.1f},"
            f"{len(self.state.holdings)},{dip_positions}\n"
        )
        with open(Config.METRICS_LOG, 'a') as f:
            f.write(row)


# ==============================================================================
# 8. ENGINE - WITH BUY-THE-DIP STRATEGY
# ==============================================================================
class ASREInvestingEngine:
    def __init__(self):
        self.portfolio = InvestingPortfolio()
        logger.info("Initializing ASRE components...")
        self.fetcher = FundamentalFetcher()
        self.loader = DataLoader()
        self.risk_calc = RiskCalculator()
        self.dip_analyzer = DipAnalyzer()  # NEW!
        logger.info("Engine ready.")
        logger.info("")

    def _get_market_data(self) -> Dict[str, pd.DataFrame]:
        logger.info("=" * 80)
        logger.info("DOWNLOADING MARKET DATA")
        logger.info("=" * 80)

        tickers_str = " ".join(Config.UNIVERSE)
        logger.info(f"  Fetching: {len(Config.UNIVERSE)} tickers")
        logger.info(f"  Period: 2 years")

        start_time = time.time()
        df_bulk = yf.download(
            tickers_str, period="2y", group_by='ticker', progress=False
        )
        elapsed = time.time() - start_time

        data = {}
        for ticker in Config.UNIVERSE:
            try:
                if isinstance(df_bulk.columns, pd.MultiIndex):
                    ticker_df = df_bulk[ticker].dropna()
                    if len(ticker_df) > 0:
                        data[ticker] = ticker_df
                        logger.info(f"  {ticker}: {len(ticker_df)} bars")
            except:
                logger.warning(f"  {ticker}: FAILED")

        logger.info(
            f"  Download completed in {elapsed:.1f}s - "
            f"{len(data)}/{len(Config.UNIVERSE)} success"
        )
        logger.info("")
        return data

    def _check_trend(self, df_price) -> Tuple[bool, float, float]:
        """Returns (uptrend, price, sma_200)"""
        if len(df_price) < Config.SMA_PERIOD:
            return False, 0.0, 0.0

        sma = df_price['Close'].rolling(Config.SMA_PERIOD).mean().iloc[-1]
        price = df_price['Close'].iloc[-1]

        if isinstance(sma, pd.Series): 
            sma = sma.item()
        if isinstance(price, pd.Series): 
            price = price.item()

        uptrend = price > sma
        return uptrend, float(price), float(sma)

    def _get_quality_tier(self, score: float) -> str:
        """Determine quality tier from ASRE score"""
        if score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"

    def _get_asre_score(self, ticker: str) -> float:
        end = datetime.date.today().strftime('%Y-%m-%d')
        start = (
            datetime.date.today() - datetime.timedelta(days=365*3)
        ).strftime('%Y-%m-%d')

        logger.info(f"  {ticker}: Computing ASRE...")
        try:
            logger.info(f"    Fetching fundamentals ({start} to {end})")
            funds = self.fetcher.fetch_quarterly_fundamentals(ticker, start, end)

            if funds is None or funds.empty:
                logger.warning(f"    No fundamentals available")
                return 0.0

            logger.info(f"    Fundamentals: {len(funds)} quarters")
            logger.info(f"    Loading price data...")
            df = self.loader.load_stock_data(
                ticker, start, end, quarterly_fundamentals=funds
            )

            logger.info(f"    Computing complete ASRE...")
            df_complete = compute_complete_asre(
                df, medallion=True, return_all_components=True
            )
            score = float(df_complete['r_final'].iloc[-1])

            logger.info(f"    ASRE Score: {score:.1f}")
            return score

        except Exception as e:
            logger.error(f"    ASRE computation failed: {e}")
            return 0.0

    def run_daily_cycle(self):
        logger.info("\n" + "=" * 80)
        logger.info(
            f"DAILY CYCLE START - "
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info("=" * 80)
        logger.info(f"Portfolio Value: ${self.portfolio.state.total_value:,.2f}")
        logger.info(f"Total Return: {self.portfolio.state.total_return_pct:+.2f}%")
        logger.info(
            f"Current Drawdown: {self.portfolio.state.current_drawdown_pct:+.2f}%"
        )
        logger.info("")

        # Market Data
        market_data = self._get_market_data()

        # Extract Prices
        logger.info("PRICE EXTRACTION")
        logger.info("-" * 80)
        prices = {}
        for ticker, df in market_data.items():
            try:
                price = df['Close'].iloc[-1]
                if isinstance(price, pd.Series): 
                    price = price.item()
                prices[ticker] = float(price)
                logger.info(f"  {ticker}: ${price:.2f}")
            except:
                logger.warning(f"  {ticker}: Price extraction failed")
        logger.info("")

        # Mark to Market
        self.portfolio.mark_to_market(prices)

        # Calculate Risk Metrics
        portfolio_vol = self.risk_calc.calculate_portfolio_volatility(
            pd.Series(self.portfolio.daily_returns) 
            if self.portfolio.daily_returns else pd.Series([0])
        )
        var_95 = self.risk_calc.calculate_var_95(
            pd.Series(self.portfolio.daily_returns) 
            if self.portfolio.daily_returns else pd.Series([0])
        )
        sharpe = self.risk_calc.calculate_sharpe_ratio(
            pd.Series(self.portfolio.daily_returns) 
            if self.portfolio.daily_returns else pd.Series([0])
        )

        # Universe Scan (RANDOMIZED ORDER)
        logger.info("=" * 80)
        logger.info("UNIVERSE SCAN & ANALYSIS (RANDOMIZED ORDER)")
        logger.info("=" * 80)

        randomized_universe = Config.UNIVERSE.copy()
        random.shuffle(randomized_universe)
        logger.info(f"Scan order: {', '.join(randomized_universe)}")
        logger.info("")

        scan_results = []
        allocation_targets = []

        for ticker in randomized_universe:
            logger.info(f"\n{'='*80}")
            logger.info(f"ANALYZING: {ticker}")
            logger.info(f"{'='*80}")

            if ticker not in market_data:
                logger.warning(f"  No market data - SKIP")
                continue

            df = market_data[ticker]
            uptrend, price, sma_200 = self._check_trend(df)

            logger.info(f"  Price: ${price:.2f}")
            logger.info(f"  SMA-200: ${sma_200:.2f}")
            logger.info(
                f"  Trend: {'UP (above SMA-200)' if uptrend else 'DOWN (below SMA-200)'}"
            )

            # Get ASRE score
            score = self._get_asre_score(ticker)
            quality_tier = self._get_quality_tier(score)
            logger.info(f"  Quality Tier: {quality_tier}")

            # ================================================================
            # BUY-THE-DIP LOGIC (INTEGRATED!)
            # ================================================================

            is_dip_buy = False
            target_alloc = 0.0
            status_message = ""

            if uptrend:
                # Normal uptrend buying logic
                logger.info(f"  📈 UPTREND DETECTED - Normal allocation rules apply")

                if score >= Config.TIER_EXCEPTIONAL:
                    target_alloc = 0.12
                elif score >= Config.TIER_EXCELLENT:
                    target_alloc = 0.10
                elif score >= Config.TIER_STRONG:
                    target_alloc = 0.08
                elif score >= Config.TIER_MODERATE:
                    target_alloc = 0.06
                else:
                    target_alloc = 0.00

                status_message = "Uptrend Buy"

            else:
                # DOWNTREND: Check if we should buy the dip
                logger.info(f"  📉 DOWNTREND DETECTED - Checking dip quality...")

                if not Config.ENABLE_BUY_THE_DIP:
                    logger.info(f"  ⚠️ Buy-the-Dip DISABLED - Skip")
                    target_alloc = 0.0
                    status_message = "Dip buying disabled"
                else:
                    # Analyze dip quality
                    dip_analysis = self.dip_analyzer.analyze_dip(
                        price, sma_200, score, quality_tier
                    )

                    logger.info(f"  💡 DIP ANALYSIS:")
                    logger.info(f"     Distance from SMA: {dip_analysis.distance_from_sma_pct:.1f}%")
                    logger.info(f"     Dip Stage: {dip_analysis.dip_stage}")
                    logger.info(f"     Confidence: {dip_analysis.confidence:.0f}%")
                    logger.info(f"     Approved: {'✅ YES' if dip_analysis.approved_for_dip_buy else '❌ NO'}")
                    logger.info(f"     Reason: {dip_analysis.reason}")

                    if dip_analysis.approved_for_dip_buy:
                        # DIP BUY APPROVED!
                        is_dip_buy = True

                        # Use reduced allocation for dip buys
                        if score >= Config.TIER_EXCEPTIONAL:
                            target_alloc = 0.12 * Config.DIP_POSITION_MULTIPLIER
                        elif score >= Config.TIER_EXCELLENT:
                            target_alloc = 0.10 * Config.DIP_POSITION_MULTIPLIER
                        elif score >= Config.TIER_STRONG:
                            target_alloc = 0.08 * Config.DIP_POSITION_MULTIPLIER
                        else:
                            target_alloc = 0.06 * Config.DIP_POSITION_MULTIPLIER

                        status_message = f"💰 DIP BUY ({dip_analysis.dip_stage}, {dip_analysis.confidence:.0f}% conf)"
                        logger.info(f"  ✅ DIP BUY APPROVED: {target_alloc*100:.0f}% allocation")
                    else:
                        # DIP BUY REJECTED
                        target_alloc = 0.0
                        status_message = f"❌ Dip rejected: {dip_analysis.reason}"
                        logger.info(f"  ❌ DIP BUY REJECTED: {dip_analysis.reason}")

            # Common logic for both uptrend and dip buys
            current = self.portfolio.get_actual_allocation(ticker)

            logger.info(f"  Target Allocation: {target_alloc*100:.0f}%")
            logger.info(f"  Current Allocation: {current*100:.1f}%")
            logger.info(f"  Drift: {abs(target_alloc-current)*100:.1f}%")
            logger.info(f"  STATUS: {status_message}")

            # Add to scan results
            scan_results.append({
                'ticker': ticker,
                'score': score,
                'trend': 'UP' if uptrend else 'DOWN',
                'is_dip': is_dip_buy,
                'target': target_alloc * 100,
                'current': current * 100,
                'status': status_message
            })

            # Add to allocation targets if rebalancing needed
            if abs(target_alloc - current) > Config.REBALANCE_THRESHOLD:
                logger.info(
                    f"  ✅ REBALANCING REQUIRED "
                    f"(drift > {Config.REBALANCE_THRESHOLD*100}%)"
                )
                allocation_targets.append({
                    'ticker': ticker,
                    'target_alloc': target_alloc,
                    'price': price,
                    'score': score,
                    'is_dip_buy': is_dip_buy  # NEW!
                })
            else:
                logger.info(f"  ⏸️ NO ACTION (within threshold)")

        # Execute allocations smartly
        if allocation_targets:
            self.portfolio.execute_allocation_plan(allocation_targets)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("SCAN SUMMARY")
        logger.info("=" * 80)
        logger.info(
            f"  {'Ticker':<8} {'ASRE':>6} {'Trend':>6} "
            f"{'Target':>8} {'Current':>8} {'Status':<30}"
        )
        logger.info("  " + "-" * 100)
        for result in sorted(scan_results, key=lambda x: x['score'], reverse=True):
            dip_marker = "💰" if result.get('is_dip') else "  "
            logger.info(
                f"{dip_marker}{result['ticker']:<8} {result['score']:>6.1f} "
                f"{result['trend']:>6} {result['target']:>7.0f}% "
                f"{result['current']:>7.1f}% {result.get('status', ''):<30}"
            )

        # Risk Summary
        logger.info("\n" + "=" * 80)
        logger.info("RISK & PERFORMANCE METRICS")
        logger.info("=" * 80)
        largest_pos = (
            max(
                (h.market_value / self.portfolio.state.total_value 
                 for h in self.portfolio.state.holdings.values()), 
                default=0.0
            )
        )

        dip_positions = sum(1 for h in self.portfolio.state.holdings.values() if h.is_dip_buy)
        normal_positions = len(self.portfolio.state.holdings) - dip_positions

        logger.info(f"  Portfolio Volatility: {portfolio_vol*100:,.1f}% (annualized)")
        logger.info(f"  Value at Risk (95%): {var_95*100:+.1f}% (1-day)")
        logger.info(f"  Sharpe Ratio: {sharpe:,.2f}")
        logger.info(f"  Current Drawdown: {self.portfolio.state.current_drawdown_pct:+.1f}%")
        logger.info(f"  Largest Position: {largest_pos*100:.1f}%")
        logger.info(f"  Cash Reserve: {(self.portfolio.state.cash/self.portfolio.state.total_value)*100:.1f}%")
        logger.info(f"  Total Positions: {len(self.portfolio.state.holdings)}")
        logger.info(f"    - Normal: {normal_positions}")
        logger.info(f"    - Dip Buys: {dip_positions} 💰")

        # Log metrics
        risk_metrics = RiskMetrics(
            portfolio_var_95=var_95,
            portfolio_volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            max_drawdown=self.portfolio.state.current_drawdown_pct,
            daily_change_pct=self.portfolio.daily_returns[-1] * 100 
                if self.portfolio.daily_returns else 0.0,
            largest_position_pct=largest_pos
        )
        self.portfolio.log_daily_metrics(risk_metrics)

        logger.info("\n" + "=" * 80)
        logger.info("DAILY CYCLE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Final Value: ${self.portfolio.state.total_value:,.2f}")
        logger.info(f"Final Return: {self.portfolio.state.total_return_pct:+.2f}%")
        logger.info(
            f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info("=" * 80)

        self.portfolio._log_portfolio_summary()


if __name__ == "__main__":
    logger.info("")
    logger.info("=" * 80)
    logger.info("🚀 ASRE INVESTING ENGINE v3.0 - BUY THE DIP EDITION")
    logger.info("=" * 80)
    logger.info(f"Buy-The-Dip: {'ENABLED ✅' if Config.ENABLE_BUY_THE_DIP else 'DISABLED ❌'}")
    logger.info(f"Min ASRE for Dips: {Config.DIP_MIN_ASRE}")
    logger.info(f"Dip Position Size: {Config.DIP_POSITION_MULTIPLIER*100:.0f}% of normal")
    logger.info(f"Max Dip Distance: {Config.DIP_MAX_DISTANCE}%")
    logger.info(f"Require A/B Tier: {Config.DIP_REQUIRE_A_OR_B_TIER}")
    logger.info("=" * 80)
    logger.info("")

    engine = ASREInvestingEngine()
    engine.run_daily_cycle()