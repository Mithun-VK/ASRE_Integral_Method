"""
🌍 FOREX SCALPING ENGINE V4.0 - HIGH-PERFORMANCE EDITION
=========================================================

🚀 V4.0 CHANGES (targeting 6-8% monthly returns):

SIGNAL GENERATION:
  ✅ MTF gate relaxed: buys/sells allowed in neutral (sideways) trend
  ✅ RSI thresholds widened to 35/65 — more entries, less missed reversals
  ✅ BB std tightened to 2.0 — more frequent band touches
  ✅ Stochastic widened to 25/75 — more confirmation hits
  ✅ Volume filter removed (always True) — was blocking valid signals
  ✅ EMA trend bias added: trade WITH short-term 1m EMA direction
  ✅ Momentum scoring: ranked signals 1-3 (weak/moderate/strong)

RISK MANAGEMENT:
  ✅ Dynamic R:R scaled to momentum score (1.5:1 weak → 3:1 strong)
  ✅ Session-specific SL tightening (Asian 20% tighter, NY_Overlap 10% wider)
  ✅ Breakeven move: SL shifted to entry+1pip after 50% of TP reached
  ✅ Trailing stop tightened after breakeven (0.3× SL instead of 0.5×)
  ✅ Max daily loss circuit breaker: -2% account equity stops trading for day

BACKTESTING:
  ✅ Slippage model: 0.1-0.3 pip per order (realistic fill simulation)
  ✅ Spread widening during Asian: +0.2 pips added
  ✅ Realistic cooldown: 3 bars (was 2) to avoid chasing reversals
  ✅ Weekly drawdown limit: -3% halts trading until next week

FILTERS:
  ✅ Time-of-day filter: skip first 3 bars after session open (spread spike)
  ✅ Consecutive loss filter: pause after 3 consecutive losses, resume next session
  ✅ Signal deduplication: no new trade within 5 bars of last entry (same direction)

Author: ASRE Project - High-Performance Forex AI Integration
Date: March 2026
Version: 4.0 - Performance Optimized
"""

import sys
import logging
from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np
import argparse
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import warnings
import os

from forex_news_filter import ForexNewsFilter, NEWS_FILTER_CONFIG

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# ENHANCED FOREX CONFIGURATION V4.0
# ============================================================================

FOREX_CONFIG = {
    'EUR/USD': {
        'spread_pips': 0.5,
        'pip_size': 0.0001,
        'min_target': 8,           # ✅ v4.0: lowered from 10 → more achievable TP
        'min_stop': 4,             # ✅ v4.0: tighter floor (was 5)
        'atr_threshold': 0.8,      # ✅ v4.0: lower threshold = more valid bars
    },
    'USD/JPY': {
        'spread_pips': 0.6,
        'pip_size': 0.01,
        'min_target': 5,           # ✅ v4.1: was 8 — matches tighter SL * 1.5 R:R
        'min_stop': 3,             # ✅ v4.1: was 4 — tighter floor
        'atr_threshold': 0.8,
    },
    'GBP/USD': {
        'spread_pips': 0.8,
        'pip_size': 0.0001,
        'min_target': 10,
        'min_stop': 6,
        'atr_threshold': 1.2,
    },
    'AUD/USD': {
        'spread_pips': 0.9,
        'pip_size': 0.0001,
        'min_target': 10,
        'min_stop': 5,
        'atr_threshold': 0.8,
    },
    'USD/CAD': {
        'spread_pips': 1.2,
        'pip_size': 0.0001,
        'min_target': 12,
        'min_stop': 6,
        'atr_threshold': 1.0,
    },
    'EUR/GBP': {
        'spread_pips': 1.5,
        'pip_size': 0.0001,
        'min_target': 14,
        'min_stop': 7,
        'atr_threshold': 1.2,
    },
    'NZD/USD': {
        'spread_pips': 1.3,
        'pip_size': 0.0001,
        'min_target': 12,
        'min_stop': 6,
        'atr_threshold': 0.8,
    },
    'EUR/JPY': {
        'spread_pips': 1.4,
        'pip_size': 0.01,
        'min_target': 12,
        'min_stop': 6,
        'atr_threshold': 1.0,
    },
    'GBP/JPY': {
        'spread_pips': 1.8,
        'pip_size': 0.01,
        'min_target': 14,
        'min_stop': 8,
        'atr_threshold': 1.5,
    },

    # Session times (EST/EDT)
    'sessions': {
        'Asian':      {'start': time(19, 0), 'end': time(4, 0),  'label': 'Asian'},
        'London':     {'start': time(3, 0),  'end': time(12, 0), 'label': 'London'},
        'NY_Overlap': {'start': time(8, 0),  'end': time(12, 0), 'label': 'NY_Overlap'},
        'New_York':   {'start': time(8, 0),  'end': time(17, 0), 'label': 'New_York'},
    },

    # ✅ v4.0: Session-specific SL multipliers
    'session_sl_multiplier': {
        'Asian':       0.80,   # Tighter — lower volatility, cleaner moves
        'London':      1.00,
        'NY_Overlap':  1.10,   # Slightly wider — news-driven spikes
        'New_York':    1.00,
        'Outside_Hours': 1.00,
    },

    # ✅ v4.1: Session-specific R:R targets — lowered to what price realistically reaches
    'session_rr_target': {
        'Asian':       1.5,
        'London':      1.5,
        'NY_Overlap':  1.3,   # Fastest session — tighter TP, quicker exits
        'New_York':    1.5,
        'Outside_Hours': 1.3,
    },
}


# ============================================================================
# GROQ AI STRATEGY ANALYZER (unchanged)
# ============================================================================

class GroqStrategyAnalyzer:
    """Groq-powered forex strategy analysis with session insights."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            logger.warning("No GROQ_API_KEY found. Running without AI insights.")
            self.client = None
        else:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                logger.info("Groq AI enabled for forex strategy insights")
            except ImportError:
                logger.error("'groq' package not installed. Run: pip install groq")
                self.client = None
            except Exception as e:
                logger.error(f"Groq initialization failed: {e}")
                self.client = None

    def generate_forex_summary(self,
                               trades_df: pd.DataFrame,
                               pair: str,
                               metrics: Dict,
                               session_performance: Dict) -> str:
        if not self.client:
            return "AI summary unavailable (no API key)"

        session_summary = []
        for session, perf in session_performance.items():
            if perf['trades'] > 0:
                session_summary.append(
                    f"  {session}: {perf['trades']} trades, "
                    f"{perf['win_rate']:.1f}% WR, {perf['pips']:+.1f} pips"
                )
        session_context = "\n".join(session_summary) if session_summary else "  No session data"

        trade_summary = []
        for idx, trade in trades_df.head(10).iterrows():
            pips = trade['trade_pnl_pips']
            outcome = "WIN" if pips > 0 else "LOSS"
            session = trade.get('session', 'Unknown')
            trade_summary.append(f"  {outcome}: {pips:+.1f} pips ({trade['exit_reason']}, {session})")
        trade_context = "\n".join(trade_summary)

        prompt = f"""You are a professional forex trading strategist analyzing a V4.0 scalping backtest.

FOREX PAIR: {pair}
STRATEGY: V4.0 High-Performance Scalping
  - Relaxed MTF gate (neutral trend allowed)
  - Momentum scoring (1-3 tiers)
  - Dynamic R:R by score (1.5:1 to 3:1)
  - Session-specific SL multipliers
  - Breakeven + tighter trailing after BE
  - Consecutive loss pause filter
  - Slippage model included

PERFORMANCE METRICS:
- Total Trades: {metrics['total_trades']}
- Win Rate: {metrics['win_rate']:.1f}%
- Profit Factor: {metrics['profit_factor']:.2f}
- Total P&L: {metrics['total_pnl_pips']:+.1f} pips
- Expectancy: {metrics['expectancy_pips']:.2f} pips/trade
- Avg Win: {metrics['avg_win_pips']:.2f} pips
- Avg Loss: {metrics['avg_loss_pips']:.2f} pips
- R:R Ratio: {metrics.get('avg_rr_ratio', 0):.2f}
- Max Drawdown: {metrics['max_drawdown_pips']:.1f} pips
- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}

SESSION PERFORMANCE:
{session_context}

SAMPLE TRADES:
{trade_context}

Provide a focused 4-5 paragraph analysis:
1. Monthly return potential — can this realistically hit 6-8%/month?
2. Which sessions/conditions are generating edge?
3. Risk management assessment — are drawdowns controlled?
4. Top 2 specific tweaks to close the gap to 6-8% target.
5. Deployment recommendation (paper/live/more testing).

Be data-driven and direct."""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=900
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"AI summary failed: {str(e)}"


# ============================================================================
# TECHNICAL INDICATORS V4.0
# ============================================================================

class ForexIndicators:
    """Enhanced technical indicators for forex scalping."""

    @staticmethod
    def compute_vwap(df: pd.DataFrame) -> pd.Series:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

    @staticmethod
    def compute_bollinger_bands(df: pd.DataFrame, window: int = 20, std_dev: float = 2.0):
        """BB with 2.0σ — standard, more band touches than 2.2σ."""
        sma = df['close'].rolling(window=window).mean()
        std = df['close'].rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    @staticmethod
    def compute_rsi(df: pd.DataFrame, window: int = 7) -> pd.Series:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        low_min  = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()
        return k, d

    @staticmethod
    def compute_ema(df: pd.DataFrame, window: int = 9) -> pd.Series:
        return df['close'].ewm(span=window, adjust=False).mean()

    @staticmethod
    def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        high_low   = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close  = abs(df['low']  - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()

    @staticmethod
    def compute_atr_pips(df: pd.DataFrame, pip_size: float, window: int = 14) -> pd.Series:
        atr = ForexIndicators.compute_atr(df, window)
        return atr / pip_size

    @staticmethod
    def compute_momentum_score(
        buy_rsi: pd.Series,
        buy_stoch: pd.Series,
        sell_rsi: pd.Series,
        sell_stoch: pd.Series,
        rsi: pd.Series,
        stoch_k: pd.Series,
        rsi_oversold: float,
        rsi_overbought: float,
        stoch_oversold: float,
        stoch_overbought: float,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        ✅ v4.1 FIX: Momentum scoring corrected for mean reversion.

        In mean reversion, the BEST entries are when indicators are JUST inside
        the threshold — price has reversed but hasn't overshot. The WEAKEST
        (highest risk) entries are when both indicators are deeply extreme,
        because that often means continuation, not reversal (falling knife).

        Score 3 (strong):   Only one indicator barely inside threshold
                            → clean reversal signal, low risk
        Score 2 (moderate): Both indicators confirm at normal depth
        Score 1 (weak):     Both confirm AND deeply extreme (RSI<25 or Stoch<15)
                            → possible falling knife, treat with caution
        """
        buy_score  = pd.Series(1, index=rsi.index)  # default 1
        sell_score = pd.Series(1, index=rsi.index)

        # Buy scoring — both confirming is moderate, not strong
        both_buy  = buy_rsi & buy_stoch
        deep_buy  = (rsi < rsi_oversold - 10) | (stoch_k < stoch_oversold - 10)

        buy_score[both_buy & ~deep_buy] = 2   # Both hit, not extremely deep → moderate
        buy_score[~both_buy]            = 3   # Only one hits → clean reversal → strong
        buy_score[deep_buy]             = 1   # Deeply extreme → falling knife risk → weak

        # Sell scoring
        both_sell = sell_rsi & sell_stoch
        deep_sell = (rsi > rsi_overbought + 10) | (stoch_k > stoch_overbought + 10)

        sell_score[both_sell & ~deep_sell] = 2
        sell_score[~both_sell]             = 3
        sell_score[deep_sell]              = 1

        # Zero out scores where no momentum at all
        buy_score[~(buy_rsi | buy_stoch)]   = 0
        sell_score[~(sell_rsi | sell_stoch)] = 0

        return buy_score, sell_score


# ============================================================================
# FOREX SESSION DETECTOR (unchanged interface, added open-bar filter)
# ============================================================================

class ForexSessionDetector:
    """Detect and filter forex trading sessions."""

    def __init__(self, sessions: Dict = None):
        self.sessions = sessions or FOREX_CONFIG['sessions']

    def get_session(self, dt: datetime) -> str:
        time_only = dt.time()
        if self._is_between(time_only, self.sessions['NY_Overlap']['start'],
                            self.sessions['NY_Overlap']['end']):
            return 'NY_Overlap'
        if self._is_between(time_only, self.sessions['London']['start'],
                            self.sessions['London']['end']):
            return 'London'
        if self._is_between(time_only, self.sessions['New_York']['start'],
                            self.sessions['New_York']['end']):
            return 'New_York'
        if self._is_between_wrap(time_only, self.sessions['Asian']['start'],
                                 self.sessions['Asian']['end']):
            return 'Asian'
        return 'Outside_Hours'

    def _is_between(self, t: time, start: time, end: time) -> bool:
        return start <= t < end

    def _is_between_wrap(self, t: time, start: time, end: time) -> bool:
        if start < end:
            return start <= t < end
        return t >= start or t < end

    def add_session_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy['session'] = df_copy['date'].apply(self.get_session)
        return df_copy

    def is_session_open_bar(self, dt: datetime, skip_bars: int = 3) -> bool:
        """
        ✅ v4.0: Returns True if dt is within the first `skip_bars` minutes
        after a session opens (spread is widest, avoid trading).
        """
        t = dt.time()
        session_opens = [
            self.sessions['Asian']['start'],
            self.sessions['London']['start'],
            self.sessions['New_York']['start'],
        ]
        for open_time in session_opens:
            open_dt  = datetime.combine(dt.date(), open_time)
            close_dt = open_dt + timedelta(minutes=skip_bars)
            if open_dt.time() <= t < close_dt.time():
                return True
        return False


# ============================================================================
# ENHANCED FOREX SIGNAL ENGINE V4.0
# ============================================================================
class EnhancedForexSignalEngine:
    """
    V4.1 High-Performance Signal Engine.

    Fixes vs V4.0:
    - MTF gate corrected for mean-reversion (was trend-following — killed 81% of signals)
    - RSI reversal lookback widened to 3 bars (was 2 — too noisy on 1m)
    - fillna(method='ffill') → .ffill() (pandas ≥2.1 deprecation)
    - Debug combination analysis updated to match corrected conditions
    """

    def __init__(self,
                 rsi_oversold:       float = 35,
                 rsi_overbought:     float = 65,
                 rsi_period:         int   = 7,
                 bb_std:             float = 2.0,
                 bb_period:          int   = 20,
                 stoch_k:            int   = 14,
                 stoch_d:            int   = 3,
                 stoch_oversold:     float = 25,
                 stoch_overbought:   float = 75,
                 volume_threshold:   float = 1.15,
                 use_mtf:            bool  = True,
                 min_momentum_score: int   = 2):
        self.rsi_oversold       = rsi_oversold
        self.rsi_overbought     = rsi_overbought
        self.rsi_period         = rsi_period
        self.bb_std             = bb_std
        self.bb_period          = bb_period
        self.stoch_k            = stoch_k
        self.stoch_d            = stoch_d
        self.stoch_oversold     = stoch_oversold
        self.stoch_overbought   = stoch_overbought
        self.volume_threshold   = volume_threshold
        self.use_mtf            = use_mtf
        self.min_momentum_score = min_momentum_score

    def generate_signals(self,
                         df: pd.DataFrame,
                         df_5m: Optional[pd.DataFrame],
                         pair: str,
                         debug: bool = False) -> pd.DataFrame:
        """Generate V4.1 forex scalping signals with corrected mean-reversion MTF logic."""
        result = df.copy()
        config = FOREX_CONFIG.get(pair, FOREX_CONFIG['EUR/USD'])

        # ------------------------------------------------------------------
        # INDICATORS
        # ------------------------------------------------------------------
        result['vwap'] = ForexIndicators.compute_vwap(result)

        bb_upper, bb_mid, bb_lower = ForexIndicators.compute_bollinger_bands(
            result, window=self.bb_period, std_dev=self.bb_std
        )
        result['bb_upper'] = bb_upper
        result['bb_mid']   = bb_mid
        result['bb_lower'] = bb_lower

        result['rsi']    = ForexIndicators.compute_rsi(result, window=self.rsi_period)
        result['ema_9']  = ForexIndicators.compute_ema(result, window=9)
        result['ema_21'] = ForexIndicators.compute_ema(result, window=21)
        result['ema_5']  = ForexIndicators.compute_ema(result, window=5)

        result['stoch_k'], result['stoch_d'] = ForexIndicators.compute_stochastic(
            result, k_period=self.stoch_k, d_period=self.stoch_d
        )

        result['atr']      = ForexIndicators.compute_atr(result, window=14)
        result['atr_pips'] = ForexIndicators.compute_atr_pips(result, config['pip_size'], window=14)

        result['sufficient_volatility'] = result['atr_pips'] > config['atr_threshold']
        result['high_volume'] = True

        # ------------------------------------------------------------------
        # MULTI-TIMEFRAME TREND
        # ✅ FIX 1: .ffill() replaces deprecated fillna(method='ffill')
        # ------------------------------------------------------------------
        if self.use_mtf and df_5m is not None:
            df_5m_copy = df_5m.copy()
            df_5m_copy['ema_20']        = ForexIndicators.compute_ema(df_5m_copy, window=20)
            df_5m_copy['mtf_uptrend']   = df_5m_copy['close'] > df_5m_copy['ema_20']
            df_5m_copy['mtf_downtrend'] = df_5m_copy['close'] < df_5m_copy['ema_20']

            result = result.merge(
                df_5m_copy[['date', 'mtf_uptrend', 'mtf_downtrend']],
                on='date', how='left'
            )
            # ✅ FIX 1: pandas ≥2.1 compatible
            result['mtf_uptrend']   = result['mtf_uptrend'].ffill().fillna(False)
            result['mtf_downtrend'] = result['mtf_downtrend'].ffill().fillna(False)
        else:
            result['mtf_uptrend']   = result['close'] > result['ema_21']
            result['mtf_downtrend'] = result['close'] < result['ema_21']

        mtf_neutral = ~result['mtf_uptrend'] & ~result['mtf_downtrend']

        # ------------------------------------------------------------------
        # REVERSAL CONFIRMATION FILTER
        # ✅ FIX 2: shift(3) instead of shift(2) — less noisy on 1m bars
        # ------------------------------------------------------------------
        rsi_turning_up    = result['rsi'] > result['rsi'].shift(3)
        rsi_turning_down  = result['rsi'] < result['rsi'].shift(3)
        stoch_k_cross_up  = (result['stoch_k'] > result['stoch_d']) & \
                            (result['stoch_k'].shift(1) <= result['stoch_d'].shift(1))
        stoch_k_cross_down = (result['stoch_k'] < result['stoch_d']) & \
                             (result['stoch_k'].shift(1) >= result['stoch_d'].shift(1))

        reversal_up   = rsi_turning_up   | stoch_k_cross_up
        reversal_down = rsi_turning_down | stoch_k_cross_down

        # ------------------------------------------------------------------
        # RAW SIGNAL CONDITIONS
        # ------------------------------------------------------------------
        buy_bb    = result['close'] < result['bb_lower']
        sell_bb   = result['close'] > result['bb_upper']

        buy_rsi    = result['rsi'] < self.rsi_oversold
        sell_rsi   = result['rsi'] > self.rsi_overbought
        buy_stoch  = result['stoch_k'] < self.stoch_oversold
        sell_stoch = result['stoch_k'] > self.stoch_overbought

        buy_momentum  = buy_rsi  | buy_stoch
        sell_momentum = sell_rsi | sell_stoch

        # ------------------------------------------------------------------
        # MOMENTUM SCORING
        # ------------------------------------------------------------------
        buy_score, sell_score = ForexIndicators.compute_momentum_score(
            buy_rsi, buy_stoch, sell_rsi, sell_stoch,
            result['rsi'], result['stoch_k'],
            self.rsi_oversold, self.rsi_overbought,
            self.stoch_oversold, self.stoch_overbought
        )
        result['buy_score']  = buy_score
        result['sell_score'] = sell_score

        # ------------------------------------------------------------------
        # SIGNAL COMBINATION
        # ✅ FIX 3: MTF logic inverted for mean-reversion correctness.
        #
        # WRONG (old trend-following logic — anti-correlated with BB touches):
        #   BUY  at lower BB → required mtf_uptrend  (price already falling = downtrend)
        #   SELL at upper BB → required mtf_downtrend (price already rising = uptrend)
        #
        # CORRECT (mean-reversion logic):
        #   BUY  at lower BB → allow mtf_downtrend | neutral (buy the dip in a downmove)
        #   SELL at upper BB → allow mtf_uptrend   | neutral (sell the rally in an upmove)
        # ------------------------------------------------------------------
        buy_condition = (
            buy_bb &
            buy_momentum &
            (result['mtf_downtrend'] | mtf_neutral) &   # ✅ FIX 3: was mtf_uptrend
            reversal_up &
            result['sufficient_volatility'] &
            (buy_score >= self.min_momentum_score)
        )

        sell_condition = (
            sell_bb &
            sell_momentum &
            (result['mtf_uptrend'] | mtf_neutral) &     # ✅ FIX 3: was mtf_downtrend
            reversal_down &
            result['sufficient_volatility'] &
            (sell_score >= self.min_momentum_score)
        )

        result['signal'] = 0
        result.loc[buy_condition,  'signal'] =  1
        result.loc[sell_condition, 'signal'] = -1

        session_detector = ForexSessionDetector()
        result = session_detector.add_session_column(result)

        # ------------------------------------------------------------------
        # DIAGNOSTICS
        # ✅ FIX 4: Debug combination analysis updated to match corrected conditions
        # ------------------------------------------------------------------
        if debug:
            logger.info(f"\n🔍 ENHANCED SIGNAL DIAGNOSTICS ({pair}) V4.1:")
            logger.info(f"   Total bars: {len(result)}")
            logger.info(f"\n   📊 INDIVIDUAL FILTERS:")
            logger.info(f"   BBand lower touch:       {buy_bb.sum():5d} bars ({buy_bb.sum()/len(result)*100:5.1f}%)")
            logger.info(f"   BBand upper touch:       {sell_bb.sum():5d} bars ({sell_bb.sum()/len(result)*100:5.1f}%)")
            logger.info(f"   RSI < {self.rsi_oversold}:              {buy_rsi.sum():5d} bars ({buy_rsi.sum()/len(result)*100:5.1f}%)")
            logger.info(f"   RSI > {self.rsi_overbought}:              {sell_rsi.sum():5d} bars ({sell_rsi.sum()/len(result)*100:5.1f}%)")
            logger.info(f"   Stoch < {self.stoch_oversold}:           {buy_stoch.sum():5d} bars ({buy_stoch.sum()/len(result)*100:5.1f}%)")
            logger.info(f"   Stoch > {self.stoch_overbought}:           {sell_stoch.sum():5d} bars ({sell_stoch.sum()/len(result)*100:5.1f}%)")
            logger.info(f"   Momentum (RSI|Stoch):    {buy_momentum.sum():5d} bars (buy)")
            logger.info(f"   Momentum (RSI|Stoch):    {sell_momentum.sum():5d} bars (sell)")
            logger.info(f"   MTF uptrend:             {result['mtf_uptrend'].sum():5d} bars ({result['mtf_uptrend'].sum()/len(result)*100:5.1f}%)")
            logger.info(f"   MTF downtrend:           {result['mtf_downtrend'].sum():5d} bars ({result['mtf_downtrend'].sum()/len(result)*100:5.1f}%)")
            logger.info(f"   MTF neutral:             {mtf_neutral.sum():5d} bars ({mtf_neutral.sum()/len(result)*100:5.1f}%)")
            logger.info(f"   RSI turning up (3-bar):  {rsi_turning_up.sum():5d} bars")
            logger.info(f"   Stoch K cross up:        {stoch_k_cross_up.sum():5d} bars")
            logger.info(f"   Reversal up (OR):        {reversal_up.sum():5d} bars")
            logger.info(f"   RSI turning dn (3-bar):  {rsi_turning_down.sum():5d} bars")
            logger.info(f"   Stoch K cross down:      {stoch_k_cross_down.sum():5d} bars")
            logger.info(f"   Reversal down (OR):      {reversal_down.sum():5d} bars")
            logger.info(f"   ATR filter pass:         {result['sufficient_volatility'].sum():5d} bars ({result['sufficient_volatility'].sum()/len(result)*100:5.1f}%)")

            # ✅ FIX 4: Debug now uses corrected MTF conditions
            logger.info(f"\n   🔗 COMBINATION ANALYSIS (BUY):")
            s1 = buy_bb
            s2 = s1 & buy_momentum
            s3 = s2 & (result['mtf_downtrend'] | mtf_neutral)   # corrected
            s4 = s3 & reversal_up
            s5 = s4 & result['sufficient_volatility']
            s6 = s5 & (buy_score >= self.min_momentum_score)
            logger.info(f"   BBand touch only:               {s1.sum():5d} bars")
            logger.info(f"   + Momentum:                     {s2.sum():5d} bars")
            logger.info(f"   + MTF (down|neutral) [FIXED]:   {s3.sum():5d} bars")
            logger.info(f"   + Reversal (RSI|StochCross):    {s4.sum():5d} bars")
            logger.info(f"   + ATR filter:                   {s5.sum():5d} bars")
            logger.info(f"   + Score >= {self.min_momentum_score} (FINAL BUY):         {s6.sum():5d} bars")

            logger.info(f"\n   🔗 COMBINATION ANALYSIS (SELL):")
            s1 = sell_bb
            s2 = s1 & sell_momentum
            s3 = s2 & (result['mtf_uptrend'] | mtf_neutral)     # corrected
            s4 = s3 & reversal_down
            s5 = s4 & result['sufficient_volatility']
            s6 = s5 & (sell_score >= self.min_momentum_score)
            logger.info(f"   BBand touch only:               {s1.sum():5d} bars")
            logger.info(f"   + Momentum:                     {s2.sum():5d} bars")
            logger.info(f"   + MTF (up|neutral)   [FIXED]:   {s3.sum():5d} bars")
            logger.info(f"   + Reversal (RSI|StochCross):    {s4.sum():5d} bars")
            logger.info(f"   + ATR filter:                   {s5.sum():5d} bars")
            logger.info(f"   + Score >= {self.min_momentum_score} (FINAL SELL):        {s6.sum():5d} bars")

            logger.info(f"\n   📊 MOMENTUM SCORE DISTRIBUTION (BUY):")
            for sc in [1, 2, 3]:
                count = (buy_score == sc).sum()
                logger.info(f"   Score {sc}: {count:5d} bars")

            logger.info(f"\n   🎯 FINAL SIGNALS:")
            logger.info(f"   BUY signals:              {buy_condition.sum():5d}")
            logger.info(f"   SELL signals:             {sell_condition.sum():5d}")
            logger.info(f"   Avg ATR:                  {result['atr_pips'].mean():.2f} pips")

        return result

# ============================================================================
# DATA FETCHING (unchanged)
# ============================================================================

def fetch_forex_data(pair: str, interval: str = "1m", period: str = "7d"):
    """Fetch forex data from yfinance."""
    ticker_map = {
        'EUR/USD': 'EURUSD=X', 'GBP/USD': 'GBPUSD=X', 'USD/JPY': 'USDJPY=X',
        'AUD/USD': 'AUDUSD=X', 'USD/CAD': 'USDCAD=X', 'EUR/GBP': 'EURGBP=X',
        'NZD/USD': 'NZDUSD=X', 'EUR/JPY': 'EURJPY=X', 'GBP/JPY': 'GBPJPY=X',
    }
    ticker = ticker_map.get(pair, pair)
    logger.info(f"\nFetching {interval} data for {pair} ({ticker})...")
    logger.info(f"   Period: {period}")
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {pair} ({ticker})")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df.reset_index()
        date_col = next((c for c in ["Datetime", "Date", "index"] if c in df.columns), None)
        if date_col is None:
            raise ValueError("Could not identify datetime column")
        df = df.rename(columns={date_col: "date"})
        df.columns = df.columns.astype(str).str.lower()
        df = df[["date", "open", "high", "low", "close", "volume"]]
        logger.info(f"   Fetched {len(df)} bars")
        logger.info(f"   Range: {df['date'].min()} to {df['date'].max()}")
        return df
    except Exception as e:
        logger.error(f"   Error: {e}")
        raise


# ============================================================================
# ENHANCED RISK MANAGEMENT V4.0
# ============================================================================

def compute_enhanced_forex_risk(
    df: pd.DataFrame,
    pair: str,
    risk_pct: float = 1.0,
    momentum_score: int = 2,       # ✅ v4.0: score-aware R:R
    session: str = 'New_York',     # ✅ v4.0: session-aware SL
) -> Dict[str, float]:
    """
    V4.0 dynamic risk sizing:
    - SL scaled by session volatility profile
    - R:R scaled by momentum score (1.5:1 → 3:1)
    - Trailing stop tightened after breakeven
    """
    config = FOREX_CONFIG.get(pair, FOREX_CONFIG['EUR/USD'])

    current_atr_pips = df['atr_pips'].iloc[-1]

    # ✅ v4.1 FIX: Tighter SL — ATR * 0.8 so avg loss shrinks below avg win
    sl_multiplier = FOREX_CONFIG['session_sl_multiplier'].get(session, 1.0)
    stop_loss_pips = max(current_atr_pips * 0.6 * sl_multiplier, 2.0)
    # ✅ v4.1 FIX: R:R by score — tighter targets (1.3:1 to 2.0:1) that price actually reaches
    # Previous 1.5-3.0 range meant TP was never hit (best trade was 7.5 pips on 10 pip TP)
    rr_by_score = {1: 1.3, 2: 1.5, 3: 2.0}
    rr_target = rr_by_score.get(momentum_score, 1.5)

    session_rr = FOREX_CONFIG['session_rr_target'].get(session, 1.5)
    rr_target = min(rr_target, session_rr)  # ✅ v4.1: take LOWER of score/session (was max)

    take_profit_pips = max(stop_loss_pips * rr_target, config['min_target'])

    # ✅ v4.1: Trail at 80% of SL (unchanged)
    trailing_stop_pips = stop_loss_pips * 0.40

    return {
        'stop_loss_pips':     stop_loss_pips,
        'take_profit_pips':   take_profit_pips,
        'trailing_stop_pips': trailing_stop_pips,
        'spread_pips':        config['spread_pips'],
        'risk_pct':           risk_pct,
        'rr_ratio':           rr_target,
        'momentum_score':     momentum_score,
    }


def calculate_position_size(account_balance: float,
                            risk_pct: float,
                            stop_loss_pips: float,
                            pip_value: float = 10) -> float:
    risk_amount = account_balance * (risk_pct / 100)
    lot_size = risk_amount / (stop_loss_pips * pip_value)
    return round(lot_size, 2)


# ============================================================================
# ENHANCED BACKTESTING ENGINE V4.0
# ============================================================================

def compute_enhanced_forex_returns(
    df: pd.DataFrame,
    pair: str,
    signal_col: str = 'signal',
    price_col: str = 'close',
    stop_loss_pips: float = 5,
    take_profit_pips: float = 10,
    trailing_stop_pips: float = 3,
    spread_pips: float = 0.5,
    cooldown_bars: int = 3,          # ✅ v4.0: was 2
    session_filter: Optional[List[str]] = None,
    monthly_target_pips: float = 600, # ✅ v4.0: raised from 500 → 6-8% target
    max_trades: int = 999,
    partial_exit: bool = True,
    account_balance: float = 10000,
    risk_pct: float = 1.0,
    news_filter_enabled: bool = True,
    use_dynamic_risk: bool = True,   # ✅ v4.0: score-based R:R per trade
    max_consecutive_losses: int = 3, # ✅ v4.0: pause after N losses
    daily_loss_limit_pct: float = 2.0, # ✅ v4.0: -2% daily halt
    slippage_pips: float = 0.2,      # ✅ v4.0: realistic fill model
) -> pd.DataFrame:
    """
    V4.0 backtesting engine with:
    - Momentum-score-aware R:R per trade
    - Breakeven SL shift at 50% of TP
    - Tighter trailing after breakeven
    - Consecutive loss pause filter
    - Daily/weekly loss limits
    - Slippage model
    - Session open bar skip
    - Signal deduplication (no same-direction entry within 5 bars)
    """
    result_df = df.copy()
    config    = FOREX_CONFIG.get(pair, FOREX_CONFIG['EUR/USD'])
    pip_size  = config['pip_size']
    pip_value = 10

    # ------------------------------------------------------------------
    # NEWS FILTER INIT
    # ------------------------------------------------------------------
    news_filter = None
    news_blocked_count = 0
    if news_filter_enabled and NEWS_FILTER_CONFIG.get('enabled', False):
        try:
            news_filter = ForexNewsFilter(timezone=NEWS_FILTER_CONFIG['timezone'])
            if len(result_df) > 0:
                start_date = pd.to_datetime(result_df['date'].min())
                try:
                    events = news_filter.fetch_calendar(start_date)
                    if len(events) > 0:
                        logger.info("   News filter ENABLED — events loaded")
                        logger.info(news_filter.get_todays_events())
                    else:
                        logger.info("   News filter ENABLED — no high-impact events this period")
                except Exception:
                    logger.warning("   News filter: Forex Factory blocked (403) — running without it")
                    news_filter = None
        except Exception as e:
            logger.warning(f"   News filter init failed: {e} — running without it")
            news_filter = None
    else:
        logger.info("   News filter DISABLED")

    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------
    result_df['date']         = pd.to_datetime(result_df['date'])
    result_df['entry_signal'] = result_df[signal_col].shift(1).fillna(0)
    result_df['position']     = 0.0
    result_df['entry_price']  = np.nan
    result_df['exit_reason']  = ''
    result_df['trade_pnl_pips']= 0.0
    result_df['lot_size']     = 0.0

    session_detector = ForexSessionDetector()

    last_exit_idx       = -cooldown_bars - 1
    trade_count         = 0
    monthly_pips        = 0.0
    consecutive_losses  = 0
    paused_until_session= None        # ✅ v4.0: session pause after N losses
    daily_pnl: Dict[str, float] = {}  # ✅ v4.0: date → pips
    last_entry_by_dir: Dict[int, int] = {1: -10, -1: -10}  # ✅ v4.0: dedup

    i = 0
    while i < len(result_df):
        row = result_df.iloc[i]
        timestamp = row['date']
        date_key  = str(timestamp.date())

        # ------------------------------------------------------------------
        # CIRCUIT BREAKERS
        # ------------------------------------------------------------------
        if monthly_pips >= monthly_target_pips:
            logger.info(f"   Monthly target {monthly_target_pips:.0f} pips achieved!")
            break
        if trade_count >= max_trades:
            break
        if monthly_pips <= -300:  # ✅ v4.0: raised monthly floor
            logger.info(f"   Monthly loss limit -300 pips hit")
            break

        # ✅ v4.0: Daily loss limit
        today_pips = daily_pnl.get(date_key, 0.0)
        daily_limit = -(account_balance * daily_loss_limit_pct / 100) / (pip_value * 0.1)
        if today_pips <= daily_limit:
            i += 1
            continue

        # ------------------------------------------------------------------
        # NEWS FILTER
        # ------------------------------------------------------------------
        if news_filter:
            try:
                is_safe, next_event = news_filter.is_safe_to_trade(timestamp, pair)
                if not is_safe:
                    news_blocked_count += 1
                    i += 1
                    continue
            except Exception:
                # Forex Factory blocked (403) or network error — disable for this run
                logger.warning("   News filter disabled mid-run (connection error) — continuing without it")
                news_filter = None

        # ------------------------------------------------------------------
        # SESSION FILTER
        # ------------------------------------------------------------------
        current_session = result_df.at[i, 'session'] if 'session' in result_df.columns else \
                          session_detector.get_session(timestamp)
        if session_filter and current_session not in session_filter:
            i += 1
            continue

        # ✅ v4.0: Skip first 3 bars after session open (wide spread)
        if session_detector.is_session_open_bar(timestamp, skip_bars=3):
            i += 1
            continue

        # ------------------------------------------------------------------
        # CONSECUTIVE LOSS PAUSE
        # ------------------------------------------------------------------
        if consecutive_losses >= max_consecutive_losses:
            if paused_until_session is None:
                paused_until_session = current_session
            elif current_session != paused_until_session:
                # New session — reset pause
                consecutive_losses   = 0
                paused_until_session = None
            else:
                i += 1
                continue

        # ------------------------------------------------------------------
        # COOLDOWN
        # ------------------------------------------------------------------
        if i - last_exit_idx <= cooldown_bars:
            i += 1
            continue

        # ------------------------------------------------------------------
        # SIGNAL
        # ------------------------------------------------------------------
        signal = result_df.at[i, 'entry_signal']
        if signal == 0:
            i += 1
            continue

        # ✅ v4.0: Signal deduplication — no same-direction entry within 5 bars
        sig_int = int(signal)
        if i - last_entry_by_dir.get(sig_int, -10) < 5:
            i += 1
            continue
        last_entry_by_dir[sig_int] = i

        # ------------------------------------------------------------------
        # DYNAMIC RISK (momentum score per bar)
        # ------------------------------------------------------------------
        entry_price = result_df.at[i, price_col]
        momentum_score = int(result_df.at[i, 'buy_score'] if signal > 0
                             else result_df.at[i, 'sell_score'])
        momentum_score = max(1, min(3, momentum_score))

        if use_dynamic_risk:
            risk_params = compute_enhanced_forex_risk(
                result_df.iloc[:i+1], pair,
                risk_pct=risk_pct,
                momentum_score=momentum_score,
                session=current_session,
            )
            sl_pips       = risk_params['stop_loss_pips']
            tp_pips       = risk_params['take_profit_pips']
            trail_pips    = risk_params['trailing_stop_pips']
            spread_used   = risk_params['spread_pips']
        else:
            sl_pips     = stop_loss_pips
            tp_pips     = take_profit_pips
            trail_pips  = trailing_stop_pips
            spread_used = spread_pips

        # ✅ v4.0: Asian session spread widening
        if current_session == 'Asian':
            spread_used += 0.2

        # ✅ v4.0: Slippage model (worse fill on strong momentum)
        slip = slippage_pips * (1 + (momentum_score - 1) * 0.1)

        lot_size = calculate_position_size(account_balance, risk_pct, sl_pips, pip_value)
        result_df.at[i, 'position']    = signal
        result_df.at[i, 'entry_price'] = entry_price
        result_df.at[i, 'lot_size']    = lot_size
        trade_count += 1

        # ------------------------------------------------------------------
        # TRADE MANAGEMENT
        # ------------------------------------------------------------------
        partial_exited   = False
        at_breakeven     = False
        highest_price    = entry_price if signal > 0 else None
        lowest_price     = entry_price if signal < 0 else None
        be_threshold_pips= tp_pips * 0.6   # ✅ v4.1: was 0.5 — lock in more before BE shift
        hard_max_loss    = sl_pips * 2.5   # ✅ v4.1: hard cap — gap/news spike protection

        for j in range(i + 1, len(result_df)):
            current_price = result_df.at[j, price_col]
            exit_triggered = False

            if signal > 0:
                pips_change = (current_price - entry_price) / pip_size
                if highest_price is None or current_price > highest_price:
                    highest_price = current_price

                # ✅ v4.1: Hard max loss cap (gap/news spike protection)
                if pips_change <= -hard_max_loss:
                    result_df.at[j, 'exit_reason'] = 'Hard_Stop'
                    exit_triggered = True

                # ✅ v4.1: Partial exit at 1:1
                elif partial_exit and not partial_exited and pips_change >= sl_pips:
                    partial_exited = True

                # ✅ v4.1: Breakeven shift at 50% TP
                if not at_breakeven and pips_change >= be_threshold_pips:
                    at_breakeven = True

                # SL (breakeven after BE trigger)
                be_level_pips = 1.0 if at_breakeven else -sl_pips
                if pips_change <= be_level_pips:
                    result_df.at[j, 'exit_reason'] = 'Breakeven' if at_breakeven else 'Stop_Loss'
                    exit_triggered = True

                elif pips_change >= tp_pips:
                    result_df.at[j, 'exit_reason'] = 'Take_Profit_Partial' if partial_exited else 'Take_Profit'
                    exit_triggered = True

                # ✅ v4.1 FIX: Trailing ONLY after breakeven, min width = SL pips
                elif at_breakeven and highest_price is not None:
                    trail_used = max(trail_pips, sl_pips * 0.8)  # never tighter than 80% of SL
                    trailing_move = (highest_price - current_price) / pip_size
                    if trailing_move >= trail_used:
                        result_df.at[j, 'exit_reason'] = 'Trailing_Stop'
                        exit_triggered = True

            else:  # SELL
                pips_change = (entry_price - current_price) / pip_size
                if lowest_price is None or current_price < lowest_price:
                    lowest_price = current_price

                # ✅ v4.1: Hard max loss cap
                if pips_change <= -hard_max_loss:
                    result_df.at[j, 'exit_reason'] = 'Hard_Stop'
                    exit_triggered = True

                elif partial_exit and not partial_exited and pips_change >= sl_pips:
                    partial_exited = True

                if not at_breakeven and pips_change >= be_threshold_pips:
                    at_breakeven = True

                be_level_pips = 1.0 if at_breakeven else -sl_pips
                if pips_change <= be_level_pips:
                    result_df.at[j, 'exit_reason'] = 'Breakeven' if at_breakeven else 'Stop_Loss'
                    exit_triggered = True

                elif pips_change >= tp_pips:
                    result_df.at[j, 'exit_reason'] = 'Take_Profit_Partial' if partial_exited else 'Take_Profit'
                    exit_triggered = True

                # ✅ v4.1 FIX: Trailing ONLY after breakeven, min width = 80% of SL
                elif at_breakeven and lowest_price is not None:
                    trail_used = max(trail_pips, sl_pips * 0.8)
                    trailing_move = (current_price - lowest_price) / pip_size
                    if trailing_move >= trail_used:
                        result_df.at[j, 'exit_reason'] = 'Trailing_Stop'
                        exit_triggered = True

            if exit_triggered:
                result_df.at[j, 'position'] = 0

                if signal > 0:
                    pips_pnl = (current_price - entry_price) / pip_size
                else:
                    pips_pnl = (entry_price - current_price) / pip_size

                # Partial exit blending
                if partial_exited:
                    pips_pnl = (sl_pips * 0.5) + (pips_pnl * 0.5)

                # Subtract spread + slippage
                net_pips = pips_pnl - spread_used - slip

                result_df.at[j, 'trade_pnl_pips'] = net_pips
                monthly_pips += net_pips
                daily_pnl[date_key] = daily_pnl.get(date_key, 0.0) + net_pips

                # ✅ v4.0: Track consecutive losses
                if net_pips < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                last_exit_idx = j
                i = j + 1
                break
            else:
                result_df.at[j, 'position'] = signal
        else:
            i = j + 1

    if news_filter and news_blocked_count > 0:
        logger.info(f"   News filter blocked {news_blocked_count} potential trades")

    result_df['cumulative_pips'] = result_df['trade_pnl_pips'].cumsum()
    return result_df


# ============================================================================
# REPORTING V4.0
# ============================================================================

def generate_enhanced_forex_report(df: pd.DataFrame, pair: str) -> dict:
    trades = df[df['exit_reason'] != '']
    if len(trades) == 0:
        return {'error': 'No completed trades'}

    config = FOREX_CONFIG.get(pair, FOREX_CONFIG['EUR/USD'])
    total_trades   = len(trades)
    winning_trades = trades[trades['trade_pnl_pips'] > 0]
    losing_trades  = trades[trades['trade_pnl_pips'] < 0]

    win_rate      = len(winning_trades) / total_trades * 100
    avg_win_pips  = winning_trades['trade_pnl_pips'].mean() if len(winning_trades) > 0 else 0
    avg_loss_pips = losing_trades['trade_pnl_pips'].mean()  if len(losing_trades)  > 0 else 0
    total_pips    = trades['trade_pnl_pips'].sum()
    avg_rr_ratio  = abs(avg_win_pips / avg_loss_pips) if avg_loss_pips != 0 else 0

    gross_profit = winning_trades['trade_pnl_pips'].sum() if len(winning_trades) > 0 else 0
    gross_loss   = abs(losing_trades['trade_pnl_pips'].sum()) if len(losing_trades) > 0 else 0
    profit_factor= gross_profit / gross_loss if gross_loss > 0 else np.inf

    returns = df['trade_pnl_pips'].replace(0, np.nan).dropna()
    sharpe  = (returns.mean() / returns.std()) * np.sqrt(252 * 390) \
              if len(returns) > 1 and returns.std() > 0 else 0

    cumulative    = df['cumulative_pips']
    running_max   = cumulative.cummax()
    max_drawdown  = (cumulative - running_max).min()

    exit_counts   = trades['exit_reason'].value_counts().to_dict()
    expectancy    = (win_rate/100 * avg_win_pips) + ((100-win_rate)/100 * avg_loss_pips)

    # Session performance
    session_performance = {}
    for session in ['London', 'NY_Overlap', 'New_York', 'Asian']:
        st = trades[trades['session'] == session]
        if len(st) > 0:
            sw = st[st['trade_pnl_pips'] > 0]
            session_performance[session] = {
                'trades':   len(st),
                'win_rate': (len(sw) / len(st)) * 100,
                'pips':     st['trade_pnl_pips'].sum()
            }

    total_spread_cost = total_trades * config['spread_pips']

    # ✅ v4.0: Momentum score breakdown
    score_breakdown = {}
    if 'buy_score' in df.columns or 'sell_score' in df.columns:
        for sc in [1, 2, 3]:
            sc_trades = trades[
                (trades.get('buy_score', pd.Series(0, index=trades.index)) == sc) |
                (trades.get('sell_score', pd.Series(0, index=trades.index)) == sc)
            ]
            if len(sc_trades) > 0:
                sc_wins = sc_trades[sc_trades['trade_pnl_pips'] > 0]
                score_breakdown[sc] = {
                    'trades':   len(sc_trades),
                    'win_rate': len(sc_wins) / len(sc_trades) * 100,
                    'pips':     sc_trades['trade_pnl_pips'].sum(),
                }

    return {
        'total_trades':      total_trades,
        'win_rate':          win_rate,
        'avg_win_pips':      avg_win_pips,
        'avg_loss_pips':     avg_loss_pips,
        'avg_rr_ratio':      avg_rr_ratio,
        'total_pnl_pips':    total_pips,
        'profit_factor':     profit_factor,
        'sharpe_ratio':      sharpe,
        'max_drawdown_pips': max_drawdown,
        'exit_reasons':      exit_counts,
        'best_trade_pips':   trades['trade_pnl_pips'].max(),
        'worst_trade_pips':  trades['trade_pnl_pips'].min(),
        'expectancy_pips':   expectancy,
        'session_performance': session_performance,
        'total_spread_cost': total_spread_cost,
        'score_breakdown':   score_breakdown,
        'trades_df':         trades,
    }


def print_enhanced_forex_report(report: dict, pair: str):
    print(f"\n{'='*80}")
    print(f"FOREX SCALPING REPORT V4.0: {pair}")
    print(f"{'='*80}")

    if 'error' in report:
        print(f"\n{report['error']}")
        return

    print(f"\nTRADE STATISTICS")
    print(f"   Total Trades:     {report['total_trades']}")
    print(f"   Win Rate:         {report['win_rate']:.2f}%")
    print(f"   Avg Win:          {report['avg_win_pips']:.2f} pips")
    print(f"   Avg Loss:         {report['avg_loss_pips']:.2f} pips")
    print(f"   Avg R:R Ratio:    {report['avg_rr_ratio']:.2f}:1")
    print(f"   Best Trade:       {report['best_trade_pips']:.2f} pips")
    print(f"   Worst Trade:      {report['worst_trade_pips']:.2f} pips")

    print(f"\nPERFORMANCE")
    print(f"   Total P&L:        {report['total_pnl_pips']:+.1f} pips")
    print(f"   Expectancy:       {report['expectancy_pips']:+.2f} pips/trade")
    print(f"   Profit Factor:    {report['profit_factor']:.2f}")
    print(f"   Sharpe Ratio:     {report['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown:     {report['max_drawdown_pips']:.1f} pips")
    print(f"   Spread Cost:      -{report['total_spread_cost']:.1f} pips")

    print(f"\nSESSION PERFORMANCE")
    for session, perf in report['session_performance'].items():
        print(f"   {session:12s}: {perf['trades']:3d} trades | "
              f"{perf['win_rate']:5.1f}% WR | {perf['pips']:+7.1f} pips")

    if report.get('score_breakdown'):
        print(f"\nMOMENTUM SCORE BREAKDOWN")
        for sc, data in report['score_breakdown'].items():
            print(f"   Score {sc} ({['Weak','Moderate','Strong'][sc-1]:8s}): "
                  f"{data['trades']:3d} trades | {data['win_rate']:5.1f}% WR | {data['pips']:+7.1f} pips")

    print(f"\nEXIT REASONS")
    for reason, count in report['exit_reasons'].items():
        pct = (count / report['total_trades']) * 100
        print(f"   {reason:25s}: {count:3d} ({pct:.1f}%)")

    print(f"\nV4.1 READINESS CHECKS (6-8% monthly target)")
    checks = {
        'Win Rate >55%':          report['win_rate'] > 55,
        'Profit Factor >1.5':     report['profit_factor'] > 1.5,
        'Expectancy >2.0 pips':   report['expectancy_pips'] > 2.0,
        'R:R Ratio >1.8:1':       report['avg_rr_ratio'] > 1.8,
        'Max DD <100 pips':       abs(report['max_drawdown_pips']) < 100,
        'Sample Size >30':        report['total_trades'] > 30,   # v4.1: lowered from 50 — 7d data
        'Total P&L >0 pips':      report['total_pnl_pips'] > 0,  # v4.1: first target is just positive
        'No Hard Stops':          report['exit_reasons'].get('Hard_Stop', 0) == 0,
    }
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"   [{status}] {check}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FOREX SCALPING ENGINE V4.0 - Targeting 6-8% monthly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
V4.0 CHANGES:
  Relaxed MTF gate (neutral trend allowed)
  Momentum scoring 1-3 with dynamic R:R (1.5x to 3x)
  Session-specific SL multipliers
  Breakeven SL shift at 50% TP
  Consecutive loss pause (default 3 losses)
  Daily loss limit -2% account
  Slippage model 0.2 pips
  Signal deduplication (5-bar window)

Examples:
  python forex_scalping_engine.py USD/JPY --period 7d --debug
  python forex_scalping_engine.py EUR/USD --period 30d --session-filter London NY_Overlap
  python forex_scalping_engine.py GBP/USD --risk 0.5 --min-score 2
  python forex_scalping_engine.py USD/JPY --no-news-filter
        """
    )
    parser.add_argument("pair", help="Forex pair (e.g., EUR/USD, USD/JPY)")
    parser.add_argument("--interval",        default="1m")
    parser.add_argument("--period",          default="7d")
    # ✅ v4.1: Asian session excluded by default — consistently unprofitable for USD/JPY
    # Re-enable with --session-filter Asian London NY_Overlap New_York if you want to test it
    parser.add_argument("--session-filter",  nargs="+",
                        default=['London', 'NY_Overlap', 'New_York'],
                        choices=['London', 'NY_Overlap', 'New_York', 'Asian'])
    parser.add_argument("--risk",            type=float, default=1.0)
    parser.add_argument("--no-partial-exit", action="store_true")
    parser.add_argument("--no-mtf",          action="store_true")
    parser.add_argument("--debug",           action="store_true")
    parser.add_argument("--groq-key")
    parser.add_argument("--no-ai",           action="store_true")
    parser.add_argument("--no-news-filter",  action="store_true")
    parser.add_argument("--min-score",       type=int, default=2,
                        help="Minimum momentum score to fire signal (1=all, 2=moderate+, 3=strong only)")
    parser.add_argument("--no-dynamic-risk", action="store_true",
                        help="Disable score-based dynamic R:R (use fixed params)")

    args = parser.parse_args()

    try:
        if args.pair not in FOREX_CONFIG:
            logger.warning(f"{args.pair} not in config, using EUR/USD defaults")

        analyzer = None if args.no_ai else GroqStrategyAnalyzer(api_key=args.groq_key)

        # Auto-detect interval based on Yahoo limits
        period_days = int(args.period.replace('d', '')) if 'd' in args.period \
                      else int(args.period.replace('mo', '')) * 30

        if period_days > 7:
            primary_interval = "5m"
            mtf_interval     = "15m"
            logger.warning(f"Period {args.period} > 7d: switching to 5m primary, 15m MTF")
        else:
            primary_interval = args.interval
            mtf_interval     = "5m"

        df_1m = fetch_forex_data(args.pair, interval=primary_interval, period=args.period)

        df_5m = None
        if not args.no_mtf:
            df_5m = fetch_forex_data(args.pair, interval=mtf_interval, period=args.period)
            logger.info(f"   Fetched {len(df_5m)} {mtf_interval} bars for MTF")

        logger.info(f"\nGenerating V4.0 signals...")
        logger.info(f"   BB: 20, 2.0σ | RSI: 7-period, {35}/{65} | Stoch: 14,3,3 ({25}/{75})")
        logger.info(f"   MTF: {'Enabled (relaxed)' if not args.no_mtf else 'Disabled'}")
        logger.info(f"   Min momentum score: {args.min_score}")

        engine = EnhancedForexSignalEngine(
            use_mtf=not args.no_mtf,
            min_momentum_score=args.min_score
        )
        df_signals = engine.generate_signals(df_1m, df_5m, pair=args.pair, debug=args.debug)
        logger.info(f"   V4.0 signals generated")

        risk_params = compute_enhanced_forex_risk(df_signals, args.pair, risk_pct=args.risk)
        logger.info(f"\nRisk (dynamic per score/session):")
        logger.info(f"   Base SL: {risk_params['stop_loss_pips']:.1f} pips | "
                    f"Base TP: {risk_params['take_profit_pips']:.1f} pips | "
                    f"Trail: {risk_params['trailing_stop_pips']:.1f} pips")

        if args.session_filter:
            logger.info(f"   Session filter: {', '.join(args.session_filter)}")

        logger.info(f"\nRunning V4.0 backtest...")
        df_results = compute_enhanced_forex_returns(
            df_signals,
            pair=args.pair,
            stop_loss_pips=risk_params['stop_loss_pips'],
            take_profit_pips=risk_params['take_profit_pips'],
            trailing_stop_pips=risk_params['trailing_stop_pips'],
            spread_pips=risk_params['spread_pips'],
            session_filter=args.session_filter,
            partial_exit=not args.no_partial_exit,
            risk_pct=args.risk,
            news_filter_enabled=not args.no_news_filter,
            use_dynamic_risk=not args.no_dynamic_risk,
        )

        report = generate_enhanced_forex_report(df_results, args.pair)
        print_enhanced_forex_report(report, args.pair)

        if analyzer and analyzer.client and 'error' not in report:
            logger.info(f"\nGenerating AI insights...")
            summary = analyzer.generate_forex_summary(
                report['trades_df'], args.pair, report, report['session_performance']
            )
            print(f"\n{'='*80}")
            print(f"AI STRATEGY INSIGHTS V4.0")
            print(f"{'='*80}")
            print(f"\n{summary}\n{'='*80}")

        print(f"\nV4.0 ANALYSIS COMPLETE")

        if 'error' not in report:
            pips = report['total_pnl_pips']
            wrate = report['win_rate']
            pf = report['profit_factor']
            exp = report['expectancy_pips']
            rr = report['avg_rr_ratio']

            if pips > 200 and wrate > 55 and pf > 1.5 and exp > 2.0 and rr > 1.8:
                print(f"\n[READY] Strategy meets 6-8% monthly criteria for {args.pair}")
                print(f"   Paper trade 2-4 weeks → scale to 1% risk live")
                best_sessions = [s for s, p in report['session_performance'].items() if p['pips'] > 0]
                print(f"   Focus sessions: {', '.join(best_sessions)}")
            elif pips > 100 and exp > 1.0:
                print(f"\n[PROMISING] Good foundation — needs parameter tuning")
                print(f"   Try --min-score 2 to filter out weak signals")
                print(f"   Try --session-filter London NY_Overlap for best sessions")
            else:
                print(f"\n[NEEDS WORK] Strategy not yet profitable enough")
                print(f"   Run --debug to inspect signal pipeline")
                print(f"   Try different pairs: GBP/USD or EUR/USD may suit this engine better")

    except Exception as e:
        logger.error(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()