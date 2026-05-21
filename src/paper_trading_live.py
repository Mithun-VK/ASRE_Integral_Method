"""
🚀 PAPER TRADING LIVE v2.0 - Corrected imports for forex_scalping_engine.py V3.0
Connects EnhancedForexSignalEngine signals to MetaTrader 5 demo account.
"""

import time
import logging
from requests import session
import schedule
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import pytz
import json
import os

# ✅ CORRECT imports from your actual forex_scalping_engine.py
from forex_scalping_engine import (
    EnhancedForexSignalEngine,       # Class - has .generate_signals() method
    compute_enhanced_forex_risk,      # Function - calculates SL/TP/Trail
    ForexSessionDetector,             # Class - session detection
)

# ✅ ForexNewsFilter lives in forex_news_filter, not forex_scalping_engine
from forex_news_filter import ForexNewsFilter, NEWS_FILTER_CONFIG

import MetaTrader5 as mt5

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

MT5_CONFIG = {
    'login':    5047959948,          # Your MT5 demo account number
    'password': 'Xw!d8ePa',         # Your MT5 demo password
    'server':   'MetaQuotes-Demo',   # Your broker's demo server
    'timeout':  10000,               # Connection timeout (ms)
    'magic':    234000,              # Unique EA identifier
}

PAPER_CONFIG = {
    'pairs':           ['USD/JPY'],
    'session_filter': ['Asian', 'London', 'New_York', 'NY_Overlap'],
    'risk_pct':        0.5,
    'max_open_trades': 3,
    'check_interval':  60,
    'timezone':        'US/Eastern',
    'enable_news_filter': False,

    # Session-specific max trades (Asian is most active for JPY)
    'session_max_trades': {
        'Asian':      3,   # Most active for JPY pairs
        'London':     1,   # Moderate
        'New_York':   3,   # Most active overall
        'NY_Overlap': 2,   # Overlap - careful of volatility
    }
}

# MT5 symbol mapping (forex_scalping_engine pair -> MT5 symbol)
SYMBOL_MAP = {
    'EUR/USD': 'EURUSD',
    'GBP/USD': 'GBPUSD',
    'USD/JPY': 'USDJPY',
    'AUD/USD': 'AUDUSD',
    'USD/CAD': 'USDCAD',
    'NZD/USD': 'NZDUSD',
    'EUR/JPY': 'EURJPY',
    'GBP/JPY': 'GBPJPY',
}

TIMEFRAME_MAP = {
    '1m':  mt5.TIMEFRAME_M1,
    '5m':  mt5.TIMEFRAME_M5,
    '15m': mt5.TIMEFRAME_M15,
    '30m': mt5.TIMEFRAME_M30,
    '1h':  mt5.TIMEFRAME_H1,
}

# ✅ CHANGE 3: Persistent signal state file — survives restarts
SIGNAL_STATE_FILE = r"D:\asre-project\src\last_signals.json"


# ============================================================================
# MT5 DATA BRIDGE
# Returns DataFrame with SAME lowercase columns as forex_scalping_engine expects
# ============================================================================

class MT5DataBridge:
    """
    Fetches data from MT5 and returns it in the exact format
    that EnhancedForexSignalEngine.generate_signals() expects.

    forex_scalping_engine expects columns: date, open, high, low, close, volume
    MT5 returns columns: time, open, high, low, close, tick_volume
    """

    def get_ohlcv(self,
                  pair: str,
                  timeframe: str = '1m',
                  bars: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV from MT5 and format for EnhancedForexSignalEngine.

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
            (matching forex_scalping_engine's fetch_forex_data output)
        """
        symbol = SYMBOL_MAP.get(pair, pair.replace('/', ''))
        tf = TIMEFRAME_MAP.get(timeframe, mt5.TIMEFRAME_M1)

        # Make symbol available in MT5 MarketWatch
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Symbol {symbol} not available in MT5")
            return None

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

        if rates is None or len(rates) == 0:
            logger.error(f"No MT5 data for {symbol}: {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)

        # ✅ Convert to match forex_scalping_engine column format exactly
        df['date'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df.rename(columns={'tick_volume': 'volume'})
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

        logger.debug(f"MT5 data: {len(df)} {timeframe} bars for {symbol}")
        return df


# ============================================================================
# MT5 ORDER EXECUTOR
# ============================================================================

class MT5OrderExecutor:
    """Handles order placement and position management on MT5."""

    def __init__(self, magic: int = 234000):
        self.magic = magic

    def get_symbol_info(self, pair: str) -> Optional[Dict]:
        """Get live symbol info from MT5."""
        symbol = SYMBOL_MAP.get(pair, pair.replace('/', ''))
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol info not available: {symbol}")
            return None
        tick = mt5.symbol_info_tick(symbol)
        return {
            'symbol':      symbol,
            'bid':         tick.bid,
            'ask':         tick.ask,
            'spread_pips': round((tick.ask - tick.bid) / info.point / 10, 1),
            'point':       info.point,
            'digits':      info.digits,
            'min_lot':     info.volume_min,
            'max_lot':     info.volume_max,
            'lot_step':    info.volume_step,
        }

    def calculate_lot_size(self,
                           pair: str,
                           risk_pct: float,
                           sl_pips: float) -> float:
        """Calculate position size matching forex_scalping_engine's 1% risk logic."""
        balance = mt5.account_info().balance
        risk_amount = balance * (risk_pct / 100)

        symbol = SYMBOL_MAP.get(pair, pair.replace('/', ''))
        info = mt5.symbol_info(symbol)
        if info is None:
            return info.volume_min

        pip_value_per_lot = (
            info.trade_tick_value / info.trade_tick_size * info.point * 10
        )
        sl_value = sl_pips * pip_value_per_lot

        if sl_value <= 0:
            return info.volume_min

        lot = risk_amount / sl_value
        lot = round(round(lot / info.volume_step) * info.volume_step, 2)
        lot = max(info.volume_min, min(info.volume_max, lot))
        return lot

    def place_order(self,
                    pair: str,
                    signal: int,       # 1 = BUY, -1 = SELL
                    sl_pips: float,
                    tp_pips: float,
                    risk_pct: float = 1.0) -> Optional[Dict]:
        """
        Place order on MT5.

        Args:
            signal: 1 for BUY, -1 for SELL (matches your engine's signal column)
        """
        symbol = SYMBOL_MAP.get(pair, pair.replace('/', ''))
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return None

        tick = mt5.symbol_info_tick(symbol)
        pip_size = sym_info.point * 10  # 5-digit broker: 1 pip = 10 points

        if signal == 1:  # BUY
            price = tick.ask
            sl_price = price - (sl_pips * pip_size)
            tp_price = price + (tp_pips * pip_size)
            order_type = mt5.ORDER_TYPE_BUY
            direction = "BUY"
        else:  # SELL
            price = tick.bid
            sl_price = price + (sl_pips * pip_size)
            tp_price = price - (tp_pips * pip_size)
            order_type = mt5.ORDER_TYPE_SELL
            direction = "SELL"

        lot = self.calculate_lot_size(pair, risk_pct, sl_pips)

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lot,
            "type":         order_type,
            "price":        price,
            "sl":           round(sl_price, sym_info.digits),
            "tp":           round(tp_price, sym_info.digits),
            "deviation":    20,
            "magic":        self.magic,
            "comment":      f"ScalpBot_NY_{direction}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                f"Order failed [{result.retcode}]: {result.comment}\n"
                f"   {direction} {symbol} | SL: {sl_price:.5f} | TP: {tp_price:.5f}"
            )
            return None

        logger.info(
            f"{direction} {symbol}\n"
            f"   Entry: {price:.5f} | SL: {sl_price:.5f} | TP: {tp_price:.5f}\n"
            f"   Lot: {lot} | Ticket: #{result.order}\n"
            f"   Risk: {risk_pct}% | SL: {sl_pips} pips | TP: {tp_pips} pips"
        )

        return {
            'ticket': result.order,
            'symbol': symbol,
            'signal': signal,
            'price':  price,
            'sl':     sl_price,
            'tp':     tp_price,
            'lot':    lot,
        }

    def get_open_positions(self, pair: str = None) -> pd.DataFrame:
        """Get open positions for a pair."""
        symbol = SYMBOL_MAP.get(pair, pair.replace('/', '')) if pair else None
        positions = (
            mt5.positions_get(symbol=symbol)
            if symbol else mt5.positions_get()
        )
        if not positions:
            return pd.DataFrame()
        df = pd.DataFrame([p._asdict() for p in positions])
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        return df

    def get_account_summary(self) -> Dict:
        """Get current account status."""
        info = mt5.account_info()
        positions = mt5.positions_get() or []
        return {
            'balance':     info.balance,
            'equity':      info.equity,
            'profit':      info.profit,
            'free_margin': info.margin_free,
            'open_trades': len(positions),
            'is_demo':     info.trade_mode == 0,
        }


# ============================================================================
# LIVE PAPER TRADER
# ============================================================================

class LivePaperTrader:
    """
    Live paper trading engine.

    Connects your existing EnhancedForexSignalEngine to MT5 demo account.
    Signal generation is 100% identical to backtesting.
    """

    def __init__(self, config: Dict = None):
        self.config = config or PAPER_CONFIG
        self.tz = pytz.timezone(self.config['timezone'])

        # ✅ CHANGE 2: Relaxed engine parameters — more signals, less strict MTF gate
        self.signal_engine = EnhancedForexSignalEngine(
            rsi_oversold=35,       # was 30 — catches bounces in downtrend
            rsi_overbought=65,     # was 70 — less strict
            rsi_period=7,
            bb_std=2.0,            # was 2.2 — more band touches
            bb_period=20,
            stoch_k=14,
            stoch_d=3,
            stoch_oversold=25,     # was 20
            stoch_overbought=75,   # was 80
            use_mtf=True,
        )

        self.session_detector = ForexSessionDetector()
        self.data_bridge = MT5DataBridge()
        self.executor = MT5OrderExecutor(magic=MT5_CONFIG['magic'])

        # News filter (bypass by default until fixed)
        self.news_filter = ForexNewsFilter(
            timezone=self.config['timezone']
        ) if self.config.get('enable_news_filter') else None

        self.trade_log: List[Dict] = []

        # ✅ CHANGE 3: Load persisted signal state — survives restarts
        if os.path.exists(SIGNAL_STATE_FILE):
            with open(SIGNAL_STATE_FILE) as f:
                self._last_signal = json.load(f)
            logger.info(f"Loaded persisted signal state: {self._last_signal}")
        else:
            self._last_signal = {}

    # --------------------------------------------------------
    # CHANGE 3: Persist signal state to disk after every update
    # --------------------------------------------------------

    def _save_signal_state(self):
        """Write _last_signal to disk so restarts don't re-fire stale signals."""
        with open(SIGNAL_STATE_FILE, 'w') as f:
            json.dump(self._last_signal, f)

    # --------------------------------------------------------
    # SESSION CHECK
    # --------------------------------------------------------

    def is_in_session(self) -> bool:
        """
        Check if current time is within any configured trading session.
        Handles Asian session midnight crossover correctly.
        """
        now_et = datetime.now(self.tz)

        # Skip weekends entirely (forex closed Sat + most of Sun)
        if now_et.weekday() == 5:  # Saturday - always closed
            return False
        if now_et.weekday() == 6 and now_et.hour < 17:  # Sunday before 5PM ET
            return False

        session = self.session_detector.get_session(now_et)
        return session in self.config['session_filter']

    # --------------------------------------------------------
    # CORE TRADING LOGIC
    # --------------------------------------------------------

    def check_and_trade(self, pair: str):
        """Main trading check - session-aware with persistent signal state."""

        # 1. Session check
        if not self.is_in_session():
            logger.debug(f"Outside session - skipping {pair}")
            return

        # 2. Get current session for dynamic limits
        now_et = datetime.now(self.tz)
        current_session = self.session_detector.get_session(now_et)

        # 3. Session-specific max trades
        session_max = self.config.get('session_max_trades', {}).get(
            current_session,
            self.config['max_open_trades']
        )

        # 4. News filter check
        if self.news_filter:
            now_utc = datetime.now(pytz.UTC)
            safe, event = self.news_filter.is_safe_to_trade(now_utc, pair)
            if not safe:
                logger.warning(
                    f"News block [{pair}]: {event['event']} "
                    f"in {event['minutes_until']}min"
                )
                return

        # 5. Max open trades check (session-aware)
        open_positions = self.executor.get_open_positions(pair)
        if len(open_positions) >= session_max:
            logger.info(f"Max trades ({session_max}) reached for {pair} in {current_session}")
            return

        # 6. Fetch live data from MT5
        df_1m = self.data_bridge.get_ohlcv(pair, timeframe='1m', bars=500)
        df_5m = self.data_bridge.get_ohlcv(pair, timeframe='5m', bars=200)

        if df_1m is None or len(df_1m) < 50:
            logger.error(f"Insufficient data for {pair}")
            return

        # 7. Generate signals
        df_signals = self.signal_engine.generate_signals(
            df_1m, df_5m, pair=pair, debug=False
        )

        latest_signal = int(df_signals['signal'].iloc[-1])
        last_signal = self._last_signal.get(pair, 0)

        if latest_signal == last_signal or latest_signal == 0:
            logger.info(f"[{pair}] Signal check → latest={latest_signal}, last={last_signal} | session={current_session}")
            return

        # ✅ CHANGE 3: Persist immediately after updating signal state
        self._last_signal[pair] = latest_signal
        self._save_signal_state()
        momentum_score = int(df_signals['buy_score'].iloc[-1] 
                     if latest_signal == 1 
                     else df_signals['sell_score'].iloc[-1])

        # 8. Risk parameters
        risk_params = compute_enhanced_forex_risk(
            df_signals, pair,
            risk_pct=self.config['risk_pct'],
            momentum_score=momentum_score,
            session=current_session,           # ← pass live session
        )

        direction = "BUY" if latest_signal == 1 else "SELL"
        logger.info(
            f"\nNEW SIGNAL [{current_session}]: {direction} {pair}\n"
            f"   SL: {risk_params['stop_loss_pips']:.1f} pips\n"
            f"   TP: {risk_params['take_profit_pips']:.1f} pips"
        )

        # 9. Execute order
        result = self.executor.place_order(
            pair=pair,
            signal=latest_signal,
            sl_pips=risk_params['stop_loss_pips'],
            tp_pips=risk_params['take_profit_pips'],
            risk_pct=self.config['risk_pct'],
        )

        if result:
            self.trade_log.append({
                'time':    datetime.now(self.tz),
                'pair':    pair,
                'signal':  direction,
                'session': current_session,
                'sl':      risk_params['stop_loss_pips'],
                'tp':      risk_params['take_profit_pips'],
                'ticket':  result['ticket'],
            })

    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    def print_dashboard(self):
        """Enhanced dashboard showing session-aware status."""
        summary = self.executor.get_account_summary()
        now = datetime.now(self.tz)
        in_session = self.is_in_session()
        current_session = self.session_detector.get_session(now)

        # Next session countdown
        if in_session:
            session_status = f"{current_session.upper()} - TRADING ACTIVE"
        else:
            if now.weekday() >= 5:
                session_status = "Weekend - Opens Sun 5PM EST"
            elif now.hour < 19:
                session_status = "Waiting - Asian opens 7PM EST"
            else:
                session_status = "Outside tracked session"

        print("\n" + "="*65)
        print(f"USD/JPY PAPER TRADING | {now.strftime('%a %Y-%m-%d %H:%M %Z')}")
        print("="*65)
        print(f"   {'MODE:':<18} DEMO ACCOUNT")
        print(f"   {'Balance:':<18} ${summary['balance']:>12,.2f}")
        print(f"   {'Equity:':<18} ${summary['equity']:>12,.2f}")
        print(f"   {'Session P&L:':<18} ${summary['profit']:>+12,.2f}")
        print(f"   {'Free Margin:':<18} ${summary['free_margin']:>12,.2f}")
        print(f"   {'Open Trades:':<18} {summary['open_trades']}")
        print(f"   {'Session:':<18} {session_status}")
        print(f"   {'Total Logged:':<18} {len(self.trade_log)} trades")

        # Session breakdown of logged trades
        if self.trade_log:
            from collections import Counter
            session_counts = Counter(t.get('session', '?') for t in self.trade_log)
            print(f"\n   Trades by session:")
            for sess, count in session_counts.items():
                print(f"   --- {sess:<12}: {count} trades")

            print(f"\n   Last 5 trades:")
            for t in self.trade_log[-5:]:
                print(
                    f"   --- [{t.get('session', '?'):10}] "
                    f"{t['time'].strftime('%a %H:%M')} | "
                    f"{t['signal']} {t['pair']} | "
                    f"#{t['ticket']}"
                )

        print("="*65)

    # --------------------------------------------------------
    # MAIN RUNNER
    # --------------------------------------------------------

    def run(self):
        """Initialize MT5 and start paper trading loop."""
        print("\n" + "="*65)
        print("LIVE PAPER TRADING STARTED")
        print("="*65)

        # Connect to MT5
        if not mt5.initialize(
            login=MT5_CONFIG['login'],
            password=MT5_CONFIG['password'],
            server=MT5_CONFIG['server'],
            timeout=MT5_CONFIG['timeout']
        ):
            logger.error(f"MT5 connection failed: {mt5.last_error()}")
            logger.error("  Check: login, password, server name in MT5_CONFIG")
            logger.error("  Ensure MT5 terminal is running and logged in")
            return

        # Verify demo account
        account = mt5.account_info()
        if account.trade_mode != 0:
            logger.error("NOT a demo account! Aborting for safety.")
            mt5.shutdown()
            return

        logger.info(
            f"\nMT5 Connected!\n"
            f"   Account: #{account.login}\n"
            f"   Broker:  {account.company}\n"
            f"   Server:  {account.server}\n"
            f"   Balance: ${account.balance:,.2f}\n"
            f"   Mode:    DEMO ACCOUNT"
        )
        logger.info(f"\nConfig:")
        logger.info(f"   Pairs:       {', '.join(self.config['pairs'])}")
        logger.info(f"   Sessions:    {', '.join(self.config['session_filter'])}")
        logger.info(f"   Risk:        {self.config['risk_pct']}% per trade")
        logger.info(f"   Interval:    every {self.config['check_interval']}s")
        logger.info(f"   News filter: {'Enabled' if self.news_filter else 'Bypassed'}")

        # Schedule jobs
        for pair in self.config['pairs']:
            schedule.every(self.config['check_interval']).seconds.do(
                self.check_and_trade, pair=pair
            )

        schedule.every(5).minutes.do(self.print_dashboard)

        # Initial dashboard + first check
        self.print_dashboard()
        for pair in self.config['pairs']:
            self.check_and_trade(pair)

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\nPaper trading stopped (Ctrl+C)")
        finally:
            mt5.shutdown()
            logger.info("MT5 disconnected")

            # Final summary
            if self.trade_log:
                print(f"\nSESSION SUMMARY: {len(self.trade_log)} trades placed")
                for t in self.trade_log:
                    print(f"   {t['time'].strftime('%H:%M')} | {t['signal']} {t['pair']} | #{t['ticket']}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # UPDATE MT5_CONFIG BEFORE RUNNING:
    # MT5_CONFIG['login']    = your demo account number
    # MT5_CONFIG['password'] = your demo password
    # MT5_CONFIG['server']   = broker server (e.g. 'ICMarketsSC-Demo01')

    trader = LivePaperTrader(PAPER_CONFIG)
    trader.run()