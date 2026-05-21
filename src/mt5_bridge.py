"""
🔗 MT5 BRIDGE - Connect forex_scalping_engine.py to MetaTrader 5
Paper trading via MT5 demo account
"""

import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# MT5 CONFIGURATION
# ============================================================================

MT5_CONFIG = {
    'login':    5046931376,          # Your MT5 demo account number
    'password': '2aNcIu@s',   # Your MT5 demo password
    'server':   'MetaQuotes-Demo', # Your broker's demo server
    'timeout':  10000,             # Connection timeout (ms)
    'magic':    234000,            # Unique EA identifier
}

# Map your pair format to MT5 symbols
SYMBOL_MAP = {
    'USD/JPY': 'USDJPY',
    'EUR/USD': 'EURUSD',
    'GBP/USD': 'GBPUSD',
    'AUD/USD': 'AUDUSD',
    'USD/CAD': 'USDCAD',
    'NZD/USD': 'NZDUSD',
    'EUR/JPY': 'EURJPY',
    'GBP/JPY': 'GBPJPY',
}

# Map your timeframe strings to MT5 timeframe constants
TIMEFRAME_MAP = {
    '1m':  mt5.TIMEFRAME_M1,
    '5m':  mt5.TIMEFRAME_M5,
    '15m': mt5.TIMEFRAME_M15,
    '30m': mt5.TIMEFRAME_M30,
    '1h':  mt5.TIMEFRAME_H1,
    '4h':  mt5.TIMEFRAME_H4,
    '1d':  mt5.TIMEFRAME_D1,
}


# ============================================================================
# MT5 CONNECTION
# ============================================================================

class MT5Bridge:
    """
    Bridge between forex_scalping_engine.py and MetaTrader 5.
    Handles connection, order execution, and position management.
    """

    def __init__(self, config: Dict = None):
        self.config = config or MT5_CONFIG
        self.connected = False
        self.account_info = None

    def connect(self) -> bool:
        """Initialize and connect to MT5 terminal."""
        if not mt5.initialize(
            login=self.config['login'],
            password=self.config['password'],
            server=self.config['server'],
            timeout=self.config['timeout']
        ):
            logger.error(f"❌ MT5 init failed: {mt5.last_error()}")
            return False

        self.account_info = mt5.account_info()
        if self.account_info is None:
            logger.error("❌ Failed to get account info")
            return False

        self.connected = True
        logger.info(
            f"✅ MT5 Connected!\n"
            f"   └─ Account: {self.account_info.login}\n"
            f"   └─ Broker:  {self.account_info.company}\n"
            f"   └─ Server:  {self.account_info.server}\n"
            f"   └─ Balance: ${self.account_info.balance:,.2f}\n"
            f"   └─ Equity:  ${self.account_info.equity:,.2f}\n"
            f"   └─ Mode:    {'DEMO ✅' if self.account_info.trade_mode == 0 else 'LIVE ⚠️'}"
        )
        return True

    def disconnect(self):
        """Shutdown MT5 connection."""
        mt5.shutdown()
        self.connected = False
        logger.info("🔌 MT5 disconnected")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    # ============================================================================
    # DATA FETCHING (replaces yfinance in your engine)
    # ============================================================================

    def get_ohlcv(self,
                  pair: str,
                  timeframe: str = '1m',
                  bars: int = 10000) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from MT5 - replaces yfinance.
        Returns DataFrame compatible with your existing signal logic.
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return None

        symbol = SYMBOL_MAP.get(pair, pair.replace('/', ''))
        tf = TIMEFRAME_MAP.get(timeframe, mt5.TIMEFRAME_M1)

        # Enable symbol in Market Watch if needed
        if not mt5.symbol_select(symbol, True):
            logger.error(f"❌ Symbol {symbol} not available")
            return None

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

        if rates is None or len(rates) == 0:
            logger.error(f"❌ No data for {symbol}: {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        }, inplace=True)

        logger.info(f"✅ Fetched {len(df)} {timeframe} bars for {symbol}")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # ============================================================================
    # ORDER EXECUTION
    # ============================================================================

    def get_symbol_info(self, pair: str) -> Optional[Dict]:
        """Get symbol details (pip size, digits, etc.)."""
        symbol = SYMBOL_MAP.get(pair, pair.replace('/', ''))
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return {
            'symbol': symbol,
            'bid': info.bid,
            'ask': info.ask,
            'spread_pips': (info.ask - info.bid) / info.point / 10,
            'point': info.point,
            'digits': info.digits,
            'pip_value': info.trade_tick_value,
            'min_lot': info.volume_min,
            'max_lot': info.volume_max,
            'lot_step': info.volume_step,
        }

    def calculate_lot_size(self,
                           pair: str,
                           risk_pct: float,
                           sl_pips: float) -> float:
        """
        Calculate lot size based on account balance, risk%, and SL distance.
        Matches your engine's 1% risk logic.
        """
        balance = mt5.account_info().balance
        risk_amount = balance * (risk_pct / 100)

        info = self.get_symbol_info(pair)
        if not info:
            return info['min_lot']

        symbol = SYMBOL_MAP.get(pair, pair.replace('/', ''))
        sym_info = mt5.symbol_info(symbol)

        pip_value_per_lot = sym_info.trade_tick_value / sym_info.trade_tick_size * sym_info.point * 10
        sl_value_per_lot = sl_pips * pip_value_per_lot

        if sl_value_per_lot <= 0:
            return sym_info.volume_min

        lot_size = risk_amount / sl_value_per_lot

        # Round to lot step
        lot_step = sym_info.volume_step
        lot_size = round(round(lot_size / lot_step) * lot_step, 2)
        lot_size = max(sym_info.volume_min, min(sym_info.volume_max, lot_size))

        return lot_size

    def place_order(self,
                    pair: str,
                    signal: str,       # 'BUY' or 'SELL'
                    sl_pips: float,
                    tp_pips: float,
                    risk_pct: float = 1.0,
                    comment: str = "ScalpBot") -> Optional[Dict]:
        """
        Place order on MT5 demo account.

        Args:
            pair: Trading pair (e.g., 'USD/JPY')
            signal: 'BUY' or 'SELL'
            sl_pips: Stop loss in pips
            tp_pips: Take profit in pips
            risk_pct: Risk percentage of account
            comment: Order comment

        Returns:
            Order result dict or None
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return None

        symbol = SYMBOL_MAP.get(pair, pair.replace('/', ''))
        sym_info = mt5.symbol_info(symbol)

        if sym_info is None:
            logger.error(f"❌ Symbol info unavailable: {symbol}")
            return None

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if signal == 'BUY' else tick.bid
        point = sym_info.point

        # Calculate SL/TP prices
        pip_size = point * 10  # 1 pip = 10 points for 5-digit brokers
        if signal == 'BUY':
            sl_price = price - (sl_pips * pip_size)
            tp_price = price + (tp_pips * pip_size)
            order_type = mt5.ORDER_TYPE_BUY
        else:
            sl_price = price + (sl_pips * pip_size)
            tp_price = price - (tp_pips * pip_size)
            order_type = mt5.ORDER_TYPE_SELL

        # Calculate lot size
        lot = self.calculate_lot_size(pair, risk_pct, sl_pips)

        request = {
            "action":     mt5.TRADE_ACTION_DEAL,
            "symbol":     symbol,
            "volume":     lot,
            "type":       order_type,
            "price":      price,
            "sl":         round(sl_price, sym_info.digits),
            "tp":         round(tp_price, sym_info.digits),
            "deviation":  20,
            "magic":      self.config['magic'],
            "comment":    comment,
            "type_time":  mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                f"❌ Order failed: {result.retcode} - {result.comment}\n"
                f"   Signal: {signal} {symbol} @ {price}\n"
                f"   SL: {sl_price:.5f} | TP: {tp_price:.5f}\n"
                f"   Lot: {lot}"
            )
            return None

        logger.info(
            f"✅ Order placed!\n"
            f"   └─ {signal} {symbol} @ {price:.5f}\n"
            f"   └─ SL: {sl_price:.5f} | TP: {tp_price:.5f}\n"
            f"   └─ Lot: {lot} | Ticket: {result.order}"
        )

        return {
            'ticket':  result.order,
            'symbol':  symbol,
            'signal':  signal,
            'price':   price,
            'sl':      sl_price,
            'tp':      tp_price,
            'lot':     lot,
            'time':    datetime.now(pytz.UTC)
        }

    def close_position(self, ticket: int) -> bool:
        """Close a position by ticket number."""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.warning(f"Position {ticket} not found")
            return False

        pos = position[0]
        symbol = pos.symbol
        lot = pos.volume
        order_type = (
            mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )
        tick = mt5.symbol_info_tick(symbol)
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action":   mt5.TRADE_ACTION_DEAL,
            "symbol":   symbol,
            "volume":   lot,
            "type":     order_type,
            "position": ticket,
            "price":    price,
            "deviation": 20,
            "magic":    self.config['magic'],
            "comment":  "Close by ScalpBot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"✅ Closed position {ticket}")
            return True
        else:
            logger.error(f"❌ Close failed: {result.comment}")
            return False

    def get_open_positions(self, pair: str = None) -> pd.DataFrame:
        """Get all open positions, optionally filtered by pair."""
        if pair:
            symbol = SYMBOL_MAP.get(pair, pair.replace('/', ''))
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if not positions:
            return pd.DataFrame()

        df = pd.DataFrame([p._asdict() for p in positions])
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        return df

    def get_account_summary(self) -> Dict:
        """Get current account P&L summary."""
        info = mt5.account_info()
        positions = mt5.positions_get()

        return {
            'balance':    info.balance,
            'equity':     info.equity,
            'margin':     info.margin,
            'free_margin': info.margin_free,
            'profit':     info.profit,
            'open_trades': len(positions) if positions else 0,
        }
