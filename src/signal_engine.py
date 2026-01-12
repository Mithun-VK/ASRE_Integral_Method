"""
ASRE Signal Engine - Hybrid Threshold Logic with Buy/Sell Separation
Mitigates risk by enforcing an absolute quality floor on dynamic thresholds.

✅ NEW: Separate Buy and Sell thresholds (Hysteresis band)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SignalEngine:

    def __init__(self, floor: float = 65.0, sell_threshold: float = 55.0, 
                 sensitivity: float = 0.5, window: int = 60):
        """
        Args:
            floor (float): Absolute minimum rating required to BUY (default: 65.0)
            sell_threshold (float): Rating below which to SELL (default: 55.0)
            sensitivity (float): Multiplier for standard deviation (higher = stricter dynamic)
            window (int): Rolling window for dynamic calculation
        
        NEW FEATURE:
        - Hysteresis band prevents whipsaw trades
        - Buy when R_final >= floor (65)
        - Sell when R_final <= sell_threshold (55)
        - Hold in between (55-65 range)
        """
        self.floor = floor
        self.sell_threshold = sell_threshold  # ✅ NEW
        self.sensitivity = sensitivity
        self.window = window
        
        # Validation
        if self.floor <= self.sell_threshold:
            logger.warning(f"⚠️ Buy floor ({floor}) should be > Sell threshold ({sell_threshold})")

    def generate_signals(self, df: pd.DataFrame, rating_col: str = 'r_final') -> pd.DataFrame:
        """
        Generate Buy/Sell signals using Hybrid Thresholds with Hysteresis.
        
        NEW LOGIC:
        - BUY:  R_final >= buy_floor (65)
        - SELL: R_final <= sell_threshold (55)
        - HOLD: Between thresholds (55-65)
        """
        result = df.copy()

        # 1. Calculate Dynamic Stats (Relative Quality)
        rolling_mean = result[rating_col].rolling(window=self.window, min_periods=5).mean()
        rolling_std = result[rating_col].rolling(window=self.window, min_periods=5).std()

        # 2. Calculate Dynamic Buy Threshold
        dynamic_buy = rolling_mean + (self.sensitivity * rolling_std)
        result['buy_threshold'] = dynamic_buy.clip(lower=self.floor)  # Never below floor
        
        # ✅ NEW: Calculate Dynamic Sell Threshold
        dynamic_sell = rolling_mean - (self.sensitivity * rolling_std)
        result['sell_threshold'] = dynamic_sell.clip(upper=self.sell_threshold)  # Never above sell_threshold

        # 3. Generate Signals with Hysteresis
        result['signal'] = 0
        result['raw_signal'] = 0  # For debugging

        # ✅ NEW: State Machine (Prevents whipsaw)
        position = 0  # 0=Cash, 1=Long
        signals = []
        
        for i in range(len(result)):
            rating = result.iloc[i][rating_col]
            buy_thresh = result.iloc[i]['buy_threshold']
            sell_thresh = result.iloc[i]['sell_threshold']
            
            # BUY Logic
            if position == 0 and rating >= buy_thresh:
                position = 1
                signals.append(1)
            # SELL Logic
            elif position == 1 and rating <= sell_thresh:
                position = 0
                signals.append(0)
            # HOLD
            else:
                signals.append(position)
        
        result['signal'] = signals

        # 4. Optional: Mark threshold crossings
        result['buy_signal_raw'] = (result[rating_col] >= result['buy_threshold']).astype(int)
        result['sell_signal_raw'] = (result[rating_col] <= result['sell_threshold']).astype(int)

        return result

    def get_current_state(self, current_rating: float, history: pd.Series) -> dict:
        """
        Get the current threshold state for live trading.
        
        ✅ UPDATED: Returns both buy and sell thresholds
        """
        # Calculate stats from recent history
        mean = history.tail(self.window).mean()
        std = history.tail(self.window).std()

        # Buy threshold
        dynamic_buy = mean + (self.sensitivity * std)
        hybrid_buy = max(dynamic_buy, self.floor)
        
        # ✅ NEW: Sell threshold
        dynamic_sell = mean - (self.sensitivity * std)
        hybrid_sell = min(dynamic_sell, self.sell_threshold)

        # Determine signal
        if current_rating >= hybrid_buy:
            signal = "BUY"
        elif current_rating <= hybrid_sell:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "current_rating": round(current_rating, 2),
            "buy_threshold": round(hybrid_buy, 2),
            "sell_threshold": round(hybrid_sell, 2),
            "hysteresis_band": round(hybrid_buy - hybrid_sell, 2),
            "safety_floor": self.floor,
            "signal": signal,
            "is_buy_floor_active": hybrid_buy == self.floor,
            "is_sell_floor_active": hybrid_sell == self.sell_threshold
        }
    
    def get_threshold_summary(self):
        """Print current configuration."""
        return {
            'buy_floor': self.floor,
            'sell_threshold': self.sell_threshold,
            'hysteresis_band': self.floor - self.sell_threshold,
            'sensitivity': self.sensitivity,
            'window': self.window
        }
