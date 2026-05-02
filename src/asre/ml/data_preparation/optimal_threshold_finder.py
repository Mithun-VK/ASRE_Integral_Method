"""Optimal Threshold Finder"""
import pandas as pd
from typing import Dict

class OptimalThresholdFinder:
    def __init__(self, backtest_engine):
        self.backtest_engine = backtest_engine
    
    def grid_search_thresholds(self, df: pd.DataFrame, date) -> Dict:
        # TODO: Implement grid search
        pass
