"""Performance Metrics Calculator"""
import numpy as np

class MetricsCalculator:
    @staticmethod
    def calculate_sharpe_ratio(returns):
        return returns.mean() / returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_cagr(total_return, years):
        return (1 + total_return) ** (1 / years) - 1
