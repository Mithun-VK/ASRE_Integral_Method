"""Pytest configuration and shared fixtures"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data"""
    np.random.seed(42)
    n = 252
    close = 100 + np.random.randn(n).cumsum()
    
    df = pd.DataFrame({
        'open': close + np.random.randn(n) * 0.5,
        'high': close + np.random.uniform(0, 2, n),
        'low': close - np.random.uniform(0, 2, n),
        'close': close,
        'volume': np.random.randint(1000000, 10000000, n),
        'pe': np.random.uniform(10, 30, n),
        'roe': np.random.uniform(0.05, 0.25, n),
        'de': np.random.uniform(0.3, 1.5, n),
    }, index=pd.date_range('2023-01-01', periods=n))
    
    return df
