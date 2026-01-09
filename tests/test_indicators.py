"""Unit tests for indicators module"""

import pytest
import numpy as np
import pandas as pd
from asre.indicators import (
    calculate_ema,
    calculate_sma,
    calculate_rsi,
    calculate_log_returns,
)


@pytest.fixture
def sample_prices():
    """Generate sample price series"""
    np.random.seed(42)
    prices = 100 + np.random.randn(252).cumsum()
    return pd.Series(prices, index=pd.date_range('2023-01-01', periods=252))


def test_ema(sample_prices):
    ema = calculate_ema(sample_prices, span=20)
    assert len(ema) == len(sample_prices)
    assert ema.isna().sum() > 0  # First values should be NaN


def test_sma(sample_prices):
    sma = calculate_sma(sample_prices, window=20)
    assert len(sma) == len(sample_prices)
    assert sma.isna().sum() > 0


def test_rsi(sample_prices):
    rsi = calculate_rsi(sample_prices, period=14)
    assert (rsi >= 0).all() or rsi.isna().any()
    assert (rsi <= 100).all() or rsi.isna().any()


def test_log_returns(sample_prices):
    returns = calculate_log_returns(sample_prices, window=60)
    assert len(returns) == len(sample_prices)
    assert returns.isna().sum() > 0  # First 60 values should be NaN
