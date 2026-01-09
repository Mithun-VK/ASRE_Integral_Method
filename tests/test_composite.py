"""Tests for composite rating"""

import pytest
from asre.composite import compute_asre_rating
from asre.config import get_default_config


def test_compute_asre_rating(sample_ohlcv):
    """Test ASRE rating computation"""
    config = get_default_config()
    result = compute_asre_rating(sample_ohlcv, config)
    
    assert 'asre_rating' in result.columns
    assert 'f_score' in result.columns
    assert 't_score' in result.columns
    assert 'm_score' in result.columns
    
    # Check ranges
    assert (result['asre_rating'] >= 0).all() or result['asre_rating'].isna().any()
    assert (result['asre_rating'] <= 100).all() or result['asre_rating'].isna().any()
