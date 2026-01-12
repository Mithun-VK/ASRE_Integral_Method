"""
ASRE - Advanced Stock Rating Engine

A production-grade quantitative rating system combining:
- Fundamental analysis (F-Score)
- Technical analysis (T-Score)
- Momentum analysis (M-Score)
- Composite rating (R_ASRE / R_final)
"""

__version__ = "1.0.0"

# Config
from .config import (
    MomentumConfig,
    TechnicalConfig,
    FundamentalsConfig,
    CompositeConfig,
    BacktestConfig,
    get_default_configs,
)

# Core modules
from .fundamentals import compute_fundamental_score_universal as compute_fundamental_score
from .technical import compute_technical_score
from .momentum import compute_momentum_score
from .composite import (
    compute_asre_rating,
    compute_asre_medallion,
    compute_complete_asre,
)

# Backtest
from .backtest import Backtester

# Data loading
from .data_loader import load_stock_data, DataLoader

# Optimization
from .optimization import ASREOptimizer

__all__ = [
    # Version
    '__version__',
    
    # Config
    'MomentumConfig',
    'TechnicalConfig',
    'FundamentalsConfig',
    'CompositeConfig',
    'BacktestConfig',
    'get_default_configs',
    
    # Core functions
    'compute_fundamental_score',
    'compute_technical_score',
    'compute_momentum_score',
    'compute_asre_rating',
    'compute_asre_medallion',
    'compute_complete_asre',
    
    # Classes
    'Backtester',
    'DataLoader',
    'ASREOptimizer',
    
    # Convenience
    'load_stock_data',
]
