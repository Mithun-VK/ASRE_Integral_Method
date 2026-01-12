"""
ASRE Configuration Module

Centralized configuration using Pydantic v2 for type safety and validation.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Base Configuration
# ---------------------------------------------------------------------------

class ASREBaseConfig(BaseModel):
    """Base configuration with common settings."""
    
    class Config:
        """Pydantic v2 config."""
        arbitrary_types_allowed = True
        validate_assignment = True


# ---------------------------------------------------------------------------
# Momentum Configuration
# ---------------------------------------------------------------------------

class MomentumConfig(ASREBaseConfig):
    """Momentum score configuration with all window parameters."""
    
    kappa: float = Field(default=0.03, ge=0.01, le=0.1)
    beta_m: float = Field(default=0.2, ge=0.05, le=0.5)
    lookback_short: int = Field(default=10, ge=5, le=20)
    lookback_long: int = Field(default=30, ge=20, le=60)
    
    # Comprehensive window parameters
    window_10d: int = Field(default=10, ge=5, le=20)
    window_30d: int = Field(default=30, ge=20, le=40)
    window_60d: int = Field(default=60, ge=50, le=80)
    window_momentum: int = Field(default=20, ge=10, le=40)
    window_vol: int = Field(default=20, ge=10, le=40)
    
    @field_validator('kappa')
    @classmethod
    def validate_kappa(cls, v: float) -> float:
        if not 0.02 <= v <= 0.05:
            raise ValueError(f"kappa should be between 0.02 and 0.05, got {v}")
        return v


# ---------------------------------------------------------------------------
# Technical Configuration
# ---------------------------------------------------------------------------

class TechnicalConfig(ASREBaseConfig):
    """Technical score configuration with all window parameters."""
    
    gamma: float = Field(default=0.1, ge=0.05, le=0.3)
    theta: float = Field(default=0.1, ge=0.05, le=0.3)
    rsi_period: int = Field(default=14, ge=10, le=20)
    bb_period: int = Field(default=20, ge=15, le=30)
    bb_std: float = Field(default=2.0, ge=1.5, le=3.0)
    
    # Comprehensive window parameters
    window_20d: int = Field(default=20, ge=15, le=30)
    window_50d: int = Field(default=50, ge=40, le=60)
    window_200d: int = Field(default=200, ge=150, le=250)
    window_rsi: int = Field(default=14, ge=10, le=20)
    window_bb: int = Field(default=20, ge=15, le=30)
    window_macd_fast: int = Field(default=12, ge=8, le=16)
    window_macd_slow: int = Field(default=26, ge=20, le=35)
    window_macd_signal: int = Field(default=9, ge=6, le=12)
    window_adx: int = Field(default=14, ge=10, le=20)
    window_atr: int = Field(default=14, ge=10, le=20)
    window_ema_short: int = Field(default=12, ge=8, le=20)
    window_ema_long: int = Field(default=26, ge=20, le=40)
    
    @field_validator('gamma', 'theta')
    @classmethod
    def validate_sensitivity(cls, v: float, info) -> float:
        if not 0.05 <= v <= 0.2:
            raise ValueError(f"{info.field_name} should be between 0.05 and 0.2, got {v}")
        return v


# ---------------------------------------------------------------------------
# Fundamentals Configuration
# ---------------------------------------------------------------------------

class FundamentalsConfig(ASREBaseConfig):
    """Fundamental score configuration with all window parameters."""
    
    alpha: float = Field(default=0.12, ge=0.01, le=0.1)
    beta_1: float = Field(default=0.4, ge=0.1, le=0.7)
    beta_2: float = Field(default=0.35, ge=0.1, le=0.6)
    beta_3: float = Field(default=0.25, ge=0.05, le=0.5)
    
    pe_optimal: float = Field(default=18.0, ge=10.0, le=30.0)
    roe_target: float = Field(default=20.0, ge=10.0, le=40.0)
    de_threshold: float = Field(default=0.5, ge=0.2, le=1.5)
    # Market proxy modulation strength
    market_proxy_weight: float = 0.15

    # Proxy smoothing window
    proxy_window: int = 60
    
    # Comprehensive window parameters
    window_252d: int = Field(default=252, ge=200, le=300)
    window_63d: int = Field(default=63, ge=40, le=80)
    window_annual: int = Field(default=252, ge=200, le=300)
    window_quarterly: int = Field(default=63, ge=40, le=80)
    window_pe: int = Field(default=252, ge=200, le=300)
    window_roe: int = Field(default=252, ge=200, le=300)
    window_de: int = Field(default=252, ge=200, le=300)
    
    @model_validator(mode='after')
    def validate_weights_sum(self) -> 'FundamentalsConfig':
        total = self.beta_1 + self.beta_2 + self.beta_3
        if not 0.95 <= total <= 1.05:
            raise ValueError(f"Beta weights must sum to ~1.0, got {total:.3f}")
        return self


# ---------------------------------------------------------------------------
# Composite Configuration
# ---------------------------------------------------------------------------

class CompositeConfig(ASREBaseConfig):
    """Composite rating configuration with all window parameters."""
    
    w_f_base: float = Field(default=0.4, ge=0.2, le=0.6)
    w_t_base: float = Field(default=0.35, ge=0.2, le=0.5)
    w_m_base: float = Field(default=0.25, ge=0.1, le=0.4)
    
    lambda_risk: float = Field(default=2.0, ge=1.0, le=3.0)
    phi: float = Field(default=0.1, ge=0.05, le=0.2)
    psi: float = Field(default=0.1, ge=0.05, le=0.2)
    eta: float = Field(default=0.02, ge=0.01, le=0.1)
    sigma_obs: float = Field(default=0.2, ge=0.1, le=0.5)
    
    # Comprehensive window parameters
    window_kalman: int = Field(default=60, ge=30, le=100)
    window_risk: int = Field(default=60, ge=30, le=100)
    window_correlation: int = Field(default=60, ge=30, le=100)
    window_regime: int = Field(default=60, ge=30, le=100)
    window_vix: int = Field(default=20, ge=10, le=40)
    
    @model_validator(mode='after')
    def validate_base_weights_sum(self) -> 'CompositeConfig':
        total = self.w_f_base + self.w_t_base + self.w_m_base
        if not 0.95 <= total <= 1.05:
            raise ValueError(f"Base weights must sum to ~1.0, got {total:.3f}")
        return self


# ---------------------------------------------------------------------------
# Backtest Configuration
# ---------------------------------------------------------------------------

class BacktestConfig(ASREBaseConfig):
    """Backtesting configuration."""
    
    threshold_long: float = Field(default=70.0, ge=50.0, le=90.0)
    threshold_short: float = Field(default=30.0, ge=10.0, le=50.0)
    
    transaction_cost: float = Field(default=0.001, ge=0.0, le=0.01)
    slippage: float = Field(default=0.0005, ge=0.0, le=0.005)
    max_position: float = Field(default=1.0, ge=0.0, le=5.0)
    
    periods_per_year: int = Field(default=252, ge=200, le=365)
    
    @model_validator(mode='after')
    def validate_thresholds(self) -> 'BacktestConfig':
        if self.threshold_long <= self.threshold_short:
            raise ValueError("Long threshold must be greater than short threshold")
        return self


# ---------------------------------------------------------------------------
# Application Settings
# ---------------------------------------------------------------------------

class AppSettings(BaseSettings):
    """Application-level settings."""
    
    cache_dir: str = Field(default="data/cache", alias="ASRE_CACHE_DIR")
    cache_ttl_days: int = Field(default=1, alias="ASRE_CACHE_TTL_DAYS")
    log_level: str = Field(default="INFO", alias="ASRE_LOG_LEVEL")
    data_source: str = Field(default="yfinance", alias="ASRE_DATA_SOURCE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------

def get_default_configs() -> Dict[str, ASREBaseConfig]:
    """Get default configuration objects for all modules."""
    return {
        'momentum': MomentumConfig(),
        'technical': TechnicalConfig(),
        'fundamentals': FundamentalsConfig(),
        'composite': CompositeConfig(),
        'backtest': BacktestConfig(),
    }


def load_configs_from_dict(config_dict: Dict[str, Dict[str, Any]]) -> Dict[str, ASREBaseConfig]:
    """Load config objects from nested dictionary."""
    configs = {}
    
    if 'momentum' in config_dict:
        configs['momentum'] = MomentumConfig(**config_dict['momentum'])
    else:
        configs['momentum'] = MomentumConfig()
    
    if 'technical' in config_dict:
        configs['technical'] = TechnicalConfig(**config_dict['technical'])
    else:
        configs['technical'] = TechnicalConfig()
    
    if 'fundamentals' in config_dict:
        configs['fundamentals'] = FundamentalsConfig(**config_dict['fundamentals'])
    else:
        configs['fundamentals'] = FundamentalsConfig()
    
    if 'composite' in config_dict:
        configs['composite'] = CompositeConfig(**config_dict['composite'])
    else:
        configs['composite'] = CompositeConfig()
    
    if 'backtest' in config_dict:
        configs['backtest'] = BacktestConfig(**config_dict['backtest'])
    else:
        configs['backtest'] = BacktestConfig()
    
    return configs


def save_configs_to_dict(configs: Dict[str, ASREBaseConfig]) -> Dict[str, Dict[str, Any]]:
    """Convert config objects to nested dictionary for JSON serialization."""
    return {
        name: config.model_dump()
        for name, config in configs.items()
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = [
    'MomentumConfig',
    'TechnicalConfig',
    'FundamentalsConfig',
    'CompositeConfig',
    'BacktestConfig',
    'AppSettings',
    'get_default_configs',
    'load_configs_from_dict',
    'save_configs_to_dict',
]
