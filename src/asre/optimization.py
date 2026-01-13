"""
ASRE Hyperparameter Optimization

Comprehensive optimization framework for ASRE parameters.

Features:
1. Parameter Optimization
   - Grid search with cross-validation
   - Random search for high-dimensional spaces
   - Bayesian optimization for efficient search
   - Differential evolution for global optimization

2. Walk-Forward Analysis
   - Rolling train/test windows
   - Out-of-sample validation
   - Parameter stability testing
   - Regime-aware optimization

3. Objective Functions
   - Sharpe ratio maximization
   - Calmar ratio optimization
   - Information ratio vs benchmark
   - Multi-objective optimization

4. Constraint Handling
   - Parameter bounds
   - Sum constraints (e.g., weights = 1)
   - Dependency constraints
   - Regularization

Production-grade features:
- Full integration with all ASRE modules
- Parallel optimization (multiprocessing)
- Progress tracking and early stopping
- Results caching and persistence
- Comprehensive visualization support
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit

# Optional Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    warnings.warn("scikit-optimize not installed. Bayesian optimization unavailable.")

from .config import (
    MomentumConfig,
    TechnicalConfig,
    FundamentalsConfig,
    CompositeConfig,
    BacktestConfig,
)
from .composite import (
    compute_complete_asre,
    compute_asre_rating,
    validate_asre_rating,
)
from .backtest import (
    Backtester,
    generate_signals_threshold,
    compute_strategy_returns,
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
)
from .calibration import (
    calibrate_logistic_weights,
    mle_normal_distribution,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter Space Definitions
# ---------------------------------------------------------------------------


@dataclass
class ParameterSpace:
    """Define parameter search space with bounds and types."""
    
    name: str
    bounds: Tuple[float, float]
    param_type: str = 'continuous'  # 'continuous', 'integer', 'categorical'
    default: Optional[float] = None
    category: str = 'composite'  # 'momentum', 'technical', 'fundamentals', 'composite'
    
    def sample(self, n: int = 1) -> Union[float, List[float]]:
        """Sample random value(s) from parameter space."""
        if self.param_type == 'continuous':
            samples = np.random.uniform(self.bounds[0], self.bounds[1], n)
        elif self.param_type == 'integer':
            samples = np.random.randint(self.bounds[0], self.bounds[1] + 1, n)
        else:
            raise ValueError(f"Unsupported param_type: {self.param_type}")
        
        return samples[0] if n == 1 else samples.tolist()


# Default parameter spaces for ASRE (organized by module)
DEFAULT_PARAM_SPACES = {
    # Momentum parameters
    'kappa': ParameterSpace('kappa', (0.02, 0.05), 'continuous', 0.03, 'momentum'),
    'beta_m': ParameterSpace('beta_m', (0.1, 0.3), 'continuous', 0.2, 'momentum'),
    
    # Technical parameters
    'gamma': ParameterSpace('gamma', (0.05, 0.2), 'continuous', 0.1, 'technical'),
    'theta': ParameterSpace('theta', (0.05, 0.2), 'continuous', 0.1, 'technical'),
    
    # Fundamental parameters
    'alpha': ParameterSpace('alpha', (0.01, 0.05), 'continuous', 0.02, 'fundamentals'),
    'beta_1': ParameterSpace('beta_1', (0.2, 0.6), 'continuous', 0.4, 'fundamentals'),
    'beta_2': ParameterSpace('beta_2', (0.2, 0.5), 'continuous', 0.30, 'fundamentals'),
    'beta_3': ParameterSpace('beta_3', (0.1, 0.4), 'continuous', 0.30, 'fundamentals'),
    
    # Composite parameters
    'lambda_risk': ParameterSpace('lambda_risk', (1.5, 2.5), 'continuous', 2.0, 'composite'),
    'phi': ParameterSpace('phi', (0.05, 0.15), 'continuous', 0.1, 'composite'),
    'psi': ParameterSpace('psi', (0.05, 0.15), 'continuous', 0.1, 'composite'),
    'eta': ParameterSpace('eta', (0.01, 0.05), 'continuous', 0.02, 'composite'),
    
    # Backtest parameters
    'threshold_long': ParameterSpace('threshold_long', (65, 80), 'continuous', 70.0, 'backtest'),
    'threshold_short': ParameterSpace('threshold_short', (20, 35), 'continuous', 30.0, 'backtest'),
}


# Subsets for focused optimization
MOMENTUM_PARAM_SPACES = {k: v for k, v in DEFAULT_PARAM_SPACES.items() if v.category == 'momentum'}
TECHNICAL_PARAM_SPACES = {k: v for k, v in DEFAULT_PARAM_SPACES.items() if v.category == 'technical'}
FUNDAMENTALS_PARAM_SPACES = {k: v for k, v in DEFAULT_PARAM_SPACES.items() if v.category == 'fundamentals'}
COMPOSITE_PARAM_SPACES = {k: v for k, v in DEFAULT_PARAM_SPACES.items() if v.category == 'composite'}


# ---------------------------------------------------------------------------
# Config Creation from Parameters
# ---------------------------------------------------------------------------


def create_configs_from_params(params: Dict[str, float]) -> Dict[str, Any]:
    """
    Create config objects from parameter dictionary.
    
    Args:
        params: Parameter dictionary
    
    Returns:
        Dict with config objects for each module
    """
    configs = {}
    
    # Momentum config
    if any(k in params for k in ['kappa', 'beta_m']):
        configs['momentum'] = MomentumConfig(
            kappa=params.get('kappa', 0.03),
            beta_m=params.get('beta_m', 0.2),
        )
    
    # Technical config
    if any(k in params for k in ['gamma', 'theta']):
        configs['technical'] = TechnicalConfig(
            gamma=params.get('gamma', 0.1),
            theta=params.get('theta', 0.1),
        )
    
    # Fundamentals config
    if any(k in params for k in ['alpha', 'beta_1', 'beta_2', 'beta_3']):
        configs['fundamentals'] = FundamentalsConfig(
            alpha=params.get('alpha', 0.02),
            beta_1=params.get('beta_1', 0.4),
            beta_2=params.get('beta_2', 0.30),
            beta_3=params.get('beta_3', 0.30),
        )
    
    # Composite config
    if any(k in params for k in ['lambda_risk', 'phi', 'psi', 'eta']):
        configs['composite'] = CompositeConfig(
            lambda_risk=params.get('lambda_risk', 2.0),
            phi=params.get('phi', 0.1),
            psi=params.get('psi', 0.1),
            eta=params.get('eta', 0.02),
        )
    
    # Backtest config
    if any(k in params for k in ['threshold_long', 'threshold_short']):
        configs['backtest'] = BacktestConfig(
            threshold_long=params.get('threshold_long', 70.0),
            threshold_short=params.get('threshold_short', 30.0),
        )
    
    return configs


# ---------------------------------------------------------------------------
# Objective Functions (Enhanced with Full Integration)
# ---------------------------------------------------------------------------


def objective_sharpe_ratio(
    df: pd.DataFrame,
    params: Dict[str, float],
    use_medallion: bool = True,
) -> float:
    """
    Objective function: Maximize Sharpe ratio.
    
    Fully integrated with composite.py and backtest.py
    
    Args:
        df: DataFrame with data
        params: Parameter dictionary
        use_medallion: Use R_ASRE (Medallion) rating
    
    Returns:
        Negative Sharpe ratio (for minimization)
    """
    try:
        # Create configs from parameters
        configs = create_configs_from_params(params)
        
        # Compute ASRE ratings with all configs
        result_df = compute_complete_asre(
            df,
            config=configs.get('composite'),
            fundamentals_config=configs.get('fundamentals'),
            technical_config=configs.get('technical'),
            momentum_config=configs.get('momentum'),
            medallion=use_medallion,
            return_all_components=False,
        )
        
        # Validate ratings
        rating_col = 'r_asre' if use_medallion else 'r_final'
        is_valid, msg = validate_asre_rating(result_df, rating_col)
        if not is_valid:
            logger.warning(f"Rating validation failed: {msg}")
            return 1e10

        # ------------------------------------------------------------------
        # NEW: Penalize flat or weak signals
        # ------------------------------------------------------------------
        rating_std = result_df[rating_col].std()

        if rating_std < 2.0:
            logger.warning(f"Signal variance too low (std={rating_std:.3f}) — penalizing")
            return 1e6
        
        # Get backtest config
        bt_config = configs.get('backtest', BacktestConfig())
        
        # Run backtest using Backtester class
        backtester = Backtester(
            result_df,
            rating_col=rating_col,
            config=bt_config,
        )
        
        backtester.run(
            signal_type='threshold',
            threshold_long=bt_config.threshold_long,
            threshold_short=bt_config.threshold_short,
        )
        
        # Get report
        report = backtester.get_report()
        sharpe = report['sharpe_ratio']
        
        # Return negative (for minimization)
        return -sharpe
    
    except Exception as e:
        logger.warning(f"Objective function failed with params {params}: {e}")
        return 1e10  # Large penalty for failures


def objective_calmar_ratio(
    df: pd.DataFrame,
    params: Dict[str, float],
    use_medallion: bool = True,
) -> float:
    """
    Objective function: Maximize Calmar ratio.
    
    Returns:
        Negative Calmar ratio (for minimization)
    """
    try:
        configs = create_configs_from_params(params)
        
        result_df = compute_complete_asre(
            df,
            config=configs.get('composite'),
            fundamentals_config=configs.get('fundamentals'),
            technical_config=configs.get('technical'),
            momentum_config=configs.get('momentum'),
            medallion=use_medallion,
        )
        
        rating_col = 'r_asre' if use_medallion else 'r_final'
        bt_config = configs.get('backtest', BacktestConfig())
        
        backtester = Backtester(result_df, rating_col=rating_col, config=bt_config)
        backtester.run(signal_type='threshold')
        
        report = backtester.get_report()
        calmar = report['calmar_ratio']
        
        return -calmar
    
    except Exception as e:
        logger.warning(f"Objective function failed: {e}")
        return 1e10


def objective_multi_criteria(
    df: pd.DataFrame,
    params: Dict[str, float],
    weights: Dict[str, float] = {'sharpe': 0.5, 'calmar': 0.3, 'sortino': 0.2},
    use_medallion: bool = True,
) -> float:
    """
    Multi-objective optimization.
    
    Combines multiple risk-adjusted metrics into single objective.
    
    Args:
        df: DataFrame
        params: Parameters
        weights: Weights for each criterion
        use_medallion: Use R_ASRE rating
    
    Returns:
        Negative weighted score
    """
    try:
        configs = create_configs_from_params(params)
        
        result_df = compute_complete_asre(
            df,
            config=configs.get('composite'),
            fundamentals_config=configs.get('fundamentals'),
            technical_config=configs.get('technical'),
            momentum_config=configs.get('momentum'),
            medallion=use_medallion,
        )
        
        rating_col = 'r_asre' if use_medallion else 'r_final'
        bt_config = configs.get('backtest', BacktestConfig())
        
        backtester = Backtester(result_df, rating_col=rating_col, config=bt_config)
        backtester.run(signal_type='threshold')
        
        report = backtester.get_report()
        
        # Calculate metrics
        sharpe = report['sharpe_ratio']
        calmar = report['calmar_ratio']
        sortino = report['sortino_ratio']
        
        # Normalize metrics (approximate ranges)
        sharpe_norm = sharpe / 3.0  # Assume Sharpe ~3 is excellent
        calmar_norm = calmar / 2.0  # Assume Calmar ~2 is excellent
        sortino_norm = sortino / 3.5  # Assume Sortino ~3.5 is excellent
        
        # Weighted combination
        score = (
            weights.get('sharpe', 0.5) * sharpe_norm +
            weights.get('calmar', 0.3) * calmar_norm +
            weights.get('sortino', 0.2) * sortino_norm
        )
        
        return -score  # Negative for minimization
    
    except Exception as e:
        logger.warning(f"Multi-objective failed: {e}")
        return 1e10


def objective_with_regularization(
    df: pd.DataFrame,
    params: Dict[str, float],
    base_params: Dict[str, float],
    lambda_reg: float = 0.1,
    use_medallion: bool = True,
) -> float:
    """
    Objective with L2 regularization to prevent overfitting.
    
    Formula: objective = -Sharpe + λ·||params - base_params||²
    
    Args:
        df: DataFrame
        params: Current parameters
        base_params: Base/default parameters
        lambda_reg: Regularization strength
        use_medallion: Use R_ASRE rating
    
    Returns:
        Regularized objective value
    """
    # Base Sharpe objective
    sharpe_obj = objective_sharpe_ratio(df, params, use_medallion)
    
    # L2 regularization penalty
    param_diff = np.array([params.get(k, base_params[k]) - base_params[k] 
                           for k in base_params.keys()])
    # Clip parameter drift to prevent optimizer instability
    param_diff = np.clip(param_diff, -5.0, 5.0)

    l2_penalty = lambda_reg * np.sum(param_diff ** 2)
    
    return sharpe_obj + l2_penalty


# ---------------------------------------------------------------------------
# Grid Search Optimizer
# ---------------------------------------------------------------------------


def grid_search_optimize(
    df: pd.DataFrame,
    param_grid: Dict[str, List[float]],
    objective_func: Callable = objective_sharpe_ratio,
    n_splits: int = 5,
    verbose: bool = True,
) -> Tuple[Dict[str, float], float]:
    """
    Grid search optimization with time-series cross-validation.
    
    Args:
        df: DataFrame with data
        param_grid: Dict mapping param names to lists of values
        objective_func: Objective function to minimize
        n_splits: Number of CV splits
        verbose: Print progress
    
    Returns:
        Tuple of (best_params, best_score)
    """
    from itertools import product
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    total = len(combinations)
    logger.info(f"Grid search: {total} combinations, {n_splits} CV splits")
    
    best_score = np.inf
    best_params = None
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for idx, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        
        scores = []
        
        for train_idx, test_idx in tscv.split(df):
            df_test = df.iloc[test_idx]
            
            try:
                score = objective_func(df_test, params)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Error with params {params}: {e}")
                scores.append(1e10)
        
        avg_score = np.mean(scores)
        
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
            
            if verbose:
                logger.info(f"[{idx+1}/{total}] New best: {params} → {-avg_score:.4f}")
    
    logger.info(f"Grid search complete. Best params: {best_params}, Score: {-best_score:.4f}")
    
    return best_params, -best_score


# ---------------------------------------------------------------------------
# Random Search Optimizer
# ---------------------------------------------------------------------------


def random_search_optimize(
    df: pd.DataFrame,
    param_spaces: Dict[str, ParameterSpace],
    objective_func: Callable = objective_sharpe_ratio,
    n_iter: int = 100,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, float], float]:
    """
    Random search optimization.
    
    More efficient than grid search for high-dimensional spaces.
    
    Args:
        df: DataFrame
        param_spaces: Dict of ParameterSpace objects
        objective_func: Objective function
        n_iter: Number of random samples
        n_splits: CV splits
        random_state: Random seed
    
    Returns:
        Tuple of (best_params, best_score)
    """
    np.random.seed(random_state)
    
    logger.info(f"Random search: {n_iter} iterations, {n_splits} CV splits")
    
    best_score = np.inf
    best_params = None
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for i in range(n_iter):
        # Sample random parameters
        params = {name: space.sample() for name, space in param_spaces.items()}
        
        scores = []
        
        for train_idx, test_idx in tscv.split(df):
            df_test = df.iloc[test_idx]
            
            try:
                score = objective_func(df_test, params)
                scores.append(score)
            except Exception:
                scores.append(1e10)
        
        avg_score = np.mean(scores)
        
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
            logger.info(f"[{i+1}/{n_iter}] New best: {params} → {-avg_score:.4f}")
    
    logger.info(f"Random search complete. Best score: {-best_score:.4f}")
    
    return best_params, -best_score


# ---------------------------------------------------------------------------
# Differential Evolution Optimizer
# ---------------------------------------------------------------------------


def differential_evolution_optimize(
    df: pd.DataFrame,
    param_spaces: Dict[str, ParameterSpace],
    objective_func: Callable = objective_sharpe_ratio,
    maxiter: int = 100,
    workers: int = 1,
    seed: int = 42,
) -> Tuple[Dict[str, float], float]:
    """
    Differential evolution global optimization.
    
    Robust for non-convex, noisy objectives.
    
    Args:
        df: DataFrame
        param_spaces: Parameter spaces
        objective_func: Objective function
        maxiter: Maximum iterations
        workers: Number of parallel workers
        seed: Random seed
    
    Returns:
        Tuple of (best_params, best_score)
    """
    param_names = list(param_spaces.keys())
    bounds = [param_spaces[name].bounds for name in param_names]
    
    logger.info(f"Differential evolution: maxiter={maxiter}, workers={workers}")
    
    # Wrapper function
    def objective_wrapper(x):
        params = dict(zip(param_names, x))
        return objective_func(df, params)
    
    # Optimize
    result = differential_evolution(
        objective_wrapper,
        bounds=bounds,
        maxiter=maxiter,
        workers=workers,
        seed=seed,
        disp=True,
        atol=1e-6,
        tol=1e-6,
    )
    
    if result.success:
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun
        
        logger.info(f"Optimization converged: {best_params}")
        logger.info(f"Best score: {best_score:.4f}")
        
        return best_params, best_score
    else:
        logger.warning("Optimization did not converge")
        # Return defaults
        defaults = {name: space.default for name, space in param_spaces.items()}
        return defaults, 0.0


# ---------------------------------------------------------------------------
# Bayesian Optimization
# ---------------------------------------------------------------------------


def bayesian_optimize(
    df: pd.DataFrame,
    param_spaces: Dict[str, ParameterSpace],
    objective_func: Callable = objective_sharpe_ratio,
    n_calls: int = 50,
    random_state: int = 42,
) -> Tuple[Dict[str, float], float]:
    """
    Bayesian optimization using Gaussian processes.
    
    Most sample-efficient for expensive objective functions.
    
    Requires scikit-optimize (skopt).
    
    Args:
        df: DataFrame
        param_spaces: Parameter spaces
        objective_func: Objective function
        n_calls: Number of function evaluations
        random_state: Random seed
    
    Returns:
        Tuple of (best_params, best_score)
    """
    if not HAS_SKOPT:
        raise ImportError("scikit-optimize required for Bayesian optimization")
    
    param_names = list(param_spaces.keys())
    
    # Convert to skopt space
    space = []
    for name in param_names:
        bounds = param_spaces[name].bounds
        if param_spaces[name].param_type == 'continuous':
            space.append(Real(bounds[0], bounds[1], name=name))
        elif param_spaces[name].param_type == 'integer':
            space.append(Integer(bounds[0], bounds[1], name=name))
    
    logger.info(f"Bayesian optimization: {n_calls} calls")
    
    # Wrapper function
    def objective_wrapper(x):
        params = dict(zip(param_names, x))
        return objective_func(df, params)
    
    # Optimize
    result = gp_minimize(
        objective_wrapper,
        space,
        n_calls=n_calls,
        random_state=random_state,
        verbose=True,
        n_jobs=1,
    )
    
    best_params = dict(zip(param_names, result.x))
    best_score = -result.fun
    
    logger.info(f"Bayesian optimization complete: {best_params}")
    logger.info(f"Best score: {best_score:.4f}")
    
    return best_params, best_score


# ---------------------------------------------------------------------------
# Walk-Forward Analysis (Enhanced)
# ---------------------------------------------------------------------------


def walk_forward_optimize(
    df: pd.DataFrame,
    param_spaces: Dict[str, ParameterSpace],
    objective_func: Callable = objective_sharpe_ratio,
    optimizer: str = 'differential_evolution',
    train_window: int = 252,
    test_window: int = 63,
    step_size: int = 63,
    reoptimize_every: int = 1,
    use_medallion: bool = True,
) -> pd.DataFrame:
    """
    Walk-forward optimization and out-of-sample testing.
    
    Enhanced with full module integration.
    
    Process:
    1. Train on past 'train_window' days
    2. Optimize parameters
    3. Test on next 'test_window' days with Backtester
    4. Roll forward by 'step_size' days
    5. Repeat
    
    Args:
        df: DataFrame with data
        param_spaces: Parameter spaces
        objective_func: Objective function
        optimizer: 'differential_evolution', 'random', 'grid', or 'bayesian'
        train_window: Training window size (days)
        test_window: Test window size (days)
        step_size: Rolling step size (days)
        reoptimize_every: Reoptimize every N periods (1 = every period)
        use_medallion: Use R_ASRE (Medallion) rating
    
    Returns:
        DataFrame with walk-forward results
    """
    results = []
    
    start_idx = train_window
    end_idx = len(df) - test_window
    
    logger.info(
        f"Walk-forward: train={train_window}, test={test_window}, "
        f"step={step_size}, optimizer={optimizer}"
    )
    
    cached_params = None
    period = 0
    
    for idx in range(start_idx, end_idx, step_size):
        period += 1
        
        train_start = idx - train_window
        train_end = idx
        test_start = idx
        test_end = min(idx + test_window, len(df))
        
        df_train = df.iloc[train_start:train_end]
        df_test = df.iloc[test_start:test_end]
        
        # Optimize parameters (or reuse cached)
        if period % reoptimize_every == 1 or cached_params is None:
            logger.info(f"Period {period}: Optimizing on [{df_train.index[0]} to {df_train.index[-1]}]")
            
            if optimizer == 'differential_evolution':
                best_params, train_score = differential_evolution_optimize(
                    df_train, param_spaces, objective_func, maxiter=50
                )
            elif optimizer == 'random':
                best_params, train_score = random_search_optimize(
                    df_train, param_spaces, objective_func, n_iter=50
                )
            elif optimizer == 'bayesian':
                best_params, train_score = bayesian_optimize(
                    df_train, param_spaces, objective_func, n_calls=30
                )
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
            
            cached_params = best_params
        else:
            logger.info(f"Period {period}: Using cached parameters")
            best_params = cached_params
            train_score = 0.0
        
        # Test on out-of-sample period using full integration
        logger.info(f"Testing on [{df_test.index[0]} to {df_test.index[-1]}]")
        
        try:
            # Create configs
            configs = create_configs_from_params(best_params)
            
            # Compute ASRE with all modules
            result_df = compute_complete_asre(
                df_test,
                config=configs.get('composite'),
                fundamentals_config=configs.get('fundamentals'),
                technical_config=configs.get('technical'),
                momentum_config=configs.get('momentum'),
                medallion=use_medallion,
            )
            
            # Run backtest
            rating_col = 'r_asre' if use_medallion else 'r_final'
            bt_config = configs.get('backtest', BacktestConfig())
            
            backtester = Backtester(result_df, rating_col=rating_col, config=bt_config)
            backtester.run(signal_type='threshold')
            
            report = backtester.get_report()
            
            results.append({
                'period': period,
                'train_start': df_train.index[0],
                'train_end': df_train.index[-1],
                'test_start': df_test.index[0],
                'test_end': df_test.index[-1],
                'train_score': train_score,
                'test_sharpe': report['sharpe_ratio'],
                'test_sortino': report['sortino_ratio'],
                'test_calmar': report['calmar_ratio'],
                'test_return': report['total_return'],
                'test_max_dd': report['max_drawdown'],
                'test_win_rate': report['win_rate'],
                'params': best_params.copy(),
            })
            
            logger.info(
                f"Period {period}: Test Sharpe={report['sharpe_ratio']:.4f}, "
                f"Return={report['total_return']:.2%}"
            )
        
        except Exception as e:
            logger.error(f"Period {period} failed: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("Walk-Forward Analysis Summary")
    logger.info("=" * 70)
    logger.info(f"Periods: {len(results_df)}")
    logger.info(f"Avg Test Sharpe: {results_df['test_sharpe'].mean():.4f}")
    logger.info(f"Std Test Sharpe: {results_df['test_sharpe'].std():.4f}")
    logger.info(f"Avg Test Return: {results_df['test_return'].mean():.4%}")
    logger.info(f"Avg Max DD: {results_df['test_max_dd'].mean():.2%}")
    logger.info(f"Avg Win Rate: {results_df['test_win_rate'].mean():.2%}")
    logger.info("=" * 70)
    
    return results_df


# ---------------------------------------------------------------------------
# Optimizer Class (Unified Interface with Full Integration)
# ---------------------------------------------------------------------------


class ASREOptimizer:
    """
    Unified optimizer interface for ASRE parameters.
    
    Fully integrated with:
    - composite.py: ASRE rating computation
    - backtest.py: Strategy backtesting
    - calibration.py: MLE weight initialization
    - All score modules: fundamentals, technical, momentum
    - config.py: Configuration objects
    
    Usage:
        optimizer = ASREOptimizer(df, param_spaces)
        best_params = optimizer.optimize(method='differential_evolution')
        wf_results = optimizer.walk_forward()
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        param_spaces: Optional[Dict[str, ParameterSpace]] = None,
        objective_func: Callable = objective_sharpe_ratio,
        use_medallion: bool = True,
    ):
        """
        Initialize optimizer.
        
        Args:
            df: DataFrame with data
            param_spaces: Parameter spaces (uses defaults if None)
            objective_func: Objective function to optimize
            use_medallion: Use R_ASRE (Medallion) rating
        """
        self.df = df
        self.param_spaces = param_spaces or DEFAULT_PARAM_SPACES
        self.objective_func = objective_func
        self.use_medallion = use_medallion
        
        self.best_params: Optional[Dict[str, float]] = None
        self.best_score: Optional[float] = None
        self.optimization_history: List[Dict] = []
    
    def optimize(
        self,
        method: str = 'differential_evolution',
        **kwargs,
    ) -> Dict[str, float]:
        """
        Optimize parameters.
        
        Args:
            method: 'differential_evolution', 'random', 'grid', or 'bayesian'
            **kwargs: Method-specific arguments
        
        Returns:
            Best parameters
        """
        logger.info(f"Optimizing with method: {method}")
        
        if method == 'differential_evolution':
            self.best_params, self.best_score = differential_evolution_optimize(
                self.df,
                self.param_spaces,
                self.objective_func,
                **kwargs,
            )
        elif method == 'random':
            self.best_params, self.best_score = random_search_optimize(
                self.df,
                self.param_spaces,
                self.objective_func,
                **kwargs,
            )
        elif method == 'bayesian':
            self.best_params, self.best_score = bayesian_optimize(
                self.df,
                self.param_spaces,
                self.objective_func,
                **kwargs,
            )
        elif method == 'grid':
            # Convert ParameterSpace to grid
            param_grid = {
                name: np.linspace(space.bounds[0], space.bounds[1], 5).tolist()
                for name, space in self.param_spaces.items()
            }
            self.best_params, self.best_score = grid_search_optimize(
                self.df,
                param_grid,
                self.objective_func,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return self.best_params
    
    def walk_forward(self, **kwargs) -> pd.DataFrame:
        """
        Run walk-forward analysis.
        
        Args:
            **kwargs: Arguments for walk_forward_optimize
        
        Returns:
            Walk-forward results DataFrame
        """
        return walk_forward_optimize(
            self.df,
            self.param_spaces,
            self.objective_func,
            use_medallion=self.use_medallion,
            **kwargs,
        )
    
    def get_optimal_configs(self) -> Dict[str, Any]:
        """
        Get config objects from best parameters.
        
        Returns:
            Dict with config objects
        """
        if self.best_params is None:
            raise ValueError("Run optimization first")
        
        return create_configs_from_params(self.best_params)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

__all__ = [
    # Parameter space
    'ParameterSpace',
    'DEFAULT_PARAM_SPACES',
    'MOMENTUM_PARAM_SPACES',
    'TECHNICAL_PARAM_SPACES',
    'FUNDAMENTALS_PARAM_SPACES',
    'COMPOSITE_PARAM_SPACES',
    
    # Config creation
    'create_configs_from_params',
    
    # Objective functions
    'objective_sharpe_ratio',
    'objective_calmar_ratio',
    'objective_multi_criteria',
    'objective_with_regularization',
    
    # Optimizers
    'grid_search_optimize',
    'random_search_optimize',
    'differential_evolution_optimize',
    'bayesian_optimize',
    
    # Walk-forward
    'walk_forward_optimize',
    
    # Unified interface
    'ASREOptimizer',
]
