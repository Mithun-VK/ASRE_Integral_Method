"""
asre/theory/overfitting_guard.py — ASRE Harvey-Liu-Zhu Overfitting Guard (v1.0)

Purpose
-------
Prevents ASRE from using factors or composite weights that are statistically
indistinguishable from noise. Every factor return series produced by
backtest.py must pass this guard before its results are used in weight
optimisation or PDF reporting.

The Single Biggest Failure Mode of Data-Driven Scoring Models
--------------------------------------------------------------
When a model is tested on the same data used to select its factors, the
resulting t-statistics are inflated by multiple testing. A factor that
appears significant at t=2.0 when tested once has only a ~5% false
discovery rate. But after testing 300 factors on the same dataset (as is
common in quantitative finance), the expected number of false discoveries
at t=2.0 is ~15 — meaning most "significant" factors are noise.

Harvey, Liu & Zhu (2016) solution
----------------------------------
Adjust the significance threshold based on the number of factors tested
and the historical publication rate of spurious factors. For new factors
(post-2000), the adjusted threshold is t > 3.0. For established factors
with decades of out-of-sample replication (Jegadeesh-Titman, Fama-French),
the original threshold t > 2.0 is retained.

ASRE application
----------------
ASRE's three factors (F, T, M) are not newly discovered — they are proxies
for published, replicated anomalies. However, the SPECIFIC WEIGHTS and the
COMPOSITE SCORE formula are ASRE-specific and require validation on Indian
market data. This guard checks:

  1. Each factor's return series independently (is this factor earning
     excess returns in India?).
  2. The composite score's return series (does the weighted combination
     add value beyond individual factors?).
  3. The walk-forward optimizer's output weights (do empirical weights
     outperform the theoretical prior at the required t-stat?).

Integration points
------------------
  backtest.py:
      from asre.theory.overfitting_guard import OverfittingGuard
      guard = OverfittingGuard(risk_free_rate=0.065)
      result = guard.check_factor(name="m_score", returns=m_returns)
      if not result.passes:
          logger.warning("M-Score factor return does not meet HLZ threshold")

  composite.py (weight optimizer):
      from asre.theory.overfitting_guard import OverfittingGuard
      guard = OverfittingGuard()
      ok = guard.check_weight_improvement(
          prior_returns=theoretical_prior_returns,
          optimized_returns=empirical_weight_returns,
      )
      if not ok:
          logger.info("Empirical weights do not beat theoretical prior — using prior")

References
----------
Harvey, C., Liu, Y., Zhu, H. (2016). ...and the cross-section of expected
  returns. Review of Financial Studies, 29(1), 5-68.
  → t > 3.0 required for new factors after multiple-testing correction.

Benjamini, Y., Hochberg, Y. (1995). Controlling the false discovery rate.
  Journal of the Royal Statistical Society B, 57(1), 289-300.
  → FDR correction used alongside HLZ for composite score testing.

White, H. (2000). A reality check for data snooping.
  Econometrica, 68(5), 1097-1126.
  → Bootstrap-based test for data snooping in trading rules.

Lo, A., MacKinlay, C. (1990). Data-snooping biases in tests of financial
  asset pricing models. Review of Financial Studies, 3(3), 431-467.
  → Foundational paper on in-sample bias in factor testing.

Jegadeesh, N., Titman, S. (1993). Returns to buying winners and selling losers.
  Journal of Finance, 48(1), 65-91.

India 10-year G-Sec yield reference:
  RBI (2025): Benchmark yield ~6.5% pa as of Q4 2025.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

OVERFITTING_GUARD_VERSION: str = "1.0"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Harvey, Liu & Zhu (2016) t-stat thresholds
TSTAT_HURDLE_NEW:         float = 3.0    # Post-2000 / India-specific factors
TSTAT_HURDLE_ESTABLISHED: float = 2.0    # Pre-2000 replicated factors
TSTAT_HURDLE_INDIA:       float = 3.5    # India-specific signals (higher bar)

# Minimum sample sizes — below these, any t-stat is unreliable
MIN_OBSERVATIONS_ANNUAL:  int   = 252    # 1 trading year
MIN_OBSERVATIONS_ROBUST:  int   = 504    # 2 trading years (preferred)
MIN_OBSERVATIONS_IDEAL:   int   = 1260   # 5 trading years (SEBI retention period)

# Default risk-free rate — India 10Y G-Sec (RBI benchmark, Q4 2025)
DEFAULT_RISK_FREE_RATE_ANNUAL: float = 0.065   # 6.5% pa
DEFAULT_RISK_FREE_RATE_DAILY:  float = DEFAULT_RISK_FREE_RATE_ANNUAL / 252

# Annualisation factor (trading days)
TRADING_DAYS_PER_YEAR: int = 252

# Minimum excess return to be considered economically significant
# (avoids statistically significant but economically trivial factors)
MIN_ECONOMIC_SIGNIFICANCE_ANNUAL: float = 0.020   # 2.0% pa

# Factor classification for HLZ threshold selection
_ESTABLISHED_FACTORS = frozenset({
    "f_score", "m_score", "momentum", "value", "hml", "rmw", "cma",
    "wml", "piotroski", "reversal",
})
_NEW_FACTORS = frozenset({
    "t_score", "technical", "vix_adjusted", "dip_score", "composite",
})


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FactorSignificanceResult:
    """
    Complete result from a single factor significance test.

    Attributes
    ----------
    factor_name       : Identifier string for the factor tested.
    n_observations    : Number of return observations used.
    mean_excess_return: Annualised mean excess return (daily rf subtracted).
    std_excess_return : Annualised standard deviation of excess returns.
    sharpe_ratio      : Annualised Sharpe ratio (mean / std).
    t_stat            : t-statistic of the mean excess return.
    p_value           : Two-tailed p-value (approximate, t-distribution).
    hlz_hurdle        : HLZ t-stat threshold applied.
    passes_hlz        : True if t_stat > hlz_hurdle.
    passes_economic   : True if |annualised excess return| > 2.0% pa.
    passes            : True only if BOTH hlz and economic tests pass.
    sample_adequacy   : 'ADEQUATE', 'MARGINAL', or 'INSUFFICIENT'.
    verdict           : Human-readable plain-English verdict.
    reference         : Citation for the threshold applied.
    warnings          : List of any non-fatal warnings.
    """
    factor_name         : str
    n_observations      : int
    mean_excess_return  : float     # annualised
    std_excess_return   : float     # annualised
    sharpe_ratio        : float
    t_stat              : float
    p_value             : float
    hlz_hurdle          : float
    passes_hlz          : bool
    passes_economic     : bool
    passes              : bool
    sample_adequacy     : str
    verdict             : str
    reference           : str
    warnings            : List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialise to dict for JSONL audit log."""
        return {
            "factor"              : self.factor_name,
            "n"                   : self.n_observations,
            "mean_excess_pa"      : round(self.mean_excess_return, 6),
            "std_excess_pa"       : round(self.std_excess_return, 6),
            "sharpe"              : round(self.sharpe_ratio, 4),
            "t_stat"              : round(self.t_stat, 4),
            "p_value"             : round(self.p_value, 6),
            "hlz_hurdle"          : self.hlz_hurdle,
            "passes_hlz"          : self.passes_hlz,
            "passes_economic"     : self.passes_economic,
            "passes"              : self.passes,
            "sample_adequacy"     : self.sample_adequacy,
            "verdict"             : self.verdict,
            "warnings"            : self.warnings,
            "guard_version"       : OVERFITTING_GUARD_VERSION,
        }

    def log_summary(self) -> None:
        """Log a concise one-line summary at the appropriate level."""
        level = logging.INFO if self.passes else logging.WARNING
        logger.log(
            level,
            "overfitting_guard [%s]: t=%.3f (hurdle=%.1f) | "
            "Sharpe=%.3f | excess=%.1f%% pa | %s | %s",
            self.factor_name,
            self.t_stat,
            self.hlz_hurdle,
            self.sharpe_ratio,
            self.mean_excess_return * 100,
            self.sample_adequacy,
            "PASS" if self.passes else "FAIL",
        )


@dataclass
class WeightImprovementResult:
    """
    Result from testing whether empirical weights beat the theoretical prior.
    Used in composite.py to decide whether to use optimized vs prior weights.
    """
    prior_sharpe      : float
    optimized_sharpe  : float
    improvement       : float      # optimized_sharpe - prior_sharpe
    t_stat_improvement: float      # paired t-test on return difference
    passes            : bool       # True = use optimized weights
    verdict           : str
    recommended_weights: Dict[str, float]  # which weights to use


# ---------------------------------------------------------------------------
# OverfittingGuard
# ---------------------------------------------------------------------------

class OverfittingGuard:
    """
    Applies Harvey-Liu-Zhu (2016) multiple-testing correction to ASRE
    factor return series and composite score series from backtest.py.

    Parameters
    ----------
    risk_free_rate : Annual risk-free rate (default: India 10Y G-Sec 6.5%).
                     Converted internally to daily rate for computation.
    trading_days   : Trading days per year (default: 252).

    Usage
    -----
        guard = OverfittingGuard(risk_free_rate=0.065)

        # Test a single factor
        result = guard.check_factor("m_score", m_score_returns)
        result.log_summary()
        if not result.passes:
            logger.warning("M-Score does not meet significance threshold")

        # Test the composite score
        composite_result = guard.check_factor("composite", composite_returns)

        # Test whether empirical weights beat theoretical prior
        weight_result = guard.check_weight_improvement(
            prior_returns=prior_weighted_returns,
            optimized_returns=empirical_weighted_returns,
            theoretical_prior=THEORETICAL_WEIGHT_PRIOR,
            empirical_weights=optimizer_output_weights,
        )

        # Full backtest report for audit log
        report = guard.full_report([
            ("f_score", f_returns),
            ("t_score", t_returns),
            ("m_score", m_returns),
            ("composite", composite_returns),
        ])
    """

    def __init__(
        self,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE_ANNUAL,
        trading_days:   int   = TRADING_DAYS_PER_YEAR,
    ) -> None:
        self.rf_annual    = risk_free_rate
        self.rf_daily     = risk_free_rate / trading_days
        self.trading_days = trading_days

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def check_factor(
        self,
        factor_name: str,
        returns:     "np.ndarray | list",
        hurdle:      Optional[float] = None,
    ) -> FactorSignificanceResult:
        """
        Test whether a factor's return series is statistically significant
        under the Harvey-Liu-Zhu (2016) multiple-testing correction.

        Parameters
        ----------
        factor_name : Factor identifier ('f_score', 't_score', 'm_score',
                      'composite', or any string).
        returns     : Daily return series for the factor-sorted long-short
                      portfolio, OR the composite score return series.
                      Must be array-like of floats. NaN values are dropped.
        hurdle      : Override the HLZ t-stat threshold (optional).
                      If None, threshold is selected automatically based
                      on factor_name.

        Returns
        -------
        FactorSignificanceResult
        """
        returns_arr = self._prepare_returns(returns)
        n           = len(returns_arr)
        warnings_   = []

        # Sample adequacy
        adequacy = self._sample_adequacy(n, warnings_)

        # Excess returns (subtract daily risk-free)
        excess_daily = returns_arr - self.rf_daily

        # Annualised statistics
        mean_daily  = float(np.mean(excess_daily))
        std_daily   = float(np.std(excess_daily, ddof=1))
        mean_annual = mean_daily * self.trading_days
        std_annual  = std_daily  * math.sqrt(self.trading_days)
        sharpe      = (mean_annual / std_annual) if std_annual > 1e-10 else 0.0

        # t-statistic: t = mean / (std / sqrt(n))
        t_stat = (mean_daily / (std_daily / math.sqrt(n))) if std_daily > 1e-10 else 0.0

        # Two-tailed p-value (approximation via normal CDF for large n)
        p_value = self._p_value(t_stat, n)

        # HLZ threshold
        if hurdle is not None:
            hlz = hurdle
            ref = "User-specified threshold"
        else:
            hlz, ref = self._select_hurdle(factor_name)

        passes_hlz      = abs(t_stat) > hlz
        passes_economic = abs(mean_annual) > MIN_ECONOMIC_SIGNIFICANCE_ANNUAL

        # Warn if sign is negative (factor earns negative excess return)
        if mean_annual < -MIN_ECONOMIC_SIGNIFICANCE_ANNUAL:
            warnings_.append(
                f"Factor earns NEGATIVE excess return ({mean_annual*100:.1f}% pa). "
                "Consider removing or inverting this factor."
            )

        # Warn if statistically significant but economically trivial
        if passes_hlz and not passes_economic:
            warnings_.append(
                f"Factor is statistically significant (t={t_stat:.2f}) but "
                f"economically trivial ({mean_annual*100:.2f}% pa < 2.0% pa minimum). "
                "Statistical significance alone is insufficient."
            )

        # Warn if sample is insufficient
        if adequacy == "INSUFFICIENT":
            warnings_.append(
                f"Only {n} observations — below minimum {MIN_OBSERVATIONS_ANNUAL}. "
                "t-statistic is unreliable. Do not use for weight optimization."
            )
            passes_hlz      = False
            passes_economic = False

        passes = passes_hlz and passes_economic
        verdict = self._build_verdict(
            factor_name, t_stat, hlz, mean_annual, sharpe, passes, adequacy
        )

        result = FactorSignificanceResult(
            factor_name        = factor_name,
            n_observations     = n,
            mean_excess_return = mean_annual,
            std_excess_return  = std_annual,
            sharpe_ratio       = sharpe,
            t_stat             = t_stat,
            p_value            = p_value,
            hlz_hurdle         = hlz,
            passes_hlz         = passes_hlz,
            passes_economic    = passes_economic,
            passes             = passes,
            sample_adequacy    = adequacy,
            verdict            = verdict,
            reference          = ref,
            warnings           = warnings_,
        )
        result.log_summary()
        return result

    def check_weight_improvement(
        self,
        prior_returns:      "np.ndarray | list",
        optimized_returns:  "np.ndarray | list",
        theoretical_prior:  Optional[Dict[str, float]] = None,
        empirical_weights:  Optional[Dict[str, float]] = None,
    ) -> WeightImprovementResult:
        """
        Test whether empirical walk-forward weights produce statistically
        significant improvement over the theoretical prior weights.

        If the improvement is not significant, composite.py should use
        THEORETICAL_WEIGHT_PRIOR rather than the optimizer's output.

        Parameters
        ----------
        prior_returns     : Daily returns of portfolio using theoretical prior.
        optimized_returns : Daily returns of portfolio using empirical weights.
        theoretical_prior : Dict {'f': 0.30, 't': 0.25, 'm': 0.45}.
        empirical_weights : Dict {'f': w_f, 't': w_t, 'm': w_m} from optimizer.

        Returns
        -------
        WeightImprovementResult
        """
        from asre.theory.factor_registry import THEORETICAL_WEIGHT_PRIOR

        prior_arr = self._prepare_returns(prior_returns)
        opt_arr   = self._prepare_returns(optimized_returns)
        n         = min(len(prior_arr), len(opt_arr))

        if n < MIN_OBSERVATIONS_ANNUAL:
            return WeightImprovementResult(
                prior_sharpe       = 0.0,
                optimized_sharpe   = 0.0,
                improvement        = 0.0,
                t_stat_improvement = 0.0,
                passes             = False,
                verdict            = (
                    f"Insufficient data ({n} obs < {MIN_OBSERVATIONS_ANNUAL} minimum). "
                    "Using theoretical prior weights."
                ),
                recommended_weights = theoretical_prior or THEORETICAL_WEIGHT_PRIOR,
            )

        prior_arr = prior_arr[-n:]
        opt_arr   = opt_arr[-n:]

        prior_sharpe = self._sharpe(prior_arr)
        opt_sharpe   = self._sharpe(opt_arr)
        improvement  = opt_sharpe - prior_sharpe

        # Paired t-test on return differences
        diff     = opt_arr - prior_arr
        t_stat   = self._paired_tstat(diff)
        passes   = abs(t_stat) > TSTAT_HURDLE_NEW and improvement > 0

        if passes:
            recommended = empirical_weights or THEORETICAL_WEIGHT_PRIOR
            verdict = (
                f"Empirical weights outperform theoretical prior: "
                f"Sharpe {prior_sharpe:.3f} → {opt_sharpe:.3f} "
                f"(+{improvement:.3f}), t={t_stat:.3f} > {TSTAT_HURDLE_NEW}. "
                f"Using empirical weights."
            )
        else:
            recommended = theoretical_prior or THEORETICAL_WEIGHT_PRIOR
            if improvement <= 0:
                reason = (
                    f"Empirical weights do not improve Sharpe "
                    f"({prior_sharpe:.3f} → {opt_sharpe:.3f})."
                )
            else:
                reason = (
                    f"Improvement is not statistically significant: "
                    f"t={t_stat:.3f} < required {TSTAT_HURDLE_NEW}."
                )
            verdict = (
                f"{reason} "
                f"Falling back to theoretical prior "
                f"(Fama-French + Carhart). "
                f"Harvey, Liu & Zhu (2016)."
            )

        logger.info(
            "overfitting_guard weight_check: prior_sharpe=%.3f "
            "opt_sharpe=%.3f t=%.3f → %s",
            prior_sharpe, opt_sharpe, t_stat,
            "USE EMPIRICAL" if passes else "USE PRIOR",
        )

        return WeightImprovementResult(
            prior_sharpe        = prior_sharpe,
            optimized_sharpe    = opt_sharpe,
            improvement         = improvement,
            t_stat_improvement  = t_stat,
            passes              = passes,
            verdict             = verdict,
            recommended_weights = recommended,
        )

    def full_report(
        self,
        factor_series: List[Tuple[str, "np.ndarray | list"]],
        apply_fdr:     bool = True,
    ) -> Dict[str, FactorSignificanceResult]:
        """
        Run check_factor() on multiple factors and apply Benjamini-Hochberg
        False Discovery Rate correction across all results.

        Parameters
        ----------
        factor_series : List of (factor_name, return_series) tuples.
        apply_fdr     : If True, apply BH FDR correction to p-values.
                        Some factors that individually pass may fail after
                        correction.

        Returns
        -------
        Dict mapping factor_name → FactorSignificanceResult.

        Usage in backtest.py
        --------------------
            report = guard.full_report([
                ("f_score",   f_returns),
                ("t_score",   t_returns),
                ("m_score",   m_returns),
                ("composite", composite_returns),
            ])
            for name, result in report.items():
                result.log_summary()
                decision_log.factor_significance[name] = result.to_dict()
        """
        results: Dict[str, FactorSignificanceResult] = {}
        for name, series in factor_series:
            results[name] = self.check_factor(name, series)

        if apply_fdr and len(results) > 1:
            results = self._apply_fdr_correction(results)

        return results

    # ------------------------------------------------------------------
    # FDR correction (Benjamini-Hochberg 1995)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_fdr_correction(
        results: Dict[str, FactorSignificanceResult],
        fdr_level: float = 0.05,
    ) -> Dict[str, FactorSignificanceResult]:
        """
        Apply Benjamini-Hochberg (1995) FDR correction across all p-values.
        Factors that pass individual HLZ but fail FDR get a warning added.
        """
        items      = list(results.items())
        n_tests    = len(items)
        p_values   = [r.p_value for _, r in items]
        sorted_idx = sorted(range(n_tests), key=lambda i: p_values[i])

        # BH threshold: p_(k) <= (k / m) * fdr_level
        bh_pass = [False] * n_tests
        for rank, idx in enumerate(sorted_idx, start=1):
            if p_values[idx] <= (rank / n_tests) * fdr_level:
                bh_pass[idx] = True

        for i, (name, result) in enumerate(items):
            if result.passes and not bh_pass[i]:
                result.warnings.append(
                    f"Passes individual HLZ (t={result.t_stat:.3f}) but "
                    f"fails Benjamini-Hochberg FDR correction "
                    f"(p={result.p_value:.4f} across {n_tests} tests). "
                    f"Treat significance as marginal. "
                    f"Benjamini & Hochberg (1995)."
                )
                logger.warning(
                    "overfitting_guard FDR: [%s] marginal after BH correction "
                    "(p=%.4f, n_tests=%d)",
                    name, result.p_value, n_tests,
                )

        return results

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def check_f_score(self, returns: "np.ndarray | list") -> FactorSignificanceResult:
        """Shorthand for check_factor('f_score', returns)."""
        return self.check_factor("f_score", returns)

    def check_t_score(self, returns: "np.ndarray | list") -> FactorSignificanceResult:
        """Shorthand for check_factor('t_score', returns)."""
        return self.check_factor("t_score", returns)

    def check_m_score(self, returns: "np.ndarray | list") -> FactorSignificanceResult:
        """Shorthand for check_factor('m_score', returns)."""
        return self.check_factor("m_score", returns)

    def check_composite(self, returns: "np.ndarray | list") -> FactorSignificanceResult:
        """
        Shorthand for check_factor('composite', returns).
        Uses TSTAT_HURDLE_NEW (3.0) — composite is an ASRE-specific combination.
        """
        return self.check_factor("composite", returns)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_returns(returns: "np.ndarray | list") -> np.ndarray:
        """
        Convert input to a clean 1-D float64 array.
        Drops NaN, Inf, and values outside [-1.0, 1.0] (daily return sanity).
        """
        arr = np.asarray(returns, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        # Sanity: clip extreme daily return values
        n_before = len(arr)
        arr = arr[np.abs(arr) <= 1.0]
        n_dropped = n_before - len(arr)
        if n_dropped > 0:
            logger.warning(
                "overfitting_guard: dropped %d observations with |return| > 100%%.",
                n_dropped,
            )
        return arr

    @staticmethod
    def _sample_adequacy(n: int, warnings_: List[str]) -> str:
        if n >= MIN_OBSERVATIONS_ROBUST:
            return "ADEQUATE"
        if n >= MIN_OBSERVATIONS_ANNUAL:
            warnings_.append(
                f"Only {n} observations (< {MIN_OBSERVATIONS_ROBUST} preferred). "
                "Results are directional only — extend backtest for robust inference."
            )
            return "MARGINAL"
        return "INSUFFICIENT"

    @staticmethod
    def _select_hurdle(factor_name: str) -> Tuple[float, str]:
        """
        Return (hlz_threshold, citation_string) for a given factor name.
        Matches against known established and new factor sets.
        """
        name_lower = factor_name.lower()
        if any(k in name_lower for k in _ESTABLISHED_FACTORS):
            return (
                TSTAT_HURDLE_ESTABLISHED,
                "Harvey, Liu & Zhu (2016): t > 2.0 for established pre-2000 factors",
            )
        if any(k in name_lower for k in _NEW_FACTORS):
            return (
                TSTAT_HURDLE_NEW,
                "Harvey, Liu & Zhu (2016): t > 3.0 for new/composite factors",
            )
        # Default: apply the stricter threshold for unrecognised factors
        return (
            TSTAT_HURDLE_INDIA,
            "Harvey, Liu & Zhu (2016): t > 3.5 for India-specific / unclassified factors",
        )

    def _sharpe(self, returns: np.ndarray) -> float:
        """Annualised Sharpe ratio of a daily return series."""
        excess = returns - self.rf_daily
        mean   = float(np.mean(excess)) * self.trading_days
        std    = float(np.std(excess, ddof=1)) * math.sqrt(self.trading_days)
        return (mean / std) if std > 1e-10 else 0.0

    @staticmethod
    def _paired_tstat(diff: np.ndarray) -> float:
        """t-statistic for a paired difference series."""
        n   = len(diff)
        std = float(np.std(diff, ddof=1))
        if std < 1e-10 or n < 2:
            return 0.0
        return float(np.mean(diff)) / (std / math.sqrt(n))

    @staticmethod
    def _p_value(t_stat: float, n: int) -> float:
        """
        Two-tailed p-value.
        Uses normal approximation for n >= 30 (accurate for finance data).
        Falls back to exact t-distribution for small samples.
        """
        try:
            from scipy.stats import t as t_dist
            return float(2 * t_dist.sf(abs(t_stat), df=n - 1))
        except ImportError:
            pass

        # Normal approximation (Abramowitz & Stegun 26.2.17)
        z = abs(t_stat)
        p = math.erfc(z / math.sqrt(2))
        return float(p)

    @staticmethod
    def _build_verdict(
        factor_name: str,
        t_stat:      float,
        hlz:         float,
        mean_annual: float,
        sharpe:      float,
        passes:      bool,
        adequacy:    str,
    ) -> str:
        """Build the plain-English verdict string."""
        result_word = "passes" if passes else "does not pass"
        sign        = "positive" if mean_annual >= 0 else "negative"
        return (
            f"{factor_name}: {result_word} the Harvey-Liu-Zhu (2016) significance test. "
            f"t-statistic={t_stat:.3f} (threshold={hlz:.1f}). "
            f"Annualised excess return={mean_annual*100:.2f}% ({sign}). "
            f"Sharpe ratio={sharpe:.3f}. "
            f"Sample adequacy: {adequacy}. "
            f"Reference: Harvey, Liu & Zhu (2016), Review of Financial Studies."
        )


# ---------------------------------------------------------------------------
# Module-level constants (importable by backtest.py and composite.py)
# ---------------------------------------------------------------------------

# The two thresholds most commonly imported directly
TSTAT_HURDLE  = TSTAT_HURDLE_ESTABLISHED   # 2.0 — backward-compatible alias
TSTAT_HURDLE3 = TSTAT_HURDLE_NEW           # 3.0 — stricter alias


# ---------------------------------------------------------------------------
# Module-level smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """
    Runs on import. Verifies core logic with synthetic return series.
    Uses only numpy — no backtest data required.
    """
    rng = np.random.default_rng(seed=42)

    # 1. Clearly significant factor: 1% daily mean, low noise
    strong = rng.normal(loc=0.001, scale=0.01, size=504)
    guard  = OverfittingGuard(risk_free_rate=0.065)
    result = guard.check_factor("m_score", strong)
    assert result.passes, (
        f"Smoke test failed: strong factor should pass. t={result.t_stat:.3f}"
    )
    assert result.t_stat > TSTAT_HURDLE_ESTABLISHED

    # 2. Noise factor: zero mean
    noise  = rng.normal(loc=0.0, scale=0.02, size=504)
    result_noise = guard.check_factor("t_score", noise)
    assert not result_noise.passes, \
        "Smoke test failed: noise factor should not pass."

    # 3. Insufficient sample
    short  = rng.normal(loc=0.002, scale=0.01, size=100)
    result_short = guard.check_factor("composite", short)
    assert result_short.sample_adequacy == "INSUFFICIENT"
    assert not result_short.passes

    # 4. Weight improvement: identical returns → should use prior
    prior_r = rng.normal(loc=0.0005, scale=0.015, size=504)
    opt_r   = prior_r + rng.normal(loc=0.0, scale=0.0001, size=504)
    wi = guard.check_weight_improvement(prior_r, opt_r)
    assert not wi.passes, \
        "Trivial improvement should not pass weight improvement test."

    # 5. Full report with FDR
    report = guard.full_report([
        ("f_score",   strong),
        ("t_score",   noise),
        ("m_score",   strong),
        ("composite", strong),
    ])
    assert len(report) == 4
    assert report["f_score"].passes
    assert not report["t_score"].passes

    # 6. to_dict is JSON-serialisable
    import json
    json.dumps(result.to_dict())


_smoke_test()


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "OverfittingGuard",
    "FactorSignificanceResult",
    "WeightImprovementResult",
    "TSTAT_HURDLE",
    "TSTAT_HURDLE3",
    "TSTAT_HURDLE_NEW",
    "TSTAT_HURDLE_ESTABLISHED",
    "TSTAT_HURDLE_INDIA",
    "MIN_OBSERVATIONS_ANNUAL",
    "MIN_OBSERVATIONS_ROBUST",
    "OVERFITTING_GUARD_VERSION",
]
