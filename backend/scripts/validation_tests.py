"""
Statistical Validation Tests
=============================

Implements rigorous statistical tests to prove a strategy's edge is real
and not the result of luck or overfitting:

1. Deflated Sharpe Ratio (DSR) — Lopez de Prado
2. Monte Carlo Permutation Test
3. Minimum Track Record Length (MLRS)
"""

import math
from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# 1. Deflated Sharpe Ratio (DSR)
# ---------------------------------------------------------------------------


def deflated_sharpe_ratio(
    daily_returns: List[float],
    n_trials: int = 30,
    annual_factor: int = 252,
) -> Dict[str, Any]:
    """Calculate the Deflated Sharpe Ratio per Lopez de Prado.

    The observed Sharpe ratio is inflated by trying many parameter combinations.
    The DSR corrects for this selection bias.

    Args:
        daily_returns: List of daily portfolio returns
        n_trials: Number of backtest variations tried (estimate from config complexity)
        annual_factor: Trading days per year

    Returns:
        Dict with DSR, observed SR, threshold, and significance assessment
    """
    returns = np.array(daily_returns)
    n = len(returns)

    if n < 30:
        return {"status": "error", "reason": "Insufficient returns (need 30+)"}

    sr_observed = _compute_sharpe(returns, annual_factor)

    # Expected maximum SR under null hypothesis (Gumbel approximation)
    # E[max SR] ≈ sqrt(2 * log(N_trials)) for large N
    threshold = math.sqrt(2.0 * math.log(max(n_trials, 2)))

    # Deflated Sharpe Ratio
    # DSR = SR_observed - E[max SR under null]
    # But properly: DSR uses the variance-adjusted formula
    var_sr = 1.0 / n  # Variance of SR estimator
    sr_std = math.sqrt(var_sr)

    # Z-score of observed SR vs threshold
    z_score = (sr_observed - threshold * sr_std) / sr_std if sr_std > 0 else 0

    # Probability that SR is statistically significant
    from scipy.stats import norm

    p_value = 1.0 - norm.cdf(z_score)

    dsr = sr_observed - threshold * sr_std

    return {
        "sr_observed": round(sr_observed, 4),
        "sr_threshold": round(threshold * sr_std, 4),
        "dsr": round(dsr, 4),
        "n_trials": n_trials,
        "n_observations": n,
        "z_score": round(z_score, 4),
        "p_value": round(p_value, 6),
        "significant": dsr > 0,
        "confidence_pct": round((1 - p_value) * 100, 2),
    }


def _compute_sharpe(returns: np.ndarray, annual_factor: int = 252) -> float:
    """Compute annualized Sharpe ratio (risk-free rate = 0)."""
    if len(returns) < 2:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret == 0:
        return 0.0
    return float(mean_ret / std_ret * math.sqrt(annual_factor))


# ---------------------------------------------------------------------------
# 2. Monte Carlo Permutation Test
# ---------------------------------------------------------------------------


def monte_carlo_permutation_test(
    daily_returns: List[float],
    n_simulations: int = 5000,
    annual_factor: int = 252,
    seed: int = 42,
) -> Dict[str, Any]:
    """Test if strategy Sharpe ratio is statistically significant.

    Uses autocorrelation-adjusted standard error to properly test whether
    the observed Sharpe ratio is significantly different from zero, accounting
    for the temporal dependence in returns that trend-following strategies exploit.

    Also performs a sign-flip permutation test on mean returns to verify
    that the strategy's mean return is significantly positive.
    """
    from scipy.stats import norm

    returns = np.array(daily_returns)
    n = len(returns)

    if n < 30:
        return {"status": "error", "reason": "Insufficient returns (need 30+)"}

    actual_sharpe = _compute_sharpe(returns, annual_factor)
    mean_ret = np.mean(returns)

    # Autocorrelation-adjusted Sharpe SE
    max_lag = min(20, n // 4)
    rho_sum = 0.0
    for k in range(1, max_lag + 1):
        rho_k = np.corrcoef(returns[:-k], returns[k:])[0, 1]
        if not np.isnan(rho_k):
            rho_sum += rho_k

    ac_factor = max(1.0, 1.0 + 2.0 * rho_sum)
    adjusted_se = math.sqrt(ac_factor / n)

    z_score = actual_sharpe / (adjusted_se * math.sqrt(annual_factor)) if adjusted_se > 0 else 0
    p_value_sharpe = 1.0 - norm.cdf(z_score)

    # Sign-flip permutation test on mean return
    rng = np.random.RandomState(seed)
    better_count = 0
    for _ in range(n_simulations):
        signs = rng.choice([-1, 1], size=n)
        flipped_mean = np.mean(returns * signs)
        if flipped_mean > mean_ret:
            better_count += 1

    p_value_mean = better_count / n_simulations
    p_value = max(p_value_sharpe, p_value_mean)

    # Bootstrap confidence interval for Sharpe
    bootstrap_sharpes = []
    for _ in range(min(n_simulations, 2000)):
        sample = rng.choice(returns, size=n, replace=True)
        bootstrap_sharpes.append(_compute_sharpe(sample, annual_factor))

    bootstrap_sharpes = np.array(bootstrap_sharpes)
    sharpe_ci_low = float(np.percentile(bootstrap_sharpes, 5))
    sharpe_ci_high = float(np.percentile(bootstrap_sharpes, 95))
    pct_positive = float(np.mean(bootstrap_sharpes > 0) * 100)

    return {
        "actual_sharpe": round(actual_sharpe, 4),
        "autocorrelation_factor": round(ac_factor, 4),
        "adjusted_se": round(adjusted_se, 6),
        "z_score": round(z_score, 4),
        "p_value_sharpe": round(float(p_value_sharpe), 6),
        "p_value_mean_return": round(p_value_mean, 6),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "confidence_pct": round((1 - p_value) * 100, 2),
        "bootstrap_sharpe_ci_low": round(sharpe_ci_low, 4),
        "bootstrap_sharpe_ci_high": round(sharpe_ci_high, 4),
        "bootstrap_pct_positive": round(pct_positive, 2),
        "n_simulations": n_simulations,
        "autocorrelation_sum": round(rho_sum, 4),
    }


# ---------------------------------------------------------------------------
# 3. Minimum Track Record Length (MLRS)
# ---------------------------------------------------------------------------


def minimum_track_record_length(
    daily_returns: List[float],
    annual_factor: int = 252,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Calculate the minimum number of observations needed for the Sharpe
    ratio to be statistically significant at the given alpha level.

    Uses the formula from Lopez de Prado:
        MLRS = [1 + (SR * skew/2) - (SR^2 * kurt/4)] * (Z_alpha / SR)^2

    Args:
        daily_returns: List of daily portfolio returns
        annual_factor: Trading days per year
        alpha: Significance level (default 0.05 = 95% confidence)

    Returns:
        Dict with MLRS, actual observations, and sufficiency assessment
    """
    returns = np.array(daily_returns)
    n = len(returns)

    if n < 10:
        return {"status": "error", "reason": "Insufficient returns (need 10+)"}

    sr = _compute_sharpe(returns, annual_factor)

    if sr <= 0:
        return {
            "sr_annualized": round(sr, 4),
            "mlrs_days": None,
            "mlrs_years": None,
            "actual_days": n,
            "actual_years": round(n / annual_factor, 2),
            "sufficient": False,
            "reason": "Sharpe ratio <= 0, no meaningful MLRS",
        }

    # Calculate skewness and kurtosis
    from scipy.stats import kurtosis, skew

    sk = float(skew(returns))
    kurt = float(kurtosis(returns, fisher=False))  # Pearson kurtosis

    # Z-value for given alpha
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha)

    # MLRS formula (Lopez de Prado)
    mlrs = (1 + (sr * sk / 2) - (sr**2 * kurt / 4)) * (z_alpha / sr) ** 2
    mlrs = max(mlrs, 10)  # Floor at 10 observations

    mlrs_years = mlrs / annual_factor
    actual_years = n / annual_factor

    return {
        "sr_annualized": round(sr, 4),
        "skewness": round(sk, 4),
        "kurtosis": round(kurt, 4),
        "alpha": alpha,
        "z_alpha": round(z_alpha, 4),
        "mlrs_days": int(math.ceil(mlrs)),
        "mlrs_years": round(mlrs_years, 2),
        "actual_days": n,
        "actual_years": round(actual_years, 2),
        "sufficient": n >= mlrs,
        "excess_days": int(n - mlrs) if n >= mlrs else None,
    }


# ---------------------------------------------------------------------------
# 4. Combined Validation Summary
# ---------------------------------------------------------------------------


def run_all_validations(
    daily_returns: List[float],
    n_trials: int = 30,
    mc_sims: int = 5000,
) -> Dict[str, Any]:
    """Run all three statistical validation tests and return combined results.

    Args:
        daily_returns: List of daily portfolio returns
        n_trials: Number of backtest variations for DSR
        mc_sims: Number of Monte Carlo simulations

    Returns:
        Combined validation results
    """
    dsr_result = deflated_sharpe_ratio(daily_returns, n_trials)
    mc_result = monte_carlo_permutation_test(daily_returns, mc_sims)
    mlrs_result = minimum_track_record_length(daily_returns)

    # Overall assessment
    tests_passed = 0
    total_tests = 3

    if dsr_result.get("significant", False):
        tests_passed += 1
    if mc_result.get("significant", False):
        tests_passed += 1
    if mlrs_result.get("sufficient", False):
        tests_passed += 1

    overall_confidence = (tests_passed / total_tests) * 100

    return {
        "dsr": dsr_result,
        "monte_carlo_permutation": mc_result,
        "minimum_track_record": mlrs_result,
        "tests_passed": tests_passed,
        "total_tests": total_tests,
        "overall_confidence_pct": round(overall_confidence, 1),
        "edge_verified": tests_passed >= 2,  # At least 2 of 3 tests must pass
    }
