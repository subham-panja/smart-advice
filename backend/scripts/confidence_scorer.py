"""
Composite Confidence Scorer
============================

Combines all validation results into a single 0-100 confidence score
that tells you whether the strategy edge is real and actionable.

Components:
- Walk-Forward Robustness (20%)
- DSR Significance (15%)
- Monte Carlo p-value (15%)
- Stress Test Pass Rate (15%)
- Parameter Stability (10%)
- Universe Consistency (10%)
- Cost Resilience (10%)
- Data Sufficiency (5%)
"""

from typing import Any, Dict, Optional


def compute_confidence_score(
    walk_forward_results: Optional[Dict[str, Any]] = None,
    validation_results: Optional[Dict[str, Any]] = None,
    stress_test_results: Optional[Dict[str, Any]] = None,
    param_sensitivity_results: Optional[list] = None,
    cost_sensitivity_results: Optional[list] = None,
    base_cagr: float = 0.0,
) -> Dict[str, Any]:
    """Compute composite confidence score (0-100).

    Args:
        walk_forward_results: Results from walk-forward MC analysis
        validation_results: Results from statistical validation (DSR, MC permutation, MLRS)
        stress_test_results: Results from regime stress tests
        param_sensitivity_results: Results from parameter sensitivity
        cost_sensitivity_results: Results from cost sensitivity
        base_cagr: Base case CAGR for comparison

    Returns:
        Dict with component scores, total score, and action recommendation
    """
    scores = {}
    weights = {}

    # 1. Walk-Forward Robustness (20%)
    if walk_forward_results and walk_forward_results.get("status") in ("completed", "success"):
        robustness = walk_forward_results.get("robustness_score", 0)
        positive_cagr_pct = walk_forward_results.get("positive_cagr_pct", 0)
        # Score: average of robustness and % positive CAGR, normalized to 0-100
        wf_score = (robustness + positive_cagr_pct) / 2
        scores["walk_forward"] = min(wf_score, 100)
        weights["walk_forward"] = 0.20
    else:
        scores["walk_forward"] = 0
        weights["walk_forward"] = 0.20

    # 2. DSR Significance (15%)
    if validation_results and validation_results.get("dsr", {}).get("significant", False):
        dsr_p = validation_results["dsr"].get("confidence_pct", 0)
        scores["dsr"] = min(dsr_p, 100)
    elif validation_results:
        scores["dsr"] = 0
    else:
        scores["dsr"] = 50  # Unknown, neutral
    weights["dsr"] = 0.15

    # 3. Monte Carlo Permutation p-value (15%)
    if validation_results and validation_results.get("monte_carlo_permutation"):
        mc_p = validation_results["monte_carlo_permutation"].get("p_value", 1.0)
        mc_confidence = (1 - mc_p) * 100
        scores["mc_permutation"] = min(mc_confidence, 100)
    elif validation_results:
        scores["mc_permutation"] = 0
    else:
        scores["mc_permutation"] = 50
    weights["mc_permutation"] = 0.15

    # 4. Stress Test Pass Rate (15%)
    if stress_test_results:
        regime = stress_test_results.get("regime_tests", [])
        if regime:
            regime_pass = sum(1 for r in regime if r.get("passed", False))
            regime_total = len(regime)
            scores["stress_tests"] = (regime_pass / regime_total) * 100
        else:
            scores["stress_tests"] = 50
    else:
        scores["stress_tests"] = 50
    weights["stress_tests"] = 0.15

    # 5. Parameter Stability (10%)
    if param_sensitivity_results:
        passed = sum(1 for r in param_sensitivity_results if r.get("passed", False))
        total = len(param_sensitivity_results)
        scores["param_stability"] = (passed / total) * 100 if total > 0 else 50
    else:
        scores["param_stability"] = 50
    weights["param_stability"] = 0.10

    # 6. Cost Resilience (10%)
    if cost_sensitivity_results:
        # Check if CAGR > 12% even at highest cost level
        # Count how many scenarios have CAGR > 12%
        resilient = sum(1 for r in cost_sensitivity_results if r.get("cagr", 0) > 12.0)
        total = len(cost_sensitivity_results)
        scores["cost_resilience"] = (resilient / total) * 100 if total > 0 else 50
    else:
        scores["cost_resilience"] = 50
    weights["cost_resilience"] = 0.10

    # 7. Data Sufficiency (5%)
    if validation_results and validation_results.get("minimum_track_record"):
        mlrs = validation_results["minimum_track_record"]
        if mlrs.get("sufficient", False):
            actual = mlrs.get("actual_days", 0)
            required = mlrs.get("mlrs_days", 1)
            # Score: how much excess data beyond minimum
            ratio = min(actual / required, 2.0)  # Cap at 2x
            scores["data_sufficiency"] = min((ratio / 2.0) * 100, 100)
        else:
            scores["data_sufficiency"] = 0
    else:
        scores["data_sufficiency"] = 50
    weights["data_sufficiency"] = 0.05

    # Weighted total
    total_score = sum(scores[k] * weights[k] for k in scores)

    # Realistic CAGR projection
    if total_score >= 80:
        haircut = 0.85
        confidence_level = "Very High"
        action = "Start with Rs 10K, scale up quickly"
    elif total_score >= 65:
        haircut = 0.70
        confidence_level = "High"
        action = "Start with Rs 10K, scale after 3 months"
    elif total_score >= 50:
        haircut = 0.50
        confidence_level = "Moderate"
        action = "Paper trade 3 months first"
    else:
        haircut = 0.25
        confidence_level = "Low"
        action = "Do not trade - refine strategy"

    realistic_cagr = base_cagr * haircut

    return {
        "component_scores": {k: round(v, 1) for k, v in scores.items()},
        "weights": weights,
        "total_score": round(total_score, 1),
        "confidence_level": confidence_level,
        "base_cagr": round(base_cagr, 2),
        "haircut": haircut,
        "realistic_cagr": round(realistic_cagr, 2),
        "action": action,
        "edge_verified": total_score >= 65,
    }


def print_confidence_report(report: Dict[str, Any]):
    """Print formatted confidence report."""
    print(f"\n{'='*70}")
    print("STRATEGY CONFIDENCE REPORT")
    print(f"{'='*70}\n")

    print(f"Overall Confidence Score: {report['total_score']}/100")
    print(f"Confidence Level: {report['confidence_level']}")
    print(f"Edge Verified: {'YES' if report['edge_verified'] else 'NO'}")
    print()

    print("Component Scores:")
    print(f"  {'Component':<25s} {'Score':>6s} {'Weight':>8s} {'Weighted':>8s}")
    print(f"  {'-'*47}")
    component_names = {
        "walk_forward": "Walk-Forward Robustness",
        "dsr": "Deflated Sharpe Ratio",
        "mc_permutation": "Monte Carlo Permutation",
        "stress_tests": "Stress Test Pass Rate",
        "param_stability": "Parameter Stability",
        "cost_resilience": "Cost Resilience",
        "data_sufficiency": "Data Sufficiency",
    }
    for key, name in component_names.items():
        score = report["component_scores"].get(key, 0)
        weight = report["weights"].get(key, 0) * 100
        weighted = score * report["weights"].get(key, 0)
        print(f"  {name:<25s} {score:>5.1f}  {weight:>6.0f}%  {weighted:>6.1f}")

    print(f"\n  {'TOTAL':<25s} {report['total_score']:>5.1f}")
    print()
    print(f"Base Backtest CAGR:    {report['base_cagr']:.1f}%")
    print(f"Haircut Applied:       {(1 - report['haircut']) * 100:.0f}%")
    print(f"Realistic CAGR:        {report['realistic_cagr']:.1f}%")
    print()
    print(f"Recommended Action: {report['action']}")
    print(f"{'='*70}\n")
