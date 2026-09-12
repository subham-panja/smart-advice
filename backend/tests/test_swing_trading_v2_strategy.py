from utils.strategy_loader import StrategyLoader


def test_load_swing_trading_v2():
    """Verify Swing_Trading_v2 strategy config loads and has compounding parameters."""
    strategy = StrategyLoader.get_strategy_by_name("Swing_Trading_v2")
    assert strategy is not None
    assert strategy.get("name") == "Swing_Trading_v2"

    risk = strategy.get("risk_management", {})
    assert risk.get("max_positions") <= 4, "Compounding strategy must keep concentrated positions"
    assert risk.get("risk_per_trade_pct", 0) >= 3.0, "Risk per trade should allow runner compounding"

    pyramid = strategy.get("pyramiding", {})
    assert pyramid.get("enabled") is True, "Pyramiding must be enabled for Swing_Trading_v2"
    assert len(pyramid.get("steps", [])) >= 1, "At least one pyramid add step required"

    regime = risk.get("regime_adaptive_risk", {})
    assert "bull" in regime and "bear" in regime, "Regime-adaptive risk rules must be present"
