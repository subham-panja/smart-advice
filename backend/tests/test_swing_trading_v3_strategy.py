from utils.strategy_loader import StrategyLoader


def test_load_swing_trading_v3():
    """Verify Swing_Trading_v3 strategy config loads, is enabled, and has Mukul Agrawal parameters."""
    # 1. Verify enabled in active swing strategies
    active_strats = StrategyLoader.load_swing_strategies()
    active_names = [s.get("name") for s in active_strats]
    assert "Swing_Trading_v3" in active_names, "Swing_Trading_v3 must be enabled"

    # 2. Verify strategy parameters
    strategy = StrategyLoader.get_strategy_by_name("Swing_Trading_v3")
    assert strategy is not None
    assert strategy.get("name") == "Swing_Trading_v3"
    assert strategy.get("enabled") is True
    assert "Mukul Agrawal" in strategy.get("description", "")

    # 3. Verify risk management
    risk = strategy.get("risk_management", {})
    assert risk.get("max_positions") == 5
    assert risk.get("risk_per_trade_pct") == 1.8
    assert risk.get("max_risk_per_trade_pct") == 2.0

    # 4. Verify exit rules & trailing stop
    exit_rules = strategy.get("exit_rules", {})
    assert exit_rules.get("atr_stop_multiplier") == 2.0
    assert exit_rules.get("trail_stop_atr") == 2.6
    assert exit_rules.get("trail_min_gain_pct") == 4.0
    assert exit_rules.get("hard_stop_loss_pct") == 7.0

    # 5. Verify market breadth filter
    breadth = strategy.get("market_breadth_filter", {})
    assert breadth.get("enabled") is True
    assert breadth.get("min_advance_pct") == 36.0


def test_swing_trading_v3_evaluator_init():
    """Verify StrategyEvaluator initializes all strategy indicators for Swing_Trading_v3 without missing parameter errors."""
    from scripts.strategy_evaluator import StrategyEvaluator

    strategy = StrategyLoader.get_strategy_by_name("Swing_Trading_v3")
    evaluator = StrategyEvaluator(strategy)
    assert "MACD_Signal_Crossover" in evaluator.instances
    assert "Volume_Breakout" in evaluator.instances
    assert "ADX_Trend_Strength" in evaluator.instances
    assert "RSI_Overbought_Oversold" in evaluator.instances
