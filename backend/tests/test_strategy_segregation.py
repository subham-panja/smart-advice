from utils.strategy_loader import StrategyLoader


def test_strategy_loader_segregation():
    """Verify strategy segregation into swing and intraday folders."""
    # 1. Swing strategies
    swing_strats = StrategyLoader.load_swing_strategies()
    assert len(swing_strats) >= 1
    for s in swing_strats:
        assert s.get("_category") == "swing"
        assert "swing" in s.get("_relative_path", "").lower()

    # 2. Intraday strategies
    intraday_strats = StrategyLoader.load_intraday_strategies()
    assert len(intraday_strats) >= 1
    for s in intraday_strats:
        assert s.get("_category") == "intraday"
        assert "intraday" in s.get("_relative_path", "").lower()

    # 3. All strategies with metadata
    all_strats = StrategyLoader.load_all_strategies_including_disabled()
    categories = {s.get("_category") for s in all_strats}
    assert "swing" in categories
    assert "intraday" in categories

    # 4. Lookup by name and by relative path
    swing_by_name = StrategyLoader.get_strategy_by_name("Swing_Trading_v2")
    assert swing_by_name["name"] == "Swing_Trading_v2"
    assert swing_by_name["_category"] == "swing"

    swing_by_path = StrategyLoader.get_strategy_by_name("swing/swing_trading_v2.json")
    assert swing_by_path["name"] == "Swing_Trading_v2"

    intraday_by_name = StrategyLoader.get_strategy_by_name("Intraday_Trading_v1")
    assert intraday_by_name["name"] == "Intraday_Trading_v1"
    assert intraday_by_name["_category"] == "intraday"

    intraday_by_path = StrategyLoader.get_strategy_by_name("intraday/intraday_trading.json")
    assert intraday_by_path["name"] == "Intraday_Trading_v1"
