import numpy as np

from database import get_mongodb
from handlers import backtests_handler
from utils.persistence_handler import PersistenceHandler


def test_create_ultimate_backtest_session():
    """Verify create_backtest_session properly tags session_type='ultimate'."""
    handler = PersistenceHandler()
    session_id = handler.create_backtest_session(
        strategy_name="Swing_Trading_v2",
        strategy_config={"test": True},
        capital_config={"initial_capital": 50000},
        symbols=["RELIANCE", "TCS"],
        session_type="ultimate",
    )
    assert session_id is not None

    db = get_mongodb()
    doc = db.backtest_sessions.find_one({"_id": session_id})
    assert doc is not None
    assert doc.get("session_type") == "ultimate"
    assert "Ultimate" in doc.get("session_name", "")
    assert doc.get("strategy_name") == "Swing_Trading_v2"

    # Cleanup test session
    db.backtest_sessions.delete_one({"_id": session_id})


def test_save_and_retrieve_ultimate_phases():
    """Verify ultimate_phases with numpy types are sanitized and retrieved."""
    handler = PersistenceHandler()
    session_id = handler.create_backtest_session(
        strategy_name="Swing_Trading_v2",
        strategy_config={"test": True},
        capital_config={"initial_capital": 50000},
        symbols=["INFY"],
        session_type="ultimate",
    )

    phases_data = {
        "confidence_score": {
            "total_score": np.float64(78.5),
            "confidence_level": "High",
            "edge_verified": np.bool_(True),
            "component_scores": {
                "walk_forward": np.float64(80.0),
                "dsr": np.float64(75.0),
            },
        },
        "validation": {
            "edge_verified": np.bool_(True),
            "dsr": {
                "sr_observed": np.float64(1.45),
                "p_value": np.float64(0.012),
                "significant": np.bool_(True),
            },
        },
        "stress_tests": {
            "regime_tests": [{"regime": "Bull 2017", "cagr": np.float64(34.2), "passed": np.bool_(True)}],
            "cost_sensitivity": [{"scenario": "Realistic", "cagr": np.float64(28.4), "slippage": 0.0015}],
        },
    }

    success = handler.save_ultimate_backtest_phases(session_id, phases_data)
    assert success is True

    # Check via backtests_handler
    retrieved = backtests_handler.get_backtest_session(str(session_id))
    assert retrieved is not None
    assert retrieved.get("session_type") == "ultimate"
    assert "ultimate_phases" in retrieved
    up = retrieved["ultimate_phases"]
    assert up["confidence_score"]["total_score"] == 78.5
    assert up["confidence_score"]["edge_verified"] is True
    assert up["validation"]["dsr"]["significant"] is True

    # Cleanup test session
    db = get_mongodb()
    db.backtest_sessions.delete_one({"_id": session_id})
