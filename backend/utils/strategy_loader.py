import json
import logging
import os
from typing import Any, Dict, List, Optional

from config import STRATEGIES_DIR

logger = logging.getLogger(__name__)


class StrategyLoader:
    """Loads and manages multiple strategy configurations from JSON files across categories."""

    @staticmethod
    def _infer_category(path: str) -> str:
        """Infer category (swing, intraday, etc.) from relative directory."""
        rel = os.path.relpath(path, STRATEGIES_DIR).lower()
        if rel.startswith("intraday") or "intraday" in rel:
            return "intraday"
        if rel.startswith("swing") or "swing" in rel:
            return "swing"
        return "swing"

    @staticmethod
    def load_all_strategies(category: Optional[str] = "swing") -> List[Dict[str, Any]]:
        """
        Scans the strategies directory (including swing and intraday subfolders)
        and returns a list of enabled strategies.
        Defaults to 'swing' strategies for trading cycle and backtests,
        or 'all' / None for everything.
        """
        strategies = []
        if not os.path.exists(STRATEGIES_DIR):
            logger.error(f"Strategies directory not found at {STRATEGIES_DIR}")
            raise FileNotFoundError(f"Strategies directory missing: {STRATEGIES_DIR}")

        for root, _dirs, files in os.walk(STRATEGIES_DIR):
            for filename in sorted(files):
                if filename.endswith(".json"):
                    path = os.path.join(root, filename)
                    strat_cat = StrategyLoader._infer_category(path)
                    if category and category.lower() != "all" and strat_cat != category.lower():
                        continue

                    try:
                        with open(path, "r") as f:
                            strat = json.load(f)
                            # Mandatory field check
                            if "enabled" not in strat:
                                raise KeyError(f"Strategy file {filename} missing mandatory 'enabled' key")

                            if strat["enabled"]:
                                if "name" not in strat:
                                    raise KeyError(f"Strategy file {filename} missing mandatory 'name' key")
                                strat["_file_name"] = filename
                                strat["_category"] = strat_cat
                                strat["_relative_path"] = os.path.relpath(path, STRATEGIES_DIR)
                                strategies.append(strat)
                                logger.info(f"Loaded strategy: {strat['name']} from {strat['_relative_path']}")
                    except Exception as e:
                        logger.error(f"Critical error loading strategy from {filename}: {e}")
                        raise e

        return strategies

    @staticmethod
    def load_swing_strategies() -> List[Dict[str, Any]]:
        """Load enabled swing trading strategies."""
        return StrategyLoader.load_all_strategies(category="swing")

    @staticmethod
    def load_intraday_strategies() -> List[Dict[str, Any]]:
        """Load enabled intraday trading strategies."""
        return StrategyLoader.load_all_strategies(category="intraday")

    @staticmethod
    def get_strategy_by_name(name: str) -> Dict[str, Any]:
        """Fetch a specific strategy by its name, filename, stem, or relative path."""
        all_strats = StrategyLoader.load_all_strategies_including_disabled(category=None)
        clean_name = name[:-5] if name.endswith(".json") else name
        clean_name_lower = clean_name.lower()
        name_lower = name.lower()

        for s in all_strats:
            strat_name = s.get("name", "")
            file_name = s.get("_file_name", "")
            file_stem = file_name[:-5] if file_name.endswith(".json") else file_name
            rel_path = s.get("_relative_path", "")
            rel_stem = rel_path[:-5] if rel_path.endswith(".json") else rel_path

            if (
                strat_name == name
                or strat_name.lower() == name_lower
                or strat_name.lower() == clean_name_lower
                or file_name == name
                or file_name.lower() == name_lower
                or file_stem.lower() == clean_name_lower
                or rel_path == name
                or rel_path.lower() == name_lower
                or rel_stem.lower() == clean_name_lower
            ):
                return s
        raise ValueError(f"Strategy '{name}' not found.")

    @staticmethod
    def load_all_strategies_including_disabled(category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns all strategies (enabled and disabled) with metadata, optionally filtered by category."""
        strategies = []
        if not os.path.exists(STRATEGIES_DIR):
            return strategies

        for root, _dirs, files in os.walk(STRATEGIES_DIR):
            for filename in sorted(files):
                if filename.endswith(".json"):
                    path = os.path.join(root, filename)
                    strat_cat = StrategyLoader._infer_category(path)
                    if category and category.lower() != "all" and strat_cat != category.lower():
                        continue

                    try:
                        with open(path, "r") as f:
                            strat = json.load(f)
                            strat["_file_name"] = filename
                            strat["_category"] = strat_cat
                            strat["_relative_path"] = os.path.relpath(path, STRATEGIES_DIR)
                            strategies.append(strat)
                    except Exception as e:
                        logger.error(f"Error loading strategy from {filename}: {e}")

        return strategies
