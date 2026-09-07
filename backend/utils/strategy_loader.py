import json
import logging
import os
from typing import Any, Dict, List

from config import STRATEGIES_DIR

logger = logging.getLogger(__name__)


class StrategyLoader:
    """Loads and manages multiple strategy configurations from JSON files."""

    @staticmethod
    def load_all_strategies() -> List[Dict[str, Any]]:
        """Scans the strategies directory and returns a list of enabled strategies."""
        strategies = []
        if not os.path.exists(STRATEGIES_DIR):
            logger.error(f"Strategies directory not found at {STRATEGIES_DIR}")
            raise FileNotFoundError(f"Strategies directory missing: {STRATEGIES_DIR}")

        for filename in os.listdir(STRATEGIES_DIR):
            if filename.endswith(".json"):
                path = os.path.join(STRATEGIES_DIR, filename)
                try:
                    with open(path, "r") as f:
                        strat = json.load(f)
                        # Mandatory field check
                        if "enabled" not in strat:
                            raise KeyError(f"Strategy file {filename} missing mandatory 'enabled' key")

                        if strat["enabled"]:
                            if "name" not in strat:
                                raise KeyError(f"Strategy file {filename} missing mandatory 'name' key")
                            strategies.append(strat)
                            logger.info(f"Loaded strategy: {strat['name']} from {filename}")
                except Exception as e:
                    logger.error(f"Critical error loading strategy from {filename}: {e}")
                    raise e

        return strategies

    @staticmethod
    def get_strategy_by_name(name: str) -> Dict[str, Any]:
        """Fetch a specific strategy by its name or filename."""
        all_strats = StrategyLoader.load_all_strategies_including_disabled()
        clean_name = name[:-5] if name.endswith(".json") else name
        for s in all_strats:
            strat_name = s.get("name", "")
            file_name = s.get("_file_name", "")
            file_stem = file_name[:-5] if file_name.endswith(".json") else file_name
            if (
                strat_name == name
                or strat_name.lower() == name.lower()
                or strat_name.lower() == clean_name.lower()
                or file_name == name
                or file_name.lower() == name.lower()
                or file_stem.lower() == clean_name.lower()
            ):
                return s
        raise ValueError(f"Strategy '{name}' not found.")

    @staticmethod
    def load_all_strategies_including_disabled() -> List[Dict[str, Any]]:
        """Returns all strategies (enabled and disabled) with metadata."""
        strategies = []
        if not os.path.exists(STRATEGIES_DIR):
            return strategies

        for filename in os.listdir(STRATEGIES_DIR):
            if filename.endswith(".json"):
                path = os.path.join(STRATEGIES_DIR, filename)
                try:
                    with open(path, "r") as f:
                        strat = json.load(f)
                        strat["_file_name"] = filename
                        strategies.append(strat)
                except Exception as e:
                    logger.error(f"Error loading strategy from {filename}: {e}")

        return strategies
