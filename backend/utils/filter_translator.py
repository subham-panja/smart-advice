import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class FilterTranslator:
    """Translates generic stock_filters into Chartink scan clauses and local evaluators."""

    @staticmethod
    def translate_to_chartink(filters: List[Dict[str, Any]]) -> str:
        """Converts a list of stock_filters into a Chartink query string."""
        clauses = []

        for f in filters:
            f_type = f["type"]

            if f_type == "price":
                op = f["op"]
                if op == "between":
                    clauses.append(f"latest close > {f['min']}")
                    clauses.append(f"latest close < {f['max']}")
                elif op == ">":
                    clauses.append(f"latest close > {f['value']}")
                elif op == "<":
                    clauses.append(f"latest close < {f['value']}")

            elif f_type == "volume":
                clauses.append(f"latest volume > {f['value']}")

            elif f_type == "market_cap":
                clauses.append(f"market cap > {f['value']}")

            elif f_type == "rsi":
                op = f["op"]
                clauses.append(f"latest rsi( {f['period']} ) {op} {f['value']}")

            elif f_type == "moving_average":
                kind = f["kind"].lower()
                period = f["period"]
                target = f["target"].lower()
                op = f["op"]
                clauses.append(f"latest {target} {op} latest {kind}( close,{period} )")

            elif f_type == "volume_spike_lookup":
                # This specific filter type does not require an 'op' key as it defines its own internal logic
                lookback = f["lookback_days"]
                multiplier = f["multiplier"]
                ma_period = f["ma_period"]

                spike_clauses = []
                for i in range(1, lookback + 1):
                    spike_clauses.append(f"{i} day ago volume > {i} day ago sma(volume,{ma_period}) * {multiplier}")

                clauses.append(f"( {' or '.join(spike_clauses)} )")

        if not clauses:
            raise ValueError("No valid filters provided for Chartink translation.")

        return f"( {{cash}} ( {' and '.join(clauses)} ) )"
