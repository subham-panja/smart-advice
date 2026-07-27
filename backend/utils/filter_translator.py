import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class FilterTranslator:
    """Translates generic stock_filters into screener scan clauses and local evaluators."""

    @staticmethod
    def translate_to_scan_clause(filters: List[Dict[str, Any]]) -> str:
        """Converts a list of stock_filters into a screener query string."""
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
                if op == "between":
                    clauses.append(f"latest rsi( {f['period']} ) > {f['min']}")
                    clauses.append(f"latest rsi( {f['period']} ) < {f['max']}")
                elif op == ">":
                    clauses.append(f"latest rsi( {f['period']} ) > {f['value']}")
                elif op == "<":
                    clauses.append(f"latest rsi( {f['period']} ) < {f['value']}")

            elif f_type == "moving_average":
                kind = f["kind"].lower()
                period = f["period"]
                target = f["target"].lower()
                op = f["op"]

                # Skip HMA (Hull Moving Average) - screener doesn't support it
                # These are local evaluation filters only
                if kind == "hma":
                    logger.debug(f"Skipping HMA filter (op={op}) - not supported by screener, evaluated locally")
                    continue

                # Skip 'monitor' op - this is a local monitoring filter, not a screener scan filter
                if op == "monitor":
                    logger.debug("Skipping moving_average monitor filter - evaluated locally in swing signals")
                    continue

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
            # All filters were local-only (HMA, monitor, etc.)
            # Return a minimal valid query that passes all cash stocks
            logger.warning("No screener-compatible filters found. Using minimal scan clause.")
            return "( {cash} ( latest close > 0 ) )"

        return f"( {{cash}} ( {' and '.join(clauses)} ) )"
