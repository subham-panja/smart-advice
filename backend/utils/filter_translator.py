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
            f_type = f.get("type")
            op = f.get("op")

            if f_type == "price":
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
                clauses.append(f"latest rsi( {f['period']} ) {op} {f['value']}")

            elif f_type == "moving_average":
                kind = f.get("kind", "SMA").lower()
                period = f.get("period")
                target = f.get("target", "close").lower()
                # Chartink format: latest close > latest sma( close,50 )
                if f.get("price_above", True) or op == ">" or op == "<":
                    operator = op if op in [">", "<"] else ">"
                    clauses.append(f"latest {target} {operator} latest {kind}( close,{period} )")

            elif f_type == "volume_spike_lookup":
                # Complex EP volume spike: ( [0] 1 day ago volume > [0] 1 day ago sma(volume,50) * 3 or ... )
                lookback = f.get("lookback_days", 5)
                multiplier = f.get("multiplier", 3.0)
                ma_period = f.get("ma_period", 50)

                spike_clauses = []
                for i in range(1, lookback + 1):
                    spike_clauses.append(
                        f"[0] {i} day ago volume > [0] {i} day ago sma(volume,{ma_period}) * {multiplier}"
                    )

                clauses.append(f"( {' or '.join(spike_clauses)} )")

        if not clauses:
            return ""

        return f"( {{cash}} ( {' and '.join(clauses)} ) )"
