"""
Enhanced Volume Analysis Utility
File: utils/volume_analysis.py
"""

from typing import Dict

import pandas as pd

from utils.logger import setup_logging

logger = setup_logging()


class VolumeAnalyzer:
    """
    Enhanced Volume Analysis for trading signal confirmation with no fallbacks.
    """

    def __init__(self, app_config: Dict):
        """Initialize the Volume Analyzer strictly from config."""
        vol_cfg = app_config["volume_analysis_config"]

        # Mandatory parameters from config
        self.volume_breakout_multiplier = vol_cfg["volume_breakout_multiplier"]
        self.volume_strong_multiplier = vol_cfg["volume_strong_multiplier"]
        self.volume_weak_threshold = vol_cfg["volume_weak_threshold"]
        self.volume_lookback = vol_cfg["volume_lookback"]
        self.trend_lookback = vol_cfg["trend_lookback"]
        self.divergence_lookback = vol_cfg["divergence_lookback"]

    def get_volume_confirmation_factor(self, data: pd.DataFrame, signal_type: str = "bullish") -> Dict:
        """Calculate comprehensive volume confirmation factor strictly."""
        try:
            if len(data) < self.volume_lookback:
                return {"factor": 1.0, "strength": "insufficient_data", "details": []}

            current_volume = data["Volume"].iloc[-1]
            avg_volume = data["Volume"].tail(self.volume_lookback).mean()

            if avg_volume == 0:
                raise ValueError("Average volume is zero; cannot calculate confirmation.")

            volume_ratio = current_volume / avg_volume
            details = []
            base_factor = 1.0

            # Basic volume confirmation
            if volume_ratio >= self.volume_strong_multiplier:
                base_factor = 1.4
                details.append(f"Strong volume: {volume_ratio:.1f}x average")
            elif volume_ratio >= self.volume_breakout_multiplier:
                base_factor = 1.2
                details.append(f"Good volume: {volume_ratio:.1f}x average")
            elif volume_ratio >= 1.0:
                base_factor = 1.0
                details.append(f"Normal volume: {volume_ratio:.1f}x average")
            elif volume_ratio >= self.volume_weak_threshold:
                base_factor = 0.9
                details.append(f"Weak volume: {volume_ratio:.1f}x average")
            else:
                base_factor = 0.7
                details.append(f"Very weak volume: {volume_ratio:.1f}x average")

            return {"factor": round(base_factor, 2), "volume_ratio": round(volume_ratio, 2), "details": details}
        except Exception as e:
            logger.error(f"Volume analysis failure: {e}")
            raise e

    def _analyze_volume_trend(self, data: pd.DataFrame) -> Dict:
        """Analyze volume trend over recent periods."""
        try:
            if len(data) < self.trend_lookback * 2:
                return {"factor": 1.0, "details": []}
            recent_volume = data["Volume"].tail(self.trend_lookback).mean()
            historical_volume = data["Volume"].tail(self.trend_lookback * 2).head(self.trend_lookback).mean()
            if historical_volume == 0:
                return {"factor": 1.0, "details": []}
            trend_ratio = recent_volume / historical_volume
            if trend_ratio >= 1.2:
                return {"factor": 1.1, "details": ["Increasing volume trend"]}
            elif trend_ratio <= 0.8:
                return {"factor": 0.9, "details": ["Declining volume trend"]}
            return {"factor": 1.0, "details": []}
        except Exception:
            return {"factor": 1.0, "details": []}

    def _analyze_volume_price_divergence(self, data: pd.DataFrame, signal_type: str) -> Dict:
        """Analyze volume-price divergence patterns."""
        try:
            if len(data) < self.divergence_lookback:
                return {"factor": 1.0, "details": []}
            recent_data = data.tail(self.divergence_lookback)
            if recent_data["Close"].iloc[0] == 0 or recent_data["Volume"].iloc[0] == 0:
                return {"factor": 1.0, "details": []}
            price_change = (recent_data["Close"].iloc[-1] - recent_data["Close"].iloc[0]) / recent_data["Close"].iloc[0]
            volume_change = (recent_data["Volume"].iloc[-1] - recent_data["Volume"].iloc[0]) / recent_data[
                "Volume"
            ].iloc[0]
            if signal_type == "bullish":
                if price_change < -0.02 and volume_change > 0.1:
                    return {"factor": 1.2, "details": ["Bullish volume divergence"]}
                elif price_change > 0.02 and volume_change < -0.1:
                    return {"factor": 0.9, "details": ["Weak volume confirmation"]}
            return {"factor": 1.0, "details": []}
        except Exception:
            return {"factor": 1.0, "details": []}

    def _analyze_volume_accumulation(self, data: pd.DataFrame) -> Dict:
        """Analyze volume accumulation patterns."""
        try:
            if len(data) < 10:
                return {"factor": 1.0, "details": []}
            recent_data = data.tail(5)
            up_v, down_v = 0, 0
            for i in range(1, len(recent_data)):
                if recent_data["Close"].iloc[i] > recent_data["Close"].iloc[i - 1]:
                    up_v += recent_data["Volume"].iloc[i]
                elif recent_data["Close"].iloc[i] < recent_data["Close"].iloc[i - 1]:
                    down_v += recent_data["Volume"].iloc[i]
            total = up_v + down_v
            if total > 0:
                ratio = up_v / total
                if ratio >= 0.7:
                    return {"factor": 1.15, "details": ["Strong buying pressure"]}
                elif ratio <= 0.3:
                    return {"factor": 0.85, "details": ["Selling pressure"]}
            return {"factor": 1.0, "details": []}
        except Exception:
            return {"factor": 1.0, "details": []}

    def detect_volume_breakout(self, data: pd.DataFrame, price_breakout: bool = False) -> Dict:
        """Detect volume breakouts strictly."""
        if len(data) < self.volume_lookback:
            return {"detected": False, "strength": 0.0, "details": []}

        current_volume = data["Volume"].iloc[-1]
        avg_volume = data["Volume"].tail(self.volume_lookback).mean()
        volume_std = data["Volume"].tail(self.volume_lookback).std()

        if avg_volume == 0 or volume_std == 0:
            raise ValueError("Insufficient data for volume breakout detection.")

        volume_z_score = (current_volume - avg_volume) / volume_std
        volume_ratio = current_volume / avg_volume

        details = []
        if volume_ratio >= self.volume_strong_multiplier and volume_z_score >= 2.0:
            strength = 1.0
        elif volume_ratio >= self.volume_breakout_multiplier and volume_z_score >= 1.5:
            strength = 0.8
        else:
            return {"detected": False, "strength": 0.0, "details": []}

        return {
            "detected": True,
            "strength": strength,
            "volume_ratio": volume_ratio,
            "z_score": volume_z_score,
            "details": details,
        }


def get_enhanced_volume_confirmation(data: pd.DataFrame, app_config: Dict, signal_type: str = "bullish") -> Dict:
    """Convenience function strictly using app_config."""
    analyzer = VolumeAnalyzer(app_config)
    confirmation = analyzer.get_volume_confirmation_factor(data, signal_type)

    # Add Trend analysis
    trend = analyzer._analyze_volume_trend(data)
    confirmation["factor"] *= trend["factor"]
    confirmation["details"].extend(trend["details"])

    # Add Divergence
    div = analyzer._analyze_volume_price_divergence(data, signal_type)
    confirmation["factor"] *= div["factor"]
    confirmation["details"].extend(div["details"])

    # Add Accumulation
    acc = analyzer._analyze_volume_accumulation(data)
    confirmation["factor"] *= acc["factor"]
    confirmation["details"].extend(acc["details"])

    return confirmation
