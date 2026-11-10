import unittest
from unittest.mock import patch

from handlers import analysis_handler


class TestRecommendationsSerialization(unittest.TestCase):
    def test_full_document_serialization(self):
        doc = {
            "symbol": "TCS",
            "company_name": "Tata Consultancy Services",
            "technical_score": 0.85,
            "fundamental_score": 0.70,
            "sentiment_score": 0.60,
            "combined_score": 0.80,
            "composite_score": 4.12,
            "is_recommended": True,
            "recommendation_strength": "BUY",
            "buy_price": 3500.0,
            "sell_price": 3850.0,
            "stop_loss": 3360.0,
            "suggested_quantity": 10,
            "allocation_pct": 35.0,
            "entry_atr": 70.0,
            "rr_ratio": 2.5,
            "expected_return_percent": 10.0,
            "pattern_detected": "vcp_breakout",
            "setup_checklist": {"trend": True, "volume": True},
            "technical_pillars": {"momentum": 0.9},
            "reason": "Multi-week VCP contraction with 3x breakout volume",
            "strategy_name": "Swing_Trading_v3",
        }

        with patch("database.get_recommended_shares_with_analytics", return_value=[doc]):
            recs = analysis_handler.get_recommendations()
            self.assertEqual(len(recs), 1)
            r = recs[0]
            self.assertEqual(r["symbol"], "TCS")
            self.assertEqual(r["stop_loss"], 3360.0)
            self.assertEqual(r["composite_score"], 4.12)
            self.assertEqual(r["suggested_quantity"], 10)
            self.assertEqual(r["allocation_pct"], 35.0)
            self.assertEqual(r["entry_atr"], 70.0)
            self.assertEqual(r["rr_ratio"], 2.5)
            self.assertEqual(r["pattern_detected"], "vcp_breakout")
            self.assertEqual(r["setup_checklist"], {"trend": True, "volume": True})
            self.assertEqual(r["technical_pillars"], {"momentum": 0.9})
            self.assertEqual(r["reason"], "Multi-week VCP contraction with 3x breakout volume")
            self.assertEqual(r["strategy_name"], "Swing_Trading_v3")

    def test_fallback_rationale_and_defaults(self):
        doc = {
            "symbol": "INFY",
            "buy_price": 1500.0,
            "sell_price": 1650.0,
            "entry_atr": 30.0,
            "technical_score": 0.75,
            "combined_score": 0.70,
            "reason": "",
        }

        with patch("database.get_recommended_shares_with_analytics", return_value=[doc]):
            recs = analysis_handler.get_recommendations()
            self.assertEqual(len(recs), 1)
            r = recs[0]
            self.assertEqual(r["symbol"], "INFY")
            self.assertEqual(r["stop_loss"], 1440.0)
            self.assertEqual(r["composite_score"], 0.70)
            self.assertEqual(r["suggested_quantity"], 6)
            self.assertEqual(r["rr_ratio"], 2.5)
            self.assertNotIn("pattern_detected", r)
            self.assertNotIn("setup_checklist", r)
            self.assertNotIn("technical_pillars", r)
            self.assertIn("Entry ₹1500.00", r["reason"])
            self.assertIn("Stop Loss ₹1440.00", r["reason"])
            self.assertEqual(r["strategy_name"], "Swing_Trading_v3")

    def test_generic_reason_replacement(self):
        doc = {
            "symbol": "RELIANCE",
            "buy_price": 2500.0,
            "sell_price": 2750.0,
            "stop_loss": 2400.0,
            "reason": "Vectorbt signal: score=0.60",
        }

        with patch("database.get_recommended_shares_with_analytics", return_value=[doc]):
            recs = analysis_handler.get_recommendations()
            self.assertEqual(len(recs), 1)
            r = recs[0]
            self.assertNotIn("Vectorbt signal", r["reason"])
            self.assertIn("Entry ₹2500.00", r["reason"])
            self.assertIn("Stop Loss ₹2400.00", r["reason"])

    def test_pattern_in_reason_extraction(self):
        doc = {
            "symbol": "SBIN",
            "buy_price": 800.0,
            "sell_price": 880.0,
            "stop_loss": 760.0,
            "reason": "Pattern(pullback_21_ema)",
        }

        with patch("database.get_recommended_shares_with_analytics", return_value=[doc]):
            recs = analysis_handler.get_recommendations()
            self.assertEqual(len(recs), 1)
            r = recs[0]
            self.assertEqual(r["pattern_detected"], "pullback_21_ema")
            self.assertIn("EMA 21 Pullback Reversal", r["reason"])


if __name__ == "__main__":
    unittest.main()
