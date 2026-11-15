import unittest

import pandas as pd

from src.forecasting import (
    create_monthly_forecast_summary,
    forecast_weekday_baseline,
)
from src.feature_engineering import create_time_features
from src.run_pipeline import select_training_features


class ForecastingTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2026-01-18", periods=14, freq="D")
        self.history = pd.DataFrame({
            "DATE": dates,
            "BOOKING_COUNT": [100 + date.dayofweek for date in dates],
        })

    def test_weekday_baseline_returns_requested_horizon(self):
        forecast = forecast_weekday_baseline(self.history, horizon=90)

        self.assertEqual(len(forecast), 90)
        self.assertEqual(
            forecast["DATE"].iloc[0],
            self.history["DATE"].iloc[-1] + pd.Timedelta(days=1),
        )
        self.assertEqual(
            forecast["FORECAST_METHOD"].unique().tolist(),
            ["weekday_mean_baseline"],
        )
        self.assertTrue((forecast["FORECAST_BOOKING_COUNT"] >= 0).all())

    def test_monthly_summary_preserves_totals_and_days(self):
        forecast = forecast_weekday_baseline(self.history, horizon=10)
        summary = create_monthly_forecast_summary(forecast)

        self.assertEqual(summary["FORECAST_DAYS"].sum(), 10)
        self.assertEqual(
            summary["FORECAST_BOOKINGS"].sum(),
            forecast["FORECAST_BOOKING_COUNT"].sum(),
        )

    def test_baseline_rejects_gaps(self):
        incomplete = self.history.drop(index=4)

        with self.assertRaisesRegex(ValueError, "missing dates"):
            forecast_weekday_baseline(incomplete)

    def test_short_history_uses_reduced_lag_features(self):
        dates = pd.date_range("2026-01-01", periods=31, freq="D")
        short_history = pd.DataFrame({
            "DATE": dates,
            "BOOKING_COUNT": [100 + date.dayofweek for date in dates],
        })
        features = create_time_features(short_history)

        selected = select_training_features(features)

        self.assertIn("lag_7", selected)
        self.assertNotIn("lag_28", selected)


if __name__ == "__main__":
    unittest.main()
