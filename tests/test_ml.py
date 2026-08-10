import math
import unittest

import numpy as np
import pandas as pd

from quant_bot.ml import MLModel, build_ml_dataset, compute_features_at_row, run_ml_experiment


def _make_df(rows: int = 300) -> pd.DataFrame:
    close = [100 + 20 * math.sin(i * 0.2) for i in range(rows)]
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h"),
        "open":   [c - 0.1 for c in close],
        "high":   [c + 0.5 for c in close],
        "low":    [c - 0.5 for c in close],
        "close":  close,
        "volume": [1_000 + (i % 100) for i in range(rows)],
        "MA_10":  [100 + 18 * math.sin(i * 0.2) for i in range(rows)],
        "MA_30":  [100 + 16 * math.sin(i * 0.2) for i in range(rows)],
        "MA_50":  [100 + 14 * math.sin(i * 0.2) for i in range(rows)],
        "MA_100": [100 + 12 * math.sin(i * 0.2) for i in range(rows)],
        "MA_200": [100 + 10 * math.sin(i * 0.2) for i in range(rows)],
        "RSI":         [55.0] * rows,
        "MACD":        [0.5 * math.sin(i * 0.2) for i in range(rows)],
        "MACD_signal": [0.4 * math.sin(i * 0.2) for i in range(rows)],
        "BB_upper": [100 + 25 * math.sin(i * 0.2) for i in range(rows)],
        "BB_lower": [100 + 15 * math.sin(i * 0.2) for i in range(rows)],
        "ATR":      [2.0] * rows,
        "VOL_MA_20":  [1_050.0] * rows,
        "VOL_STD_20": [30.0]    * rows,
        "VOL_24":     [0.01]    * rows,
        "HIGH_20":    [c + 1.0  for c in close],
        "LOW_20":     [c - 1.0  for c in close],
    })


class TestBuildMLDataset(unittest.TestCase):

    def test_creates_target_and_features(self) -> None:
        dataset = build_ml_dataset(_make_df(), horizon=12)
        self.assertFalse(dataset.empty)
        self.assertIn("target_up", dataset.columns)
        self.assertIn("ma_gap_short", dataset.columns)
        self.assertIn("volume_change", dataset.columns)
        self.assertTrue(dataset["target_up"].isin([0, 1]).all())

    def test_no_nans_in_feature_columns(self) -> None:
        from quant_bot.ml import FEATURE_COLUMNS
        dataset = build_ml_dataset(_make_df(), horizon=12)
        self.assertFalse(dataset[FEATURE_COLUMNS].isna().any().any())


class TestComputeFeaturesAtRow(unittest.TestCase):

    def test_returns_none_for_early_rows(self) -> None:
        df = _make_df()
        self.assertIsNone(compute_features_at_row(df, 0))
        self.assertIsNone(compute_features_at_row(df, 71))

    def test_returns_dict_with_all_features_after_warmup(self) -> None:
        from quant_bot.ml import FEATURE_COLUMNS
        df = _make_df()
        features = compute_features_at_row(df, 150)
        self.assertIsNotNone(features)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, features)


class TestRunMLExperiment(unittest.TestCase):

    def test_returns_ml_model(self) -> None:
        result = run_ml_experiment(_make_df(300), horizon=12)
        self.assertIsInstance(result.model, MLModel)

    def test_model_predict_proba_in_unit_interval(self) -> None:
        result = run_ml_experiment(_make_df(300), horizon=12)
        features = {col: 0.0 for col in result.model.feature_columns}
        prob = result.model.predict_proba(features)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_train_cutoff_timestamp_is_after_first_row(self) -> None:
        df = _make_df(300)
        result = run_ml_experiment(df, horizon=12)
        self.assertGreater(
            result.model.train_cutoff_timestamp,
            df["timestamp"].iloc[0],
        )


if __name__ == "__main__":
    unittest.main()
