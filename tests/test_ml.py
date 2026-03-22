import unittest

import pandas as pd

from quant_bot.ml import build_ml_dataset


class MLTests(unittest.TestCase):
    def test_build_ml_dataset_creates_target_and_features(self) -> None:
        rows = 260
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h"),
                "open": [100 + (index * 0.2) for index in range(rows)],
                "high": [101 + (index * 0.2) for index in range(rows)],
                "low": [99 + (index * 0.2) for index in range(rows)],
                "close": [100 + (index * 0.2) for index in range(rows)],
                "volume": [1_000 + index for index in range(rows)],
                "MA_10": [100 + (index * 0.2) for index in range(rows)],
                "MA_30": [99 + (index * 0.2) for index in range(rows)],
                "MA_200": [98 + (index * 0.2) for index in range(rows)],
                "RSI": [55.0 for _ in range(rows)],
            }
        )

        dataset = build_ml_dataset(df, horizon=12)

        self.assertFalse(dataset.empty)
        self.assertIn("target_up", dataset.columns)
        self.assertIn("ma_gap_short", dataset.columns)
        self.assertIn("volume_change", dataset.columns)
        self.assertTrue(dataset["target_up"].isin([0, 1]).all())


if __name__ == "__main__":
    unittest.main()
