import math  # Python's standard math library — used here for math.sin() to create an oscillating price series
import unittest  # Python's built-in testing framework — provides test discovery, TestCase, and assertion methods

import numpy as np  # NumPy: needed to check array types in assertions
import pandas as pd  # pandas: needed to build synthetic DataFrames for testing

from quant_bot.ml import MLModel, build_ml_dataset, compute_features_at_row, run_ml_experiment  # import the four ML functions/classes we want to test


def _make_df(rows: int = 300) -> pd.DataFrame:  # helper function: creates a minimal synthetic DataFrame with all columns the ML code expects — no real Binance data needed for tests
    # IMPORTANT: price must oscillate (go up AND down) so that target_up contains both 0s and 1s.
    # sklearn's LogisticRegression raises ValueError if all labels are the same class.
    close = [100 + 20 * math.sin(i * 0.2) for i in range(rows)]  # math.sin(): returns values between -1 and +1; ×20 scales the oscillation to ±$20 around $100 — creates realistic up-and-down movements
    return pd.DataFrame(  # build a synthetic OHLCV+indicator table
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h"),  # hourly timestamps starting 2025-01-01
            "open":   [c - 0.1 for c in close],  # open slightly below close
            "high":   [c + 0.5 for c in close],  # high slightly above close
            "low":    [c - 0.5 for c in close],  # low slightly below close
            "close":  close,  # the oscillating price series
            "volume": [1_000 + (i % 100) for i in range(rows)],  # % 100: modulo keeps volume oscillating 1000–1099
            "MA_10":  [100 + 18 * math.sin(i * 0.2) for i in range(rows)],  # MA10 slightly smaller amplitude than close
            "MA_30":  [100 + 16 * math.sin(i * 0.2) for i in range(rows)],  # MA30 smaller amplitude — slower-moving
            "MA_200": [100 + 10 * math.sin(i * 0.2) for i in range(rows)],  # MA200 smallest amplitude — slowest
            "RSI":    [55.0] * rows,  # constant RSI of 55 — neutral, avoids NaN issues
            # NEW columns required by the expanded feature set:
            "MACD":         [0.5 * math.sin(i * 0.2) for i in range(rows)],   # synthetic MACD line — oscillates around 0
            "MACD_signal":  [0.4 * math.sin(i * 0.2) for i in range(rows)],   # synthetic signal line — slightly smaller amplitude
            "BB_upper":     [100 + 25 * math.sin(i * 0.2) for i in range(rows)],  # upper Bollinger band — wider than close
            "BB_lower":     [100 + 15 * math.sin(i * 0.2) for i in range(rows)],  # lower Bollinger band — narrower than close (always above-below relationship preserved)
            "ATR":          [2.0] * rows,  # constant ATR = 2.0 — represents stable volatility
        }
    )


class TestBuildMLDataset(unittest.TestCase):  # test group for the build_ml_dataset() function

    def test_creates_target_and_features(self) -> None:  # verify the output has the required columns and valid label values
        dataset = build_ml_dataset(_make_df(), horizon=12)  # build the supervised dataset with a 12-candle prediction horizon
        self.assertFalse(dataset.empty)  # assertFalse(.empty): confirm the result has at least one row — empty would mean all rows were dropped as NaN
        self.assertIn("target_up", dataset.columns)  # assertIn(): check that "target_up" column was created
        self.assertIn("ma_gap_short", dataset.columns)  # check one of the feature columns was created
        self.assertIn("volume_change", dataset.columns)  # check another feature column
        self.assertTrue(dataset["target_up"].isin([0, 1]).all())  # isin([0,1]): True for each row that is 0 or 1; .all(): True if every row passes — verifies the label is a valid binary value

    def test_no_nans_in_feature_columns(self) -> None:  # verify that after dropna() the feature columns are completely clean
        from quant_bot.ml import FEATURE_COLUMNS  # import the authoritative list of feature names
        dataset = build_ml_dataset(_make_df(), horizon=12)
        self.assertFalse(dataset[FEATURE_COLUMNS].isna().any().any())  # dataset[FEATURE_COLUMNS]: select only the 7 feature columns; .isna(): True for each NaN cell; .any(): True if any cell in each column is NaN; .any() again: True if any column had a NaN — assertFalse confirms there are none


class TestComputeFeaturesAtRow(unittest.TestCase):  # test group for the row-level feature computation used by the backtester

    def test_returns_none_for_early_rows(self) -> None:  # verify that rows with insufficient history return None
        df = _make_df()  # create synthetic data
        self.assertIsNone(compute_features_at_row(df, 0))   # row 0: no history at all — must return None
        self.assertIsNone(compute_features_at_row(df, 23))  # row 23: not enough for return_24 (needs 24 rows back) — must return None

    def test_returns_dict_with_all_features_after_warmup(self) -> None:  # verify that after 24+ rows we get a complete feature dict
        from quant_bot.ml import FEATURE_COLUMNS  # the authoritative feature name list
        df = _make_df()
        features = compute_features_at_row(df, 50)  # row 50 has plenty of history
        self.assertIsNotNone(features)  # assertIsNotNone(): confirm we got a dict back (not None)
        for col in FEATURE_COLUMNS:  # loop over each expected feature name
            self.assertIn(col, features)  # assertIn(): check the key exists in the returned dict — verifies no feature is missing


class TestRunMLExperiment(unittest.TestCase):  # test group for the full ML training and evaluation function

    def test_returns_ml_model(self) -> None:  # verify the result contains a proper MLModel object (sklearn-backed)
        result = run_ml_experiment(_make_df(300), horizon=12)  # run the full ML experiment on 300 synthetic rows
        self.assertIsInstance(result.model, MLModel)  # assertIsInstance(): checks that result.model is an instance of the MLModel class — confirms sklearn training completed and produced a valid object

    def test_model_predict_proba_in_unit_interval(self) -> None:  # verify that predict_proba() always returns a value between 0 and 1
        result = run_ml_experiment(_make_df(300), horizon=12)
        features = {col: 0.0 for col in result.model.feature_columns}  # dict comprehension: create a feature dict with all zeros — a valid (though neutral) input
        prob = result.model.predict_proba(features)  # ask the model for its prediction probability
        self.assertGreaterEqual(prob, 0.0)  # assertGreaterEqual(): verify prob ≥ 0.0 — probabilities can't be negative
        self.assertLessEqual(prob, 1.0)    # assertLessEqual(): verify prob ≤ 1.0 — probabilities can't exceed 1

    def test_train_cutoff_timestamp_is_after_first_row(self) -> None:  # verify the training cutoff is sensibly placed (not at the very beginning)
        df = _make_df(300)  # synthetic data
        result = run_ml_experiment(df, horizon=12)
        self.assertGreater(  # assertGreater(a, b): assert a > b
            result.model.train_cutoff_timestamp,  # the timestamp where training ends (start of the test split)
            df["timestamp"].iloc[0],  # the very first timestamp in the dataset
        )  # if the cutoff equals the first row, the training set would be empty — this test catches that regression


if __name__ == "__main__":  # allow running the test file directly (python tests/test_ml.py) as well as via python -m unittest
    unittest.main()  # unittest.main(): discovers all TestCase subclasses in this file and runs every test_ method
