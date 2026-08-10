from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from quant_bot.config import DEFAULT_ML_HORIZON, DEFAULT_TRAIN_RATIO

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: list[str] = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "return_72",
    "ma_gap_short",
    "ma_gap_50_100",
    "ma_gap_trend",
    "ma_50_dist",
    "ma_100_dist",
    "rsi_scaled",
    "rsi_change_6",
    "macd_hist_pct",
    "price_acceleration",
    "atr_pct",
    "volatility_24",
    "bb_position",
    "bb_width_pct",
    "volume_change",
    "volume_zscore",
    "volume_ma_ratio",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "hour_of_day",
    "day_of_week",
    "dist_from_high_20",
    "dist_from_low_20",
]


@dataclass
class MLModel:
    pipeline: Pipeline
    feature_columns: list[str]
    train_cutoff_timestamp: pd.Timestamp

    def predict_proba(self, features: dict[str, float]) -> float:
        x = np.array(
            [[features.get(col, 0.0) for col in self.feature_columns]]
        )
        return float(self.pipeline.predict_proba(x)[0, 1])


def compute_features_at_row(df: pd.DataFrame, i: int) -> dict[str, float] | None:
    if i < 24:
        return None

    if i < 72:
        return None

    close    = df["close"].iloc[i]
    close_1  = df["close"].iloc[i - 1]
    close_3  = df["close"].iloc[i - 3]
    close_6  = df["close"].iloc[i - 6]
    close_12 = df["close"].iloc[i - 12]
    close_24 = df["close"].iloc[i - 24]
    close_72 = df["close"].iloc[i - 72]

    open_  = df["open"].iloc[i]
    high   = df["high"].iloc[i]
    low    = df["low"].iloc[i]
    vol    = df["volume"].iloc[i]
    vol_1  = df["volume"].iloc[i - 1]

    ma10   = df["MA_10"].iloc[i]
    ma30   = df["MA_30"].iloc[i]
    ma50   = df["MA_50"].iloc[i]
    ma100  = df["MA_100"].iloc[i]
    ma200  = df["MA_200"].iloc[i]
    rsi    = df["RSI"].iloc[i]
    rsi_6  = df["RSI"].iloc[i - 6]
    macd     = df["MACD"].iloc[i]
    macd_sig = df["MACD_signal"].iloc[i]
    bb_up  = df["BB_upper"].iloc[i]
    bb_lo  = df["BB_lower"].iloc[i]
    atr    = df["ATR"].iloc[i]
    vol_24 = df["VOL_24"].iloc[i]
    vol_ma = df["VOL_MA_20"].iloc[i]
    vol_std = df["VOL_STD_20"].iloc[i]
    high_20 = df["HIGH_20"].iloc[i]
    low_20  = df["LOW_20"].iloc[i]
    ts = df["timestamp"].iloc[i]

    required = [close, open_, high, low, ma10, ma30, ma50, ma100, ma200, rsi, rsi_6,
                macd, macd_sig, bb_up, bb_lo, atr, vol_24, vol_ma, vol_std, high_20, low_20]
    if any(pd.isna(v) for v in required):
        return None
    if close == 0 or close_1 == 0 or close_3 == 0 or close_6 == 0 or close_12 == 0 or close_24 == 0 or close_72 == 0:
        return None
    bb_range = bb_up - bb_lo
    candle_range = high - low
    if bb_range == 0 or candle_range == 0 or vol_std == 0:
        return None

    return_1 = (close - close_1) / close_1
    return_6 = (close - close_6) / close_6
    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low

    return {
        "return_1":  return_1,
        "return_3":  (close - close_3) / close_3,
        "return_6":  return_6,
        "return_12": (close - close_12) / close_12,
        "return_24": (close - close_24) / close_24,
        "return_72": (close - close_72) / close_72,
        "ma_gap_short":  (ma10 - ma30) / close,
        "ma_gap_50_100": (ma50 - ma100) / close,
        "ma_gap_trend":  (close - ma200) / close,
        "ma_50_dist":    (close - ma50) / close,
        "ma_100_dist":   (close - ma100) / close,
        "rsi_scaled":         rsi / 100.0,
        "rsi_change_6":       (rsi - rsi_6) / 100.0,
        "macd_hist_pct":      (macd - macd_sig) / close,
        "price_acceleration": return_1 - return_6,
        "atr_pct":       atr / close,
        "volatility_24": vol_24,
        "bb_position":   (close - bb_lo) / bb_range,
        "bb_width_pct":  bb_range / close,
        "volume_change":   (vol - vol_1) / vol_1 if vol_1 != 0 else 0.0,
        "volume_zscore":   (vol - vol_ma) / vol_std,
        "volume_ma_ratio": vol / vol_ma if vol_ma != 0 else 1.0,
        "body_pct":       body / candle_range,
        "upper_wick_pct": upper_wick / candle_range,
        "lower_wick_pct": lower_wick / candle_range,
        "hour_of_day":  ts.hour / 23.0,
        "day_of_week":  ts.dayofweek / 6.0,
        "dist_from_high_20": (close - high_20) / close,
        "dist_from_low_20":  (close - low_20) / close,
    }


@dataclass
class MLExperimentResult:
    metrics: dict[str, float]
    prediction_frame: pd.DataFrame
    feature_importance: pd.DataFrame
    model: MLModel


def build_ml_dataset(df: pd.DataFrame, horizon: int = DEFAULT_ML_HORIZON) -> pd.DataFrame:
    dataset = df.copy()

    dataset["return_1"]  = dataset["close"].pct_change(1)
    dataset["return_3"]  = dataset["close"].pct_change(3)
    dataset["return_6"]  = dataset["close"].pct_change(6)
    dataset["return_12"] = dataset["close"].pct_change(12)
    dataset["return_24"] = dataset["close"].pct_change(24)
    dataset["return_72"] = dataset["close"].pct_change(72)

    dataset["ma_gap_short"]  = (dataset["MA_10"] - dataset["MA_30"]) / dataset["close"]
    dataset["ma_gap_50_100"] = (dataset["MA_50"] - dataset["MA_100"]) / dataset["close"]
    dataset["ma_gap_trend"]  = (dataset["close"] - dataset["MA_200"]) / dataset["close"]
    dataset["ma_50_dist"]    = (dataset["close"] - dataset["MA_50"])  / dataset["close"]
    dataset["ma_100_dist"]   = (dataset["close"] - dataset["MA_100"]) / dataset["close"]

    dataset["rsi_scaled"]         = dataset["RSI"] / 100.0
    dataset["rsi_change_6"]       = (dataset["RSI"] - dataset["RSI"].shift(6)) / 100.0
    dataset["macd_hist_pct"]      = (dataset["MACD"] - dataset["MACD_signal"]) / dataset["close"]
    dataset["price_acceleration"] = dataset["return_1"] - dataset["return_6"]

    dataset["atr_pct"]       = dataset["ATR"] / dataset["close"]
    dataset["volatility_24"] = dataset["VOL_24"]
    bb_range = dataset["BB_upper"] - dataset["BB_lower"]
    dataset["bb_position"]  = (dataset["close"] - dataset["BB_lower"]) / bb_range.replace(0, np.nan)
    dataset["bb_width_pct"] = bb_range / dataset["close"]

    dataset["volume_change"]   = dataset["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    dataset["volume_zscore"]   = (dataset["volume"] - dataset["VOL_MA_20"]) / dataset["VOL_STD_20"].replace(0, np.nan)
    dataset["volume_ma_ratio"] = dataset["volume"] / dataset["VOL_MA_20"].replace(0, np.nan)

    candle_range = (dataset["high"] - dataset["low"]).replace(0, np.nan)
    body = (dataset["close"] - dataset["open"]).abs()
    upper_wick = dataset["high"] - dataset[["open", "close"]].max(axis=1)
    lower_wick = dataset[["open", "close"]].min(axis=1) - dataset["low"]
    dataset["body_pct"]       = body / candle_range
    dataset["upper_wick_pct"] = upper_wick / candle_range
    dataset["lower_wick_pct"] = lower_wick / candle_range

    dataset["hour_of_day"] = dataset["timestamp"].dt.hour / 23.0
    dataset["day_of_week"] = dataset["timestamp"].dt.dayofweek / 6.0

    dataset["dist_from_high_20"] = (dataset["close"] - dataset["HIGH_20"]) / dataset["close"]
    dataset["dist_from_low_20"]  = (dataset["close"] - dataset["LOW_20"])  / dataset["close"]

    dataset["future_close"] = dataset["close"].shift(-horizon)
    dataset["target_up"] = (dataset["future_close"] > dataset["close"]).astype(int)

    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["future_close"]).reset_index(drop=True)
    return dataset


def run_ml_experiment(
    df: pd.DataFrame,
    horizon: int = DEFAULT_ML_HORIZON,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
) -> MLExperimentResult:
    dataset = build_ml_dataset(df, horizon=horizon)

    split_index = max(int(len(dataset) * train_ratio), 1)
    train = dataset.iloc[:split_index].copy()
    test = dataset.iloc[split_index:].copy()
    if test.empty:
        test = train.tail(min(32, len(train))).copy()

    train_x = train[FEATURE_COLUMNS]
    test_x  = test[FEATURE_COLUMNS]
    train_y = train["target_up"].to_numpy(dtype=float)
    test_y  = test["target_up"].to_numpy(dtype=int)

    pos_count = float((train_y == 1).sum())
    neg_count = float((train_y == 0).sum())
    scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

    base_pipeline = Pipeline([
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )),
    ])

    param_grid = {
        "clf__n_estimators":  [200, 400],
        "clf__max_depth":     [3, 5],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample":     [0.8],
        "clf__colsample_bytree": [0.8],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(train_x.to_numpy(dtype=float), train_y)
    pipeline = grid.best_estimator_
    logger.info(
        "GridSearchCV best params = %s (CV accuracy %.1f%%)",
        grid.best_params_,
        grid.best_score_ * 100,
    )

    probabilities = pipeline.predict_proba(test_x.to_numpy(dtype=float))[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = float((predictions == test_y).mean())
    baseline_rate = int(train_y.mean() >= 0.5)
    baseline_predictions = np.full_like(test_y, baseline_rate)
    baseline_accuracy = float((baseline_predictions == test_y).mean())

    prediction_frame = test[["timestamp", "close", "future_close", "target_up"]].copy()
    prediction_frame["prediction"]    = predictions
    prediction_frame["probability_up"] = probabilities
    prediction_frame["correct"] = prediction_frame["prediction"] == prediction_frame["target_up"]

    importances = pipeline.named_steps["clf"].feature_importances_
    feature_importance = pd.DataFrame(
        {
            "feature":    FEATURE_COLUMNS,
            "weight":     importances,
            "abs_weight": np.abs(importances),
        }
    ).sort_values("abs_weight", ascending=False, ignore_index=True)

    train_cutoff_ts = pd.Timestamp(test["timestamp"].iloc[0])

    model = MLModel(
        pipeline=pipeline,
        feature_columns=FEATURE_COLUMNS,
        train_cutoff_timestamp=train_cutoff_ts,
    )

    metrics = {
        "train_rows":          float(len(train)),
        "test_rows":           float(len(test)),
        "accuracy":            accuracy,
        "baseline_accuracy":   baseline_accuracy,
        "positive_rate_test":  float(test_y.mean()),
    }

    logger.info(
        "ML experiment — accuracy: %.1f%% | baseline: %.1f%% | train cutoff: %s",
        accuracy * 100,
        baseline_accuracy * 100,
        train_cutoff_ts.date(),
    )

    return MLExperimentResult(
        metrics=metrics,
        prediction_frame=prediction_frame,
        feature_importance=feature_importance,
        model=model,
    )
