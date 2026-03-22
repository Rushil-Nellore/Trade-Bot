from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_bot.config import DEFAULT_ML_HORIZON, DEFAULT_TRAIN_RATIO


@dataclass
class MLExperimentResult:
    metrics: dict[str, float]
    prediction_frame: pd.DataFrame
    feature_importance: pd.DataFrame


def build_ml_dataset(df: pd.DataFrame, horizon: int = DEFAULT_ML_HORIZON) -> pd.DataFrame:
    dataset = df.copy()
    dataset["return_1"] = dataset["close"].pct_change(1)
    dataset["return_6"] = dataset["close"].pct_change(6)
    dataset["return_24"] = dataset["close"].pct_change(24)
    dataset["ma_gap_short"] = (dataset["MA_10"] - dataset["MA_30"]) / dataset["close"]
    dataset["ma_gap_trend"] = (dataset["close"] - dataset["MA_200"]) / dataset["close"]
    dataset["volume_change"] = dataset["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    dataset["rsi_scaled"] = dataset["RSI"] / 100.0
    dataset["future_close"] = dataset["close"].shift(-horizon)
    dataset["target_up"] = (dataset["future_close"] > dataset["close"]).astype(int)

    feature_columns = [
        "return_1",
        "return_6",
        "return_24",
        "ma_gap_short",
        "ma_gap_trend",
        "volume_change",
        "rsi_scaled",
    ]
    dataset = dataset.dropna(subset=feature_columns + ["future_close"]).reset_index(drop=True)
    return dataset


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logistic_regression(
    features: np.ndarray,
    targets: np.ndarray,
    learning_rate: float = 0.1,
    epochs: int = 500,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(features.shape[1], dtype=float)
    bias = 0.0
    sample_count = max(len(features), 1)

    for _ in range(epochs):
        linear_output = features @ weights + bias
        predictions = _sigmoid(linear_output)
        errors = predictions - targets
        weights -= learning_rate * (features.T @ errors) / sample_count
        bias -= learning_rate * errors.mean()

    return weights, bias


def run_ml_experiment(
    df: pd.DataFrame,
    horizon: int = DEFAULT_ML_HORIZON,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
) -> MLExperimentResult:
    dataset = build_ml_dataset(df, horizon=horizon)
    feature_columns = [
        "return_1",
        "return_6",
        "return_24",
        "ma_gap_short",
        "ma_gap_trend",
        "volume_change",
        "rsi_scaled",
    ]
    split_index = max(int(len(dataset) * train_ratio), 1)
    train = dataset.iloc[:split_index].copy()
    test = dataset.iloc[split_index:].copy()
    if test.empty:
        test = train.tail(min(32, len(train))).copy()

    train_x = train[feature_columns]
    test_x = test[feature_columns]
    mean = train_x.mean()
    std = train_x.std().replace(0, 1.0)
    train_scaled = ((train_x - mean) / std).to_numpy(dtype=float)
    test_scaled = ((test_x - mean) / std).to_numpy(dtype=float)
    train_y = train["target_up"].to_numpy(dtype=float)
    test_y = test["target_up"].to_numpy(dtype=int)

    weights, bias = _fit_logistic_regression(train_scaled, train_y)
    probabilities = _sigmoid(test_scaled @ weights + bias)
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = float((predictions == test_y).mean())
    baseline_rate = int(train_y.mean() >= 0.5)
    baseline_predictions = np.full_like(test_y, baseline_rate)
    baseline_accuracy = float((baseline_predictions == test_y).mean())

    prediction_frame = test[["timestamp", "close", "future_close", "target_up"]].copy()
    prediction_frame["prediction"] = predictions
    prediction_frame["probability_up"] = probabilities
    prediction_frame["correct"] = prediction_frame["prediction"] == prediction_frame["target_up"]

    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "weight": weights,
            "abs_weight": np.abs(weights),
        }
    ).sort_values("abs_weight", ascending=False, ignore_index=True)

    metrics = {
        "train_rows": float(len(train)),
        "test_rows": float(len(test)),
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "positive_rate_test": float(test_y.mean()),
    }
    return MLExperimentResult(
        metrics=metrics,
        prediction_frame=prediction_frame,
        feature_importance=feature_importance,
    )
