from __future__ import annotations  # enables newer type-hint syntax on Python 3.9 and earlier

import logging  # Python's built-in logging library — structured timestamped messages instead of print()
from dataclasses import dataclass  # dataclass decorator — auto-generates __init__ and __repr__ so we only list fields

import numpy as np  # NumPy: fast numerical array library — used for array operations and math
import pandas as pd  # pandas: data-table library — pd.DataFrame, pd.Timestamp

from sklearn.linear_model import LogisticRegression  # LogisticRegression: sklearn's production-grade binary classifier — replaces the scratch implementation; handles regularisation, convergence checks, and numerical stability automatically
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # GridSearchCV: tries every combination of hyperparameters and picks the best; TimeSeriesSplit: cross-validation that respects time order — never trains on data that comes after the test set
from sklearn.pipeline import Pipeline  # Pipeline: chains multiple sklearn steps (e.g. scaler → model) into one object so fit/predict applies both steps in the correct order
from sklearn.preprocessing import StandardScaler  # StandardScaler: subtracts the mean and divides by the std of each feature — puts all features on the same scale so no single feature dominates

from quant_bot.config import DEFAULT_ML_HORIZON, DEFAULT_TRAIN_RATIO  # the two default values this module needs from config

logger = logging.getLogger(__name__)  # create a logger named "quant_bot.ml" so log messages show which file they came from

# Single source of truth for the 7 feature names — used by both training and the live
# backtest so they can never accidentally use different feature sets.
FEATURE_COLUMNS: list[str] = [  # list of the 11 input feature names the model is trained on (expanded from 7 to 11 for better predictive power)
    "return_1",       # 1-candle price return (how much price moved in the last 1 hour)
    "return_6",       # 6-candle price return
    "return_24",      # 24-candle price return (roughly one day's momentum)
    "ma_gap_short",   # gap between MA10 and MA30 relative to price — measures short-term momentum
    "ma_gap_trend",   # distance of price above/below MA200 — measures long-term trend position
    "volume_change",  # percentage change in trading volume vs the previous candle
    "rsi_scaled",     # RSI divided by 100 — puts RSI on a 0-1 scale matching the other features
    "macd_hist_pct",  # NEW: MACD - MACD_signal as a fraction of price — captures momentum divergence
    "bb_position",    # NEW: where price sits within the Bollinger bands (0 = at lower band, 1 = at upper band)
    "atr_pct",        # NEW: ATR divided by price — current volatility as a percentage of price
    "hour_of_day",    # NEW: hour of the day scaled to 0-1 — captures intraday seasonality (e.g. US market open volatility)
]


# ── Trained model wrapper ─────────────────────────────────────────────────────

@dataclass  # auto-generates __init__ and __repr__
class MLModel:  # lightweight container that holds the trained sklearn pipeline and metadata needed for live prediction
    pipeline: Pipeline  # the fitted sklearn Pipeline (StandardScaler + LogisticRegression) — calling .predict_proba() on it handles scaling and prediction in one step
    feature_columns: list[str]  # the ordered list of feature names the model expects — must match what compute_features_at_row() returns
    train_cutoff_timestamp: pd.Timestamp  # the first timestamp that belongs to the test split — the backtester only applies ML gating on candles at or after this date to avoid lookahead bias

    def predict_proba(self, features: dict[str, float]) -> float:  # given a dict of feature values, return P(price goes up) as a float between 0 and 1
        x = np.array(  # np.array(): convert a Python list into a NumPy array — sklearn requires NumPy input
            [[features.get(col, 0.0) for col in self.feature_columns]]  # list comprehension builds a 1-row, 7-column 2D array; .get(col, 0.0): look up each feature, defaulting to 0.0 if it's missing
        )
        return float(self.pipeline.predict_proba(x)[0, 1])  # pipeline.predict_proba(): run the input through the scaler then the classifier; returns shape (1,2) — column 0 is P(down), column 1 is P(up); [0,1]: grab P(up) from the first (only) row; float(): convert numpy scalar to plain Python float


# ── Row-level feature computation (called by the backtester) ─────────────────

def compute_features_at_row(df: pd.DataFrame, i: int) -> dict[str, float] | None:  # compute all 7 ML features for the candle at index i using only data from rows 0 to i (no future peeking)
    if i < 24:  # we need 24 rows of history to compute return_24 — return None for any row that doesn't have enough past data
        return None  # None signals to the backtester "skip the ML gate this candle"

    close = df["close"].iloc[i]  # .iloc[i]: integer-location indexing — get the close price at row i
    close_1 = df["close"].iloc[i - 1]  # close price one candle ago
    close_6 = df["close"].iloc[i - 6]  # close price six candles ago
    close_24 = df["close"].iloc[i - 24]  # close price 24 candles ago
    vol = df["volume"].iloc[i]  # trading volume at the current candle
    vol_1 = df["volume"].iloc[i - 1]  # trading volume one candle ago
    ma10 = df["MA_10"].iloc[i]  # 10-period moving average at the current candle
    ma30 = df["MA_30"].iloc[i]  # 30-period moving average at the current candle
    ma200 = df["MA_200"].iloc[i]  # 200-period moving average at the current candle
    rsi = df["RSI"].iloc[i]  # RSI value at the current candle
    macd = df["MACD"].iloc[i]  # MACD line at the current candle (fast EMA - slow EMA)
    macd_sig = df["MACD_signal"].iloc[i]  # MACD signal line (9-period EMA of MACD)
    bb_up = df["BB_upper"].iloc[i]  # upper Bollinger band
    bb_lo = df["BB_lower"].iloc[i]  # lower Bollinger band
    atr = df["ATR"].iloc[i]  # Average True Range at the current candle
    ts = df["timestamp"].iloc[i]  # timestamp at the current candle — used for hour-of-day feature

    if any(pd.isna(v) for v in [close, ma10, ma30, ma200, rsi, macd, macd_sig, bb_up, bb_lo, atr]):  # check all values including new indicators — any NaN means we can't predict yet
        return None  # bail out early rather than propagating NaN into the model
    if close == 0 or close_1 == 0 or close_6 == 0 or close_24 == 0:  # guard against division-by-zero when computing percentage returns — price should never be 0 in real data but we check anyway
        return None  # return None so the backtester skips this candle's ML gate
    bb_range = bb_up - bb_lo  # width of the Bollinger band envelope — used as the denominator for bb_position
    if bb_range == 0:  # if bands have collapsed to a single line (zero volatility), skip — would cause division by zero
        return None

    return {  # build and return the feature dictionary — all values are pure backward-looking calculations
        "return_1":     (close - close_1) / close_1,   # (current - past) / past = fractional price change over 1 candle
        "return_6":     (close - close_6) / close_6,   # fractional price change over 6 candles
        "return_24":    (close - close_24) / close_24, # fractional price change over 24 candles (approx one day)
        "ma_gap_short": (ma10 - ma30) / close,         # how far the fast MA is above/below the slow MA, scaled by price; positive = upward momentum
        "ma_gap_trend": (close - ma200) / close,       # how far price is above/below the long-term average; positive = bullish long-term trend
        "volume_change": (vol - vol_1) / vol_1 if vol_1 != 0 else 0.0,  # percentage change in volume vs previous candle; guard against zero divisor
        "rsi_scaled":   rsi / 100.0,                   # divide RSI (0-100) by 100 to put it on the same 0-1 scale as the other features
        "macd_hist_pct": (macd - macd_sig) / close,    # NEW: MACD histogram (= MACD - signal line) normalised by price — positive = bullish momentum acceleration
        "bb_position":  (close - bb_lo) / bb_range,    # NEW: position within Bollinger bands; 0 = at lower band (oversold), 1 = at upper band (overbought)
        "atr_pct":      atr / close,                   # NEW: volatility as a fraction of price — high values = market is moving violently
        "hour_of_day":  ts.hour / 23.0,                # NEW: hour of day (0–23) scaled to 0-1 — captures recurring intraday patterns like US market open
    }


# ── ML experiment result ──────────────────────────────────────────────────────

@dataclass  # auto-generates __init__ so we just list fields
class MLExperimentResult:  # container for everything produced by run_ml_experiment()
    metrics: dict[str, float]  # performance numbers: accuracy, baseline_accuracy, train_rows, test_rows, positive_rate_test
    prediction_frame: pd.DataFrame  # test-set rows with columns: timestamp, close, future_close, target_up, prediction, probability_up, correct
    feature_importance: pd.DataFrame  # DataFrame with columns: feature, weight, abs_weight — sorted by abs_weight descending
    model: MLModel  # the trained MLModel object — passed to the backtester so it can gate entries with the ML signal


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_ml_dataset(df: pd.DataFrame, horizon: int = DEFAULT_ML_HORIZON) -> pd.DataFrame:  # turn a raw OHLCV+indicator DataFrame into a supervised ML dataset
    dataset = df.copy()  # copy(): work on an independent copy so we don't modify the original DataFrame

    dataset["return_1"]  = dataset["close"].pct_change(1)   # pct_change(1): (current - previous) / previous for every row — 1-candle percentage return
    dataset["return_6"]  = dataset["close"].pct_change(6)   # pct_change(6): percentage change vs 6 candles ago
    dataset["return_24"] = dataset["close"].pct_change(24)  # pct_change(24): percentage change vs 24 candles ago — approx one day
    dataset["ma_gap_short"] = (dataset["MA_10"] - dataset["MA_30"]) / dataset["close"]  # short-term momentum: gap between fast and slow MA relative to price
    dataset["ma_gap_trend"] = (dataset["close"] - dataset["MA_200"]) / dataset["close"]  # long-term trend position: how far price sits above/below the 200-period MA
    dataset["volume_change"] = dataset["volume"].pct_change().replace([np.inf, -np.inf], np.nan)  # volume percentage change; .replace(): swap inf/-inf with NaN — happens when previous volume was 0
    dataset["rsi_scaled"] = dataset["RSI"] / 100.0  # scale RSI from 0-100 down to 0-1 to match the other features
    dataset["macd_hist_pct"] = (dataset["MACD"] - dataset["MACD_signal"]) / dataset["close"]  # NEW: MACD histogram divided by price — captures momentum acceleration
    bb_range = dataset["BB_upper"] - dataset["BB_lower"]  # band width — denominator for bb_position
    dataset["bb_position"] = (dataset["close"] - dataset["BB_lower"]) / bb_range.replace(0, np.nan)  # NEW: where price is within bands; .replace(0, nan): avoid division by zero if bands collapse
    dataset["atr_pct"] = dataset["ATR"] / dataset["close"]  # NEW: volatility as a fraction of price
    dataset["hour_of_day"] = dataset["timestamp"].dt.hour / 23.0  # NEW: .dt.hour: extract hour from datetime column (0-23); /23.0: scale to 0-1

    dataset["future_close"] = dataset["close"].shift(-horizon)  # shift(-horizon): move the close column `horizon` rows UP — so row i now shows the closing price `horizon` candles in the future; this is the prediction target
    dataset["target_up"] = (dataset["future_close"] > dataset["close"]).astype(int)  # create the binary label: 1 if price will be higher in `horizon` candles, 0 if lower; .astype(int): convert True/False to 1/0

    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["future_close"]).reset_index(drop=True)  # dropna(): remove rows where any feature or the future_close is NaN — happens for the first ~200 rows (MA warmup) and the last `horizon` rows (no future price exists); reset_index(drop=True): renumber rows 0,1,2,...
    return dataset  # return the cleaned supervised dataset ready for ML training


# ── Main ML function ──────────────────────────────────────────────────────────

def run_ml_experiment(  # train a logistic-regression model and evaluate it on a held-out test split
    df: pd.DataFrame,  # the full price+indicator DataFrame from get_price_data()
    horizon: int = DEFAULT_ML_HORIZON,  # how many candles ahead to predict
    train_ratio: float = DEFAULT_TRAIN_RATIO,  # fraction of data used for training (80%)
) -> MLExperimentResult:  # returns a container with metrics, predictions, feature importance, and the trained model
    dataset = build_ml_dataset(df, horizon=horizon)  # build the supervised dataset (features + target labels)

    split_index = max(int(len(dataset) * train_ratio), 1)  # int(len * 0.8): the row number at the 80% mark; max(..., 1): ensure at least 1 training row to prevent empty arrays
    train = dataset.iloc[:split_index].copy()  # .iloc[:split_index]: all rows UP TO the split point — chronologically the earlier (older) data; .copy(): independent copy so changes don't affect `dataset`
    test = dataset.iloc[split_index:].copy()  # .iloc[split_index:]: all rows AFTER the split — newer data the model has never seen
    if test.empty:  # edge case: if the dataset is tiny, the test split might be empty
        test = train.tail(min(32, len(train))).copy()  # .tail(n): take the last n rows of train as a fallback test set; min(32, len(train)): don't ask for more rows than exist

    train_x = train[FEATURE_COLUMNS]  # select only the 7 feature columns from the training rows — these are the model inputs
    test_x  = test[FEATURE_COLUMNS]   # same for the test rows
    train_y = train["target_up"].to_numpy(dtype=float)  # .to_numpy(): convert the pandas Series to a NumPy array — sklearn requires arrays not Series; dtype=float: ensure numeric type
    test_y  = test["target_up"].to_numpy(dtype=int)    # test labels as integers (0 or 1) — used for accuracy calculation

    # ── Tuned logistic regression via GridSearchCV + TimeSeriesSplit ──────────
    # Improvements over the previous fixed-C version:
    #   1. class_weight="balanced" — weights samples inversely to class frequency, helping when the up/down split is not 50/50
    #   2. GridSearchCV — tries multiple C (regularisation) values and picks the best
    #   3. TimeSeriesSplit — cross-validation that respects time order (never trains on a fold AFTER its test fold), unlike normal CV which would leak future data
    base_pipeline = Pipeline([  # Pipeline: chain preprocessing and the classifier into a single object
        ("scaler", StandardScaler()),  # step 1 — StandardScaler: subtract the mean and divide by std for each feature so they're all on the same scale
        ("clf",    LogisticRegression(  # step 2 — sklearn's production logistic regression
            max_iter=2000,              # allow up to 2000 iterations to converge — needed because larger C values require more iterations
            random_state=42,            # fixes the random seed so results are reproducible
            class_weight="balanced",    # NEW: weights each class inversely proportional to its frequency — prevents the model from just predicting the majority class
            solver="lbfgs",             # lbfgs: a fast quasi-Newton optimiser, good default for small/medium datasets
        )),
    ])
    param_grid = {  # the grid of hyperparameter values to try — GridSearchCV will train one model per combination
        "clf__C": [0.01, 0.1, 1.0, 10.0],  # clf__C: the "C" parameter of the "clf" step in the pipeline; tries 4 different regularisation strengths (low C = more regularisation = simpler model)
    }
    tscv = TimeSeriesSplit(n_splits=5)  # TimeSeriesSplit(n_splits=5): split the training data into 5 chronological folds; each fold's test set comes AFTER its training set — preserves time ordering
    grid = GridSearchCV(  # GridSearchCV: exhaustively tries each parameter combination using cross-validation, picks the one with the best average score
        estimator=base_pipeline,        # the pipeline to tune
        param_grid=param_grid,          # the values to try
        cv=tscv,                        # use the time-series CV instead of random k-fold (which would leak future data)
        scoring="accuracy",             # pick the C that maximises classification accuracy across the 5 folds
        n_jobs=-1,                      # n_jobs=-1: use all CPU cores in parallel — much faster
    )
    grid.fit(train_x.to_numpy(dtype=float), train_y)  # .fit(): runs the grid search — trains 4 (C values) × 5 (folds) = 20 small models, picks the best C
    pipeline = grid.best_estimator_  # .best_estimator_: the pipeline retrained on the FULL training set using the best C found by the grid search
    logger.info("GridSearchCV best C = %s (CV accuracy %.1f%%)", grid.best_params_["clf__C"], grid.best_score_ * 100)  # log which C won and how well it scored in cross-validation

    probabilities = pipeline.predict_proba(test_x.to_numpy(dtype=float))[:, 1]  # pipeline.predict_proba(): run test data through scaler then classifier; returns shape (n_rows, 2) — column 0 is P(down), column 1 is P(up); [:, 1]: take all rows, column 1 = P(price goes up)
    predictions = (probabilities >= 0.5).astype(int)  # threshold at 0.5: probability ≥ 0.5 → predict 1 (up), below 0.5 → predict 0 (down); .astype(int): convert boolean array to 0/1

    accuracy = float((predictions == test_y).mean())  # (predictions == test_y): element-wise True/False array; .mean(): fraction that are True = accuracy score; float(): convert numpy scalar to Python float
    baseline_rate = int(train_y.mean() >= 0.5)  # always-predict-the-majority-class baseline: if more than half of training labels are 1, always predict 1; otherwise always predict 0
    baseline_predictions = np.full_like(test_y, baseline_rate)  # np.full_like(): create an array with the same shape as test_y, filled entirely with the baseline constant
    baseline_accuracy = float((baseline_predictions == test_y).mean())  # accuracy of the do-nothing baseline — our model must beat this to be meaningful

    prediction_frame = test[["timestamp", "close", "future_close", "target_up"]].copy()  # start with the key columns from the test set for inspection
    prediction_frame["prediction"]    = predictions   # add the model's 0/1 prediction for each test row
    prediction_frame["probability_up"] = probabilities  # add the raw probability (e.g. 0.62 = 62% chance of going up)
    prediction_frame["correct"] = prediction_frame["prediction"] == prediction_frame["target_up"]  # True where prediction matched the actual outcome — for easy analysis

    weights = pipeline.named_steps["clf"].coef_[0]  # .named_steps["clf"]: access the LogisticRegression step by its name; .coef_[0]: the 7 learned weights (one per feature); [0] because coef_ is 2D for binary classification
    feature_importance = pd.DataFrame(  # pd.DataFrame(): build a table from a dict of equal-length lists
        {
            "feature":    FEATURE_COLUMNS,  # column of feature names
            "weight":     weights,          # column of raw signed weights — positive = model thinks this feature predicts price going up
            "abs_weight": np.abs(weights),  # np.abs(): absolute value of each weight — used for ranking without direction
        }
    ).sort_values("abs_weight", ascending=False, ignore_index=True)  # sort_values(): reorder rows by abs_weight largest first so most important features appear at the top; ignore_index=True: renumber rows 0,1,2,...

    train_cutoff_ts = pd.Timestamp(test["timestamp"].iloc[0])  # pd.Timestamp(): ensure we have a proper Timestamp object; .iloc[0]: the first row of the test set — this is the exact point where training ends and testing begins

    model = MLModel(  # assemble the trained model wrapper
        pipeline=pipeline,                         # the fitted sklearn Pipeline (scaler + classifier)
        feature_columns=FEATURE_COLUMNS,           # the 7 feature names
        train_cutoff_timestamp=train_cutoff_ts,    # the date from which the backtester may apply ML gating
    )

    metrics = {  # collect all evaluation numbers into a plain dict
        "train_rows":          float(len(train)),       # how many rows were used for training
        "test_rows":           float(len(test)),        # how many rows were used for evaluation
        "accuracy":            accuracy,                # fraction of test predictions that were correct
        "baseline_accuracy":   baseline_accuracy,       # accuracy of always-predicting the majority class
        "positive_rate_test":  float(test_y.mean()),    # fraction of test labels that are 1 (price went up)
    }

    logger.info(  # log a one-line summary at INFO level — visible in the terminal when running bot.py
        "ML experiment — accuracy: %.1f%% | baseline: %.1f%% | train cutoff: %s",
        accuracy * 100,          # convert 0.53 to 53.0 for readability
        baseline_accuracy * 100, # convert 0.516 to 51.6 for readability
        train_cutoff_ts.date(),  # .date(): show just the date part, not the full timestamp
    )

    return MLExperimentResult(  # pack everything into the result container and return it to the caller
        metrics=metrics,
        prediction_frame=prediction_frame,
        feature_importance=feature_importance,
        model=model,
    )
