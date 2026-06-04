from __future__ import annotations  # enables newer type-hint syntax on Python 3.9 and earlier

import logging  # Python's built-in logging library — structured timestamped messages instead of print()
from dataclasses import dataclass  # dataclass decorator — auto-generates __init__ and __repr__ so we only list fields

import numpy as np  # NumPy: fast numerical array library — used for array operations and math
import pandas as pd  # pandas: data-table library — pd.DataFrame, pd.Timestamp

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # GridSearchCV: tries every combination of hyperparameters and picks the best; TimeSeriesSplit: cross-validation that respects time order — never trains on data that comes after the test set
from sklearn.pipeline import Pipeline  # Pipeline: chains multiple sklearn steps into one object so fit/predict applies them in the correct order
from xgboost import XGBClassifier  # XGBClassifier: eXtreme Gradient Boosting — an ensemble of decision trees that learns SEQUENTIALLY from previous trees' mistakes; replaces LogisticRegression for higher accuracy on non-linear patterns

from quant_bot.config import DEFAULT_ML_HORIZON, DEFAULT_TRAIN_RATIO  # the two default values this module needs from config

logger = logging.getLogger(__name__)  # create a logger named "quant_bot.ml" so log messages show which file they came from

# Single source of truth for the 7 feature names — used by both training and the live
# backtest so they can never accidentally use different feature sets.
FEATURE_COLUMNS: list[str] = [  # list of the 29 input feature names the model is trained on (expanded from 11 → 29 to give XGBoost more signal to learn from)
    # ── Returns at multiple horizons ────────────────────────────────────────
    "return_1",          # 1-candle (1-hour) percentage return
    "return_3",          # 3-candle return — short-term momentum
    "return_6",          # 6-candle return
    "return_12",         # 12-candle (half-day) return
    "return_24",         # 24-candle (1-day) return
    "return_72",         # 72-candle (3-day) return — captures multi-day trends
    # ── Trend position (multiple MAs) ───────────────────────────────────────
    "ma_gap_short",      # MA10 vs MA30 — short-term trend
    "ma_gap_50_100",     # MA50 vs MA100 — medium-term trend
    "ma_gap_trend",      # price vs MA200 — long-term trend
    "ma_50_dist",        # price vs MA50 — distance from medium-term anchor
    "ma_100_dist",       # price vs MA100 — distance from medium-long anchor
    # ── Momentum oscillators ────────────────────────────────────────────────
    "rsi_scaled",        # RSI / 100 (0-1 scale)
    "rsi_change_6",      # change in RSI over the last 6 candles — RSI acceleration
    "macd_hist_pct",     # MACD histogram (MACD − signal) as fraction of price
    "price_acceleration",# return_1 − return_6 — is momentum building (positive) or fading (negative)?
    # ── Volatility ──────────────────────────────────────────────────────────
    "atr_pct",           # ATR / price — current volatility as % of price
    "volatility_24",     # std of last 24 hourly returns — choppy vs calm regime detector
    "bb_position",       # position within Bollinger Bands (0 = lower, 1 = upper)
    "bb_width_pct",      # band width / price — volatility expansion / squeeze
    # ── Volume ──────────────────────────────────────────────────────────────
    "volume_change",     # % change in volume vs previous candle
    "volume_zscore",     # (vol − vol_mean) / vol_std over last 20 candles — catches volume spikes
    "volume_ma_ratio",   # vol / 20-period avg vol — relative volume
    # ── Candle shape / microstructure ───────────────────────────────────────
    "body_pct",          # |close − open| / (high − low) — how decisive the candle was
    "upper_wick_pct",    # upper wick size / range — rejection at highs (bearish signal)
    "lower_wick_pct",    # lower wick size / range — rejection at lows (bullish signal)
    # ── Time / seasonality ─────────────────────────────────────────────────
    "hour_of_day",       # hour of day (0-23) / 23 — intraday patterns
    "day_of_week",       # day of week (0=Mon … 6=Sun) / 6 — weekly patterns
    # ── Trend persistence / range position ──────────────────────────────────
    "dist_from_high_20", # (close − rolling 20-period high) / close — distance below recent peak
    "dist_from_low_20",  # (close − rolling 20-period low) / close — distance above recent trough
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

    if i < 72:  # need 72 candles of history for return_72 (the longest lookback feature)
        return None

    # ── Price snapshots at various lookback offsets ──────────────────────────
    close    = df["close"].iloc[i]
    close_1  = df["close"].iloc[i - 1]
    close_3  = df["close"].iloc[i - 3]
    close_6  = df["close"].iloc[i - 6]
    close_12 = df["close"].iloc[i - 12]
    close_24 = df["close"].iloc[i - 24]
    close_72 = df["close"].iloc[i - 72]

    # ── OHLCV at the current candle ──────────────────────────────────────────
    open_  = df["open"].iloc[i]   # trailing _ to avoid shadowing Python's built-in open()
    high   = df["high"].iloc[i]
    low    = df["low"].iloc[i]
    vol    = df["volume"].iloc[i]
    vol_1  = df["volume"].iloc[i - 1]

    # ── Indicators at the current candle ─────────────────────────────────────
    ma10   = df["MA_10"].iloc[i]
    ma30   = df["MA_30"].iloc[i]
    ma50   = df["MA_50"].iloc[i]
    ma100  = df["MA_100"].iloc[i]
    ma200  = df["MA_200"].iloc[i]
    rsi    = df["RSI"].iloc[i]
    rsi_6  = df["RSI"].iloc[i - 6]  # RSI 6 candles ago — used to compute rsi_change_6
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

    # ── NaN guard: bail out if ANY required value is missing ────────────────
    required = [close, open_, high, low, ma10, ma30, ma50, ma100, ma200, rsi, rsi_6,
                macd, macd_sig, bb_up, bb_lo, atr, vol_24, vol_ma, vol_std, high_20, low_20]
    if any(pd.isna(v) for v in required):
        return None
    # ── Division-by-zero guards ─────────────────────────────────────────────
    if close == 0 or close_1 == 0 or close_3 == 0 or close_6 == 0 or close_12 == 0 or close_24 == 0 or close_72 == 0:
        return None
    bb_range = bb_up - bb_lo  # Bollinger band width — denominator for bb_position
    candle_range = high - low  # full candle range — denominator for body / wick percentages
    if bb_range == 0 or candle_range == 0 or vol_std == 0:  # any of these being zero would cause a division-by-zero
        return None

    return_1 = (close - close_1) / close_1  # store these two so we can use them below for price_acceleration
    return_6 = (close - close_6) / close_6
    body = abs(close - open_)  # candle body size = |close − open|
    upper_wick = high - max(open_, close)  # upper wick = high minus the higher of open/close
    lower_wick = min(open_, close) - low  # lower wick = the lower of open/close minus low

    return {
        # ── Returns at multiple horizons ─────────────────────────────────────
        "return_1":  return_1,
        "return_3":  (close - close_3) / close_3,
        "return_6":  return_6,
        "return_12": (close - close_12) / close_12,
        "return_24": (close - close_24) / close_24,
        "return_72": (close - close_72) / close_72,
        # ── Trend position ───────────────────────────────────────────────────
        "ma_gap_short":  (ma10 - ma30) / close,
        "ma_gap_50_100": (ma50 - ma100) / close,
        "ma_gap_trend":  (close - ma200) / close,
        "ma_50_dist":    (close - ma50) / close,
        "ma_100_dist":   (close - ma100) / close,
        # ── Momentum ─────────────────────────────────────────────────────────
        "rsi_scaled":         rsi / 100.0,
        "rsi_change_6":       (rsi - rsi_6) / 100.0,  # divide by 100 to keep on same scale as rsi_scaled
        "macd_hist_pct":      (macd - macd_sig) / close,
        "price_acceleration": return_1 - return_6,    # is momentum strengthening (positive) or weakening (negative)?
        # ── Volatility ───────────────────────────────────────────────────────
        "atr_pct":       atr / close,
        "volatility_24": vol_24,
        "bb_position":   (close - bb_lo) / bb_range,
        "bb_width_pct":  bb_range / close,
        # ── Volume ───────────────────────────────────────────────────────────
        "volume_change":   (vol - vol_1) / vol_1 if vol_1 != 0 else 0.0,
        "volume_zscore":   (vol - vol_ma) / vol_std,
        "volume_ma_ratio": vol / vol_ma if vol_ma != 0 else 1.0,
        # ── Candle shape ─────────────────────────────────────────────────────
        "body_pct":       body / candle_range,
        "upper_wick_pct": upper_wick / candle_range,
        "lower_wick_pct": lower_wick / candle_range,
        # ── Time / seasonality ──────────────────────────────────────────────
        "hour_of_day":  ts.hour / 23.0,
        "day_of_week":  ts.dayofweek / 6.0,  # .dayofweek: 0=Monday, 6=Sunday; divide by 6 to scale 0-1
        # ── Trend persistence ───────────────────────────────────────────────
        "dist_from_high_20": (close - high_20) / close,  # always ≤ 0 (or 0 if at peak) — how far below the recent peak
        "dist_from_low_20":  (close - low_20) / close,   # always ≥ 0 (or 0 if at trough) — how far above the recent trough
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

    # ── Returns at multiple horizons ─────────────────────────────────────────
    dataset["return_1"]  = dataset["close"].pct_change(1)   # pct_change(n): fractional change from n rows ago to current
    dataset["return_3"]  = dataset["close"].pct_change(3)
    dataset["return_6"]  = dataset["close"].pct_change(6)
    dataset["return_12"] = dataset["close"].pct_change(12)
    dataset["return_24"] = dataset["close"].pct_change(24)
    dataset["return_72"] = dataset["close"].pct_change(72)

    # ── Trend position ──────────────────────────────────────────────────────
    dataset["ma_gap_short"]  = (dataset["MA_10"] - dataset["MA_30"]) / dataset["close"]
    dataset["ma_gap_50_100"] = (dataset["MA_50"] - dataset["MA_100"]) / dataset["close"]
    dataset["ma_gap_trend"]  = (dataset["close"] - dataset["MA_200"]) / dataset["close"]
    dataset["ma_50_dist"]    = (dataset["close"] - dataset["MA_50"])  / dataset["close"]
    dataset["ma_100_dist"]   = (dataset["close"] - dataset["MA_100"]) / dataset["close"]

    # ── Momentum oscillators ────────────────────────────────────────────────
    dataset["rsi_scaled"]         = dataset["RSI"] / 100.0
    dataset["rsi_change_6"]       = (dataset["RSI"] - dataset["RSI"].shift(6)) / 100.0  # .shift(6): RSI from 6 rows ago — change in RSI scaled to 0-1
    dataset["macd_hist_pct"]      = (dataset["MACD"] - dataset["MACD_signal"]) / dataset["close"]
    dataset["price_acceleration"] = dataset["return_1"] - dataset["return_6"]  # is momentum building or fading?

    # ── Volatility ──────────────────────────────────────────────────────────
    dataset["atr_pct"]       = dataset["ATR"] / dataset["close"]
    dataset["volatility_24"] = dataset["VOL_24"]
    bb_range = dataset["BB_upper"] - dataset["BB_lower"]
    dataset["bb_position"]  = (dataset["close"] - dataset["BB_lower"]) / bb_range.replace(0, np.nan)
    dataset["bb_width_pct"] = bb_range / dataset["close"]

    # ── Volume features ─────────────────────────────────────────────────────
    dataset["volume_change"]   = dataset["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    dataset["volume_zscore"]   = (dataset["volume"] - dataset["VOL_MA_20"]) / dataset["VOL_STD_20"].replace(0, np.nan)  # z-score: how unusual is current volume
    dataset["volume_ma_ratio"] = dataset["volume"] / dataset["VOL_MA_20"].replace(0, np.nan)  # relative volume vs 20-period average

    # ── Candle shape ────────────────────────────────────────────────────────
    candle_range = (dataset["high"] - dataset["low"]).replace(0, np.nan)  # full candle range; .replace(0, nan): avoid divide-by-zero on dojis
    body = (dataset["close"] - dataset["open"]).abs()  # absolute body size
    upper_wick = dataset["high"] - dataset[["open", "close"]].max(axis=1)  # .max(axis=1): row-wise max of open and close
    lower_wick = dataset[["open", "close"]].min(axis=1) - dataset["low"]   # .min(axis=1): row-wise min of open and close
    dataset["body_pct"]       = body / candle_range
    dataset["upper_wick_pct"] = upper_wick / candle_range
    dataset["lower_wick_pct"] = lower_wick / candle_range

    # ── Time features ───────────────────────────────────────────────────────
    dataset["hour_of_day"] = dataset["timestamp"].dt.hour / 23.0       # .dt.hour: extract hour 0-23 then scale to 0-1
    dataset["day_of_week"] = dataset["timestamp"].dt.dayofweek / 6.0   # .dt.dayofweek: 0=Mon … 6=Sun then scale to 0-1

    # ── Trend persistence ───────────────────────────────────────────────────
    dataset["dist_from_high_20"] = (dataset["close"] - dataset["HIGH_20"]) / dataset["close"]  # how far below the rolling 20-period high
    dataset["dist_from_low_20"]  = (dataset["close"] - dataset["LOW_20"])  / dataset["close"]  # how far above the rolling 20-period low

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

    # ── XGBoost classifier — tuned via GridSearchCV + TimeSeriesSplit ─────────
    # WHY XGBoost over logistic regression?
    #   1. Captures NON-LINEAR patterns (e.g. "RSI > 70 AND volume spiking" together — a tree can split on both)
    #   2. Captures FEATURE INTERACTIONS automatically (logistic regression only sums each feature independently)
    #   3. Trees are scale-invariant — no StandardScaler needed; trees split on raw feature values
    #   4. Boosting: each new tree corrects the mistakes (gradient errors) of all previous trees
    #
    # Handling class imbalance:
    #   scale_pos_weight = count(negative class) / count(positive class)
    #   tells XGBoost to weight the minority class higher when computing the loss gradient
    pos_count = float((train_y == 1).sum())  # count training examples labelled 1 (price went up)
    neg_count = float((train_y == 0).sum())  # count training examples labelled 0 (price went down or flat)
    scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0  # ratio: if negatives are more common, weight positives proportionally higher

    base_pipeline = Pipeline([  # Pipeline: keeps the same interface as before so MLModel.predict_proba() does not need to change
        ("clf", XGBClassifier(  # XGBClassifier: gradient-boosted decision trees for binary classification
            objective="binary:logistic",  # objective: tells XGBoost we're doing binary classification with logistic loss (outputs probabilities)
            eval_metric="logloss",        # eval_metric: how XGBoost measures error during training — logloss is the natural choice for probability outputs
            random_state=42,              # fixes the random seed so results are reproducible across runs
            scale_pos_weight=scale_pos_weight,  # class-imbalance correction — equivalent to logistic regression's class_weight="balanced"
            n_jobs=-1,                    # use all CPU cores for training a single model — much faster
            tree_method="hist",           # tree_method="hist": fast histogram-based tree builder; dramatically faster than "exact" on medium datasets
            verbosity=0,                  # silence XGBoost's own progress logs — we use our own logger.info() instead
        )),
    ])

    param_grid = {  # XGBoost has several knobs worth tuning — wider grid now that we have 29 features and more degrees of freedom
        "clf__n_estimators":  [200, 400],          # n_estimators: total number of sequential trees; more = better fit but slower and risks overfitting
        "clf__max_depth":     [3, 5],              # max_depth: maximum depth of each tree; shallow = simpler, less overfitting; deep = more complex feature interactions
        "clf__learning_rate": [0.05, 0.1],         # learning_rate (eta): how strongly each new tree's correction is applied; smaller = more conservative
        "clf__subsample":     [0.8],               # subsample: fraction of rows each tree sees (random); <1.0 adds regularisation by preventing trees from memorising the full dataset
        "clf__colsample_bytree": [0.8],            # colsample_bytree: fraction of FEATURES each tree sees; <1.0 forces diversity across trees, reduces overfitting on noisy features
    }
    tscv = TimeSeriesSplit(n_splits=5)  # TimeSeriesSplit(n_splits=5): 5 chronological folds — each fold's test set is strictly after its training set (no future data leakage)
    grid = GridSearchCV(  # GridSearchCV: tries every combination of the param_grid values
        estimator=base_pipeline,  # the pipeline to tune
        param_grid=param_grid,    # 2 × 2 × 2 = 8 combinations to try
        cv=tscv,                  # use time-series CV
        scoring="accuracy",       # pick the combination that maximises accuracy across the 5 folds
        n_jobs=-1,                # use all CPU cores in parallel across the 8 × 5 = 40 model fits
    )
    grid.fit(train_x.to_numpy(dtype=float), train_y)  # .fit(): runs the grid search — 40 small XGBoost fits, picks the best hyperparameter combination
    pipeline = grid.best_estimator_  # .best_estimator_: the pipeline retrained on the FULL training set using the winning hyperparameters
    logger.info(  # log which combination won and its CV accuracy
        "GridSearchCV best params = %s (CV accuracy %.1f%%)",
        grid.best_params_,
        grid.best_score_ * 100,
    )

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

    # XGBoost uses .feature_importances_ — measures how often each feature is used to split trees, weighted by the gain
    # (logistic regression used .coef_ — signed weights; XGBoost importances are always non-negative)
    importances = pipeline.named_steps["clf"].feature_importances_  # array of 11 non-negative importance scores, one per feature, summing to 1.0
    feature_importance = pd.DataFrame(  # pd.DataFrame(): build a table from a dict of equal-length lists
        {
            "feature":    FEATURE_COLUMNS,    # column of feature names
            "weight":     importances,        # column of XGBoost importance scores — bigger = more useful for splitting trees
            "abs_weight": np.abs(importances),  # np.abs(): absolute value (already non-negative for XGBoost but kept for schema compatibility)
        }
    ).sort_values("abs_weight", ascending=False, ignore_index=True)  # sort_values(): reorder rows by importance, largest first; ignore_index=True: renumber rows 0,1,2,...

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
