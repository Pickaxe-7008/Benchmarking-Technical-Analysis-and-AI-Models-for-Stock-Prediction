# =============================================================================
# OUTPUT CHEAT SHEET
# =============================================================================
#
# 1) Walk-forward CV (train-only):
#    Classification (task="clf"):
#      - ACC   : Accuracy averaged across CV folds
#      - PREC  : Precision
#      - REC   : Recall
#      - F1    : F1-score
#      - ROC   : ROC-AUC
#      → Evaluates stability of the model on training data splits.
#
#    Regression (task="reg"):
#      - MAE   : Mean Absolute Error (lower is better)
#      - RMSE  : Root Mean Squared Error (lower is better)
#      - R2    : Coefficient of Determination (higher is better)
#      - DIR_ACC: Directional accuracy on train folds (sign correct)
#      → Evaluates stability of the model on training data splits.
#
# 2) Holdout (last segment):
#    Classification:
#      - ACC, PREC, REC, F1, ROC (as above) → True out-of-sample performance.
#      - HIT_RATE: Same as ACC (clarity).
#
#    Regression:
#      - MAE, RMSE, R2 → True out-of-sample performance.
#      - DIR_ACC: Directional accuracy (sign(pred) == sign(true))
#
# 3) Top features:
#    - RandomForest (clf/reg) → "importance" = Gini/MSE-based importances
#    - Logistic (clf)         → "abs_coef"   = |regression coefficient|
#    - Linear/Ridge (reg)     → "abs_coef"   = |regression coefficient|
#    → Larger = stronger influence in the trained model.
#
# 4) Univariate (holdout):
#    - Classification → per-feature ROC-AUC when trained alone
#    - Regression     → per-feature R^2 on holdout when trained alone
#    → Measures standalone predictive power of each indicator.
#
# 5) Group permutation importance (holdout):
#    Classification:
#      - group, baseline_auc, shuffled_auc_mean, auc_drop
#    Regression:
#      - group, baseline_r2, shuffled_r2_mean, r2_drop
#    → Larger drop = more important family of indicators to the model.
#
# 6) Last 10 predictions (from holdout):
#    Classification:
#      - date, prev_close, y_true (0/1), y_prob_up, y_pred_up, pred_correct
#    Regression:
#      - date, prev_close, y_true (future return), y_pred, true_up, pred_up, correct_dir, abs_error
#    → Adds an easy-to-interpret correctness flag and previous day's close for context.
#
# =============================================================================

import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Regressors
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error
)

# ---------------------
# Config & CLI
# ---------------------

@dataclass
class Config:
    # Core setup
    ticker: str = "GOOG"           # Company symbol
    period_years: int = 5          # Fixed 5y history
    interval: str = "1d"           # Daily data
    horizon: int = 1               # Predict 1 day ahead
    test_ratio: float = 0.2        # Last 20% of data = holdout

    # TASK / MODEL:
    #   task="clf" with model in {"logit","rf"}
    #   task="reg" with model in {"linreg","ridge","rf"}
    task: str = "reg"
    model: str = "ridge"

    # Indicator windows
    sma_windows: List[int] = field(default_factory=lambda: [10, 20, 50, 200])
    ema_windows: List[int] = field(default_factory=lambda: [12, 26])
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0
    stoch_k: int = 14
    stoch_d: int = 3
    atr_window: int = 14
    obv: bool = True

    # Training setup
    n_splits: int = 5
    random_state: int = 42
    class_weight: str = "balanced" # For logistic regression


# ---------------------
# Data & Indicators
# ---------------------

def fetch_ohlcv(cfg: Config) -> pd.DataFrame:
    """
    Robust fetch that forces single-level columns and flattens MultiIndex if needed.
    """
    period = f"{cfg.period_years}y"
    df = yf.download(
        cfg.ticker,
        period=period,
        interval=cfg.interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True
    )

    if df.empty:
        # Fallback: try explicit start/end for robustness
        end = pd.Timestamp.today(tz="UTC").normalize()
        start = end - pd.DateOffset(years=cfg.period_years)
        df = yf.download(
            cfg.ticker,
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            interval=cfg.interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True
        )

    if df.empty:
        raise ValueError(f"No data returned for {cfg.ticker}.")

    # Handle MultiIndex (Price, Ticker) if present
    if isinstance(df.columns, pd.MultiIndex):
        levels = df.columns.names or []
        if "Ticker" in levels and cfg.ticker in df.columns.get_level_values("Ticker"):
            df = df.xs(cfg.ticker, axis=1, level="Ticker")
        else:
            df = df.droplevel(-1, axis=1)

    # Normalize names
    df = df.rename(columns=str.title)

    expected = ["Open", "High", "Low", "Close", "Volume"]
    got = [c for c in df.columns if c in expected]
    if len(got) < len(expected):
        raise KeyError(f"Missing expected columns: {sorted(set(expected) - set(got))}. Got: {df.columns.tolist()}")

    df = df.dropna(subset=expected)
    return df


def sma(series: pd.Series, w: int) -> pd.Series:
    return series.rolling(window=w, min_periods=w).mean()

def ema(series: pd.Series, w: int) -> pd.Series:
    return series.ewm(span=w, adjust=False, min_periods=w).mean()

def rsi(close: pd.Series, w: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/w, adjust=False).mean()
    roll_down = down.ewm(alpha=1/w, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    line = ema_fast - ema_slow
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = line - signal_line
    return line, signal_line, hist

def bollinger(close: pd.Series, w: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(window=w, min_periods=w).mean()
    sd = close.rolling(window=w, min_periods=w).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    return upper, ma, lower

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    low_k = low.rolling(window=k, min_periods=k).min()
    high_k = high.rolling(window=k, min_periods=k).max()
    pct_k = 100 * (close - low_k) / (high_k - low_k)
    pct_d = pct_k.rolling(window=d, min_periods=d).mean()
    return pct_k, pct_d

def atr(high: pd.Series, low: pd.Series, close: pd.Series, w: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=w, min_periods=w).mean()

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).fillna(0).cumsum()

def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    # Simple returns & volume change
    feats["RET_1"] = close.pct_change(1)
    feats["RET_5"] = close.pct_change(5)
    feats["VOL_CHG"] = vol.pct_change(5)

    # SMA/EMA relatives
    for w in cfg.sma_windows:
        sma_w = sma(close, w)
        feats[f"SMA_{w}"] = sma_w
        feats[f"SMA_{w}_pct"] = close / sma_w - 1.0
    for w in cfg.ema_windows:
        ema_w = ema(close, w)
        feats[f"EMA_{w}"] = ema_w
        feats[f"EMA_{w}_pct"] = close / ema_w - 1.0

    # RSI
    feats[f"RSI_{cfg.rsi_window}"] = rsi(close, cfg.rsi_window)

    # MACD
    macd_line, macd_sig, macd_hist = macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    feats["MACD_LINE"] = macd_line
    feats["MACD_SIGNAL"] = macd_sig
    feats["MACD_HIST"] = macd_hist

    # Bollinger (relative position & width)
    bb_u, bb_m, bb_l = bollinger(close, cfg.bb_window, cfg.bb_std)
    feats["BB_UPPER_PCT"] = close / bb_u - 1.0
    feats["BB_MID_PCT"] = close / bb_m - 1.0
    feats["BB_LOWER_PCT"] = close / bb_l - 1.0
    feats["BB_WIDTH"] = (bb_u - bb_l) / bb_m

    # Stochastic
    k, d = stochastic(high, low, close, cfg.stoch_k, cfg.stoch_d)
    feats[f"STO_K_{cfg.stoch_k}"] = k
    feats[f"STO_D_{cfg.stoch_d}"] = d

    # ATR & ATR as % of price
    feats[f"ATR_{cfg.atr_window}"] = atr(high, low, close, cfg.atr_window)
    feats[f"ATR_PCT_{cfg.atr_window}"] = feats[f"ATR_{cfg.atr_window}"] / close

    # OBV
    feats["OBV"] = obv(close, vol) if cfg.obv else np.nan

    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return feats


def make_labels(close: pd.Series, horizon: int, task: str) -> pd.Series:
    """
    Classification: binary UP/DOWN (future close > current close).
    Regression    : future return over 'horizon' days: (future/now - 1).
    """
    future = close.shift(-horizon)
    if task == "clf":
        y = (future > close).astype(int)
    elif task == "reg":
        y = (future / close) - 1.0
    else:
        raise ValueError("task must be 'clf' or 'reg'")
    return y


def align_features_labels(feats: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    df = feats.join(y.rename("y"), how="inner")
    df = df.dropna(subset=["y"])
    X = df.drop(columns=["y"])
    y = df["y"]
    return X, y


# ---------------------
# Modeling
# ---------------------

def make_model(cfg: Config):
    if cfg.task == "clf":
        if cfg.model == "logit":
            clf = LogisticRegression(max_iter=500, class_weight=cfg.class_weight)
            pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", clf),
            ])
            return pipe
        elif cfg.model == "rf":
            clf = RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=3,
                random_state=cfg.random_state, n_jobs=-1
            )
            return clf
        else:
            raise ValueError("Unsupported classifier. Use 'logit' or 'rf'.")
    elif cfg.task == "reg":
        if cfg.model == "linreg":
            reg = LinearRegression()
            pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("reg", reg),
            ])
            return pipe
        elif cfg.model == "ridge":
            reg = Ridge(alpha=1.0, random_state=cfg.random_state)
            pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("reg", reg),
            ])
            return pipe
        elif cfg.model == "rf":
            reg = RandomForestRegressor(
                n_estimators=400, max_depth=None, min_samples_leaf=3,
                random_state=cfg.random_state, n_jobs=-1
            )
            return reg
        else:
            raise ValueError("Unsupported regressor. Use 'linreg', 'ridge', or 'rf'.")
    else:
        raise ValueError("task must be 'clf' or 'reg'")


def make_clone(model):
    """
    Lightweight clone to avoid sklearn.clone complexities with pipelines.
    """
    if isinstance(model, Pipeline):
        steps = []
        for name, est in model.steps:
            if name == "scaler" and isinstance(est, StandardScaler):
                steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
            elif name == "clf" and isinstance(est, LogisticRegression):
                steps.append(("clf", LogisticRegression(max_iter=500, class_weight=est.class_weight)))
            elif name == "reg" and isinstance(est, LinearRegression):
                steps.append(("reg", LinearRegression()))
            elif name == "reg" and isinstance(est, Ridge):
                steps.append(("reg", Ridge(alpha=est.alpha, random_state=getattr(est, "random_state", None))))
        return Pipeline(steps)

    if isinstance(model, RandomForestClassifier):
        return RandomForestClassifier(
            n_estimators=model.n_estimators,
            max_depth=model.max_depth,
            min_samples_leaf=model.min_samples_leaf,
            random_state=model.random_state,
            n_jobs=model.n_jobs,
        )
    if isinstance(model, RandomForestRegressor):
        return RandomForestRegressor(
            n_estimators=model.n_estimators,
            max_depth=model.max_depth,
            min_samples_leaf=model.min_samples_leaf,
            random_state=model.random_state,
            n_jobs=model.n_jobs,
        )
    return model


def predict_scores(model, X: pd.DataFrame, task: str) -> np.ndarray:
    """
    For classification: returns P(y=1).
    For regression: returns point predictions (future returns).
    """
    if task == "clf":
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        if hasattr(model, "decision_function"):
            s = model.decision_function(X)
            return 1 / (1 + np.exp(-s))
        # Fallback to class labels mapped to {0,1}
        preds = model.predict(X)
        return (preds > 0).astype(float)
    else:
        return model.predict(X)


def train_test_split_time(X: pd.DataFrame, y: pd.Series, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    n = len(X)
    split = int(np.floor(n * (1 - test_ratio)))
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]
    return Xtr, Xte, ytr, yte


def feature_importance(model, X_cols: List[str], task: str) -> pd.DataFrame:
    # Tree-based: same for clf/reg
    if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        imp = pd.Series(model.feature_importances_, index=X_cols)
        return imp.sort_values(ascending=False).to_frame("importance")

    # Linear models inside pipelines
    if isinstance(model, Pipeline):
        if task == "clf" and isinstance(model.named_steps.get("clf"), LogisticRegression):
            coefs = model.named_steps["clf"].coef_.ravel()
            imp = pd.Series(np.abs(coefs), index=X_cols)
            return imp.sort_values(ascending=False).to_frame("abs_coef")
        if task == "reg":
            est = model.named_steps.get("reg")
            if isinstance(est, (LinearRegression, Ridge)):
                coefs = est.coef_.ravel()
                imp = pd.Series(np.abs(coefs), index=X_cols)
                return imp.sort_values(ascending=False).to_frame("abs_coef")

    return pd.DataFrame()


# ---------------------
# Indicator performance diagnostics
# ---------------------

def group_features(X_cols: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {
        "SMA": [], "EMA": [], "RSI": [], "MACD": [],
        "BOLL": [], "STOCH": [], "ATR": [], "OBV": [], "RET/VOL": []
    }
    for c in X_cols:
        u = c.upper()
        if u.startswith("SMA_"):
            groups["SMA"].append(c)
        elif u.startswith("EMA_"):
            groups["EMA"].append(c)
        elif u.startswith("RSI_"):
            groups["RSI"].append(c)
        elif u.startswith("MACD_"):
            groups["MACD"].append(c)
        elif u.startswith("BB_") or "BOLL" in u:
            groups["BOLL"].append(c)
        elif u.startswith("STO_"):
            groups["STOCH"].append(c)
        elif u.startswith("ATR_") or u.startswith("ATR_PCT_"):
            groups["ATR"].append(c)
        elif u == "OBV":
            groups["OBV"].append(c)
        elif u.startswith("RET_") or u == "VOL_CHG":
            groups["RET/VOL"].append(c)
        else:
            groups["RET/VOL"].append(c)
    return {g: cols for g, cols in groups.items() if len(cols) > 0}


def univariate_holdout_diag(task: str,
                            Xtr: pd.DataFrame, ytr: pd.Series,
                            Xte: pd.DataFrame, yte: pd.Series) -> pd.DataFrame:
    """
    Classification: ROC-AUC per feature (logit on single feature).
    Regression    : R^2 per feature (linear regression on single feature).
    """
    results = []
    for col in Xtr.columns:
        if np.isclose(Xtr[col].std(ddof=0), 0.0) or Xtr[col].isna().all():
            results.append((col, np.nan))
            continue
        if task == "clf":
            pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
            ])
            try:
                pipe.fit(Xtr[[col]], ytr.astype(int))
                proba = predict_scores(pipe, Xte[[col]], task="clf")
                if len(np.unique(yte)) < 2:
                    score = np.nan
                else:
                    score = roc_auc_score(yte.astype(int), proba)
            except Exception:
                score = np.nan
            results.append((col, score))
        else:
            pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("reg", LinearRegression())
            ])
            try:
                pipe.fit(Xtr[[col]], ytr.astype(float))
                pred = predict_scores(pipe, Xte[[col]], task="reg")
                score = r2_score(yte.astype(float), pred)
            except Exception:
                score = np.nan
            results.append((col, score))

    colname = "univariate_auc" if task == "clf" else "univariate_r2"
    df = pd.DataFrame(results, columns=["feature", colname]).sort_values(colname, ascending=False)
    return df


def group_permutation_importance(task: str,
                                 model,
                                 Xte: pd.DataFrame, yte: pd.Series,
                                 groups: Dict[str, List[str]],
                                 n_repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_pred = predict_scores(model, Xte, task)
    if task == "clf":
        baseline = np.nan
        if len(np.unique(yte)) >= 2:
            baseline = roc_auc_score(yte.astype(int), base_pred)
        rows = []
        for g, cols in groups.items():
            aucs = []
            for _ in range(n_repeats):
                Xperm = Xte.copy()
                for c in cols:
                    if c not in Xperm.columns:
                        continue
                    shuffled = Xperm[c].to_numpy().copy()
                    rng.shuffle(shuffled)
                    Xperm[c] = shuffled
                scores = predict_scores(model, Xperm, task)
                if len(np.unique(yte)) >= 2:
                    aucs.append(roc_auc_score(yte.astype(int), scores))
                else:
                    aucs.append(np.nan)
            mean_shuf = np.nanmean(aucs) if len(aucs) else np.nan
            drop = (baseline - mean_shuf) if (not np.isnan(baseline) and not np.isnan(mean_shuf)) else np.nan
            rows.append((g, baseline, mean_shuf, drop))
        out = pd.DataFrame(rows, columns=["group", "baseline_auc", "shuffled_auc_mean", "auc_drop"])
        return out.sort_values("auc_drop", ascending=False)
    else:
        baseline = r2_score(yte.astype(float), base_pred.astype(float))
        rows = []
        for g, cols in groups.items():
            r2s = []
            for _ in range(n_repeats):
                Xperm = Xte.copy()
                for c in cols:
                    if c not in Xperm.columns:
                        continue
                    shuffled = Xperm[c].to_numpy().copy()
                    rng.shuffle(shuffled)
                    Xperm[c] = shuffled
                pred = predict_scores(model, Xperm, task)
                r2s.append(r2_score(yte.astype(float), pred.astype(float)))
            mean_shuf = np.nanmean(r2s) if len(r2s) else np.nan
            drop = (baseline - mean_shuf) if (not np.isnan(baseline) and not np.isnan(mean_shuf)) else np.nan
            rows.append((g, baseline, mean_shuf, drop))
        out = pd.DataFrame(rows, columns=["group", "baseline_r2", "shuffled_r2_mean", "r2_drop"])
        return out.sort_values("r2_drop", ascending=False)


# ---------------------
# Orchestration
# ---------------------

def walkforward_cv(task: str, model, X: pd.DataFrame, y: pd.Series, n_splits: int) -> Dict[str, float]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    if task == "clf":
        mets = {"acc": [], "prec": [], "rec": [], "f1": [], "roc": []}
        for train_idx, val_idx in tscv.split(X):
            Xtr, Xva = X.iloc[train_idx], X.iloc[val_idx]
            ytr, yva = y.iloc[train_idx].astype(int), y.iloc[val_idx].astype(int)
            model_ = make_clone(model)
            model_.fit(Xtr, ytr)
            proba = predict_scores(model_, Xva, task)
            pred = (proba >= 0.5).astype(int)
            mets["acc"].append(accuracy_score(yva, pred))
            mets["prec"].append(precision_score(yva, pred, zero_division=0))
            mets["rec"].append(recall_score(yva, pred, zero_division=0))
            mets["f1"].append(f1_score(yva, pred, zero_division=0))
            if len(np.unique(yva)) == 2:
                mets["roc"].append(roc_auc_score(yva, proba))
        return {k: float(np.mean(v)) if v else np.nan for k, v in mets.items()}

    else:
        mets = {"mae": [], "rmse": [], "r2": [], "dir_acc": []}
        for train_idx, val_idx in tscv.split(X):
            Xtr, Xva = X.iloc[train_idx], X.iloc[val_idx]
            ytr, yva = y.iloc[train_idx].astype(float), y.iloc[val_idx].astype(float)
            model_ = make_clone(model)
            model_.fit(Xtr, ytr)
            pred = predict_scores(model_, Xva, task)
            mets["mae"].append(mean_absolute_error(yva, pred))
            mets["rmse"].append(np.sqrt(mean_squared_error(yva, pred)))
            mets["r2"].append(r2_score(yva, pred))
            # Directional accuracy on validation
            mets["dir_acc"].append(np.mean(np.sign(pred) == np.sign(yva)))
        return {k: float(np.mean(v)) if v else np.nan for k, v in mets.items()}


def main():
    # Optional CLI (kept simple/optional to avoid breaking defaults)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", choices=["clf", "reg"])
    parser.add_argument("--model")
    parser.add_argument("--ticker")
    parser.add_argument("--years", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--test_ratio", type=float)
    args, _ = parser.parse_known_args()

    cfg = Config()
    if args.task: cfg.task = args.task
    if args.model: cfg.model = args.model
    if args.ticker: cfg.ticker = args.ticker
    if args.years: cfg.period_years = args.years
    if args.horizon: cfg.horizon = args.horizon
    if args.test_ratio: cfg.test_ratio = args.test_ratio

    print(f"\nTicker: {cfg.ticker} | Period: {cfg.period_years}y | Interval: {cfg.interval}")
    print(f"Horizon: {cfg.horizon}d  | Task: {cfg.task} | Model: {cfg.model} | Test ratio: {cfg.test_ratio}\n")

    raw = fetch_ohlcv(cfg)
    feats = build_features(raw, cfg)

    # Align labels
    y = make_labels(raw["Close"].reindex(feats.index), cfg.horizon, cfg.task)
    X, y = align_features_labels(feats, y)

    # Train/test split
    Xtr, Xte, ytr, yte = train_test_split_time(X, y, cfg.test_ratio)

    # Model
    model = make_model(cfg)

    # Walk-forward CV (train only)
    cv_metrics = walkforward_cv(cfg.task, model, Xtr, ytr, cfg.n_splits)
    print("Walk-forward CV (train-only):")
    for k, v in cv_metrics.items():
        print(f"  {k.upper():<7}: {v:.4f}")
    print()

    # Fit on full train and evaluate holdout
    model.fit(Xtr, ytr if cfg.task == "reg" else ytr.astype(int))

    # Previous day's close aligned to Xte index (for contextual preview)
    prev_close_holdout = raw["Close"].shift(1).reindex(Xte.index)

    if cfg.task == "clf":
        proba_te = predict_scores(model, Xte, task="clf")
        pred_te = (proba_te >= 0.5).astype(int)
        holdout = {
            "ACC": accuracy_score(yte.astype(int), pred_te),
            "PREC": precision_score(yte.astype(int), pred_te, zero_division=0),
            "REC": recall_score(yte.astype(int), pred_te, zero_division=0),
            "F1": f1_score(yte.astype(int), pred_te, zero_division=0),
        }
        try:
            holdout["ROC"] = roc_auc_score(yte.astype(int), proba_te)
        except Exception:
            holdout["ROC"] = np.nan
        # Explicit hit rate (same as ACC) for clarity
        holdout["HIT_RATE"] = holdout["ACC"]

        print("Holdout (last segment):")
        for k, v in holdout.items():
            print(f"  {k:<8}: {v:.4f}")
        print()

        # Preview last 10 with correctness and prev day's close
        preview = pd.DataFrame({
            "prev_close": prev_close_holdout,
            "y_true": yte.astype(int),
            "y_prob_up": proba_te,
            "y_pred_up": pred_te,
        })
        preview["pred_correct"] = (preview["y_true"] == preview["y_pred_up"]).astype(int)
        preview = preview.tail(10)
        preview.index.name = "date"

    else:
        pred_te = predict_scores(model, Xte, task="reg")
        holdout = {
            "MAE": mean_absolute_error(yte.astype(float), pred_te.astype(float)),
            "RMSE": np.sqrt(mean_squared_error(yte.astype(float), pred_te.astype(float))),
            "R2": r2_score(yte.astype(float), pred_te.astype(float)),
        }
        # Directional accuracy on holdout
        dir_acc = np.mean(np.sign(pred_te) == np.sign(yte.astype(float)))
        holdout["DIR_ACC"] = dir_acc

        print("Holdout (last segment):")
        for k, v in holdout.items():
            print(f"  {k:<8}: {v:.4f}")
        print()

        # Preview last 10 with previous day's close, directional correctness, and abs error
        preview = pd.DataFrame({
            "prev_close": prev_close_holdout,
            "y_true": yte.astype(float),
            "y_pred": pred_te.astype(float),
        })
        preview["true_up"] = (preview["y_true"] > 0).astype(int)
        preview["pred_up"] = (preview["y_pred"] > 0).astype(int)
        preview["correct_dir"] = (preview["true_up"] == preview["pred_up"]).astype(int)
        preview["abs_error"] = (preview["y_true"] - preview["y_pred"]).abs()
        preview = preview.tail(10)
        preview.index.name = "date"

    # Feature importance
    imp = feature_importance(model, X.columns.tolist(), cfg.task)
    if not imp.empty:
        print("Top features:")
        print(imp.head(15).to_string())
    else:
        print("Feature importance not available for this model.")
    print()

    # --- Indicator diagnostics ---
    groups = group_features(X.columns.tolist())

    # 1) Univariate diagnostics (holdout)
    uni = univariate_holdout_diag(cfg.task, Xtr, ytr, Xte, yte)
    if cfg.task == "clf":
        print("\nPer-feature univariate ROC-AUC on holdout (top 20):")
        print(uni.head(20).to_string(index=False))
    else:
        print("\nPer-feature univariate R2 on holdout (top 20):")
        print(uni.head(20).to_string(index=False))

    # 2) Group permutation importance
    gp = group_permutation_importance(cfg.task, model, Xte, yte, groups, n_repeats=5, seed=cfg.random_state)
    if cfg.task == "clf":
        print("\nGroup permutation importance (AUC drop on holdout):")
    else:
        print("\nGroup permutation importance (R2 drop on holdout):")
    print(gp.to_string(index=False))

    # Enriched preview
    print("\nLast 10 predictions (with previous day's close and correctness):")
    print(preview.to_string())


if __name__ == "__main__":
    main()
