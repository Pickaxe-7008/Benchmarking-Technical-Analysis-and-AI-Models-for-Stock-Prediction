"""
- no look-ahead: meta-learner trained on OOF only; prediction day excluded from training.
- predicts the probability that Close_{t+h} > Close_{t} for `predict_date`.
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    # Data
    ticker: str = "AAPL"
    benchmark: str = "SPY"         # for optional controls (market return)
    period_years: int = 5
    interval: str = "1d"

    # Target
    horizon: int = 1                # predict next-day direction by default
    predict_date: str = "2025-10-16"  # <<<<<< default set to 10/16

    # Models
    ta_model: str = "rf"            # "logit" or "rf"
    meta_model: str = "ridge"       # "logit" or "ridge"

    # CV / Training
    n_splits: int = 5
    random_state: int = 42

    # Indicators
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
    use_obv: bool = True

    # Meta controls
    use_market_controls: bool = True   # add SPY day return & realized vol to meta-learner
    realized_vol_window: int = 10


# =============================================================================
# Data helpers
# =============================================================================

def fetch_ohlcv(ticker: str, years: int, interval: str) -> pd.DataFrame:
    period = f"{years}y"
    df = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=False, progress=False, group_by="column", threads=True
    )
    if df.empty:
        end = pd.Timestamp.today(tz="UTC").normalize()
        start = end - pd.DateOffset(years=years)
        df = yf.download(
            ticker, start=start.date().isoformat(), end=end.date().isoformat(),
            interval=interval, auto_adjust=False, progress=False,
            group_by="column", threads=True
        )
    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level="Ticker")
        except Exception:
            df = df.droplevel(-1, axis=1)

    df = df.rename(columns=str.title)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not set(needed).issubset(df.columns):
        raise KeyError(f"Missing columns in {ticker}: need {needed}, have {df.columns.tolist()}")
    return df.dropna(subset=needed)


# =============================================================================
# Indicators
# =============================================================================

def sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).mean()

def ema(s: pd.Series, w: int) -> pd.Series:
    return s.ewm(span=w, adjust=False, min_periods=w).mean()

def rsi(close: pd.Series, w: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/w, adjust=False).mean()
    roll_down = down.ewm(alpha=1/w, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ef = ema(close, fast)
    es = ema(close, slow)
    line = ef - es
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = line - sig
    return line, sig, hist

def bollinger(close: pd.Series, w=20, n_std=2.0):
    ma = close.rolling(w, min_periods=w).mean()
    sd = close.rolling(w, min_periods=w).std()
    up, lo = ma + n_std*sd, ma - n_std*sd
    return up, ma, lo

def stochastic(h: pd.Series, l: pd.Series, c: pd.Series, k=14, d=3):
    ll = l.rolling(k, min_periods=k).min()
    hh = h.rolling(k, min_periods=k).max()
    pct_k = 100 * (c - ll) / (hh - ll)
    pct_d = pct_k.rolling(d, min_periods=d).mean()
    return pct_k, pct_d

def atr(h: pd.Series, l: pd.Series, c: pd.Series, w=14):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(w, min_periods=w).mean()

def obv(close: pd.Series, vol: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * vol).fillna(0).cumsum()

def build_ta_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    feats = pd.DataFrame(index=df.index)

    feats["RET_1"] = c.pct_change(1)
    feats["RET_5"] = c.pct_change(5)
    feats["VOL_CHG_5"] = v.pct_change(5)

    for w in cfg.sma_windows:
        s = sma(c, w)
        feats[f"SMA_{w}"] = s
        feats[f"SMA_{w}_PCT"] = c / s - 1.0

    for w in cfg.ema_windows:
        e = ema(c, w)
        feats[f"EMA_{w}"] = e
        feats[f"EMA_{w}_PCT"] = c / e - 1.0

    feats[f"RSI_{cfg.rsi_window}"] = rsi(c, cfg.rsi_window)

    m_line, m_sig, m_hist = macd(c, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    feats["MACD_LINE"], feats["MACD_SIGNAL"], feats["MACD_HIST"] = m_line, m_sig, m_hist

    bb_u, bb_m, bb_l = bollinger(c, cfg.bb_window, cfg.bb_std)
    feats["BB_UP_PCT"] = c / bb_u - 1.0
    feats["BB_MID_PCT"] = c / bb_m - 1.0
    feats["BB_LO_PCT"] = c / bb_l - 1.0
    feats["BB_WIDTH"] = (bb_u - bb_l) / bb_m

    k, d = stochastic(h, l, c, cfg.stoch_k, cfg.stoch_d)
    feats[f"STO_K_{cfg.stoch_k}"] = k
    feats[f"STO_D_{cfg.stoch_d}"] = d

    feats[f"ATR_{cfg.atr_window}"] = atr(h, l, c, cfg.atr_window)
    feats[f"ATR_PCT_{cfg.atr_window}"] = feats[f"ATR_{cfg.atr_window}"] / c

    if cfg.use_obv:
        feats["OBV"] = obv(c, v)

    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return feats

def make_labels(close: pd.Series, horizon: int) -> pd.Series:
    fut = close.shift(-horizon)
    return (fut > close).astype(int)  # 1=UP, 0=DOWN

def align_xy(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    df = X.join(y.rename("y"), how="inner")
    df = df.dropna(subset=["y"])
    return df.drop(columns=["y"]), df["y"].astype(int)

# =============================================================================
# Models
# =============================================================================

def make_ta_model(name: str, random_state: int):
    if name == "logit":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
        ])
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, random_state=random_state, n_jobs=-1
        )
    raise ValueError("ta_model must be 'logit' or 'rf'")

def make_meta_model(name: str, random_state: int):
    if name == "logit":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
        ])
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=random_state)
    raise ValueError("meta_model must be 'logit' or 'ridge'")

# =============================================================================
# OOF generation (time-series CV)
# =============================================================================

def time_series_oof(model, X: pd.DataFrame, y: pd.Series, n_splits: int) -> pd.Series:
    """
    Returns a pandas Series of OOF predicted probabilities aligned to X.index.
    """
    oof = pd.Series(index=X.index, dtype=float)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr_idx, va_idx in tscv.split(X):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr = y.iloc[tr_idx]
        model_ = clone_estimator(model)
        model_.fit(Xtr, ytr)
        proba = predict_proba_safe(model_, Xva)
        oof.iloc[va_idx] = proba
    return oof

def clone_estimator(model):
    # Minimal clone to avoid sklearn.clone dependency nuances with pipelines
    if isinstance(model, Pipeline):
        steps = []
        for name, est in model.steps:
            if name == "scaler" and isinstance(est, StandardScaler):
                steps.append(("scaler", StandardScaler()))
            elif name == "clf" and isinstance(est, LogisticRegression):
                steps.append(("clf", LogisticRegression(max_iter=500, class_weight=est.class_weight)))
        return Pipeline(steps)
    if isinstance(model, RandomForestClassifier):
        return RandomForestClassifier(
            n_estimators=model.n_estimators,
            min_samples_leaf=model.min_samples_leaf,
            random_state=model.random_state,
            n_jobs=model.n_jobs
        )
    if isinstance(model, Ridge):
        return Ridge(alpha=model.alpha, random_state=getattr(model, "random_state", None))
    raise ValueError("Unsupported estimator type for clone.")

def predict_proba_safe(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1 / (1 + np.exp(-s))
    # If a regressor-like meta is used, map to [0,1] with logistic
    preds = model.predict(X)
    return 1 / (1 + np.exp(-preds))

# =============================================================================
# Meta features (fusion)
# =============================================================================

def build_meta_features(
    p_ta: pd.Series,
    controls: Optional[pd.DataFrame] = None,
    # ==== LLM PLACEHOLDERS ====================================================
    # p_llm: Optional[pd.Series] = None,
    # ==========================================================================
) -> pd.DataFrame:
    """
    Meta feature set: base predictions + optional controls.
    """
    meta = pd.DataFrame(index=p_ta.index)
    meta["p_ta"] = p_ta

    # ==== LLM PLACEHOLDERS ====================================================
    # if p_llm is not None:
    #     meta["p_llm"] = p_llm
    # ==========================================================================

    if controls is not None and not controls.empty:
        meta = meta.join(controls, how="left")

    meta = meta.dropna()
    return meta

def make_controls(bench_df: pd.DataFrame, realized_vol_window: int) -> pd.DataFrame:
    """
    Market controls: SPY daily return & realized vol (rolling stdev of returns).
    """
    c = bench_df["Close"].copy()
    rets = c.pct_change(1).rename("mkt_ret")
    vol = rets.rolling(realized_vol_window, min_periods=realized_vol_window).std().rename("mkt_rvol")
    out = pd.concat([rets, vol], axis=1)
    return out

# =============================================================================
# Orchestration
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str)
    parser.add_argument("--predict_date", type=str)
    args = parser.parse_args()

    cfg = Config()
    if args.ticker: cfg.ticker = args.ticker
    if args.predict_date: cfg.predict_date = args.predict_date

    # 1) Fetch data
    px = fetch_ohlcv(cfg.ticker, cfg.period_years, cfg.interval)
    bench = fetch_ohlcv(cfg.benchmark, cfg.period_years, cfg.interval) if cfg.use_market_controls else None

    # 2) TA features and labels
    X_ta_full = build_ta_features(px, cfg)
    y_full = make_labels(px["Close"].reindex(X_ta_full.index), cfg.horizon)
    X_ta_full, y_full = align_xy(X_ta_full, y_full)

    # 3) Locate prediction date
    pred_day = pd.to_datetime(cfg.predict_date)
    if pred_day.tzinfo is not None:
        pred_day = pred_day.tz_convert(None)
    # Use the index’s timezone-naive dates
    idx = pd.to_datetime(X_ta_full.index).normalize()
    # Find exact or nearest previous trading day
    if pred_day not in idx:
        prior = idx[idx < pred_day]
        if len(prior) == 0:
            raise ValueError(f"No data prior to {cfg.predict_date} for {cfg.ticker}.")
        closest = prior.max()
        print(f"[INFO] {cfg.predict_date} not found; using nearest previous trading day: {closest.date()}")
        pred_day = closest

    # 4) Split into train (< predict_date) and target row (= predict_date)
    train_mask = idx < pred_day
    if train_mask.sum() < 100:
        print("[WARN] Very small training window; results may be unstable.")

    X_ta_train = X_ta_full.loc[train_mask]
    y_train = y_full.loc[train_mask]

    X_ta_predrow = X_ta_full.loc[idx == pred_day]
    if X_ta_predrow.empty:
        raise RuntimeError("Failed to slice prediction row. Check date alignment.")

    # 5) Optional market controls
    controls_full = None
    if cfg.use_market_controls and bench is not None:
        controls_full = make_controls(bench.reindex(X_ta_full.index), cfg.realized_vol_window)
        controls_train = controls_full.loc[X_ta_train.index]
        controls_pred = controls_full.loc[X_ta_predrow.index]
    else:
        controls_train = None
        controls_pred = None

    # ==== LLM PLACEHOLDERS ====================================================
    # # Example: build or load LLM features aligned to X_ta_full.index
    # X_llm_full = load_or_build_llm_features(index=X_ta_full.index)
    # # Train an LLM base model on X_llm_train -> OOF predictions p_llm_oof
    # llm_model = make_llm_model(...)
    # p_llm_oof = time_series_oof(llm_model, X_llm_train, y_train, cfg.n_splits)
    # ==========================================================================

    # 6) TA base model OOF predictions (Level-1)
    ta_model = make_ta_model(cfg.ta_model, cfg.random_state)
    p_ta_oof = time_series_oof(ta_model, X_ta_train, y_train, cfg.n_splits)

    # 7) Build meta training frame from OOF (+ controls, + optional p_llm)
    meta_train = build_meta_features(
        p_ta=p_ta_oof,
        controls=controls_train,
        # p_llm=p_llm_oof,
    )
    y_meta = y_train.loc[meta_train.index]
    # Remove rows where y is not both classes (safety for ROC/Logit)
    if len(np.unique(y_meta)) < 2:
        raise ValueError("Training labels have a single class before predict_date; cannot fit meta-learner.")

    # 8) Fit meta-learner (Level-2)
    meta = make_meta_model(cfg.meta_model, cfg.random_state)
    meta.fit(meta_train, y_meta)

    # 9) Refit base model(s) on all training data, create meta features for pred_day
    ta_model.fit(X_ta_train, y_train)
    p_ta_pred = pd.Series(predict_proba_safe(ta_model, X_ta_predrow), index=X_ta_predrow.index, name="p_ta")

    meta_predrow = build_meta_features(
        p_ta=p_ta_pred,
        controls=controls_pred,
        # p_llm=None,  # enable when LLM base ready
    )

    # 10) Final stacked prediction for predict_date
    p_final = predict_proba_safe(meta, meta_predrow)[0]

    # 11) (Optional) quick diagnostics on last fold of training
    #     Evaluate base TA alone vs. stacked on the tail of training set
    try:
        last_split = list(TimeSeriesSplit(n_splits=cfg.n_splits).split(X_ta_train))[-1]
        tr_idx, va_idx = last_split
        Xva, yva = X_ta_train.iloc[va_idx], y_train.iloc[va_idx]
        ta_tmp = make_ta_model(cfg.ta_model, cfg.random_state)
        ta_tmp.fit(X_ta_train.iloc[tr_idx], y_train.iloc[tr_idx])
        p_ta_va = predict_proba_safe(ta_tmp, Xva)
        # Recreate meta features for that validation slice
        meta_va = build_meta_features(
            p_ta=pd.Series(p_ta_va, index=Xva.index),
            controls=(controls_train.loc[Xva.index] if controls_train is not None else None),
        )
        p_meta_va = predict_proba_safe(meta, meta_va)
        auc_ta = safe_auc(yva, p_ta_va)
        auc_stack = safe_auc(yva.loc[meta_va.index], p_meta_va)
    except Exception:
        auc_ta, auc_stack = np.nan, np.nan

    # 12) Output
    print("\n================ LATE-FUSION STACK RESULT ================")
    print(f"Ticker          : {cfg.ticker}")
    print(f"Predict date    : {pred_day.date()}  (horizon={cfg.horizon}d)")
    print(f"Base (TA) model : {cfg.ta_model}")
    print(f"Meta-learner    : {cfg.meta_model}")
    print("----------------------------------------------------------")
    print(f"Final stacked probability of UP: {p_final:.4f}")
    print("----------------------------------------------------------")
    if not np.isnan(auc_ta) and not np.isnan(auc_stack):
        print(f"Validation AUC  (last fold)  — TA only : {auc_ta:.3f}")
        print(f"Validation AUC  (last fold)  — STACKED : {auc_stack:.3f}")
    else:
        print("Validation AUC  (last fold)  — unavailable (insufficient split or labels).")
    print("==========================================================\n")


# =============================================================================
# Utilities
# =============================================================================

def safe_auc(y_true: pd.Series, scores: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, scores)
    except Exception:
        return np.nan


# =============================================================================
# LLM PLACEHOLDERS
# # =============================================================================


if __name__ == "__main__":
    main()
