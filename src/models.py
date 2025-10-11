import warnings, numpy as np, pandas as pd
from typing import Dict, Tuple, List
try:
    from prophet import Prophet
except Exception:
    Prophet = None
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf = None
    keras = None
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

def _evaluate(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}

def train_and_forecast(df: pd.DataFrame, horizon: int = 30, ticker: str = "TICKER") -> Tuple[pd.DataFrame, Dict, List[str]]:
    used = []
    metrics = {}
    preds = []
    train = df.iloc[:-horizon].copy()
    test = df.iloc[-horizon:].copy()

    if Prophet is not None:
        m = Prophet(daily_seasonality=True)
        dprop = train[["Date","Adj Close"]].rename(columns={"Date":"ds","Adj Close":"y"})
        try:
            m.fit(dprop)
            future = pd.DataFrame({"ds": test["Date"]})
            p = m.predict(future)
            yhat = p["yhat"].values
            metrics["Prophet"] = _evaluate(test["Adj Close"].values, yhat)
            preds.append(pd.DataFrame({"Date": test["Date"], "model": "Prophet", "yhat": yhat}))
            used.append("Prophet")
        except Exception as e:
            warnings.warn(f"Prophet failed: {e}")

    if XGBRegressor is not None:
        try:
            lag_df = _make_lags(train, test, nlags=10)
            model = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.9)
            model.fit(lag_df["Xtr"], lag_df["ytr"])
            yhat = model.predict(lag_df["Xte"])
            metrics["XGBoost"] = _evaluate(lag_df["yte"], yhat)
            preds.append(pd.DataFrame({"Date": test["Date"], "model": "XGBoost", "yhat": yhat}))
            used.append("XGBoost")
        except Exception as e:
            warnings.warn(f"XGBoost failed: {e}")

    if keras is not None and tf is not None:
        try:
            Xtr, ytr, Xte, yte = _make_seq(train["Adj Close"], test["Adj Close"], win=20)
            model = keras.Sequential([
                keras.layers.Input(shape=(Xtr.shape[1], 1)),
                keras.layers.LSTM(32, return_sequences=False),
                keras.layers.Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(Xtr, ytr, epochs=10, batch_size=32, verbose=0)
            yhat = model.predict(Xte, verbose=0).ravel()
            metrics["LSTM"] = _evaluate(yte, yhat)
            preds.append(pd.DataFrame({"Date": test["Date"], "model": "LSTM", "yhat": yhat}))
            used.append("LSTM")
        except Exception as e:
            warnings.warn(f"LSTM failed: {e}")

    if not preds:
        yhat = np.roll(test["Adj Close"].values, 1)
        yhat[0] = train["Adj Close"].iloc[-1]
        metrics["Naive"] = _evaluate(test["Adj Close"].values, yhat)
        preds.append(pd.DataFrame({"Date": test["Date"], "model": "Naive", "yhat": yhat}))
        used.append("Naive")

    pred_df = pd.concat(preds, ignore_index=True)
    return pred_df, metrics, used

def _make_lags(train: pd.DataFrame, test: pd.DataFrame, nlags: int = 10):
    tr = train.copy(); te = test.copy()
    for l in range(1, nlags+1):
        tr[f"AdjClose_lag{l}"] = tr["Adj Close"].shift(l)
        te[f"AdjClose_lag{l}"] = te["Adj Close"].shift(l)
    tr = tr.dropna()
    features = [c for c in tr.columns if "lag" in c]
    Xtr = tr[features].values
    ytr = tr["Adj Close"].values
    combo = pd.concat([train.tail(nlags), test], ignore_index=True)
    for l in range(1, nlags+1):
        combo[f"AdjClose_lag{l}"] = combo["Adj Close"].shift(l)
    te2 = combo.iloc[nlags+1:].copy()
    Xte = te2[[c for c in te2.columns if "lag" in c]].values
    yte = te2["Adj Close"].values
    return {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte}

def _make_seq(tr_series: pd.Series, te_series: pd.Series, win:int=20):
    s = pd.concat([tr_series, te_series], ignore_index=True).values.astype(float)
    X, y = [], []
    for i in range(win, len(s)):
        X.append(s[i-win:i])
        y.append(s[i])
    import numpy as np
    X = np.array(X); y = np.array(y)
    te_len = len(te_series)
    Xtr, ytr = X[:-te_len], y[:-te_len]
    Xte, yte = X[-te_len:], y[-te_len:]
    Xtr = Xtr[..., None]; Xte = Xte[..., None]
    return Xtr, ytr, Xte, yte
