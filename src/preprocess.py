import pandas as pd

def prep_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"]).sort_values()
    d = d.sort_values("Date").reset_index(drop=True)
    d["return_1d"] = d["Adj Close"].pct_change()
    for w in [5,10,20]:
        d[f"sma_{w}"] = d["Adj Close"].rolling(w).mean()
        d[f"ema_{w}"] = d["Adj Close"].ewm(span=w, adjust=False).mean()
    d["volatility_10"] = d["return_1d"].rolling(10).std()
    d["rsi_14"] = rsi(d["Adj Close"], 14)
    ema12 = d["Adj Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Adj Close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    mid = d["Adj Close"].rolling(20).mean()
    std = d["Adj Close"].rolling(20).std()
    d["bb_mid"] = mid
    d["bb_upper"] = mid + 2*std
    d["bb_lower"] = mid - 2*std
    return d.dropna().reset_index(drop=True)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))
