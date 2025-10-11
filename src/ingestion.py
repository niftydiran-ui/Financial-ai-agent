import pandas as pd
import yfinance as yf

def load_timeseries(ticker=None, period="5y", csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path)
    elif ticker:
        df = yf.download(ticker, period=period).reset_index()
    else:
        raise ValueError("Either ticker or csv_path must be provided")
    expected = ["Date","Open","High","Low","Close","Adj Close","Volume"]
    if "Date" not in df.columns and "date" in df.columns:
        df.rename(columns={"date":"Date"}, inplace=True)
    if "Adj Close" not in df.columns and "adj close" in df.columns:
        df.rename(columns={"adj close":"Adj Close"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df[expected].dropna().sort_values("Date").reset_index(drop=True)
