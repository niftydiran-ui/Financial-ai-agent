import plotly.graph_objects as go
import pandas as pd

def make_all_charts(df: pd.DataFrame, preds: pd.DataFrame, ticker="TICKER"):
    charts = {}
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df["Date"], y=df["Adj Close"], name=f"{ticker} Adj Close"))
    fig1.update_layout(title=f"{ticker} • Adjusted Close")
    charts["price_history"] = fig1

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["Date"], y=df["Adj Close"], name="Actual"))
    for m in preds["model"].unique():
        sub = preds[preds["model"] == m]
        fig2.add_trace(go.Scatter(x=sub["Date"], y=sub["yhat"], name=m))
    fig2.update_layout(title=f"{ticker} • Forecast Comparison")
    charts["forecast_comparison"] = fig2
    return charts
