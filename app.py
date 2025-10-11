import os, streamlit as st
from dotenv import load_dotenv
from src.ingestion import load_timeseries
from src.preprocess import prep_timeseries
from src.models import train_and_forecast
from src.visualize import make_all_charts
from src.reporting import generate_report

st.set_page_config(page_title="Agentic Financial Analyst", layout="wide")
st.title("🤖 Agentic AI • Financial Analyst (2025)")
load_dotenv()

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker (e.g., AAPL)", "AAPL")
with col2:
    period = st.selectbox("History period", ["1y","3y","5y","10y","max"], index=2)
with col3:
    horizon = st.number_input("Forecast horizon (days)", min_value=7, max_value=180, value=30)

uploaded = st.file_uploader("Or upload CSV with Date/Open/High/Low/Close/Adj Close/Volume")

if st.button("Run Analysis"):
    csv_path = None
    if uploaded:
        csv_path = os.path.join("data", uploaded.name)
        os.makedirs("data", exist_ok=True)
        with open(csv_path, "wb") as f:
            f.write(uploaded.read())

    ts = load_timeseries(ticker=ticker if not csv_path else None, period=period, csv_path=csv_path)
    feats = prep_timeseries(ts)
    preds, metrics, used_models = train_and_forecast(feats, horizon=horizon, ticker=ticker if not csv_path else "CSV")

    st.subheader("Metrics")
    st.json(metrics)

    st.subheader("Charts")
    charts = make_all_charts(feats, preds, ticker=ticker if not csv_path else "CSV")
    for name, fig in charts.items():
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Report")
    st.text(generate_report(ticker if not csv_path else "CSV", metrics, used_models, horizon))
