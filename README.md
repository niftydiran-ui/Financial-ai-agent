# Agentic AI - Financial Analyst (2025)

An **agentic AI** system for automated retrieval, cleaning, modeling, and reporting of financial time series (stocks/ETFs). This project leverages modern ML frameworks (Prophet, XGBoost, LSTM), automation (agentic workflows), and interactive reporting (Plotly/Streamlit) for robust financial analysis.

## Features

- **Automated Time-Series Pipeline:** Data ingestion (from Yahoo Finance or CSV), cleaning, feature engineering, and modeling.
- **Multi-Model Forecasting:** Supports Prophet, XGBoost, and LSTM for financial forecasting with automated performance evaluation (RMSE, MAE, MAPE).
- **Agentic Orchestration:** Can run with LLM-powered (LangChain/OpenAI) workflows or deterministic fallback.
- **Interactive Web App:** Visual reporting via Streamlit, offering line charts, forecast comparisons, and downloadable artifacts.
- **Report Generation:** Generates easy-to-interpret summaries and recommendations for finance stakeholders.
- **Works with both tickers and custom data CSVs.**

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional: copy .env.example to .env and set OPENAI_API_KEY
python main.py --ticker AAPL --period 5y --horizon 30
# or launch Streamlit app
streamlit run app.py
```

Artifacts (metrics, charts, report) will be saved in `artifacts/`.

## Usage

- Use as a **command-line pipeline**:
  ```bash
  python main.py --ticker AAPL --period 5y --horizon 30
  ```
  Options:
  - `--ticker <SYMBOL>`: Stock symbol (e.g., `AAPL`)
  - `--period <1y|3y|5y|10y|max>`: Data lookback window
  - `--horizon <DAYS>`: Forecast horizon (default: 30)
  - `--csv <path>`: Use local CSV file (Date/Open/High/Low/Close/Adj Close/Volume)
  - `--agent`: Run with LLM-powered agent (needs OpenAI API key)

- Use via **Streamlit web app**:
  - Input ticker, period, forecast horizon, or upload a CSV.
  - Click "Run Analysis" to get metrics, charts, and a finance-facing summary.

## Project Structure

- `main.py` – CLI pipeline and agentic runner
- `app.py` – Streamlit app for interactive UX
- `src/`
  - `ingestion.py` – Data loader (Yahoo Finance or CSV)
  - `preprocess.py` – Feature engineering, technical indicators
  - `models.py` – Time series forecasting (Prophet, XGBoost, LSTM, Naive)
  - `visualize.py` – Plotly chart rendering
  - `reporting.py` – Automated summary generation (LLM or fallback)
  - `agents.py` – Agentic orchestration with LangChain/OpenAI

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies:
  - pandas, numpy, plotly, streamlit, yfinance, scikit-learn, prophet, xgboost, tensorflow-cpu, langchain, openai, chromadb, faiss-cpu, python-dotenv, pydantic

## Example Output

- **Interactive charts** (Adjusted Close, Forecast Comparison)
- **Metrics JSON** (`MAE`, `RMSE`, `MAPE%` per model)
- **Automated report**: concisely summarizes best model, fit quality, and recommendations.

## License

See [LICENSE](LICENSE).

---

*Feel free to enhance or modify this README according to your organization or advanced usage — the above gives a solid, recruiter/ATS-friendly presentation and onboarding for contributors or users.*
