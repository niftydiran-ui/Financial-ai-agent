# Agentic AI for Automated Financial Data Analysis (2025)

An **agentic AI** system that retrieves, cleans, models, and reports on financial time series (stocks/ETFs).

## Quick Start
```
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional: copy .env.example to .env and set OPENAI_API_KEY
python main.py --ticker AAPL --period 5y --horizon 30
streamlit run app.py
```

Artifacts will be saved in `artifacts/<TICKER>`.
