import os, json
from typing import Optional
from dotenv import load_dotenv

def _no_agent_fallback(ticker, period, horizon, csv):
    from .ingestion import load_timeseries
    from .preprocess import prep_timeseries
    from .models import train_and_forecast
    from .visualize import make_all_charts
    from .reporting import generate_report
    import os, json

    ts = load_timeseries(ticker=ticker, period=period, csv_path=csv)
    feats = prep_timeseries(ts)
    preds, metrics, used_models = train_and_forecast(feats, horizon=horizon, ticker=ticker or "CSV")
    charts = make_all_charts(feats, preds, ticker=ticker or "CSV")

    run_dir = os.path.join("artifacts", (ticker or "CSV"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    for name, fig in charts.items():
        fig.write_html(os.path.join(run_dir, f"{name}.html"))
    with open(os.path.join(run_dir, "report.txt"), "w") as f:
        f.write(generate_report(ticker or "CSV", metrics, used_models, horizon))
    print(f"[NO-AGENT] Artifacts saved to: {run_dir}")

class AgenticRunner:
    def __init__(self):
        load_dotenv()
        self.have_llm = bool(os.getenv("OPENAI_API_KEY"))
        try:
            import langchain  # noqa
            self.have_langchain = True
        except Exception:
            self.have_langchain = False

    def run(self, ticker:Optional[str], period:str="5y", horizon:int=30, csv:Optional[str]=None):
        if not (self.have_llm and self.have_langchain):
            print("LangChain/LLM not available — running fallback deterministic pipeline.")
            return _no_agent_fallback(ticker, period, horizon, csv)

        from langchain.agents import initialize_agent, AgentType
        from langchain.tools import Tool
        from langchain_openai import ChatOpenAI

        def _t_ingest(_):
            from .ingestion import load_timeseries
            return load_timeseries(ticker=ticker, period=period, csv_path=csv)

        def _t_prep(data):
            from .preprocess import prep_timeseries
            import pandas as pd
            return prep_timeseries(data)

        def _t_model(data):
            from .models import train_and_forecast
            return train_and_forecast(data, horizon=horizon, ticker=ticker or "CSV")

        def _t_visual(args):
            from .visualize import make_all_charts
            feats, preds = args
            return make_all_charts(feats, preds, ticker=ticker or "CSV")

        def _t_report(args):
            from .reporting import generate_report
            _, (preds, metrics, used) = args
            return generate_report(ticker or "CSV", metrics, used, horizon)

        tools = [
            Tool(name="IngestData", func=_t_ingest, description="Load time series data."),
            Tool(name="PreprocessData", func=_t_prep, description="Clean and engineer features."),
            Tool(name="TrainModels", func=_t_model, description="Train Prophet/XGBoost/LSTM and forecast."),
            Tool(name="Visualize", func=_t_visual, description="Create Plotly charts."),
            Tool(name="Report", func=_t_report, description="Generate a natural language report."),
        ]
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        result = agent.run("Ingest -> Preprocess -> TrainModels -> Visualize -> Report. Return a one-paragraph summary.")
        print(result)
