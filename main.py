import argparse, os, json, pathlib
from dotenv import load_dotenv
from src.ingestion import load_timeseries
from src.preprocess import prep_timeseries
from src.models import train_and_forecast
from src.visualize import make_all_charts
from src.reporting import generate_report
from src.agents import AgenticRunner

def run_pipeline(ticker=None, period="5y", horizon=30, csv=None, outdir="artifacts"):
    ts = load_timeseries(ticker=ticker, period=period, csv_path=csv)
    feats = prep_timeseries(ts)
    preds, metrics, used_models = train_and_forecast(feats, horizon=horizon, ticker=ticker or "CSV")
    charts = make_all_charts(feats, preds, ticker=ticker or "CSV")

    run_dir = os.path.join(outdir, (ticker or pathlib.Path(csv).stem if csv else "CSV"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    for name, fig in charts.items():
        fig.write_html(os.path.join(run_dir, f"{name}.html"))
    report_txt = generate_report(ticker or "CSV", metrics, used_models, horizon)
    with open(os.path.join(run_dir, "report.txt"), "w") as f:
        f.write(report_txt)
    print(f"Artifacts saved to: {run_dir}")
    print("Models used:", used_models)
    return run_dir

def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default=None)
    p.add_argument("--period", type=str, default="5y")
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--agent", action="store_true")
    args = p.parse_args()

    if args.agent:
        runner = AgenticRunner()
        runner.run(ticker=args.ticker, period=args.period, horizon=args.horizon, csv=args.csv)
    else:
        run_pipeline(ticker=args.ticker, period=args.period, horizon=args.horizon, csv=args.csv)

if __name__ == "__main__":
    main()
