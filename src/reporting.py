import os, datetime as dt
from dotenv import load_dotenv

def generate_report(ticker, metrics:dict, used_models:list, horizon:int)->str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    header = f"Agentic Financial Analyst Report — {ticker}\nGenerated: {dt.datetime.utcnow().isoformat()}Z\nHorizon: {horizon} days\nModels used: {', '.join(used_models)}\n\n"
    body = _fallback_summary(metrics)
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI()
            prompt = header + "Metrics:\n" + str(metrics) + "\n\nWrite a concise, professional summary for a finance stakeholder."
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.2)
            return resp.choices[0].message.content
        except Exception:
            pass
    return header + body

def _fallback_summary(metrics:dict)->str:
    best_model = None
    best_rmse = float("inf")
    for m, vals in metrics.items():
        rmse = vals.get("RMSE", 1e9)
        if rmse < best_rmse:
            best_rmse = rmse; best_model = m
    lines = [
        "Summary (fallback, no LLM):",
        f"- {best_model} achieved the lowest RMSE ≈ {best_rmse:.2f}.",
        "- Lower MAE/RMSE/MAPE indicate better forecast fit.",
        "Recommendations:",
        "1) Consider a weighted ensemble around the top model.",
        "2) Monitor drift and retrain weekly or on regime change.",
        "3) Use scenario analysis (bull/base/bear)."
    ]
    return "\n".join(lines)
