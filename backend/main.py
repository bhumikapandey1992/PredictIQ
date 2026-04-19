import json

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.pipeline.features import engineer_healthcare, engineer_sales, engineer_stocks
from backend.pipeline.ingest import load_healthcare, load_sales, load_stocks
from backend.pipeline.train import train_model

app = FastAPI(title="PredictIQ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store: dict = {"data": {}, "features": {}, "models": {}, "metrics": {}, "importance": {}}

DOMAIN_TASK = {
    "stocks": "classification",
    "sales": "regression",
    "healthcare": "classification",
}

DOMAIN_LABELS = {
    "stocks": {0: "Down", 1: "Up"},
    "healthcare": {0: "Malignant", 1: "Benign"},
}


def _df_records(df: pd.DataFrame, n: int = 200) -> list:
    return json.loads(df.tail(n).to_json(orient="records", date_format="iso"))


def _df_stats(df: pd.DataFrame) -> dict:
    return json.loads(df.describe().to_json())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest/{domain}")
def ingest(domain: str, ticker: str = "AAPL", period: str = "2y"):
    if domain == "stocks":
        df = load_stocks(ticker, period)
    elif domain == "sales":
        df = load_sales()
    elif domain == "healthcare":
        df = load_healthcare()
    else:
        return JSONResponse(status_code=400, content={"error": "Unknown domain"})

    store["data"][domain] = df
    return {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "records": _df_records(df, 200),
        "stats": _df_stats(df),
    }


@app.get("/data/{domain}")
def get_data(domain: str):
    if domain not in store["data"]:
        return JSONResponse(status_code=400, content={"error": "Ingest data first"})
    df = store["data"][domain]
    return {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "records": _df_records(df, 200),
        "stats": _df_stats(df),
    }


@app.post("/train/{domain}")
def train(domain: str, model_type: str = "xgboost"):
    if domain not in store["data"]:
        return JSONResponse(status_code=400, content={"error": "Ingest data first"})

    df = store["data"][domain]

    if domain == "stocks":
        X, y, features = engineer_stocks(df)
    elif domain == "sales":
        X, y, features = engineer_sales(df)
    elif domain == "healthcare":
        X, y, features = engineer_healthcare(df)
    else:
        return JSONResponse(status_code=400, content={"error": "Unknown domain"})

    store["features"][domain] = features
    task = DOMAIN_TASK[domain]
    model, metrics, importance = train_model(X, y, task, model_type)

    store["models"][domain] = model
    store["metrics"][domain] = metrics
    store["importance"][domain] = importance

    return {"metrics": metrics, "importance": importance, "task": task}


@app.get("/metrics/{domain}")
def get_metrics(domain: str):
    if domain not in store["metrics"]:
        return JSONResponse(status_code=400, content={"error": "Train model first"})
    return {
        "metrics": store["metrics"][domain],
        "importance": store["importance"][domain],
    }


class PredictRequest(BaseModel):
    features: dict


@app.post("/predict/{domain}")
def predict(domain: str, body: PredictRequest):
    if domain not in store["models"]:
        return JSONResponse(status_code=400, content={"error": "Train model first"})

    model = store["models"][domain]
    features = store["features"][domain]
    row = pd.DataFrame([{f: body.features.get(f, 0.0) for f in features}])

    pred = model.predict(row)[0]
    result: dict = {"prediction": float(pred)}

    if DOMAIN_TASK[domain] == "classification":
        proba = model.predict_proba(row)[0]
        result["probabilities"] = [round(float(p), 4) for p in proba]
        result["label"] = DOMAIN_LABELS[domain].get(int(pred), str(pred))

    return result
