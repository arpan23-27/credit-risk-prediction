import json
from contextlib import asynccontextmanager
from pathlib import Path


import numpy as np
import pandas as pd

from app.schemas import ApplicantIn, PredictionOut
from src.train import CATEGORICAL_COLS, NUMERIC_COLS


import joblib
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "model"
CONFIG_PATH = ROOT / "artifacts" / "threshold" / "model_config.json"

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["raw"] = joblib.load(MODEL_DIR / "credit_risk_xgb.joblib")
    state["calibrated"] = joblib.load(MODEL_DIR / "credit_risk_xgb_calibrated.joblib")
    state["config"] = json.loads(CONFIG_PATH.read_text())
    yield
    state.clear()


app = FastAPI(title="Credit Risk API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "threshold": state["config"]["decision_threshold"]}



@app.post("/predict", response_model=PredictionOut)
def predict(applicant: ApplicantIn):
    row = applicant.model_dump()
    row["loan_percent_income"] = row["loan_amnt"] / row["person_income"]
    row = {k: (np.nan if v is None else v) for k, v in row.items()}
    X = pd.DataFrame([row])[NUMERIC_COLS + CATEGORICAL_COLS]

    raw = float(state["raw"].predict_proba(X)[0, 1])
    calibrated = float(state["calibrated"].predict_proba(X)[0, 1])
    threshold = state["config"]["decision_threshold"]

    return PredictionOut(
        risk_probability=round(min(max(calibrated, 0.01), 0.99), 4),
        decision="high_risk" if raw >= threshold else "low_risk",
        threshold=threshold,
    )