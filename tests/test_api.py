from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "artifacts" / "model" / "credit_risk_xgb.joblib").exists(),
    reason="model artifacts not found; run python src/train.py first",
)

RISKY = {
    "person_age": 25, "person_income": 30000, "person_home_ownership": "RENT",
    "person_emp_length": 2, "loan_intent": "PERSONAL", "loan_grade": "E",
    "loan_amnt": 15000, "loan_int_rate": 16.5, "cb_person_default_on_file": "Y",
    "cb_person_cred_hist_length": 3,
}
SAFE = {
    "person_age": 45, "person_income": 120000, "person_home_ownership": "MORTGAGE",
    "person_emp_length": 15, "loan_intent": "EDUCATION", "loan_grade": "A",
    "loan_amnt": 5000, "loan_int_rate": 7.5, "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 12,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_threshold(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert 0 < r.json()["threshold"] < 1


def test_risky_applicant_scores_higher_than_safe_applicant(client):
    risky = client.post("/predict", json=RISKY).json()
    safe = client.post("/predict", json=SAFE).json()
    assert risky["risk_probability"] > safe["risk_probability"]
    assert risky["decision"] == "high_risk"
    assert safe["decision"] == "low_risk"


def test_missing_optional_fields_are_accepted(client):
    body = {**SAFE, "person_emp_length": None, "loan_int_rate": None}
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    assert 0.01 <= r.json()["risk_probability"] <= 0.99


def test_unknown_category_is_rejected(client):
    r = client.post("/predict", json={**SAFE, "loan_grade": "Z"})
    assert r.status_code == 422


def test_underage_applicant_is_rejected(client):
    r = client.post("/predict", json={**SAFE, "person_age": 15})
    assert r.status_code == 422