from typing import Literal, Optional

from pydantic import BaseModel, Field


class ApplicantIn(BaseModel):
    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0)
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: Optional[float] = Field(default=None, ge=0, le=60)
    loan_intent: Literal["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
                         "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(gt=0)
    loan_int_rate: Optional[float] = Field(default=None, ge=0, le=40)
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(ge=0, le=60)


class PredictionOut(BaseModel):
    risk_probability: float
    decision: Literal["high_risk", "low_risk"]
    threshold: float