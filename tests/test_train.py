import numpy as np
import pandas as pd
import pytest

from src.train import (CATEGORICAL_COLS, NUMERIC_COLS, build_pipeline, clean_data,
                       cost_per_applicant, find_cost_threshold)


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "person_age": rng.integers(21, 60, n),
        "person_income": rng.integers(20_000, 120_000, n),
        "person_home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
        "person_emp_length": rng.integers(0, 15, n).astype(float),
        "loan_intent": rng.choice(["PERSONAL", "EDUCATION", "MEDICAL"], n),
        "loan_grade": rng.choice(list("ABCD"), n),
        "loan_amnt": rng.integers(1_000, 30_000, n),
        "loan_int_rate": rng.uniform(6, 20, n).round(2),
        "loan_percent_income": rng.uniform(0.05, 0.5, n).round(2),
        "cb_person_default_on_file": rng.choice(["Y", "N"], n),
        "cb_person_cred_hist_length": rng.integers(2, 20, n),
    })
    df["loan_status"] = (rng.random(n) < 0.25).astype(int)
    return df


def test_clean_data_drops_impossible_rows_but_keeps_missing(sample_df):
    df = sample_df.copy()
    df.loc[0, "person_age"] = 144            # impossible age
    df.loc[1, "person_emp_length"] = 123     # impossible employment length
    df.loc[2, "person_emp_length"] = np.nan  # missing -> must be KEPT
    df = pd.concat([df, df.iloc[[5]]])       # exact duplicate row
    cleaned = clean_data(df)
    assert len(cleaned) == len(sample_df) - 2        # 2 impossible rows removed, duplicate removed
    assert cleaned["person_emp_length"].isna().sum() == 1
    
    
    
def test_pipeline_outputs_valid_probabilities_with_missing_and_unseen_values(sample_df):
    df = sample_df.copy()
    df.loc[:20, "loan_int_rate"] = np.nan
    X, y = df[NUMERIC_COLS + CATEGORICAL_COLS], df["loan_status"]
    model = build_pipeline(scale_pos_weight=3.0, n_estimators=10).fit(X, y)
    new = X.head(5).copy()
    new.loc[new.index[0], "loan_grade"] = "Z"        # category never seen in training
    new.loc[new.index[1], "person_emp_length"] = np.nan
    proba = model.predict_proba(new)[:, 1]
    assert proba.shape == (5,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_cost_per_applicant_matches_hand_calculation():
    y = np.array([1, 0, 1, 0])
    proba = np.array([0.9, 0.8, 0.2, 0.1])
    # threshold 0.5 -> predictions [1, 1, 0, 0]: TP=1, FP=1, FN=1, TN=1
    assert cost_per_applicant(y, proba, 0.5, cost_fn=5, cost_fp=1) == pytest.approx((5 * 1 + 1 * 1) / 4)


def test_higher_false_negative_cost_never_raises_the_threshold():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.25).astype(int)
    proba = np.clip(0.25 + 0.45 * y + rng.normal(0, 0.2, 2000), 0, 1)
    t_cheap = find_cost_threshold(y, proba, cost_fn=1, cost_fp=1)
    t_costly = find_cost_threshold(y, proba, cost_fn=10, cost_fp=1)
    assert t_costly <= t_cheap
    