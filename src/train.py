from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy.stats import randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


log = logging.getLogger("train")

ROOT = Path(__file__).resolve().parents[1]


TARGET = "loan_status"
NUMERIC_COLS = ["person_age", "person_income", "person_emp_length", "loan_amnt",
                "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
CATEGORICAL_COLS = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
REQUIRED_COLS = NUMERIC_COLS + CATEGORICAL_COLS + [TARGET]


PARAM_DIST = {
    "classifier__n_estimators": randint(150, 600),
    "classifier__max_depth": randint(3, 9),
    "classifier__learning_rate": uniform(0.01, 0.5),
    "classifier__subsample": uniform(0.6, 0.4),
    "classifier__colsample_bytree": uniform(0.6, 0.4),
    "classifier__min_child_weight": randint(1, 10),
    "classifier__gamma": uniform(0, 5),
}



def load_data(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and impossible values. Rows with a missing value are kept
    (they are imputed later inside the pipeline), because comparisons with NaN are False."""
    out = df.copy()
    steps = [
        ("duplicate rows",          lambda d: d.drop_duplicates()),
        ("age outside 18-100",      lambda d: d[d["person_age"].between(18, 100)]),
        ("employment length > 60",  lambda d: d[~(d["person_emp_length"] > 60)]),
        ("employment length > age", lambda d: d[~(d["person_emp_length"] > d["person_age"])]),
        ("income <= 0",             lambda d: d[d["person_income"] > 0]),
        ("loan amount <= 0",        lambda d: d[d["loan_amnt"] > 0]),
    ]
    log.info("Rows before cleaning: %d", len(out))
    for name, step in steps:
        before = len(out)
        out = step(out)
        log.info("  removed %5d rows: %s", before - len(out), name)
    log.info("Rows after cleaning: %d", len(out))
    return out


def build_preprocessor() -> ColumnTransformer:
    """Median-impute numerics (+ 'was missing' flags), one-hot the categoricals.
    No scaling: tree models do not need it."""
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))]), NUMERIC_COLS),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_COLS),
    ])


def build_pipeline(scale_pos_weight: float, seed: int = 42, **xgb_kwargs) -> Pipeline:
    params = dict(n_estimators=300, max_depth=5, learning_rate=0.1,
                  eval_metric="logloss", random_state=seed)
    params.update(xgb_kwargs)
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", XGBClassifier(scale_pos_weight=scale_pos_weight, **params)),
    ])
    
    
    
    
    
def cost_per_applicant(y_true, proba, threshold: float, cost_fn: float, cost_fp: float) -> float:
    """Average cost: a false negative (approved a defaulter) costs cost_fn, a false positive
    (rejected a good customer) costs cost_fp."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(proba) >= threshold
    fn = int(((~flagged) & (y_true == 1)).sum())
    fp = int((flagged & (y_true == 0)).sum())
    return (cost_fn * fn + cost_fp * fp) / len(y_true)


def find_cost_threshold(y_true, proba, cost_fn: float, cost_fp: float) -> float:
    grid = np.linspace(0.01, 0.99, 99)
    costs = [cost_per_applicant(y_true, proba, t, cost_fn, cost_fp) for t in grid]
    return float(grid[int(np.argmin(costs))])


def evaluate(y_true, proba, threshold: float, cost_fn: float, cost_fp: float) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "avg_cost_per_applicant": float(cost_per_applicant(y_true, proba, threshold, cost_fn, cost_fp)),
    }
    
    
def train(args: argparse.Namespace) -> dict:
    df = clean_data(load_data(args.data))

    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    log.info("Train %s | Test %s | default rate train %.3f test %.3f",
             X_train.shape, X_test.shape, y_train.mean(), y_test.mean())

    neg, pos = np.bincount(y_train)
    scale_pos_weight = float(neg / pos)
    log.info("scale_pos_weight = %.2f", scale_pos_weight)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    log.info("Tuning XGBoost (%d iterations x 5 folds, scoring = PR-AUC)...", args.n_iter)
    search = RandomizedSearchCV(
        build_pipeline(scale_pos_weight, args.seed), PARAM_DIST, n_iter=args.n_iter, cv=cv,
        scoring="average_precision", n_jobs=-1, random_state=args.seed, verbose=1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    log.info("Best CV PR-AUC: %.4f", search.best_score_)

    # Threshold from out-of-fold predictions on TRAIN only, so the test set stays untouched
    oof = cross_val_predict(best_model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = find_cost_threshold(y_train, oof, args.cost_fn, args.cost_fp)
    log.info("Cost-optimal threshold (cost_fn=%s, cost_fp=%s): %.2f", args.cost_fn, args.cost_fp, threshold)
    
    
    
    
    log.info("Calibrating probabilities (%s)...", args.calibration)
    calibrated = CalibratedClassifierCV(best_model, method=args.calibration, cv=5).fit(X_train, y_train)

    # One-time evaluation on the held-out test set
    proba_raw = best_model.predict_proba(X_test)[:, 1]
    proba_cal = calibrated.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, proba_raw, threshold, args.cost_fn, args.cost_fp)
    test_metrics["brier_raw"] = float(brier_score_loss(y_test, proba_raw))
    test_metrics["brier_calibrated"] = float(brier_score_loss(y_test, proba_cal))
    log.info("Test metrics: %s", json.dumps(test_metrics, indent=2))

    # Save
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.threshold_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, args.model_dir / "credit_risk_xgb.joblib")
    joblib.dump(calibrated, args.model_dir / "credit_risk_xgb_calibrated.joblib")

    config = {
        "decision_threshold": threshold,
        "threshold_applies_to": "raw (uncalibrated) probability from credit_risk_xgb.joblib",
        "cost_fn": args.cost_fn, "cost_fp": args.cost_fp,
        "calibration_method": args.calibration,
        "seed": args.seed, "n_iter": args.n_iter,
        "best_cv_pr_auc": float(search.best_score_),
        "best_params": {k.replace("classifier__", ""): (int(v) if isinstance(v, (int, np.integer)) else float(v))
                        for k, v in search.best_params_.items()},
        "numeric_cols": NUMERIC_COLS, "categorical_cols": CATEGORICAL_COLS,
        "test_metrics": test_metrics,
        "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    (args.threshold_dir / "model_config.json").write_text(json.dumps(config, indent=2))
    log.info("Saved model files to %s and config to %s", args.model_dir, args.threshold_dir)
    return config



def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the credit risk model.")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "credit_risk_dataset.csv")
    p.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "model")
    p.add_argument("--threshold-dir", type=Path, default=ROOT / "artifacts" / "threshold")
    p.add_argument("--n-iter", type=int, default=50, help="randomized-search iterations")
    p.add_argument("--cost-fn", type=float, default=5.0, help="cost of approving a defaulter")
    p.add_argument("--cost-fp", type=float, default=1.0, help="cost of rejecting a good customer")
    p.add_argument("--calibration", choices=["sigmoid", "isotonic"], default="isotonic")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(parse_args(argv))


if __name__ == "__main__":
    main()