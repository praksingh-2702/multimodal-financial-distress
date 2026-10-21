"""
train_baseline.py
-----------------
Baselines on financial ratios only. These are the numbers every later model (FinBERT, GRU, GNN,
fusion) has to beat.

  1. Altman Z-score proxy  - classic rule, no training (lower Z = riskier)
  2. Logistic regression   - simple, interpretable
  3. XGBoost               - strong tabular model

Input : data/processed/dataset.csv   (build_dataset.py)
Output: reports/baseline_results.csv
        reports/baseline_feature_importance.csv
        models/xgb_baseline.json
        data/processed/preds_baseline.csv  (test-set probabilities, used later for fusion/plots)

Run from the project root:
    python src/train_baseline.py

Metrics: PR-AUC (main - robust to class imbalance), ROC-AUC, and precision/recall/F1 of the
distressed class at a threshold chosen on the VALIDATION set (never on test).
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA = Path("data/processed/dataset.csv")
REPORTS = Path("reports")
MODELS = Path("models")
SEED = 42

FEATURES = [
    "debt_to_equity", "debt_to_assets", "interest_coverage", "liquidity_proxy", "cash_to_assets",
    "roa", "ebit_margin", "asset_turnover", "cfo_to_assets", "cfo_to_debt", "mcap_to_debt",
    "altman_z_proxy", "sales_growth", "borrowings_growth", "price_change", "negative_net_worth",
    "log_assets", "d_roa", "d_debt_to_assets", "d_interest_coverage", "n_missing",
]


def winsorize(train, *others, lo=0.01, hi=0.99):
    """Clip extreme ratios using limits learned on TRAIN only."""
    q_lo, q_hi = train.quantile(lo), train.quantile(hi)
    return [d.clip(q_lo, q_hi, axis=1) for d in (train, *others)]


def best_threshold(y, p):
    """Threshold that maximises F1 on the validation set."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.nanargmax(f1[:-1])]) if len(thr) else 0.5


def evaluate(name, y_val, p_val, y_test, p_test):
    thr = best_threshold(y_val, p_val)
    pred = (p_test >= thr).astype(int)
    return {
        "model": name,
        "pr_auc": average_precision_score(y_test, p_test),
        "roc_auc": roc_auc_score(y_test, p_test),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "threshold": thr,
    }


def main():
    df = pd.read_csv(DATA)
    tr, va, te = (df[df["split"] == s] for s in ["train", "val", "test"])
    y_tr, y_va, y_te = tr["label"].values, va["label"].values, te["label"].values
    X_tr, X_va, X_te = winsorize(tr[FEATURES], va[FEATURES], te[FEATURES])

    print(f"Train {len(tr)} rows ({y_tr.sum()} pos) | Val {len(va)} ({y_va.sum()} pos) | "
          f"Test {len(te)} ({y_te.sum()} pos)")
    print(f"Random-guess PR-AUC on test = positive rate = {y_te.mean():.3f}\n")

    results, preds = [], te[["company_id", "company", "group", "fiscal_year", "label"]].copy()

    # 1. Altman Z proxy: no training, lower = riskier -> use negative Z as the score
    z_va = -X_va["altman_z_proxy"].fillna(X_tr["altman_z_proxy"].median())
    z_te = -X_te["altman_z_proxy"].fillna(X_tr["altman_z_proxy"].median())
    # convert to 0-1 rank so a threshold can be chosen on val
    rank = lambda s, ref: np.searchsorted(np.sort(ref), s) / len(ref)
    p_va_z, p_te_z = rank(z_va.values, z_va.values), rank(z_te.values, z_va.values)
    results.append(evaluate("Altman Z proxy (rule)", y_va, p_va_z, y_te, p_te_z))
    preds["p_altman"] = p_te_z

    # 2. Logistic regression (balanced class weights handle the imbalance)
    logit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5))
    logit.fit(X_tr, y_tr)
    p_va_l, p_te_l = logit.predict_proba(X_va)[:, 1], logit.predict_proba(X_te)[:, 1]
    results.append(evaluate("Logistic regression", y_va, p_va_l, y_te, p_te_l))
    preds["p_logit"] = p_te_l

    # 3. XGBoost (handles missing values natively; scale_pos_weight handles imbalance)
    spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    xgb = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
                        scale_pos_weight=spw, eval_metric="aucpr", random_state=SEED)
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    p_va_x, p_te_x = xgb.predict_proba(X_va)[:, 1], xgb.predict_proba(X_te)[:, 1]
    results.append(evaluate("XGBoost (ratios)", y_va, p_va_x, y_te, p_te_x))
    preds["p_xgb"] = p_te_x

    # ---- save ----------------------------------------------------------------
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    res = pd.DataFrame(results)
    res.to_csv(REPORTS / "baseline_results.csv", index=False)
    imp = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=False)
    imp.to_csv(REPORTS / "baseline_feature_importance.csv", header=["importance"])
    xgb.save_model(MODELS / "xgb_baseline.json")
    preds.to_csv(Path("data/processed/preds_baseline.csv"), index=False)

    print("TEST RESULTS")
    print(res.round(3).to_string(index=False))
    print("\nXGBoost - top 10 features:")
    print(imp.head(10).round(3).to_string())

    top = preds.sort_values("p_xgb", ascending=False).head(10)
    print("\nHighest-risk test rows (XGBoost):")
    print(top[["company", "fiscal_year", "label", "p_xgb"]].round(3).to_string(index=False))
    print(f"\nSaved: {REPORTS / 'baseline_results.csv'}, {MODELS / 'xgb_baseline.json'}")


if __name__ == "__main__":
    main()