"""
train_baseline.py
-----------------
Ratio-only baselines, evaluated with 5-fold cross-validation by company, for both horizons
(12 and 24 months). These are the numbers the GRU, GNN, FinBERT and fusion models must beat.

  1. Altman Z-score proxy  - textbook rule, no training (lower Z = riskier)
  2. Logistic regression   - simple, interpretable
  3. XGBoost               - strong tabular model

Input : data/processed/dataset.csv  (+ folds.csv, created automatically on first run)
Output: reports/baseline_cv.csv              mean ± std per model and horizon
        reports/baseline_cv_per_fold.csv
        reports/baseline_feature_importance.csv
        data/processed/oof_baseline.csv      out-of-fold scores (used later by fusion)
        models/xgb_baseline.json             XGBoost refit on ALL data (for the dashboard)

Run from the project root:
    python src/train_baseline.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from cv_utils import (FEATURES, HORIZONS, N_FOLDS, SEED, best_threshold, fold_metrics,
                      inner_split, load_dataset, summarize, winsorize)

warnings.filterwarnings("ignore")
REPORTS, MODELS = Path("reports"), Path("models")


def make_logit():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5))


def make_xgb(y):
    spw = (len(y) - y.sum()) / max(y.sum(), 1)
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
                         scale_pos_weight=spw, eval_metric="aucpr", random_state=SEED)


def altman_score(X_fit, X):
    """Higher score = riskier. Missing Z filled with the training median."""
    return -X["altman_z_proxy"].fillna(X_fit["altman_z_proxy"].median()).values


def main():
    df = load_dataset()
    print(f"Rows {len(df)} | companies {df['company_id'].nunique()} | {N_FOLDS}-fold CV by company")
    for h, col in HORIZONS.items():
        print(f"  {h}: {df[col].sum()} positive rows ({df[col].mean():.1%}) -> random PR-AUC ~ {df[col].mean():.3f}")

    per_fold = []
    oof = df[["company_id", "company", "group", "fiscal_year", "fold", "label", "label_24m"]].copy()
    importances = []

    for h, col in HORIZONS.items():
        for k in range(N_FOLDS):
            train, test = df[df["fold"] != k], df[df["fold"] == k]
            fit, val = inner_split(train)
            X_fit, X_val, X_test = winsorize(fit[FEATURES], val[FEATURES], test[FEATURES])
            y_fit, y_val, y_test = fit[col].values, val[col].values, test[col].values

            # 1. Altman Z rule
            s_val, s_test = altman_score(X_fit, X_val), altman_score(X_fit, X_test)
            per_fold.append({"model": "Altman Z proxy (rule)", "horizon": h, "fold": k,
                             **fold_metrics(y_test, s_test, best_threshold(y_val, s_val))})
            oof.loc[test.index, f"altman_{h}"] = s_test

            # 2. Logistic regression
            m = make_logit().fit(X_fit, y_fit)
            p_val, p_test = m.predict_proba(X_val)[:, 1], m.predict_proba(X_test)[:, 1]
            per_fold.append({"model": "Logistic regression", "horizon": h, "fold": k,
                             **fold_metrics(y_test, p_test, best_threshold(y_val, p_val))})
            oof.loc[test.index, f"logit_{h}"] = p_test

            # 3. XGBoost
            m = make_xgb(y_fit).fit(X_fit, y_fit)
            p_val, p_test = m.predict_proba(X_val)[:, 1], m.predict_proba(X_test)[:, 1]
            per_fold.append({"model": "XGBoost (ratios)", "horizon": h, "fold": k,
                             **fold_metrics(y_test, p_test, best_threshold(y_val, p_val))})
            oof.loc[test.index, f"xgb_{h}"] = p_test
            if h == "12m":
                importances.append(pd.Series(m.feature_importances_, index=FEATURES))

    # ---- save ----------------------------------------------------------------
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    table, _ = summarize(per_fold)
    table.to_csv(REPORTS / "baseline_cv.csv", index=False)
    pd.DataFrame(per_fold).to_csv(REPORTS / "baseline_cv_per_fold.csv", index=False)
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    imp.to_csv(REPORTS / "baseline_feature_importance.csv", header=["importance"])
    oof.to_csv("data/processed/oof_baseline.csv", index=False)

    # final XGBoost on all data (12m) for the dashboard / SHAP
    X_all, = winsorize(df[FEATURES])
    make_xgb(df["label"].values).fit(X_all, df["label"].values).save_model(MODELS / "xgb_baseline.json")

    print("\nCROSS-VALIDATED RESULTS (mean ± std over 5 folds)")
    print(table.to_string(index=False))
    print("\nPooled out-of-fold PR-AUC (12m): " + ", ".join(
        f"{m}={average_precision_score(oof['label'], oof[f'{m}_12m']):.3f}" for m in ["altman", "logit", "xgb"]))
    print("\nXGBoost (12m) - top 10 features, averaged over folds:")
    print(imp.head(10).round(3).to_string())
    print(f"\nSaved: {REPORTS / 'baseline_cv.csv'}, data/processed/oof_baseline.csv, {MODELS / 'xgb_baseline.json'}")


if __name__ == "__main__":
    main()