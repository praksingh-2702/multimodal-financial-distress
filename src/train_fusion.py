"""
train_fusion.py
---------------
Combines the models and produces the project's headline ABLATION TABLE.

Part A - all 1,740 company-years (12m and 24m horizons)
  Inputs are the out-of-fold scores of XGBoost (ratios), GRU (trends) and GNN (peers).
  Every score for a company came from a model that never saw that company, so fusion is fair.
  * Rank-average fusion: average of the models' rank scores (no training, no leakage)
  * Stacked fusion: logistic regression on the 3 scores, trained on the other folds
  Ablation: each model alone, every pair, all three.

Part B - the annual-report subset (~135 company-years with a usable report, 12m horizon)
  * Numbers only     : fused XGB+GRU+GNN score (from Part A)
  * FinBERT only     : sentiment features
  * Text only        : FinBERT + red-flag phrases (going concern, defaults, restructuring...)
  * Numbers + text   : all of the above together
  Same company folds as everything else.

Input : data/processed/oof_baseline.csv, oof_gru.csv, oof_gnn.csv, text_features.csv, folds.csv
Output: reports/ablation_cv.csv, reports/text_cv.csv (+ per-fold files)
        data/processed/final_scores.csv    one row per company-year: every model's score + fusion

Run from the project root (~1 minute):
    python src/train_fusion.py
"""

from itertools import combinations
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cv_utils import HORIZONS, N_FOLDS, best_threshold, fold_metrics, summarize

warnings.filterwarnings("ignore")
P = Path("data/processed")
REPORTS = Path("reports")
KEY = ["company_id", "fiscal_year"]
MODELS = {"xgb": "XGBoost (ratios)", "gru": "GRU (trends)", "gnn": "GNN (peers)"}

FB_COLS = ["fb_neg_share", "fb_pos_share", "fb_mean_neg", "fb_net_tone", "fb_tone_std",
           "fb_neg_p90", "fb_debt_share", "fb_debt_tone"]
KW_COLS = ["kw_going_concern", "kw_material_uncertainty", "kw_qualified_opinion",
           "kw_emphasis_of_matter", "kw_default", "kw_npa", "kw_restructuring",
           "kw_erosion_net_worth", "kw_liquidity_stress", "kw_nclt"]


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def rank01(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def load_scores():
    base = pd.read_csv(P / "oof_baseline.csv")
    gru = pd.read_csv(P / "oof_gru.csv")[KEY + ["gru_12m", "gru_24m"]]
    gnn = pd.read_csv(P / "oof_gnn.csv")[KEY + ["industry", "gnn_12m", "gnn_24m", "mlp_12m", "mlp_24m"]]
    for name, f in [("baseline", base), ("GRU", gru), ("GNN", gnn)]:
        if f.duplicated(KEY).any() or len(f) != len(base):
            raise SystemExit(f"{name} scores: {len(f)} rows, {f.duplicated(KEY).sum()} duplicate "
                             f"company-years (baseline has {len(base)}). Re-run build_dataset.py, "
                             "train_baseline.py, train_gru.py and train_gnn.py in that order.")
    df = base.merge(gru, on=KEY, how="inner", validate="one_to_one") \
             .merge(gnn, on=KEY, how="inner", validate="one_to_one")
    return df


def part_a(df):
    per_fold = []
    for h, col in HORIZONS.items():
        y = df[col].values
        for k in range(1, len(MODELS) + 1):
            for combo in combinations(MODELS, k):
                cols = [f"{m}_{h}" for m in combo]
                score = np.mean([rank01(df[c].values) for c in cols], axis=0)
                name = " + ".join(MODELS[m].split(" ")[0] for m in combo)
                name = MODELS[combo[0]] if k == 1 else f"Fusion: {name}"
                for f in range(N_FOLDS):
                    tr, te = df["fold"] != f, df["fold"] == f
                    thr = best_threshold(y[tr], score[tr])
                    per_fold.append({"model": name, "horizon": h, "fold": f,
                                     **fold_metrics(y[te], score[te], thr)})
                if k == len(MODELS):
                    df[f"fusion_avg_{h}"] = score

        # stacked: logistic regression on the three scores, trained on the other folds
        X = np.column_stack([logit(df[f"{m}_{h}"].values) for m in MODELS])
        stacked = np.zeros(len(df))
        for f in range(N_FOLDS):
            tr, te = (df["fold"] != f).values, (df["fold"] == f).values
            m = LogisticRegression(class_weight="balanced", C=1.0).fit(X[tr], y[tr])
            p_tr, p_te = m.predict_proba(X[tr])[:, 1], m.predict_proba(X[te])[:, 1]
            stacked[te] = p_te
            per_fold.append({"model": "Fusion: stacked (logistic on XGB, GRU, GNN)", "horizon": h,
                             "fold": f, **fold_metrics(y[te], p_te, best_threshold(y[tr], p_tr))})
        df[f"fusion_stacked_{h}"] = stacked
        weights = LogisticRegression(class_weight="balanced").fit(X, y).coef_[0]
        print(f"  stacked weights {h}: " + ", ".join(f"{MODELS[m]}={w:.2f}" for m, w in zip(MODELS, weights)))

        # reference rows
        for ref, name in [("altman", "Altman Z proxy (rule)"), ("logit", "Logistic regression")]:
            s = df[f"{ref}_{h}"].values
            for f in range(N_FOLDS):
                tr, te = df["fold"] != f, df["fold"] == f
                per_fold.append({"model": name, "horizon": h, "fold": f,
                                 **fold_metrics(y[te], s[te], best_threshold(y[tr], s[tr]))})
    return per_fold


def part_b(df):
    tf = pd.read_csv(P / "text_features.csv")
    tf = tf.dropna(subset=["company_id", "fiscal_year"])
    tf["fiscal_year"] = tf["fiscal_year"].astype(int)
    sub = tf.merge(df[KEY + ["fold", "label", "fusion_avg_12m", "fusion_stacked_12m"]],
                   on=KEY, how="inner", suffixes=("_t", ""))
    sub = sub.dropna(subset=["fb_net_tone"]).reset_index(drop=True)
    y = sub["label"].values
    print(f"\nText subset: {len(sub)} reports ({y.sum()} distressed, {len(y) - y.sum()} healthy)")

    sets = {
        "Numbers only (fused XGB+GRU+GNN)": None,
        "FinBERT sentiment only": FB_COLS,
        "Text only (FinBERT + red flags)": FB_COLS + KW_COLS,
        "Numbers + text": ["num_score"] + FB_COLS + KW_COLS,
    }
    sub["num_score"] = logit(sub["fusion_stacked_12m"].values)
    per_fold = []
    sub["text_score"] = np.nan
    for name, cols in sets.items():
        for f in range(N_FOLDS):
            tr, te = (sub["fold"] != f).values, (sub["fold"] == f).values
            if y[te].sum() == 0 or y[te].sum() == te.sum():
                continue
            if cols is None:
                s_tr, s_te = sub["fusion_stacked_12m"].values[tr], sub["fusion_stacked_12m"].values[te]
            else:
                m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(class_weight="balanced", C=0.3, max_iter=2000))
                m.fit(sub.loc[tr, cols], y[tr])
                s_tr, s_te = m.predict_proba(sub.loc[tr, cols])[:, 1], m.predict_proba(sub.loc[te, cols])[:, 1]
                if name.startswith("Text only"):
                    sub.loc[te, "text_score"] = s_te
            per_fold.append({"model": name, "horizon": "12m (report subset)", "fold": f,
                             **fold_metrics(y[te], s_te, best_threshold(y[tr], s_tr))})
    return per_fold, sub


def main():
    df = load_scores()
    print(f"Company-years: {len(df)} | models: {', '.join(MODELS.values())}")
    pf_a = part_a(df)
    pf_b, sub = part_b(df)

    REPORTS.mkdir(exist_ok=True)
    table_a, _ = summarize(pf_a)
    table_b, _ = summarize(pf_b)
    table_a.to_csv(REPORTS / "ablation_cv.csv", index=False)
    table_b.to_csv(REPORTS / "text_cv.csv", index=False)
    pd.DataFrame(pf_a).to_csv(REPORTS / "ablation_cv_per_fold.csv", index=False)
    pd.DataFrame(pf_b).to_csv(REPORTS / "text_cv_per_fold.csv", index=False)

    df = df.merge(sub[KEY + ["save_as", "text_score"]], on=KEY, how="left")
    df.to_csv(P / "final_scores.csv", index=False)

    show = ["horizon", "model", "pr_auc", "roc_auc", "f1"]
    print("\nA. ABLATION - all company-years (mean ± std over 5 folds)")
    print(table_a[show].to_string(index=False))
    print("\nB. TEXT - annual-report subset, 12m (mean ± std over folds)")
    print(table_b[show].to_string(index=False))
    print(f"\nSaved: {REPORTS / 'ablation_cv.csv'}, {REPORTS / 'text_cv.csv'}, {P / 'final_scores.csv'}")


if __name__ == "__main__":
    main()