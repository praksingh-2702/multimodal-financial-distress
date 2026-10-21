"""
explain_and_package.py
----------------------
Final step before the dashboard:
  1. Calibrated risk %  - turns the fused score into a realistic probability (Platt scaling,
                          cross-fitted by fold so it stays honest) + Low/Medium/High band
  2. Modality contributions - how much ratios / trends / peers pushed each prediction up or down
  3. SHAP               - which financial ratios drive the ratio model, for every company-year
  4. Early-warning stats - how many months before insolvency the model first raised a red flag
  5. Case-study charts  - risk over time for well-known collapses (Lanco, Alok, Jaypee...)
  6. Packages everything the dashboard needs into app_data/ (small, safe to commit to GitHub)

Input : data/processed/dataset.csv, final_scores.csv, peer_edges.csv, text_features.csv,
        top_negative_sentences.csv, models/xgb_baseline.json, reports/*_cv.csv
Output: app_data/*.csv, app_data/meta.json, app_data/figures/*.png, reports/figures/*.png

Run from the project root (~1-2 minutes):
    python src/explain_and_package.py
"""

import json
from pathlib import Path
import shutil
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from cv_utils import FEATURES, N_FOLDS, load_dataset, winsorize

warnings.filterwarnings("ignore")
plt.style.use("dark_background")                       # match the dashboard's dark theme
plt.rcParams.update({"figure.facecolor": "#0e1117", "axes.facecolor": "#0e1117",
                     "savefig.facecolor": "#0e1117", "axes.edgecolor": "#4a5563", "grid.color": "#2a313c"})
P = Path("data/processed")
APP = Path("app_data")
FIG = Path("reports/figures")
KEY = ["company_id", "fiscal_year"]
MODELS = ["xgb", "gru", "gnn"]
BANDS = [(0.20, "High"), (0.05, "Medium"), (0.0, "Low")]       # on calibrated 12-month risk
CASES = ["LANCO", "ALOK", "JAYPEE", "BHUSHAN", "TATA STEEL BSL", "AMTEK", "ABG SHIP", "IVRCL",
         "JET AIRWAYS", "VIDEOCON", "ESL STEEL", "JSW ISPAT"]
DISPLAY_RATIOS = ["debt_to_equity", "interest_coverage", "roa", "ebit_margin", "liquidity_proxy",
                  "cfo_to_assets", "altman_z_proxy", "log_assets"]


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def calibrate(score, y, folds):
    """Platt scaling, cross-fitted: each fold's calibration is learned on the other folds."""
    out = np.zeros(len(score))
    x = logit(score).reshape(-1, 1)
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        out[te] = LogisticRegression(C=1e6).fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return out


def band(p):
    for cut, name in BANDS:
        if p >= cut:
            return name
    return "Low"


def main():
    APP.mkdir(exist_ok=True)
    (APP / "figures").mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    ds = load_dataset().reset_index(drop=True)          # same row order the GNN used
    sc = pd.read_csv(P / "final_scores.csv")
    df = ds.merge(sc[KEY + [f"{m}_{h}" for m in MODELS for h in ["12m", "24m"]] +
                     ["fusion_stacked_12m", "fusion_stacked_24m", "text_score", "save_as"]],
                  on=KEY, how="left", validate="one_to_one")
    folds = df["fold"].values

    # ---- 1. calibrated risk ----------------------------------------------------
    df["risk_12m"] = calibrate(df["fusion_stacked_12m"].values, df["label"].values, folds)
    df["risk_24m"] = calibrate(df["fusion_stacked_24m"].values, df["label_24m"].values, folds)
    df["band"] = df["risk_12m"].map(band)
    df["risk_percentile"] = df["risk_12m"].rank(pct=True)

    # ---- 2. modality contributions (stacked weights x centred score) ---------------
    X = np.column_stack([logit(df[f"{m}_12m"].values) for m in MODELS])
    w = LogisticRegression(class_weight="balanced").fit(X, df["label"]).coef_[0]
    for i, m in enumerate(MODELS):
        df[f"contrib_{m}"] = w[i] * (X[:, i] - X[:, i].mean())
    weights = {m: round(float(v), 3) for m, v in zip(MODELS, w)}

    # ---- 3. SHAP on the all-data XGBoost ratio model ------------------------------
    X_all, = winsorize(ds[FEATURES])
    xgb = XGBClassifier()
    xgb.load_model("models/xgb_baseline.json")
    sv = shap.TreeExplainer(xgb).shap_values(X_all)
    shap_df = pd.DataFrame(sv, columns=FEATURES).round(4)
    shap_df.insert(0, "fiscal_year", ds["fiscal_year"].values)
    shap_df.insert(0, "company_id", ds["company_id"].values)

    plt.figure()
    shap.summary_plot(sv, X_all, show=False, max_display=12)
    plt.title("What drives the ratio model (SHAP, 12-month risk)")
    plt.tight_layout()
    plt.savefig(FIG / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close("all")
    global_imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False)

    # ---- 4. early-warning lead time -------------------------------------------------
    dist = df[df["group"] == "distressed"].copy()
    flagged = dist[dist["risk_12m"] >= BANDS[0][0]]
    first_flag = flagged.sort_values("fiscal_year").groupby("company_id").first()
    n_dist = dist["company_id"].nunique()
    healthy = df[df["group"] == "healthy"]
    lead = {
        "distressed_companies": int(n_dist),
        "flagged_high_at_least_once": int(len(first_flag)),
        "share_flagged": round(len(first_flag) / max(n_dist, 1), 3),
        "median_months_warning": round(float(first_flag["months_to_event"].median()), 1)
        if len(first_flag) else None,
        "share_flagged_12m_or_earlier": round(float((first_flag["months_to_event"] >= 12).mean()), 3)
        if len(first_flag) else None,
        "healthy_company_years_high": round(float((healthy["band"] == "High").mean()), 3),
    }

    # ---- 5. case studies -------------------------------------------------------------
    names = df["company"].str.upper()
    n_points = df.groupby("company_id").size()
    picked = []
    for c in CASES:                      # famous names first, but only with >= 3 years of history
        ids = df.loc[names.str.contains(c, regex=False), "company_id"].unique()
        if len(ids) and ids[0] not in picked and n_points[ids[0]] >= 3:
            picked.append(ids[0])
    if len(picked) < 6:                  # top up with the largest-history distressed companies
        more = (dist.groupby("company_id").size().sort_values(ascending=False).index)
        picked += [c for c in more if c not in picked][: 6 - len(picked)]
    picked = picked[:6]
    if picked:
        n = len(picked)
        fig, axes = plt.subplots((n + 2) // 3, min(n, 3), figsize=(5 * min(n, 3), 3.6 * ((n + 2) // 3)),
                                 squeeze=False)
        for ax, cid in zip(axes.flat, picked):
            d = df[df["company_id"] == cid].sort_values("fiscal_year")
            x = pd.to_datetime(d["obs_date"])
            ax.plot(x, d["risk_12m"] * 100, marker="o", color="#6fa8dc", label="12-month risk")
            ax.plot(x, d["risk_24m"] * 100, marker=".", ls="--", color="#9aa4ae", label="24-month risk")
            ev = pd.to_datetime(d["cirp_start_date"].iloc[0])
            ax.axvline(ev, color="#e6e6e6", lw=1.2)
            ax.text(ev, ax.get_ylim()[1] * 0.92, " insolvency", fontsize=8)
            ax.axhline(BANDS[0][0] * 100, color="#e69138", lw=0.8, ls=":")
            ax.set_title(d["company"].iloc[0][:32], fontsize=10)
            ax.set_ylabel("risk %")
            ax.set_ylim(0, 100)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(axis="x", labelsize=8)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=7, loc="upper left")
        fig.suptitle("Model risk before insolvency (each point uses only data public at that date)")
        fig.tight_layout()
        fig.savefig(FIG / "case_studies.png", dpi=130)
        plt.close(fig)

    # ---- 6. package app data ------------------------------------------------------------
    keep = (KEY + ["company", "group", "industry", "cin", "symbol", "obs_date", "cirp_start_date",
                   "months_to_event", "label", "label_24m", "risk_12m", "risk_24m", "band",
                   "risk_percentile", "text_score", "save_as"]
            + [f"{m}_12m" for m in MODELS] + [f"contrib_{m}" for m in MODELS] + DISPLAY_RATIOS)
    df[keep].round(5).to_csv(APP / "scores.csv", index=False)
    shap_df.to_csv(APP / "shap.csv", index=False)

    edges = pd.read_csv(P / "peer_edges.csv")
    cid = ds["company_id"].values
    edges["src_id"], edges["dst_id"] = cid[edges["src"]], cid[edges["dst"]]
    edges[["fiscal_year", "src_id", "dst_id", "kind"]].to_csv(APP / "peers.csv", index=False)

    tf_path, sent_path = P / "text_features.csv", P / "top_negative_sentences.csv"
    if tf_path.exists():
        tf = pd.read_csv(tf_path)
        tcols = ["save_as", "company_id", "fiscal_year", "report_fy"] + \
                [c for c in tf.columns if c.startswith(("fb_", "kw_"))]
        tf[[c for c in tcols if c in tf.columns]].round(4).to_csv(APP / "text.csv", index=False)
    if sent_path.exists():
        pd.read_csv(sent_path).to_csv(APP / "sentences.csv", index=False)

    (APP / "results").mkdir(exist_ok=True)
    for f in ["ablation_cv.csv", "text_cv.csv", "baseline_cv.csv", "gru_cv.csv", "gnn_cv.csv"]:
        if (Path("reports") / f).exists():
            shutil.copy(Path("reports") / f, APP / "results" / f)
    for f in ["shap_summary.png", "case_studies.png"]:
        if (FIG / f).exists():
            shutil.copy(FIG / f, APP / "figures" / f)

    meta = {
        "company_years": int(len(df)), "companies": int(df["company_id"].nunique()),
        "distressed_companies": int(n_dist), "healthy_companies": int(healthy["company_id"].nunique()),
        "positive_rows_12m": int(df["label"].sum()), "fiscal_years": [int(df.fiscal_year.min()), int(df.fiscal_year.max())],
        "stacked_weights_12m": weights, "risk_bands": {n: c for c, n in BANDS},
        "early_warning": lead, "top_shap_features": global_imp.head(8).round(4).to_dict(),
        "case_study_ids": picked,
    }
    (APP / "meta.json").write_text(json.dumps(meta, indent=2))

    # ---- summary -----------------------------------------------------------------------
    print(f"Company-years: {len(df)} | bands: " +
          ", ".join(f"{k} {v}" for k, v in df["band"].value_counts().items()))
    print(f"Mean calibrated 12m risk: {df['risk_12m'].mean():.3f} (actual positive rate {df['label'].mean():.3f})")
    print(f"Stacked weights (12m): {weights}")
    print("\nEARLY WARNING (12m risk >= 20% = 'High'):")
    for k, v in lead.items():
        print(f"  {k}: {v}")
    print("\nTop SHAP features (ratio model):")
    print(global_imp.head(8).round(4).to_string())
    print(f"\nCase studies: {', '.join(df.loc[df.company_id.isin(picked), 'company'].unique())}")
    size = sum(f.stat().st_size for f in APP.rglob("*") if f.is_file()) / 1e6
    print(f"\napp_data/ size: {size:.1f} MB  ({len(list(APP.rglob('*.*')))} files)")
    print("Saved: app_data/, reports/figures/shap_summary.png, reports/figures/case_studies.png")


if __name__ == "__main__":
    main()