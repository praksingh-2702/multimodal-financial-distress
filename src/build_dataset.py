"""
build_dataset.py
----------------
Builds the final training table: one row per company per fiscal year, with a 0/1 label.

Inputs : data/processed/financials.csv           (parse_screener.py)
         data/processed/distressed_linked.csv    (check_coverage.py)
         data/processed/distressed_industry.csv  (make_healthy_list.py)
         data/processed/download_list_healthy.csv
Output : data/processed/dataset.csv

Run from the project root:
    python src/build_dataset.py

LABEL DEFINITION (put this in the report)
  A row = company c, fiscal year FY (year ending 31 March FY).
  Its annual results are assumed public on 30 June FY  -> that is the "observation date".
  label     = 1 if the company's insolvency (CIRP) started within 12 months after the observation date
  label_24m = 1 if it started within 24 months (secondary, for robustness)
  Rows observed on/after the insolvency date are dropped (that would be using the future).
  Healthy companies: label 0 for every year.

SPLIT
  Companies (not rows) are split 70/15/15 into train/val/test, separately for distressed and
  healthy, so one company never appears in two splits.
"""

from pathlib import Path

import numpy as np
import pandas as pd

FIN = Path("data/processed/financials.csv")
DIST_LINKED = Path("data/processed/distressed_linked.csv")
DIST_IND = Path("data/processed/distressed_industry.csv")
HEALTHY_LIST = Path("data/processed/download_list_healthy.csv")
OUT = Path("data/processed/dataset.csv")

FY_MIN, FY_MAX = 2014, 2022      # common window where both groups have data
# Healthy files (current Screener exports) start at FY2017; distressed files start much earlier.
# So year-on-year features may only look back to FY2017, for EVERY company - otherwise
# "trend is missing" would give away "this is a healthy company".
LAG_MIN_FY = 2017
LAG_FEATURES = ["sales_growth", "borrowings_growth", "price_change",
                "d_roa", "d_debt_to_assets", "d_interest_coverage"]
SEED = 42
SPLIT_SHARES = (0.70, 0.15, 0.15)

# Features used by the models (all computed from information public at the observation date)
FEATURES = [
    "debt_to_equity", "debt_to_assets", "interest_coverage", "liquidity_proxy", "cash_to_assets",
    "roa", "ebit_margin", "asset_turnover", "cfo_to_assets", "cfo_to_debt", "mcap_to_debt",
    "altman_z_proxy", "sales_growth", "borrowings_growth", "price_change", "negative_net_worth",
    "log_assets", "d_roa", "d_debt_to_assets", "d_interest_coverage", "n_missing",
]


def main():
    fin = pd.read_csv(FIN, parse_dates=["fy_end"])
    fin["company_id"] = fin["source_file"].str.replace(r"\.xlsx$", "", regex=True)

    # Fiscal-year label such that results are public by the 30 June observation date:
    # year ending Jan-Mar Y -> FY Y; ending Apr-Dec Y (e.g. Castrol's December year) -> FY Y+1.
    # Companies that changed their year-end can then still have 2 reports in one FY -> keep latest.
    month = fin["fy_end"].dt.month
    non_march = int((month != 3).sum())
    fin["fiscal_year"] = np.where(month <= 3, fin["fy_end"].dt.year, fin["fy_end"].dt.year + 1)
    before_dedup = len(fin)
    fin = fin.sort_values(["company_id", "fy_end"]).drop_duplicates(["company_id", "fiscal_year"], keep="last")
    n_dupes = before_dedup - len(fin)
    fin["prefix"] = fin["source_file"].str.extract(r"^(H?\d+)_")[0]
    fin["group"] = np.where(fin["prefix"].str.startswith("H"), "healthy", "distressed")

    # ---- attach event dates + industry ---------------------------------------
    linked = pd.read_csv(DIST_LINKED, parse_dates=["cirp_start_date"])
    ind = pd.read_csv(DIST_IND)[["row_no", "industry"]]
    dist_meta = linked.merge(ind, on="row_no", how="left")
    dist_meta["prefix"] = dist_meta["row_no"].astype(int).astype(str).str.zfill(3)
    dist_meta = dist_meta[["prefix", "cin", "cirp_start_date", "industry"]]

    healthy = pd.read_csv(HEALTHY_LIST)
    healthy_meta = healthy.rename(columns={"file_prefix": "prefix", "Industry": "industry",
                                           "Symbol": "symbol"})[["prefix", "symbol", "industry"]]

    fin["prefix_num"] = fin["prefix"].where(fin["group"] == "healthy",
                                            fin["prefix"].str.zfill(3))
    df = fin.merge(dist_meta, left_on="prefix_num", right_on="prefix", how="left", suffixes=("", "_d"))
    df = df.merge(healthy_meta, left_on="prefix_num", right_on="prefix", how="left",
                  suffixes=("", "_h"))
    df["industry"] = df["industry"].fillna(df["industry_h"])
    df = df.drop(columns=[c for c in df.columns if c.endswith(("_d", "_h"))] + ["prefix_num"])

    unlinked = df[(df["group"] == "distressed") & df["cirp_start_date"].isna()]["source_file"].unique()
    df = df[~((df["group"] == "distressed") & df["cirp_start_date"].isna())]

    # ---- extra features ------------------------------------------------------
    df = df.sort_values(["company_id", "fiscal_year"])
    g = df.groupby("company_id")
    df["log_assets"] = np.log1p(df["total_assets"].clip(lower=0))
    for col in ["roa", "debt_to_assets", "interest_coverage"]:
        df[f"d_{col}"] = df[col] - g[col].shift(1)
    no_fair_lag = (df["fiscal_year"] - 1) < LAG_MIN_FY
    df.loc[no_fair_lag, LAG_FEATURES] = np.nan
    raw_core = ["sales", "net_profit", "borrowings", "total_assets", "interest", "cfo", "price"]
    df["n_missing"] = df[raw_core].isna().sum(axis=1)

    # ---- labels --------------------------------------------------------------
    df["obs_date"] = pd.to_datetime(df["fiscal_year"].astype(int).astype(str) + "-06-30")
    gap_days = (df["cirp_start_date"] - df["obs_date"]).dt.days
    df["months_to_event"] = (gap_days / 30.44).round(1)

    before = len(df)
    df = df[~((df["group"] == "distressed") & (gap_days <= 0))]          # drop post-event rows
    dropped_post = before - len(df)
    gap_days = (df["cirp_start_date"] - df["obs_date"]).dt.days
    df["label"] = ((df["group"] == "distressed") & (gap_days <= 365)).astype(int)
    df["label_24m"] = ((df["group"] == "distressed") & (gap_days <= 730)).astype(int)

    # ---- window + rows with almost no data -----------------------------------
    df = df[df["fiscal_year"].between(FY_MIN, FY_MAX)]
    too_empty = df["total_assets"].isna() | (df["total_assets"] <= 0)
    df = df[~too_empty]

    # ---- company-level split -------------------------------------------------
    rng = np.random.default_rng(SEED)
    split_of = {}
    for grp in ["distressed", "healthy"]:
        ids = np.array(sorted(df.loc[df["group"] == grp, "company_id"].unique()))
        rng.shuffle(ids)
        n = len(ids)
        n_tr, n_va = int(round(n * SPLIT_SHARES[0])), int(round(n * SPLIT_SHARES[1]))
        for i, cid in enumerate(ids):
            split_of[cid] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    df["split"] = df["company_id"].map(split_of)

    keep = ["company_id", "company", "group", "industry", "cin", "symbol", "fiscal_year", "obs_date",
            "cirp_start_date", "months_to_event", "label", "label_24m", "split"] + FEATURES
    df = df[keep].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # ---- summary -------------------------------------------------------------
    print(f"Rows: {len(df)} company-years | companies: {df['company_id'].nunique()} "
          f"(distressed {df.loc[df.group == 'distressed', 'company_id'].nunique()}, "
          f"healthy {df.loc[df.group == 'healthy', 'company_id'].nunique()})")
    print(f"Window FY{FY_MIN}-FY{FY_MAX} | post-event rows dropped: {dropped_post}")
    print(f"Reports with a non-March year end: {non_march} | duplicate company-years removed: {n_dupes}")
    assert not df.duplicated(["company_id", "fiscal_year"]).any(), "duplicate company-years remain"
    if len(unlinked):
        print(f"Distressed files with no IBBI link (skipped): {len(unlinked)}")
    print(f"\nPositive rows (insolvency within 12m): {df['label'].sum()}  "
          f"({df['label'].mean():.1%} of rows) | within 24m: {df['label_24m'].sum()}")

    print("\nBy split:")
    s = df.groupby("split").agg(rows=("label", "size"), positives=("label", "sum"),
                                companies=("company_id", "nunique"))
    print(s.reindex(["train", "val", "test"]).to_string())

    print("\nBy fiscal year (rows / positives / healthy rows):")
    y = df.groupby("fiscal_year").agg(rows=("label", "size"), positives=("label", "sum"),
                                      healthy_rows=("group", lambda x: (x == "healthy").sum()))
    print(y.to_string())

    print("\nFeature missing rates (top 8):")
    print(df[FEATURES].isna().mean().sort_values(ascending=False).head(8).round(2).to_string())
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()