"""
make_report_list.py
-------------------
Builds the annual-report download list for the FinBERT (text) part.
One report per company: ~90 distressed + ~90 healthy.

Input : data/processed/dataset.csv                 (build_dataset.py)
        data/processed/download_list_distressed.csv
        data/processed/download_list_healthy.csv
Output: data/processed/download_list_reports.csv
        data/processed/text_rows.csv   (which dataset row each report belongs to - used later)

Run from the project root:
    python src/make_report_list.py

WHICH YEAR'S REPORT (no future information)
  Indian companies must hold their AGM by 30 September, so the annual report for the year
  ending March FY is public by 30 Sept FY at the latest.
  Distressed: we take the row labelled 1 (insolvency within 12 months). If insolvency started
  AFTER 30 Sept FY, the FY report was already public -> use it. Otherwise use the FY-1 report.
  Healthy: a random year, drawn to match the distressed years, so both groups cover the same period.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

DATA = Path("data/processed/dataset.csv")
DIST_LIST = Path("data/processed/download_list_distressed.csv")
HEALTHY_LIST = Path("data/processed/download_list_healthy.csv")
OUT_LIST = Path("data/processed/download_list_reports.csv")
OUT_ROWS = Path("data/processed/text_rows.csv")

N_PER_GROUP = 90
SEED = 42
PEOPLE = ["Person 1", "Person 2", "Person 3"]


def fy_label(fy):
    return f"FY{fy} (Apr {fy - 1} - Mar {fy}, i.e. {fy - 1}-{str(fy)[2:]})"


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(DATA, parse_dates=["cirp_start_date"])
    df["prefix"] = df["company_id"].str.extract(r"^(H?\d+)_")[0]

    # ---------- distressed: the positive row, biggest cases first -------------
    dl = pd.read_csv(DIST_LIST)
    dl["prefix"] = [f"{i + 1:03d}" for i in range(len(dl))]
    pos = df[(df["group"] == "distressed") & (df["label"] == 1)].copy()
    pos = pos.merge(dl[["prefix", "search_name", "total_admitted_claims_cr"]], on="prefix", how="left")
    pos = pos.sort_values("total_admitted_claims_cr", ascending=False).head(N_PER_GROUP)
    sept30 = pd.to_datetime(pos["fiscal_year"].astype(str) + "-09-30")
    pos["report_fy"] = np.where(pos["cirp_start_date"] > sept30, pos["fiscal_year"], pos["fiscal_year"] - 1)
    pos["how_to_find"] = "Search on screener.in: " + pos["search_name"].astype(str)

    # ---------- healthy: one random row per company, years matched -------------
    hl = pd.read_csv(HEALTHY_LIST).rename(columns={"file_prefix": "prefix"})
    hrows = df[df["group"] == "healthy"].merge(hl[["prefix", "screener_url"]], on="prefix", how="left")
    target_years = pos["report_fy"].values
    picks = []
    companies = np.array(sorted(hrows["company_id"].unique()), dtype=object)
    rng.shuffle(companies)
    for cid in companies[: N_PER_GROUP * 2]:
        rows = hrows[hrows["company_id"] == cid]
        want = int(rng.choice(target_years))
        rows = rows.assign(dist=(rows["fiscal_year"] - want).abs())
        picks.append(rows.sort_values("dist").iloc[0])
        if len(picks) == N_PER_GROUP:
            break
    neg = pd.DataFrame(picks)
    neg["report_fy"] = neg["fiscal_year"]
    neg["how_to_find"] = "Open: " + neg["screener_url"].astype(str)

    # ---------- combine ---------------------------------------------------------
    cols = ["company_id", "company", "group", "fiscal_year", "label", "report_fy", "how_to_find"]
    rows = pd.concat([pos[cols], neg[cols]], ignore_index=True)
    rows["save_as"] = "AR_" + rows["company_id"].str.extract(r"^(H?\d+)_")[0] + \
                      "_FY" + rows["report_fy"].astype(int).astype(str) + ".pdf"
    # interleave so every person gets both groups and the most important cases come first
    rows["order"] = rows.groupby("group").cumcount()
    rows = rows.sort_values(["order", "group"]).reset_index(drop=True)
    rows["priority"] = range(1, len(rows) + 1)
    rows["assigned_to"] = [PEOPLE[i % len(PEOPLE)] for i in range(len(rows))]
    rows["which_report"] = rows["report_fy"].astype(int).map(fy_label)
    rows["downloaded"] = ""
    rows["notes"] = ""

    OUT_LIST.parent.mkdir(parents=True, exist_ok=True)
    rows[["priority", "save_as", "company", "group", "which_report", "how_to_find",
          "assigned_to", "downloaded", "notes"]].to_csv(OUT_LIST, index=False)
    rows[["company_id", "fiscal_year", "label", "report_fy", "save_as"]].to_csv(OUT_ROWS, index=False)

    print(f"Reports to download: {len(rows)}  (distressed {len(pos)}, healthy {len(neg)})")
    print("\nReport years (distressed vs healthy):")
    print(pd.crosstab(rows["report_fy"], rows["group"]).to_string())
    print("\nPer person:")
    print(rows["assigned_to"].value_counts().sort_index().to_string())
    print("\nFirst 6 rows:")
    print(rows[["priority", "save_as", "company", "which_report"]].head(6).to_string(index=False))
    print(f"\nSaved: {OUT_LIST}")
    print(f"Saved: {OUT_ROWS}")


if __name__ == "__main__":
    main()