"""
build_labels.py
---------------
Builds the distress-event list (our labels) from IBBI's insolvency outcome files.

Inputs : data/raw/ibbi/resolution.xlsx   (CIRPs that ended with a resolution plan)
         data/raw/ibbi/liquidation.xlsx  (CIRPs that ended with liquidation)
Outputs: data/processed/ibbi_all_cirp.csv      every company in both files
         data/processed/distressed_listed.csv  listed companies only = our distress events

Run from the project root:
    python src/build_labels.py
"""

from pathlib import Path
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

RAW = Path("data/raw/ibbi")
OUT = Path("data/processed")

# A valid Indian CIN looks like: L32105MH1985PLC035516
#   L      -> listed (U = unlisted)
#   32105  -> NIC industry code (we use this for the GNN peer graph)
#   MH     -> state
#   1985   -> year of incorporation
#   PLC    -> company type (PLC public, PTC private, GOI government...)
#   035516 -> registration number
CIN_REGEX = r"^[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}$"


def find_col(df, *keywords):
    """Return the first column whose (lower-cased) name contains all keywords.
    IBBI's two files name the same column slightly differently
    (e.g. 'CIN' vs 'CIN Number', 'Triggered' vs 'Trigerred'), so we match loosely."""
    for col in df.columns:
        name = " ".join(str(col).lower().split())
        if all(k in name for k in keywords):
            return col
    raise KeyError(f"No column containing {keywords}. Columns are: {list(df.columns)}")


def load(file_name, outcome, outcome_date_keywords):
    """Read one IBBI Excel file and return it with standard column names."""
    df = pd.read_excel(RAW / file_name, header=3)  # first 3 rows are titles; row 4 has column names
    clean = pd.DataFrame({
        "cin": df[find_col(df, "cin")],
        "company_name": df[find_col(df, "name")],
        "triggered_by": df[find_col(df, "trig")],
        "nclt_bench": df[find_col(df, "bench")],
        "cirp_start_date": df[find_col(df, "commencement")],
        "outcome_date": df[find_col(df, *outcome_date_keywords)],
        "total_admitted_claims_cr": df[find_col(df, "total admitted")],
    })
    clean["outcome"] = outcome
    return clean


def main():
    res = load("resolution.xlsx", "resolution_plan", ("date", "resolution"))
    liq = load("liquidation.xlsx", "liquidation", ("date", "liquidation"))
    df = pd.concat([res, liq], ignore_index=True)

    # --- Clean ---------------------------------------------------------------
    df["cin"] = df["cin"].astype(str).str.strip().str.upper()
    df = df[df["cin"].str.match(CIN_REGEX)].copy()  # drops blank / footer / malformed rows
    df["company_name"] = df["company_name"].astype(str).str.strip()
    df["cirp_start_date"] = pd.to_datetime(df["cirp_start_date"], errors="coerce", dayfirst=True)
    df["outcome_date"] = pd.to_datetime(df["outcome_date"], errors="coerce", dayfirst=True)
    df["total_admitted_claims_cr"] = pd.to_numeric(df["total_admitted_claims_cr"], errors="coerce")

    # --- Decode the CIN ------------------------------------------------------
    df["is_listed"] = df["cin"].str[0] == "L"
    df["nic_code"] = df["cin"].str[1:6]
    df["nic_2digit"] = df["cin"].str[1:3]
    df["state"] = df["cin"].str[6:8]
    df["incorporation_year"] = df["cin"].str[8:12].astype(int)
    df["company_type"] = df["cin"].str[12:15]
    df["event_year"] = df["cirp_start_date"].dt.year

    # A company can appear twice (e.g. a second insolvency case). Keep its FIRST event.
    df = df.sort_values("cirp_start_date").drop_duplicates("cin", keep="first")

    listed = (df[df["is_listed"]]
              .sort_values("total_admitted_claims_cr", ascending=False)
              .reset_index(drop=True))

    # --- Save ----------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ibbi_all_cirp.csv", index=False)
    listed.to_csv(OUT / "distressed_listed.csv", index=False)

    # --- Summary -------------------------------------------------------------
    print(f"All completed CIRPs (unique companies): {len(df)}")
    print(f"  of which LISTED (CIN starts with L):   {len(listed)}")
    print(f"  missing start dates among listed:      {listed['cirp_start_date'].isna().sum()}")

    print("\nListed companies by year insolvency started:")
    print(listed["event_year"].value_counts().sort_index().to_string())

    print("\nListed companies by outcome:")
    print(listed["outcome"].value_counts().to_string())

    claims = listed["total_admitted_claims_cr"]
    print("\nListed companies by size of admitted claims (Rs crore):")
    print(f"  >= 100 cr : {(claims >= 100).sum()}")
    print(f"  >= 500 cr : {(claims >= 500).sum()}")
    print(f"  >= 1000 cr: {(claims >= 1000).sum()}")

    print("\nTop 15 listed companies by admitted claims:")
    cols = ["company_name", "cirp_start_date", "outcome", "total_admitted_claims_cr", "nic_code"]
    print(listed[cols].head(15).to_string())

    print("\nSanity check - well-known distressed names:")
    known = ["videocon", "alok industries", "bhushan steel", "jet airways", "jaypee infratech",
             "amtek auto", "monnet ispat", "lanco infratech", "electrosteel steels", "abg shipyard"]
    names = df["company_name"].str.lower()
    for k in known:
        hits = df[names.str.contains(k, na=False, regex=False)]
        if hits.empty:
            print(f"  {k:22s} -> not in these two files")
        else:
            row = hits.iloc[0]
            print(f"  {k:22s} -> found | listed={row['is_listed']} | start={row['cirp_start_date'].date()} | {row['outcome']}")

    print(f"\nSaved: {OUT / 'ibbi_all_cirp.csv'}")
    print(f"Saved: {OUT / 'distressed_listed.csv'}")


if __name__ == "__main__":
    main()