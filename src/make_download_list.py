"""
make_download_list.py
---------------------
Filters the IBBI distressed list down to the companies we will download from Screener,
and splits them between 3 people.

Input : data/processed/distressed_listed.csv   (made by build_labels.py)
Output: data/processed/download_list_distressed.csv

Run from the project root:
    python src/make_download_list.py

Rules (write these in the report's Data section):
  1. Insolvency started 2017-2022 (2023+ is incomplete because ongoing cases aren't in IBBI's lists)
  2. Admitted claims >= Rs 100 crore (tiny firms rarely have usable financial data)
  3. Exclude banks / NBFCs / housing finance (their balance sheets are not comparable)
"""

from pathlib import Path
import re

import pandas as pd

IN_FILE = Path("data/processed/distressed_listed.csv")
OUT_FILE = Path("data/processed/download_list_distressed.csv")

MIN_CLAIMS_CR = 100
YEARS = range(2017, 2023)
# Financial firms are identified by NAME only. CIN industry codes can't be used here:
# older CINs use an older NIC version where codes 64-67 mean non-financial industries.
FINANCIAL_WORDS = re.compile(r"financ|housing|leasing|\bbank|capital ltd|capital limited", re.I)
PEOPLE = ["Person 1", "Person 2", "Person 3"]


def clean_name(name):
    """Name to type into Screener's search box."""
    name = re.sub(r"\(.*?\)", "", str(name))                      # drop '(In Liquidation)' etc.
    name = re.sub(r"\b(limited|ltd\.?|pvt\.?|private)\b", "", name, flags=re.I)
    return " ".join(name.split()).strip(" .,-")


def main():
    df = pd.read_csv(IN_FILE, dtype={"nic_code": str, "nic_2digit": str},
                     parse_dates=["cirp_start_date"])
    start = len(df)

    df = df[df["event_year"].isin(YEARS)]
    after_years = len(df)
    df = df[df["total_admitted_claims_cr"] >= MIN_CLAIMS_CR]
    after_size = len(df)
    is_fin = df["company_name"].str.contains(FINANCIAL_WORDS)
    excluded_fin = df[is_fin]
    df = df[~is_fin]

    # Biggest cases first, so if time runs out the most important ones are done
    df = df.sort_values("total_admitted_claims_cr", ascending=False).reset_index(drop=True)
    df["search_name"] = df["company_name"].map(clean_name)
    df["last_fy_needed"] = df["cirp_start_date"].dt.year - 1   # last full year before insolvency
    df["assigned_to"] = [PEOPLE[i % len(PEOPLE)] for i in range(len(df))]
    df["downloaded"] = ""
    df["notes"] = ""

    cols = ["search_name", "company_name", "cin", "cirp_start_date", "last_fy_needed",
            "total_admitted_claims_cr", "nic_code", "assigned_to", "downloaded", "notes"]
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(OUT_FILE, index=False)

    print(f"Listed distressed companies:          {start}")
    print(f"  after 2017-2022 filter:             {after_years}")
    print(f"  after >= Rs {MIN_CLAIMS_CR} cr claims filter:   {after_size}")
    print(f"  after excluding financial firms:    {len(df)}  ({len(excluded_fin)} removed)")
    print("\nRemoved as financial (check none are wrongly removed):")
    print("  " + ", ".join(excluded_fin["company_name"].head(25).tolist()))
    print("\nPer person:")
    print(df["assigned_to"].value_counts().sort_index().to_string())
    print("\nFirst 10 to download:")
    print(df[["search_name", "cirp_start_date", "total_admitted_claims_cr", "assigned_to"]]
          .head(10).to_string(index=False))
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()