"""
check_coverage.py
-----------------
Links each Screener file back to its IBBI record (via the 001_ number prefix = row in the
download list) and checks whether we have the financial years BEFORE insolvency.

Inputs : data/processed/download_list_distressed.csv  (made by make_download_list.py)
         data/processed/financials.csv                (made by parse_screener.py)
Output : data/processed/distressed_linked.csv         (one row per distressed company)

Run from the project root:
    python src/check_coverage.py

Timing rule (avoids using future information):
  Annual results for the year ending March Y are public by about 30 June Y.
  So the last financial year we may use is the latest FY whose 30 June is BEFORE the
  insolvency date:  insolvency Jul-Dec of year E -> FY E;  Jan-Jun of year E -> FY E-1.
"""

from pathlib import Path
import re

import pandas as pd

LIST_FILE = Path("data/processed/download_list_distressed.csv")
FIN_FILE = Path("data/processed/financials.csv")
OUT_FILE = Path("data/processed/distressed_linked.csv")
WINDOW = 3  # years of history the GRU will use


def prediction_fy(event_date):
    return event_date.year if event_date.month > 6 else event_date.year - 1


def name_matches(search_name, company):
    """True if our name and Screener's own company name share a word (3+ letters)."""
    words = [w for w in re.findall(r"[a-z]+", str(search_name).lower()) if len(w) > 2]
    return any(w in str(company).lower() for w in words)


def main():
    lst = pd.read_csv(LIST_FILE, parse_dates=["cirp_start_date"])
    lst["row_no"] = range(1, len(lst) + 1)          # file 001_ = first row of the list

    fin = pd.read_csv(FIN_FILE)
    fin["row_no"] = pd.to_numeric(fin["source_file"].str.extract(r"^(\d+)_")[0], errors="coerce")
    unnumbered = fin.loc[fin["row_no"].isna(), "source_file"].unique()

    links = []
    for (row_no, src, company), g in fin.dropna(subset=["row_no"]).groupby(["row_no", "source_file", "company"]):
        rec = lst[lst["row_no"] == row_no]
        if rec.empty:
            continue
        rec = rec.iloc[0]
        pfy = prediction_fy(rec["cirp_start_date"])
        years = set(g["fiscal_year"].dropna().astype(int))
        usable = sorted(y for y in years if y <= pfy)              # years known before the event
        in_window = [y for y in range(pfy - WINDOW + 1, pfy + 1) if y in years]
        links.append({
            "row_no": int(row_no),
            "source_file": src,
            "screener_name": company,
            "search_name": rec["search_name"],
            "cin": rec["cin"],
            "cirp_start_date": rec["cirp_start_date"].date(),
            "prediction_fy": pfy,
            "first_fy_in_file": min(years) if years else None,
            "last_fy_in_file": max(years) if years else None,
            "years_before_event": len(usable),
            "latest_usable_fy": max(usable) if usable else None,
            "years_in_window": len(in_window),
            "name_match": name_matches(rec["search_name"], company),
        })

    df = pd.DataFrame(links).sort_values("row_no")
    df["staleness_years"] = df["prediction_fy"] - df["latest_usable_fy"]
    df["coverage"] = pd.cut(df["years_in_window"], bins=[-1, 0, WINDOW - 1, WINDOW],
                            labels=["none", "partial", "full"])
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"Linked {len(df)} Screener files to IBBI records")
    if len(unnumbered):
        print(f"  ignored {len(unnumbered)} file(s) without a number prefix: {list(unnumbered)}")

    print(f"\nCoverage of the {WINDOW} financial years just before insolvency:")
    print(df["coverage"].value_counts().reindex(["full", "partial", "none"]).to_string())

    print("\nCompanies with ANY usable pre-insolvency year:", (df["years_before_event"] > 0).sum())
    stale = df[(df["years_before_event"] > 0) & (df["staleness_years"] > 0)]
    print(f"  of which latest data is older than the prediction year (stopped filing): {len(stale)}")

    none = df[df["years_before_event"] == 0]
    if len(none):
        print(f"\nNO data before insolvency ({len(none)}) - file starts too late (usually renamed survivors):")
        print(none[["row_no", "search_name", "screener_name", "cirp_start_date", "first_fy_in_file"]]
              .to_string(index=False))

    mismatch = df[~df["name_match"]]
    if len(mismatch):
        print(f"\nName check - Screener name shares no word with our name ({len(mismatch)}). "
              "Expected for renamed companies; anything else is a wrong download:")
        print(mismatch[["row_no", "search_name", "screener_name"]].to_string(index=False))

    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()