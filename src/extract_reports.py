"""
extract_reports.py
------------------
Reads every annual-report PDF, pulls out the Directors' Report + Management Discussion &
Analysis (MD&A) text for FinBERT, and counts red-flag phrases across the whole report
(auditor warnings, defaults, restructuring...).

Input : data/raw/reports/AR_<prefix>_FY<year>.pdf
        data/processed/text_rows.csv          (make_report_list.py)
Output: data/processed/report_text/<name>.txt  section text, one file per report
        data/processed/report_meta.csv         one row per report: pages, section found,
                                               red-flag counts, leakage flag

Run from the project root (~5-15 minutes for ~150 reports):
    python src/extract_reports.py
"""

from pathlib import Path
import re

import pandas as pd
import pymupdf

REPORTS = Path("data/raw/reports")
TEXT_ROWS = Path("data/processed/text_rows.csv")
OUT_DIR = Path("data/processed/report_text")
OUT_META = Path("data/processed/report_meta.csv")

MAX_SECTION_PAGES = 40
FALLBACK_PAGES = (2, 30)          # used when no section heading is found

START = re.compile(r"(directors['’`]?\s*report|board['’`]?s?\s*report|report\s+of\s+the\s+(board|directors)"
                   r"|management\s+discussion\s+(and|&)\s+analysis)", re.I)
STOP = re.compile(r"(report\s+on\s+corporate\s+governance|corporate\s+governance\s+report"
                  r"|independent\s+auditor['’`]?s?\s+report|auditors?['’`]?\s+report"
                  r"|business\s+responsibility|balance\s+sheet\s+as\s+at)", re.I)

# Red-flag phrases, counted over the WHOLE report (auditor's report included)
FLAGS = {
    "going_concern": r"going\s+concern",
    "material_uncertainty": r"material\s+uncertainty",
    "qualified_opinion": r"qualified\s+opinion|basis\s+for\s+qualified|adverse\s+opinion|disclaimer\s+of\s+opinion",
    "emphasis_of_matter": r"emphasis\s+of\s+matter",
    "default": r"\bdefault(ed|s)?\b|overdue|irregular\s+in\s+repayment",
    "npa": r"non[-\s]?performing\s+asset|\bnpa\b|special\s+mention\s+account|\bsma[-\s]?[012]\b",
    "restructuring": r"restructur|\bcdr\b|one[-\s]?time\s+settlement|\bots\b|\bs4a\b|strategic\s+debt",
    "erosion_net_worth": r"erosion\s+of\s+(net\s+worth|capital)|negative\s+net\s+worth",
    "liquidity_stress": r"liquidity\s+(crunch|stress|constraint|crisis)|cash\s+flow\s+(constraint|mismatch)",
    "nclt": r"\bnclt\b|national\s+company\s+law\s+tribunal|insolvency\s+and\s+bankruptcy\s+code|\bibc\b",
}
# The company ITSELF already under insolvency when the report was written -> would be leakage
LEAK = re.compile(r"interim\s+resolution\s+professional|resolution\s+professional\s+(of|for)\s+the\s+company"
                  r"|corporate\s+insolvency\s+resolution\s+process\s+(of|against|in\s+respect\s+of)\s+the\s+company"
                  r"|under\s+the\s+control\s+of\s+the\s+resolution\s+professional", re.I)


def is_toc_page(text):
    """Table-of-contents pages mention many section names and lots of page numbers."""
    hits = len(START.findall(text)) + len(STOP.findall(text))
    numbers = len(re.findall(r"^\s*\d{1,3}\s*$", text, re.M))
    return hits >= 3 or numbers >= 12


def heading_on_page(text, pattern):
    """True if the pattern appears near the top of the page (first ~12 lines) - i.e. a heading."""
    top = "\n".join(text.strip().splitlines()[:12])
    return bool(pattern.search(top))


def extract_one(path):
    doc = pymupdf.open(path)
    pages = [p.get_text("text") for p in doc]
    doc.close()
    full = "\n".join(pages)
    n_chars = len(full.strip())

    # find where the Directors' Report / MD&A starts and where it ends
    start = None
    for i, t in enumerate(pages):
        if i >= 2 and heading_on_page(t, START) and not is_toc_page(t):
            start = i
            break
    if start is not None:
        end = start + 1
        while end < len(pages) and end - start < MAX_SECTION_PAGES:
            if heading_on_page(pages[end], STOP) and not heading_on_page(pages[end], START):
                break
            end += 1
        section, found = "\n".join(pages[start:end]), True
    else:
        a, b = FALLBACK_PAGES
        section, found, start, end = "\n".join(pages[a:b]), False, a, min(b, len(pages))

    words = max(len(full.split()), 1)
    meta = {
        "pages": len(pages),
        "chars": n_chars,
        "scanned_or_empty": n_chars / max(len(pages), 1) < 300,   # image-only PDFs have ~no text
        "section_found": found,
        "section_pages": end - start,
        "section_words": len(section.split()),
        "leak_flag": bool(LEAK.search(full)),
    }
    for name, pat in FLAGS.items():
        meta[f"kw_{name}"] = len(re.findall(pat, full, re.I)) / words * 10_000   # per 10k words
    return section, meta


def main():
    rows = pd.read_csv(TEXT_ROWS)
    files = sorted(REPORTS.glob("AR_*.pdf"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"PDFs found: {len(files)}")

    metas = []
    for i, f in enumerate(files, 1):
        try:
            section, meta = extract_one(f)
            (OUT_DIR / f"{f.stem}.txt").write_text(section, encoding="utf-8")
        except Exception as e:
            meta = {"error": str(e)[:200]}
        meta["save_as"] = f.name
        metas.append(meta)
        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)} done")

    meta = pd.DataFrame(metas).merge(rows, on="save_as", how="left")
    meta.to_csv(OUT_META, index=False)

    ok = meta[meta.get("error").isna()] if "error" in meta else meta
    print(f"\nExtracted: {len(ok)} | errors: {len(meta) - len(ok)}")
    print(f"Section heading found: {ok['section_found'].sum()} | fallback pages used: {(~ok['section_found']).sum()}")
    print(f"Scanned / image-only (no usable text): {ok['scanned_or_empty'].sum()}")
    print(f"LEAK flag (company already in insolvency when report written): {ok['leak_flag'].sum()}")
    if ok["leak_flag"].any():
        print("  " + ", ".join(ok.loc[ok["leak_flag"], "save_as"].tolist()))
    print(f"Median section length: {int(ok['section_words'].median())} words")

    print("\nRed-flag phrases per 10k words - average by group:")
    kw = [c for c in ok.columns if c.startswith("kw_")]
    print(ok.groupby("label")[kw].mean().T.rename(columns={0: "healthy (0)", 1: "distressed (1)"})
          .round(2).to_string())
    if meta.get("label") is not None and meta["label"].isna().any():
        print(f"\nPDFs not in text_rows.csv (ignored later): {meta['label'].isna().sum()}")
    print(f"\nSaved: {OUT_META}, {OUT_DIR}/*.txt")


if __name__ == "__main__":
    main()