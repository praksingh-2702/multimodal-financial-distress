"""
make_healthy_list.py
--------------------
1. Gives every distressed company an industry (same categories NSE uses).
2. Picks healthy companies from the Nifty Total Market list, in the SAME industry mix as the
   distressed ones, so the model can't cheat by learning "industry X = distressed".
3. Writes a download list with direct Screener links.

Inputs : data/raw/nse/ind_niftytotalmarket_list.csv
         data/processed/download_list_distressed.csv   (make_download_list.py)
         data/processed/distressed_linked.csv          (check_coverage.py)
         data/processed/ibbi_all_cirp.csv              (build_labels.py)
Outputs: data/processed/distressed_industry.csv        (industry for each distressed company)
         data/processed/download_list_healthy.csv

Run from the project root:
    python src/make_healthy_list.py
"""

from pathlib import Path
import re

import pandas as pd

NSE_FILE = Path("data/raw/nse/ind_niftytotalmarket_list.csv")
DIST_LIST = Path("data/processed/download_list_distressed.csv")
DIST_LINKED = Path("data/processed/distressed_linked.csv")
IBBI_ALL = Path("data/processed/ibbi_all_cirp.csv")
OUT_DIST_IND = Path("data/processed/distressed_industry.csv")
OUT_HEALTHY = Path("data/processed/download_list_healthy.csv")

N_HEALTHY = 160            # aim for ~150 after a few fail to download
OTHER_SHARE = 0.10         # ~10% from industries with no distressed firms, so the model sees them too
SEED = 42
PEOPLE = ["Person 1", "Person 2", "Person 3"]

# 1) Name keywords are checked first (most reliable).
KEYWORDS = [
    (r"steel|ispat|iron|alloy|ferro|metal|forging|castings?|metcast|tubes", "Metals & Mining"),
    (r"textile|spinn|cotton|synthetic|fabric|silk|yarn|apparel|garment|denim|cotsyn|fibre|knit", "Textiles"),
    (r"pharma|drug|health|hospital|medic", "Healthcare"),
    (r"sugar|foods?\b|agro|dairy|beverage|edible|soya|rice|tea\b", "Fast Moving Consumer Goods"),
    (r"cement|ceramic|tiles|glass", "Construction Materials"),
    (r"paper|board|pulp|timber|wood|plywood", "Forest Materials"),
    (r"power|energy|solar|wind", "Power"),
    (r"realty|real estate|propert|developers|estate", "Realty"),
    (r"infra|construct|project|builders|buildcon|roads?|highway|epc", "Construction"),
    (r"shipyard|naval|engineering|machin|electric|cable|transformer|equipment|pumps|valves", "Capital Goods"),
    (r"auto|motors?\b|tyre|vehicle", "Automobile and Auto Components"),
    (r"chemical|petrochem|plastic|polymer|fertili|paints?|inks?\b|rubber|pack", "Chemicals"),
    (r"refiner|petroleum|\boil\b|gas\b|coal|coke", "Oil Gas & Consumable Fuels"),
    (r"electronic|appliance|consumer durable|jewel|gems|watches", "Consumer Durables"),
    (r"telecom|communications?", "Telecommunication"),
    (r"media|entertainment|television|broadcast|news", "Media Entertainment & Publication"),
    (r"retail|hotel|resort|restaurant|education|travel", "Consumer Services"),
    (r"airways|airlines|logistic|shipping|transport", "Services"),
    (r"software|infotech|technolog|computer|systems", "Information Technology"),
]

# 2) Fallback: first 2 digits of the CIN's NIC code (most CINs use NIC-2004).
NIC2004 = {
    "01": "Fast Moving Consumer Goods", "02": "Forest Materials", "05": "Fast Moving Consumer Goods",
    "10": "Oil Gas & Consumable Fuels", "11": "Oil Gas & Consumable Fuels", "13": "Metals & Mining",
    "14": "Metals & Mining", "15": "Fast Moving Consumer Goods", "16": "Fast Moving Consumer Goods",
    "17": "Textiles", "18": "Textiles", "19": "Consumer Durables", "20": "Forest Materials",
    "21": "Forest Materials", "22": "Media Entertainment & Publication", "23": "Oil Gas & Consumable Fuels",
    "24": "Chemicals", "25": "Chemicals", "26": "Construction Materials", "27": "Metals & Mining",
    "28": "Capital Goods", "29": "Capital Goods", "30": "Capital Goods", "31": "Capital Goods",
    "32": "Consumer Durables", "33": "Capital Goods", "34": "Automobile and Auto Components",
    "35": "Automobile and Auto Components", "36": "Consumer Durables", "37": "Metals & Mining",
    "40": "Power", "41": "Utilities", "45": "Construction", "50": "Consumer Services",
    "51": "Consumer Services", "52": "Consumer Services", "55": "Consumer Services", "60": "Services",
    "61": "Services", "62": "Services", "63": "Services", "64": "Telecommunication", "70": "Realty",
    "72": "Information Technology", "73": "Services", "74": "Services", "80": "Consumer Services",
    "85": "Healthcare", "92": "Media Entertainment & Publication",
}


def norm_name(s):
    s = re.sub(r"\(.*?\)", " ", str(s).lower())
    s = re.sub(r"\b(limited|ltd|pvt|private|india|the|merged)\b", " ", s)
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


def industry_for(name, nic_code):
    low = str(name).lower()
    for pattern, industry in KEYWORDS:
        if re.search(pattern, low):
            return industry, "name keyword"
    nic = str(nic_code).zfill(5)
    if nic[:4] == "2423":
        return "Healthcare", "NIC code"
    if nic[:2] in NIC2004:
        return NIC2004[nic[:2]], "NIC code"
    return "Diversified", "default"


def main():
    # ---- 1. Industry for each usable distressed company --------------------------
    lst = pd.read_csv(DIST_LIST, dtype={"nic_code": str})
    linked = pd.read_csv(DIST_LINKED)
    usable = linked[linked["years_before_event"] > 0].merge(
        lst[["cin", "company_name", "nic_code"]], on="cin", how="left")
    res = usable.apply(lambda r: industry_for(f"{r['search_name']} {r['screener_name']}", r["nic_code"]),
                       axis=1, result_type="expand")
    usable["industry"], usable["industry_source"] = res[0], res[1]
    OUT_DIST_IND.parent.mkdir(parents=True, exist_ok=True)
    usable[["row_no", "cin", "search_name", "screener_name", "nic_code", "industry", "industry_source"]] \
        .to_csv(OUT_DIST_IND, index=False)

    dist_mix = usable["industry"].value_counts()
    print(f"Distressed companies with usable data: {len(usable)}")
    print(usable["industry_source"].value_counts().to_string())
    print("\nDistressed industry mix:")
    print(dist_mix.to_string())

    # ---- 2. Candidate healthy companies -----------------------------------------
    nse = pd.read_csv(NSE_FILE)
    nse.columns = [c.strip() for c in nse.columns]
    nse = nse[nse["Industry"] != "Financial Services"].copy()   # same exclusion as distressed

    # Never pick a company that went through insolvency or is already in our distressed set
    bad = set(pd.read_csv(IBBI_ALL)["company_name"].map(norm_name))
    bad |= set(linked["screener_name"].map(norm_name)) | set(linked["search_name"].map(norm_name))
    nse["norm"] = nse["Company Name"].map(norm_name)
    removed = nse[nse["norm"].isin(bad)]
    nse = nse[~nse["norm"].isin(bad)]

    # ---- 3. Allocate N_HEALTHY across industries, proportional to distressed mix ---
    n_main = round(N_HEALTHY * (1 - OTHER_SHARE))
    shares = dist_mix / dist_mix.sum()
    alloc = (shares * n_main).round().astype(int).clip(lower=2)
    available = nse["Industry"].value_counts()

    picks = []
    for industry, n in alloc.items():
        pool = nse[nse["Industry"] == industry]
        if pool.empty:
            print(f"  note: no NSE companies in '{industry}' - its quota moves to 'other'")
            continue
        picks.append(pool.sample(n=min(n, len(pool)), random_state=SEED))
    chosen = pd.concat(picks) if picks else nse.iloc[0:0]

    other_pool = nse[~nse["Industry"].isin(alloc.index) & ~nse.index.isin(chosen.index)]
    n_other = max(0, min(N_HEALTHY - len(chosen), round(N_HEALTHY * OTHER_SHARE)))
    if len(other_pool):
        chosen = pd.concat([chosen, other_pool.sample(n=min(n_other, len(other_pool)), random_state=SEED)])

    # Some industries have fewer NSE companies than their quota (e.g. textiles, paper).
    # Top up to N_HEALTHY from the rest of the list, preferring industries the distressed firms are in.
    shortfall = N_HEALTHY - len(chosen)
    if shortfall > 0:
        rest = nse[~nse.index.isin(chosen.index)]
        preferred = rest[rest["Industry"].isin(alloc.index)]
        top_up = preferred.sample(n=min(shortfall, len(preferred)), random_state=SEED)
        if len(top_up) < shortfall:
            others = rest[~rest.index.isin(top_up.index)]
            top_up = pd.concat([top_up, others.sample(n=min(shortfall - len(top_up), len(others)),
                                                      random_state=SEED)])
        chosen = pd.concat([chosen, top_up])

    # ---- 4. Save download list --------------------------------------------------
    chosen = chosen.sort_values(["Industry", "Company Name"]).reset_index(drop=True)
    chosen["file_prefix"] = [f"H{i + 1:03d}" for i in range(len(chosen))]
    chosen["screener_url"] = "https://www.screener.in/company/" + chosen["Symbol"].astype(str) + "/"
    chosen["assigned_to"] = [PEOPLE[i % len(PEOPLE)] for i in range(len(chosen))]
    chosen["downloaded"] = ""
    chosen["notes"] = ""
    cols = ["file_prefix", "Company Name", "Symbol", "Industry", "screener_url", "assigned_to",
            "downloaded", "notes"]
    chosen[cols].to_csv(OUT_HEALTHY, index=False)

    print(f"\nNSE candidates (non-financial): {len(nse) + len(removed)} | "
          f"removed as past/current insolvency: {len(removed)}")
    if len(removed):
        print("  " + ", ".join(removed["Company Name"].head(15).tolist()))
    print(f"\nHealthy companies picked: {len(chosen)}")
    print(chosen["Industry"].value_counts().to_string())
    print("\nPer person:")
    print(chosen["assigned_to"].value_counts().sort_index().to_string())
    print(f"\nSaved: {OUT_DIST_IND}")
    print(f"Saved: {OUT_HEALTHY}")


if __name__ == "__main__":
    main()
    