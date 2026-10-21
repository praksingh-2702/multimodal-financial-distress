"""
Corporate distress early-warning model - Indian listed companies
Streamlit app. Reads only the pre-computed files in app_data/ (built by src/explain_and_package.py).

Run locally from the project root:
    streamlit run dashboard/app.py
"""

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

APP = Path(__file__).resolve().parent.parent / "app_data"
NAVY, GREY, LIGHT = "#6fa8dc", "#9aa4ae", "#4a5563"   # accent, neutral, muted (dark theme)
COMPONENTS = {"xgb": "Financial ratios (XGBoost)", "gru": "Ratio trends, 3 years (GRU)",
              "gnn": "Industry peers (GNN)"}
RATIOS = {
    "debt_to_equity": "Debt / equity", "interest_coverage": "Interest coverage",
    "roa": "Return on assets", "ebit_margin": "EBIT margin", "liquidity_proxy": "Liquidity ratio (proxy)",
    "cfo_to_assets": "Operating cash flow / assets", "altman_z_proxy": "Altman Z (proxy)",
    "log_assets": "Size: log total assets", "debt_to_assets": "Debt / assets", "cash_to_assets": "Cash / assets",
    "asset_turnover": "Asset turnover", "cfo_to_debt": "Operating cash flow / debt",
    "mcap_to_debt": "Market cap / debt", "sales_growth": "Sales growth", "borrowings_growth": "Borrowings growth",
    "price_change": "Share price change", "negative_net_worth": "Negative net worth",
    "d_roa": "Change in ROA", "d_debt_to_assets": "Change in debt / assets",
    "d_interest_coverage": "Change in interest coverage", "n_missing": "Missing data items",
}

st.set_page_config(page_title="Corporate Distress Early Warning", layout="wide")


@st.cache_data
def load():
    scores = pd.read_csv(APP / "scores.csv", parse_dates=["obs_date", "cirp_start_date"])
    shap = pd.read_csv(APP / "shap.csv")
    peers = pd.read_csv(APP / "peers.csv")
    meta = json.loads((APP / "meta.json").read_text())
    text = pd.read_csv(APP / "text.csv") if (APP / "text.csv").exists() else pd.DataFrame()
    sents = pd.read_csv(APP / "sentences.csv") if (APP / "sentences.csv").exists() else pd.DataFrame()
    res = {f.stem: pd.read_csv(f) for f in (APP / "results").glob("*.csv")}
    return scores, shap, peers, meta, text, sents, res


scores, shap_df, peers, meta, text, sents, results = load()
companies = (scores.groupby("company_id")
             .agg(company=("company", "first"), industry=("industry", "first"), group=("group", "first"))
             .reset_index().sort_values("company"))
companies["label"] = companies["company"].str.title() + " — " + companies["industry"].fillna("")


def mean_of(s):
    return float(s.split("±")[0]) if isinstance(s, str) and "±" in s else np.nan


def plain_layout(fig, height):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=10, b=10),
                      font=dict(size=12), legend=dict(orientation="h", y=1.12, x=0))
    return fig


# --------------------------------------------------------------------------- header
st.title("Corporate Distress Early-Warning Model")
st.caption("Indian listed companies, FY2014–FY2022. Estimates the probability that a company enters the "
           "insolvency resolution process (IBC) within 12 months of its annual results. "
           "Academic project using historical data; not investment advice.")

tab_overview, tab_company, tab_screen, tab_method = st.tabs(["Overview", "Company", "Screener", "Methodology"])

# =========================================================================== OVERVIEW
with tab_overview:
    abl = results.get("ablation_cv")
    base_rate = meta["positive_rows_12m"] / meta["company_years"]
    best = abl[(abl["horizon"] == "12m") & abl["model"].str.startswith("Fusion: stacked")] if abl is not None else []

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies", meta["companies"])
    c2.metric("Insolvency cases", meta["distressed_companies"])
    c3.metric("Company-years", f"{meta['company_years']:,}")
    if len(best):
        c4.metric("Final model ROC-AUC (12m)", f"{mean_of(best['roc_auc'].iloc[0]):.3f}")

    st.subheader("Model comparison")
    st.write("Five-fold cross-validation, split by company. Every score is produced by a model that did not "
             f"see that company during training. About {base_rate:.0%} of company-years are followed by "
             "insolvency within 12 months, so precision–recall AUC (PR-AUC) is the main metric; "
             f"a random ranking would score about {base_rate:.2f}.")
    if abl is not None:
        order = ["Altman Z proxy (rule)", "Logistic regression", "XGBoost (ratios)", "GRU (trends)",
                 "GNN (peers)", "Fusion: XGBoost + GRU + GNN", "Fusion: stacked (logistic on XGB, GRU, GNN)"]
        nice = {"Fusion: stacked (logistic on XGB, GRU, GNN)": "Final model (stacked fusion)",
                "Fusion: XGBoost + GRU + GNN": "Average of XGBoost, GRU, GNN"}
        rows = []
        for m in order:
            r = {"Model": nice.get(m, m)}
            for h in ["12m", "24m"]:
                x = abl[(abl["model"] == m) & (abl["horizon"] == h)]
                if len(x):
                    r[f"PR-AUC {h}"] = x["pr_auc"].iloc[0]
                    r[f"ROC-AUC {h}"] = x["roc_auc"].iloc[0]
            rows.append(r)
        table = pd.DataFrame(rows)
        left, right = st.columns([1.15, 1])
        with left:
            st.dataframe(table, hide_index=True, width="stretch")
            st.caption("Mean ± standard deviation across the 5 folds.")
        with right:
            fig = go.Figure()
            for h, color in [("12m", NAVY), ("24m", LIGHT)]:
                fig.add_bar(y=table["Model"], x=table[f"PR-AUC {h}"].map(mean_of), orientation="h",
                            name=f"{h} horizon", marker_color=color)
            fig.update_layout(barmode="group", xaxis_title="PR-AUC", yaxis=dict(autorange="reversed"))
            st.plotly_chart(plain_layout(fig, 340), width="stretch")

    a, b = st.columns(2)
    with a:
        st.subheader("Annual-report text")
        tcv = results.get("text_cv")
        if tcv is not None:
            st.write("Evaluated on the company-years for which a readable annual report was available.")
            st.dataframe(tcv[["model", "pr_auc", "roc_auc"]].rename(
                columns={"model": "Inputs", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"}),
                hide_index=True, width="stretch")
    with b:
        st.subheader("Early warning")
        ew = meta.get("early_warning", {})
        if ew:
            st.markdown(
                f"- Companies that later entered insolvency and were rated High (≥ 20%) at least once "
                f"beforehand: **{ew['flagged_high_at_least_once']} of {ew['distressed_companies']}** "
                f"({ew['share_flagged']:.0%})\n"
                f"- Median time between the first High rating and the insolvency date: "
                f"**{ew['median_months_warning']} months**\n"
                f"- Healthy company-years rated High: **{ew['healthy_company_years_high']:.1%}**")
        w = meta.get("stacked_weights_12m", {})
        if w:
            st.write("Weights of each component in the final model (12-month horizon):")
            st.dataframe(pd.DataFrame({"Component": [COMPONENTS[k] for k in w], "Weight": list(w.values())}),
                         hide_index=True, width="stretch")

    if (APP / "figures" / "case_studies.png").exists():
        st.subheader("Examples: estimated risk before known insolvencies")
        st.image(str(APP / "figures" / "case_studies.png"), width="stretch")
    if (APP / "figures" / "shap_summary.png").exists():
        with st.expander("Ratio importance (SHAP summary)"):
            st.image(str(APP / "figures" / "shap_summary.png"))

# =========================================================================== COMPANY
with tab_company:
    ids = meta.get("case_study_ids", [])
    default = companies.reset_index(drop=True).index[companies["company_id"].isin(ids[:1]).values]
    choice = st.selectbox("Company", companies["label"], index=int(default[0]) if len(default) else 0)
    cid = companies.loc[companies["label"] == choice, "company_id"].iloc[0]
    hist = scores[scores["company_id"] == cid].sort_values("fiscal_year")
    fy = st.select_slider("Financial year", hist["fiscal_year"].tolist(), value=int(hist["fiscal_year"].iloc[-1]))
    row = hist[hist["fiscal_year"] == fy].iloc[0]

    st.markdown(f"#### {row['company'].title()}")
    outcome = (f"Insolvency process started {row['cirp_start_date']:%d %b %Y} "
               f"({row['months_to_event']:.0f} months after this observation)"
               if row["group"] == "distressed" and pd.notna(row["cirp_start_date"]) else "No insolvency (comparison company)")
    st.caption(f"Industry: {row['industry']}  ·  Observation date: 30 Jun {fy}  ·  Actual outcome: {outcome}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Probability of insolvency, 12 months", f"{row['risk_12m']:.1%}")
    m2.metric("Probability of insolvency, 24 months", f"{row['risk_24m']:.1%}")
    m3.metric("Risk band", row["band"])
    m4.metric("Percentile among all company-years", f"{row['risk_percentile']:.0%}")
    st.caption("Bands: Low < 5%, Medium 5–20%, High ≥ 20% (12-month probability).")

    st.markdown("##### Risk history")
    fig = go.Figure()
    fig.add_scatter(x=hist["obs_date"], y=hist["risk_12m"] * 100, mode="lines+markers", name="12 months",
                    line=dict(color=NAVY))
    fig.add_scatter(x=hist["obs_date"], y=hist["risk_24m"] * 100, mode="lines+markers", name="24 months",
                    line=dict(color=GREY, dash="dash"))
    fig.add_hline(y=20, line_dash="dot", line_color=GREY)
    if pd.notna(row["cirp_start_date"]):
        fig.add_vline(x=row["cirp_start_date"], line_color="#e6e6e6", line_width=1)
        fig.add_annotation(x=row["cirp_start_date"], y=98, text="insolvency start", showarrow=False,
                           xanchor="right", font=dict(size=11))
    fig.update_yaxes(range=[0, 100], title="probability (%)")
    st.plotly_chart(plain_layout(fig, 280), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("##### Contribution of each component")
        comp = pd.DataFrame({
            "Component": [COMPONENTS[m] for m in ["xgb", "gru", "gnn"]],
            "Component score": [row[f"{m}_12m"] for m in ["xgb", "gru", "gnn"]],
            "Effect on final estimate": [row[f"contrib_{m}"] for m in ["xgb", "gru", "gnn"]],
        })
        comp["Direction"] = np.where(comp["Effect on final estimate"] > 0, "raises risk", "lowers risk")
        st.dataframe(comp.round(3), hide_index=True, width="stretch")
        st.caption("Effect = component weight × (score relative to the average company), on the log-odds scale.")

        st.markdown("##### Main ratio drivers (SHAP)")
        srow = shap_df[(shap_df["company_id"] == cid) & (shap_df["fiscal_year"] == fy)]
        if len(srow):
            s = srow.drop(columns=["company_id", "fiscal_year"]).iloc[0]
            s = s.reindex(s.abs().sort_values(ascending=False).index)[:8][::-1]
            fig = go.Figure(go.Bar(x=s.values, y=[RATIOS.get(k, k) for k in s.index], orientation="h",
                                   marker_color=[NAVY if v > 0 else LIGHT for v in s.values]))
            fig.update_xaxes(title="contribution to ratio-model score (positive = higher risk)")
            st.plotly_chart(plain_layout(fig, 300), width="stretch")
    with right:
        st.markdown("##### Key ratios against industry median")
        same = scores[(scores["fiscal_year"] == fy) & (scores["industry"] == row["industry"])]
        keys = [k for k in RATIOS if k in scores.columns]
        st.dataframe(pd.DataFrame({"Ratio": [RATIOS[k] for k in keys],
                                   "Company": [row[k] for k in keys],
                                   "Industry median": [same[k].median() for k in keys]}).round(3),
                     hide_index=True, width="stretch")
        st.caption(f"{len(same)} companies in {row['industry']} for FY{fy}.")

    st.markdown("##### Annual report")
    trow = text[text["company_id"] == cid] if len(text) else pd.DataFrame()
    if len(trow):
        t = trow.iloc[0]
        avg = text.merge(scores[["company_id", "fiscal_year", "group"]], on=["company_id", "fiscal_year"])
        grp = avg.groupby("group")[["fb_net_tone", "fb_debt_tone", "fb_neg_share"]].mean()
        st.write(f"Report analysed: FY{int(t['report_fy'])}, the latest report published before the "
                 "observation date. Sentiment is scored sentence by sentence with FinBERT.")
        comp_t = pd.DataFrame({
            "Measure": ["Net tone (positive − negative)", "Net tone of debt / lending sentences",
                        "Share of negative sentences"],
            "This report": [t["fb_net_tone"], t["fb_debt_tone"], t["fb_neg_share"]],
            "Average, healthy companies": [grp.loc[g, c] if g in grp.index else np.nan for g, c in
                                           [("healthy", "fb_net_tone"), ("healthy", "fb_debt_tone"), ("healthy", "fb_neg_share")]],
            "Average, distressed companies": [grp.loc[g, c] if g in grp.index else np.nan for g, c in
                                              [("distressed", "fb_net_tone"), ("distressed", "fb_debt_tone"), ("distressed", "fb_neg_share")]],
        })
        st.dataframe(comp_t.round(3), hide_index=True, width="stretch")
        flags = {k.replace("kw_", "").replace("_", " "): t[k] for k in t.index if k.startswith("kw_") and t[k] > 0}
        if flags:
            st.write("Warning terms found (occurrences per 10,000 words): " +
                     "; ".join(f"{k} {v:.1f}" for k, v in sorted(flags.items(), key=lambda x: -x[1])))
        top = sents[sents["save_as"] == t["save_as"]].sort_values("p_negative", ascending=False) if len(sents) else []
        if len(top):
            with st.expander("Most negative sentences in the report"):
                for _, s_ in top.head(5).iterrows():
                    st.markdown(f"> {html.escape(str(s_['sentence']))}")
                    st.caption(f"FinBERT negative probability: {s_['p_negative']:.2f}")
    else:
        st.caption("No machine-readable annual report was analysed for this company.")

    st.markdown("##### Peers used by the graph model")
    e = peers[(peers["fiscal_year"] == fy) & ((peers["src_id"] == cid) | (peers["dst_id"] == cid))]
    rel = pd.concat([e.loc[e["src_id"] == cid, ["dst_id", "kind"]].rename(columns={"dst_id": "peer"}),
                     e.loc[e["dst_id"] == cid, ["src_id", "kind"]].rename(columns={"src_id": "peer"})])
    if len(rel):
        rel = rel.groupby("peer")["kind"].agg(lambda k: ", ".join(sorted(set(k)))).reset_index()
        yr = scores[scores["fiscal_year"] == fy].set_index("company_id")
        rel = rel[rel["peer"].isin(yr.index)]
        peer_tbl = pd.DataFrame({
            "Peer company": yr.loc[rel["peer"], "company"].str.title().values,
            "Link": rel["kind"].str.replace("industry", "same industry").str.replace("similar", "similar ratios").values,
            "12-month probability": (yr.loc[rel["peer"], "risk_12m"].values * 100).round(1),
        }).sort_values("12-month probability", ascending=False)
        st.dataframe(peer_tbl, hide_index=True, width="stretch", height=260)
        st.caption("The graph model combines a company's ratios with these peers' ratios. "
                   "Peers' outcomes are never used.")
        with st.expander("Peer network graph"):
            try:
                from pyvis.network import Network
                net = Network(height="420px", width="100%", bgcolor="#0e1117", font_color="#e6e6e6",
                              cdn_resources="remote")
                net.barnes_hut(gravity=-5000, spring_length=130)
                net.add_node(cid, label=row["company"].title()[:28], color=NAVY, size=22)
                for p in rel["peer"].head(25):
                    r = yr.loc[p]
                    net.add_node(p, label=str(r["company"]).title()[:24], color=GREY, size=12,
                                 title=f"{r['company']} | 12m probability {r['risk_12m']:.0%}")
                for _, ed in e.iterrows():
                    if ed["src_id"] in net.get_nodes() and ed["dst_id"] in net.get_nodes():
                        net.add_edge(ed["src_id"], ed["dst_id"], color=LIGHT)
                page = net.generate_html()
                if hasattr(st, "iframe"):
                    st.iframe(page, height=440)
                else:
                    components.html(page, height=440)
            except Exception as ex:
                st.caption(f"Graph could not be drawn: {ex}")
    else:
        st.caption("No peers recorded for this company-year.")

# =========================================================================== SCREENER
with tab_screen:
    st.write("Companies ranked by estimated 12-month probability of insolvency for a selected year.")
    s1, s2, s3 = st.columns(3)
    yr_sel = s1.selectbox("Financial year", sorted(scores["fiscal_year"].unique(), reverse=True))
    ind_sel = s2.selectbox("Industry", ["All industries"] + sorted(scores["industry"].dropna().unique()))
    band_sel = s3.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium"])
    view = scores[(scores["fiscal_year"] == yr_sel) & scores["band"].isin(band_sel)]
    if ind_sel != "All industries":
        view = view[view["industry"] == ind_sel]
    view = view.sort_values("risk_12m", ascending=False)
    st.dataframe(pd.DataFrame({
        "Company": view["company"].str.title(), "Industry": view["industry"],
        "12-month probability (%)": (view["risk_12m"] * 100).round(1),
        "24-month probability (%)": (view["risk_24m"] * 100).round(1),
        "Band": view["band"],
        "Actual outcome": np.where(view["group"] == "distressed",
                                   "Insolvency, " + view["cirp_start_date"].dt.strftime("%b %Y"), "None"),
    }), hide_index=True, width="stretch", height=520)
    st.caption(f"{len(view)} companies. The actual outcome is shown for evaluation only; it is not a model input.")

# =========================================================================== METHODOLOGY
with tab_method:
    st.markdown("""
#### Target
A company-year is labelled **1** if the company's Corporate Insolvency Resolution Process (CIRP) under the
Insolvency and Bankruptcy Code, 2016 began within 12 months after its annual results became public
(taken as 30 June following the financial year). A 24-month label is reported as a robustness check.

#### Data
| Source | Use |
|---|---|
| IBBI lists of completed CIRPs (resolution and liquidation), as on 31 March 2025 | insolvency events and dates |
| Screener.in company exports (from exchange filings) | profit and loss, balance sheet, cash flow, share price |
| NSE Nifty Total Market constituents | healthy comparison companies and industry classification |
| Annual reports (BSE) | Directors' Report and Management Discussion & Analysis text |

Sample: listed non-financial companies; insolvency cases starting 2017–2022 with admitted claims of at
least ₹100 crore; healthy companies drawn in the same industry mix.

#### Models
1. **XGBoost** on about 20 financial ratios covering leverage, liquidity, profitability, cash flow and market value.
2. **GRU** (recurrent neural network) on each company's last three years of ratios, to capture deterioration over time.
3. **Graph neural network** (GraphSAGE) on a yearly peer graph linking companies in the same industry and the five
   most financially similar companies, to capture sector-wide stress.
4. **FinBERT** sentence-level sentiment on the annual report, plus counts of warning terms (going concern,
   qualified opinion, default, restructuring).
5. **Final model:** logistic regression on the component scores, with Platt calibration.

#### Controls against look-ahead bias
- Cross-validation by company; reported scores are out-of-fold.
- Only information public at the observation date is used; financial years not ending in March are dated accordingly.
- Year-on-year features use the same history window for all companies.
- Annual reports written after insolvency had begun were identified and excluded.
- The graph model shares features between peers, never outcomes.

#### Limitations
- Healthy companies are current index members and are, on average, larger than the insolvency cases.
- The peer graph is based on industry and similarity, not on actual supplier, customer or group relationships.
- Text results cover only company-years with a readable annual report.
- Banks and non-bank financial companies are excluded.
""")