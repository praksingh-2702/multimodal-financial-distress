# Multimodal Financial Distress Prediction for Indian Listed Companies

**Live dashboard:** https://multimodal-financial-distress-dnhvvfxayrcv47hc4upxj.streamlit.app

An early-warning model that estimates the probability that an Indian listed company enters the
**Corporate Insolvency Resolution Process (CIRP)** under the Insolvency and Bankruptcy Code within
**12 months** of its annual results. It combines four kinds of information:

| Component | Input | Model |
|---|---|---|
| Financial ratios | ~20 ratios: leverage, liquidity, profitability, cash flow, market value, Altman Z proxy | XGBoost |
| Trends | each company's last 3 years of ratios | GRU (recurrent neural network) |
| Industry peers | yearly peer graph: same industry + 5 most financially similar companies | Graph neural network (GraphSAGE) |
| Annual-report text | Directors' Report and MD&A | FinBERT sentiment + warning-term counts |

The component scores are combined by a stacked logistic regression and calibrated to a probability.

AI in Finance course project.

---

## Results

Five-fold cross-validation **split by company**: every score comes from a model that never saw that
company. About 9% of company-years are followed by insolvency within 12 months, so **PR-AUC** is the
main metric (random ranking ≈ 0.09).

| Model | PR-AUC 12m | ROC-AUC 12m | PR-AUC 24m | ROC-AUC 24m |
|---|---|---|---|---|
| Altman Z-score proxy (rule) | 0.299 | 0.803 | 0.482 | 0.809 |
| Logistic regression (ratios) | 0.370 | 0.845 | 0.551 | 0.855 |
| XGBoost (ratios) | 0.352 | 0.849 | 0.596 | 0.870 |
| GRU (3-year trends) | 0.386 | 0.870 | 0.586 | 0.872 |
| GNN (industry peers) | 0.403 | 0.877 | 0.605 | 0.880 |
| **Final model (stacked fusion)** | **0.420** | **0.883** | **0.625** | **0.887** |

- The final model's PR-AUC is about **4.7× a random ranking**, and every AI component improves on the
  Altman Z-score rule.
- The **graph model beat the identical network without the graph in 9 of 10 fold comparisons**, so
  peer information adds signal beyond a company's own ratios.
- Weights in the final model (12m): GRU 0.41, GNN 0.38, XGBoost 0.14.

**Annual-report text** (135 company-years with a readable report):

| Inputs | PR-AUC | ROC-AUC |
|---|---|---|
| FinBERT sentiment only | 0.801 | 0.783 |
| FinBERT + warning terms | 0.961 | 0.943 |
| Financial models only | 0.961 | 0.954 |
| **Financial models + text** | **0.984** | **0.978** |

**Early warning:** 110 of 211 companies that later entered insolvency (52%) were rated High (≥ 20%)
at least once beforehand, with a median of **19 months** between the first High rating and the
insolvency date. Only 1.9% of healthy company-years were rated High.

Exact figures (mean ± standard deviation across folds) are in [`reports/`](reports/).

---

## Data

| Source | Use |
|---|---|
| IBBI lists of completed CIRPs (resolution plans and liquidation), as on 31 March 2025 | insolvency events and start dates |
| Screener.in company exports (from exchange filings) | profit and loss, balance sheet, cash flow, year-end share price |
| NSE Nifty Total Market constituents | healthy comparison companies and industry classification |
| Annual reports (BSE, via Screener.in) | Directors' Report and MD&A text |

**Sample:** 368 listed non-financial companies (211 insolvency cases, 157 healthy) and 1,740
company-years, FY2014–FY2022. Insolvency cases started 2017–2022 with admitted claims of at least
₹100 crore; healthy companies were drawn in the same industry mix. Raw data is not committed to this
repository; the scripts below rebuild it.

**Label:** a company-year is 1 if CIRP began within 12 months after 30 June following the financial
year (when annual results are taken to be public). A 24-month label is reported as a robustness check.

---

## Controls against look-ahead bias

- Cross-validation by company; all reported scores are out-of-fold.
- Only information public at the observation date is used. Financial years not ending in March are
  dated accordingly (184 such reports), and duplicate company-years from year-end changes are removed.
- Year-on-year features use the same history window (from FY2017) for distressed and healthy
  companies, because healthy companies' data starts in FY2017.
- Annual reports written after insolvency had begun (mentioning an Interim Resolution Professional)
  were detected and excluded (7 reports).
- The graph model passes features between peers, never outcomes.

## Limitations

- Healthy companies are current index members and are larger on average than the insolvency cases.
- The peer graph uses industry and financial similarity, not actual supplier, customer or
  business-group links.
- Text results cover only company-years with a readable annual report.
- Banks and non-bank financial companies are excluded.
- The IBBI lists cover completed processes only; ongoing cases are not included.

---

## Repository structure

```
src/
  build_labels.py          IBBI files -> listed companies that entered CIRP, with dates
  make_download_list.py    distressed companies to download (scope rules)
  parse_screener.py        Screener exports -> financial statements and ratios
  check_coverage.py        links files to IBBI records; checks pre-insolvency years
  make_healthy_list.py     industry tags; healthy companies in the same industry mix
  build_dataset.py         company-year table with labels and features
  cv_utils.py              shared 5-fold company-grouped cross-validation
  train_baseline.py        Altman Z proxy, logistic regression, XGBoost
  train_gru.py             GRU on 3-year ratio sequences
  train_gnn.py             GraphSAGE on the yearly peer graph (+ no-graph ablation)
  make_report_list.py      which annual report to use for each company (no look-ahead)
  extract_reports.py       Directors' Report / MD&A extraction, warning terms, leak check
  finbert_features.py      FinBERT sentence sentiment aggregated per report
  train_fusion.py          stacked fusion, ablation table, text evaluation
  explain_and_package.py   calibration, SHAP, early-warning statistics, case studies, app data
dashboard/app.py           Streamlit dashboard
app_data/                  pre-computed scores and explanations used by the dashboard
reports/                   cross-validation results and figures
```

## Reproducing

```bash
python -m venv .venv && source .venv/Scripts/activate      # Windows Git Bash
pip install -r requirements.txt

# 1. labels and financial data (place the IBBI and Screener files in data/raw/ first)
python src/build_labels.py
python src/make_download_list.py
python src/parse_screener.py
python src/check_coverage.py
python src/make_healthy_list.py
python src/build_dataset.py

# 2. models
python src/train_baseline.py
python src/train_gru.py
python src/train_gnn.py

# 3. annual-report text (place PDFs in data/raw/reports/)
python src/make_report_list.py
python src/extract_reports.py
python src/finbert_features.py

# 4. fusion, explanations, dashboard
python src/train_fusion.py
python src/explain_and_package.py
streamlit run dashboard/app.py
```

All models run on a laptop CPU. FinBERT scoring of ~135 reports takes about 20–30 minutes.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, PyTorch, PyTorch Geometric, Hugging Face Transformers
(ProsusAI/finbert), PyMuPDF, SHAP, Streamlit, Plotly, PyVis.

*Academic project on historical data. Not investment advice.*