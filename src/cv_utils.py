"""
cv_utils.py
-----------
Shared helpers so EVERY model (baselines, GRU, GNN, FinBERT, fusion) is evaluated the same way.

5-fold cross-validation by COMPANY: every company is in exactly one test fold, and a company's
rows are never split between training and testing. Folds are saved once to
data/processed/folds.csv and reused by all models, so results are directly comparable and
out-of-fold predictions can be stacked in the fusion model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

DATA = Path("data/processed/dataset.csv")
FOLDS = Path("data/processed/folds.csv")
N_FOLDS = 5
SEED = 42
HORIZONS = {"12m": "label", "24m": "label_24m"}

FEATURES = [
    "debt_to_equity", "debt_to_assets", "interest_coverage", "liquidity_proxy", "cash_to_assets",
    "roa", "ebit_margin", "asset_turnover", "cfo_to_assets", "cfo_to_debt", "mcap_to_debt",
    "altman_z_proxy", "sales_growth", "borrowings_growth", "price_change", "negative_net_worth",
    "log_assets", "d_roa", "d_debt_to_assets", "d_interest_coverage", "n_missing",
]


def load_dataset():
    """Dataset with a 'fold' column (0-4). Creates folds.csv the first time."""
    df = pd.read_csv(DATA)
    if not FOLDS.exists():
        make_folds(df)
    folds = pd.read_csv(FOLDS)
    df = df.drop(columns=["fold"], errors="ignore").merge(folds, on="company_id", how="left")
    if df["fold"].isna().any():
        raise ValueError("Some companies have no fold - delete data/processed/folds.csv and rerun")
    df["fold"] = df["fold"].astype(int)
    return df


def make_folds(df):
    """Stratify by company type (healthy / distressed-with-positive / distressed-without)."""
    comp = df.groupby("company_id").agg(group=("group", "first"), has_pos=("label", "max"))
    comp["stratum"] = comp["group"] + "_" + comp["has_pos"].astype(str)
    rng = np.random.default_rng(SEED)
    rows = []
    for _, ids in comp.groupby("stratum").groups.items():
        ids = np.array(sorted(ids), dtype=object)
        rng.shuffle(ids)
        rows += [(cid, i % N_FOLDS) for i, cid in enumerate(ids)]
    pd.DataFrame(rows, columns=["company_id", "fold"]).to_csv(FOLDS, index=False)


def inner_split(train_df, frac=0.2, seed=SEED):
    """Split a training fold by company into (fit, val) for threshold choice / early stopping."""
    ids = np.array(sorted(train_df["company_id"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    val_ids = set(ids[: max(1, int(len(ids) * frac))])
    is_val = train_df["company_id"].isin(val_ids)
    return train_df[~is_val], train_df[is_val]


def winsorize(train, *others, lo=0.01, hi=0.99):
    """Clip extreme ratios using limits learned on the training data only."""
    q_lo, q_hi = train.quantile(lo), train.quantile(hi)
    return [d.clip(q_lo, q_hi, axis=1) for d in (train, *others)]


def best_threshold(y, p):
    if y.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.nanargmax(f1[:-1])]) if len(thr) else 0.5


def fold_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def summarize(per_fold):
    """per_fold: list of dicts with keys model, horizon, fold, metrics... -> mean ± std table."""
    pf = pd.DataFrame(per_fold)
    metrics = ["pr_auc", "roc_auc", "precision", "recall", "f1"]
    g = pf.groupby(["horizon", "model"], sort=False)[metrics]
    mean, std = g.mean(), g.std()
    out = mean.round(3).astype(str) + " ± " + std.round(3).astype(str)
    return out.reset_index(), mean.reset_index()