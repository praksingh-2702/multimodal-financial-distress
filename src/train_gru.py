"""
train_gru.py
------------
GRU on each company's recent ratio history: for a row (company, FY t) the model reads the
ratio vectors of years t-2, t-1, t (oldest first) and predicts distress in the next 12 / 24 months.
It can learn TRENDS ("debt rising while margins fall for 3 years") that a single-year model misses.

Fairness rule (same as build_dataset.py): history only goes back to FY2017 for every company,
because healthy companies' files start there. Shorter histories are zero-padded at the front
and a 0/1 "year present" flag is added to each step, so padding is never mistaken for data.

Input : data/processed/dataset.csv (+ folds.csv from cv_utils)
Output: reports/gru_cv.csv, reports/gru_cv_per_fold.csv
        data/processed/oof_gru.csv     out-of-fold scores (for fusion)

Run from the project root (CPU is fine, ~2-4 minutes):
    python src/train_gru.py
"""

from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score

from cv_utils import (FEATURES, HORIZONS, N_FOLDS, best_threshold, fold_metrics, inner_split,
                      load_dataset, summarize)

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

SEQ_LEN = 3
HIST_MIN_FY = 2017
HIDDEN = 32
EPOCHS = 150
PATIENCE = 20
LR = 3e-3
SEEDS = [0, 1, 2]           # average 3 runs per fold -> more stable scores
REPORTS = Path("reports")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class GRUClassifier(nn.Module):
    def __init__(self, n_in, hidden=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.gru(x)                     # h: (1, batch, hidden) = summary of the sequence
        return self.head(self.drop(h[-1])).squeeze(-1)


class Preprocessor:
    """Winsorize -> median impute -> standardize, fitted on training rows only."""

    def fit(self, X):
        self.lo, self.hi = X.quantile(0.01), X.quantile(0.99)
        Xc = X.clip(self.lo, self.hi, axis=1)
        self.med = Xc.median()
        Xc = Xc.fillna(self.med)
        self.mu, self.sd = Xc.mean(), Xc.std().replace(0, 1)
        return self

    def transform(self, X):
        Xc = X.clip(self.lo, self.hi, axis=1)
        miss = Xc.isna().astype(float).values
        Xc = ((Xc.fillna(self.med) - self.mu) / self.sd).fillna(0).values
        return np.concatenate([Xc, miss], axis=1)   # values + per-feature missing flags


def build_sequences(df, Z):
    """For each row of df, stack the transformed vectors of the same company's years t-2..t."""
    pos = {(c, y): i for i, (c, y) in enumerate(zip(df["company_id"], df["fiscal_year"]))}
    n_feat = Z.shape[1]
    seq = np.zeros((len(df), SEQ_LEN, n_feat + 1), dtype=np.float32)
    for i, (c, y) in enumerate(zip(df["company_id"], df["fiscal_year"])):
        for j, lag in enumerate(range(SEQ_LEN - 1, -1, -1)):      # oldest -> newest
            yy = y - lag
            if lag > 0 and yy < HIST_MIN_FY:
                continue
            k = pos.get((c, yy))
            if k is not None:
                seq[i, j, :n_feat] = Z[k]
                seq[i, j, n_feat] = 1.0                            # "year present" flag
    return seq


def train_one(seq_fit, y_fit, seq_val, y_val, seed):
    set_seed(seed)
    model = GRUClassifier(seq_fit.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    pos_w = torch.tensor((len(y_fit) - y_fit.sum()) / max(y_fit.sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xf, yf = torch.tensor(seq_fit), torch.tensor(y_fit, dtype=torch.float32)
    Xv = torch.tensor(seq_val)

    best, best_state, wait = -1, None, 0
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xf))
        for b in range(0, len(Xf), 64):
            idx = perm[b:b + 64]
            opt.zero_grad()
            loss_fn(model(Xf[idx]), yf[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv)).numpy()
        score = average_precision_score(y_val, pv) if y_val.sum() else 0
        if score > best:
            best, best_state, wait = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model, seq):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(seq))).numpy()


def main():
    df = load_dataset().sort_values(["company_id", "fiscal_year"]).reset_index(drop=True)
    print(f"Rows {len(df)} | sequence length {SEQ_LEN} (history from FY{HIST_MIN_FY}) | "
          f"{N_FOLDS}-fold CV x {len(SEEDS)} seeds")

    per_fold = []
    oof = df[["company_id", "company", "group", "fiscal_year", "fold", "label", "label_24m"]].copy()

    for h, col in HORIZONS.items():
        for k in range(N_FOLDS):
            train, test = df[df["fold"] != k], df[df["fold"] == k]
            fit, val = inner_split(train)
            prep = Preprocessor().fit(fit[FEATURES])
            Z = prep.transform(df[FEATURES])          # transform all rows with train-only stats
            seq = build_sequences(df, Z)              # history rows are features only, never labels
            s_fit, s_val, s_test = seq[fit.index], seq[val.index], seq[test.index]
            y_fit, y_val, y_test = fit[col].values, val[col].values, test[col].values

            p_val = np.zeros(len(val))
            p_test = np.zeros(len(test))
            for s in SEEDS:
                m = train_one(s_fit, y_fit, s_val, y_val, seed=s)
                p_val += predict(m, s_val) / len(SEEDS)
                p_test += predict(m, s_test) / len(SEEDS)

            per_fold.append({"model": "GRU (3-year history)", "horizon": h, "fold": k,
                             **fold_metrics(y_test, p_test, best_threshold(y_val, p_val))})
            oof.loc[test.index, f"gru_{h}"] = p_test
            print(f"  {h} fold {k}: PR-AUC {per_fold[-1]['pr_auc']:.3f}")

    REPORTS.mkdir(exist_ok=True)
    table, _ = summarize(per_fold)
    table.to_csv(REPORTS / "gru_cv.csv", index=False)
    pd.DataFrame(per_fold).to_csv(REPORTS / "gru_cv_per_fold.csv", index=False)
    oof.to_csv("data/processed/oof_gru.csv", index=False)

    print("\nGRU - CROSS-VALIDATED RESULTS (mean ± std over 5 folds)")
    print(table.to_string(index=False))
    base = Path("reports/baseline_cv.csv")
    if base.exists():
        print("\nBaselines for comparison:")
        print(pd.read_csv(base)[["horizon", "model", "pr_auc", "roc_auc"]].to_string(index=False))
    print("\nSaved: reports/gru_cv.csv, data/processed/oof_gru.csv")


if __name__ == "__main__":
    main()