"""
train_gnn.py
------------
Graph Neural Network over a COMPANY PEER GRAPH.

Graph (one per fiscal year, all stored as one big disconnected graph):
  node  = a company in that fiscal year (one dataset row)
  edges = (1) same industry, same year          -> sector-wide stress (e.g. steel/infra 2015-17)
          (2) 5 most financially similar companies that year (k-nearest neighbours on ratios)
  This is a PEER graph (industry + similarity) - not a supply-chain graph. Say so in the report.

Model: 2-layer GraphSAGE. Each company's representation mixes its own ratios with the average
of its peers', so the model can learn "a weak firm in a stressed sector is riskier".
Only FEATURES flow along edges - labels are never shared between nodes. Features are public at
the observation date, so using test-fold companies as neighbours is not leakage
(transductive node classification; the loss only ever sees training labels).

To isolate what the GRAPH adds, the same network is also trained with NO edges ("MLP, no graph").

Input : data/processed/dataset.csv (+ folds.csv)
Output: reports/gnn_cv.csv, reports/gnn_cv_per_fold.csv
        data/processed/oof_gnn.csv       out-of-fold scores (for fusion)
        data/processed/peer_edges.csv    peer graph built on all data (for the dashboard)

Run from the project root (CPU, ~3-6 minutes):
    python src/train_gnn.py
"""

from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import SAGEConv

from cv_utils import (FEATURES, HORIZONS, N_FOLDS, best_threshold, fold_metrics, inner_split,
                      load_dataset, summarize)
from train_gru import Preprocessor

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

K_SIMILAR = 5
MAX_INDUSTRY_PEERS = 15     # cap industry links per company (keeps big industries from dominating)
HIDDEN = 32
EPOCHS = 300
PATIENCE = 30
LR = 5e-3
SEEDS = [0, 1, 2]
REPORTS = Path("reports")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def build_edges(df, Z, seed=0):
    """Edges within each fiscal year: same-industry links + k most similar companies."""
    rng = np.random.default_rng(seed)
    src, dst, kind = [], [], []
    for _, idx in df.groupby("fiscal_year").groups.items():
        idx = np.array(list(idx))
        # (1) same industry
        ind = df.loc[idx, "industry"].fillna("Unknown").values
        for name in np.unique(ind):
            if name == "Unknown":
                continue
            members = idx[ind == name]
            for i in members:
                others = members[members != i]
                if len(others) > MAX_INDUSTRY_PEERS:
                    others = rng.choice(others, MAX_INDUSTRY_PEERS, replace=False)
                src += [i] * len(others)
                dst += list(others)
                kind += ["industry"] * len(others)
        # (2) k most similar companies by (standardized) ratio profile
        if len(idx) > K_SIMILAR:
            nn_ = NearestNeighbors(n_neighbors=K_SIMILAR + 1).fit(Z[idx])
            _, nbrs = nn_.kneighbors(Z[idx])
            for row, ns in zip(idx, nbrs):
                for j in idx[ns[1:]]:
                    src += [row, j]
                    dst += [j, row]
                    kind += ["similar", "similar"]
    e = pd.DataFrame({"src": src, "dst": dst, "kind": kind}).drop_duplicates(["src", "dst"])
    return e


class PeerGNN(nn.Module):
    def __init__(self, n_in, hidden=HIDDEN):
        super().__init__()
        self.conv1 = SAGEConv(n_in, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Sequential(nn.Linear(hidden + n_in, hidden), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(hidden, 1))

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = self.drop(h)
        h = torch.relu(self.conv2(h, edge_index))
        return self.head(torch.cat([h, x], dim=1)).squeeze(-1)   # skip: own features always seen


def train_one(x, edge_index, y, fit_idx, val_idx, seed):
    set_seed(seed)
    model = PeerGNN(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    y_fit = y[fit_idx]
    pos_w = torch.tensor((len(y_fit) - y_fit.sum()) / max(y_fit.sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    yt = torch.tensor(y, dtype=torch.float32)
    best, best_state, wait = -1, None, 0
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index)
        loss_fn(out[fit_idx], yt[fit_idx]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(x, edge_index)).numpy()
        score = average_precision_score(y[val_idx], p[val_idx]) if y[val_idx].sum() else 0
        if score > best:
            best, best_state, wait = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x, edge_index)).numpy()


def main():
    df = load_dataset().reset_index(drop=True)
    print(f"Nodes (company-years): {len(df)} | {N_FOLDS}-fold CV x {len(SEEDS)} seeds")

    per_fold = []
    oof = df[["company_id", "company", "group", "industry", "fiscal_year", "fold",
              "label", "label_24m"]].copy()
    no_edges = torch.zeros((2, 0), dtype=torch.long)

    for k in range(N_FOLDS):
        train, test = df[df["fold"] != k], df[df["fold"] == k]
        fit, val = inner_split(train)
        prep = Preprocessor().fit(fit[FEATURES])
        Z = prep.transform(df[FEATURES]).astype(np.float32)
        edges = build_edges(df, Z[:, :len(FEATURES)], seed=k)
        edge_index = torch.tensor(edges[["src", "dst"]].values.T, dtype=torch.long)
        x = torch.tensor(Z)
        if k == 0:
            deg = edges.groupby("src").size()
            print(f"Edges: {len(edges)} ({(edges.kind == 'industry').sum()} industry, "
                  f"{(edges.kind == 'similar').sum()} similarity) | avg peers per node {deg.mean():.1f}")
        fit_idx, val_idx, test_idx = fit.index.values, val.index.values, test.index.values

        for h, col in HORIZONS.items():
            y = df[col].values.astype(float)
            for name, ei in [("GNN (peer graph)", edge_index), ("MLP (no graph)", no_edges)]:
                p = np.mean([train_one(x, ei, y, fit_idx, val_idx, s) for s in SEEDS], axis=0)
                per_fold.append({"model": name, "horizon": h, "fold": k,
                                 **fold_metrics(y[test_idx], p[test_idx],
                                                best_threshold(y[val_idx], p[val_idx]))})
                tag = "gnn" if name.startswith("GNN") else "mlp"
                oof.loc[test_idx, f"{tag}_{h}"] = p[test_idx]
            print(f"  fold {k} {h}: GNN {per_fold[-2]['pr_auc']:.3f} | MLP {per_fold[-1]['pr_auc']:.3f}")

    # peer graph on all data, for the dashboard
    prep = Preprocessor().fit(df[FEATURES])
    Z = prep.transform(df[FEATURES])
    edges = build_edges(df, Z[:, :len(FEATURES)], seed=0)
    edges["src_company"] = df.loc[edges["src"], "company"].values
    edges["dst_company"] = df.loc[edges["dst"], "company"].values
    edges["fiscal_year"] = df.loc[edges["src"], "fiscal_year"].values
    edges.to_csv("data/processed/peer_edges.csv", index=False)

    REPORTS.mkdir(exist_ok=True)
    table, _ = summarize(per_fold)
    table.to_csv(REPORTS / "gnn_cv.csv", index=False)
    pd.DataFrame(per_fold).to_csv(REPORTS / "gnn_cv_per_fold.csv", index=False)
    oof.to_csv("data/processed/oof_gnn.csv", index=False)

    print("\nGNN - CROSS-VALIDATED RESULTS (mean ± std over 5 folds)")
    print(table.to_string(index=False))
    for f in ["baseline_cv.csv", "gru_cv.csv"]:
        if (REPORTS / f).exists():
            print(f"\n{f}:")
            print(pd.read_csv(REPORTS / f)[["horizon", "model", "pr_auc", "roc_auc"]].to_string(index=False))
    print("\nSaved: reports/gnn_cv.csv, data/processed/oof_gnn.csv, data/processed/peer_edges.csv")


if __name__ == "__main__":
    main()