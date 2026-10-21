"""
finbert_features.py
-------------------
Runs FinBERT (ProsusAI/finbert, a BERT model trained on financial text) on the sentences of
each Directors' Report / MD&A and turns the sentence-level sentiment into report-level features.

Input : data/processed/report_text/*.txt   (extract_reports.py)
        data/processed/report_meta.csv
Output: data/processed/text_features.csv   one row per report
        data/processed/top_negative_sentences.csv   5 most negative sentences per report
                                                    (shown in the dashboard as explanation)

Run from the project root:
    python src/finbert_features.py
First run downloads the model (~440 MB). On a laptop CPU expect ~15-30 minutes for ~150
reports at 200 sentences each. To go faster, lower MAX_SENTENCES or run on Google Colab (GPU).
"""

import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TEXT_DIR = Path("data/processed/report_text")
META = Path("data/processed/report_meta.csv")
OUT = Path("data/processed/text_features.csv")
OUT_SENT = Path("data/processed/top_negative_sentences.csv")

MODEL = os.environ.get("FINBERT_MODEL", "ProsusAI/finbert")   # env override only for testing
MAX_SENTENCES = 200           # per report, sampled evenly across the section
BATCH = 32
DEBT_WORDS = re.compile(r"debt|loan|borrow|lender|bank|interest|repay|liquidity|working\s+capital|"
                        r"default|restructur|npa|credit", re.I)


def split_sentences(text):
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    out = []
    for s in parts:
        s = s.strip()
        letters = sum(ch.isalpha() for ch in s)
        if 40 <= len(s) <= 500 and letters / max(len(s), 1) > 0.6:   # skip tables / numbers
            out.append(s)
    return list(dict.fromkeys(out))          # drop repeated sentences (headers, boilerplate)


def sample_evenly(items, k):
    if len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idx]


def main():
    meta = pd.read_csv(META)
    if "error" in meta:
        meta = meta[meta["error"].isna()]
    meta = meta[~meta["leak_flag"] & ~meta["scanned_or_empty"]]
    print(f"Reports to score: {len(meta)} (excluding leak-flagged and scanned ones)")

    torch.set_num_threads(max(1, torch.get_num_threads()))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    labels = {i: l.lower() for i, l in model.config.id2label.items()}     # positive/negative/neutral
    print(f"Model labels: {labels}")

    feats, top_neg = [], []
    for n, save_as in enumerate(meta["save_as"], 1):
        stem = Path(save_as).stem
        text = (TEXT_DIR / f"{stem}.txt").read_text(encoding="utf-8", errors="ignore")
        sents = sample_evenly(split_sentences(text), MAX_SENTENCES)
        if len(sents) < 10:
            feats.append({"save_as": save_as, "n_sentences": len(sents)})
            continue

        probs = []
        with torch.no_grad():
            for b in range(0, len(sents), BATCH):
                enc = tok(sents[b:b + BATCH], padding=True, truncation=True, max_length=128,
                          return_tensors="pt")
                probs.append(torch.softmax(model(**enc).logits, dim=-1).numpy())
        probs = np.vstack(probs)
        p = {labels[i]: probs[:, i] for i in range(probs.shape[1])}
        net = p["positive"] - p["negative"]
        is_debt = np.array([bool(DEBT_WORDS.search(s)) for s in sents])
        pred = probs.argmax(axis=1)
        neg_id = [i for i, l in labels.items() if l == "negative"][0]
        pos_id = [i for i, l in labels.items() if l == "positive"][0]

        feats.append({
            "save_as": save_as,
            "n_sentences": len(sents),
            "fb_neg_share": float((pred == neg_id).mean()),
            "fb_pos_share": float((pred == pos_id).mean()),
            "fb_mean_neg": float(p["negative"].mean()),
            "fb_net_tone": float(net.mean()),
            "fb_tone_std": float(net.std()),
            "fb_neg_p90": float(np.quantile(p["negative"], 0.9)),       # how bad the worst part is
            "fb_debt_share": float(is_debt.mean()),
            "fb_debt_tone": float(net[is_debt].mean()) if is_debt.any() else 0.0,
        })
        for i in np.argsort(-p["negative"])[:5]:
            top_neg.append({"save_as": save_as, "p_negative": float(p["negative"][i]), "sentence": sents[i]})

        if n % 10 == 0 or n == len(meta):
            print(f"  {n}/{len(meta)} reports scored")

    out = pd.DataFrame(feats)
    fb_cols = ["fb_neg_share", "fb_pos_share", "fb_mean_neg", "fb_net_tone", "fb_tone_std",
               "fb_neg_p90", "fb_debt_share", "fb_debt_tone"]
    out = out.reindex(columns=["save_as", "n_sentences"] + fb_cols)   # reports with <10 sentences -> NaN
    print(f"Reports with too little text (<10 sentences, features left blank): "
          f"{(out['n_sentences'] < 10).sum()}")
    out = out.merge(meta, on="save_as", how="left")
    out.to_csv(OUT, index=False)
    pd.DataFrame(top_neg).to_csv(OUT_SENT, index=False)

    cols = ["fb_neg_share", "fb_pos_share", "fb_net_tone", "fb_neg_p90", "fb_debt_tone"]
    print("\nFinBERT features - average by group:")
    print(out.groupby("label")[cols].mean().T.rename(columns={0: "healthy (0)", 1: "distressed (1)"})
          .round(3).to_string())
    print(f"\nSaved: {OUT}, {OUT_SENT}")


if __name__ == "__main__":
    main() 