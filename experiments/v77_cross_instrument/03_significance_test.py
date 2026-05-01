"""
Permutation test for v77 gate. Question: when the gate keeps the
top-quintile (e.g. 107 Oracle trades at thr=0.70 → PF 8.04 vs
baseline 3.47), is that PF jump significant or could a random subset
of the same size have hit it?

Method: 5000 random subsets of size N. Compute PF for each. Report
where the gate's PF lands in that distribution.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = "/home/jay/Desktop/new-model-zigzag"
DATA = os.path.join(os.path.dirname(__file__), "data")

HOLDOUT_START = pd.Timestamp("2024-12-12")
RNG = np.random.default_rng(0)


def pf(rs):
    pos = rs[rs > 0].sum(); neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def perm_pf(rs, n_keep, n_iter=5000):
    """Distribution of PF when picking n_keep trades at random."""
    out = np.empty(n_iter)
    idx = np.arange(len(rs))
    for i in range(n_iter):
        chosen = RNG.choice(idx, size=n_keep, replace=False)
        out[i] = pf(rs[chosen])
    return out


def fit_clf():
    feats = pd.read_parquet(os.path.join(DATA, "features_labels.parquet"))
    fcols = [c for c in feats.columns
             if c not in {"date","n_trades","sum_R","label","split"}]
    midas = pd.read_csv(os.path.join(ROOT, "data/v6_trades_holdout_xau.csv"),
                         parse_dates=["time"])
    midas["date"] = midas["time"].dt.normalize()
    m = midas.merge(feats[["date"]+fcols], on="date", how="inner")
    train = m[m["date"] < HOLDOUT_START]
    Xtr = train[fcols].values
    ytr = (train["pnl_R"] > 0).astype(int).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
    clf.fit(sc.transform(Xtr), ytr)
    return clf, sc, fcols, feats


def test_one(name, trades_csv, clf, sc, fcols, feats):
    df = pd.read_csv(os.path.join(ROOT, trades_csv), parse_dates=["time"])
    df["date"] = df["time"].dt.normalize()
    df = df.merge(feats[["date"]+fcols], on="date", how="inner")
    df = df[df["date"] >= HOLDOUT_START]
    rs = df["pnl_R"].values
    p  = clf.predict_proba(sc.transform(df[fcols].values))[:, 1]
    base = pf(rs)
    print(f"\n=== {name} ===  n={len(rs)}  baseline PF={base:.3f}")
    for thr in [0.55, 0.60, 0.65, 0.70]:
        keep = p > thr
        if keep.sum() < 30:
            continue
        gate_pf = pf(rs[keep])
        null_pf = perm_pf(rs, int(keep.sum()))
        pct = (null_pf >= gate_pf).mean()
        ge_baseline_share = (null_pf >= base).mean()
        print(f"  thr={thr:.2f}  kept={keep.sum():>4}  gate PF={gate_pf:.2f}"
              f"  null mean={null_pf.mean():.2f}  null 95th={np.percentile(null_pf,95):.2f}"
              f"  p-value={pct:.3f}"
              f"  (null≥baseline share={ge_baseline_share:.2f})")


def main():
    clf, sc, fcols, feats = fit_clf()
    test_one("midas_holdout",  "data/v6_trades_holdout_xau.csv",
             clf, sc, fcols, feats)
    test_one("oracle_holdout", "data/v72l_trades_holdout.csv",
             clf, sc, fcols, feats)


if __name__ == "__main__":
    main()
