"""
Test cross-instrument signal as a POSITION SIZING multiplier rather
than a hard gate. The hard-gate test (03) showed Oracle PF jumps
significantly at p≥0.70 (p-value=0.001) but absolute R drops because
we keep too few trades. If we instead SIZE UP high-p trades and SIZE
DOWN low-p trades, we keep all volume but tilt R toward the signal.

Sizing schemes tested:
  A) Linear:    lot = 0.5 + 1.0 * p_good           (range 0.5 → 1.5)
  B) Stepped:   lot = 0.5 if p<.50, 1.0 if .50<=p<.65, 1.5 if p>=.65
  C) Aggressive: lot = 0.25 if p<.40, 0.75 if .40<=p<.60, 1.5 if p>=.60
  D) Top-heavy:  lot = 0.75 if p<.65, 2.0 if p>=.65

Compare total R, PF (in lot-weighted R), max DD vs baseline (lot=1.0).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = "/home/jay/Desktop/new-model-zigzag"
DATA = os.path.join(os.path.dirname(__file__), "data")
REPS = os.path.join(os.path.dirname(__file__), "reports")

HOLDOUT_START = pd.Timestamp("2024-12-12")


def pf(rs):
    pos = rs[rs > 0].sum(); neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def max_dd(rs):
    """Max drawdown of cumulative R (in R units)."""
    eq = np.cumsum(rs)
    peak = np.maximum.accumulate(eq)
    return (eq - peak).min()


SCHEMES = {
    "A_linear":     lambda p: 0.5 + 1.0 * p,
    "B_stepped":    lambda p: np.where(p < 0.50, 0.5,
                                       np.where(p < 0.65, 1.0, 1.5)),
    "C_aggressive": lambda p: np.where(p < 0.40, 0.25,
                                       np.where(p < 0.60, 0.75, 1.5)),
    "D_topheavy":   lambda p: np.where(p < 0.65, 0.75, 2.0),
}


def fit():
    feats = pd.read_parquet(os.path.join(DATA, "features_labels.parquet"))
    fcols = [c for c in feats.columns
             if c not in {"date","n_trades","sum_R","label","split"}]
    midas = pd.read_csv(os.path.join(ROOT, "data/v6_trades_holdout_xau.csv"),
                         parse_dates=["time"])
    midas["date"] = midas["time"].dt.normalize()
    m = midas.merge(feats[["date"]+fcols], on="date", how="inner")
    train = m[m["date"] < HOLDOUT_START]
    sc = StandardScaler().fit(train[fcols].values)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
    clf.fit(sc.transform(train[fcols].values),
            (train["pnl_R"] > 0).astype(int).values)
    return clf, sc, fcols, feats


def evaluate(name, trades_csv, clf, sc, fcols, feats):
    df = pd.read_csv(os.path.join(ROOT, trades_csv), parse_dates=["time"])
    df["date"] = df["time"].dt.normalize()
    df = df.merge(feats[["date"]+fcols], on="date", how="inner")
    df = df[df["date"] >= HOLDOUT_START].sort_values("time").reset_index(drop=True)
    if len(df) == 0: return
    rs = df["pnl_R"].values
    p  = clf.predict_proba(sc.transform(df[fcols].values))[:, 1]

    base_r = rs.sum(); base_pf = pf(rs); base_dd = max_dd(rs)
    print(f"\n=== {name} ===  n={len(rs)}")
    print(f"  baseline (lot=1.0): R={base_r:+.1f} PF={base_pf:.3f} maxDD={base_dd:+.1f}")

    rows = [{"scheme": "baseline_1.0", "R": round(base_r,1),
             "PF": round(base_pf,3), "maxDD": round(base_dd,1),
             "avg_lot": 1.0, "R_per_lot": round(base_r,3)}]
    for sname, fn in SCHEMES.items():
        lots = fn(p).astype(float)
        wr = rs * lots
        rows.append({
            "scheme":     sname,
            "R":          round(wr.sum(), 1),
            "PF":         round(pf(wr), 3),
            "maxDD":      round(max_dd(wr), 1),
            "avg_lot":    round(lots.mean(), 3),
            "R_per_lot":  round(wr.sum() / lots.sum(), 3),
        })
    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))
    df_out.to_csv(os.path.join(REPS, f"sizing_{name}.csv"), index=False)


def main():
    clf, sc, fcols, feats = fit()
    evaluate("midas",  "data/v6_trades_holdout_xau.csv", clf, sc, fcols, feats)
    evaluate("oracle", "data/v72l_trades_holdout.csv",   clf, sc, fcols, feats)


if __name__ == "__main__":
    main()
