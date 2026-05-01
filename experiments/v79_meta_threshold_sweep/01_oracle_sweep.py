"""
Sweep Oracle XAU meta gate threshold ABOVE the deployed 0.675.

The saved holdout trades file already has meta@0.675 applied. So we can
only sweep UPWARD (tighter) — which is exactly what we want for "higher WR".

Method:
  1. Load v72l_trades_holdout.csv (post-meta@0.675 trades, with realized pnl_R).
  2. Look up V72L features at each trade time from setups_*_v72l.csv.
  3. Score each trade with the pickled meta model → p_meta.
  4. Sweep thresholds 0.675 → 0.95, report n / WR / PF / R / DD.
"""
from __future__ import annotations
import os, glob, pickle
import numpy as np
import pandas as pd

ROOT = "/home/jay/Desktop/new-model-zigzag"
PKL  = "/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine/models/oracle_xau_validated.pkl"
HOLDOUT = pd.Timestamp("2024-12-12")


def pf(rs):
    pos = rs[rs > 0].sum(); neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def max_dd(rs):
    eq = np.cumsum(rs); peak = np.maximum.accumulate(eq)
    return (eq - peak).min()


def main():
    print("loading pickle…")
    obj = pickle.load(open(PKL, "rb"))
    meta_mdl  = obj["meta_mdl"]
    META_FEATS = obj["meta_feats"]
    V72L_FEATS = obj["v72l_feats"]
    print(f"  meta_threshold deployed: {obj['meta_threshold']}")
    print(f"  meta_feats ({len(META_FEATS)}): {META_FEATS}")

    print("loading holdout trades…")
    trades = pd.read_csv(os.path.join(ROOT, "data/v72l_trades_holdout.csv"),
                          parse_dates=["time"])
    trades["cid"] = trades["cid"].astype(int)
    trades["direction"] = trades["direction"].astype(int)

    print("loading + concat setups_*_v72l.csv…")
    setups = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data/setups_*_v72l.csv"))):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        setups.append(df)
    setups = pd.concat(setups, ignore_index=True)
    setups["cid"] = setups["cid"].astype(int)
    setups["direction"] = setups["direction"].astype(int)

    # Match on (time, cid, rule, direction) — sufficient
    merged = trades.merge(setups[["time","cid","rule","direction"] + V72L_FEATS],
                            on=["time","cid","rule","direction"], how="left")
    missing = merged[V72L_FEATS].isna().any(axis=1).sum()
    print(f"  matched {len(merged) - missing}/{len(merged)} trades to setup features")
    merged = merged.dropna(subset=V72L_FEATS)

    # Score with meta
    X = merged[META_FEATS].fillna(0).values
    p_meta = meta_mdl.predict_proba(X)[:, 1]
    merged["p_meta"] = p_meta
    print(f"  p_meta distribution: min={p_meta.min():.3f} med={np.median(p_meta):.3f}"
          f"  p25={np.percentile(p_meta,25):.3f}  p75={np.percentile(p_meta,75):.3f}"
          f"  max={p_meta.max():.3f}")

    # Baseline (current deployed thr=0.675) is the file as-is
    rs0 = merged["pnl_R"].values
    base_n  = len(rs0); base_wr = (rs0 > 0).mean()
    base_pf = pf(rs0); base_R  = rs0.sum(); base_dd = max_dd(rs0)
    print(f"\nDeployed (thr ≥ 0.675):  n={base_n}  WR={base_wr:.1%}  PF={base_pf:.2f}"
          f"  R={base_R:+.1f}  DD={base_dd:+.1f}")

    print("\nSweep above deployed:")
    print(f"{'thr':>6} {'n':>5} {'kept%':>6} {'WR':>6} {'PF':>5} {'R':>9} {'DD':>7}  vs base")
    rows = []
    for thr in [0.675, 0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875, 0.90]:
        keep = p_meta >= thr
        if keep.sum() < 30: break
        rs = merged.loc[keep, "pnl_R"].values
        n  = len(rs); wr = (rs > 0).mean()
        pf_ = pf(rs); R_ = rs.sum(); dd_ = max_dd(rs)
        kept_pct = keep.sum() / base_n
        delta_R  = R_ - base_R
        rows.append({"thr": thr, "n": n, "kept_pct": round(kept_pct, 3),
                     "WR": round(wr, 4), "PF": round(pf_, 2),
                     "R": round(R_, 1), "DD": round(dd_, 1),
                     "ΔR": round(delta_R, 1),
                     "ΔWR_pp": round((wr - base_wr) * 100, 1)})
        print(f"{thr:>6.3f} {n:>5} {kept_pct:>6.1%} {wr:>6.1%} {pf_:>5.2f} "
              f"{R_:>+9.1f} {dd_:>+7.1f}  ΔR={delta_R:+8.1f}  ΔWR={ (wr-base_wr)*100:+.1f}pp")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(os.path.dirname(__file__), "oracle_sweep.csv"), index=False)


if __name__ == "__main__":
    main()
