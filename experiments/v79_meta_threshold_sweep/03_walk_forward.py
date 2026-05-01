"""
Walk-forward validation of the kill-bad-cohorts rule.

The forensic identified cohorts on the FULL holdout — that's data
dredging. Real test: identify on H1 of holdout, apply to H2.

If the rule holds (WR up, R neutral or up) on H2 — it generalizes.
If H2 reverses (the "bad" cohorts in H1 were good in H2) — it's noise.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

ROOT = "/home/jay/Desktop/new-model-zigzag"


def pf(rs):
    pos = rs[rs > 0].sum(); neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def find_bad_cohorts(df, wr_thr=0.50, min_n=30):
    """Return set of (cid_rule) cohorts where WR < wr_thr and n >= min_n."""
    df = df.copy()
    df["cr"] = df["cid"].astype(str) + "_" + df["rule"]
    bad = set()
    for k, g in df.groupby("cr"):
        if len(g) >= min_n and (g["pnl_R"] > 0).mean() < wr_thr:
            bad.add(k)
    return bad


def stats(rs):
    return {
        "n":  len(rs),
        "WR": (rs > 0).mean() if len(rs) else 0.0,
        "PF": pf(rs),
        "R":  rs.sum(),
    }


def evaluate(df, name, wr_thr=0.50, min_n=30):
    print(f"\n=== {name} ===")
    df = df.sort_values("time").reset_index(drop=True).copy()
    df["cr"] = df["cid"].astype(str) + "_" + df["rule"]
    n = len(df)
    h1 = df.iloc[:n // 2]
    h2 = df.iloc[n // 2:]
    print(f"  H1: {len(h1)}  ({h1.time.min().date()} → {h1.time.max().date()})")
    print(f"  H2: {len(h2)}  ({h2.time.min().date()} → {h2.time.max().date()})")

    # Identify bad cohorts on H1 only
    bad = find_bad_cohorts(h1, wr_thr=wr_thr, min_n=min_n)
    print(f"  Bad cohorts on H1 (WR<{wr_thr}, n>={min_n}): {sorted(bad)}")

    # Apply to H2
    h2_keep = h2[~h2["cr"].isin(bad)]
    h2_drop = h2[ h2["cr"].isin(bad)]

    s_full = stats(h2["pnl_R"].values)
    s_keep = stats(h2_keep["pnl_R"].values)
    s_drop = stats(h2_drop["pnl_R"].values) if len(h2_drop) else None

    print(f"\n  H2 baseline: n={s_full['n']} WR={s_full['WR']:.1%} PF={s_full['PF']:.2f} R={s_full['R']:+.1f}")
    print(f"  H2 after kill: n={s_keep['n']} WR={s_keep['WR']:.1%} PF={s_keep['PF']:.2f} R={s_keep['R']:+.1f}"
          f"  ΔWR={(s_keep['WR']-s_full['WR'])*100:+.1f}pp"
          f"  ΔPF={s_keep['PF']-s_full['PF']:+.2f}"
          f"  ΔR={s_keep['R']-s_full['R']:+.1f}")
    if s_drop:
        print(f"  H2 dropped trades (the cohorts we killed): "
              f"n={s_drop['n']} WR={s_drop['WR']:.1%} R={s_drop['R']:+.1f}")
        verdict = "✅ PASS — bad H1 cohorts also bad in H2" if s_drop["R"] <= 0 \
            else "⚠ MIXED — H1 bad cohorts were profitable in H2"
        if s_drop["WR"] < s_full["WR"]:
            verdict += " (but WR still below average → likely real)"
        print(f"  Verdict: {verdict}")


def main():
    oracle = pd.read_csv(os.path.join(ROOT, "data/v72l_trades_holdout.csv"),
                          parse_dates=["time"])
    midas  = pd.read_csv(os.path.join(ROOT, "data/v6_trades_holdout_xau.csv"),
                          parse_dates=["time"])

    print("ORACLE walk-forward (v7.2-lite, deployed meta=0.675)")
    evaluate(oracle, "Oracle WR<50% n>=30", wr_thr=0.50, min_n=30)
    evaluate(oracle, "Oracle WR<55% n>=30", wr_thr=0.55, min_n=30)

    print("\n" + "="*70)
    print("MIDAS walk-forward (v6, no meta)")
    evaluate(midas, "Midas WR<50% n>=30", wr_thr=0.50, min_n=30)
    evaluate(midas, "Midas WR<55% n>=30", wr_thr=0.55, min_n=30)


if __name__ == "__main__":
    main()
