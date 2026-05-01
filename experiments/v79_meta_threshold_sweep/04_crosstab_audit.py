"""
Direction × cluster + hour × cluster crosstabs on holdout trades.
Looking for narrower-than-cohort kill rules that the global meta gate
+ v79 cohort kills missed.

Output:
  - Per (cid, direction) WR/PF/R/p-value
  - Per (cid, hour-band) WR/PF/R/p-value
  - Walk-forward H1→H2 validation of any kill that looks promising
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = "/home/jay/Desktop/new-model-zigzag"

# Already-killed cohorts from v7.9 — apply BEFORE this audit so we don't
# double-count. Same as production configs.
KILLED = {
    "oracle_xau": {(2, "R0e_nr4_break")},
    "midas_xau":  {(1, "R0c_doubletouch"),
                    (2, "R0d_squeeze"),
                    (2, "R0e_nr4_break"),
                    (2, "R0h_3bar_reversal")},
    "oracle_btc": {(2, "R0h_3bar_reversal")},
}


def pf(rs):
    pos = rs[rs > 0].sum(); neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def post_v79(df, slug):
    df = df.copy()
    df["cr"] = list(zip(df["cid"], df["rule"]))
    return df[~df["cr"].isin(KILLED.get(slug, set()))].drop(columns=["cr"])


def crosstab(df, group_cols, base_wr, min_n):
    rows = []
    for k, g in df.groupby(group_cols):
        if len(g) < min_n: continue
        wr = (g["pnl_R"] > 0).mean()
        wins = int((g["pnl_R"] > 0).sum())
        p = binomtest(wins, len(g), base_wr, alternative="less").pvalue
        rows.append({
            **dict(zip(group_cols, k if isinstance(k, tuple) else (k,))),
            "n": len(g), "WR": round(wr, 3),
            "R": round(g["pnl_R"].sum(), 1),
            "PF": round(pf(g["pnl_R"].values), 2),
            "p_low_WR": round(p, 4),
        })
    return pd.DataFrame(rows).sort_values("WR")


def kill_test(df, name, mask_fn, label):
    keep = ~mask_fn(df)
    rs0 = df["pnl_R"].values; rs1 = df.loc[keep, "pnl_R"].values
    base_wr = (rs0 > 0).mean(); base_pf = pf(rs0); base_r = rs0.sum()
    new_wr = (rs1 > 0).mean(); new_pf = pf(rs1); new_r = rs1.sum()
    print(f"  KILL {label}: n {len(rs0)}→{len(rs1)} "
          f"WR {base_wr*100:.1f}%→{new_wr*100:.1f}% "
          f"PF {base_pf:.2f}→{new_pf:.2f} "
          f"R {base_r:+.0f}→{new_r:+.0f} "
          f"(ΔWR={(new_wr-base_wr)*100:+.1f}pp ΔPF={new_pf-base_pf:+.2f} ΔR={new_r-base_r:+.0f})")


def walk_forward_dir(df, slug):
    df = df.sort_values("time").reset_index(drop=True)
    h1 = df.iloc[:len(df)//2]; h2 = df.iloc[len(df)//2:]
    base_wr_h1 = (h1["pnl_R"] > 0).mean()
    print(f"\n  WALK-FORWARD: identify (cid, direction) bad on H1, apply to H2")
    h1_xt = crosstab(h1, ["cid", "direction"], base_wr_h1, min_n=20)
    bad_h1 = set()
    for _, row in h1_xt.iterrows():
        if row["WR"] < 0.50 and row["p_low_WR"] < 0.05:
            bad_h1.add((int(row["cid"]), int(row["direction"])))
    print(f"    H1 bad (cid, dir) cohorts (WR<50%, p<0.05): {sorted(bad_h1)}")
    if not bad_h1:
        print("    (none — no clean kill candidate from direction × cluster)"); return

    h2["cd"] = list(zip(h2["cid"], h2["direction"]))
    keep = ~h2["cd"].isin(bad_h1)
    rs0 = h2["pnl_R"].values; rs1 = h2.loc[keep, "pnl_R"].values
    print(f"    H2 base: n={len(rs0)} WR={(rs0>0).mean():.1%} PF={pf(rs0):.2f} R={rs0.sum():+.1f}")
    print(f"    H2 kill: n={len(rs1)} WR={(rs1>0).mean():.1%} PF={pf(rs1):.2f} R={rs1.sum():+.1f} "
          f"ΔWR={((rs1>0).mean()-(rs0>0).mean())*100:+.1f}pp "
          f"ΔPF={pf(rs1)-pf(rs0):+.2f} ΔR={rs1.sum()-rs0.sum():+.1f}")


def analyze(name, slug, path):
    print("="*72); print(f"=== {name} (post-v7.9 cohort kills) ===")
    df = pd.read_csv(path, parse_dates=["time"])
    df = post_v79(df, slug)
    df["hour"] = df["time"].dt.hour
    base_wr = (df["pnl_R"] > 0).mean()
    print(f"  n={len(df)} WR={base_wr:.1%} PF={pf(df['pnl_R'].values):.2f} "
          f"R={df['pnl_R'].sum():+.1f}")

    print("\n[1] Direction × cluster crosstab (n>=30, sorted by WR):")
    xt = crosstab(df, ["cid", "direction"], base_wr, min_n=30)
    print(xt.to_string(index=False))

    # Kill test: any (cid, dir) with WR<50% AND p<0.05
    bad_cd = set()
    for _, row in xt.iterrows():
        if row["WR"] < 0.50 and row["p_low_WR"] < 0.05:
            bad_cd.add((int(row["cid"]), int(row["direction"])))
    if bad_cd:
        print(f"\n  Bad (cid, dir) cohorts: {sorted(bad_cd)}")
        df["cd"] = list(zip(df["cid"], df["direction"]))
        kill_test(df, name, lambda d: d["cd"].isin(bad_cd),
                   f"(cid,dir) WR<50% p<0.05")
        df.drop(columns=["cd"], inplace=True)
    else:
        print("\n  No (cid, dir) cohort with WR<50% AND p<0.05.")

    print("\n[2] Hour × cluster crosstab (n>=20, sorted by WR), top 15 worst:")
    xt_h = crosstab(df, ["cid", "hour"], base_wr, min_n=20)
    print(xt_h.head(15).to_string(index=False))

    bad_ch = set()
    for _, row in xt_h.iterrows():
        if row["WR"] < 0.45 and row["p_low_WR"] < 0.05 and row["R"] < 0:
            bad_ch.add((int(row["cid"]), int(row["hour"])))
    if bad_ch:
        print(f"\n  Bad (cid, hour) cohorts (WR<45%, p<0.05, losing money): {sorted(bad_ch)}")
        df["ch"] = list(zip(df["cid"], df["hour"]))
        kill_test(df, name, lambda d: d["ch"].isin(bad_ch),
                   "(cid,hour) WR<45% p<0.05 R<0")
        df.drop(columns=["ch"], inplace=True)
    else:
        print("\n  No (cid, hour) cohort meeting strict bar (WR<45% & p<0.05 & R<0).")

    walk_forward_dir(df, slug)


if __name__ == "__main__":
    analyze("Oracle XAU", "oracle_xau",
             os.path.join(ROOT, "data/v72l_trades_holdout.csv"))
    analyze("Midas XAU",  "midas_xau",
             os.path.join(ROOT, "data/v6_trades_holdout_xau.csv"))
    analyze("Oracle BTC", "oracle_btc",
             os.path.join(ROOT, "data/v72l_trades_holdout_btc.csv"))
