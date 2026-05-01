"""
Forensic analysis of LOSING trades on Oracle holdout. Goal: find any
clean cohort (rule, cluster, hour, exit type, feature pattern) where
WR is materially below average AND we can kill it with little impact
on winners.

We're looking for something the meta gate missed because the meta gate
is a single global classifier. Local pockets ("C2_R1c is bad", "Asian
session is bad") are exactly what a global gate can underweight if
the bad-cohort signal averages out across the rest of the data.

Outputs:
  - Per-rule WR + R contribution
  - Per-cluster WR
  - Per-hour WR
  - Per-exit-type WR (hard_sl vs ml_exit vs max)
  - Per-direction WR
  - Significance test: is any low-WR cohort statistically different
    from baseline? (z-test on proportion)
  - Combined kill filter: drop all cohorts with WR < threshold AND
    low total R contribution. Report new WR/PF/R.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = "/home/jay/Desktop/new-model-zigzag"


def pf(rs):
    pos = rs[rs > 0].sum(); neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def cohort_table(trades, group_col, base_wr, min_n=20):
    rows = []
    for k, g in trades.groupby(group_col):
        if len(g) < min_n: continue
        wr = (g["pnl_R"] > 0).mean()
        r  = g["pnl_R"].sum()
        wins = int((g["pnl_R"] > 0).sum())
        # binomial p-value: probability of seeing ≤wins out of len(g) under base_wr
        p = binomtest(wins, len(g), base_wr, alternative="less").pvalue
        rows.append({
            group_col: k, "n": len(g), "WR": round(wr, 3),
            "R": round(r, 1), "p_low_WR": round(p, 4),
            "PF": round(pf(g["pnl_R"].values), 2),
        })
    out = pd.DataFrame(rows).sort_values("WR")
    return out


def main():
    print("loading Oracle holdout trades…")
    df = pd.read_csv(os.path.join(ROOT, "data/v72l_trades_holdout.csv"),
                      parse_dates=["time"])
    df["hour"]   = df["time"].dt.hour
    df["dow"]    = df["time"].dt.dayofweek
    df["is_win"] = (df["pnl_R"] > 0).astype(int)
    base_wr = df["is_win"].mean()
    base_pf = pf(df["pnl_R"].values)
    print(f"baseline: n={len(df)}  WR={base_wr:.1%}  PF={base_pf:.2f}  "
          f"R={df['pnl_R'].sum():+.1f}")

    # Per-cluster
    print("\n=== Per-cluster ===")
    print(cohort_table(df, "cid", base_wr).to_string(index=False))

    # Per-rule (top + bottom)
    print("\n=== Per-rule (sorted by WR ascending) ===")
    rt = cohort_table(df, "rule", base_wr, min_n=15)
    print(rt.head(15).to_string(index=False))
    print("...")
    print(rt.tail(5).to_string(index=False))

    # Per-cluster x rule
    print("\n=== (cid, rule) cohorts with WR < 50% AND n ≥ 20 ===")
    df["cid_rule"] = df["cid"].astype(str) + "_" + df["rule"]
    crt = cohort_table(df, "cid_rule", base_wr, min_n=20)
    bad = crt[(crt["WR"] < 0.50) & (crt["n"] >= 20)]
    print(bad.to_string(index=False) if len(bad) else "  (none — no (cid,rule) cohort under 50% WR with n≥20)")

    # Per-hour
    print("\n=== Per-hour (UTC) ===")
    ht = cohort_table(df, "hour", base_wr, min_n=30)
    print(ht.to_string(index=False))

    # Per-exit type
    print("\n=== Per-exit type ===")
    print(cohort_table(df, "exit", base_wr, min_n=10).to_string(index=False))

    # Per-direction
    print("\n=== Per-direction ===")
    print(cohort_table(df, "direction", base_wr).to_string(index=False))

    # ---- Kill the worst cohorts and see effect ----
    print("\n=== KILL TEST: drop all (cid, rule) cohorts with WR < 55% AND n ≥ 30 ===")
    bad_keys = set(crt[(crt["WR"] < 0.55) & (crt["n"] >= 30)]["cid_rule"])
    print(f"  killing {len(bad_keys)} cohorts: {sorted(bad_keys)[:10]}{'…' if len(bad_keys)>10 else ''}")
    keep = ~df["cid_rule"].isin(bad_keys)
    rs0 = df["pnl_R"].values
    rs1 = df.loc[keep, "pnl_R"].values
    print(f"  before: n={len(rs0)} WR={(rs0>0).mean():.1%} PF={pf(rs0):.2f} R={rs0.sum():+.1f}")
    print(f"  after:  n={len(rs1)} WR={(rs1>0).mean():.1%} PF={pf(rs1):.2f} R={rs1.sum():+.1f}")
    print(f"  Δ:      n={len(rs1)-len(rs0):+d}  ΔWR={((rs1>0).mean()-(rs0>0).mean())*100:+.1f}pp"
          f"  ΔPF={pf(rs1)-pf(rs0):+.2f}  ΔR={rs1.sum()-rs0.sum():+.1f}")

    # Try harder: WR < 60% with n ≥ 30
    print("\n=== HARDER KILL: drop (cid, rule) with WR < 60% AND n ≥ 30 ===")
    bad_keys = set(crt[(crt["WR"] < 0.60) & (crt["n"] >= 30)]["cid_rule"])
    print(f"  killing {len(bad_keys)} cohorts")
    keep = ~df["cid_rule"].isin(bad_keys)
    rs1 = df.loc[keep, "pnl_R"].values
    print(f"  after:  n={len(rs1)} WR={(rs1>0).mean():.1%} PF={pf(rs1):.2f} R={rs1.sum():+.1f}"
          f"  Δn={len(rs1)-len(rs0):+d}"
          f"  ΔWR={((rs1>0).mean()-(rs0>0).mean())*100:+.1f}pp"
          f"  ΔPF={pf(rs1)-pf(rs0):+.2f}  ΔR={rs1.sum()-rs0.sum():+.1f}")

    # ALSO drop hours with WR < 55%
    print("\n=== ADD-ON: ALSO drop hours with WR < 55% AND n ≥ 30 ===")
    bad_hours = set(ht[(ht["WR"] < 0.55) & (ht["n"] >= 30)]["hour"])
    print(f"  killing hours: {sorted(bad_hours)}")
    keep2 = keep & ~df["hour"].isin(bad_hours)
    rs2 = df.loc[keep2, "pnl_R"].values
    print(f"  after:  n={len(rs2)} WR={(rs2>0).mean():.1%} PF={pf(rs2):.2f} R={rs2.sum():+.1f}"
          f"  Δn={len(rs2)-len(rs0):+d}"
          f"  ΔWR={((rs2>0).mean()-(rs0>0).mean())*100:+.1f}pp"
          f"  ΔPF={pf(rs2)-pf(rs0):+.2f}  ΔR={rs2.sum()-rs0.sum():+.1f}")


if __name__ == "__main__":
    main()
