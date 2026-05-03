"""v8.9b — same as 02 but using 5m flow agreement instead of 4h."""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

ROOT = "/home/jay/Desktop/new-model-zigzag"
sys.path.insert(0, os.path.dirname(__file__))
from importlib.machinery import SourceFileLoader
qf = SourceFileLoader("qf01", os.path.join(os.path.dirname(__file__), "01_port_and_test.py")).load_module()


def pf(rs):
    pos = rs[rs > 0].sum(); neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def attach_flow(trades_path, swing_path):
    swing = pd.read_csv(swing_path, parse_dates=["time"])
    swing = swing[["time", "open", "high", "low", "close", "spread"]].sort_values("time").reset_index(drop=True)
    print(f"  swing: {len(swing)} bars", flush=True)

    print("  computing 5m flow...", flush=True)
    flow_5m = qf.quantum_flow(swing)
    swing["flow_5m"] = flow_5m.values

    trades = pd.read_csv(trades_path, parse_dates=["time"])
    trades["direction"] = trades["direction"].astype(int)
    trades = trades.sort_values("time").reset_index(drop=True)

    out = pd.merge_asof(trades, swing[["time", "flow_5m"]], on="time",
                          direction="backward",
                          tolerance=pd.Timedelta("10min"))
    out = out.dropna(subset=["flow_5m"]).reset_index(drop=True)
    out["agree_5m"] = (np.sign(out.flow_5m) == out.direction)
    return out


def evaluate(name, df):
    rs0 = df.pnl_R.values
    base_n = len(rs0); base_wr = (rs0>0).mean(); base_pf = pf(rs0); base_R = rs0.sum()
    keep = df.agree_5m.values
    rs1 = df.loc[keep, "pnl_R"].values
    new_n = len(rs1); new_wr = (rs1>0).mean(); new_pf = pf(rs1); new_R = rs1.sum()
    skipped = df.loc[~keep, "pnl_R"].values
    skipped_R = skipped.sum() if len(skipped) else 0
    skipped_wr = (skipped > 0).mean() if len(skipped) else 0
    print(f"\n=== {name} ===")
    print(f"  baseline:  n={base_n} WR={base_wr:.1%} PF={base_pf:.2f} R={base_R:+.0f}")
    print(f"  5m-agree:  n={new_n} WR={new_wr:.1%} PF={new_pf:.2f} R={new_R:+.0f}  "
          f"ΔWR={(new_wr-base_wr)*100:+.1f}pp ΔPF={new_pf-base_pf:+.2f} ΔR={new_R-base_R:+.0f}")
    print(f"  skipped:   n={base_n-new_n} WR={skipped_wr:.1%} R={skipped_R:+.0f}  "
          f"({(base_n-new_n)/base_n:.0%} of trades dropped)")


def main():
    print("--- Oracle XAU ---")
    df_o = attach_flow(os.path.join(ROOT, "data/v72l_trades_holdout.csv"),
                         os.path.join(ROOT, "data/swing_v5_xauusd.csv"))
    evaluate("Oracle XAU", df_o)

    print("\n--- Midas XAU ---")
    df_m = attach_flow(os.path.join(ROOT, "data/v6_trades_holdout_xau.csv"),
                         os.path.join(ROOT, "data/swing_v5_xauusd.csv"))
    evaluate("Midas XAU", df_m)

    print("\n--- Oracle BTC ---")
    df_b = attach_flow(os.path.join(ROOT, "data/v72l_trades_holdout_btc.csv"),
                         os.path.join(ROOT, "data/swing_v5_btc.csv"))
    evaluate("Oracle BTC", df_b)


if __name__ == "__main__":
    main()
