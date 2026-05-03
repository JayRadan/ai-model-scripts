"""Redo v8.9 + v8.9b flow filter tests using REAL Dukascopy volume.
Compare to spread-proxy results."""
import sys, os, glob
import pandas as pd
import numpy as np

sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/experiments/v89_quantum_flow_tiebreaker")
from importlib.machinery import SourceFileLoader
qf = SourceFileLoader("qf01", "/home/jay/Desktop/new-model-zigzag/experiments/v89_quantum_flow_tiebreaker/01_port_and_test.py").load_module()

ROOT = "/home/jay/Desktop/new-model-zigzag"


def pf(rs):
    pos = rs[rs > 0].sum(); neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def attach_flow(trades_path, swing_path, sym_label):
    swing = pd.read_csv(swing_path, parse_dates=["time"])
    print(f"  swing {sym_label}: {len(swing)} bars  {swing.time.min()} → {swing.time.max()}", flush=True)
    # Need: time, open, high, low, close, plus 'volume' or 'tick_volume'
    swing = swing.rename(columns={"tick_volume": "volume"}) if "tick_volume" in swing.columns else swing
    print(f"  cols: {list(swing.columns)[:10]}", flush=True)
    flow_5m = qf.quantum_flow(swing)
    flow_4h = qf.quantum_flow_mtf(swing)
    swing["flow_5m"] = flow_5m.values
    swing["flow_4h"] = flow_4h.values

    trades = pd.read_csv(trades_path, parse_dates=["time"])
    trades["direction"] = trades["direction"].astype(int)
    trades = trades.sort_values("time").reset_index(drop=True)

    out = pd.merge_asof(trades, swing[["time", "flow_5m", "flow_4h"]], on="time",
                          direction="backward",
                          tolerance=pd.Timedelta("10min"))
    out = out.dropna(subset=["flow_5m"]).reset_index(drop=True)
    out["agree_5m"] = (np.sign(out.flow_5m) == out.direction)
    out["agree_4h"] = (np.sign(out.flow_4h) == out.direction)
    return out


def evaluate(name, df, filter_col, filter_label):
    rs0 = df.pnl_R.values
    base_n = len(rs0); base_wr = (rs0 > 0).mean(); base_pf = pf(rs0); base_R = rs0.sum()
    keep = df[filter_col].fillna(False).values
    rs1 = df.loc[keep, "pnl_R"].values
    if len(rs1) == 0:
        print(f"  {filter_label}: ALL skipped"); return
    new_n = len(rs1); new_wr = (rs1 > 0).mean(); new_pf = pf(rs1); new_R = rs1.sum()
    print(f"  {filter_label:8s}: kept={new_n} ({new_n/base_n:.0%})  WR {new_wr:.1%}  PF {new_pf:.2f}  R {new_R:+.0f}  "
          f"ΔR {new_R-base_R:+.0f} ({(new_R-base_R)/base_R*100:+.1f}%)  ΔPF {new_pf-base_pf:+.2f}")


def run_one(name, trades_path, swing_path, sym):
    print(f"\n{'='*78}\n=== {name} ===")
    df = attach_flow(trades_path, swing_path, sym)
    rs0 = df.pnl_R.values
    print(f"  baseline:  n={len(rs0)} WR={(rs0>0).mean():.1%} PF={pf(rs0):.2f} R={rs0.sum():+.0f}")
    evaluate(name, df, "agree_5m", "5m only")
    evaluate(name, df, "agree_4h", "4h only")
    df["both"] = df.agree_5m & df.agree_4h
    evaluate(name, df, "both",     "BOTH")


if __name__ == "__main__":
    # Use Dukascopy XAU bars for Oracle/Midas (XAU instruments)
    run_one("Oracle XAU (real volume)",
             os.path.join(ROOT, "data/v72l_trades_holdout.csv"),
             "/tmp/duk/xau_m5.csv", "XAU")
    run_one("Midas XAU (real volume)",
             os.path.join(ROOT, "data/v6_trades_holdout_xau.csv"),
             "/tmp/duk/xau_m5.csv", "XAU")
    run_one("Oracle BTC (real volume)",
             os.path.join(ROOT, "data/v72l_trades_holdout_btc.csv"),
             "/tmp/duk/btc_m5.csv", "BTC")
