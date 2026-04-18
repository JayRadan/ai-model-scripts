"""
v6 holdout backtest: uses the ACTUAL trained v6 models on unseen data.
Reports PF, win rate, expectancy, max DD, trade count, PnL.

Uses:
  - models/confirm_v6_c{cid}_{rule}.json (per-rule confirmation classifiers)
  - models/exit_v6.json (ML exit classifier)
  - Last 20% of data as holdout (never seen by any model)
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paths as P

import sys
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import importlib.machinery
p04b = importlib.machinery.SourceFileLoader(
    "p04b",
    "/home/jay/Desktop/new-model-zigzag/model_pipeline/04b_compute_physics_features.py",
).load_module()

V6_CONFIRM_FEATS = [
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "vwap_dist", "hour_enc", "dow_enc",
    "quantum_flow", "quantum_flow_h4", "quantum_momentum", "quantum_vwap_conf",
    "quantum_divergence", "quantum_div_strength",
]
EXIT_FEATS = [
    "unrealized_pnl_R", "bars_held", "pnl_velocity",
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "quantum_flow", "quantum_flow_h4", "vwap_dist",
]

MAX_HOLD = 60
MIN_HOLD = 2
SL_HARD = 4.0
EXIT_THRESHOLD = 0.55

def main():
    print("Loading swing data + physics features...", flush=True)
    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)

    c = swing["close"].values.astype(np.float64)
    h = swing["high"].values.astype(np.float64)
    l = swing["low"].values.astype(np.float64)
    o = swing["open"].values.astype(np.float64)
    vol = np.maximum(swing["spread"].values.astype(np.float64), 1.0)
    n = len(c)

    tr = np.concatenate([[h[0]-l[0]],
          np.maximum.reduce([h[1:]-l[1:], np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    ret = np.concatenate([[0.0], np.diff(np.log(c))])

    # Compute physics at every bar (needed for exit model)
    print("  Computing physics features (may take a moment)...", flush=True)
    swing["hurst_rs"]     = p04b.compute_hurst_rs(ret)
    swing["ou_theta"]     = p04b.compute_ou_theta(ret)
    swing["entropy_rate"] = p04b.compute_entropy(ret)
    swing["kramers_up"]   = p04b.compute_kramers_up(c)
    swing["wavelet_er"]   = p04b.compute_wavelet_er(c)
    swing["vwap_dist"]    = p04b.compute_vwap_dist(swing, atr)
    swing["quantum_flow"] = p04b.compute_quantum_flow(o, h, l, c, vol)
    df_h4 = swing.set_index("time")[["open","high","low","close","spread"]].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last","spread":"sum"}).dropna()
    qf_h4 = p04b.compute_quantum_flow(
        df_h4["open"].values, df_h4["high"].values, df_h4["low"].values,
        df_h4["close"].values, np.maximum(df_h4["spread"].values, 1.0))
    qf_h4_s = pd.Series(qf_h4, index=df_h4.index).shift(1)
    swing["quantum_flow_h4"] = qf_h4_s.reindex(swing.set_index("time").index, method="ffill").values
    swing = swing.fillna(0)
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)

    # Load all v6 setup rows (pre-built in data/setups_*_v6.csv)
    print("  Loading v6 setup CSVs...", flush=True)
    all_setups = []
    for f in sorted(glob.glob(P.data("setups_*_v6.csv"))):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        all_setups.append(df)
    all_df = pd.concat(all_setups, ignore_index=True).sort_values("time").reset_index(drop=True)
    print(f"  Total setups: {len(all_df):,}")

    # Holdout: last 20% by time
    split_time = all_df["time"].quantile(0.80)
    holdout = all_df[all_df["time"] > split_time].copy().reset_index(drop=True)
    print(f"  Holdout: {len(holdout):,} setups from {holdout['time'].iat[0]} to {holdout['time'].iat[-1]}")

    # Run per-rule confirmation on holdout setups
    print("\nRunning confirmation classifiers...", flush=True)
    confirmed_rows = []
    for (cid, rule), grp in holdout.groupby(["cid", "rule"]):
        meta_path = P.model(f"confirm_v6_c{cid}_{rule}_meta.json")
        model_path = P.model(f"confirm_v6_c{cid}_{rule}.json")
        if not (os.path.exists(meta_path) and os.path.exists(model_path)): continue
        meta = json.load(open(meta_path))
        if meta.get("disabled", False): continue
        thr = meta["threshold"]

        mdl = XGBClassifier(); mdl.load_model(model_path)
        X = grp[V6_CONFIRM_FEATS].fillna(0).values
        proba = mdl.predict_proba(X)[:, 1]
        mask = proba >= thr
        confirmed_rows.append(grp[mask].copy())

    confirmed = pd.concat(confirmed_rows, ignore_index=True).sort_values("time").reset_index(drop=True)
    print(f"  Confirmed entries: {len(confirmed):,}")

    # Load exit model
    exit_mdl = XGBClassifier(); exit_mdl.load_model(P.model("exit_v6.json"))

    # Simulate trades with ML exit (exactly matching MQL5 live logic)
    print("\nSimulating trades with ML exit...", flush=True)
    trades = []
    equity = [0.0]

    for _, setup in confirmed.iterrows():
        t = setup["time"]
        if t not in time_to_idx.index: continue
        entry_idx = int(time_to_idx[t])
        direction = int(setup["direction"])
        entry_price = c[entry_idx]; entry_atr = atr[entry_idx]
        if not np.isfinite(entry_atr) or entry_atr <= 0: continue

        exit_idx = None
        exit_reason = "max"
        for k in range(1, MAX_HOLD+1):
            bar = entry_idx + k
            if bar >= n: break
            cur_pnl = direction * (c[bar] - entry_price) / entry_atr
            # Hard SL safety
            if cur_pnl < -SL_HARD:
                exit_idx = bar; exit_reason = "hard_sl"; break
            # ML exit after min hold
            if k >= MIN_HOLD:
                pnl_3ago = direction * (c[bar-3] - entry_price) / entry_atr if k >= 3 else cur_pnl
                row = {
                    "unrealized_pnl_R": cur_pnl,
                    "bars_held": float(k),
                    "pnl_velocity": cur_pnl - pnl_3ago,
                    "hurst_rs": swing["hurst_rs"].iat[bar],
                    "ou_theta": swing["ou_theta"].iat[bar],
                    "entropy_rate": swing["entropy_rate"].iat[bar],
                    "kramers_up": swing["kramers_up"].iat[bar],
                    "wavelet_er": swing["wavelet_er"].iat[bar],
                    "quantum_flow": swing["quantum_flow"].iat[bar],
                    "quantum_flow_h4": swing["quantum_flow_h4"].iat[bar],
                    "vwap_dist": swing["vwap_dist"].iat[bar],
                }
                X_exit = np.array([[row[f] for f in EXIT_FEATS]])
                p_exit = exit_mdl.predict_proba(X_exit)[0, 1]
                if p_exit >= EXIT_THRESHOLD:
                    exit_idx = bar; exit_reason = "ml_exit"; break
        if exit_idx is None:
            exit_idx = min(entry_idx + MAX_HOLD, n-1); exit_reason = "max"

        pnl_R = direction * (c[exit_idx] - entry_price) / entry_atr
        trades.append({
            "time": t, "cid": int(setup["cid"]), "rule": setup["rule"],
            "dir": direction, "bars": exit_idx - entry_idx,
            "pnl_R": pnl_R, "exit": exit_reason,
        })
        equity.append(equity[-1] + pnl_R)

    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        print("  NO TRADES"); return

    # Stats
    wins = trades_df[trades_df["pnl_R"] > 0]
    losses = trades_df[trades_df["pnl_R"] <= 0]
    total_win = wins["pnl_R"].sum()
    total_loss = -losses["pnl_R"].sum()
    pf = total_win / total_loss if total_loss > 0 else float("inf")
    wr = len(wins) / len(trades_df)
    expectancy_R = trades_df["pnl_R"].mean()
    total_R = trades_df["pnl_R"].sum()

    # Max drawdown from equity curve
    eq = np.array(equity)
    running_max = np.maximum.accumulate(eq)
    dd = running_max - eq
    max_dd_R = dd.max()
    max_dd_idx = dd.argmax()

    # Simulate $ PnL at $1000/R (representative sizing)
    dollar_per_R = 100.0  # $100 per R assumed
    total_usd = total_R * dollar_per_R
    max_dd_usd = max_dd_R * dollar_per_R

    print(f"\n{'='*60}")
    print(f"V6 HOLDOUT BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Period:           {trades_df['time'].iat[0]} to {trades_df['time'].iat[-1]}")
    print(f"Total trades:     {len(trades_df):,}")
    print(f"Winning trades:   {len(wins):,}")
    print(f"Losing trades:    {len(losses):,}")
    print(f"Win rate:         {wr:.1%}")
    print(f"Avg win (R):      +{wins['pnl_R'].mean():.2f}")
    print(f"Avg loss (R):     {losses['pnl_R'].mean():.2f}")
    print(f"Expectancy (R):   {expectancy_R:+.3f}")
    print(f"Profit factor:    {pf:.2f}")
    print(f"Total PnL (R):    {total_R:+.1f}")
    print(f"Max DD (R):       -{max_dd_R:.1f}")
    print(f"\nAt ${dollar_per_R}/R sizing:")
    print(f"  Total PnL:      ${total_usd:+,.0f}")
    print(f"  Max drawdown:   ${max_dd_usd:,.0f}")

    # Exit reason breakdown
    print(f"\nExit reasons:")
    print(trades_df["exit"].value_counts().to_string())

    # Per-cluster breakdown
    print(f"\nPer-cluster:")
    for cid, grp in trades_df.groupby("cid"):
        w = grp[grp["pnl_R"]>0]; ll = grp[grp["pnl_R"]<=0]
        ppf = w["pnl_R"].sum() / (-ll["pnl_R"].sum()) if len(ll)>0 and ll["pnl_R"].sum()<0 else float("inf")
        print(f"  C{cid}: n={len(grp):,} WR={len(w)/len(grp):.1%} PF={ppf:.2f} R={grp['pnl_R'].sum():+.1f}")

    # Plot equity curve
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(eq, linewidth=1.2)
    ax.set_title(f"V6 Holdout: PF={pf:.2f}  WR={wr:.1%}  n={len(trades_df)}  "
                 f"Total={total_R:+.1f}R  MaxDD=-{max_dd_R:.1f}R", fontsize=13)
    ax.set_xlabel("Trade #"); ax.set_ylabel("Cumulative PnL (R)")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    out_png = "/home/jay/Desktop/new-model-zigzag/experiments/v6_holdout_backtest.png"
    plt.savefig(out_png, dpi=120)
    print(f"\nEquity curve saved: {out_png}")

    # Trade log
    trades_df.to_csv("/home/jay/Desktop/new-model-zigzag/experiments/v6_holdout_trades.csv", index=False)


if __name__ == "__main__":
    main()
