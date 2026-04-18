"""
Honest holdout backtest for the per-rule confirmation system.

Loads each rule's classifier + threshold, replays the last 20% of that rule's
setup pool, and simulates the trades. Combines across all rules chronologically
for the equity curve.

Generates:
  backtest_confirmation.png
  backtest_confirmation_summary.json
"""
from __future__ import annotations
import glob
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CLUSTER_NAMES = {0:"Ranging", 1:"Downtrend", 3:"Uptrend"}
COLORS = {0:"#FFD700", 1:"#00E5FF", 3:"#69FF94"}

TP_MULT = 2.0
SL_MULT = 1.0
SPREAD  = 0.40


def list_rule_models(cid: int):
    """Return sorted list of rule names that have models for this cluster."""
    pattern = f"models/confirm_c{cid}_*.json"
    names = []
    for p in glob.glob(pattern):
        base = os.path.basename(p)
        if base.endswith("_meta.json"):
            continue
        stem = base[:-len(".json")]
        prefix = f"confirm_c{cid}_"
        if stem.startswith(prefix):
            names.append(stem[len(prefix):])
    return sorted(names)


def run_rule(cid: int, rule_name: str, df_all: pd.DataFrame):
    """Replay holdout for one rule; return list of per-trade dicts."""
    meta_path = f"models/confirm_c{cid}_{rule_name}_meta.json"
    model_path = f"models/confirm_c{cid}_{rule_name}.json"
    if not os.path.exists(meta_path) or not os.path.exists(model_path):
        return []

    with open(meta_path) as f:
        meta = json.load(f)
    thr = meta["threshold"]
    feat_cols = meta["feature_cols"]

    sub = (df_all[df_all["rule"] == rule_name]
           .sort_values("time").reset_index(drop=True))
    if len(sub) == 0:
        return []
    cutoff = int(len(sub) * 0.80)
    ho = sub.iloc[cutoff:].reset_index(drop=True)
    if len(ho) == 0:
        return []

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    X = ho[feat_cols].fillna(0).values
    proba = model.predict_proba(X)[:, 1]

    mask = proba >= thr
    selected = ho[mask].copy().reset_index(drop=True)
    if len(selected) == 0:
        return []

    trades = []
    for _, row in selected.iterrows():
        atr = float(row["atr"])
        if int(row["label"]) == 1:
            pnl = TP_MULT * atr - SPREAD
        else:
            pnl = -(SL_MULT * atr + SPREAD)
        trades.append({
            "time": pd.Timestamp(row["time"]),
            "cid": cid,
            "rule": rule_name,
            "direction": int(row["direction"]),
            "pnl": float(pnl),
        })
    return trades


def run_cluster(cid: int):
    name = CLUSTER_NAMES[cid]
    df = pd.read_csv(f"setups_{cid}.csv", parse_dates=["time"])
    rule_names = list_rule_models(cid)

    all_trades = []
    per_rule = {}
    for r in rule_names:
        tr = run_rule(cid, r, df)
        per_rule[r] = tr
        all_trades.extend(tr)

    all_trades.sort(key=lambda t: t["time"])
    pnls = np.array([t["pnl"] for t in all_trades], dtype=float)
    equity = np.cumsum(pnls)

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    wr = float((pnls > 0).mean()) if len(pnls) > 0 else 0.0
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 0.0
    if len(equity) > 0:
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        max_dd = float(dd.min())
    else:
        max_dd = 0.0
    span_days = ((all_trades[-1]["time"] - all_trades[0]["time"]).days
                 if len(all_trades) > 1 else 1)
    tpd = len(all_trades) / max(span_days, 1)

    return {
        "cid": cid, "name": name,
        "rules": rule_names,
        "n": int(len(all_trades)),
        "winrate": wr,
        "pf": pf,
        "pnl_total": float(pnls.sum()) if len(pnls) > 0 else 0.0,
        "max_dd": max_dd,
        "trades_per_day": tpd,
        "trades": all_trades,
        "equity": equity.tolist(),
        "per_rule_counts": {r: len(v) for r, v in per_rule.items()},
    }


def plot_results(results):
    fig = plt.figure(figsize=(18, 11), facecolor="#080c12")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3)

    for col_idx, cid in enumerate([0, 1, 3]):
        r = results[cid]
        color = COLORS[cid]
        ax = fig.add_subplot(gs[0, col_idx])
        ax.set_facecolor("#0d1117")

        if r["n"] > 0:
            ax.plot(r["equity"], color=color, linewidth=1.8)
            ax.fill_between(range(len(r["equity"])), r["equity"], 0,
                            alpha=0.15, color=color)
        ax.axhline(0, color="#444", linewidth=0.6)

        mark = "OK" if r["pf"] >= 1.5 else "~"
        ax.set_title(
            f"{mark} C{cid} {r['name']}\n"
            f"PF {r['pf']:.2f}  WR {r['winrate']:.0%}  n {r['n']}\n"
            f"PnL ${r['pnl_total']:+.1f}  DD ${r['max_dd']:.1f}",
            color=color, fontsize=10, fontfamily="monospace", pad=10,
        )
        ax.set_xlabel("trade #", color="#5a7080", fontsize=8)
        ax.set_ylabel("cumulative $", color="#5a7080", fontsize=8)
        ax.tick_params(colors="#5a7080", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2a3a")

    # Combined
    ax_all = fig.add_subplot(gs[1, :])
    ax_all.set_facecolor("#0d1117")
    all_trades = []
    for cid in [0, 1, 3]:
        all_trades.extend(results[cid]["trades"])
    all_trades.sort(key=lambda t: t["time"])
    if all_trades:
        combined = pd.DataFrame(all_trades)
        combined["equity"] = combined["pnl"].cumsum()
        ax_all.plot(combined["time"], combined["equity"],
                    color="#FFD700", linewidth=2.0)
        ax_all.fill_between(combined["time"], combined["equity"], 0,
                            alpha=0.1, color="#FFD700")
        ax_all.axhline(0, color="#444", linewidth=0.6)

        pnl = combined["pnl"].to_numpy()
        total = float(pnl.sum())
        n = len(pnl)
        wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
        wr = float((pnl > 0).mean())
        pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 0.0
        peak = np.maximum.accumulate(combined["equity"].to_numpy())
        dd = combined["equity"].to_numpy() - peak
        max_dd = float(dd.min())
        span_days = (combined["time"].iloc[-1] - combined["time"].iloc[0]).days
        tpd = n / max(span_days, 1)

        ax_all.set_title(
            f"COMBINED — PF {pf:.2f}  WR {wr:.0%}  n {n}  "
            f"PnL ${total:+.1f}  DD ${max_dd:.1f}  ~{tpd:.2f} trades/day",
            color="#FFD700", fontsize=12, fontfamily="monospace", pad=12,
            fontweight="bold",
        )
    ax_all.set_xlabel("time", color="#5a7080", fontsize=9)
    ax_all.set_ylabel("cumulative $ (combined)", color="#5a7080", fontsize=9)
    ax_all.tick_params(colors="#5a7080", labelsize=8)
    for sp in ax_all.spines.values(): sp.set_edgecolor("#1e2a3a")

    plt.suptitle(
        "PER-RULE CONFIRMATION — HONEST HOLDOUT BACKTEST",
        color="#FFD700", fontsize=14, fontfamily="monospace",
        fontweight="bold", y=1.015,
    )
    out = "backtest_confirmation.png"
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#080c12")
    print(f"\nSaved: {out}")


def main():
    results = {cid: run_cluster(cid) for cid in [0, 1, 3]}

    print(f"\n{'═'*66}\nHONEST HOLDOUT BACKTEST — per-rule confirmation\n{'═'*66}")
    for cid in [0, 1, 3]:
        r = results[cid]
        print(f"\n  C{cid} {r['name']}: {r['n']} trades  "
              f"WR={r['winrate']:.0%}  PF={r['pf']:.2f}  "
              f"PnL={r['pnl_total']:+.1f}  DD={r['max_dd']:.1f}  "
              f"tpd={r['trades_per_day']:.2f}")
        for rule, cnt in r["per_rule_counts"].items():
            print(f"    {rule:<22} {cnt}")

    plot_results(results)

    summary = {}
    for cid, r in results.items():
        summary[str(cid)] = {
            k: v for k, v in r.items()
            if k not in ("trades", "equity")
        }
    with open("backtest_confirmation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("Saved: backtest_confirmation_summary.json")


if __name__ == "__main__":
    main()
