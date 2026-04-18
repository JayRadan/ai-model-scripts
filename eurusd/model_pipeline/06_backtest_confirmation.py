"""
EURUSD honest holdout backtest — replays last 20% of each rule's setups
through the trained confirmation filter.
"""
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paths as P

CLUSTER_NAMES = {0: "Uptrend", 1: "MeanRevert", 2: "TrendRange", 3: "Downtrend", 4: "HighVol"}
CLUSTER_COLORS = {0: "#f5c518", 1: "#3b82f6", 2: "#00E5FF", 3: "#ef4444", 4: "#10b981"}
SPREAD_PNL = 1.5 * 0.00001 * 100000  # 1.5 pip spread in $ per 0.01 lot on EURUSD ≈ $0.15

FEATURE_COLS = [
    "f01_CPR", "f02_WickAsym", "f03_BEF", "f04_TCS", "f05_SPI",
    "f06_LRSlope", "f07_RECR", "f08_SCM", "f09_HLER", "f10_EP",
    "f11_KE", "f12_MCS", "f13_Work", "f14_EDR", "f15_AI",
    "f16_PPShigh", "f16_PPSlow", "f17_SCR", "f18_RVD", "f19_WBER",
    "f20_NCDE",
    "stoch_k", "rsi14", "bb_pct", "vol_ratio", "range_atr",
    "dist_sma20", "dist_sma50", "body_ratio", "consec_dir",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

df = pd.read_csv(P.data("setup_signals_eurusd.csv"), parse_dates=["time"])
print(f"Loaded {len(df):,} setups")

rules = sorted(df["rule"].unique())
cluster_results = {c: {"times": [], "pnls": []} for c in range(5)}
rule_stats = {}

for rule_name in rules:
    rdf = df[df["rule"] == rule_name].sort_values("time").reset_index(drop=True)
    cid = int(rdf["cluster"].iloc[0])

    meta_path = P.model(f"confirm_c{cid}_{rule_name}_meta.json")
    model_path = P.model(f"confirm_c{cid}_{rule_name}.json")

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except FileNotFoundError:
        continue

    threshold = meta["threshold"]
    if meta.get("disabled", False):
        rule_stats[rule_name] = {"n": 0, "cid": cid, "disabled": True}
        continue

    model = XGBClassifier()
    model.load_model(model_path)

    split = int(len(rdf) * 0.8)
    test = rdf.iloc[split:]
    if len(test) < 5:
        continue

    X_test = test[FEATURE_COLS].values
    probs = model.predict_proba(X_test)[:, 1]
    mask = probs >= threshold

    filtered = test[mask]
    if len(filtered) == 0:
        rule_stats[rule_name] = {"n": 0, "cid": cid, "disabled": False}
        continue

    wins = filtered["outcome"].sum()
    losses = len(filtered) - wins
    # PnL: TP=2×ATR ~20 pips avg on EURUSD → $2.00/0.01lot, SL=1×ATR → $1.00/0.01lot
    pnl_per_trade = filtered["outcome"].apply(lambda x: 2.0 if x == 1 else -1.0).values
    pnls = pnl_per_trade - SPREAD_PNL / 100  # spread cost per trade

    for t, p in zip(filtered["time"].values, pnls):
        cluster_results[cid]["times"].append(t)
        cluster_results[cid]["pnls"].append(p)

    total_pnl = pnls.sum()
    wr = wins / len(filtered) if len(filtered) > 0 else 0
    pf = (wins * 2.0) / max(losses * 1.0, 0.01)

    rule_stats[rule_name] = {
        "n": len(filtered), "wins": wins, "losses": losses,
        "wr": wr, "pf": pf, "pnl": total_pnl, "cid": cid, "disabled": False,
    }

# Print results
print("\n" + "="*80)
print("EURUSD HONEST HOLDOUT BACKTEST RESULTS")
print("="*80)

all_times = []
all_pnls = []

for cid in range(5):
    cr = cluster_results[cid]
    if not cr["times"]:
        print(f"\n  C{cid} {CLUSTER_NAMES[cid]}: 0 trades")
        continue

    idx = np.argsort(cr["times"])
    times = np.array(cr["times"])[idx]
    pnls = np.array(cr["pnls"])[idx]
    eq = np.cumsum(pnls)
    dd = eq - np.maximum.accumulate(eq)

    n = len(pnls)
    wins = (pnls > 0).sum()
    wr = wins / n
    pf = pnls[pnls > 0].sum() / max(-pnls[pnls < 0].sum(), 0.01)
    days = (times[-1] - times[0]).astype("timedelta64[D]").astype(int)
    tpd = n / max(days, 1)

    print(f"\n  C{cid} {CLUSTER_NAMES[cid]}: {n} trades  WR={wr:.0%}  PF={pf:.2f}  "
          f"PnL={eq[-1]:+.1f}  DD={dd.min():.1f}  tpd={tpd:.2f}")

    for rn, rs in sorted(rule_stats.items()):
        if rs.get("cid") == cid and rs.get("n", 0) > 0:
            print(f"    {rn:25s} {rs['n']:>5} trades  WR={rs['wr']:.0%}  PF={rs['pf']:.2f}  PnL={rs['pnl']:+.1f}")

    all_times.extend(times)
    all_pnls.extend(pnls)

# Combined
if all_times:
    idx = np.argsort(all_times)
    all_times = np.array(all_times)[idx]
    all_pnls = np.array(all_pnls)[idx]
    eq_all = np.cumsum(all_pnls)
    dd_all = eq_all - np.maximum.accumulate(eq_all)
    n_all = len(all_pnls)
    wins_all = (all_pnls > 0).sum()
    wr_all = wins_all / n_all
    pf_all = all_pnls[all_pnls > 0].sum() / max(-all_pnls[all_pnls < 0].sum(), 0.01)
    days_all = (all_times[-1] - all_times[0]).astype("timedelta64[D]").astype(int)
    tpd_all = n_all / max(days_all, 1)

    print(f"\n{'='*80}")
    print(f"COMBINED: PF {pf_all:.2f}  WR {wr_all:.0%}  n {n_all}  "
          f"PnL {eq_all[-1]:+.1f}  DD {dd_all.min():.1f}  ~{tpd_all:.1f} trades/day")

    # Save summary
    summary = {
        "n": n_all, "total_pnl": round(float(eq_all[-1]), 2),
        "max_dd": round(float(dd_all.min()), 2),
        "calendar_days": int(days_all),
        "wr": round(wr_all, 4), "pf": round(pf_all, 4),
        "trades_per_day": round(tpd_all, 2),
    }
    with open(P.data("backtest_eurusd_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plot — matching gold ZigZag style
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(20, 11), facecolor="#080c12")
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.45, wspace=0.3)

    for col_idx, cid in enumerate([0, 1, 2, 3, 4]):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.set_facecolor("#0d1117")
        cr = cluster_results[cid]
        color = CLUSTER_COLORS[cid]

        if cr["times"]:
            idx_s = np.argsort(cr["times"])
            pnls_s = np.array(cr["pnls"])[idx_s]
            eq_c = np.cumsum(pnls_s)
            ax.plot(eq_c, color=color, linewidth=1.8)
            ax.fill_between(range(len(eq_c)), eq_c, 0, alpha=0.15, color=color)

            n_c = len(pnls_s)
            wins_c = (pnls_s > 0).sum()
            wr_c = wins_c / n_c
            pf_c = pnls_s[pnls_s > 0].sum() / max(-pnls_s[pnls_s < 0].sum(), 0.01)
            dd_c = (eq_c - np.maximum.accumulate(eq_c)).min()
            ax.set_title(
                f"C{cid} {CLUSTER_NAMES[cid]}\n"
                f"PF {pf_c:.2f}  WR {wr_c:.0%}  n {n_c}\n"
                f"PnL ${eq_c[-1]:+.1f}  DD ${dd_c:.1f}",
                color=color, fontsize=10, fontfamily="monospace", pad=10,
            )
        else:
            ax.set_title(f"C{cid} {CLUSTER_NAMES[cid]}\n(no trades)",
                         color=color, fontsize=10, fontfamily="monospace", pad=10)

        ax.axhline(0, color="#444", linewidth=0.6)
        ax.set_xlabel("trade #", color="#5a7080", fontsize=8)
        ax.set_ylabel("cumulative $", color="#5a7080", fontsize=8)
        ax.tick_params(colors="#5a7080", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2a3a")

    # Combined
    ax_all = fig.add_subplot(gs[1, :])
    ax_all.set_facecolor("#0d1117")
    ax_all.plot(all_times, eq_all, color="#FFD700", linewidth=2.0)
    ax_all.fill_between(all_times, eq_all, 0, alpha=0.1, color="#FFD700")
    ax_all.axhline(0, color="#444", linewidth=0.6)
    ax_all.set_title(
        f"COMBINED — PF {pf_all:.2f}  WR {wr_all:.0%}  n {n_all}  "
        f"PnL ${eq_all[-1]:+.1f}  DD ${dd_all.min():.1f}  ~{tpd_all:.1f} trades/day",
        color="#FFD700", fontsize=12, fontfamily="monospace", pad=12, fontweight="bold",
    )
    ax_all.set_xlabel("time", color="#5a7080", fontsize=9)
    ax_all.set_ylabel("cumulative $ (combined)", color="#5a7080", fontsize=9)
    ax_all.tick_params(colors="#5a7080", labelsize=8)
    for sp in ax_all.spines.values(): sp.set_edgecolor("#1e2a3a")

    plt.suptitle(
        "EURUSD PER-RULE CONFIRMATION — HONEST HOLDOUT BACKTEST",
        color="#FFD700", fontsize=14, fontfamily="monospace",
        fontweight="bold", y=1.015,
    )
    plt.savefig(P.data("backtest_eurusd.png"), dpi=140, bbox_inches="tight", facecolor="#080c12")
    print(f"\nSaved: backtest_eurusd.png")
