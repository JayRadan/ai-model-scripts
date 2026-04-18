"""
GBPJPY backtest with PRUNED rules — only the 11 surviving rules.
Compares before vs after pruning side by side.
"""
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import paths as P

CLUSTER_NAMES = {0: "Uptrend", 1: "Ranging", 2: "Downtrend", 3: "HighVol"}
CLUSTER_COLORS = {0: "#f5c518", 1: "#3b82f6", 2: "#ef4444", 3: "#10b981"}
SPREAD_PNL = 0.20

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

# Rules that SURVIVE the quality gate (AUC>=0.52, hPF>=1.25, hN>=20)
KEEP_RULES = {
    "R0a_pullback", "R0c_breakout_pb", "R0g_close_streak",
    "R0j_ema_pullback", "R0k_tokyo_buy",
    "R1b_stoch", "R1f_mean_revert", "R1g_close_extreme", "R1n_range_fade",
    "R3a_v_reversal", "R3d_stoch_vol",
}

df = pd.read_csv(P.data("setup_signals_gbpjpy.csv"), parse_dates=["time"])
print(f"Loaded {len(df):,} setups")

rules = sorted(df["rule"].unique())


def run_backtest(keep_only=None, label="ALL"):
    """Run backtest, optionally filtering to keep_only set of rules."""
    cluster_results = {c: {"times": [], "pnls": []} for c in range(4)}
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

        # Disabled by original pipeline
        if meta.get("disabled", False):
            rule_stats[rule_name] = {"n": 0, "cid": cid, "disabled": True}
            continue

        # Disabled by pruning
        if keep_only is not None and rule_name not in keep_only:
            rule_stats[rule_name] = {"n": 0, "cid": cid, "disabled": True, "pruned": True}
            continue

        threshold = meta["threshold"]
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
        pnl_per_trade = filtered["outcome"].apply(lambda x: 2.0 if x == 1 else -1.0).values
        pnls = pnl_per_trade - SPREAD_PNL / 100

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

    # Combine
    all_times = []
    all_pnls = []

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    for cid in range(4):
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

        print(f"\n  C{cid} {CLUSTER_NAMES[cid]}: {n} trades  WR={wr:.0%}  PF={pf:.2f}  "
              f"PnL={eq[-1]:+.1f}  DD={dd.min():.1f}")

        for rn, rs in sorted(rule_stats.items()):
            if rs.get("cid") == cid and rs.get("n", 0) > 0:
                print(f"    {rn:25s} {rs['n']:>5} trades  WR={rs['wr']:.0%}  PF={rs['pf']:.2f}  PnL={rs['pnl']:+.1f}")

        all_times.extend(times)
        all_pnls.extend(pnls)

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

        print(f"\n  COMBINED: PF {pf_all:.2f}  WR {wr_all:.0%}  n {n_all}  "
              f"PnL {eq_all[-1]:+.1f}  DD {dd_all.min():.1f}  ~{tpd_all:.1f} trades/day")

        return {
            "all_times": all_times, "all_pnls": all_pnls,
            "eq": eq_all, "dd": dd_all,
            "cluster_results": cluster_results,
            "n": n_all, "pf": pf_all, "wr": wr_all, "tpd": tpd_all,
            "rule_stats": rule_stats,
        }
    return None


# Run both
print("\n" + "#"*80)
print("# BEFORE vs AFTER PRUNING")
print("#"*80)

before = run_backtest(keep_only=None, label="BEFORE PRUNING (all 45 active rules)")
after = run_backtest(keep_only=KEEP_RULES, label="AFTER PRUNING (11 quality rules)")

# Comparison chart
if before and after:
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), facecolor="#080c12")

    for ax, data, title, color in [
        (axes[0], before, "BEFORE PRUNING", "#ef4444"),
        (axes[1], after, "AFTER PRUNING", "#10b981"),
    ]:
        ax.set_facecolor("#0d1117")
        ax.plot(data["all_times"], data["eq"], color=color, linewidth=2.0)
        ax.fill_between(data["all_times"], data["eq"], 0, alpha=0.12, color=color)
        ax.axhline(0, color="#444", linewidth=0.6)
        ax.set_title(
            f"{title}\n"
            f"PF {data['pf']:.2f}  WR {data['wr']:.0%}  n {data['n']}  "
            f"PnL ${data['eq'][-1]:+.1f}  DD ${data['dd'].min():.1f}  ~{data['tpd']:.1f} tpd",
            color=color, fontsize=12, fontfamily="monospace", pad=12,
        )
        ax.set_xlabel("time", color="#5a7080", fontsize=9)
        ax.set_ylabel("cumulative $", color="#5a7080", fontsize=9)
        ax.tick_params(colors="#5a7080", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e2a3a")

    plt.suptitle(
        "GBPJPY SAMURAI — BEFORE vs AFTER RULE PRUNING",
        color="#FFD700", fontsize=14, fontfamily="monospace",
        fontweight="bold", y=1.02,
    )
    out = P.data("backtest_gbpjpy_pruned_comparison.png")
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#080c12")
    print(f"\nSaved: {out}")
