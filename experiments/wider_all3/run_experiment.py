"""
Wider SL/TP grid on all 3 production models.
Proven on Samurai: wider TP → higher PF. Does it generalize?
"""
import os, json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = "/home/jay/Desktop/new-model-zigzag/experiments/wider_all3"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_FWD = 90

INSTRUMENTS = {
    "XAUUSD": {
        "raw": "/home/jay/Desktop/new-model-zigzag/data/swing_v5_xauusd.csv",
        "models": "/home/jay/Desktop/new-model-zigzag/models",
        "setups_pattern": "/home/jay/Desktop/new-model-zigzag/data/setups_{cid}.csv",
        "spread": 0.05, "meta_strict": True,
    },
    "GBPJPY": {
        "raw": "/home/jay/Desktop/new-model-zigzag/gbpjpy/data/swing_v5_gbpjpy.csv",
        "models": "/home/jay/Desktop/new-model-zigzag/gbpjpy/models",
        "setups_csv": "/home/jay/Desktop/new-model-zigzag/gbpjpy/data/setup_signals_gbpjpy.csv",
        "spread": 0.10, "meta_strict": False,
    },
    "EURUSD": {
        "raw": "/home/jay/Desktop/new-model-zigzag/eurusd/data/swing_v5_eurusd.csv",
        "models": "/home/jay/Desktop/new-model-zigzag/eurusd/models",
        "setups_csv": "/home/jay/Desktop/new-model-zigzag/eurusd/data/setup_signals_eurusd.csv",
        "spread": 0.04, "meta_strict": False,
    },
}

SL_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
TP_GRID = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0]


def load_instrument(name, cfg):
    print(f"\n── {name} ──")
    raw = pd.read_csv(cfg["raw"], parse_dates=["time"])
    raw = raw[raw["time"] >= "2016-01-01"].reset_index(drop=True)
    c = raw["close"].values.astype(np.float64)
    h = raw["high"].values.astype(np.float64)
    l = raw["low"].values.astype(np.float64)
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    t2i = dict(zip(raw["time"], range(len(raw))))

    # Load setups
    if "setups_csv" in cfg:
        setups = pd.read_csv(cfg["setups_csv"], parse_dates=["time"])
        if "cluster" in setups.columns and "cluster_id" not in setups.columns:
            setups["cluster_id"] = setups["cluster"]
    else:
        parts = []
        for cid in range(5):
            p = cfg["setups_pattern"].format(cid=cid)
            if os.path.exists(p):
                s = pd.read_csv(p, parse_dates=["time"])
                s["cluster_id"] = cid
                parts.append(s)
        setups = pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True)

    return {"c": c, "h": h, "l": l, "atr": atr, "n_bars": len(raw),
            "t2i": t2i, "setups": setups, "spread": cfg["spread"],
            "models": cfg["models"], "meta_strict": cfg["meta_strict"]}


def confirm(data):
    setups = data["setups"]
    parts = []
    for rule_name in setups["rule"].unique():
        rdf = setups[setups["rule"] == rule_name].sort_values("time").reset_index(drop=True)
        cid = int(rdf["cluster_id"].iloc[0])
        mp = f"{data['models']}/confirm_c{cid}_{rule_name}_meta.json"
        modp = f"{data['models']}/confirm_c{cid}_{rule_name}.json"
        if not os.path.exists(mp): continue
        meta = json.load(open(mp))
        if data["meta_strict"]:
            if not meta.get("passes_strict", True): continue
        else:
            if meta.get("disabled", False): continue
        thr = meta["threshold"]
        try:
            model = XGBClassifier(); model.load_model(modp)
        except Exception:
            continue
        fc = meta.get("feature_cols") or []
        avail = [c_ for c_ in fc if c_ in rdf.columns] if fc else []
        if len(avail) < 20:
            skip = {"time","rule","cluster","cluster_id","direction","outcome","label","idx"}
            cand = [c_ for c_ in rdf.columns if c_ not in skip and pd.api.types.is_numeric_dtype(rdf[c_])]
            if len(cand) >= model.n_features_in_:
                avail = cand[:model.n_features_in_]
            else:
                continue
        split = int(len(rdf) * 0.8)
        test = rdf.iloc[split:].reset_index(drop=True)
        if len(test) == 0: continue
        X = test[avail].fillna(0).values
        if X.shape[1] != model.n_features_in_: continue
        probs = model.predict_proba(X)[:, 1]
        mask = probs >= thr
        p = test[mask].copy()
        p["ml_prob"] = probs[mask]
        parts.append(p)
    if not parts: return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True)
    df["idx"] = df["time"].map(data["t2i"])
    df = df.dropna(subset=["idx"]).copy()
    df["idx"] = df["idx"].astype(int)
    return df


def simulate(df, data, tp_mult, sl_mult):
    c, h_arr, l_arr, atr, n_bars, sp = data["c"], data["h"], data["l"], data["atr"], data["n_bars"], data["spread"]
    trades = []
    for _, row in df.iterrows():
        idx = int(row["idx"])
        if idx + MAX_FWD >= n_bars: continue
        a = atr[idx]
        if a < 1e-10: continue
        entry = c[idx]
        dirv = row["direction"] if isinstance(row["direction"], (int, np.integer)) else (1 if row["direction"] in ("buy", 1) else -1)
        if dirv == 1:
            sl_p = entry - sl_mult * a; tp_p = entry + tp_mult * a
        else:
            sl_p = entry + sl_mult * a; tp_p = entry - tp_mult * a
        exit_pnl = None
        for k in range(1, MAX_FWD + 1):
            bi = idx + k
            if bi >= n_bars: break
            if dirv == 1:
                sl_hit = l_arr[bi] <= sl_p; tp_hit = h_arr[bi] >= tp_p
            else:
                sl_hit = h_arr[bi] >= sl_p; tp_hit = l_arr[bi] <= tp_p
            if sl_hit:
                exit_pnl = -sl_mult - sp; break
            if tp_hit:
                exit_pnl = tp_mult - sp; break
        if exit_pnl is None:
            if dirv == 1:
                exit_pnl = (c[idx + MAX_FWD] - entry) / a - sp
            else:
                exit_pnl = (entry - c[idx + MAX_FWD]) / a - sp
        trades.append({"time": row["time"], "pnl": exit_pnl})
    return pd.DataFrame(trades)


def stats(trades):
    if len(trades) == 0: return None
    wins = trades[trades["pnl"] > 0.01]; losses = trades[trades["pnl"] < -0.01]
    n = len(trades); wr = len(wins)/n
    pf = wins["pnl"].sum() / max(-losses["pnl"].sum(), 0.01)
    pnl = trades["pnl"].sum()
    eq = np.cumsum(trades.sort_values("time")["pnl"].values)
    dd = (eq - np.maximum.accumulate(eq)).min()
    return {"n": n, "wr": wr, "pf": pf, "pnl": pnl, "dd": dd, "eq": eq}


all_results = {}
for inst, cfg in INSTRUMENTS.items():
    data = load_instrument(inst, cfg)
    confirmed = confirm(data)
    if len(confirmed) < 50:
        print(f"  {inst}: only {len(confirmed)} confirmed trades — skipping")
        continue
    print(f"  Confirmed holdout: {len(confirmed)}")

    print(f"\n  {'SL':>5} {'TP':>5} {'R:R':>5} {'n':>6} {'WR':>6} {'PF':>6} {'PnL':>9} {'DD':>8}")
    print("  " + "-"*65)

    inst_res = []
    for sl in SL_GRID:
        for tp in TP_GRID:
            if tp <= sl: continue
            trades = simulate(confirmed, data, tp, sl)
            s = stats(trades)
            if not s: continue
            s["sl"] = sl; s["tp"] = tp; s["rr"] = tp/sl
            inst_res.append(s)
            mark = "★" if s["pf"] >= 1.5 else ""
            print(f"  {sl:>5.1f} {tp:>5.1f} {s['rr']:>5.1f} {s['n']:>6} {s['wr']:>6.0%} {s['pf']:>6.2f} {s['pnl']:>+9.1f} {s['dd']:>+8.1f} {mark}")

    # Baseline reference
    baseline = next((r for r in inst_res if r["sl"]==1.0 and r["tp"]==2.0), None)
    print(f"\n  Baseline SL=1 TP=2: WR={baseline['wr']:.0%} PF={baseline['pf']:.2f} PnL={baseline['pnl']:+.0f}")
    top_pnl = sorted(inst_res, key=lambda r: -r["pnl"])[:3]
    top_pf = sorted(inst_res, key=lambda r: -r["pf"])[:3]
    print(f"\n  Top 3 by PnL:")
    for r in top_pnl:
        print(f"    SL={r['sl']} TP={r['tp']}  WR={r['wr']:.0%}  PF={r['pf']:.2f}  PnL={r['pnl']:+.0f}  DD={r['dd']:+.0f}")
    print(f"  Top 3 by PF:")
    for r in top_pf:
        print(f"    SL={r['sl']} TP={r['tp']}  WR={r['wr']:.0%}  PF={r['pf']:.2f}  PnL={r['pnl']:+.0f}  DD={r['dd']:+.0f}")
    all_results[inst] = {"results": inst_res, "baseline": baseline}


# ── Plot: 3-row stack (one row per instrument), 2 heatmaps each ──
fig = plt.figure(figsize=(22, 18), facecolor="#080c12")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.25)

for row_idx, (inst, info) in enumerate(all_results.items()):
    results = info["results"]
    pf_grid = np.full((len(SL_GRID), len(TP_GRID)), np.nan)
    pnl_grid = np.full((len(SL_GRID), len(TP_GRID)), np.nan)
    for r in results:
        i = SL_GRID.index(r["sl"]); j = TP_GRID.index(r["tp"])
        pf_grid[i, j] = r["pf"]; pnl_grid[i, j] = r["pnl"]

    ax1 = fig.add_subplot(gs[row_idx, 0])
    ax1.set_facecolor("#0d1117")
    im1 = ax1.imshow(pf_grid, cmap="RdYlGn", aspect="auto", vmin=0.9, vmax=1.8)
    ax1.set_xticks(range(len(TP_GRID))); ax1.set_xticklabels([f"{t:.1f}" for t in TP_GRID])
    ax1.set_yticks(range(len(SL_GRID))); ax1.set_yticklabels([f"{s:.1f}" for s in SL_GRID])
    ax1.set_xlabel("TP (ATR)", color="#ccc")
    ax1.set_ylabel("SL (ATR)", color="#ccc")
    ax1.set_title(f"{inst} — Profit Factor (baseline PF={info['baseline']['pf']:.2f})", color="#FFD700", fontsize=12)
    for i in range(len(SL_GRID)):
        for j in range(len(TP_GRID)):
            if not np.isnan(pf_grid[i, j]):
                ax1.text(j, i, f"{pf_grid[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    plt.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(gs[row_idx, 1])
    ax2.set_facecolor("#0d1117")
    im2 = ax2.imshow(pnl_grid, cmap="RdYlGn", aspect="auto")
    ax2.set_xticks(range(len(TP_GRID))); ax2.set_xticklabels([f"{t:.1f}" for t in TP_GRID])
    ax2.set_yticks(range(len(SL_GRID))); ax2.set_yticklabels([f"{s:.1f}" for s in SL_GRID])
    ax2.set_xlabel("TP (ATR)", color="#ccc")
    ax2.set_ylabel("SL (ATR)", color="#ccc")
    ax2.set_title(f"{inst} — PnL (baseline {info['baseline']['pnl']:+.0f})", color="#FFD700", fontsize=12)
    for i in range(len(SL_GRID)):
        for j in range(len(TP_GRID)):
            if not np.isnan(pnl_grid[i, j]):
                ax2.text(j, i, f"{pnl_grid[i, j]:+.0f}", ha="center", va="center", color="black", fontsize=8)
    plt.colorbar(im2, ax=ax2)

plt.suptitle("All 3 Models — Wider SL/TP Grid",
             color="#FFD700", fontsize=18, fontweight="bold", y=0.995)
out = os.path.join(OUT_DIR, "wider_all3.png")
plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#080c12")
print(f"\nSaved: {out}")
