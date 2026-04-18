"""
Samurai (GBPJPY) — bigger SL/TP grid to give trades more room.

Baseline: SL=1R TP=2R → WR 44%, PF 1.35
Test wider stops and wider targets on the same ML-confirmed holdout trades.
"""
import os, json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = "/home/jay/Desktop/new-model-zigzag/experiments/samurai_wider"
os.makedirs(OUT_DIR, exist_ok=True)

RAW_CSV = "/home/jay/Desktop/new-model-zigzag/gbpjpy/data/swing_v5_gbpjpy.csv"
MODELS_DIR = "/home/jay/Desktop/new-model-zigzag/gbpjpy/models"
SETUPS_CSV = "/home/jay/Desktop/new-model-zigzag/gbpjpy/data/setup_signals_gbpjpy.csv"
SPREAD = 0.10
MAX_FWD = 90  # extend forward window since TP can be far

print("Loading GBPJPY M5...")
raw = pd.read_csv(RAW_CSV, parse_dates=["time"])
raw = raw[raw["time"] >= "2016-01-01"].reset_index(drop=True)
c = raw["close"].values.astype(np.float64)
h = raw["high"].values.astype(np.float64)
l = raw["low"].values.astype(np.float64)
prev_c = np.roll(c, 1); prev_c[0] = c[0]
tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
atr14 = pd.Series(tr).rolling(14, min_periods=1).mean().values
n_bars = len(raw)
t2i = dict(zip(raw["time"], range(n_bars)))
print(f"  {n_bars:,} bars")

# ── Load setups + apply ML confirmation (holdout only) ──
print("\nLoading + applying ML confirmation (holdout 20%)...")
setups = pd.read_csv(SETUPS_CSV, parse_dates=["time"])
if "cluster" in setups.columns and "cluster_id" not in setups.columns:
    setups["cluster_id"] = setups["cluster"]

confirmed_parts = []
for rule_name in setups["rule"].unique():
    rdf = setups[setups["rule"] == rule_name].sort_values("time").reset_index(drop=True)
    cid = int(rdf["cluster_id"].iloc[0])
    meta_path = f"{MODELS_DIR}/confirm_c{cid}_{rule_name}_meta.json"
    model_path = f"{MODELS_DIR}/confirm_c{cid}_{rule_name}.json"
    if not os.path.exists(meta_path): continue
    meta = json.load(open(meta_path))
    if meta.get("disabled", False): continue
    thr = meta["threshold"]
    try:
        model = XGBClassifier(); model.load_model(model_path)
    except Exception:
        continue
    # GBPJPY meta lacks feature_cols — derive from model.feature_names_in_ / n_features_in_
    fc = meta.get("feature_cols") or []
    avail = [c_ for c_ in fc if c_ in rdf.columns] if fc else []
    if len(avail) < 20:
        # Fallback: use all numeric non-meta columns (exclude time/rule/cluster/direction/outcome/label)
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
    confirmed_parts.append(p)

confirmed = pd.concat(confirmed_parts, ignore_index=True).sort_values("time").reset_index(drop=True)
print(f"  Confirmed holdout trades: {len(confirmed):,}")

# Map idx
confirmed["idx"] = confirmed["time"].map(t2i)
confirmed = confirmed.dropna(subset=["idx"]).copy()
confirmed["idx"] = confirmed["idx"].astype(int)

# ── Simulate with configurable SL/TP ──
def simulate(df, tp_mult, sl_mult, max_fwd=MAX_FWD):
    trades = []
    for _, row in df.iterrows():
        idx = int(row["idx"])
        if idx + max_fwd >= n_bars: continue
        a = atr14[idx]
        if a < 1e-10: continue
        entry = c[idx]
        dirv = row["direction"] if isinstance(row["direction"], (int, np.integer)) else (1 if row["direction"] in ("buy", 1) else -1)
        if dirv == 1:
            sl_p = entry - sl_mult * a; tp_p = entry + tp_mult * a
        else:
            sl_p = entry + sl_mult * a; tp_p = entry - tp_mult * a
        exit_pnl = None
        for k in range(1, max_fwd + 1):
            bi = idx + k
            if bi >= n_bars: break
            if dirv == 1:
                sl_hit = l[bi] <= sl_p; tp_hit = h[bi] >= tp_p
            else:
                sl_hit = h[bi] >= sl_p; tp_hit = l[bi] <= tp_p
            if sl_hit:
                exit_pnl = -sl_mult - SPREAD; break
            if tp_hit:
                exit_pnl = tp_mult - SPREAD; break
        if exit_pnl is None:
            if dirv == 1:
                exit_pnl = (c[idx + max_fwd] - entry) / a - SPREAD
            else:
                exit_pnl = (entry - c[idx + max_fwd]) / a - SPREAD
        trades.append({"time": row["time"], "pnl": exit_pnl, "cluster": int(row["cluster_id"])})
    return pd.DataFrame(trades)


def stats(trades):
    if len(trades) == 0: return None
    wins = trades[trades["pnl"] > 0.01]
    losses = trades[trades["pnl"] < -0.01]
    n = len(trades); wr = len(wins) / n
    pf = wins["pnl"].sum() / max(-losses["pnl"].sum(), 0.01)
    pnl = trades["pnl"].sum()
    eq = np.cumsum(trades.sort_values("time")["pnl"].values)
    dd = (eq - np.maximum.accumulate(eq)).min() if len(eq) > 0 else 0
    return {"n": n, "wr": wr, "pf": pf, "pnl": pnl, "dd": dd, "eq": eq}


# ── Grid sweep ──
print("\n" + "="*80)
print("SL/TP GRID — Samurai holdout (bigger stops, bigger targets)")
print("="*80)

SL_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
TP_GRID = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0]

results = []
print(f"\n  {'SL':>5} {'TP':>5} {'R:R':>5} {'n':>6} {'WR':>6} {'PF':>6} {'PnL':>9} {'DD':>8}")
print("  " + "-"*65)
for sl in SL_GRID:
    for tp in TP_GRID:
        if tp <= sl: continue  # require positive R:R
        trades = simulate(confirmed, tp_mult=tp, sl_mult=sl)
        s = stats(trades)
        if not s: continue
        s["sl"] = sl; s["tp"] = tp; s["rr"] = tp / sl
        results.append(s)
        best = "★" if s["pf"] > 1.5 and s["pnl"] > 500 else ""
        print(f"  {sl:>5.1f} {tp:>5.1f} {s['rr']:>5.1f} {s['n']:>6} {s['wr']:>6.0%} {s['pf']:>6.2f} {s['pnl']:>+9.1f} {s['dd']:>+8.1f} {best}")


# Highlight best
results.sort(key=lambda r: r["pnl"], reverse=True)
print(f"\n  Top 5 by PnL:")
for r in results[:5]:
    print(f"    SL={r['sl']} TP={r['tp']}  n={r['n']}  WR={r['wr']:.0%}  PF={r['pf']:.2f}  PnL={r['pnl']:+.1f}  DD={r['dd']:+.1f}")

best_pf = sorted(results, key=lambda r: -r["pf"])[:5]
print(f"\n  Top 5 by PF:")
for r in best_pf:
    print(f"    SL={r['sl']} TP={r['tp']}  n={r['n']}  WR={r['wr']:.0%}  PF={r['pf']:.2f}  PnL={r['pnl']:+.1f}  DD={r['dd']:+.1f}")


# ── Plot ──
fig = plt.figure(figsize=(20, 12), facecolor="#080c12")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

# Heatmap: PF over SL×TP grid
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#0d1117")
pf_grid = np.full((len(SL_GRID), len(TP_GRID)), np.nan)
pnl_grid = np.full((len(SL_GRID), len(TP_GRID)), np.nan)
for r in results:
    i = SL_GRID.index(r["sl"]); j = TP_GRID.index(r["tp"])
    pf_grid[i, j] = r["pf"]; pnl_grid[i, j] = r["pnl"]
im1 = ax1.imshow(pf_grid, cmap="RdYlGn", aspect="auto", vmin=0.8, vmax=1.7)
ax1.set_xticks(range(len(TP_GRID))); ax1.set_xticklabels([f"{t:.1f}" for t in TP_GRID])
ax1.set_yticks(range(len(SL_GRID))); ax1.set_yticklabels([f"{s:.1f}" for s in SL_GRID])
ax1.set_xlabel("TP (ATR)", color="#ccc")
ax1.set_ylabel("SL (ATR)", color="#ccc")
ax1.set_title("Profit Factor — SL × TP", color="#FFD700", fontsize=12)
for i in range(len(SL_GRID)):
    for j in range(len(TP_GRID)):
        if not np.isnan(pf_grid[i,j]):
            ax1.text(j, i, f"{pf_grid[i,j]:.2f}", ha="center", va="center", color="black", fontsize=9)
plt.colorbar(im1, ax=ax1)

# Heatmap: PnL
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#0d1117")
im2 = ax2.imshow(pnl_grid, cmap="RdYlGn", aspect="auto")
ax2.set_xticks(range(len(TP_GRID))); ax2.set_xticklabels([f"{t:.1f}" for t in TP_GRID])
ax2.set_yticks(range(len(SL_GRID))); ax2.set_yticklabels([f"{s:.1f}" for s in SL_GRID])
ax2.set_xlabel("TP (ATR)", color="#ccc")
ax2.set_ylabel("SL (ATR)", color="#ccc")
ax2.set_title("PnL — SL × TP", color="#FFD700", fontsize=12)
for i in range(len(SL_GRID)):
    for j in range(len(TP_GRID)):
        if not np.isnan(pnl_grid[i,j]):
            ax2.text(j, i, f"{pnl_grid[i,j]:+.0f}", ha="center", va="center", color="black", fontsize=8)
plt.colorbar(im2, ax=ax2)

# Equity curves — top 5 by PnL
ax3 = fig.add_subplot(gs[1, :])
ax3.set_facecolor("#0d1117")
colors = ["#FFD700", "#10b981", "#00E5FF", "#3b82f6", "#a855f7", "#ef4444", "#fd79a8"]
top = sorted(results, key=lambda r: -r["pnl"])[:6]
# include baseline SL=1 TP=2 for reference
baseline = next((r for r in results if r["sl"]==1.0 and r["tp"]==2.0), None)
if baseline and baseline not in top:
    top.append(baseline)
for i, r in enumerate(top):
    label = f"SL={r['sl']} TP={r['tp']}  WR={r['wr']:.0%}  PF={r['pf']:.2f}  PnL={r['pnl']:+.0f}"
    if r["sl"]==1.0 and r["tp"]==2.0: label += "  (baseline)"
    ax3.plot(r["eq"], color=colors[i % len(colors)], linewidth=1.4, alpha=0.9, label=label)
ax3.axhline(0, color="#444", linewidth=0.6)
ax3.legend(fontsize=9, loc="upper left")
ax3.set_title("Equity Curves — Top configs + Baseline", color="#FFD700", fontsize=12)
ax3.tick_params(colors="#5a7080")
for sp in ax3.spines.values(): sp.set_edgecolor("#1e2a3a")

plt.suptitle("Samurai (GBPJPY) — Bigger SL/TP Grid",
             color="#FFD700", fontsize=16, fontweight="bold", y=1.0)
out = os.path.join(OUT_DIR, "samurai_wider.png")
plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#080c12")
print(f"\nSaved: {out}")
