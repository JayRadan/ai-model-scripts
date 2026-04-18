"""
Regime v4 — HONEST vol-only classifier.

After 3 failed attempts to predict direction (midas_v2, regime_v2, regime_v3) +
nearest-neighbor test (vol corr 0.80, direction acc 54% < always-up 56%), we
accept that fingerprint features predict ONLY volatility.

Design:
  3-state regime: Quiet / Normal / HighVol (by forward vol quantile)
  Trained as 3-class XGBoost on same 7 fingerprint features
  Runs AS AN OVERLAY on existing K=5 — does NOT replace it.

Usage in live EA:
  Quiet   → SKIP trade (low expected movement, TP unlikely to hit)
  Normal  → trade as usual
  HighVol → trade but widen SL by 1.5× (optional) OR allow only high-conviction

Deliverable:
  1. Classifier trained + AUC validated
  2. Backtest overlay: do the production confirmed trades perform better
     when we filter out Quiet-regime entries?
  3. Export to ONNX for MQL5 ingestion later
"""
import os, json, time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = "/home/jay/Desktop/new-model-zigzag/experiments/regime_v4_vol_only"
os.makedirs(OUT_DIR, exist_ok=True)

RAW_CSV = "/home/jay/Desktop/new-model-zigzag/data/swing_v5_xauusd.csv"
WINDOW = 288
STEP = 12

NAMES = {0: "Quiet", 1: "Normal", 2: "HighVol"}
COLORS = {0: "#3b82f6", 1: "#10b981", 2: "#ef4444"}

print("Loading XAUUSD M5...")
raw = pd.read_csv(RAW_CSV, parse_dates=["time"])
raw = raw[raw["time"] >= "2016-01-01"].sort_values("time").reset_index(drop=True)
c = raw["close"].values.astype(np.float64)
h = raw["high"].values.astype(np.float64)
l = raw["low"].values.astype(np.float64)
o = raw["open"].values.astype(np.float64)
n_bars = len(raw)
t2i = dict(zip(raw["time"], range(n_bars)))
print(f"  {n_bars:,} bars")


def fingerprint(c_, h_, l_, o_):
    n = len(c_)
    if n < 10: return None
    r = np.diff(c_) / c_[:-1]
    br = (h_ - l_) / c_
    fp = np.zeros(7)
    fp[0] = r.sum(); fp[1] = r.std()
    mean_r = r.mean()
    fp[2] = np.mean(np.sign(r) == np.sign(mean_r)) if abs(mean_r) > 1e-12 else 0.5
    fp[3] = r.sum() / (r.std() + 1e-9)
    fp[4] = br.mean()
    total_range = (h_.max() - l_.min()) / c_.mean()
    fp[5] = total_range / (br.mean() + 1e-9)
    if len(r) > 2:
        d = r[:-1].std() * r[1:].std()
        fp[6] = np.corrcoef(r[:-1], r[1:])[0, 1] if d > 1e-12 else 0.0
    return fp


# ── Compute fingerprints + forward vol ──
print(f"\nComputing fingerprints + forward vol (step={STEP})...")
t0 = time.time()
records = []
for idx in range(WINDOW, n_bars - WINDOW, STEP):
    fp = fingerprint(c[idx-WINDOW:idx], h[idx-WINDOW:idx], l[idx-WINDOW:idx], o[idx-WINDOW:idx])
    if fp is None: continue
    fwd_r = np.diff(c[idx:idx+WINDOW]) / c[idx:idx+WINDOW-1]
    fwd_vol = fwd_r.std() * 100
    fwd_ret = (c[idx+WINDOW-1] - c[idx]) / c[idx] * 100
    records.append({
        "time": raw["time"].iloc[idx], "idx": idx,
        "fp0": fp[0], "fp1": fp[1], "fp2": fp[2], "fp3": fp[3],
        "fp4": fp[4], "fp5": fp[5], "fp6": fp[6],
        "fwd_vol": fwd_vol, "fwd_ret": fwd_ret,
    })
df = pd.DataFrame(records).sort_values("time").reset_index(drop=True)
print(f"  {len(df):,} points in {time.time()-t0:.1f}s")


# ── Vol tercile labels ──
q25 = df["fwd_vol"].quantile(0.25)
q75 = df["fwd_vol"].quantile(0.75)
labels = np.ones(len(df), dtype=np.int8)
labels[df["fwd_vol"] <= q25] = 0  # Quiet
labels[df["fwd_vol"] > q75] = 2   # HighVol
df["label"] = labels
print(f"\nVol thresholds: Quiet <= {q25:.4f}%  Normal ({q25:.4f}, {q75:.4f}]  HighVol > {q75:.4f}%")
for cls in range(3):
    sub = df[df["label"] == cls]
    print(f"  {cls} {NAMES[cls]:<8}: {len(sub)/len(df)*100:>5.1f}%  fwd_vol={sub['fwd_vol'].mean():.4f}%")


# ── Train ──
print("\n" + "="*70)
print("TRAINING XGBoost 3-class vol classifier")
print("="*70)

feat_cols = [f"fp{i}" for i in range(7)]
s1, s2 = int(len(df)*0.70), int(len(df)*0.85)
X_tr, y_tr = df.iloc[:s1][feat_cols].values, df.iloc[:s1]["label"].values
X_vl, y_vl = df.iloc[s1:s2][feat_cols].values, df.iloc[s1:s2]["label"].values
X_te, y_te = df.iloc[s2:][feat_cols].values, df.iloc[s2:]["label"].values
print(f"  train={len(X_tr)}  val={len(X_vl)}  test={len(X_te)}")

model = XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.04,
    objective="multi:softprob", num_class=3,
    subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=40, verbosity=0
)
model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

probs = model.predict_proba(X_te)
preds = probs.argmax(axis=1)
print(f"\n  Test accuracy: {(preds == y_te).mean():.3f}")
print(f"\n  Per-class AUC:")
aucs = []
for cls in range(3):
    y_bin = (y_te == cls).astype(int)
    a = roc_auc_score(y_bin, probs[:, cls])
    aucs.append(a)
    print(f"    {cls} {NAMES[cls]:<8}: AUC={a:.3f}  base={y_bin.mean():.0%}")

cm = confusion_matrix(y_te, preds, labels=list(range(3)))
print(f"\n  Confusion matrix (rows=true, cols=pred):")
print(f"  {'':<10}" + " ".join(f"{NAMES[i]:>9}" for i in range(3)))
for i in range(3):
    print(f"  {NAMES[i]:<10}" + " ".join(f"{cm[i][j]:>9}" for j in range(3)))

# Forward-vol validation
test_df = df.iloc[s2:].copy().reset_index(drop=True)
test_df["pred"] = preds
print(f"\n  Forward-vol per predicted class:")
for cls in range(3):
    sub = test_df[test_df["pred"] == cls]
    if len(sub) == 0: continue
    print(f"    {NAMES[cls]:<8}: n={len(sub):>5}  fwd_vol={sub['fwd_vol'].mean():.4f}%  (expected: "
          f"{'low' if cls==0 else 'mid' if cls==1 else 'high'})")


# ── BACKTEST OVERLAY: do production trades perform better when filtering out Quiet? ──
print("\n" + "="*70)
print("BACKTEST OVERLAY on production ML-confirmed trades")
print("="*70)

# Load production confirmed trades (holdout) from samurai_wider / wider_all3
# Instead, apply same logic: confirm trades with production models, join with vol regime, split by regime.
models_dir = "/home/jay/Desktop/new-model-zigzag/models"

setups_parts = []
for cid in range(5):
    s = pd.read_csv(f"/home/jay/Desktop/new-model-zigzag/data/setups_{cid}.csv", parse_dates=["time"])
    s["cluster_id"] = cid
    setups_parts.append(s)
setups = pd.concat(setups_parts, ignore_index=True).sort_values("time").reset_index(drop=True)

confirmed_parts = []
for rule_name in setups["rule"].unique():
    rdf = setups[setups["rule"] == rule_name].sort_values("time").reset_index(drop=True)
    cid = int(rdf["cluster_id"].iloc[0])
    meta_path = f"{models_dir}/confirm_c{cid}_{rule_name}_meta.json"
    model_path = f"{models_dir}/confirm_c{cid}_{rule_name}.json"
    if not os.path.exists(meta_path): continue
    meta = json.load(open(meta_path))
    if not meta.get("passes_strict", True): continue
    thr = meta["threshold"]
    fc = meta.get("feature_cols")
    avail = [x for x in fc if x in rdf.columns] if fc else []
    if len(avail) < 30: continue
    m = XGBClassifier(); m.load_model(model_path)
    split = int(len(rdf) * 0.8)
    te = rdf.iloc[split:].reset_index(drop=True)
    if len(te) == 0: continue
    X = te[avail].fillna(0).values
    if X.shape[1] != m.n_features_in_: continue
    p = m.predict_proba(X)[:, 1]
    msk = p >= thr
    keep = te[msk].copy()
    keep["ml_prob"] = p[msk]
    confirmed_parts.append(keep)

confirmed = pd.concat(confirmed_parts, ignore_index=True).sort_values("time").reset_index(drop=True)
confirmed["idx"] = confirmed["time"].map(t2i)
confirmed = confirmed.dropna(subset=["idx"]).copy()
confirmed["idx"] = confirmed["idx"].astype(int)
print(f"  Production confirmed holdout: {len(confirmed)}")

# For each confirmed trade, compute fingerprint at entry + run classifier
print(f"  Assigning vol regime to each confirmed trade...")
vol_regime = np.full(len(confirmed), -1, dtype=np.int8)
max_prob = np.zeros(len(confirmed))
for i, idx in enumerate(confirmed["idx"].values):
    if idx < WINDOW: continue
    fp = fingerprint(c[idx-WINDOW:idx], h[idx-WINDOW:idx], l[idx-WINDOW:idx], o[idx-WINDOW:idx])
    if fp is None: continue
    p = model.predict_proba(np.array([fp]))[0]
    vol_regime[i] = int(p.argmax())
    max_prob[i] = float(p.max())
confirmed["vol_regime"] = vol_regime
confirmed["vol_prob"] = max_prob
confirmed = confirmed[confirmed["vol_regime"] != -1].reset_index(drop=True)


# Simulate wider-SL/TP (SL=2, TP=6 — what we deployed)
def simulate_trades(df_, tp_mult=6.0, sl_mult=2.0, max_fwd=90, spread=0.05):
    atr_ = pd.Series(np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))).rolling(14, min_periods=1).mean().values
    trades = []
    for _, r in df_.iterrows():
        idx = int(r["idx"])
        if idx + max_fwd >= n_bars: continue
        a = atr_[idx]
        if a < 1e-10: continue
        entry = c[idx]
        dirv = r["direction"] if isinstance(r["direction"], (int, np.integer)) else (1 if r["direction"] in ("buy", 1) else -1)
        if dirv == 1:
            sl_p = entry - sl_mult * a; tp_p = entry + tp_mult * a
        else:
            sl_p = entry + sl_mult * a; tp_p = entry - tp_mult * a
        pnl = None
        for k in range(1, max_fwd + 1):
            bi = idx + k
            if bi >= n_bars: break
            if dirv == 1:
                if l[bi] <= sl_p: pnl = -sl_mult - spread; break
                if h[bi] >= tp_p: pnl = tp_mult - spread; break
            else:
                if h[bi] >= sl_p: pnl = -sl_mult - spread; break
                if l[bi] <= tp_p: pnl = tp_mult - spread; break
        if pnl is None:
            pnl = ((c[idx+max_fwd] - entry)/a - spread) * dirv
        trades.append({"time": r["time"], "pnl": pnl, "vol_regime": int(r["vol_regime"])})
    return pd.DataFrame(trades)

print("  Simulating (SL=2, TP=6 — deployed settings)...")
trades = simulate_trades(confirmed)

def stats(trs, name):
    if len(trs) == 0: return None
    w = (trs["pnl"] > 0.01).sum(); n = len(trs)
    pf = trs[trs["pnl"] > 0]["pnl"].sum() / max(-trs[trs["pnl"] < 0]["pnl"].sum(), 0.01)
    return {"name": name, "n": n, "wr": w/n, "pf": pf, "pnl": trs["pnl"].sum()}

overall = stats(trades, "ALL regimes")
by_regime = [stats(trades[trades["vol_regime"] == cls], NAMES[cls]) for cls in range(3)]
filtered = stats(trades[trades["vol_regime"] != 0], "Normal + HighVol only")

print(f"\n  {'Filter':<28}{'n':>6}{'WR':>7}{'PF':>7}{'PnL':>9}")
print("  " + "-"*58)
for s in [overall] + by_regime + [filtered]:
    if s is None: continue
    print(f"  {s['name']:<28}{s['n']:>6}{s['wr']:>7.0%}{s['pf']:>7.2f}{s['pnl']:>+9.1f}")

improvement = ""
if overall and filtered:
    pf_delta = filtered["pf"] - overall["pf"]
    pnl_pct = (filtered["pnl"] / max(overall["pnl"], 0.01) - 1) * 100
    improvement = f"\n  Filtering Quiet: PF change = {pf_delta:+.2f}  (removed {(overall['n']-filtered['n'])/overall['n']*100:.0f}% of trades)"
    print(improvement)


# ── Plots ──
fig = plt.figure(figsize=(22, 12), facecolor="#080c12")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# Per-class AUC
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#0d1117")
ax1.bar(range(3), aucs, color=[COLORS[c] for c in range(3)])
ax1.axhline(0.5, color="#666", linestyle="--", alpha=0.5)
ax1.axhline(0.75, color="#FFD700", linestyle="--", alpha=0.4, label="target 0.75")
ax1.set_xticks(range(3)); ax1.set_xticklabels([NAMES[c] for c in range(3)], color="#ccc")
ax1.set_ylabel("AUC", color="#ccc"); ax1.set_title("Per-class test AUC", color="#FFD700", fontsize=12)
ax1.legend(fontsize=9); ax1.tick_params(colors="#5a7080")
for sp in ax1.spines.values(): sp.set_edgecolor("#1e2a3a")

# Forward vol per predicted class
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#0d1117")
for cls in range(3):
    sub = test_df[test_df["pred"] == cls]
    if len(sub) == 0: continue
    ax2.bar(cls, sub["fwd_vol"].mean(), color=COLORS[cls],
            yerr=sub["fwd_vol"].std()/np.sqrt(len(sub)))
ax2.axhline(q25, color="#3b82f6", linestyle="--", alpha=0.4, label=f"Quiet cutoff {q25:.3f}")
ax2.axhline(q75, color="#ef4444", linestyle="--", alpha=0.4, label=f"HighVol cutoff {q75:.3f}")
ax2.set_xticks(range(3)); ax2.set_xticklabels([NAMES[c] for c in range(3)], color="#ccc")
ax2.set_ylabel("Mean fwd vol %", color="#ccc")
ax2.set_title("Forward vol per predicted class", color="#FFD700", fontsize=12)
ax2.legend(fontsize=8); ax2.tick_params(colors="#5a7080")
for sp in ax2.spines.values(): sp.set_edgecolor("#1e2a3a")

# Trade stats per regime
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor("#0d1117")
if by_regime[0]:
    x = range(3)
    pf_vals = [by_regime[c]["pf"] if by_regime[c] else 0 for c in range(3)]
    wr_vals = [by_regime[c]["wr"]*100 if by_regime[c] else 0 for c in range(3)]
    ax3.bar([i-0.2 for i in x], pf_vals, 0.35, color="#FFD700", label="PF")
    ax3_b = ax3.twinx()
    ax3_b.bar([i+0.2 for i in x], wr_vals, 0.35, color="#00E5FF", label="WR %")
    ax3.axhline(1.5, color="#10b981", linestyle="--", alpha=0.4)
    ax3.set_xticks(x); ax3.set_xticklabels([NAMES[c] for c in range(3)], color="#ccc")
    ax3.set_ylabel("PF (gold)", color="#FFD700")
    ax3_b.set_ylabel("WR% (cyan)", color="#00E5FF")
    ax3.set_title("Production trades split by vol regime (SL=2 TP=6)", color="#FFD700", fontsize=12)
    ax3.tick_params(colors="#5a7080"); ax3_b.tick_params(colors="#5a7080")
for sp in ax3.spines.values(): sp.set_edgecolor("#1e2a3a")

# Summary text
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor("#0d1117")
ax4.axis("off")
txt = "Regime v4 — Vol-Only\n" + "═"*45 + "\n\n"
txt += f"Test accuracy: {(preds == y_te).mean():.1%}\n"
txt += "Per-class AUC:\n"
for cls in range(3):
    txt += f"  {NAMES[cls]:<8} {aucs[cls]:.3f}\n"
txt += "\nTrade stats at SL=2 TP=6:\n"
if overall:
    txt += f"  {overall['name']:<22}n={overall['n']} WR={overall['wr']:.0%} PF={overall['pf']:.2f}\n"
for s in by_regime:
    if s: txt += f"  {s['name']:<22}n={s['n']:>4} WR={s['wr']:.0%} PF={s['pf']:.2f}\n"
if filtered:
    txt += f"  {filtered['name']:<22}n={filtered['n']} WR={filtered['wr']:.0%} PF={filtered['pf']:.2f}\n"
ax4.text(0, 1, txt, fontsize=10, color="#ccc", family="monospace", va="top")

plt.suptitle("Regime v4 — Vol-Only Classifier (overlay)",
             color="#FFD700", fontsize=16, fontweight="bold", y=0.995)
out = os.path.join(OUT_DIR, "regime_v4.png")
plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#080c12")
print(f"\nSaved: {out}")

# Save model
model.save_model(os.path.join(OUT_DIR, "regime_v4_classifier.json"))
meta = {
    "names": NAMES, "window": WINDOW, "step": STEP, "feat_cols": feat_cols,
    "vol_thresholds": {"q25": float(q25), "q75": float(q75)},
    "test_accuracy": float((preds == y_te).mean()),
    "per_class_auc": {NAMES[c]: float(aucs[c]) for c in range(3)},
    "overall_stats": overall,
    "by_regime": by_regime,
    "filtered_stats": filtered,
}
# serialize numpy stuff
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(x) for x in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return o
json.dump(clean(meta), open(os.path.join(OUT_DIR, "regime_v4_meta.json"), "w"), indent=2)
print(f"Saved: regime_v4_classifier.json + meta.json")
