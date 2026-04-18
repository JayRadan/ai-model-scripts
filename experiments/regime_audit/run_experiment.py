"""
Regime audit — does the K=5 classifier actually assign the right cluster?

Runs the live-equivalent classifier at every bar on holdout, measures:
  1. Confidence margin (distance to 2nd-nearest centroid)
  2. Cluster persistence (bars-between-flips)
  3. Forward-behavior confusion: does each cluster's physics match its label?
     - Uptrend    → positive forward return
     - Downtrend  → negative forward return
     - MeanRevert → negative autocorr
     - TrendRange → high vol AND high autocorr
     - HighVol    → highest volatility band
  4. Live vs training distribution of assigned clusters

Output: table + PNG showing which clusters are reliable and which misfire.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = "/home/jay/Desktop/new-model-zigzag/experiments/regime_audit"
os.makedirs(OUT_DIR, exist_ok=True)

RAW_CSV = "/home/jay/Desktop/new-model-zigzag/data/swing_v5_xauusd.csv"
SELECTOR_PATH = "/home/jay/Desktop/new-model-zigzag/data/regime_selector_K4.json"

CLUSTER_NAMES = {0: "Uptrend", 1: "MeanRevert", 2: "TrendRange", 3: "Downtrend", 4: "HighVol"}

# ── Load raw + selector ──
print("Loading raw + K=5 selector...")
raw = pd.read_csv(RAW_CSV, parse_dates=["time"])
raw = raw[raw["time"] >= "2016-01-01"].sort_values("time").reset_index(drop=True)
c = raw["close"].values.astype(np.float64)
h = raw["high"].values.astype(np.float64)
l = raw["low"].values.astype(np.float64)
o = raw["open"].values.astype(np.float64)
n_bars = len(raw)
print(f"  {n_bars:,} bars")

sel = json.load(open(SELECTOR_PATH))
WINDOW = sel["window"]           # 288
FEAT_NAMES = sel["feat_names"]   # 7
MEAN = np.array(sel["scaler_mean"])
STD = np.array(sel["scaler_std"])
PCA_MEAN = np.array(sel["pca_mean"])
PCA_COMPS = np.array(sel["pca_components"])  # shape (7, 7)
CENTROIDS = np.array(sel["centroids"])       # shape (5, 7)
print(f"  K={sel['K']}  window={WINDOW}  features={len(FEAT_NAMES)}")


def fingerprint(c_, h_, l_, o_):
    n = len(c_)
    if n < 10: return None
    r = np.diff(c_) / c_[:-1]
    br = (h_ - l_) / c_
    fp = np.zeros(7)
    fp[0] = r.sum()
    fp[1] = r.std()
    mean_r = r.mean()
    fp[2] = np.mean(np.sign(r) == np.sign(mean_r)) if abs(mean_r) > 1e-12 else 0.5
    fp[3] = r.sum() / (r.std() + 1e-9)
    fp[4] = br.mean()
    total_range = (h_.max() - l_.min()) / c_.mean()
    fp[5] = total_range / (br.mean() + 1e-9)
    if len(r) > 2:
        r1, r2 = r[:-1], r[1:]
        denom = r1.std() * r2.std()
        fp[6] = np.corrcoef(r1, r2)[0, 1] if denom > 1e-12 else 0.0
    return fp


def classify(fp_vec):
    """Return (cluster_id, margin, distances[5])."""
    scaled = (fp_vec - MEAN) / STD
    pca = (scaled - PCA_MEAN) @ PCA_COMPS.T
    dists = np.linalg.norm(CENTROIDS - pca, axis=1)
    order = np.argsort(dists)
    cid = int(order[0])
    margin = float(dists[order[1]] - dists[order[0]])
    return cid, margin, dists


# ── Classify every bar (live-style: last 288 bars, step 1) ──
# For speed, step by 12 bars (1 hour) — finer than WINDOW but still fast
STEP = 12
print(f"\nClassifying every {STEP} bars (live-style rolling, window={WINDOW})...")

records = []
for idx in range(WINDOW, n_bars, STEP):
    start = idx - WINDOW
    fp = fingerprint(c[start:idx], h[start:idx], l[start:idx], o[start:idx])
    if fp is None: continue
    cid, margin, dists = classify(fp)

    # Forward outcome: next 288 bars (match window)
    fwd_end = min(idx + WINDOW, n_bars)
    fwd_ret = (c[fwd_end - 1] - c[idx]) / c[idx] * 100 if fwd_end > idx else 0
    fwd_vol = np.std(np.diff(c[idx:fwd_end]) / c[idx:fwd_end-1]) * 100 if fwd_end - idx > 2 else 0
    # Forward autocorr
    fwd_r = np.diff(c[idx:fwd_end]) / c[idx:fwd_end-1] if fwd_end - idx > 2 else np.array([0])
    if len(fwd_r) > 2:
        r1, r2 = fwd_r[:-1], fwd_r[1:]
        den = r1.std() * r2.std()
        fwd_ac = float(np.corrcoef(r1, r2)[0, 1]) if den > 1e-12 else 0.0
    else:
        fwd_ac = 0.0

    records.append({
        "time": raw["time"].iloc[idx], "idx": idx, "cluster": cid, "margin": margin,
        "d0": dists[0], "d1": dists[1], "d2": dists[2], "d3": dists[3], "d4": dists[4],
        "fwd_ret": fwd_ret, "fwd_vol": fwd_vol, "fwd_ac": fwd_ac,
    })

df = pd.DataFrame(records)
print(f"  Classified {len(df):,} points")


# ── Analysis ──

# 1. Cluster distribution (training vs here-measured)
print("\n" + "="*70)
print("1. CLUSTER DISTRIBUTION (live-style rolling vs. expected ~20% each)")
print("="*70)
for cid in range(5):
    pct = (df["cluster"] == cid).mean() * 100
    print(f"  C{cid} {CLUSTER_NAMES[cid]:<12}: {pct:>5.1f}%  ({(df['cluster']==cid).sum():,} points)")


# 2. Margin distribution — low margin = uncertain
print("\n" + "="*70)
print("2. CONFIDENCE MARGIN (distance to 2nd-nearest - distance to 1st)")
print("="*70)
for cid in range(5):
    sub = df[df["cluster"] == cid]
    if len(sub) == 0: continue
    q = np.quantile(sub["margin"], [0.10, 0.50, 0.90])
    low_conf = (sub["margin"] < 0.2).mean() * 100
    print(f"  C{cid} {CLUSTER_NAMES[cid]:<12}: p10={q[0]:.2f}  p50={q[1]:.2f}  p90={q[2]:.2f}  (low-conf <0.2: {low_conf:.0f}%)")


# 3. Persistence — bars between cluster flips
print("\n" + "="*70)
print("3. PERSISTENCE (run-length before cluster changes)")
print("="*70)
clusters_seq = df["cluster"].values
flips = np.where(np.diff(clusters_seq) != 0)[0]
if len(flips) > 1:
    run_lens = np.diff(np.concatenate([[0], flips, [len(clusters_seq)]]))
    print(f"  Mean run length:   {run_lens.mean():.1f} points ({run_lens.mean() * STEP / 12:.1f} hours)")
    print(f"  Median run length: {np.median(run_lens):.1f}")
    print(f"  % runs ≤ 2 points: {(run_lens <= 2).mean()*100:.0f}%  (noisy flicker)")
    print(f"  % runs > 24 points: {(run_lens > 24).mean()*100:.0f}%  (stable regimes)")


# 4. Forward-behavior confusion — does cluster physics match label?
print("\n" + "="*70)
print("4. FORWARD BEHAVIOR per cluster (validates cluster labels)")
print("="*70)
print(f"  {'Cluster':<18} {'n':>6} {'mean_ret%':>10} {'vol%':>8} {'autocorr':>10}")
print("  " + "-"*60)
for cid in range(5):
    sub = df[df["cluster"] == cid]
    if len(sub) == 0: continue
    print(f"  C{cid} {CLUSTER_NAMES[cid]:<14}: {len(sub):>6}  {sub['fwd_ret'].mean():>+10.2f}  {sub['fwd_vol'].mean():>8.3f}  {sub['fwd_ac'].mean():>+10.3f}")

# Expected physics:
#   Uptrend    → fwd_ret > 0
#   Downtrend  → fwd_ret < 0
#   MeanRevert → autocorr < 0
#   TrendRange → autocorr > 0 AND some ret
#   HighVol    → highest vol
print("\n  Expected labels:")
print("    Uptrend:    fwd_ret should be positive")
print("    Downtrend:  fwd_ret should be negative")
print("    MeanRevert: autocorr should be negative")
print("    TrendRange: autocorr should be positive")
print("    HighVol:    fwd_vol should be highest of the 5")


# 5. Low-confidence trade audit
print("\n" + "="*70)
print("5. LOW-CONFIDENCE MOMENTS (margin < 0.15)")
print("="*70)
low_conf = df[df["margin"] < 0.15]
print(f"  {len(low_conf):,} low-conf points ({len(low_conf)/len(df)*100:.0f}% of time)")
# Worst of them
worst = low_conf.nsmallest(5, "margin")
print(f"  Tightest 5 decisions:")
for _, r in worst.iterrows():
    second_best = int(np.argsort([r.d0, r.d1, r.d2, r.d3, r.d4])[1])
    print(f"    {r['time']}: C{int(r['cluster'])} ({CLUSTER_NAMES[int(r['cluster'])]}) vs C{second_best} ({CLUSTER_NAMES[second_best]}) — margin={r['margin']:.3f}")


# ── Plots ──
fig = plt.figure(figsize=(22, 14), facecolor="#080c12")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

colors = {0: "#f5c518", 1: "#3b82f6", 2: "#00E5FF", 3: "#ef4444", 4: "#a855f7"}

# (1) Cluster time series
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor("#0d1117")
ax1.scatter(df["time"], df["cluster"], c=[colors[c] for c in df["cluster"]], s=2, alpha=0.5)
ax1.set_yticks(range(5)); ax1.set_yticklabels([f"C{i} {CLUSTER_NAMES[i]}" for i in range(5)], color="#ccc")
ax1.set_title("Cluster assignment over time (finer = live simulation)", color="#FFD700", fontsize=13)
ax1.tick_params(colors="#5a7080")
for sp in ax1.spines.values(): sp.set_edgecolor("#1e2a3a")

# (2) Margin histogram per cluster
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor("#0d1117")
for cid in range(5):
    sub = df[df["cluster"] == cid]
    if len(sub) == 0: continue
    ax2.hist(sub["margin"], bins=40, color=colors[cid], alpha=0.5, label=f"C{cid} {CLUSTER_NAMES[cid]}")
ax2.axvline(0.15, color="#ef4444", linestyle="--", label="conf=0.15")
ax2.set_xlabel("Confidence margin", color="#ccc")
ax2.set_title("Margin distribution per cluster", color="#FFD700", fontsize=12)
ax2.legend(fontsize=8)
ax2.tick_params(colors="#5a7080")
for sp in ax2.spines.values(): sp.set_edgecolor("#1e2a3a")

# (3) Forward-return per cluster (box or bar)
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor("#0d1117")
for cid in range(5):
    sub = df[df["cluster"] == cid]
    if len(sub) == 0: continue
    ax3.bar(cid, sub["fwd_ret"].mean(), color=colors[cid], yerr=sub["fwd_ret"].std()/np.sqrt(len(sub)),
            label=f"C{cid} {CLUSTER_NAMES[cid]}")
ax3.axhline(0, color="#666", linewidth=0.6)
ax3.set_xticks(range(5)); ax3.set_xticklabels([f"C{i}\n{CLUSTER_NAMES[i][:6]}" for i in range(5)], color="#ccc")
ax3.set_ylabel("Mean forward 288-bar return %", color="#ccc")
ax3.set_title("Forward behavior check (labels should match)", color="#FFD700", fontsize=12)
ax3.tick_params(colors="#5a7080")
for sp in ax3.spines.values(): sp.set_edgecolor("#1e2a3a")

# (4) Persistence — run length histogram
ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor("#0d1117")
if len(flips) > 1:
    ax4.hist(run_lens, bins=50, color="#10b981", alpha=0.7, edgecolor="#ccc")
    ax4.axvline(np.median(run_lens), color="#FFD700", linestyle="--", label=f"median={np.median(run_lens):.0f}")
ax4.set_xlabel("Run length (points between flips)", color="#ccc")
ax4.set_title("Cluster persistence — how long before it flips", color="#FFD700", fontsize=12)
ax4.legend(fontsize=9)
ax4.tick_params(colors="#5a7080")
for sp in ax4.spines.values(): sp.set_edgecolor("#1e2a3a")

# (5) Text summary
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor("#0d1117")
ax5.axis("off")
txt = "Cluster distribution (live-style):\n"
for cid in range(5):
    pct = (df["cluster"] == cid).mean() * 100
    txt += f"  C{cid} {CLUSTER_NAMES[cid]:<12}: {pct:>5.1f}%\n"
txt += "\nForward behavior (mean fwd 288-bar return):\n"
for cid in range(5):
    sub = df[df["cluster"] == cid]
    if len(sub) == 0: continue
    ok = ""
    if cid == 0 and sub["fwd_ret"].mean() > 0: ok = "✓"
    elif cid == 3 and sub["fwd_ret"].mean() < 0: ok = "✓"
    elif cid == 1 and sub["fwd_ac"].mean() < 0: ok = "✓"
    elif cid == 2 and sub["fwd_ac"].mean() > 0: ok = "✓"
    elif cid == 4: ok = ""  # vol-check below
    else: ok = "⚠"
    txt += f"  C{cid} {CLUSTER_NAMES[cid]:<12}: ret={sub['fwd_ret'].mean():+.2f}%  ac={sub['fwd_ac'].mean():+.3f}  {ok}\n"
txt += f"\nLow-confidence (margin<0.15): {len(df[df['margin']<0.15])/len(df)*100:.0f}% of time"
if len(flips) > 1:
    txt += f"\nMean run length: {run_lens.mean():.1f} points"
    txt += f"\nFlicker rate (runs ≤ 2): {(run_lens<=2).mean()*100:.0f}%"
ax5.text(0, 1, txt, fontsize=10, color="#ccc", family="monospace", va="top")

plt.suptitle("XAUUSD Regime Audit — is K=5 classifier picking the right cluster?",
             color="#FFD700", fontsize=16, fontweight="bold", y=0.995)
out = os.path.join(OUT_DIR, "regime_audit.png")
plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#080c12")
print(f"\nSaved: {out}")

# Save classified data for further analysis
df.to_csv(os.path.join(OUT_DIR, "classified_bars.csv"), index=False)
print(f"Saved: {os.path.join(OUT_DIR, 'classified_bars.csv')}")
