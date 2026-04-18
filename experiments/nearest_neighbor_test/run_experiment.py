"""
Nearest-neighbor test — Jay's intuition:
"current week is like past week X → next week is like past week X's next"

Direct test:
  For every week in holdout:
    Find k=10 nearest neighbors in training period (by fingerprint)
    Check: did neighbors' forward direction match this week's actual forward direction?
    Also: did neighbors' forward vol match?

Baselines:
  - Random: 50% WR on direction
  - Trivial "always up": true for gold (secular uptrend), higher than 50%

Compare k-NN accuracy vs baselines. If k-NN beats trivial baseline on direction,
Jay is right. If it matches on vol but not direction, the three-experiment
finding is confirmed.
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = "/home/jay/Desktop/new-model-zigzag/experiments/nearest_neighbor_test"
os.makedirs(OUT_DIR, exist_ok=True)

RAW_CSV = "/home/jay/Desktop/new-model-zigzag/data/swing_v5_xauusd.csv"
WINDOW = 288
STEP = 24  # daily

print("Loading XAUUSD M5...")
raw = pd.read_csv(RAW_CSV, parse_dates=["time"])
raw = raw[raw["time"] >= "2016-01-01"].sort_values("time").reset_index(drop=True)
c = raw["close"].values.astype(np.float64)
h = raw["high"].values.astype(np.float64)
l = raw["low"].values.astype(np.float64)
n_bars = len(raw)
print(f"  {n_bars:,} bars")


def fingerprint(c_, h_, l_):
    n = len(c_)
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
        den = r[:-1].std() * r[1:].std()
        fp[6] = np.corrcoef(r[:-1], r[1:])[0, 1] if den > 1e-12 else 0.0
    return fp


print(f"\nComputing {WINDOW}-bar fingerprints + forward outcomes (step={STEP})...")
t0 = time.time()
records = []
for idx in range(WINDOW, n_bars - WINDOW, STEP):
    fp = fingerprint(c[idx-WINDOW:idx], h[idx-WINDOW:idx], l[idx-WINDOW:idx])
    fwd_ret = (c[idx+WINDOW-1] - c[idx]) / c[idx] * 100
    fwd_r = np.diff(c[idx:idx+WINDOW]) / c[idx:idx+WINDOW-1]
    fwd_vol = fwd_r.std() * 100
    records.append({
        "time": raw["time"].iloc[idx], "idx": idx,
        "fp0": fp[0], "fp1": fp[1], "fp2": fp[2], "fp3": fp[3],
        "fp4": fp[4], "fp5": fp[5], "fp6": fp[6],
        "fwd_ret": fwd_ret, "fwd_vol": fwd_vol,
        "fwd_sign": 1 if fwd_ret > 0 else -1,
    })

df = pd.DataFrame(records)
print(f"  {len(df)} weekly-ish points in {time.time()-t0:.1f}s")

# Split 70/30 time-ordered: neighbors searched in train, tested on holdout
df = df.sort_values("time").reset_index(drop=True)
split = int(len(df) * 0.70)
train = df.iloc[:split].reset_index(drop=True)
test = df.iloc[split:].reset_index(drop=True)
print(f"  train={len(train)}  test={len(test)}")

# Standardize fingerprints
feat_cols = [f"fp{i}" for i in range(7)]
scaler = StandardScaler()
X_train = scaler.fit_transform(train[feat_cols].values)
X_test = scaler.transform(test[feat_cols].values)

# Baseline 1: always-up (secular bias)
always_up_acc = (test["fwd_sign"] == 1).mean()
print(f"\nBaseline 'always up' accuracy: {always_up_acc:.0%}")

# Baseline 2: random (50%)
print(f"Baseline 'random' accuracy:   50%")


# ── k-NN test across multiple k values ──
print("\n" + "="*70)
print("NEAREST-NEIGHBOR TEST — does 'similar past week' predict direction?")
print("="*70)

results = []
for k in [1, 3, 5, 10, 20, 50]:
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_train)
    dists, idxs = nn.kneighbors(X_test)

    # For each test point, aggregate neighbors' outcomes
    neighbor_ret = train["fwd_ret"].values[idxs]        # (n_test, k)
    neighbor_vol = train["fwd_vol"].values[idxs]
    neighbor_sign = train["fwd_sign"].values[idxs]

    # Prediction: majority vote for direction, mean for vol/ret
    pred_sign = np.sign(neighbor_sign.sum(axis=1))
    pred_sign[pred_sign == 0] = 1  # break tie to up
    pred_ret = neighbor_ret.mean(axis=1)
    pred_vol = neighbor_vol.mean(axis=1)

    # Accuracies
    dir_acc = (pred_sign == test["fwd_sign"].values).mean()
    vol_corr = np.corrcoef(pred_vol, test["fwd_vol"].values)[0, 1]
    ret_corr = np.corrcoef(pred_ret, test["fwd_ret"].values)[0, 1]

    print(f"\n  k={k}:")
    print(f"    Direction accuracy:      {dir_acc:.0%}  (random=50%, always-up={always_up_acc:.0%})")
    print(f"    Vol correlation:         {vol_corr:+.3f}  (0=useless, 1=perfect)")
    print(f"    Return correlation:      {ret_corr:+.3f}  (0=useless)")

    results.append({"k": k, "dir_acc": dir_acc, "vol_corr": vol_corr,
                    "ret_corr": ret_corr, "pred_ret": pred_ret, "pred_vol": pred_vol})


# ── Focused case study — show 5 examples ──
print("\n" + "="*70)
print("CASE STUDIES — 5 random test weeks and their nearest neighbors (k=10)")
print("="*70)

k = 10
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_train)
dists, idxs = nn.kneighbors(X_test)

np.random.seed(42)
sample_idx = np.random.choice(len(test), size=5, replace=False)
for si in sample_idx:
    actual_ret = test["fwd_ret"].iloc[si]
    actual_sign = "UP" if actual_ret > 0 else "DN"
    test_time = test["time"].iloc[si]
    neighbor_rets = train["fwd_ret"].values[idxs[si]]
    neighbor_signs_up = (neighbor_rets > 0).sum()
    neighbor_mean = neighbor_rets.mean()
    print(f"\n  Test week {test_time.strftime('%Y-%m-%d')}: actual next 288 bars = {actual_ret:+.2f}% ({actual_sign})")
    print(f"    Nearest 10 past-week neighbors: {neighbor_signs_up}/{k} went UP, mean fwd ret = {neighbor_mean:+.2f}%")
    print(f"    Neighbor forward returns: {', '.join(f'{r:+.1f}' for r in sorted(neighbor_rets))}")


# ── Plots ──
fig = plt.figure(figsize=(22, 12), facecolor="#080c12")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

# (1) Direction accuracy vs k
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#0d1117")
ks = [r["k"] for r in results]
ax1.plot(ks, [r["dir_acc"]*100 for r in results], "o-", color="#10b981", linewidth=2, markersize=10, label="k-NN dir accuracy")
ax1.axhline(50, color="#ef4444", linestyle="--", alpha=0.6, label="random 50%")
ax1.axhline(always_up_acc*100, color="#FFD700", linestyle="--", alpha=0.6, label=f"always-up {always_up_acc:.0%}")
ax1.set_xlabel("k (neighbors)", color="#ccc")
ax1.set_ylabel("Direction accuracy %", color="#ccc")
ax1.set_title("Direction accuracy — does k-NN beat baselines?", color="#FFD700", fontsize=13)
ax1.set_xscale("log")
ax1.legend(fontsize=10)
ax1.tick_params(colors="#5a7080")
for sp in ax1.spines.values(): sp.set_edgecolor("#1e2a3a")

# (2) Vol correlation vs k
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#0d1117")
ax2.plot(ks, [r["vol_corr"] for r in results], "o-", color="#00E5FF", linewidth=2, markersize=10, label="vol corr")
ax2.plot(ks, [r["ret_corr"] for r in results], "o-", color="#ef4444", linewidth=2, markersize=10, label="return corr")
ax2.axhline(0, color="#666", linestyle="--", alpha=0.5)
ax2.set_xlabel("k (neighbors)", color="#ccc")
ax2.set_ylabel("Correlation", color="#ccc")
ax2.set_title("Correlation: neighbors' forward outcome vs actual", color="#FFD700", fontsize=13)
ax2.set_xscale("log")
ax2.legend(fontsize=10)
ax2.tick_params(colors="#5a7080")
for sp in ax2.spines.values(): sp.set_edgecolor("#1e2a3a")

# (3) Predicted vs actual return (k=10)
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor("#0d1117")
r10 = next(r for r in results if r["k"] == 10)
ax3.scatter(r10["pred_ret"], test["fwd_ret"], c="#10b981", s=15, alpha=0.4)
lim = max(abs(test["fwd_ret"].min()), abs(test["fwd_ret"].max())) * 1.1
ax3.plot([-lim, lim], [-lim, lim], color="#FFD700", linestyle="--", alpha=0.5, label="perfect prediction")
ax3.axhline(0, color="#666", linewidth=0.4); ax3.axvline(0, color="#666", linewidth=0.4)
ax3.set_xlabel("k=10 predicted return %", color="#ccc")
ax3.set_ylabel("Actual return %", color="#ccc")
ax3.set_title(f"Return prediction (corr={r10['ret_corr']:+.3f})", color="#FFD700", fontsize=12)
ax3.legend(fontsize=9)
ax3.tick_params(colors="#5a7080")
for sp in ax3.spines.values(): sp.set_edgecolor("#1e2a3a")

# (4) Predicted vs actual vol (k=10)
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor("#0d1117")
ax4.scatter(r10["pred_vol"], test["fwd_vol"], c="#00E5FF", s=15, alpha=0.4)
lim_v = max(test["fwd_vol"].max(), r10["pred_vol"].max()) * 1.05
ax4.plot([0, lim_v], [0, lim_v], color="#FFD700", linestyle="--", alpha=0.5)
ax4.set_xlabel("k=10 predicted vol %", color="#ccc")
ax4.set_ylabel("Actual vol %", color="#ccc")
ax4.set_title(f"Volatility prediction (corr={r10['vol_corr']:+.3f})", color="#FFD700", fontsize=12)
ax4.tick_params(colors="#5a7080")
for sp in ax4.spines.values(): sp.set_edgecolor("#1e2a3a")

plt.suptitle("Nearest-Neighbor Test — 'similar past week' predicts what?",
             color="#FFD700", fontsize=16, fontweight="bold", y=1.0)
out = os.path.join(OUT_DIR, "nn_test.png")
plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#080c12")
print(f"\nSaved: {out}")
