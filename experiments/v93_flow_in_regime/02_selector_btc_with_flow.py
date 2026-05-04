"""BTC v9.3 — K=5 K-means selector with flow_4h_mean as 8th fingerprint feature.

Adapts experiments/v93_flow_in_regime/01_selector_with_flow_feature.py for BTC.
Outputs data/regime_selector_btc_K5.json + regime_fingerprints_btc_K5.csv.
"""
import sys, os, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
from importlib.machinery import SourceFileLoader
qf = SourceFileLoader("qf01",
    "/home/jay/Desktop/new-model-zigzag/experiments/v89_quantum_flow_tiebreaker/01_port_and_test.py"
).load_module()

ROOT = "/home/jay/Desktop/new-model-zigzag"
DATA = ROOT + "/data"
WINDOW = 288; STEP = 288; K = 5
MIN_DATE = "2018-01-01"

print("Loading BTC swing data...", flush=True)
df = pd.read_csv(DATA + "/swing_v5_btc.csv", parse_dates=["time"])
df = df.rename(columns={"tick_volume": "volume"}) if "tick_volume" in df.columns else df
df = df.sort_values("time").reset_index(drop=True)
df = df[df["time"] >= MIN_DATE].reset_index(drop=True)
print(f"  {len(df):,} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}", flush=True)

print("Computing flow_4h on full bars...", flush=True)
flow_4h = qf.quantum_flow_mtf(df[["time","open","high","low","close","volume"]])
print(f"  flow_4h med={float(np.nanmedian(flow_4h.values)):.2f}", flush=True)

C = df["close"].values.astype(np.float64)
H = df["high"].values.astype(np.float64)
L = df["low"].values.astype(np.float64)
O = df["open"].values.astype(np.float64)
F = flow_4h.values.astype(np.float64)


def fp(c, h, l, o, flow):
    if len(c) < 10: return None
    r = np.diff(c) / c[:-1]
    br = (h - l) / c
    out = {}
    out["weekly_return_pct"] = float(r.sum())
    out["volatility_pct"] = float(r.std())
    mr = r.mean()
    out["trend_consistency"] = float(np.mean(np.sign(r) == np.sign(mr))) if abs(mr) > 1e-12 else 0.5
    out["trend_strength"] = float(r.sum() / (r.std() + 1e-9))
    out["volatility"] = float(br.mean())
    tr = (h.max() - l.min()) / c.mean()
    out["range_vs_atr"] = float(tr / (br.mean() + 1e-9))
    if len(r) > 2:
        d = r[:-1].std() * r[1:].std()
        out["return_autocorr"] = float(np.corrcoef(r[:-1], r[1:])[0,1]) if d > 1e-12 else 0.0
    else:
        out["return_autocorr"] = 0.0
    fc = flow[~np.isnan(flow)]
    out["flow_4h_mean"] = float(fc.mean()) if len(fc) else 0.0
    return out


print("Computing rolling fingerprints...", flush=True)
rows = []
for s in range(0, len(df) - WINDOW, STEP):
    e = s + WINDOW
    d = fp(C[s:e], H[s:e], L[s:e], O[s:e], F[s:e])
    if d is not None:
        d["start_idx"] = s; d["end_idx"] = e
        rows.append(d)
fp_df = pd.DataFrame(rows)
feats = ["weekly_return_pct","volatility_pct","trend_consistency","trend_strength",
         "volatility","range_vs_atr","return_autocorr","flow_4h_mean"]
X = fp_df[feats].values
print(f"  {len(fp_df)} fingerprints  feats={len(feats)}", flush=True)

scaler = StandardScaler(); Xs = scaler.fit_transform(X)
mask = np.all(np.abs(Xs) < 4, axis=1)
fp_df = fp_df[mask].reset_index(drop=True)
X = fp_df[feats].values
scaler = StandardScaler(); Xs = scaler.fit_transform(X)
print(f"  {len(fp_df)} after outlier removal", flush=True)

pca = PCA(n_components=len(feats))
Xp = pca.fit_transform(Xs)
print(f"  PCA var explained: {pca.explained_variance_ratio_.sum()*100:.1f}%", flush=True)

km = KMeans(n_clusters=K, n_init=20, random_state=42)
raw_lbl = km.fit_predict(Xp)

stats = {}
for c in range(K):
    m = raw_lbl == c
    stats[c] = dict(ret=fp_df.loc[m,"weekly_return_pct"].mean(),
                    vol=fp_df.loc[m,"volatility_pct"].mean(),
                    autocorr=fp_df.loc[m,"return_autocorr"].mean(),
                    flow=fp_df.loc[m,"flow_4h_mean"].mean(),
                    n=int(m.sum()))
sb = sorted(stats.items(), key=lambda x: x[1]["ret"])
down = sb[0][0]; up = sb[-1][0]
rest = [c for c in range(K) if c not in (down, up)]
hv = sorted(rest, key=lambda c: stats[c]["vol"], reverse=True)[0]
rest2 = [c for c in rest if c != hv]
mr = sorted(rest2, key=lambda c: stats[c]["autocorr"])[0]
tr = [c for c in rest2 if c != mr][0]
lblmap = {down:3, up:0, hv:4, mr:1, tr:2}

print("\nCluster mapping:")
names = {0:"Uptrend", 1:"MeanRevert", 2:"TrendRange", 3:"Downtrend", 4:"HighVol"}
for raw, new in lblmap.items():
    s = stats[raw]
    print(f"  raw C{raw} → C{new} {names[new]:>11s}  n={s['n']:>4}  "
          f"ret={s['ret']:+.2%}  vol={s['vol']:.3%}  flow={s['flow']:+.1f}", flush=True)

centroids = np.empty_like(km.cluster_centers_)
for raw, new in lblmap.items():
    centroids[new] = km.cluster_centers_[raw]

out = {
    "K": K, "window": WINDOW, "step": STEP,
    "n_feats": len(feats), "feat_names": feats,
    "scaler_mean": scaler.mean_.tolist(), "scaler_std": scaler.scale_.tolist(),
    "pca_mean": pca.mean_.tolist(), "pca_components": pca.components_.tolist(),
    "centroids": centroids.tolist(),
    "cluster_names": {str(k):v for k,v in names.items()},
    "tradeable": {"0":True,"1":True,"2":True,"3":True,"4":True},
    "thresholds": {"weekly_return_pct": 0.0},
}
with open(DATA + "/regime_selector_btc_K5.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {DATA}/regime_selector_btc_K5.json", flush=True)
fp_df["new_label"] = [lblmap[r] for r in raw_lbl]
fp_df.to_csv(DATA + "/regime_fingerprints_btc_K5.csv", index=False)
print(f"Saved: {DATA}/regime_fingerprints_btc_K5.csv", flush=True)
