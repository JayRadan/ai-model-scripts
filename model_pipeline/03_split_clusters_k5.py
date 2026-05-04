"""
Split labeled XAUUSD data into per-cluster CSVs using K=5 rolling-window assignments.
"""
import pandas as pd
import numpy as np
import json
import paths as P

CLUSTER_NAMES = {0: "Uptrend", 1: "MeanRevert", 2: "TrendRange", 3: "Downtrend", 4: "HighVol"}
K = 5; WINDOW = 288; STEP = 288; MIN_DATE = "2016-01-01"

df = pd.read_csv(P.data("labeled_v4.csv"), parse_dates=["time"])
print(f"Labeled data: {len(df):,} rows")

raw = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
raw = raw[raw["time"] >= MIN_DATE].reset_index(drop=True)

# v9.3: optionally compute flow_4h_mean if selector demands it
sel_json = json.load(open(P.data("regime_selector_K4.json")))
_FLOW_4H = None
if "flow_4h_mean" in sel_json["feat_names"]:
    import sys, os
    sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/experiments/v89_quantum_flow_tiebreaker")
    from importlib.machinery import SourceFileLoader
    qf = SourceFileLoader("qf01",
        "/home/jay/Desktop/new-model-zigzag/experiments/v89_quantum_flow_tiebreaker/01_port_and_test.py"
    ).load_module()
    raw_flow = raw.rename(columns={"tick_volume": "volume"}) if "tick_volume" in raw.columns else raw
    _FLOW_4H = qf.quantum_flow_mtf(raw_flow[["time","open","high","low","close","volume"]]).values
scaler_mean = np.array(sel_json["scaler_mean"])
scaler_std = np.array(sel_json["scaler_std"])
pca_mean = np.array(sel_json["pca_mean"])
pca_comp = np.array(sel_json["pca_components"])
centroids = np.array(sel_json["centroids"])
feat_names = sel_json["feat_names"]

def compute_fp(c, h, l, o, flow=None):
    n = len(c)
    if n < 10: return None
    returns = np.diff(c) / c[:-1]; bar_ranges = (h - l) / c
    fp = {}
    fp["weekly_return_pct"] = float(returns.sum())
    fp["volatility_pct"] = float(returns.std())
    mean_ret = returns.mean()
    fp["trend_consistency"] = float(np.mean(np.sign(returns) == np.sign(mean_ret))) if abs(mean_ret) > 1e-12 else 0.5
    fp["trend_strength"] = float(returns.sum() / (returns.std() + 1e-9))
    fp["volatility"] = float(bar_ranges.mean())
    total_range = (h.max() - l.min()) / c.mean()
    fp["range_vs_atr"] = float(total_range / (bar_ranges.mean() + 1e-9))
    if len(returns) > 2:
        r1, r2 = returns[:-1], returns[1:]
        denom = r1.std() * r2.std()
        fp["return_autocorr"] = float(np.corrcoef(r1, r2)[0, 1]) if denom > 1e-12 else 0.0
    else:
        fp["return_autocorr"] = 0.0
    if flow is not None:
        flow_clean = flow[~np.isnan(flow)]
        fp["flow_4h_mean"] = float(flow_clean.mean()) if len(flow_clean) else 0.0
    return fp

def classify_fp(fp_dict):
    vec = np.array([fp_dict[f] for f in feat_names])
    scaled = (vec - scaler_mean) / scaler_std
    rotated = (scaled - pca_mean) @ pca_comp.T
    dists = np.sum((rotated - centroids) ** 2, axis=1)
    return int(np.argmin(dists))

closes = raw["close"].values.astype(np.float64)
highs = raw["high"].values.astype(np.float64)
lows = raw["low"].values.astype(np.float64)
opens = raw["open"].values.astype(np.float64)

bar_clusters = np.full(len(raw), -1, dtype=int)
for start in range(0, len(raw) - WINDOW, STEP):
    end = start + WINDOW
    fp = compute_fp(closes[start:end], highs[start:end], lows[start:end], opens[start:end],
                     flow=_FLOW_4H[start:end] if _FLOW_4H is not None else None)
    if fp is not None:
        bar_clusters[start:end] = classify_fp(fp)

last = -1
for i in range(len(bar_clusters)):
    if bar_clusters[i] >= 0: last = bar_clusters[i]
    elif last >= 0: bar_clusters[i] = last

raw["cluster"] = bar_clusters
time_to_cluster = dict(zip(raw["time"], raw["cluster"]))
df["cluster"] = df["time"].map(time_to_cluster)
df = df.dropna(subset=["cluster"])
df["cluster"] = df["cluster"].astype(int)
df = df[df["cluster"] >= 0].reset_index(drop=True)

BUY, FLAT, SELL = 0, 1, 2
for cid in range(K):
    sub = df[df["cluster"] == cid].copy()
    if cid == 0: sub.loc[sub["entry_class"] == SELL, "entry_class"] = FLAT  # Uptrend: drop sells
    elif cid == 3: sub.loc[sub["entry_class"] == BUY, "entry_class"] = FLAT  # Downtrend: drop buys
    out = P.data(f"cluster_{cid}_data.csv")
    sub.to_csv(out, index=False)
    buys = (sub["entry_class"] == BUY).sum()
    sells = (sub["entry_class"] == SELL).sum()
    flats = (sub["entry_class"] == FLAT).sum()
    print(f"C{cid} {CLUSTER_NAMES[cid]:>12}: {len(sub):>7,} rows  BUY={buys:>6,}  FLAT={flats:>6,}  SELL={sells:>6,}")
