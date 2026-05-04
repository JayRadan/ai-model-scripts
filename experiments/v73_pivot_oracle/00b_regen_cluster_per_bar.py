"""Regenerate cluster_per_bar_v73.csv from current v9.3 selector + Dukascopy XAU bars.

Mirrors model_pipeline/03_split_clusters_k5.py logic but writes a per-bar
(time, cid) CSV at the path Janus's 03_build_setups_v74.py expects.
"""
import os, sys, json
import numpy as np
import pandas as pd

ROOT = "/home/jay/Desktop/new-model-zigzag"
OUT_DIR = ROOT + "/experiments/v73_pivot_oracle/data"
OUT = OUT_DIR + "/cluster_per_bar_v73.csv"
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW = 288; STEP = 288; MIN_DATE = "2016-01-01"

raw = pd.read_csv(ROOT + "/data/swing_v5_xauusd.csv", parse_dates=["time"])
raw = raw[raw["time"] >= MIN_DATE].sort_values("time").reset_index(drop=True)
print(f"swing rows: {len(raw):,}", flush=True)

sel = json.load(open(ROOT + "/data/regime_selector_K4.json"))
feat_names = sel["feat_names"]
scaler_mean = np.array(sel["scaler_mean"])
scaler_std = np.array(sel["scaler_std"])
pca_mean = np.array(sel["pca_mean"])
pca_comp = np.array(sel["pca_components"])
centroids = np.array(sel["centroids"])
print(f"selector feat_names: {feat_names}", flush=True)

_FLOW_4H = None
if "flow_4h_mean" in feat_names:
    sys.path.insert(0, ROOT + "/experiments/v89_quantum_flow_tiebreaker")
    from importlib.machinery import SourceFileLoader
    qf = SourceFileLoader("qf01",
        ROOT + "/experiments/v89_quantum_flow_tiebreaker/01_port_and_test.py").load_module()
    rf = raw.rename(columns={"tick_volume": "volume"}) if "tick_volume" in raw.columns else raw
    _FLOW_4H = qf.quantum_flow_mtf(rf[["time","open","high","low","close","volume"]]).values
    print("computed flow_4h on full bars", flush=True)


def fp(c, h, l, o, flow=None):
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
    if flow is not None:
        fc = flow[~np.isnan(flow)]
        out["flow_4h_mean"] = float(fc.mean()) if len(fc) else 0.0
    return out


def classify(d):
    v = np.array([d[f] for f in feat_names])
    s = (v - scaler_mean) / scaler_std
    p = (s - pca_mean) @ pca_comp.T
    return int(np.argmin(np.sum((p - centroids)**2, axis=1)))


C = raw["close"].values.astype(np.float64)
H = raw["high"].values.astype(np.float64)
L = raw["low"].values.astype(np.float64)
O = raw["open"].values.astype(np.float64)

bar_cid = np.full(len(raw), -1, dtype=int)
for s in range(0, len(raw) - WINDOW, STEP):
    e = s + WINDOW
    d = fp(C[s:e], H[s:e], L[s:e], O[s:e],
           flow=_FLOW_4H[s:e] if _FLOW_4H is not None else None)
    if d is not None:
        bar_cid[s:e] = classify(d)

last = -1
for i in range(len(bar_cid)):
    if bar_cid[i] >= 0: last = bar_cid[i]
    elif last >= 0: bar_cid[i] = last

mask = bar_cid >= 0
out = pd.DataFrame({"time": raw.loc[mask, "time"].values, "cid": bar_cid[mask]})
out.to_csv(OUT, index=False)
print(f"wrote {len(out):,} rows → {OUT}", flush=True)
print(out["cid"].value_counts().sort_index())
