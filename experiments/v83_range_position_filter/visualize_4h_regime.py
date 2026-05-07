"""Interactive regime chart — clean version.

Top panel: H1-resampled XAU candlesticks (clear price action).
Bottom panel: regime ribbon — solid color band per regime over time.
Shaded vertical bands behind candles also tint by regime.
"""
import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from importlib.machinery import SourceFileLoader

ROOT = "/home/jay/Desktop/new-model-zigzag"
DATA = ROOT + "/data"
OUT_HTML = ROOT + "/experiments/v83_range_position_filter/regimes_4h_holdout.html"

WINDOW = 288
STEP = 288
HOLDOUT_START = "2024-12-01"
RESAMPLE = "1H"  # candles for display

qf = SourceFileLoader("qf01",
    ROOT + "/experiments/v89_quantum_flow_tiebreaker/01_port_and_test.py"
).load_module()

print("Loading selector...")
sel = json.load(open("experiments/v83_range_position_filter/regime_selector_4h.json"))
feat_cols = sel["feat_names"]
scaler_mean = np.array(sel["scaler_mean"]); scaler_std = np.array(sel["scaler_std"])
pca_mean = np.array(sel["pca_mean"]); pca_components = np.array(sel["pca_components"])
centroids = np.array(sel["centroids"])
cluster_names = {int(k): v for k, v in sel["cluster_names"].items()}

print("Loading swing bars...")
df = pd.read_csv(DATA + "/swing_v5_xauusd.csv", parse_dates=["time"])
if "tick_volume" in df.columns: df = df.rename(columns={"tick_volume":"volume"})
df = df.sort_values("time").reset_index(drop=True)

print("Computing flow_4h...")
flow_4h = qf.quantum_flow_mtf(df[["time","open","high","low","close","volume"]])
closes = df["close"].values.astype(np.float64)
highs  = df["high"].values.astype(np.float64)
lows   = df["low"].values.astype(np.float64)
opens  = df["open"].values.astype(np.float64)
flow_arr = flow_4h.values.astype(np.float64)


def fingerprint(c, h, l, o, flow):
    if len(c) < 10: return None
    returns = np.diff(c) / c[:-1]
    bar_ranges = (h - l) / c
    mean_ret = returns.mean()
    fp = [
        float(returns.sum()),
        float(returns.std()),
        float(np.mean(np.sign(returns) == np.sign(mean_ret))) if abs(mean_ret) > 1e-12 else 0.5,
        float(returns.sum() / (returns.std() + 1e-9)),
        float(bar_ranges.mean()),
        float(((h.max() - l.min()) / c.mean()) / (bar_ranges.mean() + 1e-9)),
    ]
    if len(returns) > 2:
        r1, r2 = returns[:-1], returns[1:]
        denom = r1.std() * r2.std()
        fp.append(float(np.corrcoef(r1, r2)[0,1]) if denom > 1e-12 else 0.0)
    else:
        fp.append(0.0)
    flow_clean = flow[~np.isnan(flow)]
    fp.append(float(flow_clean.mean()) if len(flow_clean) else 0.0)
    return np.array(fp)


def predict_cluster(fp_vec):
    scaled = (fp_vec - scaler_mean) / scaler_std
    pcs = (scaled - pca_mean) @ pca_components.T
    return int(np.argmin(np.linalg.norm(centroids - pcs, axis=1)))


print("Sliding windows...")
labels_full = np.full(len(df), -1, dtype=np.int8)
for start in range(0, len(df) - WINDOW, STEP):
    end = start + WINDOW
    fp = fingerprint(closes[start:end], highs[start:end], lows[start:end],
                     opens[start:end], flow_arr[start:end])
    if fp is None: continue
    labels_full[start:end] = predict_cluster(fp)

df["regime"] = labels_full
hold = df[(df["time"] >= HOLDOUT_START) & (df["regime"] >= 0)].reset_index(drop=True)
print(f"Holdout: {len(hold):,} M5 bars  {hold['time'].iloc[0]} → {hold['time'].iloc[-1]}")

# Resample to H1 candles, regime = mode of the hour
hold = hold.set_index("time")
ohlc = hold[["open","high","low","close"]].resample(RESAMPLE).agg(
    {"open":"first","high":"max","low":"min","close":"last"}).dropna()
reg = hold["regime"].resample(RESAMPLE).agg(lambda s: s.mode().iloc[0] if len(s) else -1).reindex(ohlc.index)
ohlc["regime"] = reg.astype(int)
ohlc = ohlc.reset_index()
print(f"Resampled to {len(ohlc):,} {RESAMPLE} candles")

COLORS = {0:"#26a69a", 1:"#42a5f5", 2:"#ab47bc", 3:"#ef5350", 4:"#ffa726"}
NAMES  = {k: f"C{k} {cluster_names.get(k,'?')}" for k in COLORS}

# Build regime "runs" so we draw fewer, wider shapes (much faster + cleaner)
runs = []
cur = ohlc["regime"].iloc[0]; start = ohlc["time"].iloc[0]
for i in range(1, len(ohlc)):
    if ohlc["regime"].iloc[i] != cur:
        runs.append((cur, start, ohlc["time"].iloc[i]))
        cur = ohlc["regime"].iloc[i]; start = ohlc["time"].iloc[i]
runs.append((cur, start, ohlc["time"].iloc[-1]))
print(f"Regime runs: {len(runs)}")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.85, 0.15], vertical_spacing=0.02,
                    subplot_titles=("XAU H1 — Background tinted by detected regime",
                                    "Regime ribbon"))

# Vertical regime tints behind candles (light alpha)
shapes = []
for cid, t0, t1 in runs:
    rgba = COLORS[cid].replace("#","")
    r,g,b = int(rgba[0:2],16), int(rgba[2:4],16), int(rgba[4:6],16)
    shapes.append(dict(type="rect", xref="x", yref="paper",
                       x0=t0, x1=t1, y0=0, y1=1,
                       fillcolor=f"rgba({r},{g},{b},0.12)", line=dict(width=0), layer="below"))

# Candles
fig.add_trace(go.Candlestick(
    x=ohlc["time"], open=ohlc["open"], high=ohlc["high"],
    low=ohlc["low"], close=ohlc["close"], name="XAU H1",
    increasing_line_color="#d8d8d8", decreasing_line_color="#888",
    showlegend=False,
), row=1, col=1)

# Ribbon: one bar trace per regime, value=1, colored solid
for cid in sorted(COLORS):
    mask = ohlc["regime"] == cid
    fig.add_trace(go.Bar(
        x=ohlc.loc[mask, "time"], y=[1]*int(mask.sum()),
        marker=dict(color=COLORS[cid], line=dict(width=0)),
        name=f"{NAMES[cid]}  ({int(mask.sum())}h)",
        hovertemplate=f"%{{x}}<br>{NAMES[cid]}<extra></extra>",
        width=3600*1000,  # 1h in ms — touching bars
    ), row=2, col=1)

fig.update_layout(
    title=f"Oracle XAU — v9.3 Regime Detection on Holdout  ({hold.index[0].date()} → {hold.index[-1].date()})",
    template="plotly_dark", height=820,
    barmode="stack", bargap=0,
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=-0.08, x=0),
    shapes=shapes,
    margin=dict(l=50, r=20, t=70, b=40),
)
fig.update_yaxes(title_text="XAU", row=1, col=1)
fig.update_yaxes(visible=False, row=2, col=1, range=[0,1])
fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), row=2, col=1)

os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
fig.write_html(OUT_HTML, include_plotlyjs="cdn")
print(f"\n✓ Saved: {OUT_HTML}")
