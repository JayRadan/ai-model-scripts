"""
Final deployed backtest — what the 3 EAs will ACTUALLY do live.

Applies to each instrument's holdout ML-confirmed trades:
  1. Deployed SL/TP (Midas/Samurai SL=2 TP=6, Meridian SL=1.5 TP=6)
  2. Vol-regime sizing multiplier (0.5× Quiet, 1.0× Normal, 1.3× HighVol)
     — drives per-trade $-PnL in live

Reports 3 scenarios per instrument:
  A) Baseline — old SL=1 TP=2, flat lots
  B) Wider SL/TP — new SL=2or1.5 TP=6, flat lots (already shipped)
  C) Wider + vol sizing — new SL/TP + regime_v4 multiplier (final live behavior)
"""
import os, json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = "/home/jay/Desktop/new-model-zigzag/experiments/final_deployed_backtest"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_FWD = 90
WINDOW = 288

INSTRUMENTS = {
    "Midas (XAUUSD)": {
        "raw": "/home/jay/Desktop/new-model-zigzag/data/swing_v5_xauusd.csv",
        "models": "/home/jay/Desktop/new-model-zigzag/models",
        "setups_pattern": "/home/jay/Desktop/new-model-zigzag/data/setups_{cid}.csv",
        "spread": 0.05, "meta_strict": True,
        "sl": 2.0, "tp": 6.0,
        "regime_onnx": "/home/jay/Desktop/new-model-zigzag/experiments/regime_v4_vol_only/regime_v4_xau.onnx",
    },
    "Samurai (GBPJPY)": {
        "raw": "/home/jay/Desktop/new-model-zigzag/gbpjpy/data/swing_v5_gbpjpy.csv",
        "models": "/home/jay/Desktop/new-model-zigzag/gbpjpy/models",
        "setups_csv": "/home/jay/Desktop/new-model-zigzag/gbpjpy/data/setup_signals_gbpjpy.csv",
        "spread": 0.10, "meta_strict": False,
        "sl": 2.0, "tp": 6.0,
        "regime_onnx": "/home/jay/Desktop/new-model-zigzag/experiments/regime_v4_vol_only/regime_v4_gj.onnx",
    },
    "Meridian (EURUSD)": {
        "raw": "/home/jay/Desktop/new-model-zigzag/eurusd/data/swing_v5_eurusd.csv",
        "models": "/home/jay/Desktop/new-model-zigzag/eurusd/models",
        "setups_csv": "/home/jay/Desktop/new-model-zigzag/eurusd/data/setup_signals_eurusd.csv",
        "spread": 0.04, "meta_strict": False,
        "sl": 1.5, "tp": 6.0,
        "regime_onnx": "/home/jay/Desktop/new-model-zigzag/experiments/regime_v4_vol_only/regime_v4_eu.onnx",
    },
}

LOT_MULT = {0: 0.5, 1: 1.0, 2: 1.3}
REGIME_NAMES = {0: "Quiet", 1: "Normal", 2: "HighVol"}


def load_instrument(name, cfg):
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    raw = pd.read_csv(cfg["raw"], parse_dates=["time"])
    raw = raw[raw["time"] >= "2016-01-01"].reset_index(drop=True)
    c = raw["close"].values.astype(np.float64)
    h = raw["high"].values.astype(np.float64)
    l = raw["low"].values.astype(np.float64)
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    t2i = dict(zip(raw["time"], range(len(raw))))

    if "setups_csv" in cfg:
        s = pd.read_csv(cfg["setups_csv"], parse_dates=["time"])
        if "cluster" in s.columns and "cluster_id" not in s.columns:
            s["cluster_id"] = s["cluster"]
    else:
        parts = []
        for cid in range(5):
            p = cfg["setups_pattern"].format(cid=cid)
            if os.path.exists(p):
                x = pd.read_csv(p, parse_dates=["time"]); x["cluster_id"] = cid
                parts.append(x)
        s = pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True)

    print(f"  Raw bars: {len(raw):,}  Setups: {len(s):,}")
    return {"c": c, "h": h, "l": l, "atr": atr, "n": len(raw), "t2i": t2i,
            "setups": s, "cfg": cfg}


def confirm(data):
    cfg = data["cfg"]; setups = data["setups"]
    parts = []
    for rule in setups["rule"].unique():
        rdf = setups[setups["rule"] == rule].sort_values("time").reset_index(drop=True)
        cid = int(rdf["cluster_id"].iloc[0])
        mp = f"{cfg['models']}/confirm_c{cid}_{rule}_meta.json"
        modp = f"{cfg['models']}/confirm_c{cid}_{rule}.json"
        if not os.path.exists(mp): continue
        meta = json.load(open(mp))
        if cfg["meta_strict"]:
            if not meta.get("passes_strict", True): continue
        else:
            if meta.get("disabled", False): continue
        thr = meta["threshold"]
        try:
            m = XGBClassifier(); m.load_model(modp)
        except Exception:
            continue
        fc = meta.get("feature_cols") or []
        avail = [c_ for c_ in fc if c_ in rdf.columns] if fc else []
        if len(avail) < 20:
            skip = {"time","rule","cluster","cluster_id","direction","outcome","label","idx"}
            cand = [c_ for c_ in rdf.columns if c_ not in skip and pd.api.types.is_numeric_dtype(rdf[c_])]
            if len(cand) >= m.n_features_in_: avail = cand[:m.n_features_in_]
            else: continue
        split = int(len(rdf) * 0.8)
        te = rdf.iloc[split:].reset_index(drop=True)
        if len(te) == 0: continue
        X = te[avail].fillna(0).values
        if X.shape[1] != m.n_features_in_: continue
        p = m.predict_proba(X)[:, 1]
        msk = p >= thr
        keep = te[msk].copy(); keep["ml_prob"] = p[msk]
        parts.append(keep)

    df = pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True)
    df["idx"] = df["time"].map(data["t2i"])
    df = df.dropna(subset=["idx"]).copy()
    df["idx"] = df["idx"].astype(int)
    print(f"  ML-confirmed holdout: {len(df):,}")
    return df


def fingerprint(c_, h_, l_):
    n = len(c_)
    r = np.diff(c_) / c_[:-1]
    br = (h_ - l_) / c_
    fp = np.zeros(7)
    fp[0] = r.sum(); fp[1] = r.std()
    mean_r = r.mean()
    fp[2] = np.mean(np.sign(r) == np.sign(mean_r)) if abs(mean_r) > 1e-12 else 0.5
    fp[3] = r.sum() / (r.std() + 1e-9)
    fp[4] = br.mean()
    tr = (h_.max() - l_.min()) / c_.mean()
    fp[5] = tr / (br.mean() + 1e-9)
    if len(r) > 2:
        d = r[:-1].std() * r[1:].std()
        fp[6] = np.corrcoef(r[:-1], r[1:])[0, 1] if d > 1e-12 else 0.0
    return fp


def assign_vol_regime(df, data, onnx_path):
    sess = ort.InferenceSession(onnx_path)
    c, h, l, n = data["c"], data["h"], data["l"], data["n"]
    reg = np.full(len(df), 1, dtype=np.int8)
    for i, idx in enumerate(df["idx"].values):
        if idx < WINDOW: reg[i] = 1; continue
        fp = fingerprint(c[idx-WINDOW:idx], h[idx-WINDOW:idx], l[idx-WINDOW:idx])
        x = np.array([fp], dtype=np.float32)
        out = sess.run(None, {"input": x})
        reg[i] = int(out[0][0])
    df = df.copy(); df["vol_regime"] = reg
    return df


def simulate(df, data, tp_mult, sl_mult):
    c, h, l, atr, n, spread = data["c"], data["h"], data["l"], data["atr"], data["n"], data["cfg"]["spread"]
    trades = []
    for _, r in df.iterrows():
        idx = int(r["idx"])
        if idx + MAX_FWD >= n: continue
        a = atr[idx]
        if a < 1e-10: continue
        entry = c[idx]
        dirv = r["direction"] if isinstance(r["direction"], (int, np.integer)) else (1 if r["direction"] in ("buy", 1) else -1)
        if dirv == 1:
            sl_p = entry - sl_mult * a; tp_p = entry + tp_mult * a
        else:
            sl_p = entry + sl_mult * a; tp_p = entry - tp_mult * a
        pnl = None
        for k in range(1, MAX_FWD + 1):
            bi = idx + k
            if bi >= n: break
            if dirv == 1:
                if l[bi] <= sl_p: pnl = -sl_mult - spread; break
                if h[bi] >= tp_p: pnl = tp_mult - spread; break
            else:
                if h[bi] >= sl_p: pnl = -sl_mult - spread; break
                if l[bi] <= tp_p: pnl = tp_mult - spread; break
        if pnl is None:
            pnl = ((c[idx+MAX_FWD] - entry) / a - spread) * dirv
        trades.append({"time": r["time"], "pnl_R": pnl,
                       "vol_regime": int(r.get("vol_regime", 1))})
    return pd.DataFrame(trades)


def stats(trades, weighted=False):
    if len(trades) == 0: return None
    if weighted:
        trades = trades.copy()
        trades["mult"] = trades["vol_regime"].map(LOT_MULT)
        trades["pnl"] = trades["pnl_R"] * trades["mult"]
    else:
        trades["pnl"] = trades["pnl_R"]

    w = (trades["pnl"] > 0.01).sum()
    lo = (trades["pnl"] < -0.01).sum()
    n = len(trades)
    wr = w / n
    pf = trades[trades["pnl"] > 0]["pnl"].sum() / max(-trades[trades["pnl"] < 0]["pnl"].sum(), 0.01)
    pnl = trades["pnl"].sum()
    eq = np.cumsum(trades.sort_values("time")["pnl"].values)
    dd = (eq - np.maximum.accumulate(eq)).min()
    return {"n": n, "wr": wr, "pf": pf, "pnl": pnl, "dd": dd, "eq": eq, "trades": trades}


# ── Run ──
all_results = {}
for name, cfg in INSTRUMENTS.items():
    data = load_instrument(name, cfg)
    confirmed = confirm(data)
    if len(confirmed) < 50: continue

    # Assign vol regime to each confirmed trade
    print(f"  Assigning vol regime...")
    confirmed = assign_vol_regime(confirmed, data, cfg["regime_onnx"])
    vol_dist = confirmed["vol_regime"].value_counts().sort_index()
    for v, c_ in vol_dist.items():
        print(f"    {REGIME_NAMES.get(int(v),'?')}: {c_} ({c_/len(confirmed)*100:.0f}%)")

    # Three scenarios
    base = simulate(confirmed, data, tp_mult=2.0, sl_mult=1.0)
    wide = simulate(confirmed, data, tp_mult=cfg["tp"], sl_mult=cfg["sl"])
    # Vol-weighted = wide result with multiplier applied
    s_base = stats(base)
    s_wide = stats(wide)
    s_final = stats(wide, weighted=True)

    print(f"\n  {'Scenario':<32} {'n':>5} {'WR':>6} {'PF':>6} {'PnL(R)':>10} {'DD(R)':>9}")
    print(f"  " + "-"*72)
    print(f"  {'A) Baseline SL=1 TP=2':<32} {s_base['n']:>5} {s_base['wr']:>6.0%} {s_base['pf']:>6.2f} {s_base['pnl']:>+10.1f} {s_base['dd']:>+9.1f}")
    print(f"  {'B) Wider SL/TP (shipped)':<32} {s_wide['n']:>5} {s_wide['wr']:>6.0%} {s_wide['pf']:>6.2f} {s_wide['pnl']:>+10.1f} {s_wide['dd']:>+9.1f}")
    print(f"  {'C) Wider + Vol Sizing (LIVE)':<32} {s_final['n']:>5} {s_final['wr']:>6.0%} {s_final['pf']:>6.2f} {s_final['pnl']:>+10.1f} {s_final['dd']:>+9.1f}")

    improvement_B = (s_wide['pnl'] / max(s_base['pnl'], 0.01) - 1) * 100
    improvement_C = (s_final['pnl'] / max(s_base['pnl'], 0.01) - 1) * 100
    print(f"\n  PnL improvement: B vs A = {improvement_B:+.0f}%,  C vs A = {improvement_C:+.0f}%,  C vs B = {(s_final['pnl']/max(s_wide['pnl'],0.01)-1)*100:+.0f}%")

    all_results[name] = {"A": s_base, "B": s_wide, "C": s_final,
                         "vol_dist": vol_dist.to_dict()}


# ── Plot ──
fig = plt.figure(figsize=(22, 14), facecolor="#080c12")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.25)
inst_colors = {"Midas (XAUUSD)": "#FFD700", "Samurai (GBPJPY)": "#ef4444", "Meridian (EURUSD)": "#3b82f6"}

for row, (name, res) in enumerate(all_results.items()):
    color = inst_colors[name]
    ax1 = fig.add_subplot(gs[row, 0])
    ax1.set_facecolor("#0d1117")
    ax1.plot(res["A"]["eq"], color="#6b7280", linewidth=1.2, alpha=0.7, label=f"A) Baseline  WR={res['A']['wr']:.0%} PF={res['A']['pf']:.2f}  PnL={res['A']['pnl']:+.0f}")
    ax1.plot(res["B"]["eq"], color="#10b981", linewidth=1.4, alpha=0.85, label=f"B) Wider TP  WR={res['B']['wr']:.0%} PF={res['B']['pf']:.2f}  PnL={res['B']['pnl']:+.0f}")
    ax1.plot(res["C"]["eq"], color=color, linewidth=1.6, alpha=0.95, label=f"C) +Vol Size  WR={res['C']['wr']:.0%} PF={res['C']['pf']:.2f}  PnL={res['C']['pnl']:+.0f}")
    ax1.axhline(0, color="#444", linewidth=0.6)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_title(f"{name} — equity curves (R-units)", color=color, fontsize=13)
    ax1.tick_params(colors="#5a7080")
    for sp in ax1.spines.values(): sp.set_edgecolor("#1e2a3a")

    ax2 = fig.add_subplot(gs[row, 1])
    ax2.set_facecolor("#0d1117")
    scenarios = ["A) Baseline", "B) Wider", "C) Final LIVE"]
    pf_vals = [res[k]["pf"] for k in ["A", "B", "C"]]
    pnl_vals = [res[k]["pnl"] for k in ["A", "B", "C"]]
    x = np.arange(3)
    b1 = ax2.bar(x - 0.2, pf_vals, 0.35, color="#FFD700", label="PF")
    ax2_b = ax2.twinx()
    b2 = ax2_b.bar(x + 0.2, pnl_vals, 0.35, color=color, alpha=0.6, label="PnL (R)")
    ax2.axhline(1.5, color="#10b981", linestyle="--", alpha=0.4)
    ax2.set_xticks(x); ax2.set_xticklabels(scenarios, color="#ccc")
    ax2.set_ylabel("PF", color="#FFD700")
    ax2_b.set_ylabel("PnL (R)", color=color)
    ax2.set_title(f"{name} — PF & PnL comparison", color=color, fontsize=12)
    ax2.tick_params(colors="#5a7080"); ax2_b.tick_params(colors="#5a7080")
    for sp in ax2.spines.values(): sp.set_edgecolor("#1e2a3a")

plt.suptitle("Final Deployed Backtest — what the 3 EAs will do LIVE",
             color="#FFD700", fontsize=17, fontweight="bold", y=0.995)
out = os.path.join(OUT_DIR, "final_deployed.png")
plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#080c12")
print(f"\nSaved: {out}")

# Save summary JSON
summary = {}
for name, res in all_results.items():
    summary[name] = {
        "baseline": {"wr": res["A"]["wr"], "pf": res["A"]["pf"], "pnl_R": res["A"]["pnl"], "dd_R": res["A"]["dd"], "n": res["A"]["n"]},
        "wider": {"wr": res["B"]["wr"], "pf": res["B"]["pf"], "pnl_R": res["B"]["pnl"], "dd_R": res["B"]["dd"], "n": res["B"]["n"]},
        "live": {"wr": res["C"]["wr"], "pf": res["C"]["pf"], "pnl_R": res["C"]["pnl"], "dd_R": res["C"]["dd"], "n": res["C"]["n"]},
        "vol_regime_dist": {REGIME_NAMES.get(int(k), str(k)): int(v) for k, v in res["vol_dist"].items()},
    }
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return o
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(clean(summary), f, indent=2)
print(f"Saved: summary.json")
