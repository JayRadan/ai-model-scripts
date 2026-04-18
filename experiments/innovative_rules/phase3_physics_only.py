"""
Phase 3: Retrain XAU confirmation with ONLY physics + candle structure features.
Drop all classic retail indicators (RSI, stoch, BB, momentum).

Feature sets compared:
  A) CURRENT:  f01-f20 + rsi + stoch + bb + mom + tech (36 features)
  B) PHYSICS:  f01-f20 + physics + VWAP + time encoding (28 features)
     - f01-f20 (21 candle/structure features — NOT retail indicators)
     - hurst_rs, ou_theta, entropy_rate, kramers_up, wavelet_er (5 physics)
     - vwap_dist (session VWAP distance / ATR)
     - hour_enc, dow_enc (2 time)

Big holdout: train on first 80%, test on last 20%, all rules aggregated.
"""
from __future__ import annotations
import glob, json, os, time as _time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jay/Desktop/new-model-zigzag"
TP_MULT, SL_MULT, MAX_FWD = 6.0, 2.0, 40

# ═══════ Feature definitions ═══════

BASE_F01_F20 = [
    "f01_CPR","f02_WickAsym","f03_BEF","f04_TCS","f05_SPI",
    "f06_LRSlope","f07_RECR","f08_SCM","f09_HLER","f10_EP",
    "f11_KE","f12_MCS","f13_Work","f14_EDR","f15_AI",
    "f16_PPShigh","f16_PPSlow","f17_SCR","f18_RVD","f19_WBER","f20_NCDE",
]

CURRENT_36 = BASE_F01_F20 + [
    "rsi14","rsi6","stoch_k","stoch_d","bb_pct",
    "mom5","mom10","mom20","ll_dist10","hh_dist10",
    "vol_accel","atr_ratio","spread_norm","hour_enc","dow_enc",
]

PHYSICS_COLS = ["hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er"]

PHYSICS_28 = BASE_F01_F20 + PHYSICS_COLS + ["vwap_dist", "hour_enc", "dow_enc"]


# ═══════ Physics + VWAP computation ═══════

def _ffill(arr):
    last = np.nan
    for i in range(len(arr)):
        if np.isfinite(arr[i]): last = arr[i]
        else: arr[i] = last

def compute_hurst_rs(ret, window=120, step=6):
    n = len(ret); h = np.full(n, np.nan)
    for i in range(window, n, step):
        seg = ret[i-window:i]; m = seg.mean()
        y = np.cumsum(seg - m); R = y.max() - y.min(); S = seg.std()
        if S > 1e-15 and R > 0: h[i] = np.log(R/S) / np.log(window)
    _ffill(h); return h

def compute_ou_theta(ret, window=60, step=6):
    n = len(ret); theta = np.full(n, np.nan); x = np.cumsum(ret)
    for i in range(window, n, step):
        seg = x[i-window:i]; mu = seg.mean()
        dx = np.diff(seg); xm = seg[:-1] - mu; d = np.sum(xm**2)
        if d > 1e-15: theta[i] = -np.sum(dx*xm)/d
    _ffill(theta); return theta

def compute_entropy(ret, window=100, n_bins=10, step=6):
    n = len(ret); ent = np.full(n, np.nan)
    for i in range(window, n, step):
        seg = ret[i-window:i]
        counts, _ = np.histogram(seg, bins=n_bins)
        p = counts/counts.sum(); p = p[p>0]
        ent[i] = -np.sum(p*np.log2(p)) / np.log2(n_bins)
    _ffill(ent); return ent

def compute_kramers_up(c, window=100, step=6):
    n = len(c); esc = np.full(n, np.nan)
    for i in range(window, n, step):
        seg = c[i-window:i]; hi = seg.max()
        sigma = np.std(np.diff(np.log(seg+1e-10)))
        if sigma < 1e-12: continue
        esc[i] = np.exp(-(hi-c[i])/(sigma*c[i]+1e-10))
    _ffill(esc); return esc

def _ricker(points, a):
    A = 2/(np.sqrt(3*a)*(np.pi**0.25)); wsq = a**2
    vec = np.arange(0, points)-(points-1.0)/2
    return A*(1-vec**2/wsq)*np.exp(-vec**2/(2*wsq))

def compute_wavelet_er(c, window=120, step=12):
    n = len(c); er = np.full(n, np.nan)
    kt = _ricker(min(200,window), 20); kn = _ricker(min(30,window), 3)
    for i in range(window, n, step):
        seg = c[i-window:i]; sn = (seg-seg.mean())/(seg.std()+1e-10)
        ct = np.convolve(sn, kt, mode='same'); cn = np.convolve(sn, kn, mode='same')
        er[i] = np.mean(ct**2)/(np.mean(cn**2)+1e-15)
    _ffill(er); return er

def compute_vwap_dist(df, atr):
    """Session VWAP (daily reset). Distance = (close - VWAP) / ATR."""
    c = df["close"].values.astype(np.float64)
    # Use tick_volume if available, else spread as proxy
    if "tick_volume" in df.columns:
        v = df["tick_volume"].values.astype(np.float64)
    else:
        v = df["spread"].values.astype(np.float64)
    v = np.maximum(v, 1.0)  # avoid zero volume

    typical = (df["high"].values + df["low"].values + c) / 3.0
    dates = df["time"].dt.date.values

    vwap = np.zeros(len(c))
    cum_tv = 0.0; cum_v = 0.0
    prev_date = dates[0]
    for i in range(len(c)):
        if dates[i] != prev_date:
            cum_tv = 0.0; cum_v = 0.0; prev_date = dates[i]
        cum_tv += typical[i] * v[i]
        cum_v  += v[i]
        vwap[i] = cum_tv / cum_v if cum_v > 0 else c[i]

    dist = np.where(atr > 1e-10, (c - vwap) / atr, 0.0)
    return dist


def compute_all_features(swing_path):
    """Return swing df with physics + VWAP features added."""
    print("Loading swing + computing features...", flush=True)
    df = pd.read_csv(swing_path, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    c = df["close"].values.astype(np.float64)
    h, l = df["high"].values.astype(np.float64), df["low"].values.astype(np.float64)

    # ATR
    tr = np.concatenate([[h[0]-l[0]],
          np.maximum.reduce([h[1:]-l[1:], np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values

    ret = np.concatenate([[0.0], np.diff(np.log(c))])

    t0 = _time.time()
    df["hurst_rs"]     = compute_hurst_rs(ret)
    df["ou_theta"]     = compute_ou_theta(ret)
    df["entropy_rate"] = compute_entropy(ret)
    df["kramers_up"]   = compute_kramers_up(c)
    df["wavelet_er"]   = compute_wavelet_er(c)
    df["vwap_dist"]    = compute_vwap_dist(df, atr)

    # Time encoding (matching Python training convention)
    hour = df["time"].dt.hour.astype(float)
    dow  = df["time"].dt.dayofweek.astype(float)
    df["hour_enc"] = np.sin(2*np.pi*hour/24)
    df["dow_enc"]  = np.sin(2*np.pi*dow/5)

    print(f"  done in {_time.time()-t0:.1f}s", flush=True)
    return df, atr


def forward_outcome(H, L, C, atr, i, direction):
    a = atr[i]
    if not np.isfinite(a) or a <= 0: return -1
    e = C[i]; n = len(C); end = min(i+1+MAX_FWD, n)
    if direction == 1:
        tp, sl = e + TP_MULT*a, e - SL_MULT*a
        for k in range(i+1, end):
            if L[k] <= sl: return 0
            if H[k] >= tp: return 1
    else:
        tp, sl = e - TP_MULT*a, e + SL_MULT*a
        for k in range(i+1, end):
            if H[k] >= sl: return 0
            if L[k] <= tp: return 1
    return 2


def main():
    print("="*70)
    print("PHASE 3: Physics-only features vs current 36 — big holdout")
    print("="*70, flush=True)

    swing, atr = compute_all_features(f"{ROOT}/data/swing_v5_xauusd.csv")
    H, L, C = swing["high"].values, swing["low"].values, swing["close"].values
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)

    # Load all setups
    all_setups = []
    for f in sorted(glob.glob(f"{ROOT}/data/setups_*.csv")):
        cid = int(os.path.basename(f).replace("setups_","").replace(".csv",""))
        df = pd.read_csv(f, parse_dates=["time"]); df["cid"] = cid
        all_setups.append(df)
    all_df = pd.concat(all_setups, ignore_index=True).sort_values("time").reset_index(drop=True)

    # Merge swing features (physics + VWAP + time enc) into setups by time
    swing_feats = swing[["time"] + PHYSICS_COLS + ["vwap_dist", "hour_enc", "dow_enc"]].copy()
    all_df = all_df.merge(swing_feats, on="time", how="left", suffixes=("","_new"))
    # Use new hour_enc/dow_enc if original exists
    for col in ["hour_enc","dow_enc"]:
        if col+"_new" in all_df.columns:
            all_df[col] = all_df[col+"_new"].fillna(all_df[col])
            all_df.drop(columns=[col+"_new"], inplace=True)
    for col in PHYSICS_COLS + ["vwap_dist"]:
        all_df[col] = all_df[col].fillna(0)

    print(f"Total setups: {len(all_df):,}", flush=True)

    # Relabel at TP=6/SL=2
    print("Relabeling...", flush=True)
    labels = []
    for _, row in all_df.iterrows():
        t = row["time"]
        if t not in time_to_idx.index: labels.append(-1); continue
        idx = int(time_to_idx[t]); d = int(row["direction"])
        labels.append(forward_outcome(H, L, C, atr, idx, d))
    all_df["label"] = labels
    all_df = all_df[all_df["label"].isin([0,1])].reset_index(drop=True)
    print(f"Valid: {len(all_df):,} (wins={all_df['label'].sum():,})", flush=True)

    # Three configs to compare
    configs = {
        "A) Current 36":    CURRENT_36,
        "B) Physics 28":    PHYSICS_28,
        "C) Current+Phys 41": CURRENT_36 + PHYSICS_COLS + ["vwap_dist"],
    }

    results = {}
    for cfg_name, feat_cols in configs.items():
        missing = [c for c in feat_cols if c not in all_df.columns]
        if missing:
            print(f"\n{cfg_name}: MISSING {missing[:5]}"); continue
        print(f"\n{cfg_name} ({len(feat_cols)} features):", flush=True)

        combined = {"pnls": [], "wins": 0, "losses": 0}
        rules_done = 0

        for (cid, rule), grp in all_df.groupby(["cid","rule"]):
            meta_path = f"{ROOT}/models/confirm_c{cid}_{rule}_meta.json"
            if not os.path.exists(meta_path): continue
            meta = json.load(open(meta_path))
            if meta.get("disabled", False): continue
            thr = meta["threshold"]

            rdf = grp.sort_values("time").reset_index(drop=True)
            split = int(len(rdf) * 0.8)
            train, test = rdf.iloc[:split], rdf.iloc[split:]
            if len(train) < 50 or len(test) < 20: continue

            y_tr, y_te = train["label"].values, test["label"].values
            X_tr = train[feat_cols].fillna(0).values
            X_te = test[feat_cols].fillna(0).values

            mdl = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                eval_metric="logloss", verbosity=0)
            mdl.fit(X_tr, y_tr)
            proba = mdl.predict_proba(X_te)[:,1]
            mask = proba >= thr
            if mask.sum() == 0: continue

            sel = y_te[mask]
            for s in sel:
                pnl = TP_MULT if s == 1 else -SL_MULT
                combined["pnls"].append(pnl)
                if s == 1: combined["wins"] += 1
                else: combined["losses"] += 1
            rules_done += 1

        pnls = np.array(combined["pnls"])
        n = len(pnls); w = combined["wins"]; lo = combined["losses"]
        if n == 0: print("  no trades"); continue
        wr = w/n; pf = (w*TP_MULT)/(lo*SL_MULT+1e-9)
        eq = np.cumsum(pnls); dd = (eq - np.maximum.accumulate(eq)).min()
        print(f"  n={n}  WR={wr:.1%}  PF={pf:.3f}  PnL={pnls.sum():+.1f}R  DD={dd:.1f}R  rules={rules_done}")
        results[cfg_name] = {"pnls": pnls, "n": n, "wr": wr, "pf": pf,
                             "total": pnls.sum(), "dd": dd}

    # Equity curves
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0b0e14")
    ax.set_facecolor("#0d1117")
    colors = {"A) Current 36": "#3b82f6", "B) Physics 28": "#f5c518", "C) Current+Phys 41": "#10b981"}
    for name, r in results.items():
        eq = np.cumsum(r["pnls"])
        ax.plot(eq, color=colors.get(name, "#fff"), linewidth=1.5,
                label=f"{name} PF={r['pf']:.2f} n={r['n']}")
    ax.axhline(0, color="#444", linewidth=0.5)
    ax.legend(facecolor="#111", edgecolor="#333", fontsize=10, loc="upper left")
    ax.set_title("XAU — Feature Set Comparison — Full Holdout",
                 color="#ffd700", fontsize=13)
    ax.set_xlabel("trade #", color="#888"); ax.set_ylabel("cumulative R", color="#888")
    ax.tick_params(colors="#888")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2332")
    plt.tight_layout()
    out = f"{ROOT}/experiments/innovative_rules/phase3_comparison.png"
    plt.savefig(out, dpi=140, facecolor="#0b0e14")
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
