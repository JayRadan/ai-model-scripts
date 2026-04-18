"""
Phase 2b: Physics features — BIG holdout test.

Train on first 80% of setups chronologically, test on last 20%.
Aggregate ALL rules together for maximum sample size.
Compare baseline (36 feat) vs enhanced (36 + 5 physics) confirmation.

Also test: physics as regime selector features (7 → 12 dim fingerprint).
"""
from __future__ import annotations
import glob, json, os, sys, time as _time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jay/Desktop/new-model-zigzag"
TP_MULT, SL_MULT, MAX_FWD = 6.0, 2.0, 40
SPREAD = 0.40

PHYSICS_COLS = ["hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er"]


# ═══════════════ Physics features (reuse from phase2) ═══════════════
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
        dx = np.diff(seg); xm = seg[:-1] - mu; denom = np.sum(xm**2)
        if denom > 1e-15: theta[i] = -np.sum(dx * xm) / denom
    _ffill(theta); return theta

def compute_entropy(ret, window=100, n_bins=10, step=6):
    n = len(ret); ent = np.full(n, np.nan)
    for i in range(window, n, step):
        seg = ret[i-window:i]
        counts, _ = np.histogram(seg, bins=n_bins)
        p = counts / counts.sum(); p = p[p > 0]
        ent[i] = -np.sum(p * np.log2(p)) / np.log2(n_bins)
    _ffill(ent); return ent

def compute_kramers_up(c, window=100, step=6):
    n = len(c); esc = np.full(n, np.nan)
    for i in range(window, n, step):
        seg = c[i-window:i]; hi = seg.max()
        sigma = np.std(np.diff(np.log(seg + 1e-10)))
        if sigma < 1e-12: continue
        barrier = (hi - c[i]) / (sigma * c[i] + 1e-10)
        esc[i] = np.exp(-barrier)
    _ffill(esc); return esc

def _ricker(points, a):
    A = 2 / (np.sqrt(3*a) * (np.pi**0.25))
    wsq = a**2; vec = np.arange(0, points) - (points-1.0)/2
    return A * (1 - vec**2/wsq) * np.exp(-vec**2/(2*wsq))

def compute_wavelet_er(c, window=120, step=12):
    n = len(c); er = np.full(n, np.nan)
    kt = _ricker(min(200, window), 20); kn = _ricker(min(30, window), 3)
    for i in range(window, n, step):
        seg = c[i-window:i]; seg_n = (seg-seg.mean())/(seg.std()+1e-10)
        ct = np.convolve(seg_n, kt, mode='same'); cn = np.convolve(seg_n, kn, mode='same')
        er[i] = np.mean(ct**2) / (np.mean(cn**2) + 1e-15)
    _ffill(er); return er


def compute_physics_df(swing_path):
    print("Computing physics features on full swing...", flush=True)
    df = pd.read_csv(swing_path, usecols=["time","close"], parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    c = df["close"].values.astype(np.float64)
    ret = np.concatenate([[0.0], np.diff(np.log(c))])
    t0 = _time.time()
    h = compute_hurst_rs(ret); ou = compute_ou_theta(ret)
    ent = compute_entropy(ret); kr = compute_kramers_up(c)
    wv = compute_wavelet_er(c)
    print(f"  done in {_time.time()-t0:.1f}s", flush=True)
    return pd.DataFrame({"time": df["time"], "hurst_rs": h, "ou_theta": ou,
                         "entropy_rate": ent, "kramers_up": kr, "wavelet_er": wv})


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
    print("PHASE 2b: Big holdout — physics as confirmation features")
    print("="*70, flush=True)

    # Load swing
    swing = pd.read_csv(f"{ROOT}/data/swing_v5_xauusd.csv",
                        parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    H, L, C = swing["high"].values, swing["low"].values, swing["close"].values
    tr = np.concatenate([[H[0]-L[0]],
          np.maximum.reduce([H[1:]-L[1:], np.abs(H[1:]-C[:-1]), np.abs(L[1:]-C[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)

    # Physics
    physics = compute_physics_df(f"{ROOT}/data/swing_v5_xauusd.csv")

    # Collect ALL setups across all clusters
    all_setups = []
    for f in sorted(glob.glob(f"{ROOT}/data/setups_*.csv")):
        cid = int(os.path.basename(f).replace("setups_","").replace(".csv",""))
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        all_setups.append(df)
    all_df = pd.concat(all_setups, ignore_index=True).sort_values("time").reset_index(drop=True)
    print(f"Total setups: {len(all_df):,}", flush=True)

    # Merge physics
    all_df = all_df.merge(physics, on="time", how="left")
    for col in PHYSICS_COLS:
        all_df[col] = all_df[col].fillna(0)

    # Relabel all at TP=6, SL=2
    print("Relabeling at TP=6/SL=2...", flush=True)
    labels = []
    for _, row in all_df.iterrows():
        t = row["time"]
        if t not in time_to_idx.index: labels.append(-1); continue
        idx = int(time_to_idx[t])
        d = int(row["direction"])
        labels.append(forward_outcome(H, L, C, atr, idx, d))
    all_df["label_6x2"] = labels
    all_df = all_df[all_df["label_6x2"].isin([0, 1])].reset_index(drop=True)
    print(f"Valid labeled setups: {len(all_df):,} (wins={all_df['label_6x2'].sum():,})", flush=True)

    # Get feature cols from a model meta
    sample_meta = json.load(open(glob.glob(f"{ROOT}/models/confirm_c0_*_meta.json")[0]))
    base_feats = sample_meta["feature_cols"]
    enh_feats = base_feats + PHYSICS_COLS
    missing = [c for c in base_feats if c not in all_df.columns]
    if missing:
        print(f"MISSING features in setup CSVs: {missing}")
        return

    # Per-rule train/test, but aggregate results
    combined_base = {"wins": 0, "losses": 0, "n": 0, "pnls": []}
    combined_enh  = {"wins": 0, "losses": 0, "n": 0, "pnls": []}

    rules_done = 0
    for (cid, rule), grp in all_df.groupby(["cid", "rule"]):
        meta_path = f"{ROOT}/models/confirm_c{cid}_{rule}_meta.json"
        if not os.path.exists(meta_path): continue
        meta = json.load(open(meta_path))
        if meta.get("disabled", False): continue
        thr = meta["threshold"]
        rule_feats = meta.get("feature_cols", base_feats)

        rdf = grp.sort_values("time").reset_index(drop=True)
        split = int(len(rdf) * 0.8)
        train, test = rdf.iloc[:split], rdf.iloc[split:]
        if len(train) < 50 or len(test) < 20: continue

        y_tr = train["label_6x2"].values
        y_te = test["label_6x2"].values

        # A) Baseline
        X_tr_b = train[rule_feats].fillna(0).values
        X_te_b = test[rule_feats].fillna(0).values
        mdl_b = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric="logloss", verbosity=0)
        mdl_b.fit(X_tr_b, y_tr)
        p_b = mdl_b.predict_proba(X_te_b)[:,1]
        mask_b = p_b >= thr
        if mask_b.sum() > 0:
            sel = y_te[mask_b]
            for s in sel:
                pnl = TP_MULT if s == 1 else -SL_MULT
                combined_base["pnls"].append(pnl)
                combined_base["n"] += 1
                if s == 1: combined_base["wins"] += 1
                else: combined_base["losses"] += 1

        # B) Enhanced
        rule_feats_enh = rule_feats + PHYSICS_COLS
        miss = [c for c in rule_feats_enh if c not in train.columns]
        if miss: continue
        X_tr_e = train[rule_feats_enh].fillna(0).values
        X_te_e = test[rule_feats_enh].fillna(0).values
        mdl_e = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric="logloss", verbosity=0)
        mdl_e.fit(X_tr_e, y_tr)
        p_e = mdl_e.predict_proba(X_te_e)[:,1]
        mask_e = p_e >= thr
        if mask_e.sum() > 0:
            sel = y_te[mask_e]
            for s in sel:
                pnl = TP_MULT if s == 1 else -SL_MULT
                combined_enh["pnls"].append(pnl)
                combined_enh["n"] += 1
                if s == 1: combined_enh["wins"] += 1
                else: combined_enh["losses"] += 1

        rules_done += 1

    print(f"\nRules processed: {rules_done}", flush=True)

    # Results
    for label, d in [("BASELINE (36 features)", combined_base),
                     ("ENHANCED (36+5 physics)", combined_enh)]:
        pnls = np.array(d["pnls"])
        if len(pnls) == 0:
            print(f"\n{label}: no trades"); continue
        w = d["wins"]; l = d["losses"]; n = d["n"]
        wr = w / n; pf = (w * TP_MULT) / (l * SL_MULT + 1e-9)
        eq = np.cumsum(pnls)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak).min()
        print(f"\n{label}:")
        print(f"  Trades: {n}  Wins: {w}  Losses: {l}")
        print(f"  Win rate: {wr:.1%}")
        print(f"  PF: {pf:.3f}")
        print(f"  Total PnL (R): {pnls.sum():+.1f}")
        print(f"  Max DD (R): {dd:.1f}")

    # Equity curves
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0b0e14")
    ax.set_facecolor("#0d1117")
    if combined_base["pnls"]:
        eq_b = np.cumsum(combined_base["pnls"])
        ax.plot(eq_b, color="#3b82f6", linewidth=1.5, label=f"Baseline (36f) PF={combined_base['wins']*TP_MULT/(combined_base['losses']*SL_MULT+1e-9):.2f}")
    if combined_enh["pnls"]:
        eq_e = np.cumsum(combined_enh["pnls"])
        ax.plot(eq_e, color="#f5c518", linewidth=1.5, label=f"Enhanced (36+5 physics) PF={combined_enh['wins']*TP_MULT/(combined_enh['losses']*SL_MULT+1e-9):.2f}")
    ax.axhline(0, color="#444", linewidth=0.5)
    ax.legend(facecolor="#111", edgecolor="#333", fontsize=10)
    ax.set_title("XAU Confirmation: Baseline vs Physics-Enhanced — Full Holdout",
                 color="#ffd700", fontsize=13)
    ax.set_xlabel("trade #", color="#888"); ax.set_ylabel("cumulative R", color="#888")
    ax.tick_params(colors="#888")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2332")
    plt.tight_layout()
    out = f"{ROOT}/experiments/innovative_rules/phase2b_equity_curves.png"
    plt.savefig(out, dpi=140, facecolor="#0b0e14")
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
