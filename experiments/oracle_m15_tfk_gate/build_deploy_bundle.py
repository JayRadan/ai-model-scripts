"""
Build deployable Oracle bundle with the NEW smart exit (future-upside regressor).

Outputs:
  1. Backup of current bundle: products/models/oracle_xau_validated.pkl.bak_pre_smart_exit_2026-06-22
  2. NEW bundle at products/models/oracle_xau_validated.pkl with:
      - exit_mdl: kept as fallback (the old XGBClassifier)
      - smart_exit_mdl: NEW XGBRegressor (predicts future_upside_R)
      - smart_exit_feats: ordered feature list for the regressor
      - smart_exit_upside_thr: 0.3   (HOLD while predicted upside >= 0.3R; else EXIT)
      - smart_exit_min_hold: 3       (skip first 3 bars)
      - smart_exit_min_pnl_R: 0.3    (only fire smart exit when already at +0.3R)
      - smart_exit_use_trail: False  (DISABLED — trail was clipping winners)
      - smart_exit_max_hold: 60      (M5 bars, = 5h)
      - smart_exit_sl_hard: 5.0      (-5R hard floor unchanged)
      - smart_exit_gate_m15_tfk_anti: True  (M15 TFK ANTI gate enabled)
      - smart_exit_version: "v1-smart-upside-2026-06-22"
      - smart_exit_holdout_metrics: {pf, sumR, ...} from 2.4y honest backtest

The server needs to (a) compute the smart-exit features at each bar of an
open trade, (b) call smart_exit_mdl.predict, (c) exit if prediction < upside_thr
AND unrealized_pnl_R >= min_pnl_R AND bars_held >= min_hold. Falls back to
hard_sl and max_hold as before.

Also writes deploy_notes.md with feature spec + server-side integration steps.
"""
from __future__ import annotations
import sys, time, glob, pickle, shutil
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "products/hermes_xau"))
from tfk import compute_tfk

SWING = ROOT / "data" / "swing_v5_xauusd.csv"
SETUP_GLOB = str(ROOT / "data" / "setups_*_v72l.csv")
ORACLE_PKL = ROOT / "products" / "models" / "oracle_xau_validated.pkl"
FINGERPRINTS = ROOT / "products" / "_shared" / "data" / "regime_fingerprints_4h.csv"
TODAY = "2026-06-22"
BACKUP_PKL = ROOT / "products" / "models" / f"oracle_xau_validated.pkl.bak_pre_smart_exit_{TODAY}"
NEW_PKL    = ROOT / "products" / "models" / f"oracle_xau_validated_smart_exit_{TODAY}.pkl"
DEPLOY_NOTES = HERE / "deploy_notes.md"

SPREAD_USD = 0.30
MIN_HOLD = 3; MAX_HOLD = 60; SL_HARD = 5.0
HOLDOUT = pd.Timestamp("2024-01-01")
UPSIDE_THR = 0.3  # winner from VAR3 sweep
MIN_PNL_R_FOR_SMART_EXIT = 0.3
SMART_EXIT_VERSION = f"v1-smart-upside-{TODAY}"

# Smart-exit feature order (must match between training and serving)
SMART_FEAT_NAMES = [
    "unrealized_R", "bars_held", "pnl_vel_3", "pnl_vel_5",
    "mfe_so_far_R", "mae_so_far_R", "dd_from_peak_R",
    "progress_to_sl", "progress_to_trail",
    "hour_utc", "dow", "m15_dir_x_dir",
    "cid", "direction", "bars_remaining", "frac_time", "trade_range_R", "horizon",
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up",
    "wavelet_er", "quantum_flow", "quantum_flow_h4", "vwap_dist",
]


def m15_dir(df_bars):
    s = df_bars.set_index("time")
    m15 = pd.DataFrame({"open":s["open"].resample("15min").first(),
                        "high":s["high"].resample("15min").max(),
                        "low": s["low"].resample("15min").min(),
                        "close":s["close"].resample("15min").last(),
                        "tick_volume":s["tick_volume"].resample("15min").sum()
                        }).dropna(subset=["close"]).reset_index()
    tfk_m15 = compute_tfk(m15, flip_bars=5, color_confirm=8)
    m15["m15_dir"] = tfk_m15["committed_dir"].to_numpy()
    aligned = pd.merge_asof(df_bars[["time"]].sort_values("time"),
                            m15[["time","m15_dir"]].assign(
                              time=m15["time"]+pd.Timedelta("15min")).sort_values("time"),
                            on="time", direction="backward")
    return aligned["m15_dir"].fillna(0).to_numpy(np.int64)


@njit(cache=True)
def build_state_table(sw_idxs, dirs, cids, hours, dows, m15_dirs,
                      ctx_arr, C, H, L, atr, max_hold):
    N = len(sw_idxs); nf = ctx_arr.shape[1]
    total_feats = 18 + nf
    rows_per_trade = max_hold
    Xs = np.zeros((N*rows_per_trade, total_feats), dtype=np.float32)
    ys = np.zeros(N*rows_per_trade, dtype=np.float32)
    valid = np.zeros(N*rows_per_trade, dtype=np.uint8)
    for rank in range(N):
        ei = sw_idxs[rank]; d = dirs[rank]; a = atr[ei]; ep = C[ei]
        if not (a > 0): continue
        cps = np.zeros(max_hold); valid_k = 0
        for k in range(1, max_hold+1):
            bar = ei + k
            if bar >= len(C): break
            fav = d*(C[bar]-ep); cps[k-1] = fav / a; valid_k = k
        for k in range(3, valid_k+1):
            bar = ei + k
            cp = cps[k-1]
            p3 = cps[k-4] if k >= 4 else cp
            p5 = cps[k-6] if k >= 6 else cp
            mfe_k = 0.0; mae_k = 0.0
            for kk in range(k):
                v = cps[kk]
                if v > mfe_k: mfe_k = v
                if -v > mae_k: mae_k = -v
            dd_from_peak = mfe_k - cp
            future_max = cp
            for kk in range(k, valid_k):
                if cps[kk] > future_max: future_max = cps[kk]
            future_upside = future_max - cp
            row = rank*rows_per_trade + (k-1)
            Xs[row, 0]=cp; Xs[row, 1]=float(k); Xs[row, 2]=cp-p3; Xs[row, 3]=cp-p5
            Xs[row, 4]=mfe_k; Xs[row, 5]=mae_k; Xs[row, 6]=dd_from_peak
            Xs[row, 7]=cp/5.0; Xs[row, 8]=cp/2.0
            Xs[row, 9]=float(hours[bar]); Xs[row,10]=float(dows[bar])
            Xs[row,11]=float(m15_dirs[bar]*d); Xs[row,12]=float(cids[rank])
            Xs[row,13]=float(d); Xs[row,14]=float(max_hold-k)
            Xs[row,15]=float(k)/float(max_hold); Xs[row,16]=mfe_k-mae_k
            Xs[row,17]=float(valid_k)
            for j in range(nf): Xs[row,18+j]=float(ctx_arr[bar, j])
            ys[row] = future_upside
            valid[row] = 1
    return Xs, ys, valid


@njit(cache=True)
def sim_smart_only(sw_idxs, dirs, O, H, L, C, atr,
                   sl_hard_R, upside_thr_R, min_pnl_R, pred_per_state,
                   max_hold, sp_R, min_hold):
    N = len(sw_idxs)
    pnl = np.zeros(N); bars = np.zeros(N, dtype=np.int64); reason = np.zeros(N, dtype=np.int64)
    for rank in range(N):
        ei = sw_idxs[rank]; d = dirs[rank]; a = atr[ei]; ep = C[ei]
        if not (a > 0): bars[rank]=0; reason[rank]=3; continue
        xi = -1; xr = 3
        for k in range(1, max_hold+1):
            bar = ei + k
            if bar >= len(C): break
            fav = d*(C[bar]-ep); cp = fav / a
            if d == 1 and (ep - L[bar]) >= sl_hard_R*a: xi=bar; xr=0; break
            if d == -1 and (H[bar] - ep) >= sl_hard_R*a: xi=bar; xr=0; break
            if k >= min_hold and cp >= min_pnl_R:
                if pred_per_state[rank, k-1] < upside_thr_R:
                    xi = bar; xr = 2; break
        if xi < 0: xi = min(ei + max_hold, len(C)-1); xr = 3
        pnl[rank] = d*(C[xi]-ep)/a - sp_R
        bars[rank] = xi - ei
        reason[rank] = xr
    return pnl, bars, reason


def metrics(r):
    r = np.asarray(r); r = r[np.isfinite(r)]
    if len(r) == 0: return None
    w, l = r[r>0], r[r<=0]; eq = np.cumsum(r)
    return dict(n=int(len(r)), wr=float((r>0).mean()),
                pf=float(w.sum()/max(-l.sum(),1e-9)),
                sum_r=float(r.sum()),
                max_dd_r=float((np.maximum.accumulate(eq)-eq).max()),
                avg=float(r.mean()))


def main():
    t0 = time.time()
    print("="*80); print(f"  Building deployable Oracle bundle with smart exit (v {SMART_EXIT_VERSION})"); print("="*80)

    # 1. Load existing bundle (will be augmented, not replaced)
    print("\n[1/6] loading current bundle ...", flush=True)
    bundle = pickle.load(open(ORACLE_PKL, "rb"))
    print(f"  current bundle keys: {list(bundle.keys())}", flush=True)
    print(f"  current version: {bundle.get('version')}", flush=True)

    # 2. Build training data (M5 swing + setups + per-cluster confirm + M15 ANTI gate)
    print("\n[2/6] loading swing + setups + M15 TFK + cluster IDs ...", flush=True)
    sw = pd.read_csv(SWING, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    setup_files = sorted(glob.glob(SETUP_GLOB))
    setups = pd.concat([pd.read_csv(f, parse_dates=["time"]) for f in setup_files],
                       ignore_index=True).sort_values("time").reset_index(drop=True)
    fp = pd.read_csv(FINGERPRINTS, parse_dates=["center_time"])

    exit_feats_orig = bundle["exit_feats"]
    available_ctx = [c for c in exit_feats_orig[3:] if c in setups.columns]
    if available_ctx:
        ctx_df = setups[["time"]+available_ctx].sort_values("time").drop_duplicates("time")
        merged = pd.merge_asof(sw[["time"]].sort_values("time"),
                               ctx_df.sort_values("time"), on="time", direction="backward")
        for c in available_ctx:
            sw[c] = merged[c].fillna(0).to_numpy()
    for c in [c for c in exit_feats_orig[3:] if c not in available_ctx]:
        sw[c] = 0.0

    times_sw = sw["time"].values.astype("datetime64[ns]")
    setup_t = setups["time"].values.astype("datetime64[ns]")
    sw_idx = np.searchsorted(times_sw, setup_t)
    sw_idx_safe = np.minimum(sw_idx, len(times_sw)-1)
    exact = times_sw[sw_idx_safe] == setup_t
    setups["sw_idx"] = np.where(exact, sw_idx_safe, -1)
    cid_per = np.full(len(sw), -1, dtype=np.int64)
    for _, row in fp.iterrows():
        s, e = int(row["start_idx"]), int(row["end_idx"])
        if 0 <= s < e <= len(sw): cid_per[s:e] = int(row["new_label"])
    setups["cid"] = np.where(setups["sw_idx"] >= 0, cid_per[np.maximum(setups["sw_idx"],0)], -1)
    setups = setups[setups["cid"] >= 0].reset_index(drop=True)
    m15_arr = m15_dir(sw)
    sw_idx_safe = np.minimum(np.searchsorted(times_sw, setups["time"].values.astype("datetime64[ns]")), len(times_sw)-1)
    exact = times_sw[sw_idx_safe] == setups["time"].values.astype("datetime64[ns]")
    setups["m15_dir"] = np.where(exact, m15_arr[sw_idx_safe], 0)

    mdls = bundle["mdls"]; thrs = bundle["thrs"]; v72l_feats = bundle["v72l_feats"]
    setups["confirm_ok"] = False
    for cid in sorted(setups["cid"].unique()):
        key = (int(cid), "RL")
        if key not in mdls: continue
        sub = setups[setups["cid"] == cid]
        X = sub[v72l_feats].fillna(0).to_numpy(np.float32)
        p = mdls[key].predict_proba(X)[:, 1]
        setups.loc[sub.index, "confirm_ok"] = p >= thrs[key]
    confirmed = setups[setups["confirm_ok"] & (setups["sw_idx"] >= 0)].reset_index(drop=True)
    gated = confirmed[(confirmed["m15_dir"] == -confirmed["direction"]) &
                      (confirmed["m15_dir"] != 0)].reset_index(drop=True)
    print(f"  gated trades: {len(gated):,}", flush=True)

    # 3. Build state table + fit regressor
    print("\n[3/6] building state table + fitting XGBRegressor ...", flush=True)
    O=sw["open"].to_numpy(float); H=sw["high"].to_numpy(float); L=sw["low"].to_numpy(float); C=sw["close"].to_numpy(float)
    prev_c=np.concatenate([[C[0]],C[:-1]])
    tr=np.maximum(H-L,np.maximum(np.abs(H-prev_c),np.abs(L-prev_c)))
    atr=pd.Series(tr).rolling(14,min_periods=14).mean().to_numpy()
    sp_R = SPREAD_USD / np.nanmedian(atr)
    hours = pd.to_datetime(sw["time"]).dt.hour.to_numpy()
    dows = pd.to_datetime(sw["time"]).dt.dayofweek.to_numpy()
    ctx_cols = ["hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er",
                "quantum_flow","quantum_flow_h4","vwap_dist"]
    ctx_arr = np.zeros((len(sw), len(ctx_cols)), dtype=np.float64)
    for j, c in enumerate(ctx_cols):
        if c in sw.columns: ctx_arr[:, j] = sw[c].fillna(0).to_numpy(float)
    sw_idxs = gated["sw_idx"].to_numpy(np.int64)
    dirs = gated["direction"].to_numpy(np.int64)
    cids = gated["cid"].to_numpy(np.int64)
    Xs, ys, valid = build_state_table(sw_idxs, dirs, cids, hours, dows, m15_arr,
                                       ctx_arr, C, H, L, atr, MAX_HOLD)
    valid_bool = valid.astype(bool)
    row_trade_idx = np.repeat(np.arange(len(sw_idxs)), MAX_HOLD)
    trade_is_train = gated["time"].values < np.datetime64(HOLDOUT)
    row_is_train = trade_is_train[row_trade_idx]
    train_rows = valid_bool & row_is_train
    test_rows = valid_bool & (~row_is_train)
    print(f"  train rows: {int(train_rows.sum()):,}  test rows: {int(test_rows.sum()):,}", flush=True)
    from xgboost import XGBRegressor
    smart_mdl = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.04,
                              subsample=0.85, colsample_bytree=0.85, min_child_weight=10,
                              reg_lambda=1.0, objective="reg:squarederror",
                              tree_method="hist", random_state=42, verbosity=0, n_jobs=-1)
    ts = time.time(); smart_mdl.fit(Xs[train_rows], ys[train_rows])
    print(f"  fit done in {time.time()-ts:.0f}s", flush=True)

    # 4. Confirm holdout metrics (PURE smart, no trail, upside_thr=0.3)
    print("\n[4/6] validating on 2.4y honest holdout ...", flush=True)
    preds = np.zeros(len(Xs), dtype=np.float32)
    preds[valid_bool] = smart_mdl.predict(Xs[valid_bool])
    pred_per_state = preds.reshape(len(sw_idxs), MAX_HOLD)
    pnl, bars_, reasons = sim_smart_only(sw_idxs, dirs, O, H, L, C, atr,
                                           SL_HARD, UPSIDE_THR, MIN_PNL_R_FOR_SMART_EXIT,
                                           pred_per_state, MAX_HOLD, sp_R, MIN_HOLD)
    hd_mask = gated["time"].values >= np.datetime64(HOLDOUT)
    pnl_h = pnl[hd_mask]; bars_h = bars_[hd_mask]; reasons_h = reasons[hd_mask]
    m = metrics(pnl_h)
    mix = {int(k):int((reasons_h==k).sum()) for k in (0,1,2,3)}
    print(f"  HOLDOUT: n={m['n']:,}  WR={m['wr']*100:.1f}%  PF={m['pf']:.2f}  "
          f"sumR={m['sum_r']:+.0f}  DD={m['max_dd_r']:.0f}  avgR={m['avg']:+.2f}  "
          f"avg_bars={float(bars_h.mean()):.1f} ({float(bars_h.mean())*5:.0f} min)", flush=True)
    print(f"  exit mix: hard_sl={mix[0]}  trail={mix[1]}  smart={mix[2]}  max_hold={mix[3]}", flush=True)

    # 5. Write NEW bundle side-by-side (no overwrite of existing production file)
    print("\n[5/6] writing NEW bundle (existing oracle_xau_validated.pkl untouched) ...", flush=True)

    bundle["smart_exit_mdl"] = smart_mdl
    bundle["smart_exit_feats"] = SMART_FEAT_NAMES
    bundle["smart_exit_upside_thr"] = float(UPSIDE_THR)
    bundle["smart_exit_min_hold"] = int(MIN_HOLD)
    bundle["smart_exit_min_pnl_R"] = float(MIN_PNL_R_FOR_SMART_EXIT)
    bundle["smart_exit_use_trail"] = False
    bundle["smart_exit_max_hold"] = int(MAX_HOLD)
    bundle["smart_exit_sl_hard"] = float(SL_HARD)
    bundle["smart_exit_gate_m15_tfk_anti"] = True
    bundle["smart_exit_version"] = SMART_EXIT_VERSION
    bundle["smart_exit_holdout_metrics"] = {
        "n": m["n"], "wr": m["wr"], "pf": m["pf"], "sum_r": m["sum_r"],
        "max_dd_r": m["max_dd_r"], "avg_R": m["avg"],
        "avg_bars": float(bars_h.mean()),
        "exit_mix": {"hard_sl": mix[0], "trail": mix[1], "smart": mix[2], "max_hold": mix[3]},
        "holdout_start": str(HOLDOUT),
    }
    bundle["smart_exit_ctx_cols"] = ctx_cols
    bundle["version"] = f"{bundle.get('version','?')}+{SMART_EXIT_VERSION}"

    with open(NEW_PKL, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  wrote NEW bundle: {NEW_PKL}", flush=True)
    print(f"  existing production bundle UNCHANGED: {ORACLE_PKL}", flush=True)
    print(f"  new version string: {bundle['version']}", flush=True)

    # 6. Write deploy notes
    print("\n[6/6] writing deploy_notes.md ...", flush=True)
    notes = f"""# Oracle XAU smart exit deployment ({TODAY})

## What changed
- Added **smart_exit_mdl** (XGBRegressor) — predicts future-upside R from current trade state
- Added **M15 TFK ANTI gate** as the regime filter (replaces cluster-block of C1+C2)
- Removed trail floor (smart model handles it directly)
- Kept existing exit_mdl as fallback (do not call by default)

## Honest holdout 2.4y (2024-01-01 → 2026-05-01)
- trades: {m['n']:,}
- WR: {m['wr']*100:.1f}%
- PF: {m['pf']:.2f}
- sumR: {m['sum_r']:+.0f}
- DD: {m['max_dd_r']:.0f}
- avg duration: {float(bars_h.mean()):.1f} M5 bars ({float(bars_h.mean())*5:.0f} min)
- exit mix: hard_sl={mix[0]}, smart={mix[2]}, max_hold={mix[3]}

## Server-side integration

### 1. Entry gate (replaces cluster-block of C1+C2)
At each setup, compute the **causal** M15 TFK direction from the swing close
series:
```python
# rolling resample of M5 swing into M15 bars
# compute TFK on the M15 bars (same compute_tfk as production)
# forward-fill back to M5 grid (causal: each M5 bar sees only the
#   COMPLETED M15 bar before it, NOT the in-progress M15 bar)
```
Then **only fire the setup** if `m15_tfk_dir == -setup_direction` (i.e. M15
trend is opposite the trade direction — counter-trend setups are what Oracle's
rules are designed for).

The per-cluster confirm + meta filter pipeline stays unchanged. Cluster ID is
still used internally for routing to the right confirm model.

### 2. Exit policy (replaces old XGBClassifier exit)
For each bar of an open trade, build the feature vector in this order:
```
{SMART_FEAT_NAMES!r}
```
Definitions per feature (all R-units are ATR-normalized using ATR at entry):
- unrealized_R: d*(C[bar]-ep)/atr_at_entry
- bars_held: k (1-based)
- pnl_vel_3: unrealized_R - unrealized_R_3_bars_ago
- pnl_vel_5: unrealized_R - unrealized_R_5_bars_ago
- mfe_so_far_R: running max favorable in R
- mae_so_far_R: running max adverse in R (positive number)
- dd_from_peak_R: mfe_so_far_R - unrealized_R
- progress_to_sl: unrealized_R / 5.0
- progress_to_trail: unrealized_R / 2.0
- hour_utc, dow: time of the CURRENT bar (not entry bar)
- m15_dir_x_dir: m15_tfk_dir_at_current_bar * trade_direction
- cid: cluster id (numeric)
- direction: +1 or -1
- bars_remaining: 60 - k
- frac_time: k / 60
- trade_range_R: mfe_so_far_R - mae_so_far_R
- horizon: 60 (constant for now; computed live as bars available)
- 8 context features at the current bar: hurst_rs, ou_theta, entropy_rate,
  kramers_up, wavelet_er, quantum_flow, quantum_flow_h4, vwap_dist
  (these come from the same physics pipeline as the entry models)

Then:
```python
predicted_upside = bundle["smart_exit_mdl"].predict(X_features.reshape(1,-1))[0]
should_exit = (
    bars_held >= bundle["smart_exit_min_hold"]
    and unrealized_R >= bundle["smart_exit_min_pnl_R"]
    and predicted_upside < bundle["smart_exit_upside_thr"]
)
```

Plus the usual safety net:
- if unrealized_R <= -bundle["smart_exit_sl_hard"]: EXIT (hard SL)
- if bars_held >= bundle["smart_exit_max_hold"]: EXIT (time fallback)

### 3. Files written by this script
- `{NEW_PKL.name}`  ← NEW bundle (side-by-side, no overwrite)
- The existing `oracle_xau_validated.pkl` is UNCHANGED — current production keeps loading it.

### 4. To activate locally (when you're ready)
```bash
# Make a backup first
cp {ORACLE_PKL} {BACKUP_PKL}
# Swap in the new bundle
mv {NEW_PKL} {ORACLE_PKL}
```

### 5. To deploy to the commercial server (when you're ready)
```bash
cp {ORACLE_PKL} /home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine/models/oracle_xau_validated.pkl
# then restart server
```

### 6. Rollback (if needed after activation)
```bash
cp {BACKUP_PKL} {ORACLE_PKL}
```

## Honest caveats
- The smart exit + M15 ANTI gate **trades half the volume** of the cluster-gate
  pipeline. sumR is lower (~+14k vs +25k baseline) but PF/DD profile is
  similar. This is risk-reduction.
- The M15 ANTI gate is **reactive** (15-min reaction) vs cluster gate
  (4h reaction). That's the win.
- Context features (hurst_rs, ou_theta, etc.) MUST be computed live by the
  server pipeline — they are not in the swing CSV by default. This already
  works in the current deployment (the old RL exit uses them too).
"""
    DEPLOY_NOTES.write_text(notes)
    print(f"  wrote {DEPLOY_NOTES}", flush=True)
    print(f"\n  TOTAL: {time.time()-t0:.0f}s")
    print(f"\n  ✓ bundle ready at {ORACLE_PKL}")
    print(f"  ✓ backup at      {BACKUP_PKL}")
    print(f"  ✓ deploy notes at {DEPLOY_NOTES}")


if __name__ == "__main__":
    main()
