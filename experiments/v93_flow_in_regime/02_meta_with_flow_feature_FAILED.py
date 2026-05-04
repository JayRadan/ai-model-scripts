"""v9.2 — Retrain Oracle XAU meta gate WITH flow_5m as 19th v72L feature
Using Dukascopy bars + real volume. Compare to vanilla Dukascopy retrain
(PF 3.61, WR 65.1%) which we just produced.

Method:
  1. Load Dukascopy XAU swing data
  2. Compute flow_5m on full bars (real volume from tick_volume column)
  3. Re-train confirm heads with V72L + flow_5m (19 features)
  4. Re-train meta head with META_FEATS + flow_5m (21 features = 18+dir+cid+flow)
  5. Simulate trades on holdout, compare to vanilla
"""
from __future__ import annotations
import os, sys, glob, time as _time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = "/home/jay/Desktop/new-model-zigzag"
sys.path.insert(0, ROOT + "/model_pipeline")
sys.path.insert(0, ROOT + "/experiments/v89_quantum_flow_tiebreaker")
sys.path.insert(0, ROOT + "/experiments/v72_lite_deploy")

# Reuse the indicator port + the validation simulator
from importlib.machinery import SourceFileLoader
qf  = SourceFileLoader("qf01", ROOT + "/experiments/v89_quantum_flow_tiebreaker/01_port_and_test.py").load_module()
val = SourceFileLoader("val01", ROOT + "/experiments/v72_lite_deploy/01_validate_v72_lite.py").load_module()

GLOBAL_CUTOFF = pd.Timestamp("2024-12-12 00:00:00")

V72L_FEATS = list(val.V72L_FEATS)               # 18 features
V72L_FEATS_PLUS = V72L_FEATS + ["flow_5m"]       # 19
META_FEATS_PLUS = V72L_FEATS_PLUS + ["direction", "cid"]   # 21


def load_setups_with_flow():
    print("loading swing + computing flow_5m on full Dukascopy bars...", flush=True)
    swing = pd.read_csv(ROOT + "/data/swing_v5_xauusd.csv", parse_dates=["time"])
    swing = swing.rename(columns={"tick_volume": "volume"}) if "tick_volume" in swing.columns else swing
    flow = qf.quantum_flow(swing[["open","high","low","close","volume"]])
    flow_df = pd.DataFrame({"time": swing["time"].values, "flow_5m": flow.values})
    print(f"  swing bars={len(swing):,} flow med={float(np.nanmedian(flow)):.2f}", flush=True)

    print("loading setups + merging flow...", flush=True)
    setups = []
    for f in sorted(glob.glob(ROOT + "/data/setups_*_v72l.csv")):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        setups.append(df)
    setups = pd.concat(setups, ignore_index=True).sort_values("time").reset_index(drop=True)
    setups["cid"] = setups["cid"].astype(int)
    setups["direction"] = setups["direction"].astype(int)
    setups = setups.loc[:, ~setups.columns.duplicated()]
    setups = setups.merge(flow_df, on="time", how="left")
    setups["flow_5m"] = setups["flow_5m"].fillna(0)
    return setups


def main():
    t_total = _time.time()
    setups = load_setups_with_flow()

    train = setups[setups["time"] < GLOBAL_CUTOFF].reset_index(drop=True)
    test  = setups[setups["time"] >= GLOBAL_CUTOFF].reset_index(drop=True)
    print(f"train: {len(train):,}  test: {len(test):,}", flush=True)

    print("\n[1] Training confirm heads (with flow_5m as 19th feat)...", flush=True)
    t0 = _time.time()
    mdls, thrs = val.train_conf(train, V72L_FEATS_PLUS, "v72l-conf+flow")
    print(f"  conf train {_time.time()-t0:.0f}s", flush=True)
    tc = val.confirm(train, mdls, thrs, V72L_FEATS_PLUS)
    print(f"  {len(tc):,} confirmed train setups", flush=True)

    print("\n[2] Loading swing+physics for exit/sim...", flush=True)
    swing, atr = val.load_swing_with_physics()

    t0 = _time.time()
    exit_mdl = val.train_exit(tc, swing, atr)
    print(f"  exit train {_time.time()-t0:.0f}s", flush=True)

    print("\n[3] Sim train trades for meta labels...", flush=True)
    t0 = _time.time()
    tt = val.simulate(tc, swing, atr, exit_mdl)
    print(f"  simulated {len(tt):,} train trades in {_time.time()-t0:.0f}s", flush=True)
    tc["direction"] = tc["direction"].astype(int); tc["cid"] = tc["cid"].astype(int)
    md = tt.merge(tc[["time","cid","rule"] + V72L_FEATS_PLUS],
                   on=["time","cid","rule"], how="left")
    md["meta_label"] = (md["pnl_R"] > 0).astype(int)
    md = md.sort_values("time").reset_index(drop=True)
    s = int(len(md) * 0.80)
    mtr, mvd = md.iloc[:s], md.iloc[s:]
    meta_mdl = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", verbosity=0)
    meta_mdl.fit(mtr[META_FEATS_PLUS].fillna(0).values, mtr["meta_label"].values)
    pv = meta_mdl.predict_proba(mvd[META_FEATS_PLUS].fillna(0).values)[:, 1]
    yv = mvd["meta_label"].values; pn = mvd["pnl_R"].values
    base_pf = pn[pn>0].sum() / max(-pn[pn<=0].sum(), 1e-9)
    print(f"  meta-validation baseline: n={len(yv)} WR={yv.mean():.1%} PF={base_pf:.2f}", flush=True)

    # Threshold sweep — pick best (mirrors validate script)
    cands = []
    for thr in np.arange(0.40, 0.80, 0.025):
        m = pv >= thr
        if m.sum() < 50: continue
        wr = yv[m].mean(); pt = pn[m]
        pf = pt[pt>0].sum() / max(-pt[pt<=0].sum(), 1e-9)
        rt = m.sum() / len(yv)
        cands.append((thr, wr, pf, m.sum(), rt))
    valid = [c for c in cands if c[2] >= base_pf * 0.95 and c[4] >= 0.50] or \
            [c for c in cands if c[2] >= base_pf * 0.90] or cands
    valid.sort(key=lambda c: (-c[1], -c[2]))
    best_thr = valid[0][0]
    print(f"  selected meta threshold: {best_thr:.3f}", flush=True)

    # Feature importances
    fi = meta_mdl.feature_importances_
    flow_imp = float(fi[V72L_FEATS_PLUS.index("flow_5m") if "flow_5m" in V72L_FEATS_PLUS else -1])
    print(f"  flow_5m importance in meta: {flow_imp:.4f}  (rank {1+sorted(-fi).index(-fi[V72L_FEATS_PLUS.index('flow_5m')])}/{len(fi)})", flush=True)

    print("\n[4] Holdout backtest...", flush=True)
    tec = val.confirm(test, mdls, thrs, V72L_FEATS_PLUS)
    print(f"  v72l+flow confirmed in holdout: {len(tec):,}", flush=True)
    t72_no_meta = val.simulate(tec, swing, atr, exit_mdl)
    val.report(t72_no_meta, "v72l+flow NO meta")
    tec["direction"] = tec["direction"].astype(int); tec["cid"] = tec["cid"].astype(int)
    pm = meta_mdl.predict_proba(tec[META_FEATS_PLUS].fillna(0).values)[:, 1]
    tec_m = tec[pm >= best_thr].copy()
    print(f"  After meta filter: {len(tec_m):,}  (dropped {len(tec)-len(tec_m):,})", flush=True)
    t72_meta = val.simulate(tec_m, swing, atr, exit_mdl)
    r = val.report(t72_meta, "v72l+flow + META")

    print("\n" + "="*78)
    print(f"WITH flow_5m feature:  n={r['n']}  WR={r['wr']:.1%}  PF={r['pf']:.2f}  DD=-{r['dd']:.1f}  R={r['total']:+.0f}")
    print(f"WITHOUT (just done):   n=1010   WR=65.1%  PF=3.61  DD=-47.8  R=+3052")
    print(f"vs Eightcap (real V):  n=1619   WR=67.4%  PF=3.89  DD=-75    R=+5218")
    print(f"vs current production: n=1367   WR=65.3%  PF=3.48  DD=-67    R=+3817")
    print("="*78)

    print(f"\nTotal elapsed: {_time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
