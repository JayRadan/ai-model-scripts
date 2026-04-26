"""
Experiment: Oracle XAU holdout with a C4 (HighVol) direction-filter —
only SHORT entries allowed when the regime is HighVol. Everything else
(other clusters, other rules, meta threshold, exit head, SL/TP geometry)
is left exactly as the validated Oracle pipeline.

Compare against the baseline Oracle holdout (PF 3.96 / WR 68.5%).

Run:
    python experiments/v72_lite_deploy/99_exp_c4_shorts_only.py
"""
from __future__ import annotations
import sys, os, time as _time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "val", os.path.join(THIS_DIR, "01_validate_v72_lite.py"))
val = importlib.util.module_from_spec(spec); spec.loader.exec_module(val)


C4_CID = 4      # HighVol
ALLOW_LONGS_IN_C4 = False     # experiment switch


def filter_c4_shorts(setups: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Drop C4 entries with direction == +1 (i.e. longs)."""
    if len(setups) == 0: return setups
    before = len(setups)
    c4_long_mask = (setups["cid"] == C4_CID) & (setups["direction"].astype(int) == +1)
    kept = setups[~c4_long_mask].reset_index(drop=True)
    dropped = before - len(kept)
    print(f"  [{tag}] dropped {dropped} C4-long entries "
          f"({before} → {len(kept)}, kept {len(kept)/max(before,1):.1%})")
    return kept


def main():
    print("="*80)
    print("EXPERIMENT: Oracle XAU holdout — shorts-only in HighVol (C4)")
    print("="*80)

    train, test = val.load_and_split()
    swing, atr = val.load_swing_with_physics()

    # Stage A — train confirmations + exit (identical to baseline)
    print("\n[1/4] Training confirmations + exit (identical to baseline)...")
    t0 = _time.time()
    mdls, thrs = val.train_conf(train, val.V72L_FEATS, "v72l-conf")
    tc = val.confirm(train, mdls, thrs, val.V72L_FEATS)
    exit_mdl = val.train_exit(tc, swing, atr)
    print(f"  train+exit: {_time.time()-t0:.0f}s, {len(tc):,} train confirms")

    # Stage B — meta head + threshold (identical to baseline)
    print("\n[2/4] Meta head + threshold pick (identical to baseline)...")
    tt = val.simulate(tc, swing, atr, exit_mdl)
    tc2 = tc.copy()
    tc2["direction"] = tc2["direction"].astype(int)
    tc2["cid"]       = tc2["cid"].astype(int)
    md = tt.merge(tc2[["time","cid","rule"] + list(val.V72L_FEATS)],
                  on=["time","cid","rule"], how="left")
    md["meta_label"] = (md["pnl_R"] > 0).astype(int)
    md = md.sort_values("time").reset_index(drop=True)
    s = int(len(md) * 0.80); mtr, mvd = md.iloc[:s], md.iloc[s:]
    meta_mdl = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", verbosity=0)
    meta_mdl.fit(mtr[val.META_FEATS].fillna(0).values, mtr["meta_label"].values)
    pv = meta_mdl.predict_proba(mvd[val.META_FEATS].fillna(0).values)[:, 1]
    yv = mvd["meta_label"].values; pn = mvd["pnl_R"].values
    baseline_pf = pn[pn>0].sum() / max(-pn[pn<=0].sum(), 1e-9)
    cands = []
    for thr in np.arange(0.40, 0.80, 0.025):
        m = pv >= thr
        if m.sum() < 50: continue
        wr = yv[m].mean()
        pt = pn[m]; pf = pt[pt>0].sum() / max(-pt[pt<=0].sum(), 1e-9)
        rt = m.sum() / len(yv)
        cands.append((thr, wr, pf, m.sum(), rt))
    valid = [c for c in cands if c[2] >= baseline_pf * 0.95 and c[4] >= 0.50] or \
            [c for c in cands if c[2] >= baseline_pf * 0.90] or cands
    valid.sort(key=lambda c: (-c[1], -c[2]))
    best_thr = valid[0][0]
    print(f"  meta threshold: {best_thr:.3f}")

    # Stage C — run the holdout twice: baseline vs experiment
    print("\n[3/4] Running holdout WITHOUT filter (baseline)...")
    tec = val.confirm(test, mdls, thrs, val.V72L_FEATS)
    tec["direction"] = tec["direction"].astype(int); tec["cid"] = tec["cid"].astype(int)
    pm = meta_mdl.predict_proba(tec[val.META_FEATS].fillna(0).values)[:, 1]
    tec_m = tec[pm >= best_thr].copy()
    print(f"  after meta filter: {len(tec_m):,} setups")
    base_trades = val.simulate(tec_m, swing, atr, exit_mdl)
    r_base = val.report(base_trades, "Oracle BASELINE (no C4 filter)")

    print("\n[4/4] Running holdout WITH C4-shorts-only filter (experiment)...")
    tec_m_exp = filter_c4_shorts(tec_m, "post-meta")
    exp_trades = val.simulate(tec_m_exp, swing, atr, exit_mdl)
    r_exp = val.report(exp_trades, "Oracle EXPERIMENT (C4: shorts only)")

    # Stage D — side-by-side summary + per-cluster breakdown on C4 only
    print("\n" + "="*80); print("SIDE-BY-SIDE"); print("="*80)
    if r_base and r_exp:
        print(f"  BASELINE      : n={r_base['n']:<5} WR={r_base['wr']:.1%} PF={r_base['pf']:.2f} "
              f"DD=-{r_base['dd']:.1f}  Total={r_base['total']:+.1f}R")
        print(f"  C4-SHORTS-ONLY: n={r_exp['n']:<5} WR={r_exp['wr']:.1%} PF={r_exp['pf']:.2f} "
              f"DD=-{r_exp['dd']:.1f}  Total={r_exp['total']:+.1f}R")
        d_n  = r_exp['n']     - r_base['n']
        d_pf = r_exp['pf']    - r_base['pf']
        d_wr = r_exp['wr']    - r_base['wr']
        d_r  = r_exp['total'] - r_base['total']
        print(f"  Δ             : Δn={d_n:+d}  ΔWR={d_wr:+.1%}  ΔPF={d_pf:+.2f}  ΔR={d_r:+.1f}")

    # C4-only zoom
    print("\nC4 (HighVol) only:")
    for df, tag in [(base_trades, "BASE    "), (exp_trades, "EXP(-L) ")]:
        c4 = df[df["cid"] == C4_CID]
        if len(c4) == 0:
            print(f"  {tag}: no C4 trades"); continue
        w = c4[c4["pnl_R"] > 0]; l = c4[c4["pnl_R"] <= 0]
        pf = w["pnl_R"].sum() / max(-l["pnl_R"].sum(), 1e-9)
        longs  = c4[c4["direction"] == +1]
        shorts = c4[c4["direction"] == -1]
        print(f"  {tag}: n={len(c4)} ({len(longs)}L / {len(shorts)}S)  "
              f"WR={len(w)/len(c4):.1%}  PF={pf:.2f}  Total={c4['pnl_R'].sum():+.1f}R")

    # Save
    out = "/home/jay/Desktop/new-model-zigzag/data/v72l_exp_c4_shorts_only.csv"
    exp_trades.to_csv(out, index=False)
    print(f"\n  Saved experiment trades to {out}")


if __name__ == "__main__":
    main()
