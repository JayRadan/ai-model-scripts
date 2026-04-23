"""
Dump the DEPLOYED Oracle v7.2-lite holdout trades (post-meta-filter) to a JSON
that the /lab/oracle-coach page can consume.

Runs the exact same confirm → meta → simulate pipeline as 01_validate_v72_lite.py
and 05_gen_backtest_json.py, but saves the per-trade list (time, rule, direction,
entry, sl, tp, pnl, outcome, cluster) — the thing those scripts compute internally
but never persist.

Pairs with a bundled OHLC candle window from labeled_v4.csv so the coach can render
the full holdout chart + every deployed trade as a marker.
"""
from __future__ import annotations
import json, os, sys, gzip, importlib, pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from datetime import datetime, timezone

# Cache trained models so we don't retrain on re-runs
CACHE_PATH = "/tmp/oracle_deployed_pipeline_cache.pkl"

sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/experiments/v72_lite_deploy")
val = importlib.import_module("01_validate_v72_lite")

DST = "/home/jay/Desktop/my-agents-and-website/commercial/website/public/lab/oracle-coach-deployed.json"
META_THRESHOLD = 0.675
DOLLAR_PER_R = 0.51  # same constant 05_gen_backtest_json.py uses

BLURBS = {
    "R3a_pullback": "Pullback to the 20-SMA holding a higher-low — trend-continuation buy.",
    "R3b_higherlow": "Higher-low above last swing — long, stop below.",
    "R3c_breakpullback": "Break-and-retest of prior resistance — continuation long.",
    "R3d_oversold": "Oversold dip inside an uptrend — shallow dip buy.",
    "R3e_false_breakdown": "Failed breakdown — traps shorts, fuels next leg up.",
    "R3f_sma_bounce": "Bounce off rising MA — MA as support.",
    "R3g_three_green": "Three green closes in uptrend — momentum continuation.",
    "R3h_close_streak": "Close streak above average — momentum long.",
    "R3i_inside_break": "Inside-bar break up in uptrend — compression release.",
    "R1a_swinghigh": "Swing-high rejection — sellers defending, short.",
    "R1b_lowerhigh": "Lower-high confirms downtrend — short with stop above.",
    "R1c_bouncefade": "Counter-trend bounce into resistance — fade it.",
    "R1d_overbought": "Overbought in downtrend — sell signal, not reversal.",
    "R1f_sma_reject": "Falling MA rejects price — short the rejection.",
    "R1g_three_red": "Three red closes — momentum continuation short.",
    "R0a_bb": "Outer Bollinger tag in range — fade the extreme.",
    "R0b_stoch": "Stochastic extreme in range — mean-revert.",
    "R0c_doubletouch": "Second touch of range boundary — fade it.",
    "R0d_squeeze": "Bollinger squeeze — compression release incoming.",
    "R0e_nr4_break": "NR4 break — narrowest-range breakout.",
    "R0f_mean_revert": "Stretched from midline — pull back to mean.",
    "R0g_inside_break": "Inside-bar breakout of range — take the direction.",
    "R0i_close_extreme": "Close at range extreme — fade toward middle.",
    "R2b_v_reversal": "Shock regime V-reversal — sharp mean-revert.",
}


def main():
    import time as _time
    t0 = _time.time()

    if os.path.exists(CACHE_PATH):
        print(f"Loading cached models from {CACHE_PATH}...", flush=True)
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        mdls, thrs, exit_mdl, meta_mdl, train, test, swing, atr = (
            cache["mdls"], cache["thrs"], cache["exit_mdl"], cache["meta_mdl"],
            cache["train"], cache["test"], cache["swing"], cache["atr"],
        )
        print(f"  loaded in {_time.time()-t0:.0f}s", flush=True)
    else:
        print("Loading splits + physics...", flush=True)
        train, test = val.load_and_split()
        swing, atr = val.load_swing_with_physics()

        print(f"Training per-rule confirmation models ({_time.time()-t0:.0f}s elapsed)...", flush=True)
        mdls, thrs = val.train_conf(train, val.V72L_FEATS, "v72l-conf")
        tc = val.confirm(train, mdls, thrs, val.V72L_FEATS)
        exit_mdl = val.train_exit(tc, swing, atr)

        print(f"Simulating train trades for meta labels ({_time.time()-t0:.0f}s)...", flush=True)
        tt = val.simulate(tc, swing, atr, exit_mdl)
        tc["direction"] = tc["direction"].astype(int); tc["cid"] = tc["cid"].astype(int)
        md = tt.merge(tc[["time","cid","rule"] + val.V72L_FEATS], on=["time","cid","rule"], how="left")
        md["meta_label"] = (md["pnl_R"] > 0).astype(int)
        md = md.sort_values("time").reset_index(drop=True)
        s_idx = int(len(md) * 0.80)
        mtr = md.iloc[:s_idx]

        print(f"Training meta model ({_time.time()-t0:.0f}s, {len(mtr):,} rows × {len(val.META_FEATS)} feats)...", flush=True)
        meta_mdl = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="logloss", verbosity=0, n_jobs=-1)
        meta_mdl.fit(mtr[val.META_FEATS].fillna(0).values, mtr["meta_label"].values)
        print(f"  meta trained ({_time.time()-t0:.0f}s)", flush=True)

        print("Saving cache for faster re-runs...", flush=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"mdls": mdls, "thrs": thrs, "exit_mdl": exit_mdl, "meta_mdl": meta_mdl,
                         "train": train, "test": test, "swing": swing, "atr": atr}, f)

    print(f"Holdout: confirm → meta filter → simulate ({_time.time()-t0:.0f}s)...", flush=True)
    tec = val.confirm(test, mdls, thrs, val.V72L_FEATS)
    tec["direction"] = tec["direction"].astype(int); tec["cid"] = tec["cid"].astype(int)
    pm = meta_mdl.predict_proba(tec[val.META_FEATS].fillna(0).values)[:, 1]
    tec_m = tec[pm >= META_THRESHOLD].copy()
    print(f"  confirmed {len(tec):,} → after meta {len(tec_m):,}", flush=True)

    trades = val.simulate(tec_m, swing, atr, exit_mdl)
    trades["time"] = pd.to_datetime(trades["time"])
    trades = trades.sort_values("time").reset_index(drop=True)
    print(f"  {len(trades):,} deployed trades", flush=True)

    # Attach entry/sl/tp/rule via merge back to tec_m
    tec_key = tec_m[["time", "cid", "rule", "direction", "entry_price", "atr"]].copy()
    trades = trades.merge(tec_key, on=["time", "cid", "rule", "direction"], how="left")

    # Load OHLC window
    print("Loading OHLC bars for the holdout window...", flush=True)
    df = pd.read_csv("/home/jay/Desktop/new-model-zigzag/data/labeled_v4.csv",
                     usecols=["time", "open", "high", "low", "close"],
                     parse_dates=["time"])
    win_start = trades["time"].min() - pd.Timedelta(hours=6)
    win_end   = trades["time"].max() + pd.Timedelta(hours=6)
    df = df[(df["time"] >= win_start) & (df["time"] <= win_end)].reset_index(drop=True)
    print(f"  {len(df):,} bars in window {win_start} → {win_end}", flush=True)

    bars = []
    for _, r in df.iterrows():
        ts = int(r["time"].tz_localize("UTC").timestamp()) if r["time"].tzinfo is None else int(r["time"].timestamp())
        bars.append({"t": ts, "o": round(r["open"], 2), "h": round(r["high"], 2),
                     "l": round(r["low"], 2), "c": round(r["close"], 2)})

    time_to_idx = {b["t"]: i for i, b in enumerate(bars)}

    out_signals = []
    for _, r in trades.iterrows():
        ts = int(r["time"].tz_localize("UTC").timestamp()) if r["time"].tzinfo is None else int(r["time"].timestamp())
        bi = time_to_idx.get(ts)
        if bi is None:
            # Snap to nearest preceding bar if exact timestamp missing
            bi = max(0, int(np.searchsorted([b["t"] for b in bars], ts) - 1))
        direction = "long" if int(r["direction"]) == 1 else "short"
        entry = float(r.get("entry_price", 0.0))
        atr_v = float(r.get("atr", 1.0))
        sl = entry - atr_v if direction == "long" else entry + atr_v
        tp = entry + 2*atr_v if direction == "long" else entry - 2*atr_v
        pnl_r = float(r["pnl_R"])
        pnl_usd = pnl_r * DOLLAR_PER_R
        outcome = "win" if pnl_r > 0 else "loss"
        rule = r["rule"]
        why = BLURBS.get(rule, f"{rule} — Oracle deployed signal.")
        verb = "buy" if direction == "long" else "sell"
        narrative = f"{why} {verb.capitalize()} at {entry:.2f}, stop {sl:.2f}, target {tp:.2f}."
        out_signals.append({
            "bi": int(bi), "t": ts, "r": rule, "d": direction,
            "c": min(1.0, 0.55 + abs(pnl_r) * 0.05),
            "b": why, "n": narrative,
            "e": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
            "rg": int(r["cid"]), "o": outcome,
            "p": round(pnl_usd, 2),
        })

    payload = {
        "symbol": "XAUUSD", "timeframe": "M5",
        "window_start": str(win_start), "window_end": str(win_end),
        "bars": bars, "signals": out_signals,
        "note": "Deployed Oracle v7.2-lite (post-ML-confirmation + meta-label) — matches /backtest page stats.",
        "source": "01_validate_v72_lite pipeline, clean holdout Dec 2024 → Mar 2026",
        "meta_threshold": META_THRESHOLD,
        "dollar_per_r": DOLLAR_PER_R,
    }
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    sz = os.path.getsize(DST) / 1e6
    print(f"\nWrote {DST}: {sz:.2f} MB")

    with open(DST, "rb") as f, gzip.open(DST + ".gz", "wb", compresslevel=6) as g:
        g.write(f.read())
    print(f"Gzipped: {os.path.getsize(DST + '.gz')/1e6:.2f} MB")

    wins = sum(1 for s in out_signals if s["o"] == "win")
    total_pnl = sum(s["p"] for s in out_signals)
    print(f"\nDeployed stats: {len(out_signals)} trades, {wins} wins ({wins/len(out_signals)*100:.1f}% WR), "
          f"total PnL ${total_pnl:.2f} (0.01 lot)")


if __name__ == "__main__":
    main()
