"""
Replay today's BTCUSD M5 bars through the FULL Python v7.2-lite inference
pipeline and print every rule candidate + per-rule confirmation probability +
meta-label probability. Output is formatted to be directly comparable to the
MQL5 EA log lines like:

  V7 META reject: rule=C2_R0d_squeeze dir=-1 cid=2 P_conf=0.407 P_win=0.399 (thr=0.525)

Inputs:
  Wine MT5 CSV: MQL5/Files/swing_v5_btc_today.csv   (exporter-20-math output)
  ONNX models:  models/confirm_v7_btc_*.onnx, models/meta_v7_btc.onnx
  Regime JSON:  data/regime_selector_btc_K5.json
  Threshold:    experiments/v72_lite_btc_deploy/meta_threshold_v72l_btc.txt
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
import onnxruntime as ort

ROOT = "/home/jay/Desktop/new-model-zigzag"
sys.path.insert(0, ROOT + "/model_pipeline")
sys.path.insert(0, ROOT + "/experiments/v72_lite_btc_deploy")

from importlib.machinery import SourceFileLoader
tech = SourceFileLoader("labeler_v4", ROOT + "/model_pipeline/01_labeler_v4.py").load_module()
phys = SourceFileLoader("physics_btc", ROOT + "/experiments/v72_lite_btc_deploy/04b_compute_physics_features_btc.py").load_module()
step1 = SourceFileLoader("step1_btc", ROOT + "/experiments/v72_lite_btc_deploy/00_compute_v72l_features_step1_btc.py").load_module()
setup = SourceFileLoader("setup_btc", ROOT + "/experiments/v72_lite_btc_deploy/04_build_setup_signals_btc.py").load_module()

CSV = os.path.expanduser(
    "~/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Files/swing_v5_btc_today.csv")

OLD_FEATS = [
    "hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er",
    "vwap_dist","hour_enc","dow_enc",
    "quantum_flow","quantum_flow_h4","quantum_momentum","quantum_vwap_conf",
    "quantum_divergence","quantum_div_strength",
]
V72L_EXTRA = ["vpin","sig_quad_var","har_rv_ratio","hawkes_eta"]
V72L_FEATS = OLD_FEATS + V72L_EXTRA                     # 18
META_FEATS = V72L_FEATS + ["direction","cid"]           # 20

CLUSTER_NAMES = {0:"Uptrend", 1:"MeanRevert", 2:"TrendRange", 3:"Downtrend", 4:"HighVol"}


def load_csv():
    raw = open(CSV, "rb").read()
    txt = raw.decode("utf-16-le").lstrip("\ufeff")
    import io
    df = pd.read_csv(io.StringIO(txt))
    df["time"] = pd.to_datetime(df["time"], format="%Y.%m.%d %H:%M")
    df = df.sort_values("time").reset_index(drop=True)
    print(f"  Loaded {len(df):,} bars  {df['time'].iat[0]} → {df['time'].iat[-1]}")
    return df


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    vol = np.maximum(df["spread"].values.astype(np.float64), 1.0)

    # Tech features (rsi/stoch/bb/mom/ll_dist/hh_dist/atr_ratio/etc.)
    df = tech.compute_tech_features(df)

    # Physics features (4b: hurst, ou_theta, entropy, kramers, wavelet, vwap,
    #                       hour_enc, dow_enc, quantum_* M5 + H4)
    tr = np.concatenate([[h[0]-l[0]],
          np.maximum.reduce([h[1:]-l[1:], np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    ret = np.concatenate([[0.0], np.diff(np.log(c))])
    df["hurst_rs"]     = phys.compute_hurst_rs(ret)
    df["ou_theta"]     = phys.compute_ou_theta(ret)
    df["entropy_rate"] = phys.compute_entropy(ret)
    df["kramers_up"]   = phys.compute_kramers_up(c)
    df["wavelet_er"]   = phys.compute_wavelet_er(c)
    df["vwap_dist"]    = phys.compute_vwap_dist(df, atr)
    hour = df["time"].dt.hour.astype(float)
    dow  = df["time"].dt.dayofweek.astype(float)
    df["hour_enc"] = np.sin(2*np.pi*hour/24)
    df["dow_enc"]  = np.sin(2*np.pi*dow/5)
    df["quantum_flow"] = phys.compute_quantum_flow(o, h, l, c, vol)
    df_h4 = df.set_index("time")[["open","high","low","close","spread"]].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last","spread":"sum"}).dropna()
    qf_h4 = phys.compute_quantum_flow(df_h4["open"].values, df_h4["high"].values,
                                       df_h4["low"].values, df_h4["close"].values,
                                       np.maximum(df_h4["spread"].values, 1.0))
    qf_h4_s = pd.Series(qf_h4, index=df_h4.index).shift(1)
    df["quantum_flow_h4"] = qf_h4_s.reindex(df.set_index("time").index, method="ffill").values
    df["quantum_momentum"] = np.concatenate([[0.0], np.diff(df["quantum_flow"].values)])
    vwap_dir = np.sign(c - np.where(atr > 1e-10, c - df["vwap_dist"].values * atr, c))
    qf_sign  = np.sign(df["quantum_flow"].values)
    df["quantum_vwap_conf"] = qf_sign * vwap_dir
    qf_m5 = df["quantum_flow"].values
    qf_h4v = df["quantum_flow_h4"].values
    n = len(df)
    div = np.zeros(n)
    div[(qf_h4v > 0) & (qf_m5 < 0)] = +1.0
    div[(qf_h4v < 0) & (qf_m5 > 0)] = -1.0
    df["quantum_divergence"] = div
    df["quantum_div_strength"] = np.where(div != 0, np.abs(qf_h4v), 0.0)

    # v7.2-lite extras (step=1, smoothed)
    vpin_raw = step1.compute_vpin(h, l, c)
    qv_raw   = step1.compute_sig_quad_var(c)
    har_raw  = step1.compute_har_rv_ratio(c)
    eta_raw  = step1.compute_hawkes_eta(c)
    smooth = lambda a: pd.Series(a).rolling(6, min_periods=1).mean().values
    df["vpin"]         = smooth(np.where(np.isfinite(vpin_raw), vpin_raw, 0.0))
    df["sig_quad_var"] = smooth(np.where(np.isfinite(qv_raw),   qv_raw,   0.0))
    df["har_rv_ratio"] = smooth(np.where(np.isfinite(har_raw),  har_raw,  0.0))
    df["hawkes_eta"]   = smooth(np.where(np.isfinite(eta_raw),  eta_raw,  0.0))

    df = df.fillna(0)
    return df


def scan_c2_rules(df: pd.DataFrame) -> list[dict]:
    """Only scan the 4 C2 rules armed in the BTC EA when regime==C2.
    These are the SAME Python rule definitions used to train the ONNX models
    (from 04_build_setup_signals_btc.py), so any divergence vs MQL5 is a true
    MQL5↔Python rule-parity gap, not an artefact of my hand-coding."""
    events  = setup.rule_ranging_squeeze(df)                # R0d_squeeze
    events += setup.rule_ranging_nr4_breakout(df)           # R0e_nr4_break
    events += setup.rule_ranging_inside_break(df)           # R0g_inside_break
    events += setup.rule_ranging_three_bar_reversal(df)     # R0h_3bar_reversal
    return events


def load_onnx_sessions() -> dict:
    """Return dict keyed by the rule name used in MQL5 log output."""
    base = ROOT + "/models"
    mapping = {
        "R0d_squeeze":        "confirm_v7_btc_c2_R0d_squeeze.onnx",
        "R0e_nr4_break":      "confirm_v7_btc_c2_R0e_nr4_break.onnx",
        "R0g_inside_break":   "confirm_v7_btc_c2_R0g_inside_break.onnx",
        "R0h_3bar_reversal":  "confirm_v7_btc_c2_R0h_3bar_reversal.onnx",
    }
    sess = {}
    for rule, fname in mapping.items():
        p = os.path.join(base, fname)
        if os.path.exists(p):
            sess[rule] = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    meta = ort.InferenceSession(base + "/meta_v7_btc.onnx", providers=["CPUExecutionProvider"])
    return sess, meta


def predict_proba(session, feats: np.ndarray) -> float:
    """Return P(class=1) from a 1-row sklearn→ONNX binary classifier."""
    out = session.run(None, {"features": feats.reshape(1, -1).astype(np.float32)})
    # zipmap disabled → probabilities is a [1,2] float array
    probs = out[1]
    if isinstance(probs, list):
        # zipmap path: list[dict{0:p0, 1:p1}]
        return float(probs[0][1])
    return float(probs[0, 1])


def main():
    print("="*90)
    print("BTC v7.2-lite Python replay — comparing to MQL5 EA inference")
    print("="*90)
    df = load_csv()
    print("\n[1/3] Computing features (tech + physics + v7.2-lite extras)...")
    df = compute_all_features(df)

    print("\n[2/3] Scanning C2 TrendRange rules (same 4 the EA arms in C2)...")
    today = df["time"].dt.date.max()
    events = scan_c2_rules(df)
    events = [e for e in events if df["time"].iat[e["idx"]].date() == today]
    print(f"  {len(events)} rule candidates on {today}")

    if not events:
        print("\n  No C2 rule candidates fired today (Python side). Nothing to compare.")
        return

    print("\n[3/3] Running ONNX confirm + meta per candidate...")
    sessions, meta_sess = load_onnx_sessions()
    thr = float(open(ROOT + "/experiments/v72_lite_btc_deploy/meta_threshold_v72l_btc.txt").read().strip())
    print(f"  meta threshold: {thr}")

    # Per-rule confirmation thresholds — same ones the EA uses
    # (from confirmation_router_v7_btc.mqh). For C2 rules:
    PER_RULE_THR = {
        "R0d_squeeze":       0.40,
        "R0e_nr4_break":     0.50,
        "R0g_inside_break":  0.40,
        "R0h_3bar_reversal": 0.50,
    }

    print("\n" + "-"*108)
    print(f"  {'time (UTC)':<19} {'rule':<22} {'dir':>4} {'P_conf':>7} {'p-thr':>6} {'stg1':<6} {'P_win':>7} {'m-thr':>6}  {'verdict':<10}")
    print("-"*108)
    stage1_pass = 0; stage2_pass = 0
    for ev in events:
        i = ev["idx"]; d = ev["direction"]; rule = ev["rule"]
        t = df["time"].iat[i]
        feats = df.iloc[i][V72L_FEATS].values.astype(np.float32)
        sess = sessions.get(rule)
        if sess is None:
            print(f"  {t}  {rule:<22} {d:+d}   (no ONNX session found)")
            continue
        p_conf = predict_proba(sess, feats)
        p_thr = PER_RULE_THR.get(rule, 0.5)
        stg1 = "PASS" if p_conf >= p_thr else "FAIL"
        if stg1 == "FAIL":
            print(f"  {t}  {rule:<22} {d:+d}  {p_conf:>7.3f} {p_thr:>6.2f} {stg1:<6} {'-':>7} {thr:>6.3f}  REJECT-conf")
            continue
        stage1_pass += 1
        meta_in = np.concatenate([feats, [float(d), 2.0]])
        p_win = predict_proba(meta_sess, meta_in)
        verdict = "ACCEPT" if p_win >= thr else "REJECT-meta"
        if verdict == "ACCEPT": stage2_pass += 1
        print(f"  {t}  {rule:<22} {d:+d}  {p_conf:>7.3f} {p_thr:>6.2f} {stg1:<6} {p_win:>7.3f} {thr:>6.3f}  {verdict}")
    print("-"*108)
    print(f"  Total candidates: {len(events)}  Passed Stage-1 (per-rule): {stage1_pass}  Accepted (meta): {stage2_pass}")
    print("\nEA log for today (for quick visual diff):")
    print("  19:20 C2_R0d_squeeze       -1  P_conf=0.407  P_win=0.399  REJECT")
    print("  19:50 C2_R0d_squeeze       +1  P_conf=0.409  P_win=0.378  REJECT")
    print("  20:25 C2_R0g_inside_break  +1  P_conf=0.436  P_win=0.383  REJECT")
    print("  20:35 C2_R0g_inside_break  -1  P_conf=0.438  P_win=0.384  REJECT")


if __name__ == "__main__":
    main()
