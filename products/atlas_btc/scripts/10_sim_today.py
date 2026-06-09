"""
10_sim_today.py — Atlas BTC live-equivalent simulation for the current trading day

WHAT THIS DOES
==============
Pulls fresh Dukascopy BTC/USD M1 bars for the last 3 days, runs Atlas's
STRICT-candle-reversal entry rule + the deployed Q model, and reports
every trade Atlas BTC would have opened today plus its outcome.

DEPLOYED CONFIG MIRRORED (commercial/server/decision_engine/configs/atlas_btc.py)
================================================================================
  symbol_base:    BTCUSD
  q_thr:          2.0     (lowered 3.0 → 2.0 on 2026-06-09, paired with kage≥1.
                           kage≥1 + Q≥2.0 beats kage≥3 + Q≥3.0 on 8-month
                           holdout: +31% $, +11% DD. BTC 24/7 market sees the
                           most short-lived Kalman flips during sharp moves.)
  strong_body:    0.8 ATR (entry needs a "strong" previous candle: body >= 0.8 ATR)
  kf_age_min:     1       (lowered 3 → 1 on 2026-06-09. Short Kalman flips during
                           impulse moves now qualify as valid pullback entries.)
  require_both:   true    (close must be on the correct side of BOTH
                           the Kalman line AND the Hermes TFK line)
  sl_hard_atr:    6.0
  trail_atr:      2.0
  be_trigger_r:   0.5
  max_hold_bars:  300
  max_concurrent: 4 slots, switch_delta=0.5, cooldown=5 bars

ENTRY RULE (STRICT — both indicators in confluence + reversal candle)
=====================================================================
BUY when ALL of:
  • TFK regime is GREEN (cdir == +1)
  • Kalman regime is RED, age >= 3 bars (kdir == -1)
  • Previous bar was STRONG BEAR (body >= 0.8 ATR, negative direction)
  • Previous close BELOW Kalman line AND BELOW TFK line
  • Current bar prints GREEN (close > open)

SELL when ALL of:
  • TFK regime is RED (cdir == -1)
  • Kalman regime is GREEN, age >= 3 bars (kdir == +1)
  • Previous bar was STRONG BULL (body >= 0.8 ATR, positive direction)
  • Previous close ABOVE Kalman line AND ABOVE TFK line
  • Current bar prints RED (close < open)

LIMITATIONS — OFFLINE, NOT EXACTLY THE DEPLOYED SERVER
======================================================
  • Order-flow features are zeroed (live uses tick stream)
  • Regime gating not applied
For exact live trades, query the funnel_log via /decide/_log-dump.

USAGE
=====
    python3 products/atlas_btc/scripts/10_sim_today.py
"""
import importlib.util, sys, pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit

ROOT = Path("/home/jay/Desktop/new-model-zigzag")
SERVER = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(ROOT / "experiments/kalman_color_flip"))
sys.path.insert(0, str(ROOT / "products/hermes_xau"))
from kalman import compute_kalman, bars_in_regime_array
from tfk import compute_tfk
_spec = importlib.util.spec_from_file_location(
    "ofm1", ROOT / "products/_shared/m1_with_orderflow.py")
_of = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_of)
add_standard_features = _of.add_standard_features
FLOW = list(_of.FLOW_FEATS)
import dukascopy_python

# ── DEPLOYED CONFIG ────────────────────────────────────────────────────
SPREAD       = 0.30
SL_ATR       = 6.0
TRAIL_ATR    = 2.0
MAX_HOLD     = 300
BE_R         = 0.5
MAX_CONC     = 4
SWITCH_DELTA = 0.5
COOLDOWN     = 5
STRONG_ATR   = 0.8       # strong-body threshold for the prior bar
RMULT        = 1.0       # Kalman R multiplier
KF_AGE_MIN   = 1         # lowered 3 → 1 on 2026-06-09 (kage sweep)
Q_THR        = 2.0       # lowered 3.0 → 2.0 on 2026-06-09 (kage sweep)
R_to_USD     = 2.30      # 0.01 lot BTCUSD (broker-dependent)

BUNDLE_PATH = SERVER / "decision_engine/models/atlas_btc_validated.pkl"


def streak_up(v):
    n = len(v); o = np.zeros(n, np.int32); c = 0
    for i in range(1, n):
        c = c + 1 if v[i] > v[i - 1] else 0
        o[i] = c
    return o


# ── 1. Fetch 3 days of XAU/USD M1 ──────────────────────────────────────
end = datetime.now(timezone.utc)
start = end - timedelta(days=3)
print(f"[atlas_btc] Fetching BTC/USD M1 {start.isoformat()} → {end.isoformat()}")
df = dukascopy_python.fetch(instrument="BTC/USD", interval=dukascopy_python.INTERVAL_MIN_1,
                             offer_side=dukascopy_python.OFFER_SIDE_BID, start=start, end=end)
df = df.reset_index().rename(columns={"timestamp": "time"})
df["time"] = pd.to_datetime(df["time"]).dt.tz_convert("UTC").dt.tz_localize(None)
df = df.sort_values("time").reset_index(drop=True)
if "tick_volume" not in df.columns:
    vc = "volume" if "volume" in df.columns else [c for c in df.columns if "vol" in c.lower()][0]
    df["tick_volume"] = df[vc]
print(f"  fetched {len(df):,} bars  last bar: {df.time.iloc[-1]}")

# ── 2. Features (TFK + Kalman + Atlas-specific) ────────────────────────
tfk_df = compute_tfk(df)
kf = compute_kalman(df, r_mult=RMULT)
fdf = add_standard_features(kf)
for c in ["force", "velocity", "x_est", "regime_w", "trend_raw", "trend", "committed_dir"]:
    fdf[c] = tfk_df[c].to_numpy()
fdf["tfk_line"] = tfk_df["tfk_line"].to_numpy()

O = df["open"].to_numpy(float); H = df["high"].to_numpy(float)
L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
atr = fdf["atr14"].to_numpy(float)
kdir = fdf["kf_dir"].to_numpy(np.int64)
kline = fdf["kf_p"].to_numpy(float)
cdir = tfk_df["committed_dir"].to_numpy(np.int64)
tline = tfk_df["tfk_line"].to_numpy(float)
vel = fdf["f_velRaw"].to_numpy(float)
kage = bars_in_regime_array(kdir)
fdf["kf_regime_age"] = kage
fdf["bar_range_atr"] = (H - L) / np.maximum(atr, 1e-9)
fdf["dist_kf"] = np.where(atr > 0, (C - kline) / atr, 0.0)
fdf["dist_tfk"] = np.where(atr > 0, (C - tline) / atr, 0.0)
fdf["vel_up_streak"] = streak_up(vel)
body = C - O
body_atr = np.where(atr > 0, np.abs(body) / atr, 0.0)
fdf["body_atr"] = body_atr
sb = (body < 0) & (body_atr >= STRONG_ATR)  # strong bear bar
su = (body > 0) & (body_atr >= STRONG_ATR)  # strong bull bar
fdf["strong_bear_prev"] = np.concatenate([[False], sb[:-1]])
fdf["strong_bull_prev"] = np.concatenate([[False], su[:-1]])
n = len(fdf)
sp = SPREAD / np.nanmedian(atr)

# ── 3. STRICT candle reversal rule ──────────────────────────────────────
pbk = np.concatenate([[False], C[:-1] < kline[:-1]])  # prev close below Kalman
pak = np.concatenate([[False], C[:-1] > kline[:-1]])  # prev close above Kalman
pbt = np.concatenate([[False], C[:-1] < tline[:-1]])  # prev close below TFK
pat = np.concatenate([[False], C[:-1] > tline[:-1]])  # prev close above TFK
g = C > O  # green candle
r = C < O  # red candle
ok = np.isfinite(atr) & (atr > 0)
ok[:250] = False; ok[-(MAX_HOLD + 1):] = False
buy = ok & (cdir == 1) & (kdir == -1) & (fdf["strong_bear_prev"].to_numpy()) & pbk & pbt & g & (kage >= KF_AGE_MIN)
sell = ok & (cdir == -1) & (kdir == 1) & (fdf["strong_bull_prev"].to_numpy()) & pak & pat & r & (kage >= KF_AGE_MIN)
mask = buy | sell
idxs = np.where(mask)[0]
dirs = np.where(cdir[idxs] == 1, 1, -1).astype(np.int64)
print(f"  Atlas candidates over 3-day window: {len(idxs)}")

# ── 4. Q score, gate ────────────────────────────────────────────────────
bundle = pickle.load(open(BUNDLE_PATH, "rb"))
M = bundle["q_model"]; feat_cols = bundle["feat_cols"]
for f in FLOW:
    if f not in fdf.columns: fdf[f] = 0.0
Xall = fdf[feat_cols].fillna(0).to_numpy(np.float32)
q_live = M.predict(Xall[idxs])
print(f"  Q dist: median={np.median(q_live):.2f}  p75={np.quantile(q_live, 0.75):.2f}  max={q_live.max():.2f}")
print(f"  candidates with Q >= {Q_THR}: {int((q_live >= Q_THR).sum())}")

times = df["time"].to_numpy()

# ── 5. Multi-pos sim with BE, switch, cooldown ─────────────────────────
active = []; executed = []
last_open = {-1: -10**9, 1: -10**9}
info = {int(idxs[k]): (int(dirs[k]), float(q_live[k]))
        for k in range(len(idxs)) if q_live[k] >= Q_THR}
if not info:
    print("  no Atlas candidates passed q_thr filter")
    sys.exit(0)
b0 = min(info.keys()); b1 = min(max(info.keys()) + MAX_HOLD + 1, n)

for i in range(b0, b1):
    still = []
    for t in active:
        if i <= t["entry_idx"]: still.append(t); continue
        if i > min(t["entry_idx"] + MAX_HOLD, n - 1):
            cp = C[min(t["entry_idx"] + MAX_HOLD, n - 1)]
            t["pnl_R"] = float(t["dir"] * (cp - t["ep"]) / t["a"] - sp)
            t["exit_idx"] = min(t["entry_idx"] + MAX_HOLD, n - 1)
            t["exit_px"] = cp; t["exit_reason"] = "max_hold"
            executed.append(t); continue
        dd = t["dir"]; ep = t["ep"]; a = t["a"]; fav = dd * (C[i] - ep)
        if fav > t["mf"]: t["mf"] = fav
        hit = False
        if t["sl_r"] == 0:
            if dd == 1 and L[i] <= ep:
                t["pnl_R"] = -sp; t["exit_px"] = ep; t["exit_reason"] = "BE"; hit = True
            elif dd == -1 and H[i] >= ep:
                t["pnl_R"] = -sp; t["exit_px"] = ep; t["exit_reason"] = "BE"; hit = True
        else:
            dist = abs(t["sl_r"]) * a
            if dd == 1 and (ep - L[i]) >= dist:
                t["pnl_R"] = float(t["sl_r"] - sp); t["exit_px"] = ep - dist; t["exit_reason"] = "SL"; hit = True
            elif dd == -1 and (H[i] - ep) >= dist:
                t["pnl_R"] = float(t["sl_r"] - sp); t["exit_px"] = ep + dist; t["exit_reason"] = "SL"; hit = True
        if hit: t["exit_idx"] = i; executed.append(t); continue
        td_ = TRAIL_ATR * a
        if t["mf"] >= td_ and (t["mf"] - fav) >= td_:
            xp = ep + dd * (t["mf"] - td_)
            t["pnl_R"] = float((t["mf"] - td_) / a - sp)
            t["exit_idx"] = i; t["exit_px"] = xp; t["exit_reason"] = "trail"
            executed.append(t); continue
        still.append(t)
    active = still
    if i not in info: continue
    d_, q_ = info[i]
    if i - last_open[d_] < COOLDOWN: continue
    for t in active:
        if t["sl_r"] == 0: continue
        cur = t["dir"] * (C[i] - t["ep"]) / t["a"]
        if cur >= BE_R: t["sl_r"] = 0; t["be_moved_at"] = i
    ei = i + 1
    if ei >= n or not (np.isfinite(atr[i]) and atr[i] > 0): continue
    if len(active) >= MAX_CONC:
        worst = min(active, key=lambda x: x["q"])
        if q_ >= worst["q"] + SWITCH_DELTA:
            worst["pnl_R"] = float(worst["dir"] * (C[i] - worst["ep"]) / worst["a"] - sp)
            worst["exit_idx"] = i; worst["exit_px"] = C[i]; worst["exit_reason"] = "switch_closed"
            executed.append(worst); active.remove(worst)
        else: continue
    active.append({"sig_idx": i, "entry_idx": ei, "entry_time": pd.Timestamp(times[ei]),
                   "dir": d_, "ep": float(O[ei]), "a": float(atr[i]),
                   "sl_r": float(-SL_ATR), "mf": 0.0, "q": float(q_), "pnl_R": None})
    last_open[d_] = i

for t in active:
    eb = min(t["entry_idx"] + MAX_HOLD, n - 1)
    t["pnl_R"] = float(t["dir"] * (C[eb] - t["ep"]) / t["a"] - sp)
    t["exit_idx"] = eb; t["exit_px"] = C[eb]; t["exit_reason"] = "open_at_now"
    executed.append(t)

# ── 6. Filter to today + summary ───────────────────────────────────────
today_date = df.time.iloc[-1].date()
today = [t for t in executed if pd.Timestamp(t["entry_time"]).date() == today_date]
today.sort(key=lambda t: t["entry_time"])

print(f"\n=== Atlas BTC sim — {today_date} ===")
print(f"({len(today)} trades opened today)")
if not today:
    print("(no Atlas BTC trades fired today)")
    sys.exit(0)

rs = np.array([t["pnl_R"] for t in today])
wins = int((rs > 0).sum())
pf = float(rs[rs > 0].sum() / max(-rs[rs <= 0].sum(), 1e-9)) if (rs <= 0).any() else float("inf")
eq = np.cumsum(rs); dd_R = float((np.maximum.accumulate(eq) - eq).max())

print(f"sumR {rs.sum():+.2f} | WR {wins / len(rs) * 100:.1f}% | PF {pf:.2f} | DD {dd_R:.2f}R")
print(f"Estimated USD @ 0.01 lot: {rs.sum() * R_to_USD:+.2f} (DD ${dd_R * R_to_USD:.2f})")
print(f"Estimated USD @ 0.10 lot: {rs.sum() * R_to_USD * 10:+.2f} (DD ${dd_R * R_to_USD * 10:.2f})")
print()
print(f"{'#':>3} {'entry_time':>17} {'dir':>4} {'entry_px':>9} {'exit_time':>17} {'exit_px':>9} {'atr':>5} {'Q':>5} {'R':>7} {'reason':>10} {'BE?':>4}")
for k, t in enumerate(today):
    et = pd.Timestamp(times[t["exit_idx"]])
    be = "yes" if t.get("be_moved_at") is not None else ""
    print(f"{k:>3} {t['entry_time'].strftime('%Y-%m-%d %H:%M'):>17} {'BUY' if t['dir'] == 1 else 'SELL':>4} "
          f"{t['ep']:>9.3f} {et.strftime('%Y-%m-%d %H:%M'):>17} {t['exit_px']:>9.3f} "
          f"{t['a']:>5.2f} {t['q']:>5.2f} {t['pnl_R']:>+7.2f} {t['exit_reason']:>10} {be:>4}")

# ── 7. Save equity PNG ─────────────────────────────────────────────────
out_dir = ROOT / "products/atlas_btc/scripts/_out"
out_dir.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(13, 5))
eq_usd = eq * R_to_USD
eq_times = [t["entry_time"] for t in today]
ax.plot(eq_times, eq_usd, lw=1.8, color="#8b5cf6", marker="o", markersize=4)
ax.axhline(0, color="k", lw=0.7)
ax.set_title(f"Atlas BTC sim {today_date} — {len(today)} trades, WR {wins / len(rs) * 100:.1f}%, "
             f"PF {pf:.2f}, ${rs.sum() * R_to_USD:+.2f} @ 0.01 lot", fontsize=12)
ax.set_ylabel("Cumulative profit ($)")
plt.tight_layout()
out_png = out_dir / "sim_today.png"
plt.savefig(out_png, dpi=110)
print(f"\nwrote {out_png}")
