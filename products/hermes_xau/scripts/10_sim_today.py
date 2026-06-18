"""
10_sim_today.py — Hermes XAU live-equivalent simulation for the current trading day

WHAT THIS DOES
==============
Pulls fresh Dukascopy XAU/USD M1 bars for the last 3 days, runs the SAME
feature pipeline + decision logic that the deployed server uses, and
reports every trade Hermes XAU would have opened today plus its outcome
(entry time, direction, Q value, dist, exit reason, R outcome).

Use this to:
  • Compare against MT5 history to see if your fills match the model
  • Sanity-check the deployed config after a tuning change
  • Diagnose specific bad days ("why did it stack 9 buys at the peak?")

DEPLOYED CONFIG MIRRORED (commercial/server/decision_engine/configs/hermes_xau.py)
=================================================================================
  symbol_base:    XAUUSD
  near_thr:       0.50    (pullback bars: |dist| ≤ 0.50 ATR)
  counter_thr:    1.5     (counter bars: dist_signed × cdir ≤ -1.5 ATR)
  q_thr:          4.0     (combined-Q upgrade, 2026-05-26)
  dist_cap:       3.0     (NEW 2026-06-04 — block counter entries when
                           |dist| > 3.0 ATR. Prevents stretched-counter
                           disasters like 2026-06-04 14:13-14:35.)
  sl_hard_atr:    6.0
  trail_atr:      2.0
  be_trigger_r:   0.5
  max_hold_bars:  300
  max_concurrent: 4 slots, switch_delta=0.5, cooldown=5 bars

LIMITATIONS — this is OFFLINE, NOT EXACTLY THE DEPLOYED SERVER
==============================================================
  • Order-flow features (signed_flow, vpin_proxy, tick_intensity_50, etc.)
    are ZEROED here — the live server gets these from each customer's tick
    stream. Q values will differ slightly from live.
  • Regime cluster gating (if any) is not applied.
  • Drawdown guard / kill-switch state is not applied.
  • Look-ahead HTF features match deployed (resample+ffill) — that's the
    only "look-ahead" present and it's bug-compatible with production.

For TRULY EXACT live trades, query the server's funnel_log:
  curl -s "https://edge-predictor.onrender.com/decide/_log-dump?hours=24
          &x_admin_secret=$ADMIN_SECRET"

USAGE
=====
    python3 products/hermes_xau/scripts/10_sim_today.py

OUTPUT
======
  • Stdout table: every trade today with entry, exit, Q, dist, R, reason
  • Saved PNG: products/hermes_xau/scripts/_out/sim_today.png
"""
import sys, pickle
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
sys.path.insert(0, str(ROOT / "products/hermes_xau"))
from tfk import compute_tfk
import dukascopy_python

# ── DEPLOYED CONFIG — mirrors configs/hermes_xau.py exactly ────────────
SYMBOL_BASE  = "XAUUSD"
SPREAD       = 0.30      # ATR units of spread cost applied to every trade
SL_ATR       = 6.0
TRAIL_ATR    = 2.0
MAX_HOLD     = 300
BE_R         = 0.5
MAX_CONC     = 4
SWITCH_DELTA = 0.5
COOLDOWN     = 5
NEAR_THR     = 0.50
COUNTER_THR  = 1.5
Q_THR        = 2.0   # 2026-06-18 dynamic Q: strict default for chop
Q_THR_TREND  = 0.5   # 2026-06-18 dynamic Q: looser when trend_strong
TREND_AGE_MIN     = 30
TREND_SLOPE_MIN   = 1.0
TREND_DEMA50_MIN  = 1.0
DIST_CAP     = 3.0       # 2026-06-04 stretched-counter veto
TIME_BLOCK   = (20, 1)   # 2026-06-09 — block entries at UTC hours [20, 1)
                         # +3.5% $ vs no time-block; (0, 0) = disabled
TREND_SLOPE_BLOCK = 0.0  # disabled on Hermes XAU (HURT in backtest)
R_to_USD     = 1.50      # 0.01 lot XAUUSD ≈ $1.50 / R

BUNDLE_PATH = SERVER / "decision_engine/models/hermes_xau_validated.pkl"


@njit
def trail_labels(idxs, dirs, O, H, L, C, atr, sp, SL, TRAIL, MAXH, n):
    """Mirror of server's trail+SL+max_hold exit logic, no BE (used for
    standalone outcome labelling; multi-pos sim below applies BE itself)."""
    m = len(idxs); pnl = np.empty(m); xit = np.empty(m, np.int64)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]; ei = i + 1
        if ei >= n or not (a > 0):
            pnl[k] = 0.0; xit[k] = ei; continue
        ep = O[ei]; hard = SL * a; trd = TRAIL * a; mf = 0.0
        end = min(ei + MAXH, n - 1); done = False
        for j in range(ei, end + 1):
            fav = d * (C[j] - ep)
            if fav > mf: mf = fav
            if d == 1 and (ep - L[j]) >= hard: pnl[k] = -SL - sp; xit[k] = j; done = True; break
            if d == -1 and (H[j] - ep) >= hard: pnl[k] = -SL - sp; xit[k] = j; done = True; break
            if mf >= trd and (mf - fav) >= trd: pnl[k] = (mf - trd) / a - sp; xit[k] = j; done = True; break
        if not done:
            pnl[k] = d * (C[end] - ep) / a - sp; xit[k] = end
    return pnl, xit


def bars_in_regime_array(cdir):
    n = len(cdir); out = np.zeros(n, np.int64); cur = 1
    for i in range(1, n):
        cur = cur + 1 if cdir[i] == cdir[i - 1] else 1
        out[i] = cur
    return out


_HTF_EMA_A = 2.0 / 51.0

def add_features_CAUSAL(df):
    """Causal HTF — matches deployed _causal_htf_partial (server-side).
    Each M1 bar sees only HTF bars that have COMPLETED. Replaces the prior
    add_features_LOOKAHEAD which used resample+ffill (peeks at the rest of
    the in-progress HTF bucket → inflated PF 2-5x; fixed server-side 2026-05-28)."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])
    c = df["close"]; h = df["high"]; l = df["low"]
    prev = c.shift(1).fillna(c.iloc[0])
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=14).mean()
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    dn = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = up / dn.replace(0, np.nan); df["rsi14"] = 100 - 100 / (1 + rs)
    for nm in (20, 50, 100, 200):
        ema = c.ewm(span=nm, adjust=False).mean()
        df[f"dist_ema{nm}"] = (c - ema) / df["atr14"]
    for nm in (5, 10, 20):
        df[f"slope{nm}"] = (c - c.shift(nm)) / df["atr14"]
    df["atr_ratio"] = df["atr14"] / df["atr14"].rolling(50, min_periods=50).mean()
    # CAUSAL HTF (ported from server's _causal_htf_partial)
    s = df.set_index("time")["close"]
    for tf_name, tf_min in [("m5", 5), ("m15", 15), ("h1", 60)]:
        g = s.resample(f"{tf_min}min").last().dropna()
        hc = g.to_numpy(np.float64)
        end_times = (g.index + pd.Timedelta(minutes=tf_min)).values
        n = len(df)
        rsi = np.full(n, np.nan); slope = np.full(n, np.nan); emad = np.full(n, np.nan)
        if len(hc) >= 16:
            ema = np.empty(len(hc)); e = hc[0]
            for i in range(len(hc)):
                e = hc[0] if i == 0 else e * (1 - _HTF_EMA_A) + hc[i] * _HTF_EMA_A
                ema[i] = e
            d = np.diff(hc, prepend=hc[0])
            cumg = np.cumsum(np.clip(d, 0, None))
            cuml = np.cumsum(np.clip(-d, 0, None))
            m1t = df["time"].values
            cc_all = df["close"].to_numpy(np.float64)
            j = np.searchsorted(end_times, m1t, side="right") - 1
            ok = j >= 14
            jj = j[ok]; cc = cc_all[ok]
            slope[ok] = cc - hc[jj - 4]
            emad[ok] = cc - (ema[jj] * (1 - _HTF_EMA_A) + cc * _HTF_EMA_A)
            dc = cc - hc[jj]
            gsum = (cumg[jj] - cumg[jj - 13]) + np.clip(dc, 0, None)
            lsum = (cuml[jj] - cuml[jj - 13]) + np.clip(-dc, 0, None)
            rs = (gsum / 14.0) / np.where(lsum == 0, np.nan, lsum / 14.0)
            rsi[ok] = 100 - 100 / (1 + rs)
        df[f"{tf_name}_rsi14"] = rsi
        df[f"{tf_name}_slope5"] = slope
        df[f"{tf_name}_ema50_dist"] = emad
    return df


# ── 1. Fetch 3 days of XAU/USD M1 ──────────────────────────────────────
end = datetime.now(timezone.utc)
start = end - timedelta(days=3)
print(f"[hermes_xau] Fetching XAU/USD M1 {start.isoformat()} → {end.isoformat()}")
df = dukascopy_python.fetch(instrument="XAU/USD", interval=dukascopy_python.INTERVAL_MIN_1,
                             offer_side=dukascopy_python.OFFER_SIDE_BID, start=start, end=end)
df = df.reset_index().rename(columns={"timestamp": "time"})
df["time"] = pd.to_datetime(df["time"]).dt.tz_convert("UTC").dt.tz_localize(None)
df = df.sort_values("time").reset_index(drop=True)
if "tick_volume" not in df.columns:
    vc = "volume" if "volume" in df.columns else [c for c in df.columns if "vol" in c.lower()][0]
    df["tick_volume"] = df[vc]
print(f"  fetched {len(df):,} bars  last bar: {df.time.iloc[-1]}")

# ── 2. Features (TFK + standard) ───────────────────────────────────────
tfk_df = compute_tfk(df)
fdf = add_features_CAUSAL(df)
for c in ["force", "velocity", "x_est", "regime_w", "trend_raw", "trend",
          "committed_dir", "confirmed_dir", "tfk_line"]:
    fdf[c] = tfk_df[c].to_numpy()
cdir = tfk_df["committed_dir"].to_numpy(np.int64)
tline = tfk_df["tfk_line"].to_numpy(float)
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float)
L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
atr = fdf["atr14"].to_numpy(float); n = len(fdf)
sp = SPREAD / np.nanmedian(atr)
dist_signed = np.where(atr > 0, (C - tline) / atr, 0.0)
fdf["dist_at_signal"] = dist_signed
fdf["dist_abs"] = np.abs(dist_signed)
fdf["regime_age"] = bars_in_regime_array(cdir)
fdf["bar_range_atr"] = (H - L) / np.maximum(atr, 1e-9)

# Order-flow columns zeroed — see docstring caveat
FLOW = ["imbalance_ratio", "bid_ask_vol_ratio", "vpin_proxy", "median_spread",
        "cum_signed_5", "flow_persistence_5", "cum_signed_15", "flow_persistence_15",
        "cum_signed_60", "flow_persistence_60", "spread_vol_50", "tick_intensity_50",
        "signed_flow", "n_ticks"]
for f in FLOW:
    if f not in fdf.columns: fdf[f] = 0.0

# Path B features (deployed 2026-06-17 — required by 62-feature Q bundle)
_C = fdf["close"].to_numpy(np.float64)
_H = fdf["high"].to_numpy(np.float64); _L = fdf["low"].to_numpy(np.float64)
_atr14 = fdf["atr14"].to_numpy(np.float64)
for _w, _lab in [(5,"5m"),(15,"15m"),(60,"60m"),(240,"240m")]:
    _r = pd.Series(_C).pct_change(_w) * 100
    _roll = _r.rolling(500, min_periods=100)
    _z = (_r - _roll.mean()) / _roll.std().replace(0, np.nan)
    fdf[f"ret_{_lab}"]  = _r.fillna(0).to_numpy()
    fdf[f"retz_{_lab}"] = _z.fillna(0).to_numpy()
_atr_s = pd.Series(_atr14)
fdf["atr_slope10"]   = (_atr_s - _atr_s.shift(10)).fillna(0).to_numpy() / np.maximum(_atr14, 1e-6)
fdf["atr_pct_500"]   = _atr_s.rolling(500, min_periods=100).rank(pct=True).fillna(0.5).to_numpy()
fdf["vol_of_vol_60"] = _atr_s.rolling(60, min_periods=20).std().fillna(0).to_numpy() / np.maximum(_atr14, 1e-6)
_t = pd.to_datetime(fdf["time"])
_sid = _t.dt.normalize().astype("int64").to_numpy()
_tp = (_H + _L + _C) / 3.0
_v = fdf.get("tick_volume", pd.Series(1.0, index=fdf.index)).to_numpy(np.float64)
_v = np.maximum(_v, 1.0)
_vwap = np.zeros(len(fdf)); _last = -1; _cpv = 0.0; _cv = 0.0
for _i in range(len(fdf)):
    if _sid[_i] != _last: _cpv = 0.0; _cv = 0.0; _last = _sid[_i]
    _cpv += _tp[_i]*_v[_i]; _cv += _v[_i]; _vwap[_i] = _cpv / max(_cv, 1.0)
fdf["sess_vwap"] = _vwap
fdf["dist_vwap_atr"] = (_C - _vwap) / np.maximum(_atr14, 1e-6)
_vwap_s = pd.Series(_vwap)
fdf["vwap_slope_30"] = (_vwap_s - _vwap_s.shift(30)).fillna(0).to_numpy() / np.maximum(_atr14, 1e-6)
_hrs = _t.dt.hour.to_numpy()
fdf["hour_sin"] = np.sin(2*np.pi*_hrs/24)
fdf["hour_cos"] = np.cos(2*np.pi*_hrs/24)
fdf["sess_asia"]    = ((_hrs >= 0)  & (_hrs < 7)).astype(np.float32)
fdf["sess_london"]  = ((_hrs >= 7)  & (_hrs < 13)).astype(np.float32)
fdf["sess_overlap"] = ((_hrs >= 13) & (_hrs < 17)).astype(np.float32)
fdf["sess_ny"]      = ((_hrs >= 13) & (_hrs < 21)).astype(np.float32)

# ── 3. Candidate selection (pullback OR counter) ───────────────────────
counter_score = dist_signed * cdir
is_pullback   = fdf["dist_abs"].to_numpy() <= NEAR_THR
is_counter    = counter_score <= -COUNTER_THR
ok = np.isfinite(atr) & (atr > 0) & (cdir != 0)
ok[:250] = False; ok[-(MAX_HOLD + 1):] = False
mask = ok & (is_pullback | is_counter)
idxs = np.where(mask)[0]
dirs = cdir[idxs].copy()

# Stretched-counter veto (new 2026-06-04). Block counter entries (not
# pullback) with |dist| > DIST_CAP — these are the disasters we saw
# at 14:13-14:35 today.
dist_abs_at_cand = np.abs(dist_signed[idxs])
is_counter_at_cand = is_counter[idxs]
is_pullback_at_cand = is_pullback[idxs]
stretched_block = is_counter_at_cand & ~is_pullback_at_cand & (dist_abs_at_cand > DIST_CAP)
keep = ~stretched_block
idxs = idxs[keep]; dirs = dirs[keep]

# 2026-06-09 — per-product time-of-day filter mirrors deployed config
if TIME_BLOCK != (0, 0):
    hours = pd.to_datetime(df["time"].iloc[idxs].values).hour
    s, e = TIME_BLOCK
    bad = (hours >= s) & (hours < e) if s < e else (hours >= s) | (hours < e)
    keep_time = ~bad
    n_time_blocked = int(bad.sum())
    idxs = idxs[keep_time]; dirs = dirs[keep_time]
    print(f"  candidates after dist_cap={DIST_CAP} + time_block={TIME_BLOCK} UTC: "
          f"{len(idxs)} (blocked {int(stretched_block.sum())} stretched-counter, "
          f"{n_time_blocked} time-windowed)")
else:
    print(f"  candidates after dist_cap={DIST_CAP} veto: {len(idxs)} "
          f"(blocked {int(stretched_block.sum())} stretched-counter)")

# ── 4. Q score, gate ────────────────────────────────────────────────────
bundle = pickle.load(open(BUNDLE_PATH, "rb"))
M = bundle["q_model"]; feat_cols = bundle["feat_cols"]
Xall = fdf[feat_cols].fillna(0).to_numpy(np.float32)
q_live = M.predict(Xall[idxs])
# Dynamic Q threshold per candidate based on trend_strong flag
ra = fdf["regime_age"].to_numpy(float)[idxs]
sl20 = fdf["slope20"].to_numpy(float)[idxs]
de50 = fdf["dist_ema50"].to_numpy(float)[idxs]
trend_strong = (ra >= TREND_AGE_MIN) & (np.abs(sl20) >= TREND_SLOPE_MIN) & (np.abs(de50) >= TREND_DEMA50_MIN)
q_thr_per_cand = np.where(trend_strong, Q_THR_TREND, Q_THR)
pass_q = q_live >= q_thr_per_cand
print(f"  Q dist: median={np.median(q_live):.2f}  p75={np.quantile(q_live, 0.75):.2f}  max={q_live.max():.2f}")
print(f"  trend_strong candidates: {int(trend_strong.sum())} ({100*trend_strong.mean():.0f}%)")
print(f"  candidates passing dynamic Q (≥{Q_THR}/chop or ≥{Q_THR_TREND}/trend): {int(pass_q.sum())}")

times = df["time"].to_numpy()

# ── 5. Multi-position simulation with BE@+0.5R, switch-rule, cooldown ──
active = []; executed = []; last_open = {-1: -10**9, 1: -10**9}
info = {int(idxs[k]): (int(dirs[k]), float(q_live[k]))
        for k in range(len(idxs)) if pass_q[k]}
if not info:
    print("  no candidates survived q_thr filter")
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
        if t["sl_r"] == 0:  # BE armed
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

# Flush still-open at the end of fetched window
for t in active:
    eb = min(t["entry_idx"] + MAX_HOLD, n - 1)
    t["pnl_R"] = float(t["dir"] * (C[eb] - t["ep"]) / t["a"] - sp)
    t["exit_idx"] = eb; t["exit_px"] = C[eb]; t["exit_reason"] = "open_at_now"
    executed.append(t)

# ── 6. Filter to today's entries + summarise ───────────────────────────
today_date = df.time.iloc[-1].date()
today = [t for t in executed if pd.Timestamp(t["entry_time"]).date() == today_date]
today.sort(key=lambda t: t["entry_time"])

print(f"\n=== Hermes XAU sim — {today_date} ===")
print(f"({len(today)} trades opened today)")
if not today:
    print("(no Hermes XAU trades fired today)")
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
out_dir = ROOT / "products/hermes_xau/scripts/_out"
out_dir.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(13, 5))
eq_usd = eq * R_to_USD
eq_times = [t["entry_time"] for t in today]
ax.plot(eq_times, eq_usd, lw=1.8, color="#10b981", marker="o", markersize=4)
ax.axhline(0, color="k", lw=0.7)
ax.set_title(f"Hermes XAU sim {today_date} — {len(today)} trades, WR {wins / len(rs) * 100:.1f}%, "
             f"PF {pf:.2f}, ${rs.sum() * R_to_USD:+.2f} @ 0.01 lot", fontsize=12)
ax.set_ylabel("Cumulative profit ($)")
plt.tight_layout()
out_png = out_dir / "sim_today.png"
plt.savefig(out_png, dpi=110)
print(f"\nwrote {out_png}")
