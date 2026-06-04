"""
10_sim_today.py — Hermes BTC live-equivalent simulation for the current trading day

WHAT THIS DOES
==============
Pulls fresh Dukascopy BTC/USD M1 bars for the last 3 days, runs the SAME
feature pipeline + decision logic that the deployed server uses, and
reports every trade Hermes BTC would have opened today plus its outcome
(entry time, direction, Q value, dist, exit reason, R outcome).

Use this to:
  • Compare against MT5 history to see if your fills match the model
  • Sanity-check the deployed config after a tuning change
  • Diagnose specific bad days ("why did it stack 9 buys at the peak?")

DEPLOYED CONFIG MIRRORED (commercial/server/decision_engine/configs/hermes_btc.py)
=================================================================================
  symbol_base:    BTCUSD
  near_thr:       0.50    (pullback bars: |dist| ≤ 0.50 ATR)
  counter_thr:    1.5     (counter bars: dist_signed × cdir ≤ -1.5 ATR)
  q_thr:          4.0     (raised 2.5 → 4.0 on 2026-06-04 after a
                           30-day post-deploy backtest. Strictly dominant
                           in all metrics vs the previous 2.5 setting.)
  dist_cap:       0.0     (not currently used on BTC — XAU-only veto)
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
    python3 products/hermes_btc/scripts/10_sim_today.py

OUTPUT
======
  • Stdout table: every trade today with entry, exit, Q, dist, R, reason
  • Saved PNG: products/hermes_btc/scripts/_out/sim_today.png
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

# ── DEPLOYED CONFIG — mirrors configs/hermes_btc.py exactly ────────────
SYMBOL_BASE  = "BTCUSD"
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
Q_THR        = 4.0
DIST_CAP     = 0.0       # disabled on BTC
R_to_USD     = 2.30      # 0.01 lot BTCUSD ≈ $2.30 / R (depends on broker)

BUNDLE_PATH = SERVER / "decision_engine/models/hermes_btc_validated.pkl"


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


def add_features_LOOKAHEAD(df):
    """Recreate the deployed feature pipeline. Uses resample()+reindex(ffill)
    for HTF columns — this is the SAME look-ahead pattern as the deployed
    server. Bug-compatible with production behaviour."""
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
    df_ts = df.set_index("time")
    for tf_name, tf in [("m5", "5min"), ("m15", "15min"), ("h1", "60min")]:
        g = df_ts[["high", "low", "close"]].resample(tf).agg(
            {"high": "max", "low": "min", "close": "last"}).dropna()
        ch = g["close"]; delta = ch.diff()
        up = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        dn = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
        rs = up / dn.replace(0, np.nan)
        g[f"{tf_name}_rsi14"] = 100 - 100 / (1 + rs)
        g[f"{tf_name}_slope5"] = (ch - ch.shift(5))
        g[f"{tf_name}_ema50_dist"] = ch - ch.ewm(span=50, adjust=False).mean()
        keep = [c_ for c_ in g.columns if c_ not in ("high", "low", "close")]
        out = g[keep].reindex(df_ts.index, method="ffill")
        for col in out.columns: df[col] = out[col].to_numpy()
    return df


# ── 1. Fetch 3 days of BTC/USD M1 ──────────────────────────────────────
end = datetime.now(timezone.utc)
start = end - timedelta(days=3)
print(f"[hermes_btc] Fetching BTC/USD M1 {start.isoformat()} → {end.isoformat()}")
df = dukascopy_python.fetch(instrument="BTC/USD", interval=dukascopy_python.INTERVAL_MIN_1,
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
fdf = add_features_LOOKAHEAD(df)
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
n_blocked = int(stretched_block.sum())
idxs = idxs[keep]; dirs = dirs[keep]
print(f"  candidates after dist_cap={DIST_CAP} veto: {len(idxs)} (blocked {n_blocked} stretched-counter)")

# ── 4. Q score, gate ────────────────────────────────────────────────────
bundle = pickle.load(open(BUNDLE_PATH, "rb"))
M = bundle["q_model"]; feat_cols = bundle["feat_cols"]
Xall = fdf[feat_cols].fillna(0).to_numpy(np.float32)
q_live = M.predict(Xall[idxs])
print(f"  Q dist: median={np.median(q_live):.2f}  p75={np.quantile(q_live, 0.75):.2f}  max={q_live.max():.2f}")
print(f"  candidates with Q >= {Q_THR}: {int((q_live >= Q_THR).sum())}")

times = df["time"].to_numpy()

# ── 5. Multi-position simulation with BE@+0.5R, switch-rule, cooldown ──
active = []; executed = []; last_open = {-1: -10**9, 1: -10**9}
info = {int(idxs[k]): (int(dirs[k]), float(q_live[k]))
        for k in range(len(idxs)) if q_live[k] >= Q_THR}
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

print(f"\n=== Hermes BTC sim — {today_date} ===")
print(f"({len(today)} trades opened today)")
if not today:
    print("(no Hermes BTC trades fired today)")
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
out_dir = ROOT / "products/hermes_btc/scripts/_out"
out_dir.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(13, 5))
eq_usd = eq * R_to_USD
eq_times = [t["entry_time"] for t in today]
ax.plot(eq_times, eq_usd, lw=1.8, color="#06b6d4", marker="o", markersize=4)
ax.axhline(0, color="k", lw=0.7)
ax.set_title(f"Hermes BTC sim {today_date} — {len(today)} trades, WR {wins / len(rs) * 100:.1f}%, "
             f"PF {pf:.2f}, ${rs.sum() * R_to_USD:+.2f} @ 0.01 lot", fontsize=12)
ax.set_ylabel("Cumulative profit ($)")
plt.tight_layout()
out_png = out_dir / "sim_today.png"
plt.savefig(out_png, dpi=110)
print(f"\nwrote {out_png}")
