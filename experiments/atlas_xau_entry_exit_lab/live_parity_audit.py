"""
LIVE PARITY AUDIT (2026-07-07): for every live trade today, compare live fill
prices against the Dukascopy M1 series the server decides on. Per trade:
  sim_pnl  = d * (duk_open(exit_min) - duk_open(entry_min))   [what the honest
             backtest would credit for the same timestamps, offset-free]
  live_pnl = broker realized
  friction = live_pnl - sim_pnl   -> total execution cost (spread+slip+offset drift)
Broker-time -> UTC offset inferred per symbol by best price alignment.
"""
import sys, warnings
from datetime import datetime, timezone
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import dukascopy_python
from dukascopy_python.instruments import (INSTRUMENT_FX_METALS_XAU_USD,
                                          INSTRUMENT_VCCY_BTC_USD,
                                          INSTRUMENT_IDX_AMERICA_E_D_J_IND)

# (symbol, dir, entry_broker_time, entry_fill, exit_broker_time, exit_fill, live_pnl_usd, lots, tag)
T = [
 ("XAU",  1, "2026-07-07 01:29", 4165.58, "2026-07-07 01:36", 4166.91,    6.65, 0.05, "ATL"),
 ("BTC",  1, "2026-07-07 02:42", 64067.0, "2026-07-07 03:22", 64107.5,    4.05, 0.10, "ATB"),
 ("XAU",  1, "2026-07-07 02:45", 4162.05, "2026-07-07 03:38", 4151.03,  -55.10, 0.05, "V7"),
 ("XAU",  1, "2026-07-07 03:40", 4150.45, "2026-07-07 04:27", 4126.78, -118.35, 0.05, "V7"),
 ("XAU", -1, "2026-07-07 04:30", 4132.64, "2026-07-07 09:20", 4126.30,   31.70, 0.05, "V7"),
 ("XAU", -1, "2026-07-07 09:20", 4126.16, "2026-07-07 14:20", 4140.41,  -71.25, 0.05, "V7"),
 ("XAU", -1, "2026-07-07 09:46", 4131.79, "2026-07-07 12:16", 4128.00,   18.95, 0.05, "HRM"),
 ("XAU", -1, "2026-07-07 10:58", 4125.32, "2026-07-07 11:13", 4138.98,  -68.30, 0.05, "ATL"),
 ("XAU", -1, "2026-07-07 11:47", 4129.66, "2026-07-07 11:58", 4126.43,   16.15, 0.05, "ATL"),
 ("XAU", -1, "2026-07-07 12:05", 4126.99, "2026-07-07 12:35", 4127.65,   -3.30, 0.05, "ATL"),
 ("XAU", -1, "2026-07-07 12:16", 4127.84, "2026-07-07 14:46", 4150.68, -114.20, 0.05, "HRM"),
 ("XAU", -1, "2026-07-07 12:35", 4127.18, "2026-07-07 14:18", 4141.60,  -72.10, 0.05, "ATL"),
 ("BTC", -1, "2026-07-07 13:57", 63172.5, "2026-07-07 14:07", 63165.0,    0.75, 0.10, "HRB"),
 ("XAU",  1, "2026-07-07 14:20", 4140.40, "2026-07-07 18:25", 4145.43,   25.15, 0.05, "V7"),
 ("BTC", -1, "2026-07-07 14:49", 63277.5, "2026-07-07 15:06", 63614.0,  -33.65, 0.10, "ATB"),
 ("BTC",  1, "2026-07-07 16:03", 63390.0, "2026-07-07 16:24", 63100.5,  -28.95, 0.10, "ATB"),
 ("XAU",  1, "2026-07-07 17:41", 4148.22, "2026-07-07 20:11", 4144.18,  -20.20, 0.05, "HRM"),
 ("XAU",  1, "2026-07-07 18:25", 4145.67, "2026-07-07 22:10", 4107.83, -189.20, 0.05, "V7"),
 ("DJI", -1, "2026-07-07 18:55", 52877.16, "2026-07-07 19:25", 52932.96, -27.90, 0.05, "ATD"),
 ("DJI", -1, "2026-07-07 19:23", 52903.66, "2026-07-07 19:53", 52913.46,  -4.90, 0.05, "HRD"),
 ("DJI", -1, "2026-07-07 19:58", 52920.16, "2026-07-07 20:22", 52918.96,   0.60, 0.05, "HRD"),
 ("DJI", -1, "2026-07-07 20:22", 52916.16, "2026-07-07 22:08", 52887.21,  14.48, 0.05, "HRD"),
 ("BTC",  1, "2026-07-07 21:11", 64024.0, "2026-07-07 21:41", 63974.5,   -4.95, 0.10, "HRB"),
 ("DJI", -1, "2026-07-07 21:32", 52964.16, "2026-07-07 21:38", 52970.96,  -3.40, 0.05, "ATD"),
 ("DJI", -1, "2026-07-07 22:23", 52900.16, "2026-07-07 22:46", 52913.46,  -6.65, 0.05, "HRD"),
 ("DJI", -1, "2026-07-07 22:27", 52903.16, "2026-07-07 22:47", 52917.96,  -7.40, 0.05, "ATD"),
]
UNIT = {"XAU": 100, "BTC": 1, "DJI": 1}     # contract size per 1.0 lot
INSTR = {"XAU": INSTRUMENT_FX_METALS_XAU_USD, "BTC": INSTRUMENT_VCCY_BTC_USD,
         "DJI": INSTRUMENT_IDX_AMERICA_E_D_J_IND}

def fetch(sym):
    r = dukascopy_python.fetch(instrument=INSTR[sym], interval=dukascopy_python.INTERVAL_MIN_1,
            offer_side=dukascopy_python.OFFER_SIDE_BID,
            start=datetime(2026, 7, 6, 18, 0, tzinfo=timezone.utc),
            end=datetime(2026, 7, 7, 23, 59, tzinfo=timezone.utc))
    d = r.reset_index().rename(columns={"timestamp": "time"})
    d["time"] = pd.to_datetime(d["time"]).dt.tz_convert("UTC").dt.tz_localize(None)
    return d.set_index("time")

DUK = {s: fetch(s) for s in ("XAU", "BTC", "DJI")}
print({s: f"{len(DUK[s])} bars" for s in DUK})

def infer_offset(sym):
    """broker-time = UTC + off hours; pick off minimizing |fill - duk_open| across trades"""
    best, boff = 1e18, None
    for off in range(0, 4):
        tot = cnt = 0
        for (s, d, et, ef, xt, xf, pnl, lots, tag) in T:
            if s != sym: continue
            t = pd.Timestamp(et) - pd.Timedelta(hours=off)
            if t in DUK[s].index:
                tot += abs(float(DUK[s].loc[t, "open"]) - ef); cnt += 1
        if cnt and tot / cnt < best: best, boff = tot / cnt, off
    return boff, best

OFF = {}
for s in ("XAU", "BTC", "DJI"):
    off, err = infer_offset(s)
    OFF[s] = off
    print(f"{s}: broker = UTC+{off}  (mean |fill-duk_open| {err:.2f})")

print(f"\n{'tag':<5}{'sym':<5}{'live$':>9}{'sim$':>9}{'friction$':>10}   entry/exit slip (pts)")
rows = []
for (s, d, et, ef, xt, xf, pnl, lots, tag) in T:
    te = pd.Timestamp(et) - pd.Timedelta(hours=OFF[s]); tx = pd.Timestamp(xt) - pd.Timedelta(hours=OFF[s])
    duk = DUK[s]
    def opn(t):
        if t in duk.index: return float(duk.loc[t, "open"])
        i = duk.index.searchsorted(t)
        return float(duk["open"].iloc[min(i, len(duk) - 1)])
    oe, ox = opn(te), opn(tx)
    sim = d * (ox - oe) * lots * UNIT[s]
    fr = pnl - sim
    se = d * (ef - oe); sx = -d * (xf - ox)   # signed slip vs duk (entry pays +, exit pays +)
    rows.append((tag, s, pnl, sim, fr))
    print(f"{tag:<5}{s:<5}{pnl:>9.2f}{sim:>9.2f}{fr:>10.2f}   {se:+.2f} / {sx:+.2f}")

df = pd.DataFrame(rows, columns=["tag", "sym", "live", "sim", "fr"])
print("\nBY PRODUCT: live vs sim(same timestamps, duk opens) vs friction")
print(df.groupby("tag")[["live", "sim", "fr"]].sum().round(2))
print("\nBY SYMBOL friction per trade:")
g = df.groupby("sym").agg(n=("fr", "size"), tot=("fr", "sum"), per_trade=("fr", "mean")).round(2)
print(g)
print(f"\nTOTAL live {df.live.sum():+.2f} | sim {df.sim.sum():+.2f} | friction {df.fr.sum():+.2f}")
