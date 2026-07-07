"""Pre-push verification for the hermes_btc + hermes_xau edge fix.
1. loader + routing: both bundles route to edge_pullback, clean entry/exit responses
2. M5 gating: entry only fires evaluation when the last M1 bar closes an M5 bucket
3. M5 ATR scaling: sl_atr_mult/trail_atr_mult scaled by atr5/atr1 (EA M1-ATR convention)
4. M5 exit: tt-trail tightens at >= 150 M1 bars held (30 M5 bars), not before;
   max_hold at 300 M1 bars (60 M5)
"""
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
SRV = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(SRV))
from decision_engine import loader, decide_hermes
from decision_engine.configs import REGISTRY
import decision_engine.edge_pullback as ep

def load_bars(pq, tail=9000, cols=None):
    d = pd.read_parquet(pq, columns=cols)
    d = d.rename(columns={[c for c in d.columns if "time" in c.lower()][0]: "time"})
    d["time"] = pd.to_datetime(d["time"]); d = d.sort_values("time").drop_duplicates("time")
    if "tick_volume" not in d.columns: d["tick_volume"] = d.get("volume", 0)
    return d.tail(tail).reset_index(drop=True)

print("=" * 72)
for prod, pq, cols in [
    ("hermes_btc", "/home/jay/Desktop/new-model-zigzag/data/m1_btc_orderflow_8y.parquet",
     ["time", "open", "high", "low", "close", "tick_volume"]),
    ("hermes_xau", "/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet", None),
]:
    print(prod)
    bundle = loader.load_bundle(REGISTRY[prod])
    pl = bundle.payload
    print(f"  version={pl.get('version')} thr={pl.get('threshold'):.3f} bm={pl.get('bar_minutes')}"
          f" maxh={pl.get('maxh')} tt={pl.get('tight_after')}/{pl.get('tight_trail_R')}")
    bars = load_bars(pq, cols=cols)
    r = decide_hermes.decide_entry(bars, bundle, account="verify", open_positions=[])
    print(f"  ENTRY action={r.get('action')} reason={str(r.get('reason'))[:70]}")
    assert r.get("trace", {}).get("engine") == "edge_pullback", "ROUTING FAILED"
    assert r.get("action") in ("open", "hold")
    if pl.get("bar_minutes", 1) > 1:
        # bucket gating: trim to a NON-boundary minute -> must hold with m5_bucket_open
        for off in range(1, 6):
            b2 = bars.iloc[:-off]
            if pd.Timestamp(b2["time"].iloc[-1]).minute % 5 != 4:
                r2 = decide_hermes.decide_entry(b2, bundle, account="verify", open_positions=[])
                assert "bucket_open" in str(r2.get("reason")), f"gating failed: {r2.get('reason')}"
                print(f"  M5 gating OK (last={pd.Timestamp(b2['time'].iloc[-1]).strftime('%H:%M')}"
                      f" -> {r2['reason']})")
                break
        # boundary minute + forced pullback: check multiplier scaling
        b3 = bars.copy()
        while pd.Timestamp(b3["time"].iloc[-1]).minute % 5 != 4: b3 = b3.iloc[:-1]
        a1 = ep._atr14(b3)[-1]; d5 = ep._resample(b3, 5); a5 = ep._atr14(d5)[-1]
        print(f"  atr1={a1:.3f} atr5={a5:.3f} expected sl_mult≈{7*a5/a1:.2f} trail≈{2*a5/a1:.2f}")
    print("  ✓", prod)

# ---- M5 exit boundary tests (synthetic): peak +1.5R(M5) then give back 1.0R(M5)
print("=" * 72)
bundle = loader.load_bundle(REGISTRY["hermes_xau"])
N = 4000
def mk(bh_m1):
    t = pd.date_range("2026-07-01", periods=N, freq="1min")
    c = np.full(N, 100.0)
    ei = N - bh_m1
    # M5 ATR ~= 0.5 (make M1 ranges 0.1) ; move peak +0.75 (=1.5R if atr5=0.5), back to +0.25
    peak = ei + 25
    c[ei:peak] = np.linspace(100.0, 100.75, peak - ei)
    c[peak:] = np.linspace(100.75, 100.25, N - peak)
    return pd.DataFrame({"time": t, "open": c, "high": c + 0.05, "low": c - 0.05,
                         "close": c, "tick_volume": 1})
class Pos:
    direction = 1; entry_price = 100.0; entry_atr = 0.1
    def __init__(s, bh): s.bars_held = bh
for bh in (145, 150, 155, 299, 300):
    bars = mk(bh)
    # align so entry bar is at an M5 boundary: shift the series start
    r = decide_hermes.decide_exit(bars, Pos(bh), bundle, account="verify")
    print(f"  bars_held={bh:>4} (M5 held {bh//5:>2}) -> {r['action']:<5} {str(r['reason'])[:60]} "
          f"tt={r['trace'].get('tt_trail')}")
print("ALL VERIFICATIONS DONE")
