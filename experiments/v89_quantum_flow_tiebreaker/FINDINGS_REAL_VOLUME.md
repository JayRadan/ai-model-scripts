# v8.9 RE-RUN with REAL Volume (from Dukascopy)

**Date**: 2026-05-03
**Status**: ✅✅✅ MASSIVE positive signal. Original spread-as-volume tests were INVALID.
This is the single largest positive result in the entire optimization thread.

## What changed

Earlier v8.9/v8.9b/v9.0 tests used `spread` as a fallback for `volume` because
the swing CSV doesn't have a volume column. Spread ≠ volume — completely
different signal. All three earlier tests were testing a different indicator
than Jay's original Pine version.

Downloaded real M5 volume from Dukascopy (105k XAU bars, 157k BTC bars,
Nov 2024 → May 2026). Re-ran the same agreement filter test using REAL volume.

## Results — comparison

```
                    Spread proxy (v8.9b)            Real volume (NOW)
                    ΔWR    ΔPF    ΔR              ΔWR    ΔPF    ΔR
Oracle XAU 5m       +1pp   +0.23  -41%            +13pp  +4.44  -1%   ← 
Midas XAU  5m       +2pp   +0.41  -38%            +14pp  +3.03  +20%  ← R UP, not down!
Oracle BTC 5m       +3pp   +0.75  -63%            +18pp  +2.91  +40%  ← BTC R +40%!
```

## Full breakdown (Dukascopy real volume)

```
Oracle XAU baseline:  WR 65.7% PF 3.56 R+3668  (n=1301)
  5m flow agree:      kept 829 (64%)  WR 78.9%  PF 8.00  R+3629  ΔR  -1%
  4h flow agree:      kept 844 (65%)  WR 66.8%  PF 3.56  R+2293  ΔR -38%
  BOTH 5m+4h agree:   kept 544 (42%)  WR 80.1%  PF 8.26  R+2295  ΔR -37%

Midas XAU baseline:   WR 58.4% PF 2.31 R+4347  (n=2193)
  5m flow agree:      kept 1284 (59%) WR 72.9%  PF 5.34  R+5198  ΔR +20%
  4h flow agree:      kept 1303 (59%) WR 60.2%  PF 2.63  R+3022  ΔR -31%
  BOTH:               kept 784 (36%)  WR 74.6%  PF 6.50  R+3512  ΔR -19%

Oracle BTC baseline:  WR 54.7% PF 1.82 R+3234  (n=2424)
  5m flow agree:      kept 1301 (54%) WR 72.5%  PF 4.73  R+4515  ΔR +40%
  4h flow agree:      kept 1371 (57%) WR 53.0%  PF 1.81  R+1806  ΔR -44%
  BOTH:               kept 737 (30%)  WR 72.2%  PF 4.89  R+2564  ΔR -21%
```

## Interpretation

**5m flow filter is the winner.** On every product:
- Win rate jumps 13-18 percentage points
- Profit factor jumps 2.9-4.4 points
- Total R either holds (Oracle -1%) or actively improves (Midas +20%, BTC +40%)

**This is exactly what Jay originally wanted**: "same R but higher WR" —
and on Midas/BTC we get MORE R *plus* higher WR.

The 4h-flow filter alone is essentially neutral/negative. The 5m signal
carries all the value. The 4h adds nothing on top.

## Why the spread-proxy hid this so completely

The Quantum Flow formula is:
```
flow = EMA(trend × volume_ratio × 1000, 21)
```
where `volume_ratio = current_vol / SMA(vol, 50)`.

When you substitute `spread` for `volume`:
- `spread_ratio` is HIGH on news/illiquidity bars (when spread blows out)
- `spread_ratio` is LOW on calm bars
- The signal becomes "moments of broker stress" instead of "moments of participation"
- Completely different (and noisier) signal

With real volume, `vol_ratio` is HIGH on bars with heavy participation
(big moves with conviction) and LOW on chop. This is what the indicator
is *designed* to detect.

## Caveats before deploying

1. **Broker mismatch.** Dukascopy data ≠ Eightcap data. Bar boundaries
   and tick counts differ. For deployment we need volume from the user's
   actual broker (or at least from MT5's `tick_volume` field which we
   already added to the exporter).

2. **Sample alignment.** The merge dropped 5-18% of holdout trades that
   didn't have a Dukascopy bar within 10min (different bar boundaries,
   broker-specific weekend/holiday handling). The kept sample is still
   1300-2400 trades per product though — large enough.

3. **Need to re-validate with REAL Eightcap volume** before shipping live.
   This is now possible because the exporter MQL5 was modified to write
   `tick_volume`. Once Jay re-exports + sends, we redo this test on
   broker-native data.

## Three deployment options once Eightcap volume is in

**Option A — Binary filter (cheapest, lowest risk)**:
Add `flow_5m_agrees_with_direction` check at the END of the cascade
(after meta gate passes). If sign(flow_5m) ≠ direction → return hold.
- ~5 lines in decide.py
- Server-side compute of flow_5m on the bar feature row
- Expected impact: +13-18pp WR, ~0% to +40% R

**Option B — Feature in retrained models (medium effort)**:
Add `flow_5m` (and maybe `flow_4h` for context) as new V72L features.
Retrain confirm + meta heads. The trees can learn nonlinear gating
(e.g. "high p_conf AND flow_5m disagrees → still hold"). Cleaner than
binary filter but requires 6-9 hours of retrain effort.

**Option C — Position-size modifier**:
lot_mult = 1.0 if 5m agrees, 0.5 if disagrees. Halfway between A and B
in invasiveness.

## Recommendation

Ship **Option A on all 3 products** as soon as Eightcap volume is
available. Even with the worst-case interpretation (most of the gain is
broker-mismatch artifact), the signal is strong enough that even half
of it would be the biggest improvement we've shipped.

Order of work:
1. User recompiles modified `exporter-20-math.ex5` and re-exports XAU + BTC
2. Re-run this test on Eightcap data → validate the dramatic Dukascopy
   result is real (not just broker-quirk)
3. If validated → ship Option A as v9.1 (decide.py addition)
4. Monitor live for 2-4 weeks
5. If healthy → consider Option B (full retrain with flow_5m as feature)

## Files

- `01_port_and_test.py` — Pine→Python port (now also handles `volume` column properly in 4h MTF)
- `04_real_volume_test.py` — this script (reads Dukascopy CSV, runs filters)
- `/tmp/duk/{xau,btc}_m5.csv` — Dukascopy data (gitignored)
- This memo
