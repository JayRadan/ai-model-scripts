# v7.7 Cross-Instrument Regime — Findings

**Date**: 2026-05-01
**Status**: ⚠ Weak real signal. Not ship-worthy as-is. Worth a v7.7b follow-up with intraday cross-instrument.

## What we tested

Hypothesis: cross-instrument state (DXY, SPX, US10Y/TNX, VIX) is
orthogonal information our XAU-only confirm heads can't see. A
classifier trained on cross-instrument features should add real signal.

This is the only one of the four "genuinely new info" candidates we
identified after v75/v76 disproved the regime-overlay-on-XAU-itself
hypothesis. We deliberately tested the highest-EV candidate first
instead of running 4 in parallel.

## Build

```
00_pull_cross_instruments.py    yfinance daily DXY/SPX/TNX/VIX (2018→26-04)
01_build_features_and_labels.py 18 cross-instrument features per date,
                                forward-filled to all calendar days
02_train_and_apply.py           LogisticRegression on Midas pre-2024-12-12
                                (436 trades), apply to holdout
03_significance_test.py         Permutation null at top-quintile gate
04_sizing_test.py               Use predicted prob as size multiplier
```

Train: 436 Midas trades 2024-01-02 → 2024-12-11 (win rate 0.525)
Holdout: 2200 Midas + 1292 Oracle trades, 2024-12-12 → 2026-04-13

## Results

### 1. Trade-level correlation (the bar v76 set)

```
in-sample (Midas train):  corr(p_good, pnl_R) = +0.254
holdout Midas:            corr(p_good, pnl_R) = +0.022   ← noise
holdout Oracle:           corr(p_good, pnl_R) = +0.039   ← noise
```

By the strict v76 bar (≥+0.05 to be useful), it fails.

### 2. Top-quintile gate IS significant on Oracle

Permutation test (5000 random subsets of same size):

| Product | Threshold | Kept | Gate PF | Null mean PF | Null 95th | p-value |
|---|---:|---:|---:|---:|---:|---:|
| Midas  | 0.65 | 428 | 2.61 | 2.32 | 2.76 | 0.128 |
| Midas  | 0.70 | 240 | 2.56 | 2.32 | 2.94 | 0.236 |
| Oracle | 0.65 | 228 | 4.43 | 3.52 | 4.47 | **0.056** |
| Oracle | 0.70 | 107 | **8.04** | 3.59 | 5.26 | **0.001** |

**Oracle p=0.001 at thr=0.70 is the most encouraging result we've gotten
across v75/v76/v77.** It says: of the 1292 Oracle holdout trades, the
107 the model rated highest had PF=8.04, and a random 107-trade subset
hit that PF less than 0.1% of the time.

### 3. But absolute R drops at every gate threshold

Hard gate trades quality for quantity. ΔR is negative everywhere:
Oracle thr=0.70 keeps 107 trades for R=+522 vs baseline R=+3630.
The gate finds quality but throws away too much volume.

### 4. Position sizing is the better mechanism — marginally

Use predicted probability to scale lots instead of binary gate:

**Oracle holdout:**
| Scheme | R | PF | maxDD |
|---|---:|---:|---:|
| baseline lot=1.0          | +3630 | 3.47 | -67 |
| A_linear (0.5 + 1.0·p)    | +3539 | 3.52 | **-56** |
| D_topheavy (0.75 / 2.0)   | +3675 | 3.66 | -97 |

**Midas holdout:**
| Scheme | R | PF | maxDD |
|---|---:|---:|---:|
| baseline lot=1.0          | +4355 | 2.30 | -80 |
| A_linear                  | +4270 | 2.32 | -79 |
| D_topheavy                | +4516 | 2.37 | -123 |

A_linear on Oracle is the only honest win: -2.5% R, +0.05 PF, -16% DD.
Everything else is either flat or trades DD for PF.

### Top model coefficients (which features mattered)

```
dxy_ret_20d       -0.71   ← falling DXY → good gold day (expected)
spx_dist_sma50_z  +0.47   ← stretched SPX above 50d → good for gold
dxy_ret_1d        -0.37
vix_level_z60     +0.37   ← elevated VIX → good for gold (safe haven)
dxy_dist_sma50_z  +0.36
corr_xau_dxy_20d  -0.34   ← stronger negative XAU/DXY corr → good
```

These coefficients are economically sensible — they match textbook
gold dynamics. So the model isn't fitting noise, it's encoding a
real macro structure. The issue is that the structure is **slow-moving
and weak relative to M5 trade variance**.

## Why this only half-works

1. **Daily resolution is too coarse for M5 trades.** Gold can react to
   a DXY tick within minutes; daily features can't capture that.
2. **Holdout regime drift.** Train was 2024 (mixed), holdout includes
   2025-2026 strong gold uptrend (label balance 51% → 67%). Even a
   degraded classifier "looks ok" on a high-baseline holdout.
3. **N=436 training trades is too small** for an 18-feature classifier.
   The Oracle p=0.001 result happened despite that, which is why it
   feels real — but the Midas evaluation correctly punishes the
   small-train problem with non-significance.

## Verdict

- v77 is **not ship-worthy as-is**. The Oracle PF lift is real but
  small, R is flat-to-negative, Midas shows no transferable signal.
- **It is the first regime overlay (across v75/v76/v77) where ANY
  configuration produced a statistically significant top-quintile
  result.** That's a step up from "uncorrelated noise."
- The economic-coefficient sanity check passing is encouraging.

## What WOULD plausibly work (concrete next step)

**v7.7b — Intraday cross-instrument**. yfinance gives 1h cross-instrument
data for the last 730 days, which covers our entire holdout window.
Replace daily features with 1h features (last 4h DXY return, last 1h
SPX move, intraday VIX spike) and re-run. Hypothesis: intraday
cross-instrument moves are what XAU actually reacts to within an
M5 trade's lifetime.

If intraday features still don't beat ΔR≥0 with PF≥+0.10, we accept
the result: cross-instrument context is too slow / too weak to gate
M5 trades. At that point we move on to candidates #2-4 from the
v76 plan (longer-timeframe model, online drift adaptation).

## Files retained

3 scripts + this memo. Cross-instrument parquets in `data/` are
~200KB total — kept for v7.7b reuse.
