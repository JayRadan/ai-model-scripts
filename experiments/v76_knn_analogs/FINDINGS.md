# v7.6 kNN Analog Regime Detector — Findings

**Date**: 2026-05-01
**Status**: ❌ Hypothesis disproven. No deployment.

## What we tested

Pattern-based regime detector (your idea): for every M5 bar, take the
trailing 288-bar (24h) close shape, z-normalize, downsample to 24-D,
search the pre-2024-12-12 history for the top-100 most-similar windows,
look at what those analogs did in their next 4h.

If the analog forward returns clustered in one direction → directional
signal → veto contradicting trades.

## Build

```
00_build_index.py   → 82,094 indexed historical windows (2018→2024-12-12),
                      every 6th bar, BallTree over 24-D z-normalized shapes
01_query_holdout.py → for every Oracle+Midas holdout trade timestamp
                      (3,419 unique), find top-100 analogs
02_apply_veto.py    → sweep veto thresholds on both rules, report PF impact
```

## Results

### Distribution of analog forward direction

```
pct_neg_4h (fraction of K=100 analogs going down >0.1% in next 4h):
   10%-ile: 0.24      (i.e. even 'bearish-leaning' bars only had 24% down-moves)
   50%-ile: 0.32
   90%-ile: 0.41
   95%-ile: 0.43
   max:     <0.60
```

**No query found a strong bearish bias (≥60%)** from its historical analogs.
Same for bullish. The analogs are essentially balanced — knowing today's
24h shape doesn't tell you what the next 4h will do.

### Veto effect on holdout PF

**Oracle XAU baseline**: PF=3.48, R=+954.3, DD=-16.7

| Rule | Threshold | Blocked | New PF | New R | ΔR |
|---|---:|---:|---:|---:|---:|
| pct ≥0.40 | bear blk | 263 | 3.52 | +760.5 | **-193.8** |
| pct ≥0.45 | bear blk | 74 | 3.64 | +921.7 | -32.6 |
| pct ≥0.50 | bear blk | 22 | 3.52 | +941.3 | -13.0 |
| mean<-0.001 | mean blk | 29 | 3.50 | +937.7 | -16.7 |

Same pattern on Midas — every veto setting either (a) blocks too few trades
to matter or (b) blocks trades that net made profit.

### Sanity test (the kill shot)

Correlation between analog-signal-direction and trade pnl:
```
Oracle: corr = -0.004    ← effectively zero
Midas:  corr = +0.027    ← essentially zero
```

A useful predictor needs correlation ≥ ~+0.05. Both are < +0.03. The
analog signal is **statistically uncorrelated with trade outcome**.

## Why this fails

Three converging reasons:

1. **288-bar shapes are too noisy at M5**. Two charts can look identical
   for 24h and then diverge wildly because the proximate cause of the
   next move is sub-1h microstructure, not 24h shape.

2. **Z-normalization removes the level information** (intentionally — to
   match shape). But the next 4h direction is partially driven by
   absolute level (overbought/oversold relative to longer history),
   which we've thrown away.

3. **Reference market doesn't repeat**. Gold's character has changed
   dramatically since 2020 (lockdowns, inflation, geopolitics). 2018-2020
   analogs may simply not apply to 2025 patterns. The index has them
   anyway, diluting signal with stale matches.

## Combined v75 + v76 takeaway

After **5 separate experiments** (K-means K=5..8, HMM N=3..5, daily
veto threshold sweep, kNN analog), **no regime overlay improves PF on
the validated 15-month holdout**. The trained Oracle/Midas confirm
heads already extract what's available at M5. Today's losses are
within the validated DD of -16.7R — normal model variance.

## What WOULD plausibly help (ranked, but none tested)

1. **Cross-instrument signal** (DXY/SPX/US10Y) — feature space orthogonal
   to anything we've used. Gold reacts to dollar/risk-off, and those
   live in other instruments' M5 bars. Genuinely new information, not a
   re-hash of XAU's own past.

2. **Longer-timeframe model** — train a separate D1 confirm head (one
   prediction per day). Use it as a *high-confidence* veto only —
   block M5 trade when daily model is ≥95% confident in opposite
   direction. The 5% blocked might overlap exactly with the bad days.

3. **Online learning / drift adaptation** — retrain the meta head on a
   rolling 6-month window. Whichever sub-regime is currently dominant
   gets more weight automatically.

These are real hypotheses, not the same regime-overlay idea in a new
costume. Saving for whoever wants to tackle them later.

## Files retained

Just the 3 scripts + this memo + README. Heavy intermediates (BallTree,
parquet vectors, signals) cleared — re-runnable from scripts.
