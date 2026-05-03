# v9.0 Step 1 — flow_5m as Oracle XAU Meta-Head Feature

**Date**: 2026-05-03
**Status**: ❌ Negative. flow_5m doesn't add residual signal as a continuous feature. Don't proceed to Step 2 or 3.

## What we tested

Jay proposed using the Quantum Flow indicator at two scales:
- `flow_4h` → as a feature in K=5 regime classifier (cluster training)
- `flow_5m` → as a feature in per-rule confirm + meta gate (trade decisions)

This is a sensible split. But before committing to a 6-9 hour full retrain
(which would break the existing v7.9 cohort kill), ran a 1-hour cheap test:
add flow_5m to the meta head ONLY (1 product, 21 features instead of 20),
see if it carries non-trivial weight + improves holdout PF/WR.

## Method

1. Compute flow_5m on full XAU history (~1M bars)
2. Merge into setups_*_v72l.csv at trade times
3. Chronological 50/50 split of v72l_trades_holdout.csv (1367 trades)
4. Train baseline XGB meta (20 META_FEATS) on first half
5. Train new XGB meta (21 META_FEATS = +flow_5m) on first half
6. Evaluate both on second half — compare AUC, feature_importance, and
   trade outcomes at threshold 0.675

## Result

```
                      Baseline (20 feats)    +flow_5m (21 feats)    Delta
Test AUC                  0.568                  0.558              -0.010
flow_5m importance        —                      0.037 (rank 20/21) —
At thr 0.675:
  trades kept             388/684                382/684            -6
  WR                      72.4%                  71.7%              -0.7pp
  PF                      4.88                   4.65               -0.23
  total R                 +1354                  +1312              -42
```

flow_5m gets a small weight but **slightly hurts** the model. AUC drops, PF
drops, R drops. Adding the dimension lets the model overfit on noise.

## Why this fails

The 18 v72L features already capture the directional bias flow_5m measures:
- `quantum_flow`, `quantum_flow_h4`, `quantum_momentum` — directional momentum
- `vwap_dist` — distance from VWAP
- `wavelet_er`, `entropy_rate` — multi-scale market state
- H1/H4 trend features (in setup features)

flow_5m (smoothed Heikin-Ashi × volume EMA) is redundant. The meta gate
already "knows" what the indicator measures. Adding it as a 21st feature
gives no new info, just noise.

## Why the binary filter "worked" but the feature doesn't

In v8.9b we saw flow_5m as a binary filter improved PF (+0.23 to +0.75)
on holdout. But here as a continuous feature in the meta head, it hurts.

These are different mechanisms:
- **Binary filter**: catches signal in the *tails* (extreme directional
  flows) by hard-thresholding. The +0.23-0.75 PF lift came from rejecting
  trades whose flow was extremely opposite — a small tail effect.
- **Continuous feature**: gradient-boosted trees split on it but can't
  express "if flow_5m is in the bottom 10% AND direction is buy, hold".
  The signal in the tails gets averaged out with the noise in the middle.

So the indicator does have *some* exploitable structure, but only through
a hard-threshold mechanism. As a feature it adds nothing.

## Decision

**Don't proceed to Step 2** (full retrain with flow_5m in confirm heads).
**Don't proceed to Step 3** (regime retrain with flow_4h).

The 6-9 hour effort would likely produce the same result, and Step 3 would
break v7.9 cohort kill (which is shipped and validated to add real WR/PF).

## What's left as plausible action

Two tepid candidates from this thread, neither auto-shipped:

1. **flow_5m binary filter as opt-in for BTC only** (v8.9b finding):
   PF 1.84 → 2.59 (+0.75) at cost of -63% R. If you accept smaller-but-
   smoother BTC equity, could enable per-product. Default OFF.

2. **flow_5m as POSITION SIZER** (untested):
   Like v8.6 variance head, but: lot_mult = clip(0.5 + alpha*sign_agreement, 0.5, 1.5).
   Trades against the indicator get smaller lots; trades with it get bigger.
   Different from binary filter (no skip), different from meta feature
   (still uses tail signal). Could be tested in another session if you want.

## Files retained

- `01_meta_head_with_flow5m.py` — this experiment
- This memo
