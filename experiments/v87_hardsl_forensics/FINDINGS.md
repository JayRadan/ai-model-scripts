# v8.7 — Hard-SL Trade Forensics

**Date**: 2026-05-03
**Status**: 🟡 Real physics signal found, but not exploitable as a filter or sizer at the precision required. Documents the failure mode for future reference.

## Goal

Take all trades that hit -4R hard SL (~20-30% of trades on each product),
study their pre-entry feature signatures with multivariate physics-based
methods, and determine if there is exploitable structure that distinguishes
them from winners — using ONLY features available at fire time.

Different from v8.5 (post-entry kill switch which got AUC 0.83 but lost
money): pre-entry features can't suffer the "winners-in-drawdown look like
losers" confusion because no trade has any drawdown yet.

## Method

For each product (Oracle XAU, Midas XAU, Oracle BTC):
1. Load holdout trades + merge feature rows from setups files
2. Three classes: winners (pnl_R > 0), small losers (-4R < pnl_R ≤ 0), hard_sl (-4R)
3. Univariate analysis: per-feature Mann-Whitney + Cohen's d for winner vs hard_sl
4. Multivariate: PCA + Mahalanobis distance from winner centroid + K-means sub-typing
5. Predictive test: XGB binary classifier on H1, evaluated on H2 (walk-forward)
6. Filter sweep: skip predicted-hardsl entries, measure ΔR
7. Sizing variant: scale lots DOWN (not skip) by predicted hard_sl probability

## Physics-based univariate findings (REAL signal)

### Oracle BTC — strongest signal:
```
feature       cohen_d    p          win_med   hsl_med
vpin          -0.292     0.00000    0.570     0.529
sig_quad_var  -0.242     0.00000    0.694     0.571
hawkes_eta    -0.231     0.00000    0.811     0.455
```

### Oracle XAU:
```
feature       cohen_d    p
sig_quad_var  -0.268     0.00001
vpin          -0.229     0.00058
```

### Midas XAU:
```
feature           cohen_d    p
quantum_flow_h4   -0.193     0.00014
hour_enc          +0.182     0.00006
```

**The pattern (consistent across products):** hard_sl trades fire on bars
with **LOWER ambient market activity**:
- Lower VPIN (less informed trader flow)
- Lower signature quadratic variation (smoother price path)
- Lower Hawkes eta (fewer self-exciting jumps)
- Lower H4 quantum flow

**Physical interpretation**: rule-based entries look for "clean setups" —
patterns like NR4 breakouts, inside breaks, mean reverts. These tend to
fire when the market is QUIET. But sometimes that quiet IS the calm
before a strong directional move — and the rule is firing on the wrong
side of it. "False breakouts" in a regime that's about to trend hard.

This explains why C2 (TrendRange) breakdown rules dominated the v7.9
cohort kill list: trend-range regimes specifically have the
quiet-then-violent pattern.

## Multivariate findings — but NOT exploitable

### Mahalanobis distance from winner centroid

```
                Oracle XAU  Midas XAU  Oracle BTC
winner p50/p95  3.60/6.35   3.18/5.36  3.56/6.12
hardsl p50/p95  3.56/5.95   3.09/5.77  3.41/5.80
```

Hard_sl trades are **NOT outliers** — their distribution overlaps
the winner cloud. They live INSIDE the same feature space, just shifted
slightly toward the "quieter" axis.

### Predictive test (walk-forward H1 → H2)

```
              AUC      Best filter ΔR
Oracle XAU    0.573    -0.9 R (essentially 0 at thr=0.70)
Midas XAU     0.584    +0.5 R (effectively 0)
Oracle BTC    0.581    all negative
```

AUC 0.57-0.58 is **better than random but too low for a binary skip
filter**. Same failure mode as v8.5 / v77b: at every threshold, winners
killed > hard_sl saved.

### Lot-down sizing test (scale rather than skip)

```
              Best PF lift    R cost
Oracle XAU    +0.34 (α=1.0)   -16%
Midas XAU     +0.20           -22%
Oracle BTC    +0.17           -26%
```

Same pattern. Improves capital efficiency (R/lot) marginally but
costs total R. Not worth shipping.

## What this means

**The physics insight is genuine and reproducible**: hard_sl trades cluster
on quiet, low-VPIN, low-signature-variance bars. There IS structural
information that distinguishes them.

**But the structure isn't strong enough to act on at the per-trade level.**
The winner cloud overlaps the hard_sl cloud in the same feature space.
Per-trade decisions (skip / size down) catch winners proportional to
losers caught.

The v7.9 cohort kill already extracts most of what's available from
this signal at the discrete cohort level (specific (cluster, rule) combos
that have systematically high hard_sl rates).

## What this does NOT prove

- It does not prove there's NO exploitable signal in pre-entry features.
  A different feature set (microstructure features computed at sub-bar
  level — e.g. trade-by-trade tick imbalance) might separate the two
  clouds better. We don't have those features.
- It does not prove the rule cascade itself is unfixable. Rebuilding
  the rule library to avoid quiet-bar setups would address this from the
  other side, but is a fundamental redesign, not a config tweak.

## Recommendation

Do NOT ship a hard_sl-prediction filter or sizer. The math is the same
as everything else we've tested in the bad-trade-removal axis — small
real signal, but precision insufficient to overcome the false-positive
cost.

The v7.9 cohort kill (already deployed) and v8.6 variance sizing
(already deployed) extract the actionable parts of this physics insight.
Further per-trade modifications hit the same precision wall.

## Files retained

- `01_analyze.py` — univariate + multivariate + predictive sweep
- `02_size_down.py` — lot-scaling variant (also fails)
- `univariate_*.csv`, `multivariate_*.json`, `filter_sweep_*.csv`
- This memo
