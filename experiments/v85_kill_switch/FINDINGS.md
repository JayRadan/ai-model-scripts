# v8.5 — Kill-Switch Model: Predict & Exit Hard-SL-Bound Trades Early

**Date**: 2026-05-03
**Status**: ❌ Fails on all 3 products. Don't ship. Confirms v83's similar "kill-switch adds marginal value" finding with stricter math.

## What we tested

Train an XGB classifier that, for each bar after entry (≥ bar 5), predicts
P(this trade ends in hard_sl). If the probability exceeds threshold, exit
immediately at current pnl_R. Replaces the slow drift to -4R hard SL with
an early cut at, say, -1R or -2R.

Per-bar features: bars_held, current pnl_R, velocity (3-bar Δ), max/min
so far, bars_since_max/min, direction, cluster_id.
Label: 1 if trade ended with exit='hard_sl', else 0.

Walk-forward: train on H1 chronological half of holdout, test on H2.
Sweep thresholds {0.50, 0.60, 0.70, 0.80, 0.90}.

## Results — model is highly predictive but kill action is net-negative

```
H2 results per product:

Oracle XAU  (baseline WR 67.4% PF 3.68 R +1985 hard_sl_R -642)
  AUC train 0.980  test 0.837   ← model genuinely predicts hard_sl
  thr 0.50: kill 200 (29%)  WR 57.7% PF 3.18  ΔR -390  (saved +179, killed winners -561)
  thr 0.70: kill 102 (15%)  WR 63.7% PF 3.51  ΔR -146  (+86 / -228)
  thr 0.90: kill  28 ( 4%)  WR 66.8% PF 3.70  ΔR  -13  (+24 / -37)

Midas XAU  (baseline WR 59.9% PF 2.37 R +2665 hard_sl_R -1720)
  AUC train 0.923  test 0.827
  thr 0.50: kill 569 (43%)  WR 46.7% PF 2.32  ΔR -483  (+684 / -1137)
  thr 0.70: kill 326 (25%)  WR 54.6% PF 2.31  ΔR -210  (+324 / -500)
  thr 0.90: kill  26 ( 2%)  WR 59.4% PF 2.35  ΔR  -34  (+21 / -51)

Oracle BTC  (baseline WR 56.6% PF 1.98 R +1898 hard_sl_R -1674)
  AUC train 0.920  test 0.821
  thr 0.50: kill 535 (44%)  WR 47.2% PF 1.99  ΔR -336  (+700 / -972)
  thr 0.70: kill 246 (20%)  WR 53.5% PF 1.95  ΔR  -98  (+226 / -251)
  thr 0.80: kill 113 ( 9%)  WR 55.9% PF 2.00  ΔR  +12  (+97 / -52)  ← only positive
  thr 0.90: kill  36 ( 3%)  WR 56.3% PF 1.96  ΔR  -18
```

**Best result across the entire sweep**: BTC thr=0.80 gives ΔR **+12**
(+0.6% of baseline). Statistical noise. Every other configuration loses R.

## Why this fails — the asymmetric damage

The model is genuinely predictive (test AUC 0.82-0.84). The issue is
that "trade currently in drawdown" features look identical regardless
of whether the trade ultimately recovers or hits hard_sl. The model
catches both.

Math at the per-trade kill level:
- Killing a trade that WAS going to hard_sl: save ≈ 4R - |kill_pnl| ≈ +2R per save
- Killing a trade that WAS going to be a winner: lose ≈ |winner_pnl| - |kill_pnl| ≈ -3R to -5R per damage

For each hard_sl trade the model correctly catches, it incorrectly catches
**more than 1 winner**, and each winner kill costs more than each save
recovers. Even at thr=0.90 (most confident kills only):
- Oracle: 28 kills, +24R saved, -37R damage → net negative

## Comparison to v83's kill-switch result

v83 README said: "Kill-switch adds marginal value: Saves ~200 trades,
minimal PF gain." With our stricter walk-forward + per-trade R accounting,
the result is actually NEGATIVE — the v83 measurement either:
- Used a different evaluation that didn't account for winner damage
- Conflated "saved a hard_sl" with "improved total R" (those are not the
  same — saving a -4R loss but turning a +5R winner into a +1R kill is
  net negative)

Same architectural lesson as v83/v8.3: validating something that's not
what the deployment will actually do produces misleading numbers.

## Why max_hold (60 bars) ISN'T the problem either

Original exit-reason analysis showed:
- max_hold bucket: 54-61% of trades, WR 71-80%, avg +2.8 to +3.8R
- These are profitable trades that just exit when time runs out
- Cutting them earlier would harm them, not help

The whole exit logic is structurally well-tuned. There's no obvious
slack to exploit.

## What this means for "improving exits"

After v83, v8.3 retest, v8.4 (softer guard sweep), and now v8.5
(kill switch), the message is consistent: **the existing exit logic
(ML exit threshold 0.55 + 4R hard SL + 60-bar max hold) is near-optimal
for this trade distribution**.

Any modification we tried either:
- Removes wins (guard exits, kill switch on winners-in-drawdown)
- Doesn't change anything (high-confidence kill switch only catches a
  handful of trades)

The exit head, ML at 0.55, is conservative on purpose — it catches
peaks (WR 88-95% when it fires) precisely BECAUSE it doesn't fire often.
Lowering its threshold or adding a kill switch destroys this property.

## Recommendation

Do NOT ship a kill switch on any product. Stay with current exit logic.

The honest remaining unshipped wins:
1. **v78 magnitude → variance sizing**: validated +3% R, p=0.024,
   risk-neutral. Only validated improvement that survived rigorous testing.

We've now exhaustively tested:
- Bad-trade-removal axis (v75/v76/v77/v77b/v79/v80/v81): cohort kill (v79)
  was the only winner; rest hit the noise floor
- Better-exit axis (v83/v8.3/v8.4/v8.4b/v8.5): all fail for same reason —
  capping/cutting destroys the long-tail winners that carry the system

The v7.9.1 cohort-kill baseline (currently live) is at the local maximum
for this trade distribution + feature set + timeframe. Real upside from
here requires a fundamentally different system, not parameter tweaks.

## Files retained

- `01_train_eval.py` — vectorized trajectory builder + per-bar feature
  matrix + threshold sweep
- `sweep_*.csv` — per-product result tables
- This memo
