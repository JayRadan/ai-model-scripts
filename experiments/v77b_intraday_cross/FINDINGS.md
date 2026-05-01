# v7.7b Intraday Cross-Instrument — Findings

**Date**: 2026-05-01
**Status**: ❌ Failed worse than v77 daily. Macro regime drift between train and holdout produces anti-predictive model.

## What we tested

The natural follow-up to v77 daily: replace daily features with 1h
intraday features (last 1h / 4h / 24h returns + 24h-z) for 5 instruments
(EURUSD, SPX, TNX, VIX, UUP). Hypothesis: gold reacts to intraday
DXY/SPX/VIX moves within an M5 trade's lifetime, so intraday features
should beat daily.

yfinance 1h data limit: 730 days. Training window for Midas:
2024-05-05 → 2024-12-11 (~407 trades, 7 months).

## Results

### Trade-level correlation flipped sign vs in-sample

```
in-sample (Midas train):  corr(p, pnl_R) = +0.279
holdout Midas:            corr(p, pnl_R) = -0.040   ← flipped
holdout Oracle:           corr(p, pnl_R) = -0.042   ← flipped
```

The classifier didn't just fail to generalize — it **anti-correlates**.
Predicted-good trades did slightly worse than predicted-bad on holdout.

### Gate sweep: significance-failed at every threshold

All p-values ≥ 0.61 except the very tail (Oracle thr=0.75: p=0.16,
83 trades). Gate PF is **below baseline** at most thresholds:

| Product | Best gate PF | Baseline PF | Worst gate PF |
|---|---:|---:|---:|
| Midas  | 3.19 (105 trades, p=0.087) | 2.30 | 1.79 (459 trades) |
| Oracle | 4.64 (83 trades, p=0.158)  | 3.48 | 2.82 (575 trades) |

ΔR is massively negative everywhere.

### Sizing made things worse

A_linear sizing scheme (best on v77 daily Oracle) on intraday:
- Midas: R 4355 → 4150 (-205), PF 2.30 → 2.26
- Oracle: R 3817 → 3635 (-182), PF 3.48 → 3.42

Net loss on both products.

## Why this failed worse than daily

Look at the top coefficients trained on May→Dec 2024:

```
spx_ret_24h    +0.73   "SPX up = good gold day"
tnx_ret_24h_z  -0.63   "yields up = bad gold day"
vix_ret_24h    -0.41   "VIX up = bad gold day"
```

Two of these are **textbook gold dynamics that BROKE in 2025**:
1. Gold and SPX rallied together in 2025 on macro fears (fiscal,
   geopolitics). The "SPX up → bad gold" relationship reversed.
2. Yields rose along with gold in 2025 (debt concerns). The "yields
   up → bad gold" relationship reversed.
3. VIX stayed compressed while gold rallied. Inverse correlation
   with safe-haven demand decoupled.

The model learned a 2024 macro structure and applied it to a 2025-2026
holdout where the structure had inverted. **Cross-instrument relationships
themselves drift across regimes.** The thing we hoped would solve regime
drift is itself regime-dependent.

## Combined v77 + v77b takeaway

After running both, the picture is:

| Variant | Train period | corr holdout | Gate sig | Sizing R | Sizing PF |
|---|---|---:|---|---:|---:|
| v77 daily   | 2024-01 → 12 | +0.02/+0.04 | Oracle thr=0.70 p=0.001 | -2.5% / -2.0% | +0.05 / +0.02 |
| v77b 1h     | 2024-05 → 12 | -0.04/-0.04 | none significant         | -4.7% / -4.8% | -0.04 / -0.07 |

v77 daily was marginal-positive due to longer training window covering
more of 2024's mixed regime. v77b intraday's shorter window captured
only the tail of 2024's downtrend regime, then crashed when 2025 flipped.

## What this rules out

We can now state with reasonable confidence:
1. **Cross-instrument features alone do not reliably predict M5 XAU
   trade quality** in a way that survives regime drift.
2. **More data resolution doesn't fix the problem** — intraday made it
   worse, not better, because the underlying macro relationships shift.
3. **Static cross-instrument classifiers won't ship.** Any deployment
   would need online retraining on a rolling window, since the
   2024-trained model was already anti-predictive 6 months later.

## What's left from the v76 "what would plausibly help" list

1. ~~Cross-instrument signal~~ — TESTED v77 (marginal) and v77b (failed).
2. **Longer-timeframe model** — train a separate D1 confirm head, use
   as high-confidence (≥95%) directional veto only. Different
   mechanism: the model itself is the prediction, not a regime
   wrapper around it. Untested.
3. **Online learning / drift adaptation** — retrain confirm-head meta
   gate on rolling 6-month window. Whichever sub-regime is currently
   dominant gets more weight. Untested. v77b's failure mode argues
   for this — static cross-instrument classifiers can't survive drift,
   but a rolling retrain might.

## Recommendation

Of the two remaining ideas, **online retraining (#3) is the most
defensible**. It directly addresses the failure mode we just observed
(macro relationship drift). It's also the lowest-risk to ship: we
already trust the trained heads; adding rolling-window retraining
on the meta gate is an additive change, not a new model.

The longer-timeframe model (#2) requires building a D1 confirm head
from scratch, which is more work and offers no guarantee that D1
predictions transfer cleanly to M5 entries (we already showed in v75
that daily-veto from XAU itself doesn't help).

## Files retained

3 scripts + this memo. Intraday parquets (~3MB) kept for any
follow-up online-learning experiment that wants to use them as features
*inside* a rolling-retrain meta gate rather than a standalone classifier.
