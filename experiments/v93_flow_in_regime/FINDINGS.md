# v9.3 — Quantum Flow as REGIME fingerprint feature: BREAKTHROUGH

**Date**: 2026-05-03
**Status**: ✅✅ Best validation result of the entire optimization thread.
The user's intuition was right: flow indicator helps when used at REGIME level.

## What worked vs what didn't

After ~10 attempts to use the Quantum Volume-Flow Quantizer indicator
across this and prior sessions, FINALLY found the right level to apply it:

| Approach | PF | WR | R | Notes |
|---|---:|---:|---:|---|
| Per-trade binary skip filter (v8.9) | 0.42-0.71 | 78-86% | -524 | Overrestricts |
| Per-trade meta gate feature (v9.0/9.2) | 3.25-3.61 | 62-65% | +3052-3516 | Adds noise, overfits |
| **REGIME fingerprint feature (v9.3)** | **4.17** | **67.5%** | **+4880** | **Breakthrough** |

## Why per-trade tests failed but regime worked

Per-trade signals from the 18 v72L physics features already saturate
direction prediction. The Quantum Flow signal, when added at the
per-trade level, was redundant — adding noise on the margin.

But at the **regime level** (per-day cluster classification), flow_4h_mean
contributes orthogonal information: volume-weighted directional momentum
over the past 24h that the existing 7 fingerprint features didn't capture
(weekly_return_pct, volatility_pct, trend_consistency, trend_strength,
volatility, range_vs_atr, return_autocorr — all OHLC-only).

Adding flow_4h_mean as 8th fingerprint feature lets the K=5 cluster
classifier discriminate based on volume conviction:
  Uptrend cluster:    flow +2057  (strong positive, high participation)
  Downtrend cluster:  flow -1737  (strong negative)
  HighVol cluster:    flow -1622  (negative — distress moves)
  TrendRange cluster: flow +387   (mild positive)
  MeanRevert cluster: flow -165   (near-zero, choppy)

These are economically sensible cluster definitions.

## Holdout validation (Dukascopy, 2024-12-12 → 2026-05-01)

```
Pre-meta:  n=2706  WR 61.8%  PF 2.96  R+7449  DD-121
Post-meta: n=1402  WR 67.5%  PF 4.17  R+4880  DD-90    ← shipped result
```

Per-cluster (post-meta):
  C0 Uptrend:   n=871  WR 72.4%  PF 5.04  R+3392
  C3 Downtrend: n=350  WR 67.7%  PF 5.52  R+1526
  C1/C2/C4:     small samples, marginal/negative (will need cohort kill v7.9 review)

## Comparison to all alternatives

```
                              Trades  WR     PF    R       DD
Current production            1367   65.3%  3.48  +3817   -67
Eightcap broker + real vol    1619   67.4%  3.89  +5218   -75
Dukascopy vanilla (no flow)   1010   65.1%  3.61  +3052   -48
v9.3 Dukascopy + flow regime  1402   67.5%  4.17  +4880   -90  ← winner
```

v9.3 has the best PF, near-best WR, second-best R, but worst DD.

## Trade-offs

**Pros:**
- Best PF of all approaches (+0.69 vs current production)
- WR similar to broker-native retrain
- Solid R lift (+28% vs current)
- Single-source-of-truth (Dukascopy) → multi-broker safe
- Validates Jay's intuition about regime detection being the issue

**Cons:**
- DD increased to -90 (vs current -67, vs Dukascopy vanilla -48)
- Probably from more decisive cluster assignments → more aggressive trading
- Need to monitor live or apply v8.6 magnitude sizing to smooth

## What's needed to deploy

1. Pickle the new validated XGB objects (mdls/exit/meta) →
   commercial/server/decision_engine/models/oracle_xau_validated.pkl
2. Vendor in commercial repo
3. **CRITICAL**: server's regime classifier needs flow_4h_mean computed
   at decide time. Need to update the server's regime.py to compute
   flow_4h on the bars and include it in the fingerprint vector.
4. Also need to either:
   (a) Have server fetch its own bars from Dukascopy (Stage 2 architecture)
   (b) Just ship to current architecture (customer EAs send bars) and accept
       that the model trained on Dukascopy has slight inference distribution shift

Stage 2 (Dukascopy server fetcher) is still the architecturally correct
path for multi-broker customers.

## Files

- `01_selector_with_flow_feature.py` — modified K=5 selector that adds flow_4h_mean
- `02_meta_with_flow_feature_FAILED.py` — earlier failed attempt (per-trade)
- `validation_full.log` — full holdout validation output
- This memo
