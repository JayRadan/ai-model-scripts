# v8.4 — v8.3 Strategy Implemented Correctly: HONEST VALIDATION

**Date**: 2026-05-03
**Status**: ❌ Strategy fails on Oracle/Midas. Do not ship.

## What we tested

After v8.3 was reverted (it had a routing bug that returned hold on every
Oracle/Midas bar), Jay asked to re-implement the strategy correctly:
guard exit at +0.50R + pivot pre-filter as an additive gate (not a
routing trigger). Goal: WR ≥ 75% with PF preserved near baseline.

## Method

For each holdout trade, we walked bar-by-bar from entry to original exit.
If pnl_R reached +0.50 at any bar, we replaced the exit with the guard
(pnl_R = 0.50, exit = "guard"). Otherwise we kept the original exit
(including any -4R hard SL hits — guard is profit-taking only, doesn't
affect losers).

Pivot pre-filter: scored each entry with Janus's pivot model (32 features
including v72L + 14 Janus-specific). Swept thresholds [0.20, 0.25, 0.30,
0.35, 0.40].

BTC was excluded — `btc_pivot_v2.pkl` includes 'label' in its training
feature list (data leak). Need to retrain BTC pivot before adding it.

## Results — Guard alone (no pivot filter)

```
                  baseline               guard@0.50R only
Oracle XAU:  WR 65.3% PF 3.48 R +3817  →  WR 87.8% PF 0.90 R  -64
Midas XAU:   WR 57.4% PF 2.24 R +5093  →  WR 85.5% PF 0.71 R -470
```

The guard converts ~87% of trades into +0.50R wins, but caps all winners
at +0.50 while losers still hit full hard SL. Result: WR jumps but PF
collapses below 1.0 and total R goes negative.

## Results — Guard + Pivot pre-filter

| Threshold | Oracle WR | Oracle PF | Oracle R | Midas WR | Midas PF | Midas R |
|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 85.3% | 0.67 | -87 | 84.1% | 0.62 | -206 |
| 0.25 | 79.5% | 0.44 | -106 | 84.2% | 0.61 | -113 |
| 0.30 | 73.8% | 0.31 | -89 | 85.4% | 0.64 | -57 |
| 0.35 | 72.9% | 0.27 | -48 | 85.3% | 0.61 | -36 |

The pivot filter drops the trade count by 90% but keeps trades with
NEGATIVE expected value (R/trade -0.83 Oracle, -0.27 Midas). The pivot
model trained for Janus's single-rule cascade actively MIS-selects for
Oracle/Midas's 28-rule cascade. Filter makes it worse, not better.

## Why this fails — the math

Oracle XAU baseline averages +2.79 R per trade. The distribution has a
long winner tail: most trades go to +1-2R, a few go to +4-5R. Those big
winners carry the system.

Capping winners at +0.50R while losers still hit -4R hard SL gives:

```
expected R = 0.88 × 0.50 + 0.12 × (-4.0) = +0.44 - 0.48 = -0.04 per trade
```

Negative. WR can be 88%, system still loses money.

## Why v83 README's numbers were wrong

The README claimed: WR 74.1% / PF 2.06 / R +1011 with same strategy.
Two possible explanations:

1. **The validation was broken** (most likely). Same engineering disconnect
   that made the deployment broken — measured something different from
   what the deployment would actually do.
2. **Numbers were measured on Janus's trade distribution** (single rule,
   PF 2.50 baseline), not Oracle/Midas's 28-rule distribution. Possible
   guard works on Janus, doesn't translate to Oracle/Midas.

## What this proves about WR-targeting

The PF-WR identity is unavoidable:

```
PF = (WR × avg_win) / ((1-WR) × avg_loss)
```

To hit WR 75% with PF 2.0, avg_win/avg_loss must be ≤ 0.67. Oracle's
current ratio is 1.85. You'd need to halve the winners (guard does this →
kills R) or double the losses (unsafe SL widening). No free path exists.

## Recommendation

Do NOT ship v8.4. Options:

1. **Stay at v7.9.1 baseline** (current production): WR 65% / PF 3.48 /
   profitable. Validated and live.
2. **Ship v78 magnitude sizing** as the only validated unshipped win:
   +3% R, p=0.024, risk-neutral.
3. **Soft guard** (e.g. +1.5R or +2R instead of +0.50R) to preserve some
   winner upside. Worth testing as v8.4b. Might give WR 70% / PF 2.5
   instead of 88% / 0.9.
4. **Different system entirely** — different timeframe, different rules,
   naturally different winner distribution. Months of work.

Definitely do NOT ship guard@0.50 + pivot filter as designed. Validation
is honest this time and shows the strategy loses money.

## Files retained

- `01_validate.py` — bar-by-bar guard simulation + pivot threshold sweep
- `sweep_oracle_xau.csv`, `sweep_midas_xau.csv` — full sweep results
- This memo
