# v8.9 — Quantum Volume-Flow Tiebreaker

**Date**: 2026-05-03
**Status**: ❌ Holdout validation killed it. The live-week result was coincidence. Do not ship.

## What this is

Port of Jay's Pine v6 "Quantum Volume-Flow Quantizer" indicator to Python:

```
ha_close  = (O+H+L+C)/4
ha_open   = (O[-1] + C[-1]) / 2
trend     = ha_close - ha_open
vol_ratio = volume / SMA(volume, 50)
raw       = trend × vol_ratio × 1000
flow      = EMA(raw, 21)
flow_q    = round(flow / (ATR×0.5)) × (ATR×0.5)
```

A **smoothed volume-weighted Heikin-Ashi momentum** with ATR-step quantization.
Computed on M5 (current TF) and 4H MTF; positive = upward bias, negative = downward.

## Test on live window (April 27 → May 1)

Replayed all 70 XAU broker trades, looked up indicator at the bar prior to each entry, checked whether the indicator's sign agreed with the broker's trade direction.

```
                      n_kept  n_won    WR     total_$    avg_$
baseline (no filter)     70     20    28.6%   -524.51   -7.49
5m flow agrees            46     12    26.1%    -40.87   -0.89
4h flow agrees            31      8    25.8%    +55.20   +1.78  ← 
BOTH agree                 0    -      -        -        -
```

**The 4h-flow agreement filter would have turned the week from -$524 to +$55.**
That's a $580 swing on 5 days of trading.

**BUT note** the WR is essentially unchanged (28.6% baseline vs 25.8% filtered).
The improvement isn't in win rate — it's in **rejecting the largest losers**.

## What's actually happening

The 5m and 4h flows almost never agree with each other on the same bar (0/70 BOTH-agree). Both signals are computed on the same instrument but at different time scales, and during a choppy week they disagree constantly.

The 4h filter blocks trades whose direction conflicts with the 4-hour trend. The biggest live losers (-$134, -$117, -$115, -$107, -$103) were all SELL trades during 4h uptrend windows — exactly the trades the EA shouldn't have taken if the 4h flow had been consulted.

This is consistent with the diagnosis from the regime-classifier analysis: on ambiguous days the K=5 selector flips between Uptrend/Downtrend cluster assignments and the rules fire on both sides. **The 4h flow is a higher-time-scale anchor that prevents flipping.**

## Two important caveats

1. **Tiny sample** — 70 trades over 5 days. Could be noise / luck.
2. **Tick volume on retail brokers is unreliable** — MT5's `volume` field is tick count, not real traded volume. We used `spread` as a proxy in this test (since the export didn't include volume). Real volume would change the signal.

## Where this could realistically help

If the holdout validation confirms the live-window result, this becomes a **direction-confirmation gate**:

```
After meta gate passes, BEFORE returning Decision(action="open"):
  if sign(flow_4h_at_fire_bar) != sign(direction):
      return Decision(action="hold", reason="4h flow disagrees")
```

This would be additive to v7.9 cohort kill — not a replacement. It addresses a different failure mode: **trades that fire against the 4h trend on regime-ambiguous days**.

Conservative estimate of impact (if confirmed on holdout):
- Cuts trade count by ~50% (filter is strict)
- Maintains or slightly drops WR (it's a direction filter, not a win-rate filter)
- Should reduce DD significantly (blocks the big-loss trades that go against 4h)
- Net R impact unknown without holdout sim

## Holdout backtest result — FILTER FAILS

Applied 4h-flow direction-agreement filter to ALL holdout trades:

```
                  baseline (no filter)        with 4h-flow agree filter
Oracle XAU     n=1367 WR 65.3% PF 3.48 R+3817   →   n=862  WR 66.2% PF 3.42 R+2298  (ΔR -1520, -40%)
Midas XAU      n=2636 WR 57.4% PF 2.24 R+5093   →   n=1484 WR 58.4% PF 2.42 R+3178  (ΔR -1915, -38%)
Oracle BTC     n=2439 WR 54.9% PF 1.84 R+3298   →   n=1284 WR 52.5% PF 1.77 R+1677  (ΔR -1621, -49%)
```

The skipped trades (where 4h flow disagreed) actually performed
**comparably or better** than the kept ones:
- Oracle XAU skipped: WR 63.8%, +1520R (almost as good as kept set)
- Midas XAU skipped:  WR 56.2%, +1915R
- Oracle BTC skipped: WR **57.7%**, +1621R (BETTER than kept set's 52.5%)

The filter throws out 37-47% of trades for tiny WR gain and large R loss.

## Why this fails — same failure mode as v77b/v85/v8.7

The Quantum Flow indicator is essentially a **smoothed momentum proxy**.
Like every other "directional confirmation" filter we've tested at
per-trade granularity:

1. It has marginal predictive AUC (~0.55)
2. The cost of false-positives (winners blocked) outweighs the benefit
   of true-positives (losers blocked)
3. Tiny WR improvement, large R loss

The live-week result (+$580 swing) was a coincidence on 70 trades.
Holdout's 6,442 trades show the true effect: net negative.

## What this rules out

- Tick-volume-based momentum signals don't add precision beyond what
  the meta gate already extracts
- The 4h MTF anchor sounds appealing but doesn't generalize across
  17 months of XAU and 24 months of BTC

This is the 7th "direction filter" we've tested across v75/v76/v77/v77b/
v85/v8.7/v8.9 — all fail the same way. The pattern is unambiguous:
**per-trade direction filtering is not a viable axis on M5 with our feature set.**

## Follow-up: 5m-only filter (better than 4h)

Tested same agreement filter using only 5m flow (not 4h MTF):

```
                  baseline                       with 5m flow agree
Oracle XAU      WR 65.3% PF 3.48 R+3817   →   WR 66.5% PF 3.71 R+2244 (-41%R, +0.23 PF)
Midas XAU       WR 57.4% PF 2.24 R+5093   →   WR 59.5% PF 2.64 R+3155 (-38%R, +0.41 PF)
Oracle BTC      WR 54.9% PF 1.84 R+3298   →   WR 57.5% PF 2.59 R+1214 (-63%R, +0.75 PF)
```

**5m filter beats 4h on every product** (4h: PF -0.06/+0.19/-0.07; 5m: +0.23/+0.41/+0.75).

But same R-trading-down trade-off:
- Drops 44-77% of trades
- Loses 38-63% of total R
- Skipped trades are themselves profitable (+1574/+1938/+2083 R)

Not what Jay originally wanted (same R + higher WR). Different axis: smoother
equity curve at cost of total profit. Most interesting on BTC: PF crosses 2.0
(1.84 → 2.59). Oracle/Midas not worth the R cost. Flagged as opt-in candidate
for BTC if Jay wants smoother equity over max profit.

## Files retained

- `01_port_and_test.py` — Pine→Python port + live-window replay
- `02_holdout_backtest.py` — 4h-flow filter holdout backtest
- `03_holdout_5m_only.py` — 5m-flow filter holdout backtest
- `trades_with_flow.csv` — live window
- `holdout_*.csv` — full-holdout per-trade flow values + agreement flags
- This memo

## Files retained

- `01_port_and_test.py` — Pine→Python port + live-window replay test
- `trades_with_flow.csv` — per-trade flow values + agreement flags
- This memo
