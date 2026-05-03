# v8.4b — Softer Guard Sweep (no pivot filter)

**Date**: 2026-05-03
**Status**: ❌ No guard level meets WR≥75% + PF≥2.0 simultaneously on any product. Strategy is structurally limited by the long-tail winner distribution.

## Method

Bar-by-bar simulation of a guard exit at each level G ∈ {0.50, 0.75,
1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0}R. For each holdout trade, walk
forward from entry; if pnl_R hits +G first, replace exit with the guard
(pnl_R = G). Otherwise keep original exit (including hard SL hits).

No pivot pre-filter (v8.4 proved that hurts more than helps). Pure
exit modification.

## Results

### Oracle XAU — baseline WR 65.3% / PF 3.48 / R +3817 / DD -67

| Guard | WR | PF | R | DD | ΔWR_pp | ΔPF | ΔR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 87.8% | 0.90 | -64 | -120 | +22.5 | -2.58 | -3882 |
| 0.75 | 84.9% | 1.09 | +73 | -72 | +19.6 | -2.39 | -3744 |
| 1.00 | 82.4% | 1.26 | +227 | -67 | +17.1 | -2.23 | -3590 |
| 1.50 | 77.8% | 1.49 | +517 | -48 | +12.4 | -1.99 | -3300 |
| 2.00 | 74.2% | 1.66 | +773 | -53 | +8.9 | -1.83 | -3045 |
| 2.50 | 72.3% | 1.83 | +1047 | -62 | +6.9 | -1.65 | -2770 |
| 3.00 | 71.0% | 2.00 | +1325 | -65 | +5.7 | -1.48 | -2492 |

### Midas XAU — baseline WR 57.4% / PF 2.24 / R +5093 / DD -80

| Guard | WR | PF | R | DD |
|---:|---:|---:|---:|---:|
| 0.50 | 85.6% | 0.71 | -470 | -487 |
| 1.00 | 79.5% | 0.95 | -104 | -221 |
| 1.50 | 75.6% | 1.16 | +400 | -102 |
| 2.00 | 71.2% | 1.24 | +693 | -81 |
| 2.50 | 68.4% | 1.35 | +1105 | -62 |
| 3.00 | 65.7% | 1.42 | +1409 | -70 |

### Oracle BTC — baseline WR 54.9% / PF 1.84 / R +3298 / DD -137

| Guard | WR | PF | R | DD |
|---:|---:|---:|---:|---:|
| 0.50 | 86.5% | 0.77 | -323 | -380 |
| 1.00 | 80.6% | 1.04 | +84 | -128 |
| 1.50 | 75.1% | 1.15 | +345 | -109 |
| 2.00 | 70.0% | 1.18 | +507 | -108 |
| 2.50 | 66.5% | 1.23 | +713 | -91 |
| 3.00 | 63.6% | 1.28 | +923 | -83 |

## Verdict

**No product / no guard level achieves the original target (WR ≥ 75%
AND PF ≥ 2.0 AND R > 0).** The automated selector returned "NO guard
level meets bar (PF≥0.9×base AND R>0)" for all 3 products.

Closest by product:
- Oracle XAU G=3.0R: WR 71.0% / PF 2.00 / R +1325 (R drops 65%, WR up only 6pp)
- Midas XAU: no level reaches PF 2.0 — best is G=3.0 with PF 1.42
- Oracle BTC: same story — best is G=3.0 with PF 1.28

## Why every level loses R

The winner distribution has a long tail. Original Oracle XAU avg = +2.79 R/trade
because some trades go to +4R or +5R. Capping winners at +G:
- All winners that would have been ≥G become exactly G
- Losers still hit full hard SL (-4R)
- Avg winner shrinks; avg loser unchanged
- Even with WR rising 20-30pp, total R drops 65-90%

The PF-WR identity holds: there's no free path from current PF/WR to
your target.

## Optional ship-able point (if you accept the trade-off)

If you DO accept losing ~70% of R for slightly better WR + better DD:
- **Oracle XAU G=2.5R**: WR 72.3%, PF 1.83, R +1047, DD -62
  (vs baseline R +3817, DD -67 — slight DD improvement)

This would NOT improve total profit. It would only smooth the equity
curve. Current baseline produces +3817R; this would produce +1047R.
You'd be paying 73% of your profit for a smoother ride.

I would NOT recommend shipping this for Midas or BTC under any guard
setting — both have insufficient PF headroom.

## Recommendation

Do NOT ship a guard exit on any product. The math doesn't support it.

Real options remaining:
1. Stay at v7.9.1 baseline (current production, profitable)
2. Ship v78 magnitude sizing (+3% R, only validated unshipped win)
3. Build a fundamentally different system with a naturally tighter
   winner distribution (different timeframe, different rules) — months of work

## Files retained

- `01_sweep.py` — bar-by-bar guard sweep simulator
- `sweep_oracle_xau.csv`, `sweep_midas_xau.csv`, `sweep_oracle_btc.csv`
- This memo
