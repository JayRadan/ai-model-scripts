# Atlas XAU — M1 Kalman + TFK Dual-Indicator STRICT Reversal

> **Version:** v1 (deployed 2026-06-02)
> **Architecture:** Kalman state-space regime + Hermes TFK confluence + STRICT 2-bar reversal candle
> **Bundle:** `atlas_xau_validated.pkl` (Q-regressor trained on MFE≥2R movers)
> **Entry gate:** STRICT pattern + Q ≥ **2.0** (raised from 1.0 on 2026-06-04)
> **Exit:** SL 6×ATR · TRAIL 2×ATR · MAX_HOLD 300 · BE@+0.5R · 4 slots

## Entry rule (STRICT)

**BUY** when ALL of:
- TFK regime is GREEN (`cdir == +1`)
- Kalman regime is RED, age ≥ 3 bars (`kdir == -1`, `kf_age >= 3`)
- Previous bar was STRONG BEAR (body ≥ 0.8 × ATR, close < open)
- Previous close BELOW Kalman line AND BELOW TFK line
- Current bar prints GREEN (close > open)

**SELL** mirror — TFK RED, Kalman GREEN+age≥3, prior strong bull, etc.

The mover-filter trained Q (only on candidates whose forward favourable
excursion reached ≥2R) doubled trade rate vs baseline while keeping the
edge tight.

## Holdout

| Metric | Value |
|---|---|
| Trades | ~14 trd/day |
| WR | 66% |
| PF | 1.18 |
| sumR | +1,008 R |
| DD | ~249 R |
| Note | Live-equivalent: PF ~1.0-1.2 (HTF look-ahead inflation removed) |

## Scripts

| Script | Purpose |
|---|---|
| `scripts/10_sim_today.py` | Pull fresh Dukascopy bars, simulate today's trades with deployed config |

## Config history

| Date | Change | Reason |
|---|---|---|
| 2026-06-02 | Initial deploy, q_thr=1.0 | Maximize trade rate; PF acceptable |
| 2026-06-04 | q_thr 1.0 → **2.0** | Halve DD on stretched days; small 30d cost (-5.5%) for better post-deploy quality |
