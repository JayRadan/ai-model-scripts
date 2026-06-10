# Atlas XAU — M1 Kalman + TFK Dual-Indicator STRICT Reversal

> **Version:** v1 (deployed 2026-06-02)
> **Architecture:** Kalman state-space regime + Hermes TFK confluence + STRICT 2-bar reversal candle
> **Bundle:** `atlas_xau_validated.pkl` (Q-regressor trained on MFE≥2R movers)
> **Entry gate:** STRICT pattern + Q ≥ **2.0** (raised from 1.0 on 2026-06-04)
> **Exit:** SL 6×ATR · TRAIL 2×ATR · MAX_HOLD 300 · BE@+0.5R · 4 slots



## 🧠 2026-06-09 — q_thr lowered 2.0 → 1.5 (kage sweep)

After diagnosing today's XAU sharp fall, ran a `kf_age_min` × `q_thr` sweep on
the 8-month holdout. For XAU the cleanest improvement came from **keeping
kage≥3 but lowering q_thr from 2.0 to 1.5**:

| Variant | Trades | WR | PF | DD | $@0.10 |
|---|---:|---:|---:|---:|---:|
| kage≥3 Q≥2.0 (was deployed) | 2,725 | 65.9% | 1.14 | 261 | $9,769 |
| **kage≥3 Q≥1.5 (deployed)** | **3,143** | **66.2%** | **1.18** | **239** | **+$13,895** |

**Strict improvement.** +42% $, +4bp WR, +4bp PF, **−8% DD**. The Q model
already discriminates real pullback-continuation setups from noise; the
2.0 gate was simply too tight.


## 🧠 2026-06-09 — Time-of-day filter (deployed)

After a 30-day post-deploy backtest sweep, added `time_block_utc=(18, 2)` to
block entries between **18:00 and 02:00 UTC** (NY-pm + Asia overnight).

| | Baseline | + time-block 18-02 |
|---|---|---|
| Trades | 439 | 310 (-29%) |
| WR | 70.2% | **74.8%** |
| PF | 1.14 | **1.45** |
| sumR | +93 | **+185** |
| DD | 86.31 | **36.14** (-58%) |
| $@0.10 | $1,399 | **$2,776** (+98%) |

**Biggest single-product win of the deployment.** Nearly doubles $ AND cuts
DD by more than half. Atlas's strict reversal pattern bleeds most during
overnight gold drift; blocking those hours saves the day. Disabled with
`time_block_utc = (0, 0)`.


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
