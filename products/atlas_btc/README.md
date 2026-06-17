# Atlas BTC — M15 Macro Kalman + M1 U-shape Reversal (Bitcoin)

> **Version:** **ushape_m15** (deployed 2026-06-17) — replaces prior STRICT-candle architecture
> **Architecture:** M15 Kalman macro regime + M1 Kalman U-shape edge-detected reversal (mirror of Atlas XAU deploy)
> **Bundle:** `atlas_btc_validated.pkl` (40-feature Q with M15 + M1 Kalman state)
> **Entry rule:** BUY iff M15 kf_dir=+1 AND M1 kf_dir=−1 AND M1 kf_v<0 AND M1 f_accel>0 AND edge-bar; SELL mirror.
> **Q threshold:** Q ≥ **1.5** (q_model_holdout used at inference for calibration consistency)
> **Exit:** SL 6×ATR · TRAIL 1.0×ATR · MAX_HOLD 300 · BE@+0.5R · 4 slots
> **Rollback:** `cp atlas_btc_validated.pkl.bak_pre_m15_ushape_2026-06-17 atlas_btc_validated.pkl` + revert config flag.

## 🆕 2026-06-17 — Full architecture replacement: ushape_m15

Replaces the prior STRICT-2-bar reversal candle with M15-macro + M1 U-shape edge
(same change as Atlas XAU deploy, commit 7bf5a37).

| | Before (strict_candle) | Now (ushape_m15) |
|---|---|---|
| Macro regime | M1 TFK committed_dir | **M15 Kalman kf_dir** |
| Entry trigger | strong bear/bull prev bar + close past lines | **M1 Kalman U-shape edge** |
| 8mo backtest PF | 1.31 | **1.58** (+21%) |
| 8mo backtest DD | ~95R | **57R** (−40%) |
| Trade rate | ~7/day | ~6/day @ Q≥1.5 |

### 3-day live sim (06-14 → 06-17) — head-to-head

| Date | OLD strict_candle $ | NEW ushape_m15 $ |
|---|---:|---:|
| 06-14 | +$6,890 | +$2,190 |
| 06-15 | +$8,365 | +$3,261 |
| 06-16 | −$8,423 | −$15,084 |
| 06-17 | −$9,543 | **+$11,677** |
| **TOTAL** | **−$2,710** | **+$2,044** |

NEW wins by $4,754 over 3 days. ($ figures inflated by R_to_USD calibration —
actual likely 50-100× smaller; PF/sumR comparisons are valid.)

### 14-day live sim (06-04 → 06-17)

75 trades · 8 green days vs 5 red days · +$3,761 @ 0.10 lot (scale inflated)

### Files changed in this deploy

| File | Change |
|---|---|
| `commercial/server/decision_engine/configs/atlas_btc.py` | `entry_mode="ushape_m15"`, `macro_tf_min=15`, `q_thr=1.5` |
| `commercial/server/decision_engine/models/atlas_btc_validated.pkl` | New 40-feature bundle (backup: `.bak_pre_m15_ushape_2026-06-17`) |
| `products/atlas_btc/scripts/03_train_q_production.py` | Replaced with M15 U-shape training recipe |
| `products/atlas_btc/scripts/10_sim_today.py` | Rewritten for new architecture |

Server-side `decide_atlas.py` and `atlas_features.py` already support `ushape_m15`
from the XAU deploy (commit 7bf5a37) — only config flag flip needed for BTC.

EA changes: **none**. ATB- magic numbers, slot management unchanged.

---

## Pre-2026-06-17 strict_candle config (archived)

> **Prior version:** STRICT 2-bar reversal candle + Kalman/TFK confluence
> **Prior bundle:** `atlas_btc_validated.pkl.bak_pre_m15_ushape_2026-06-17` (54 features)

## Strict_candle deployed config (archived for reference)

| Param | Value | Notes |
|---|---|---|
| `strong_body_atr` | 0.8 | |
| `kf_age_min` | **3** | raised 1 → 3 on 2026-06-12 after −$107 chop day (kage=1 let too many fast Kalman flips through) |
| `require_both_lines` | True | |
| `q_thr` | **1.5** | lowered 2.0 → 1.5 on 2026-06-17 (more activity pivot) |
| `time_block_utc` | (0, 0) | disabled — 24/7 market |
| `trend_slope_block` | 0.0 | disabled |
| `sl_hard_atr` | 6.0 | |
| `trail_atr` | **1.0** | tightened 1.5 → 1.0 on 2026-06-16 (DD −51% on 8mo holdout, config-only no retrain) |
| `use_orderflow` | True | |
| `max_concurrent` | 4 | |

---



## 🧠 2026-06-09 — kf_age_min 3 → 1 + q_thr 3.0 → 2.0

Diagnosis on today's XAU sharp fall (same Atlas architecture as BTC) showed
the `kage ≥ 3` requirement was filtering out the 1-2 bar Kalman flips that
happen during sharp impulse pullbacks. BTC is the most-affected product
because it's 24/7 with frequent sharp moves.

8-month holdout sweep (per-product, multi-pos sim):

| Variant | Trades | WR | PF | DD | $@0.10 |
|---|---:|---:|---:|---:|---:|
| kage≥3 Q≥3.0 (was deployed) | 2,522 | 71.1% | 1.38 | 128 | $34,386 |
| **kage≥1 Q≥2.0 (deployed)** | **4,308** | **68.7%** | 1.28 | 142 | **+$45,172** |

**+31% $ at +11% DD** — biggest absolute lift of the deployment. WR dips
2.4pp but trade volume rises 71%, so absolute winners grow meaningfully.


## 🧠 2026-06-09 — No filter applied

Backtest sweep tested time-filter and trend-gate variants — **none beat
baseline**. Atlas BTC's STRICT 2-bar reversal pattern already filters trend
context internally; additional filters cut volume without removing losers.

Both config fields exist (`time_block_utc = (0, 0)`, `trend_slope_block = 0.0`)
but are disabled. Baseline ($1,506 @0.10 on 30d unseen) is the current best.


## Holdout (post-Sep 2025, 8 months unseen)

| Q | trades | WR | PF | sumR |
|---:|---:|---:|---:|---:|
| 1.0 | 4,675 | 72.3% | 1.32 | +2,401 |
| 2.5 | 3,710 | 72.5% | 1.36 | +2,112 |
| **3.0** | **2,779** | **72.7%** | **1.37** | **+1,660** |
| 4.0 | 960 | 73.8% | 1.43 | +626 |

Q=3.0 chosen for best PF at usable trade volume.

## Entry rule (STRICT)

Identical to Atlas XAU — see `products/atlas_xau/README.md` for details.

## Training data

- `data/m1_btc_orderflow.parquet` (741k bars, Nov 2024 → May 2026)
- 7,855 MFE≥2R candidates (72.8% of all candidates have a ≥2R favourable move)

## Scripts

| Script | Purpose |
|---|---|
| `scripts/10_sim_today.py` | Pull fresh Dukascopy bars, simulate today's trades with deployed config |
