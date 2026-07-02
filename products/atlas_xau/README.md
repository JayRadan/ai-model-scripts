# Atlas XAU — now runs the `edge_pullback` engine (was: strict-candle, EDGE DEAD)

> **🆕 2026-07-02 — time-boxed-patience trail DEPLOYED (bundle `edge_pullback_v3_tt30_atlas_xau`).**
> Server-side exit tweak: trail stays 2×ATR for the first **30 bars**, then tightens to **0.75×ATR**
> (`tight_after`/`tight_trail_R` bundle fields, read by `edge_pullback.decide_exit`; EA unchanged, its 2×ATR
> real-time trail remains as backstop). Model/threshold/labels unchanged (label-matched retrain added nothing).
> Rationale: winners resolve fast (median hold ~30 bars); what's still open after the grace window is drift that
> bleeds R and blocks the 1-slot. Honest 8y WF (train-only thr ~11/day, net @ $0.20 spread): dev 2020-24
> **−1,787R → +2,745R (8/9 windows +)**, untouched holdout 2025-26 **+1,080R → +2,301R (3/3, WR 70→74%,
> maxDD 181→130R)**; wins at $0.10/0.20/0.30 spread and in gross. Broad parameter plateau (tighten-after 10-60
> bars × trail 0.5-1.0 all positive); uniform tight trail from bar 1 is ~breakeven — the edge is the time-boxing.
> Also disproven in the same lab: limit entries (0.3/0.5×ATR), confirm-bar entry, q30 quantile gate, **BE-lock
> (hurts here, unlike Oracle)**, SL5, tiered trail. ⚠️ The deployed SL7/T2 exit at $0.20 spread LOST money
> 2020→mid-2024 and only turned positive when XAU M1 ATR tripled in 2025 — the tt-trail was profitable through
> the hard years too. Lab: `experiments/atlas_xau_entry_exit_lab/`. Rollback:
> `models/atlas_xau_validated.pkl.bak_pre_tt_2026-07-02` (+ revert `edge_pullback.py` reads defaults-off anyway).

> **🚀 STATUS 2026-06-30 — REPLACED with the `edge_pullback` engine (commit `064aeee`).**
> The strict-candle XAU strategy was **DEAD** (8y holdout PF 0.95, 6 disconfirmations). It is **gone**.
> Atlas XAU now runs the same **pullback + XGB-expected-R** edge deployed on hermes_dji.
> Server engine: `decision_engine/edge_pullback.py`, routed by bundle `version="edge_pullback_v1_atlas_xau"`.
> **1 slot, ~11 trades/day.** Rollback: `models/atlas_xau_validated.pkl.bak_pre_edge_pullback_2026-06-30`.
>
> **Deployed config:** entry = `committed_dir != 0` AND `|close−tfk_line|/ATR ≤ 1.0`; XGBRegressor predicts gross R,
> take if `pred_R ≥ 0.573`; exit = EA-side **SL 6×ATR + trail 2×ATR**, max_hold 300. Trained through 2026-06-30 (8y, ~740k candidates).
>
> **8-year walk-forward (1pt spread, causal HTF):** same recipe generalises to XAU — standalone +3890R, daily Sharpe ~2.2;
> DJI+XAU portfolio Sharpe **3.6**, daily correlation **0.04** (true diversification).
>
> **🚨 BIGGEST CAVEAT — read this:** this edge **contradicts the prior "XAU M1 is dead" evidence** (PF 0.95 strict-candle, RL
> breakeven, etc.). It used a **0.15R flat spread**, but real XAU spread-in-R is likely ~0.3R (**double**), which could halve
> the XAU edge to near-breakeven. **XAU is the WEAKER, less-trusted of the two deployments** — watch it closely, and be ready
> to revert via the `.bak`. The DJI edge is far more validated. Re-run an XAU-specific cost-sensitivity check before trusting size.

---
<details><summary>Historical (pre-2026-06-30: strict-candle / ushape, EDGE DEAD)</summary>
>
> **Version (HISTORICAL, REVERTED 2026-06-25):** **ushape_m15** (deployed 2026-06-17) — replaced prior STRICT-candle architecture
> **Architecture:** M15 Kalman macro regime + M1 Kalman U-shape edge-detected reversal
> **Bundle:** `atlas_xau_validated.pkl` (40-feature Q-regressor, M15 + M1 Kalman state)
> **Entry rule:** BUY iff M15 kf_dir=+1 AND M1 kf_dir=−1 AND M1 kf_v<0 AND M1 f_accel>0 AND edge-bar; SELL mirror.
> **Q threshold:** Q ≥ **1.5** (q_model_holdout used at inference for calibration consistency)
> **Exit:** SL 6×ATR · TRAIL 1.0×ATR · MAX_HOLD 300 · BE@+0.5R · 4 slots
> **Rollback:** `cp atlas_xau_validated.pkl.bak_pre_m15_ushape_2026-06-17 atlas_xau_validated.pkl` + revert decide_atlas/atlas_features/configs.

## 🆕 2026-06-17 — Full architecture replacement: ushape_m15

Replaces the prior STRICT 2-bar reversal candle (TFK + M1 Kalman confluence
+ strong-body-prev-bar) with a regime-disagreement reversal pattern:

| Component | Before (strict_candle) | Now (ushape_m15) |
|---|---|---|
| Macro regime | M1 TFK committed_dir | **M15 Kalman kf_dir** (causal forward-fill) |
| Entry trigger | Strong bear/bull prev bar + close past Kalman line + current bar follow-through | **M1 Kalman U-shape edge** (velocity bottomed, accel turning) against macro |
| Direction | Opposite to TFK committed_dir | Same as M15 macro (catches reversal back to macro trend) |
| Trade rate (8mo backtest) | ~9-14/day | ~9/day @ Q≥1.5 (~18/day @ Q≥1.0) |
| PF (8mo holdout) | 1.23 | **1.24** (similar) |
| $ @ 0.10 lot (8mo) | $13,895 | **$21,912** (+58%) |
| DD | 108R | 153R (+42%) |
| WR | 69.8% | 72.4% |

### 14-day live sim validation (2026-06-03 → 2026-06-17)

170 trades · WR 72.9% · sumR +69.5R · **+$3,215 @ 0.10 lot** · ~14 trd/day
· 8 green days vs 4 red days · max peak-to-trough DD ~$1,100

| Date | Trades | WR | $@0.10 |
|---|---:|---:|---:|
| 06-03 | 24 | 45.8% | −$1,186 |
| 06-05 | 29 | 86.2% | +$1,963 |
| 06-09 | 17 | 94.1% | +$1,440 |
| 06-10 | 43 | 76.7% | +$1,651 |
| 06-11 | 6 | 50.0% | −$1,103 |
| 06-17 | 5 | 60.0% | −$97 |

### Files changed in this deploy

| File | Change |
|---|---|
| `commercial/server/decision_engine/atlas_features.py` | Added M15 Kalman resample + causal forward-fill (`kf_p_m15`, `kf_dir_m15`, `kf_v_m15`, `f_accel_m15`, `f_velPct_m15`) + `dist_m15kf` + `kv_pos_50` |
| `commercial/server/decision_engine/decide_atlas.py` | Added `_ushape_macro_signal()`; dispatch via `cfg.entry_mode`; uses `q_model_holdout` at inference for `ushape_m15` |
| `commercial/server/decision_engine/configs/atlas_xau.py` | `entry_mode="ushape_m15"`, `macro_tf_min=15`, `q_thr=1.5` |
| `products/atlas_xau/scripts/03_train_q_production.py` | Replaced with M15 U-shape training recipe |
| `products/atlas_xau/scripts/10_sim_today.py` | Rewritten for new architecture |
| `models/atlas_xau_validated.pkl` | New 40-feature bundle (backup: `.bak_pre_m15_ushape_2026-06-17`) |

EA changes: **none**. ATL- magic numbers, slot management, switch/BE/cooldown rules unchanged.

---

## Pre-2026-06-17 architecture (archived)

> **Prior version (strict_candle):** STRICT 2-bar reversal candle + Kalman confluence + Q ≥ 1.0
> **Prior bundle:** `atlas_xau_validated.pkl.bak_pre_m15_ushape_2026-06-17` (54 features)

## 🆕 Current deployed config (2026-06-17)

| Param | Value | Notes |
|---|---|---|
| `strong_body_atr` | 0.8 | strong-candle threshold |
| `kf_age_min` | **4** | rolled back 7 → 4 on 2026-06-17 (kage=7 was net-negative live) |
| `require_both_lines` | True | bar must close past both Kalman + TFK lines |
| `q_thr` | **1.0** | lowered 1.5 → 1.0 on 2026-06-17 (more trades + accept lower PF pivot) |
| `time_block_utc` | (18, 2) | block 18:00–02:00 UTC (NY-pm + Asia) |
| `trend_slope_block` | 0.0 | disabled |
| `sl_hard_atr` | 6.0 | |
| `trail_atr` | **1.0** | tightened 1.5 → 1.0 on 2026-06-16 (DD −42% on 8mo holdout) |
| `use_orderflow` | True | |
| `max_concurrent` | 4 | |

### 2026-06-17 — Path B tested, NOT deployed
Same Path B feature augmentation (19 extra features) that lifted Hermes XAU was
tested on Atlas XAU and showed **no meaningful discrimination**:

| Q≥  | Baseline PF | Path B PF |
|-----|-------------|-----------|
| 1.0 | 1.23        | 1.23      |
| 2.5 | 1.25        | 1.27      |
| 4.0 | 1.17        | 0.92      |

Atlas XAU stays on the baseline 54-feature bundle. Experiment kept at
[experiments/atlas_xau_pathb_experiment.py](../../experiments/atlas_xau_pathb_experiment.py).

### 2026-06-16 — trail 1.5 → 1.0
8-month backtest: PF 1.39→1.44, DD 116→67R (−42%), WR ~74→77%. Same trade count,
slight $ drop (−8%) for huge DD improvement. Bundle re-trained on trail=1.5 labels
(not 1.0) but the exit-only change is robust enough to ship without another retrain.

---



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

</details>
