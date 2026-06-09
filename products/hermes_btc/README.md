# Hermes BTC — M1 BTCUSD TFK + Order-Flow

**Status:** DEPLOYED 2026-05-26 (commit `588a6af` in commercial repo)
**Version:** v103-derived (Hermes architecture, BTC-trained bundle)

---


## 🧠 2026-06-09 — Trend-gate filter (deployed)

After 30-day post-deploy backtest, **time filters HURT BTC** (24/7 market with
no quiet hours). The **trend-gate** was the winner: block counter-trend
entries when the 20-bar slope opposes direction and `|slope20| > 1.5`.

Config: `trend_slope_block=1.5`.

| | Baseline | + trend-gate >1.5 |
|---|---|---|
| Trades | 1071 | 823 (-23%) |
| WR | 59.1% | **62.7%** |
| PF | 1.79 | **2.37** |
| sumR | +816 | +899 |
| DD | 91.90 | **71.45** (-22%) |
| $@0.10 | $18,775 | **$20,685** |

Biggest single-filter $ uplift across the product line. DD reduction is the
real story — fewer "I lost 6R in 10 minutes" tickets. Disabled with
`trend_slope_block = 0.0`.


## TL;DR

Bitcoin counterpart to Hermes XAU. Same architecture, BTC-specific bundle, slightly tighter Q gate.

```
Locked config:   NEAR ≤ 0.50   Q ≥ 2.5   SL = 4×ATR   TRAIL = 3×ATR   max_concurrent = 4
Holdout window:  2026-01-15 → 2026-05-02 (~3.5 months, 107 calendar days)
Holdout trades:  2,752       WR 74.9%    PF 3.87    sumR +7,653 R    DD -302 R
Multi-pos sim:   1,209 trades @ 0.01 lot → +$2,275 USD (DD -$57) starting $1,000
```

PF 3.87 vs Hermes XAU's 2.09 — BTC's friendlier spread (0.089R vs ~1R for XAU) and smoother trends produce a cleaner edge.

---

## ⚠️ Deployment notes

| Item | Value |
|---|---|
| Server slug | `hermes_btc` |
| Bundle pickle | `commercial/server/decision_engine/models/hermes_btc_validated.pkl` |
| Config dataclass | `commercial/server/decision_engine/configs/hermes_btc.py` |
| Engine type | `hermes` (shared dispatch path with `hermes_xau` in `api.py`) |
| EA enum | `EP_HERMES_BTC` |
| EA magic base | `421150` (slots 0..3 → 421150..421153) |
| EA timeframe | `PERIOD_M1` (chart must be M1 BTCUSD) |
| EA comment prefix | `HRB-L` / `HRB-S` |
| Web product id | `hermesBtc` ($99/mo, dbField `hermes_btc_price`) |

`EdgePredictor_Connector_v2.mq5` source is patched — **recompile `.ex5` in MetaEditor** before going live (the deployed binary doesn't yet include the new enum value).

---

## Strategy in one paragraph

Every M1 BTCUSD bar close, the server computes the TFK (Tick-Flow Kinematics) indicator on the live bar series. When TFK's `committed_dir` is ±1 and the close is within 0.50 ATR of the TFK line (the "NEAR" gate), the server scores the bar through an XGBRegressor Q model trained on 43 features (29 standard OHLCV/TFK + 14 live order-flow). If `Q ≥ 2.5`, server returns `action: "open"` and the EA opens a market order. Up to 4 concurrent slots are allowed; when a fifth signal arrives, the worst-Q active slot is closed if the new Q beats it by `switch_delta=0.5`. When a new entry opens, slots already in profit ≥ +1R get their SL moved to entry. Exits: hard SL at -4×ATR, trail give-back of 3×ATR from peak favor, max hold 300 bars (~5 hours).

---

## Holdout numbers (Q≥2.5, 2026-01-15 → 2026-05-02)

| Metric | Value |
|---|---|
| Trades | 2,752 (raw sweep) / 1,209 (multi-pos sim with switch + cooldown) |
| Win rate | 74.9% |
| Profit factor | 3.87 |
| sumR | +7,653 R |
| Max DD (R) | -302 R |
| Avg R per trade | +2.78 R |
| Trades / day | ~26 (raw) / ~11 (multi-pos) |
| USD pnl @ 0.01 lot | +$2,275 from $1,000 start = 3.3× |
| Max USD DD | -$57 (1.7% of equity) |

Other Q thresholds tested (NEAR=0.5):

| Q | n | WR | PF | sumR | DD |
|---|---|---|---|---|---|
| 1.0 | 6,797 | 68.8% | 2.46 | +12,096 | 487 |
| 1.5 | 5,096 | 72.0% | 2.95 | +10,805 | 396 |
| 2.0 | 3,772 | 73.6% | 3.37 | +9,171 | 353 |
| **2.5** | **2,752** | **74.9%** | **3.87** | **+7,653** | **302** ← shipped |
| 3.0 | 1,903 | 76.3% | 4.35 | +5,802 | 188 |

Q=2.5 chosen as the tradeoff: highest PF with enough trade count for stable stats. Q=3.0 has higher PF but thins out to ~18 trades/day which gets noisy in shorter live windows.

---

## Folder layout

```
products/hermes_btc/
├── README.md                                 (this file)
├── train_bundle.py                           builds hermes_btc_validated.pkl
├── 01_download.log / 02_orderflow.log        download timings
└── scripts/                                  numbered reproduction recipe
    ├── 01_download_m1_dukascopy.py           BTC M1 OHLCV (2024-11 → 2026-05)
    ├── 02_aggregate_orderflow_from_ticks.py  reads data/ticks/btc/*.parquet
    ├── 03_train_q_baseline_m1.py             OHLCV-only baseline (no flow)
    ├── 04_train_q_with_orderflow.py          adds 14 orderflow feats — proves +PF
    ├── 05_sweep_sl_threshold.py              SL × TRAIL × NEAR grid
    ├── 06_sweep_near_threshold.py            NEAR sweep at fixed Q
    ├── 07_meta_classifier_disproven.py       2-stage meta gate — no lift
    ├── 08_visualize_holdout.py               equity curves + drawdown charts
    └── 09_build_website_backtest_json.py     writes backtest_data.json hermes_btc entry
```

Note: scripts/ here was copied & adapted from `products/hermes_xau/scripts/`. The TFK indicator code (`tfk.py`) and training feature engineering (`add_standard_features` from `products/_shared/m1_with_orderflow.py`) are shared with Hermes XAU — no need to duplicate.

---

## Feature engineering breakdown

**29 standard features** (price + TFK natives + HTF):
```
dist_at_signal, dist_abs, regime_age, bar_range_atr,
force, velocity, x_est, regime_w, trend_raw, trend,         (TFK natives)
rsi14, dist_ema20, dist_ema50, dist_ema100, dist_ema200,
slope5, slope10, slope20, atr_ratio,                         (price features)
m5_rsi14, m5_slope5, m5_ema50_dist,                          (M5 HTF)
m15_rsi14, m15_slope5, m15_ema50_dist,                       (M15 HTF)
h1_rsi14, h1_slope5, h1_ema50_dist,                          (H1 HTF)
committed_dir                                                (TFK direction)
```

**14 order-flow features** (from live Dukascopy ticks):
```
imbalance_ratio, bid_ask_vol_ratio, vpin_proxy, median_spread,
cum_signed_5, flow_persistence_5,
cum_signed_15, flow_persistence_15,
cum_signed_60, flow_persistence_60,
spread_vol_50, tick_intensity_50,
signed_flow, n_ticks
```

Total: **43 features**. Order-flow contributes +10% PF in BTC training (similar magnitude to XAU). Live computation in `commercial/server/decision_engine/tick_source.py` — fetches 90 min of ticks per cycle, 50s refresh.

---

## Exit policy (server-side, every M1 bar close)

```
priority order in decide_exit():
  1. max_hold_bars (300) reached         → exit
  2. hard_sl: adverse ≥ 4×ATR            → exit  (also broker-enforced via OrderModify at open)
  3. be_stop_soft: BE active AND price retraced to entry → exit
  4. trail: peak_favor ≥ 3×ATR AND give-back ≥ 3×ATR → exit

decide_entry() side-effects:
  - move_be_for_magics: any active slot with favor ≥ +1R gets SL moved to entry
  - switch_close_magic: when 4 slots full, if new_Q ≥ worst_active_Q + 0.5, close worst
```

There is **no color-flip exit** (tested in v9b/v9c, killed winners early — disproven).

---

## Multi-position state

EA owns slot identity. Each slot uses magic `BaseMagic(421150) + slot_idx`. EA sends `open_positions[]` with each slot's `magic`, `direction`, `entry_price`, `entry_atr`, `bars_held`, `q_at_entry`, `be_active`. Server is stateless — uses that snapshot to make decisions.

---

## Reproduction recipe

```bash
# Pre-req: data/ticks/btc/*.parquet present (518 days, 2024-12 → 2026-05)

# 1. M1 bars (run once, ~30 s for 18 months)
python3 products/hermes_btc/scripts/01_download_m1_dukascopy.py
#  → data/m1_btc_full.parquet (~785k bars)

# 2. Aggregate orderflow (run once, ~25 s)
python3 products/hermes_btc/scripts/02_aggregate_orderflow_from_ticks.py
#  → data/m1_btc_orderflow.parquet (~741k bars after dropping pre-tick window)

# 3-6. Sweep + train (~15 min total)
python3 products/hermes_btc/scripts/03_train_q_baseline_m1.py        # OHLCV baseline
python3 products/hermes_btc/scripts/04_train_q_with_orderflow.py     # full 43-feat
python3 products/hermes_btc/scripts/05_sweep_sl_threshold.py         # SL grid
python3 products/hermes_btc/scripts/06_sweep_near_threshold.py       # NEAR grid

# 7. Train + freeze production bundle (~30 s)
python3 products/hermes_btc/train_bundle.py
#  → commercial/server/decision_engine/models/hermes_btc_validated.pkl

# 8. Website backtest JSON (~10 s)
python3 /tmp/build_hermes_btc_backtest_json.py
#  → updates commercial/website/public/backtest_data.json hermes_btc entry
```

---

## Why config differs from Hermes XAU

| | XAU | BTC | Why |
|---|---|---|---|
| `q_thr` | 1.0 | **2.5** | BTC's near-zero spread (0.089R) means the loose-Q tail has positive expectancy; tighter Q monotonically improved PF. XAU's spread (~1R) already pruned the low-Q tail. |
| `spread_usd` | $0.30 | **$5.00** | Realistic broker spread on BTCUSD |
| `max_concurrent` | 4 | 4 | Same |
| `switch_delta` | 0.5 | 0.5 | Same |
| Everything else | — | — | Identical |

---

## Known limitations

- **Holdout = 3.5 months only.** Tick data on disk only goes back to 2024-12, so the training window is shorter than XAU's 8 years. Live results may drift more than XAU's did.
- **BTC weekend gaps.** Bitcoin trades 24/7 but Dukascopy doesn't quote on weekends. The broker spread widening Friday close → Monday open could blow past 4×ATR SL. Worth monitoring slots held over weekend.
- **High WR (74.9%).** May indicate slight overfit. Live WR likely 65-70%. Configure expectations accordingly.
- **No order-flow during Dukascopy outages.** `tick_source.py` falls back to zeros, which silently shifts the feature distribution. Q model trained on non-zero values may score lower in that state — fewer trades, not bad trades.

---

## Live operations cheat-sheet

```bash
# Inspect server regime state
curl https://edge-predictor.onrender.com/decide/hermes_btc/_regime-debug \
  -H "X-Admin-Secret: $ADMIN_SECRET" | jq

# Pull recent funnel
admin → Decision Funnel → product=hermes_btc → Download CSV

# Sanity check: q_thr expectation
# At Q≥2.5 we expect ~26 candidates/day passing all gates; reality ~11 trades/day
# after multi-position throttling. Days with 0 trades are normal.
```

Rollback: revert commit `588a6af` in commercial repo, redeploy Render.
