# EdgePredictor — v7.2-lite Trading Model Pipeline

Regime-aware, rule-based scalping with two-stage ML filtering and ML-driven
exits. Currently deployed on **XAUUSD** (Oracle, Apr 2026) and **BTCUSD**
(BTC, Apr 2026). Same architecture, separate training per instrument.

This README is the canonical onboarding doc for any agent/developer picking
up the project. It documents the end-to-end pipeline from raw MT5 CSV
export to live-deployed EA.

---

## 1. Architecture at a glance

```
 MT5 exporter ──► swing_v5_<inst>.csv  (M5 OHLC + 21 micro + H1/H4 context + label)
                                │
                                ▼
       ┌────────────────────────────────────────────────┐
       │ Stage 1: Regime selector (K=5 rolling windows) │  02_build_selector_k5*.py
       │ Stage 2: Tech features + cluster assignment    │  01_prepare_<inst>.py
       │ Stage 3: Rule scanner → per-cluster setups     │  04_build_setup_signals*.py
       │ Stage 4: Physics features (hurst/ou/entropy/.) │  04b_compute_physics_features*.py
       │ Stage 5: v7.2-lite extras (vpin/har/hawkes)    │  00_compute_v72l_features_step1*.py
       │ Stage 6: Per-(cluster,rule) XGB + exit + meta  │  01_validate_v72_lite*.py
       │ Stage 7: Train on full data, export ONNX       │  02_train_and_export_v72l*.py
       │ Stage 8: Generate MQL5 router + regime mqh     │  04_gen_router_v72l*.py + 08_gen_mql5_selector*.py
       │ Stage 9: Build EA (SwingScalperEA_*.mq5)       │  live_deployment/MQL5_Experts/
       │ Stage 10: Upload ONNX to server MongoDB        │  commercial/server/upload_k5_models.py
       │ Stage 11: Generate website backtest JSON + PNG │  experiments/.../08_gen_backtest_assets*.py
       └────────────────────────────────────────────────┘
```

**Three models per instrument:**
- **Confirm** (29 ONNX, one per cluster×rule): `P(rule win)` — 18 features.
- **Exit** (1 ONNX): `P(close now is optimal)` — 11 features (unrealized_R, bars_held, velocity + 8 physics).
- **Meta** (1 ONNX): `P(setup wins | confirm passes)` — 20 features (18 + direction + cid).

**Runtime gates per trade:**
```
confirm_proba >= rule_threshold  AND  meta_proba >= meta_threshold
```

**Exit logic:** check ML exit every bar after `MIN_HOLD=2`; hard SL at `4×ATR` safety stop; max hold `60 bars` (5h M5).

**Regime selector:** 7-feature fingerprint on trailing 288 M5 bars (~1 UTC day), scaler → PCA → nearest-centroid → `g_active_cluster`. Refreshed on first tick of each new UTC day (calendar-anchored so all users converge).

---

## 2. File layout

```
new-model-zigzag/
├── README.md                          ← you are here
│
├── data/                              ← all CSVs + JSON artifacts
│   ├── samples/                          **head -200 of every key CSV, gitted.**
│   │                                     Open these to see column schemas without
│   │                                     regenerating the (gitignored) full files.
│   ├── swing_v5_xauusd.csv               M5 OHLC + 21 micro + H1/H4 context + label (XAU)
│   ├── swing_v5_btc.csv                  same schema (BTC)
│   ├── regime_selector_K4.json           XAU K=5 centroids/scaler/PCA
│   ├── regime_selector_btc_K5.json       BTC K=5 centroids/scaler/PCA
│   ├── regime_fingerprints_*.csv         rolling-window fingerprints (per instrument)
│   ├── regime_clusters_*.png             cluster scatter visualization
│   ├── cluster_{0..4}_data_btc.csv       BTC bars split by cluster + tech features
│   ├── setups_{0..4}_btc.csv             rule-fired setups (BTC)
│   ├── setups_{0..4}_v6_btc.csv          + 14 physics features
│   ├── setups_{0..4}_v72l_btc.csv        + 4 v7 extras (vpin/sig/har/hawkes)
│   └── v72l_trades_holdout_*.csv         holdout trade dumps (used by website backtest JSON)
│
├── models/                            ← all trained ONNX artifacts
│   ├── confirm_v7_c{0..4}_*.onnx         Oracle (XAU) per-rule confirm heads (28)
│   ├── confirm_v7_btc_c{0..4}_*.onnx     BTC per-rule confirm heads (28)
│   ├── exit_v7.onnx  / exit_v7_btc.onnx
│   ├── meta_v7.onnx  / meta_v7_btc.onnx
│   ├── regime_v4_xau.onnx                XAU vol-regime lot sizer (not yet ported to BTC)
│   ├── v7_deploy.json  / v7_deploy_btc.json   feature orders + per-rule thresholds + meta threshold
│
├── model_pipeline/                    ← XAU-specific pipeline (shared helpers in paths.py)
│   ├── paths.py                         P.data() / P.model() wrappers
│   ├── 01_labeler_v4.py                 XAU: tech features + entry_class labels
│   ├── 02_build_selector_k5.py          XAU: K=5 regime selector
│   ├── 03_split_clusters_k5.py          XAU: split labeled bars per cluster
│   ├── 04_build_setup_signals.py        XAU: 26-rule scanner → setups_{cid}.csv
│   ├── 04b_compute_physics_features.py  XAU: merge 14 physics features onto setups
│   ├── 08_gen_mql5_selector.py          XAU: regime_selector.mqh generator
│   └── 09_gen_mql5_confirmation_router.py
│
├── experiments/
│   ├── v72_lite_deploy/               ← XAU Oracle (deployed)
│   │   ├── 00_compute_v72l_features_step1.py   VPIN + HAR + Hawkes + sig_quad_var
│   │   ├── 01_validate_v72_lite.py             honest holdout validation
│   │   ├── 02_train_and_export_v72l.py         train on full data → ONNX
│   │   ├── 04_gen_router_v72l.py               confirmation_router_v7.mqh generator
│   │   ├── 05_gen_backtest_json.py             website assets for XAU
│   │   ├── 07_regime_refresh_sensitivity.py    cadence diagnostic
│   │   └── meta_threshold_v72l.txt             0.6750
│   │
│   └── v72_lite_btc_deploy/           ← BTC (deployed)
│       ├── 01_prepare_btc.py                   01+03 merged for BTC
│       ├── 02_build_selector_k5_btc.py
│       ├── 04_build_setup_signals_btc.py
│       ├── 04b_compute_physics_features_btc.py
│       ├── 00_compute_v72l_features_step1_btc.py
│       ├── 01_validate_v72_lite_btc.py
│       ├── 02_train_and_export_v72l_btc.py     batched simulate() — fast
│       ├── 04_gen_router_v72l_btc.py
│       ├── 08_gen_mql5_selector_btc.py
│       ├── 08_gen_backtest_assets_btc.py       website JSON + PNG for BTC
│       └── meta_threshold_v72l_btc.txt         0.5250
│
└── live_deployment/                   ← what gets copied to MT5
    ├── MQL5_Experts/
    │   ├── SwingScalperEA_v6.mq5        Midas (XAU) — 14-feature base
    │   ├── SwingScalperEA_v7.mq5        Oracle (XAU) — 18 features + meta
    │   └── SwingScalperEA_BTC.mq5       BTC Oracle (magic 420805)
    ├── MQL5_Include/
    │   ├── regime_selector.mqh           XAU K=5 constants
    │   ├── regime_selector_btc.mqh       BTC K=5 constants
    │   ├── confirmation_router_v7.mqh    XAU per-rule loader
    │   ├── confirmation_router_v7_btc.mqh
    │   ├── setup_rules.mqh               26 rule detectors (shared)
    │   └── v7_features.mqh               VPIN/HAR/Hawkes ring-buffer (shared)
    └── MQL5_Files/                      ONNX staging dir (copied by MT5 on license validate)
```

---

## 3. Full build — step by step (BTC example)

### Step 1 — Export raw data from MT5

Run `exporter-20-math` script in MT5 on the target symbol's M5 chart. Output:
```
MQL5/Files/swing_<inst>_5min.csv   (UTF-16, ~800MB for 10 years)
```

Schema: `time,open,high,low,close,spread,f01..f20,h1_*,h4_*,label` (42 cols).

**Gotcha:** MT5 writes UTF-16. Transcode before reading in Python:
```bash
iconv -f UTF-16 -t UTF-8 "<MT5>/MQL5/Files/swing_btc_5min.csv" \
  > data/swing_v5_btc.csv
```

### Step 2 — Build the regime selector (K=5)

```bash
cd experiments/v72_lite_btc_deploy
python 02_build_selector_k5_btc.py
```
Produces: `data/regime_selector_btc_K5.json`, `regime_fingerprints_btc_K5.csv`, `regime_clusters_btc_K5.png`.

Fingerprint is 7 features on non-overlapping 288-bar windows (~1 day):
`weekly_return_pct, volatility_pct, trend_consistency, trend_strength, volatility, range_vs_atr, return_autocorr`.

Cluster roles are **auto-detected** by the script (no manual tagging):
`C0=Uptrend (highest ret), C3=Downtrend (lowest), C4=HighVol (highest vol among middle), C1=MeanRevert (lowest autocorr), C2=TrendRange (remaining)`.

### Step 3 — Tech features + cluster assignment

```bash
python 01_prepare_btc.py
```
Reads `swing_v5_btc.csv`, computes 15 tech features (RSI, Stoch, BB%B, momentum, etc.), assigns cluster per bar using the selector JSON, writes `cluster_{0..4}_data_btc.csv`.

**XAU equivalent:** `model_pipeline/01_labeler_v4.py` then `03_split_clusters_k5.py`.

### Step 4 — Build setups with the 26-rule scanner

```bash
python 04_build_setup_signals_btc.py
```
Scans all 26 Midas-style rules (Bollinger touches, swing rejections, false breakouts, etc.). Writes `setups_{cid}_btc.csv` per cluster, each row has features + direction + forward outcome label (TP=2×ATR / SL=1×ATR / 40-bar max).

### Step 5 — Physics features (14 columns)

```bash
python 04b_compute_physics_features_btc.py
```
Computes on the full swing CSV then merges onto setups → `setups_{cid}_v6_btc.csv`:
- `hurst_rs, ou_theta, entropy_rate, kramers_up, wavelet_er, vwap_dist`
- `hour_enc, dow_enc` (**note**: `/5` divisor for `dow_enc`, XAU-style weekday encoding — applies to BTC too for consistency)
- `quantum_flow, quantum_flow_h4, quantum_momentum, quantum_vwap_conf, quantum_divergence, quantum_div_strength`

### Step 6 — v7.2-lite extras

```bash
python 00_compute_v72l_features_step1_btc.py
```
Adds 4 features with **step=1 (every bar)** so MQL5 live inference matches:
- `vpin` — range-based VPIN (BVC, Gaussian CDF, 50 buckets)
- `sig_quad_var` — Σ(Δy)² over last 60 log-returns
- `har_rv_ratio` — RV(last 288) / RV(last 8640)
- `hawkes_eta` — event rate last 60 / last 600, event = |r|>2σ_500

Writes `setups_{cid}_v72l_btc.csv` — the final input to training.

### Step 7 — Honest holdout validation

```bash
python 01_validate_v72_lite_btc.py
```
- Chrono split at `2024-12-12` (train / holdout)
- Per-(cluster, rule) XGB confirm heads (disables rules with base PF<1)
- ML exit head on confirmed train trades (peak-from-here target)
- Meta head trained on real ML-exit simulator outcomes
- Meta-threshold auto-selected by sweep on meta-validation slice
- **Batched simulate()** — builds (N_trades × MAX_HOLD) feature matrix, one `predict_proba` call (critical for 2-min runs, vs hours unbatched)

Output: `data/v72l_trades_holdout_btc.csv`, per-cluster WR/PF, auto-selected meta threshold → `meta_threshold_v72l_btc.txt`.

### Step 8 — Production train + ONNX export

```bash
python 02_train_and_export_v72l_btc.py
```
Retrains on **full history** (no holdout reserve — architecture already validated at step 7). Exports:
- 28-29 `confirm_v7_btc_c{cid}_{rule}.onnx`
- `exit_v7_btc.onnx`, `meta_v7_btc.onnx`
- `models/v7_deploy_btc.json` — feature orders, per-rule thresholds, meta threshold

ONNX parity is verified at export time (XGB predict_proba vs ONNX Runtime, target ≤ 1e-6 max abs diff).

### Step 9 — Generate MQL5 regime + router includes

```bash
python 08_gen_mql5_selector_btc.py   # → live_deployment/MQL5_Include/regime_selector_btc.mqh
python 04_gen_router_v72l_btc.py     # → MT5 Include/confirmation_router_v7_btc.mqh
```

The router mqh embeds ONNX filenames + thresholds + a `RULE_DISABLED` flag for rules with base PF<1 (marked with threshold 1.01 so they never pass).

### Step 10 — Build/adapt the EA

Copy `SwingScalperEA_v7.mq5` → `SwingScalperEA_BTC.mq5` and rewire:
- `#include <regime_selector_btc.mqh>` + `<confirmation_router_v7_btc.mqh>`
- `InpMagic = 420805` (distinct per instrument: v6=420305, v7=420705, BTC=420805)
- `META_THRESHOLD = 0.525` (from step 7)
- `exit_v7_btc.onnx`, `meta_v7_btc.onnx` filenames
- All `ConfirmV7_*` → `ConfirmV7_BTC_*`, `RULE_*_V7` → `RULE_*_V7_BTC`
- `InpMaxSpread` — **BTC is ~40× wider than XAU**; default `5000` points is sensible
- Log prefixes: `"v7:"` → `"BTC:"` (verify UpdateDashboard, watermark, init print)

Features built by the EA in `BuildFeatures()` must **exactly match** `feature_order_v72l` in `v7_deploy_btc.json`. Meta appends `direction, cid` (20 total). Exit uses a different 11-feature order starting with `unrealized_pnl_R, bars_held, pnl_velocity`.

### Step 11 — Deploy to MT5

```bash
cp live_deployment/MQL5_Experts/SwingScalperEA_BTC.mq5       "<MT5>/MQL5/Experts/"
cp live_deployment/MQL5_Include/regime_selector_btc.mqh      "<MT5>/MQL5/Include/"
# confirmation_router_v7_btc.mqh already written there by 04_gen_router
cp models/confirm_v7_btc_*.onnx models/exit_v7_btc.onnx models/meta_v7_btc.onnx "<MT5>/MQL5/Files/"
```
Compile the `.mq5` in MetaEditor (F7). Attach to BTCUSD M5 chart.

### Step 12 — Upload ONNX to the license server

```bash
cd commercial/server
source venv/bin/activate
python upload_k5_models.py   # uploads to MongoDB GridFS
```

**Also update `server/server.py`** to whitelist the new prefix:
```python
is_confirm = (
    ... or name.startswith("confirm_v7_btc_c") or ...
)
```
Otherwise the EA's license-based model download will return HTTP 400 and abort.

Exit/meta models auto-match existing prefix rules (`exit_v*`, `meta_v*`).

### Step 13 — Generate website backtest assets

```bash
python experiments/v72_lite_btc_deploy/08_gen_backtest_assets_btc.py
```
Post-processes `v72l_trades_holdout_btc.csv` + swing data (for ATR→USD conversion) into:
- `public/btc_backtest.png` — equity curve
- `public/backtest_data.json` — adds `"btc"` key matching the Midas/Oracle schema

**Schema gotcha:** `regimes[]` uses `{name, color, trades, wr, pf, pnl}` and `top_rules[]` uses `{name, pf, trades}`. Don't write `win_rate/profit_factor/rule` — the page will crash.

### Step 14 — Add product to website

Edit `commercial/website/lib/products.ts`:
- Add `"btc"` to `type ProductId`
- Add a `btc` product entry (color, price, description, files, install steps)
- Append to `ALL_PRODUCT_IDS` + `STANDALONE_PRODUCT_IDS`

Edit `app/download/[token]/page.tsx`:
- Add `"btc"` to the local `PackageId` union
- Add install steps in `PACKAGE_META`

Edit `app/api/admin/settings/route.ts` + `app/admin/page.tsx`:
- Add `btc_price` field + input

Copy `.ex5` to website:
```bash
cp "<MT5>/MQL5/Experts/SwingScalperEA_BTC.ex5" commercial/website/public/files/EdgePredictor_BTC.ex5
```
(Needs `git add -f` — `.ex5` is in `.gitignore`.)

### Step 15 — Type-check + commit

```bash
cd commercial/website && npx tsc --noEmit
cd ../ && git add <changes> && git commit -m "..." && git push origin main
```

The Render deploy picks up the server whitelist change; Next.js deploy picks up the website.

---

## 4. Deployed numbers (Apr 2026)

| Product | Asset | Period | Trades | WR | PF | Total R | MaxDD R |
|---|---|---|---|---|---|---|---|
| Midas  | XAUUSD | v6 per-rule 80/20 holdout        | 2,636 | 57.4% | 2.24 | +5,093R | -80R  |
| Oracle | XAUUSD | v7.2-lite, Dec 2024+             | 1,367 | 65.3% | 3.48 | +1,947R | -34R  |
| BTC    | BTCUSD | v7.2-lite, Dec 2024+             | 2,439 | 54.9% | 1.84 | +21,674R | -522R |

Midas numbers are from the April 2026 revalidation (`experiments/v6_xau_deploy/01_validate_v6.py`),
which is the script that produces the pickle the live server serves; the
older v6 backtest summary file (`data/backtest_v6_summary.json`, PF 1.34)
used a fixed-TP labelling that no longer matches the production ML-exit
path and is retained only for historical reference.

All at `0.01 lot`, clean holdout. Gold PnL absolute is smaller because XAU at 0.01 = $0.01/dollar-move vs BTC at 0.01 = $0.01/dollar-move but BTC ATR is ~10× larger in dollars.

### v7.9 cohort kill — May 2026 (deployed)

Forensic on each product's holdout per `(cluster_id, rule)` cohort
identified specific (cid, rule) pairs with statistically-significant
low WR that the global meta gate could not filter — meta is a single
classifier that smooths cohort-level signal away. Walk-forward H1→H2
of holdout (out-of-sample for the rule) confirmed the kills generalize:

| Product | Killed | ΔWR | ΔPF | ΔR |
|---|---|---:|---:|---:|
| Oracle XAU | C2_R0e_nr4_break | +0.6pp | +0.20 | ~0 |
| Midas XAU  | C1_R0c_doubletouch, C2_R0d_squeeze, C2_R0e_nr4_break, C2_R0h_3bar_reversal | +5.2pp | +0.85 | -3.3% |
| Oracle BTC | C2_R0h_3bar_reversal | +9.0pp | +1.81 | ~0 |
| Janus XAU  | (no kill — single-rule cascade has no clean cohort candidate) | — | — | — |

Implementation: `disabled_cohorts: tuple = ((cid, rule), ...)` field on
each ProductConfig (`commercial/server/decision_engine/configs/*.py`);
`decide._armed_pairs_for()` strips disabled pairs as the final step so it
applies equally to native and overlay-borrowed rules. Hard rule, no retrain.
Re-run forensics after every product retrain — cohorts may shift.

Forensic + walk-forward scripts:
`experiments/v79_meta_threshold_sweep/{02_loser_forensics,03_walk_forward}.py`

### What was tested and didn't ship (May 2026)

For agents picking up this project: the following have been rigorously
disproven on the validated holdout. Don't propose them again without a
genuinely new input source.

- **v75 K-means / HMM regime overlays** on XAU-only fingerprints —
  no PF separation across clusters
- **v75 daily-veto** from XAU daily features — every threshold
  net-removes profit
- **v76 kNN analog** 24h-shape matching — corr(signal, pnl) ≈ 0
- **v77 daily cross-instrument** (DXY/SPX/TNX/VIX) — top-quintile
  on Oracle p=0.001 (real signal in tail) but ΔR negative
- **v77b intraday cross-instrument** (1h) — in-sample +0.28 →
  holdout **-0.04** (sign flipped, macro relationships drift in
  ~6 months)
- **v78 magnitude prediction → variance-based sizing** — first
  positive cross-product transfer (spearman +0.10/+0.08) but
  modest (+3% R Midas, p=0.024). Not yet shipped pending a clean
  per-product variance head and `lot_mult` wire-in.

The takeaway from 6 experiments: **trained confirm heads + meta gate
already saturate directional signal at M5**. Residual is mostly noise.
Real wins came from (a) cohort kills (v79) and (b) different prediction
target like magnitude (v78) — both compound with existing models, not
replace them.

---

## 5. Critical conventions to preserve

### MQL5 ↔ Python parity
Static audit checklist (do this before every new-instrument ship):
1. `feature_order_v72l` (18) in deploy JSON matches EA `BuildFeatures()` index-by-index.
2. `meta_feature_order` (20) = 18 base + `direction, cid`.
3. `exit_feature_order` (11) = `unrealized_pnl_R, bars_held, pnl_velocity` + 8 physics in documented order.
4. `RULE_THRESHOLD_V7_*[]` in router mqh == thresholds from `confirm_models` in deploy JSON.
5. Regime `REGIME_SCALER_MEAN/STD/PCA/CENTROIDS` in the `.mqh` come from the auto-generator (`08_gen_mql5_selector*.py`) — do **not** hand-edit.
6. `dow_enc` divisor: `/5` (matches XAU labeler, applied to BTC too by 04b).

### Calendar-anchored regime refresh
EAs refresh the active cluster on the **first tick of each new UTC day**, not every N bars from EA start. This ensures all users converge on the same cluster regardless of when they attached the EA. Implemented via `g_last_regime_day = TimeCurrent()/86400` in `OnTick()`.

### Position recovery across restarts
`OnInit()` scans `PositionsTotal()` for a position matching the EA's magic+symbol and re-adopts it (`g_entry_price/dir/atr/bars_held`). Required because `g_entry_atr == 0` would silently disable the ML exit after recompile.

### Batched simulate()
Inside training scripts, `simulate()` **must batch** `predict_proba` — build a `(N_trades × MAX_HOLD, n_feats)` matrix and call once. Unbatched (per-bar) takes hours on the full training set; batched takes seconds.

### ONNX filename prefixes (server whitelist)
`commercial/server/server.py` explicitly whitelists prefixes: `confirm_c`, `confirm_v6_c`, `confirm_v7_c`, `confirm_v7_btc_c`, `gj_confirm_c`, `eu_confirm_c`. When adding a new instrument, add its prefix here or the license server returns HTTP 400 on download.

### Website JSON schema
`backtest_data.json` entries (`midas`, `oracle`, `btc`) share the schema in `app/backtest/page.tsx` → `interface ModelData`. `regimes[]` = `{name, color, trades, wr, pf, pnl}`; `top_rules[]` = `{name, pf, trades}`. Rebuilding with different names crashes the tab.

---

## 6. Quick commands reference

```bash
# Fresh build for a new instrument (e.g. ETH):
iconv -f UTF-16 -t UTF-8 "<MT5>/.../swing_eth_5min.csv" > data/swing_v5_eth.csv

# Duplicate the BTC pipeline folder, s/btc/eth/, then run:
cd experiments/v72_lite_eth_deploy
python 02_build_selector_k5_eth.py
python 01_prepare_eth.py
python 04_build_setup_signals_eth.py
python 04b_compute_physics_features_eth.py
python 00_compute_v72l_features_step1_eth.py
python 01_validate_v72_lite_eth.py          # ← read auto-selected threshold
python 02_train_and_export_v72l_eth.py
python 08_gen_mql5_selector_eth.py
python 04_gen_router_v72l_eth.py
python 08_gen_backtest_assets_eth.py

# Deploy
cp live_deployment/MQL5_Experts/SwingScalperEA_ETH.mq5      "<MT5>/MQL5/Experts/"
cp live_deployment/MQL5_Include/regime_selector_eth.mqh     "<MT5>/MQL5/Include/"
cp models/confirm_v7_eth_*.onnx models/exit_v7_eth.onnx models/meta_v7_eth.onnx "<MT5>/MQL5/Files/"

# Server
cd commercial/server && python upload_k5_models.py
# Edit server.py: add 'confirm_v7_eth_c' to is_confirm whitelist

# Website
# Edit lib/products.ts + app/download/[token]/page.tsx + admin settings
cp "<MT5>/.../SwingScalperEA_ETH.ex5" commercial/website/public/files/EdgePredictor_ETH.ex5
git add -f <files> && git commit && git push
```

---

## 7. Things agents commonly get wrong

1. **Skipping step 1 (UTF-16 transcode)** → pandas reads garbage headers.
2. **Hand-writing the regime mqh** → use `08_gen_mql5_selector*.py`; hand-written centroids drift.
3. **Forgetting to batch `simulate()`** → training hangs for hours.
4. **Wrong schema in `backtest_data.json`** → BTC tab crashed exactly this way; use `wr/pf/name` not `win_rate/profit_factor/rule`.
5. **Copying XAU EA and forgetting `InpMaxSpread`** → BTC spreads are 40× wider; default 80 blocks every trade.
6. **Forgetting `Midas` / `v7` strings in log prints and dashboards** → user-facing inconsistency.
7. **Not updating `server.py` whitelist** → HTTP 400, EA aborts at init despite models being uploaded.
8. **Modifying `InpVerbose` default expecting it to update already-attached EAs** → MT5 preserves old input values across recompile; user must re-attach the EA fresh or change in the Inputs panel.

---

## 8. Where to find answers

| I need to know... | Look at |
|---|---|
| How an EA decides direction | `SwingScalperEA_v7.mq5` → `ScanAndConfirm()` + `setup_rules.mqh` |
| Which rules belong to which cluster | `setup_rules.mqh` → `RULE_CLUSTER[]` array |
| ML exit formula | `experiments/v72_lite_*/01_validate_v72_lite*.py` → `train_exit()` (peak-from-here target) |
| Current per-rule thresholds | `models/v7_deploy_<inst>.json` → `confirm_models[<rule>].threshold` |
| Holdout trade log | `data/v72l_trades_holdout_<inst>.csv` |
| What changed in the last model rev | `git log models/v7_deploy_<inst>.json` + the commit message |
| MT5 Experts log location | `~/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Logs/` |
| License server endpoint | `commercial/server/server.py` → `/license` + `/model` routes |
| How to ship a retrained model to prod | `commercial/server/decision_engine/DEPLOY.md` |
| Sample rows of every training CSV | `data/samples/README.md` |

---

## 9. Shipping a retrained model

**Note:** since April 2026, live inference runs on the FastAPI "decision
engine" (the `/decide/{product}` endpoint on Render), *not* from ONNX
files on the customer's MT5. The EA is a thin client that POSTs bars.

That means retraining is no longer "regen ONNX → upload to license
server → user downloads" — it is "regen pickled XGB bundle → push to
server repo → Render auto-deploy → all customers live on the new
models in ~60 s". Full steps:
`commercial/server/decision_engine/DEPLOY.md`.
