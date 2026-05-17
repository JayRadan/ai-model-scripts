# EdgePredictor — v9.3 Trading Model Pipeline

Regime-aware, rule-based scalping with two-stage ML filtering and ML-driven
exits. **4 products live as of May 2026**: Oracle XAU, Midas XAU, Janus XAU,
Oracle BTC.

This README is the canonical onboarding doc. It documents the end-to-end
pipeline from raw MT5 CSV export to live-deployed EA.

> **NEW:** see [`RETRAIN.md`](RETRAIN.md) for step-by-step retrain commands
> per product. See `## Current state (v9.3, May 2026)` below for what's
> actually deployed today.

---

## Current state (v9.3, May 2026) — read this first

### What's deployed

| Product | Symbol | Architecture | Holdout PF (live) |
|---|---|---|---|
| Oracle XAU | XAUUSD | v7.2-lite + flow regime + meta gate | ~3.70 |
| Midas XAU | XAUUSD | v6 (14 feat) + flow regime, no meta | ~2.92 |
| Janus XAU | XAUUSD | v7.4 pivot-score + flow regime | ~2.78 |
| Oracle BTC | BTCUSD | v7.2-lite + flow regime + meta gate (BTC selector) | ~2.98 |

### Architecture changes in v9.3 (May 3 2026 deploy)

1. **K=5 regime classifier now uses 8 features** (was 7). 8th feature is
   `flow_4h_mean` — the mean of the Quantum Volume-Flow Quantizer (Pine
   indicator, ported in `experiments/v89_quantum_flow_tiebreaker/`)
   computed on 4h-resampled bars over the 288-bar window.
2. **All training data switched to Dukascopy** (single source of truth for
   real `tick_volume`, since broker CSVs were exporting `spread` masquerading
   as volume — broke the volume-dependent flow indicator).
3. **Path A live**: server fetches its own Dukascopy bars per /decide call,
   ignoring whatever bars the customer EA sends. Means every customer on
   every broker gets the **identical** decision for the same closed bar.
   Source: `commercial/server/decision_engine/dukascopy_source.py`.
   Activate via `USE_DUKASCOPY_BARS=1` env var on Render.

### Critical pinned params (do not change without testing)

```
MIN_DATE = "2016-01-01"          # XAU & Midas selectors. NOT 2020.
MIN_DATE = "2018-01-01"          # BTC selector
random_state = 42                # KMeans + every XGBClassifier
WINDOW = 288, STEP = 288         # regime fingerprint window
K = 5                             # number of regime clusters
Cutoff = "2024-12-12"             # train/test holdout split
```

### File map for v9.3

| Role | File |
|---|---|
| XAU regime selector builder | `experiments/v93_flow_in_regime/01_selector_with_flow_feature.py` |
| BTC regime selector builder | `experiments/v93_flow_in_regime/02_selector_btc_with_flow.py` |
| Quantum Flow port (Pine→Python) | `experiments/v89_quantum_flow_tiebreaker/01_port_and_test.py` |
| XAU regime selector (live) | `data/regime_selector_K4.json` |
| BTC regime selector (live) | `data/regime_selector_btc_K5.json` |
| Server-side regime classifier | `commercial/server/decision_engine/regime.py` (Quantum Flow inlined) |
| Server-side Dukascopy fetcher | `commercial/server/decision_engine/dukascopy_source.py` |
| Step-by-step retrain commands | `RETRAIN.md` |

### Live-ops tooling shipped 2026-05-17

Beyond model retrains, three server-side hardening features run alongside
every product. Shared infrastructure, no per-product training.

| Feature | Commit | Where to use it |
|---|---|---|
| **Stack-gate** — blocks new same-direction slot while a prior slot is underwater. Caps regime-lag losses to 1 slot. | `1374daa` | Automatic, no UI. Filter funnel by `reason LIKE '%stack_gate%'`. |
| **Admin regime override** — pin/clear live regime per product. Survives Render restarts via `regime_overrides.json`. | `f22903c` + fixes `730f112`, `fe27c3e` | `/admin/regime` panel. |
| **Funnel model vs effective regime** — per-product card shows classifier verdict + active override. Amber border when pinned. | `d84b505` | `/admin` → Decision Funnel. |

See product READMEs (`products/oracle_*/README.md` → "Live Operations
Tooling" section) for per-product wiring details.

### Known caveats from May 2026 deploy events

1. **The original v9.3 winner script was edited 4 minutes after winning and
   not committed to git.** Result: we measured PF 4.17 last night but
   today's reproducible rebuild gives PF ~3.40. The `MIN_DATE=2020 → 2016`
   revert recovered most of the gap; remaining 0.77 PF is currently
   unrecoverable. Current scripts in repo produce the verifiable PF 3.40
   number.
2. **The deployed Oracle XAU pkl has `meta_threshold=0.55` with 29 confirm
   heads**, while a fresh retrain produces `meta_threshold=0.675` with 28
   heads. The deployed version performs *better* on holdout (PF 3.70 vs
   3.40) — we don't fully know why. **Don't overwrite the deployed pkl
   without comparing first.**
3. **Junk-file bug** — `04b_compute_physics_features.py` and
   `00_compute_v72l_features_step1.py` previously globbed their own
   outputs and created recursive `setups_0_v6_v6.csv` style junk. Both
   patched on May 4 to filter to canonical files. If junk reappears,
   delete with: `rm data/setups_*_v6_v6*.csv data/setups_*_v72l_v6*.csv`.

---

---

## 1. Architecture at a glance

**Server-driven (Apr 2026+):** customer EA is a thin client. Every closed
M5 bar it POSTs the latest bars to a FastAPI service hosted on Render;
the server runs the pickled `XGBClassifier` objects from validation and
returns a `Decision`. The EA only places/manages broker orders.

```
                     ┌─────────────────────────────────────┐
 ML training         │    new-model-zigzag (this repo)     │
 + backtests         │  validate → /tmp/*_pipeline.pkl     │
                     └────────────────┬────────────────────┘
                                      │ pickle_validated_models.py
                                      ▼
                     ┌─────────────────────────────────────┐
                     │  commercial/server/decision_engine  │
                     │  • configs/<product>.py             │
                     │  • models/<product>_validated.pkl   │
                     │  • api.py (FastAPI /decide/{prod})  │
                     │  • decide.py (cascade)              │
                     │  • funnel_log.py (sqlite trace)     │
                     └────────────────┬────────────────────┘
                                      │ git push → Render auto-deploy
                                      ▼
                     ┌─────────────────────────────────────┐
   Customer MT5 ◄─── │   https://edge-predictor.onrender   │
   EdgePredictor_    │       /decide/{product}             │
   Connector_v2.ex5  └─────────────────────────────────────┘
```

**Decide cascade per closed bar (per product):**
```
regime classify (K=5 fingerprint)
   → look up armed (confirm_cid, rule) pairs for live cluster
   → drop disabled_cohorts (v7.9 hard kills)
   → fire detector — does any rule trigger on the last bar?
   → Stage-1: per-rule XGB confirm head → drop if p_conf < threshold
   → Stage-2 (Oracle/Janus only): meta head → drop if p_win < meta_thr
   → optional Janus pivot-score upstream cascade
   → Decision { open / hold / exit, direction, sl_atr_mult, trace }
```

**Three (or four) models per product, all `XGBClassifier` pickles:**
- **Confirm** (1 per cluster×rule, ~28-30 heads): `P(rule wins)` — 18 features (v7.2-lite) or 14 (Midas v6).
- **Exit** (1 head): `P(close now is optimal)` — 11 features (`unrealized_R`, `bars_held`, `pnl_velocity` + 8 physics). Threshold 0.55.
- **Meta** (Oracle XAU/BTC + Janus only — Midas v6 has no meta): `P(setup wins | confirm passes)` — 20 features (18 + direction + cid).
- **Pivot-score + direction** (Janus only): per-bar pivot detector that runs upstream of the cluster cascade.

**Exit logic:** ML exit checked every bar after `MIN_HOLD=2`; hard SL at `4×ATR` safety stop; max hold `60 bars` (5h M5).

**Regime selector:** 7-feature fingerprint on trailing 288 M5 bars (~1 UTC day), scaler → PCA → nearest-centroid → cluster id. Refreshed on first tick of each new UTC day (calendar-anchored so all users converge).

**Why pickles, not ONNX.** ONNX round-tripping introduces ~1e-6 numerical drift. Over thousands of bars and a confidence cutoff that drift can flip decisions. Pickles keep the live model bit-identical to the one that produced the holdout result. The `.ex5` file customers download (`EdgePredictor_Connector_v2.ex5`) never changes when we retrain — only the server-side pickle does.

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
└── live_deployment/                   ← LEGACY (pre-Apr 2026 ONNX path).
    │                                      Kept for reference only — current
    │                                      deployment is server-driven; the
    │                                      customer .ex5 lives in the
    │                                      commercial repo (see Section 3).
    ├── MQL5_Experts/SwingScalperEA_*.mq5
    └── MQL5_Include/{regime_selector,confirmation_router_v7,setup_rules,v7_features}.mqh
```

The **current** customer-facing artifact is
`commercial/server/MQL5_Experts/EdgePredictor_Connector_v2.mq5` (compiled
to `commercial/website/public/files/EdgePredictor_Connector.ex5`). Same
.ex5 for every product — product is selected via the EA's `EP_PRODUCT`
input enum, server-side configs decide everything else.

---

## 3. Full build — step by step (current server-driven architecture)

This is the end-to-end path for **(a) retraining an existing product** or
**(b) shipping a brand-new product**. Steps 1-7 are the same in both cases
(prepare data → train → backtest). After step 7 the path is much shorter
than the old ONNX-per-EA model: pickle, vendor, push, Render auto-deploys.

> **Retraining an existing product?** You only need steps 7-10. The data
> and feature artifacts are already on disk; you're just refreshing the
> trained weights.
>
> **Adding a new product?** All 12 steps. Mirror the directory pattern
> of an existing product (e.g. `experiments/v72_lite_btc_deploy/` is the
> reference for "asset N of an Oracle-shape engine").
>
> **Detailed cheat-sheet:** `commercial/server/decision_engine/DEPLOY.md`
> has copy-pasteable commands and a troubleshooting table.

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

### Step 8 — Pickle the validated XGB bundle

```bash
cd ~/Desktop/my-agents-and-website/commercial/server
python3 decision_engine/scripts/pickle_validated_models.py oracle_btc
# → ~/Desktop/new-model-zigzag/models/oracle_btc_validated.pkl
```

The pickler reads the trained XGB objects that step 7 stashed (Oracle: at
`/tmp/oracle_deployed_pipeline_cache.pkl`; Midas: at the validate
script's `_raw.pkl` output) and freezes them into a single bundle
containing `mdls`, `thrs`, `exit_mdl`, `meta_mdl`, `meta_threshold`,
`v72l_feats`, `meta_feats`, `exit_feats`, `trained_on`, `git_rev`.

For a Midas-shape engine (no meta gate) the script picks the
`_build_midas_payload` branch automatically; the resulting pickle has
`meta_mdl = None` and `decide.py` already branches on this.

### Step 9 — Vendor the pickle into the server repo

```bash
cp ~/Desktop/new-model-zigzag/models/oracle_btc_validated.pkl \
   ~/Desktop/my-agents-and-website/commercial/server/decision_engine/models/
```

The server loads pickles from this vendored directory at boot. **The
`new-model-zigzag/models/` copy is the source of truth** (regenerated by
the pickler); the vendored copy is what gets deployed. Always copy from
left to right, never edit the vendored copy directly.

### Step 10 — Sanity-load + commit + push (Render auto-deploys)

```bash
cd ~/Desktop/my-agents-and-website/commercial/server

# Sanity check — load the bundle, verify head count + meta threshold
python3 -c "
from decision_engine import loader
from decision_engine.configs import ORACLE_BTC
b = loader.load_bundle(ORACLE_BTC)
print('heads=', len(b.payload['mdls']),
      'meta_thr=', b.payload['meta_threshold'])
"

cd ~/Desktop/my-agents-and-website/commercial
git add server/decision_engine/models/oracle_btc_validated.pkl \
        server/decision_engine/configs/*.py
git commit -m "Oracle BTC retrain: holdout PF X.XX / WR YY.Y%"
git push
```

That's it for an existing-product retrain. **Render auto-deploys in
~60 s; next closed-bar POST hits the new models.** Verify with:

```bash
curl https://edge-predictor.onrender.com/decide/_health | jq
```

The customer's `EdgePredictor_Connector_v2.ex5` does **not** change.

---

The remaining steps (11-12) only apply when **adding a new product**, not
when retraining.

### Step 11 — Wire the product server-side (NEW PRODUCTS ONLY)

**11a. Add a config** in `commercial/server/decision_engine/configs/<slug>.py`
(copy `oracle_btc.py` as the closest template):

```python
@dataclass(frozen=True)
class OracleEthConfigT:
    name: str = "oracle_eth"
    symbol_base: str = "ETHUSD"
    validated_pkl: str = "oracle_eth_validated.pkl"
    regime_selector_json: str = "regime_selector_eth_K5.json"
    v72l_feats: tuple = V72L_FEATS
    meta_feats: tuple = META_FEATS
    exit_feats: tuple = EXIT_FEATS
    exit_threshold: float = 0.55
    min_hold_bars:  int = 2
    max_hold_bars:  int = 60
    sl_hard_atr:    float = 4.0
    c4_directional_overlay: bool = False    # ship after holdout validates
    disabled_cohorts: tuple = ()             # populate after v79 forensics
```

**11b. Register** in `configs/__init__.py` `REGISTRY` dict.

**11c. Extend** `scripts/pickle_validated_models.py` with the new product's
`PRODUCTS` entry (`pipeline_dir`, `validate_mod`, `threshold_file`,
`cache_path`, `out_name`) — pattern matches whichever existing product
shares its engine shape.

**11d. Wire the EA enum** in
`commercial/server/MQL5_Experts/EdgePredictor_Connector_v2.mq5`:
- add slug to `ENUM_EP_PRODUCT`,
- add to `ProductSlug()`, `DefaultMaxSpreadFor()`, `RulesArmedForCluster()`,
- pick a magic-number block (Oracle XAU 421000+, BTC 421050+, Midas 421100+,
  Janus 421150+ — pick the next free range).
- Recompile in MetaEditor (F7), drop the new `.ex5` into
  `commercial/website/public/files/EdgePredictor_Connector.ex5` (single
  `.ex5` for all products).

### Step 12 — Wire the product website-side (NEW PRODUCTS ONLY)

**12a. Generate website backtest assets:**
```bash
python experiments/v72_lite_eth_deploy/08_gen_backtest_assets_eth.py
```
Produces `public/eth_backtest.png` + appends an `"eth"` key to
`public/backtest_data.json`. Schema: `regimes[] = {name, color, trades, wr,
pf, pnl}`, `top_rules[] = {name, pf, trades}` (don't use `win_rate /
profit_factor / rule` — the page will crash).

**12b. Edit `commercial/website/lib/products.ts`:**
- add `"eth"` to `type ProductId`,
- add an `eth` product entry (color, price, description, files, install steps),
- append to `ALL_PRODUCT_IDS` + `STANDALONE_PRODUCT_IDS`.

**12c. Edit `app/download/[token]/page.tsx`** — add `"eth"` to the local
`PackageId` union + install steps in `PACKAGE_META`.

**12d. Edit `app/api/admin/settings/route.ts` + `app/admin/page.tsx`** —
add `eth_price` field + input.

**12e. Type-check + push:**
```bash
cd commercial/website && npx tsc --noEmit
cd ../ && git add <changes> && git commit -m "Ship Oracle ETH" && git push
```

Render redeploys server + website. New product is live.

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

### Server-side feature parity
Static audit checklist (server-driven architecture — features are
computed server-side from the bars the Connector EA POSTs):
1. `cfg.v72l_feats` tuple in `configs/<product>.py` matches the column
   order the validator's confirm models were trained on (cross-checked
   at load time against the pickle's `v72l_feats` field).
2. `cfg.meta_feats` (Oracle/Janus only) = `v72l_feats + ("direction", "cid")`.
3. `cfg.exit_feats` (11) = `("unrealized_pnl_R", "bars_held", "pnl_velocity",
   <8 physics features in the documented order>)`.
4. Per-rule `thrs[(cid, rule)]` thresholds and the regime selector JSON
   come straight from the pickled bundle — never hand-edit.
5. `dow_enc` divisor: `/5` (matches XAU labeler, applied to BTC too by 04b).
6. `disabled_cohorts` (v7.9) — verify after every retrain by re-running
   `experiments/v79_meta_threshold_sweep/02_loser_forensics.py` against
   the fresh holdout. Bad cohorts may shift.

### Calendar-anchored regime refresh
The decision engine refreshes the active cluster on the **first decision
of each new UTC day**, not every N bars. This ensures all users converge
on the same cluster regardless of when they attached the EA. Implemented
in `decide.py` using the bar timestamp the EA posts — server doesn't
care about wallclock.

### Position recovery across restarts
The Connector v2 EA's `OnInit()` calls `ReconcileSlotsWithBroker()` —
scans `PositionsTotal()` for positions matching the EA's magic block and
re-adopts each into a slot. Required because slot state lives in the
EA's `EpSlot[]` array, which a recompile clears.

### Batched simulate()
Inside training scripts, `simulate()` **must batch** `predict_proba` — build a `(N_trades × MAX_HOLD, n_feats)` matrix and call once. Unbatched (per-bar) takes hours on the full training set; batched takes seconds.

### Pickle, never ONNX (live path)
The decision engine loads `XGBClassifier` pickles directly. ONNX
round-tripping introduces ~1e-6 numerical drift that, over thousands of
bars and a confidence cutoff, can flip decisions. Live numbers must
match validation numbers bit-for-bit. **Don't reintroduce ONNX as a
shortcut** — it broke parity twice in 2025.

### Website JSON schema
`backtest_data.json` entries (`midas`, `oracle`, `btc`) share the schema in `app/backtest/page.tsx` → `interface ModelData`. `regimes[]` = `{name, color, trades, wr, pf, pnl}`; `top_rules[]` = `{name, pf, trades}`. Rebuilding with different names crashes the tab.

---

## 6. Quick commands reference

```bash
# ── Retrain an existing product (e.g. Oracle BTC) ────────────────────
cd ~/Desktop/new-model-zigzag
python experiments/v72_lite_btc_deploy/01_validate_v72_lite_btc.py

cd ~/Desktop/my-agents-and-website/commercial/server
python3 decision_engine/scripts/pickle_validated_models.py oracle_btc
cp ~/Desktop/new-model-zigzag/models/oracle_btc_validated.pkl \
   decision_engine/models/

# Sanity load
python3 -c "from decision_engine import loader; from decision_engine.configs import ORACLE_BTC; \
            b = loader.load_bundle(ORACLE_BTC); \
            print('heads=', len(b.payload['mdls']), 'meta=', b.payload['meta_threshold'])"

cd ~/Desktop/my-agents-and-website/commercial
git add server/decision_engine/ && git commit -m "Oracle BTC retrain" && git push
# Render auto-deploys in ~60s.

# ── Re-run cohort forensics after retrain (recommended) ──────────────
cd ~/Desktop/new-model-zigzag
python experiments/v79_meta_threshold_sweep/02_loser_forensics.py
python experiments/v79_meta_threshold_sweep/03_walk_forward.py
# If bad cohorts changed, update disabled_cohorts in the product config.

# ── Fresh build for a new instrument ─────────────────────────────────
iconv -f UTF-16 -t UTF-8 "<MT5>/.../swing_eth_5min.csv" > data/swing_v5_eth.csv

# Duplicate the BTC pipeline folder, s/btc/eth/, then run:
cd experiments/v72_lite_eth_deploy
python 02_build_selector_k5_eth.py
python 01_prepare_eth.py
python 04_build_setup_signals_eth.py
python 04b_compute_physics_features_eth.py
python 00_compute_v72l_features_step1_eth.py
python 01_validate_v72_lite_eth.py
# Then steps 8-12 of Section 3 (config + pickle + EA enum + website).

# ── Live ops ─────────────────────────────────────────────────────────
curl https://edge-predictor.onrender.com/decide/_health | jq
# Force-flush server caches (e.g. stuck regime):
curl -X POST https://edge-predictor.onrender.com/decide/_flush-cache \
     -H "x-admin-secret: $ADMIN_SECRET"
```

---

## 7. Things agents commonly get wrong

1. **Skipping step 1 (UTF-16 transcode)** → pandas reads garbage headers.
2. **Forgetting to batch `simulate()`** → training hangs for hours.
3. **Editing the vendored pickle in `commercial/server/decision_engine/models/` directly** → it gets overwritten by the next pickler run. Always regenerate via `pickle_validated_models.py`.
4. **Reintroducing ONNX in the live path** → 1e-6 numerical drift flips decisions over thousands of bars. The decision engine loads `XGBClassifier` pickles for a reason.
5. **Wrong schema in `backtest_data.json`** → BTC tab crashed exactly this way; use `wr/pf/name` not `win_rate/profit_factor/rule`.
6. **Forgetting to register a new product in `configs/__init__.py`'s `REGISTRY`** → server returns HTTP 404 `unknown product`.
7. **Adding a new product without copying the new `.ex5` to `commercial/website/public/files/`** → customers download the old binary (single `.ex5` for all products; the `EP_PRODUCT` enum in the EA selects which slug it talks to).
8. **Modifying EA inputs and expecting already-attached charts to pick them up** → MT5 preserves old input values across recompile. User must re-attach fresh or change in the Inputs panel.

---

## 8. Where to find answers

| I need to know... | Look at |
|---|---|
| How the server decides per bar | `commercial/server/decision_engine/decide.py` → `decide_entry()` cascade |
| How rules are detected from bars | `commercial/server/decision_engine/rules.py` (`RULE_FNS` dispatcher) + `experiments/.../setup_signals.py` |
| Which rules belong to which cluster | The pickled `mdls` dict (`(cid, rule)` keys) — load via `loader.load_bundle()` |
| ML exit formula | `experiments/v72_lite_*/01_validate_v72_lite*.py` → `train_exit()` (peak-from-here target) |
| Current per-rule thresholds | `payload['thrs']` in the pickle (`models/<product>_validated.pkl`) |
| Per-product runtime config | `commercial/server/decision_engine/configs/<product>.py` |
| Holdout trade log | `data/v72l_trades_holdout_<inst>.csv` |
| What changed in the last model rev | `git log -- server/decision_engine/models/<product>_validated.pkl` |
| Server live state / health | `curl https://edge-predictor.onrender.com/decide/_health \| jq` |
| Decision funnel (per-bar trace) | Admin panel → Decision Funnel, or `/decide/_log-stats?product=…` |
| Connector EA source | `commercial/server/MQL5_Experts/EdgePredictor_Connector_v2.mq5` |
| Customer-facing .ex5 | `commercial/website/public/files/EdgePredictor_Connector.ex5` |
| Detailed retrain/new-product cheat-sheet | `commercial/server/decision_engine/DEPLOY.md` |
| Sample rows of every training CSV | `data/samples/README.md` |

---

## 9. Shipping a retrained model — TL;DR

Full steps are in [Section 3 Steps 7-10](#step-7--honest-holdout-validation)
above; here's the 30-second version:

```bash
# 1. Validate (in zigzag repo) — produces /tmp/<product>_pipeline_cache.pkl
python experiments/v72_lite_<inst>_deploy/01_validate_v72_lite_<inst>.py

# 2. Pickle (in commercial repo)
cd ~/Desktop/my-agents-and-website/commercial/server
python3 decision_engine/scripts/pickle_validated_models.py <product>

# 3. Vendor + commit + push (Render auto-deploys in ~60s)
cp ~/Desktop/new-model-zigzag/models/<product>_validated.pkl decision_engine/models/
cd .. && git add server/decision_engine/ && git commit -m "<product> retrain" && git push
```

No customer action required. Their `.ex5` doesn't change. Old models
roll back via `git revert` of the pickle commit + push.

**Detailed cheat-sheet** with troubleshooting:
`commercial/server/decision_engine/DEPLOY.md`.
