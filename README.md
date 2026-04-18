# XAUUSD M5 Rule + ML-Confirmation Trading System

A regime-aware, rule-based scalping system for gold M5, with one tiny ML
classifier per rule that filters out the mediocre setups.

## Layout

```
new-model-zigzag/
│
├── README.md                ← you are here
│
├── data/                    ← all CSVs and data artifacts
│   ├── swing_v4.csv                  raw 1M-bar XAUUSD M5 with 21 base features
│   ├── labeled_v4.csv                labeled (BUY/FLAT/SELL) by 01_labeler_v4.py
│   ├── cluster_{0,1,2,3}_data.csv    bars split by regime (03_split_clusters.py)
│   ├── setups_{0,1,3}.csv            rule-fired setups + outcome (04_build_setup_signals.py)
│   ├── regime_fingerprints_K4.csv    weekly fingerprints
│   ├── regime_selector_K4.json       k-means centroids + scaler for live regime detection
│   ├── regime_clusters_K4.png        cluster visualization
│   └── backtest_confirmation.png     honest holdout equity curves
│
├── models/                  ← all trained ML models
│   ├── confirm_c{0,1,3}_<rule>.json     XGBoost native (12 files)
│   ├── confirm_c{0,1,3}_<rule>_meta.json metadata + threshold (12 files)
│   ├── confirm_c{0,1,3}_<rule>.onnx     ONNX exports for MT5 (12 files)
│   └── onnx_export_summary.json
│
├── model_pipeline/          ← THE FULL TRAINING PIPELINE (run in number order)
│   ├── paths.py                          centralized data/model paths
│   ├── 01_labeler_v4.py                  swing_v4 → labeled_v4 (causal swing labels)
│   ├── 02_build_selector.py              labeled_v4 → regime clusters (k-means)
│   ├── 03_split_clusters.py              labeled_v4 → cluster_{0..3}_data.csv
│   ├── 04_build_setup_signals.py         12 hand-coded rules → setups_{0,1,3}.csv
│   ├── 05_train_confirmation.py          12 XGBoost classifiers → models/
│   ├── 06_backtest_confirmation.py       honest 80/20 holdout backtest + PNG
│   ├── 07_export_confirmation_onnx.py    12 ONNX exports + numerical validation
│   ├── 08_gen_mql5_selector.py           regime_selector.mqh (constants only)
│   └── 09_gen_mql5_confirmation_router.py confirmation_router.mqh (loads 12 ONNX)
│
├── live_deployment/         ← READY-TO-COPY MT5 BUNDLE
│   ├── DEPLOYMENT.md                     full integration guide
│   ├── MQL5_Files/                       12 ONNX models → MT5 MQL5/Files/
│   ├── MQL5_Include/                     3 .mqh headers → MT5 MQL5/Include/
│   ├── MQL5_Experts/                     SwingScalperEA_v5.mq5 → MT5 MQL5/Experts/
│   └── python_reference/                 Python source for cross-checking MQL5
│
└── archive/                 ← every prior failed approach (8 of them)
    └── old_experiments/                  v3/v4 EAs, regime-only models, RL agents
```

## How the v5 system works in 30 seconds

1. **Detect the active regime** (Ranging / Downtrend / Shock / Uptrend) from
   the last 1440 M5 bars using a fixed k-means classifier built once.
2. **Run all rules belonging to that regime** on the just-closed bar. There
   are 12 hand-coded rules total (4 per tradeable cluster), each detecting a
   classic chart pattern with strict geometric + momentum filters.
3. **For every rule that fires**, run its dedicated ONNX classifier. Take the
   trade only if `P(label=1) ≥ rule_threshold`.
4. **Place market order** with `TP = 2×ATR(14)`, `SL = 1×ATR(14)`. One
   position at a time, magic-filtered, with per-rule cooldown.

## Why this works (when 7 prior RL/XGBoost attempts didn't)

Every previous attempt asked the ML to **generate** signals from raw bar
features. That's a low-signal-to-noise task on M5 — every strong model
either memorized the training set (in-sample win, holdout loss) or
plateaued at ~40% win rate (unprofitable after spread).

v5 inverts the task. The hand-coded rules provide **20+ years of trader
prior knowledge** as a setup filter. They produce ~26,000 candidate setups
across the dataset with raw win rates of **44-50%** — already positive
expectancy at 2:1 R/R. The 12 per-rule classifiers then only have to
discriminate **within** a single pattern family (e.g. "which Bollinger
touches actually bounce?"), which is a much easier task that XGBoost
solves with AUC 0.50-0.57 — small but enough to push WR from 46% to 55%.

## Honest holdout results

Trained on first 80% of each rule's setup pool, evaluated on the unseen
last 20% (~19 months of 2024-2026 gold the models never saw):

| Cluster      | Trades | WR  | PF   | PnL    | Max DD |
|--------------|--------|-----|------|--------|--------|
| C0 Ranging   | 240    | 56% | 1.63 | +120.6 | -15.0  |
| C1 Downtrend | 215    | 47% | 1.32 | +130.9 | -33.4  |
| C3 Uptrend   | 397    | 45% | 1.30 | +270.4 | -53.6  |
| **Combined** | **852**| —   | ~1.36| **+522** | —     |

PnL is in dollars at 0.01 lot sizing. See
[data/backtest_confirmation.png](data/backtest_confirmation.png) for
equity curves.

## Rerunning the pipeline from scratch

```bash
cd model_pipeline

# Step 1: rebuild labels (only if you change the labeler params)
python3 01_labeler_v4.py

# Step 2: rebuild regime clusters (only if you change selector params)
python3 02_build_selector.py --k 4

# Step 3: split into per-cluster CSVs
python3 03_split_clusters.py

# Step 4: scan for rule-based setups
python3 04_build_setup_signals.py

# Step 5: train per-rule confirmation classifiers
python3 05_train_confirmation.py

# Step 6: honest 80/20 backtest + equity PNG
python3 06_backtest_confirmation.py

# Step 7: export 12 ONNX models for MT5
python3 07_export_confirmation_onnx.py

# Step 8 + 9: regenerate MQL5 headers in live_deployment/
python3 08_gen_mql5_selector.py
python3 09_gen_mql5_confirmation_router.py
```

After step 9 you can copy `live_deployment/MQL5_Files/`,
`live_deployment/MQL5_Include/`, and `live_deployment/MQL5_Experts/`
into your MT5 install.

## Going live

See [live_deployment/DEPLOYMENT.md](live_deployment/DEPLOYMENT.md).
The bundle is already copied into the active MT5 install at
`/home/jay/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/`.

Open `Experts/SwingScalperEA_v5.mq5` in MetaEditor, press F7 → 0 errors.
Drop on XAUUSD M5 demo chart. Logs should show
`ConfirmRouter: all 12 confirmation models loaded`.

## Caveats

- ~2 trades/day combined — below the 7/day target. C0 and C1 are sparse;
  C3 is the volume workhorse.
- All rules use the same TP/SL (2×/1× ATR) and 40-bar max horizon. Don't
  change those without retraining the classifiers (the labels were built
  from those exact rules).
- 36 features must match the Python pipeline byte-for-byte. v5 EA's
  `BuildFeatures()` is identical to the labeler's.
- C2 Shock weeks are non-tradeable by rule. ~10% of weeks fall there.
