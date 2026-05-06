# Edge Predictor — Production Model Catalog

> **Last updated:** 2026-05-06  
> **GitHub:** `JayRadan/edge_predictor` (auto-deploys to Render on push to `main`)

Four production trading models for XAUUSD and BTCUSD (M5 timeframe).
All trained on Dukascopy tick data, validated on a strict 2024-12-12+
out-of-sample holdout.

---

## Quick Reference

| Product | Asset | Entry | PF | WR | Trades | Bundle |
|---|---|---|---|---|---|---|
| **Oracle XAU** | XAUUSD | RL v84 | **4.21** | 70.2% | 1,207 | 4.8 MB |
| **Oracle BTC** | BTCUSD | RL v84 | **3.82** | 67.9% | 1,135 | 4.8 MB |
| **Midas XAU** | XAUUSD | Rules v83c | **5.25** | 73.4% | 1,693 | 31 MB |
| **Janus XAU** | XAUUSD | Pivot-score | **1.85** | 58.3% | 892 | 5.3 MB |

---

## Shared Infrastructure

### Regime Detection (v83c — 4h-step)
All products use K=5 K-means on 8 fingerprint features, computed every
4 hours (window=288 M5 bars, step=48). Catch regime changes 6× faster
than the original bar-by-bar classification.

**XAU relabel:** `highvol_only` — C4 overridden if |24h return| > 0.3%  
**BTC relabel:** `full_directional` — any cluster overridden if |24h return| > 1.0%

### v83c Improvements (all products)
1. **Range-position entry filter** — skips entries at poor range positions
2. **Consecutive-loss kill-switch** — 3 SL → 12h cooldown per (cid, dir)
3. **4h-step regime** — faster regime detection
4. **Cohort kills** — disabled weak rules

### v84 RL Entry (Oracle only)
Replaces 28-rule catalog with 5 XGBRegressor Q-functions. PF gains:
- Oracle XAU: +0.03 PF, +101 more trades
- Oracle BTC: +0.79 PF, +102 more trades
- Midas: attempted but failed (14 features insufficient)

---

## Folder Layout

```
products/
├── README.md                     ← THIS FILE
├── models/                       ← All .pkl bundles (gitignored, on disk)
│   ├── oracle_xau_validated.pkl
│   ├── oracle_btc_validated.pkl
│   ├── midas_xau_validated.pkl
│   └── janus_xau_validated.pkl
├── _shared/                      ← Cross-product infrastructure
│   ├── regime_selector_xau.json
│   ├── regime_selector_btc.json
│   ├── v83c_changes.md
│   └── scripts/
│       ├── build_regime_selector.py    ← K=5 K-means + relabel rules
│       └── visualize_regime.py         ← 4h-step regime chart
├── oracle_xau/                   ← Flagship RL model
│   ├── README.md
│   ├── train_rl_entry.py              ← Quick: just RL
│   ├── deploy_bundle.py               ← Save bundle to server
│   └── scripts/                       ← Full pipeline (5 steps)
│       ├── 01_validate_v72l.py         ← Original v72l training
│       ├── 02_train_export.py          ← Export models
│       ├── 03_train_rl_entry.py        ← RL Q-learning entry
│       ├── 04_full_rl_exit.py          ← RL exit (experimental)
│       └── 05_deploy_bundle.py         ← Deploy to server
├── oracle_btc/                   ← BTC RL model
│   ├── README.md
│   ├── train_rl_entry.py
│   └── scripts/
│       ├── 01_validate_v72l.py
│       ├── 02_train_export.py
│       ├── 02b_build_selector.py       ← BTC K=5 regime selector
│       ├── 03_v83c_pipeline.py         ← v83c range filter + kill-switch
│       └── 04_train_rl_entry.py
├── midas_xau/                    ← Entry-level rule-based
│   ├── README.md
│   └── scripts/
│       └── 01_validate_v6.py           ← v6 (14-feature) training
└── janus_xau/                    ← Pivot-score experimental
    ├── README.md
    ├── models/
    └── scripts/
        ├── 00_compute_features.py
        ├── 01_label_bars.py
        ├── 02_train_pivot.py
        ├── 03_build_setups.py
        ├── 04_validate.py
        └── 05_pickle_janus.py
```

---

## Deployment Architecture

```
MT5 EA (MQL5)                    Render (FastAPI)
─────────────                    ─────────────────
M5 bar close ──POST /decide/{product}──► decide_entry()
                                              │
                              ┌───────────────┴──────────────┐
                              ▼                              ▼
                         RL Entry Path               Rule-based Path
                    (if rl_entry_mode=True)     (hand-coded rule catalog)
                              │                              │
                         Q-function ──► Q>0.3?         Rule detection
                              │                              │
                         confirm head                 per-rule confirm
                              │                              │
                         meta gate                    meta gate (Oracle)
                              │                              │
                              └──────────┬───────────────────┘
                                         ▼
                                    Decision JSON
```

---

## Retraining from Scratch

### 1. Build Regime Selector
```bash
python3 experiments/v83_range_position_filter/01_build_selector_4h.py
# Writes regime_selector_4h.json + regime_fingerprints_4h.csv
```

### 2. Train Each Product
```bash
# Oracle XAU (RL)
python3 products/oracle_xau/train_rl_entry.py

# Oracle BTC (RL)
python3 products/oracle_btc/train_rl_entry.py

# Midas retraining — see experiments/v83_range_position_filter/
```

### 3. Deploy
```bash
cd /home/jay/Desktop/my-agents-and-website/commercial
cp ../new-model-zigzag/products/models/*.pkl server/decision_engine/models/
git add server/decision_engine/models/ server/decision_engine/configs/
git commit -m "deploy updated model bundles"
git push origin main
```

---

## Key Experiment Logs

| Experiment | Location | Result |
|---|---|---|
| v83c range filter + kill-switch | `experiments/v83_range_position_filter/` | +0.5-2.7 PF across products |
| v84 RL Entry (XAU) | `experiments/v84_rl_entry/01_q_learning.py` | PF 4.21 ✅ |
| v84 RL Entry (BTC) | `experiments/v84_rl_entry/07_btc_rl.py` | PF 3.82 ✅ |
| v84 RL Entry (Midas) | `experiments/v84_rl_entry/06_midas_rl.py` | PF ~2.0 ❌ |
| v84 RL Exit | `experiments/v84_rl_entry/02_full_rl.py` | PF 3.85 (worse than ML exit) |
| v84 Improved Exit | `experiments/v84_rl_entry/03_improved_exit.py` | PF 3.65 (worse) |
