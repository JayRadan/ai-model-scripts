# Oracle XAU — Flagship RL-Enhanced XAUUSD Model

> **Version:** v84 (RL Entry) | **Holdout PF:** 4.21 | **WR:** 70.2% | **Trades:** 1,207  
> **Bundle:** `oracle_xau_validated.pkl` (4.8 MB) | **Deployed:** 2026-05-06

The premium-tier product. Uses 5 RL Q-functions (one per regime) to select
entries, replacing the 28-rule hand-coded catalog. Two-stage confirmation:
per-regime confirm head → meta-labeling gate.

---

## Performance History

| Version | Entry Type | PF | WR | Trades | Key Change |
|---|---|---|---|---|---|
| v7.2 | Rule-based (28 rules) | 3.45 | 65.9% | 1,903 | Baseline Oracle |
| v83c | Rule-based + filters | 4.18 | 69.1% | 1,487 | 4h regime + range filter + kill-switch |
| **v84** | **RL Q-functions** | **4.21** | **70.2%** | **1,207** | **Q-learning replaces rules** |

### v84 Holdout by Regime (2024-12-12 → 2026-05-01)

| Regime | Trades | WR | PF | Total R |
|---|---|---|---|---|
| C0 Uptrend | 388 | 76.5% | **5.43** | +1,724 |
| C1 MeanRevert | 168 | 56.5% | 1.62 | +104 |
| C2 TrendRange | 161 | 52.8% | 1.35 | +57 |
| C3 Downtrend | 387 | 75.7% | **5.98** | +1,886 |
| C4 HighVol | 103 | 63.1% | 3.55 | +321 |

C0 + C3 carry 64% of trades and ~85% of total profit.

---

## Architecture

```
M5 bar ──► K=5 regime selector (4h-step) ──► cid ∈ {0..4}
                       │
                       ▼
              RL Q-function per regime
              XGBRegressor(300 trees, depth=4)
              predicts expected PnL in R multiples
                       │
                  Q > 0.3R ?
                       │ YES
                       ▼
              Per-regime confirm head
              XGBClassifier(200 trees, depth=3)
              P(win | entry) ≥ threshold ?
                       │ YES
                       ▼
              Meta gate
              XGBClassifier(300 trees, depth=4)
              P(win | features, cid, dir) ≥ 0.775 ?
                       │ YES
                       ▼
                    OPEN TRADE
```

---

## Features

**18 v72l features** for entry/confirm/meta:
`hurst_rs`, `ou_theta`, `entropy_rate`, `kramers_up`, `wavelet_er`,
`vwap_dist`, `hour_enc`, `dow_enc`, `quantum_flow`, `quantum_flow_h4`,
`quantum_momentum`, `quantum_vwap_conf`, `quantum_divergence`,
`quantum_div_strength`, `vpin`, `sig_quad_var`, `har_rv_ratio`, `hawkes_eta`

**11 exit features:** `unrealized_pnl_R`, `bars_held`, `pnl_velocity` + 8 context features.

**20 meta features:** all 18 v72l + `direction` + `cid`.

---

## Hyperparameters

| Component | n_estimators | max_depth | lr | Other |
|---|---|---|---|---|
| Q-model (XGBRegressor) | 300 | 4 | 0.05 | subsample=0.8 |
| Confirm (XGBClassifier) | 200 | 3 | 0.05 | subsample=0.8 |
| Exit (XGBClassifier) | 300 | 5 | 0.05 | eval_metric=logloss |
| Meta (XGBClassifier) | 300 | 4 | 0.05 | eval_metric=logloss |

| Parameter | Value |
|---|---|
| MIN_Q (entry gate) | 0.3R expected PnL |
| MAX_HOLD | 60 bars (5 hours) |
| MIN_HOLD | 2 bars |
| SL_HARD | -4.0R (ATR-based) |
| TP target | 2:1 reward-to-risk |
| Train/holdout cutoff | 2024-12-12 |
| Meta threshold (XAU) | 0.775 (auto-swept) |

---

## Training Pipeline

### Prerequisites
```bash
# Required data files (in data/):
swing_v5_xauusd.csv              # 590K M5 bars from Dukascopy (~407 MB)
setups_*_v72l.csv                # 5 files, one per regime cid
regime_fingerprints_4h.csv       # 4h-step regime labels
regime_selector_4h.json          # K=5 centroids + relabel rules
```

### Train RL Entry
```bash
cd /home/jay/Desktop/new-model-zigzag
source .venv/bin/activate
python3 products/oracle_xau/train_rl_entry.py
```

The script (`train_rl_entry.py`, originally `experiments/v84_rl_entry/01_q_learning.py`):

1. Loads swing data (590K bars, ~407 MB)
2. Merges 4h-step regime labels onto all setups
3. Computes PnL labels (2:1 RR, 40-bar max, -4R hard stop)
4. Trains 5 XGBRegressor Q-models — one per regime, target = `pnl_r`
5. Filters setups where Q > 0.3R → assigns `rule="RL"`
6. Trains 5 per-regime confirm XGBClassifiers on binary `label`
7. Generates exit training data — bar-level MTM snapshots with "exit now?" labels
8. Trains exit XGBClassifier
9. Simulates all trades through exit model → trade-level PnL
10. Trains meta XGBClassifier on `pnl_R > 0`
11. Sweeps meta threshold [0.40-0.80] for max PF
12. Saves bundle to `products/models/oracle_xau_validated.pkl`

---

## v83c Shared Improvements

All products inherit these from `experiments/v83_range_position_filter/`:

1. **4h-step regime** (window=288, step=48) — catches regime changes 6× faster
   than the original bar-by-bar classification
2. **Range-position filter** — blocks R1a/R1b shorts below 65% of 20-bar range;
   blocks R3e longs above 35%. Eliminates mid-range false signals.
3. **Consecutive-loss kill-switch** — 3 SL in same (cid, direction) → 12-hour
   cooldown on that regime+direction. Prevents death-spirals during transitions.
4. **Cohort kills** — C2_R0h disabled (3-bar reversal cohort had 47.9% WR on
   walk-forward; sister rules similarly weak).

See `_shared/v83c_changes.md` for the full experiment breakdown.

---

## Bundle Structure

```python
{
    'q_entry': {0: XGBRegressor, ..., 4: XGBRegressor},  # 5 Q-models
    'mdls': {(0,'RL'): XGBClassifier, ..., (4,'RL'): XGBClassifier},  # 5 confirms
    'thrs': {(0,'RL'): 0.50, ...},           # Per-regime confirm thresholds
    'exit_mdl': XGBClassifier,                # 11 features
    'meta_mdl': XGBClassifier,                # 20 features
    'meta_threshold': 0.775,                  # Auto-selected
    'min_q': 0.3,                             # Q-value entry gate
    'rl_rule_name': 'RL',
    'version': 'v84-rl',
}
```

---

## Deployment

### Server config (`configs/oracle_xau.py`)
```python
rl_entry_mode: bool = True
rl_min_q: float = 0.3
regime_selector_json: str = "regime_selector_4h.json"
kill_switch_losses: int = 3
kill_switch_cooldown_hours: int = 12
range_filter_lookback: int = 20
```

### Push to Render
```bash
cd /home/jay/Desktop/my-agents-and-website/commercial
cp ../new-model-zigzag/products/models/oracle_xau_validated.pkl \
   server/decision_engine/models/
git add server/decision_engine/models/oracle_xau_validated.pkl \
        server/decision_engine/configs/oracle_xau.py \
        server/decision_engine/decide.py
git commit -m "deploy oracle_xau v84 RL bundle"
git push origin main  # Render auto-deploys
```

### EA Endpoint
The MT5 EA sends M5 bars to `POST /decide/oracle_xau`. Response:
```json
{
  "action": "open",
  "direction": "buy",
  "rule": "RL",
  "cluster_id": 0,
  "cluster_name": "Uptrend",
  "p_conf": 0.723,
  "p_win": 0.812,
  "sl_atr_mult": 4.0,
  "tp_atr_mult": 2.0
}
```

Exit signals via `POST /decide/oracle_xau/exit` with position context
(entry_price, entry_atr, bars_held, direction). Server computes PnL
and records kill-switch state.
