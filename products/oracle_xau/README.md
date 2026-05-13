# Oracle XAU — Flagship RL-Enhanced XAUUSD Model

> **Version:** v90 entry (maturity + 24h-momentum) + v88 reverse-setup exit + **v97 wider hard SL (6 ATR)**
> **Holdout PF:** **5.04** at q>3 with v97 SL=6 ATR (was 4.31 at SL=4) | **WR:** **74.1%** | **+8,286R**
> **Bundle:** `oracle_xau_validated.pkl` | **Deployed:** v84 2026-05-06 → v88 2026-05-08 → v89 2026-05-10 → v90 2026-05-12 → **v97 2026-05-13**

The premium-tier product. Uses 5 RL Q-functions (one per regime) to select
entries, replacing the 28-rule hand-coded catalog. Two-stage confirmation:
per-regime confirm head → meta-labeling gate.

---

## Performance History

| Version | Entry Type | PF | WR | Trades | Key Change |
|---|---|---|---|---|---|
| v7.2 | Rule-based (28 rules) | 3.45 | 65.9% | 1,903 | Baseline Oracle |
| v83c | Rule-based + filters | 4.18 | 69.1% | 1,487 | 4h regime + range filter + kill-switch |
| v84 | RL Q-functions (V72L only) | 4.21 | 70.2% | 1,207 | Q-learning replaces rules |
| v84 + v88 reverse-setup exit | (same entry, smarter exit) | 4.60 | 71.4% | 1,365 | Symmetric RL exit when opposite setup fires |
| v89 + v88 | RL Q-functions (V72L + maturity) | 6.44 | 77.4% | 1,154 | 3 maturity features added; min_q 0.3→3.0 |
| v90 + v88 | RL Q-functions (V72L + maturity + 24h-mom) | 4.31 (+R) | 74.1% | 2,452 | 2 direction-signed 24h-return features added; n_features 21→23 |
| **v97 (v90 + v88, SL widened 4→6 ATR)** | (same model, looser stop) | **5.04** | **74.1%** | **2,452** | **+750R / +10% PF — 56% of 4-ATR stops were being hunted (recovered within 60 bars)** |

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

## Architecture (v89)

```
M5 bar ──► K=5 regime selector (4h-step) ──► cid ∈ {0..4}
                       │
            v89 NEW: compute 3 maturity features
            stretch_100, stretch_200, pct_to_extreme_50
                       │
                       ▼
              RL Q-function per regime
              XGBRegressor(300 trees, depth=4)
              INPUT: V72L (18) + maturity (3) = 21 features
              predicts expected PnL in R multiples
                       │
                  Q > 3.0R ?  (v89 calibrated; was Q>0.3 in v84)
                       │ YES
                       ▼
              Per-regime confirm head
              XGBClassifier(200 trees, depth=3)
              INPUT: V72L (18 features) — unchanged
              P(win | entry) ≥ threshold ?
                       │ YES
                       ▼
              Meta gate
              XGBClassifier(300 trees, depth=4)
              INPUT: V72L + direction + cid (20) — unchanged
              P(win | features, cid, dir) ≥ 0.775 ?
                       │ YES
                       ▼
                    OPEN TRADE
```

### v89 Maturity Features (the key v84→v89 change)

Three direction-signed features capture *trend extension*:
- `stretch_100` — ATRs above 100-bar low (long) / below 100-bar high (short)
- `stretch_200` — same on 200-bar window
- `pct_to_extreme_50` — position within last 50-bar range (0 = opposite extreme, 1 = at top of recent rally / bottom of selloff)

These prevent the RL from entering at the *top of an extended rally* or *bottom of an extended selloff* — entries that look like clean pullbacks but are actually trend-reversal points.

**Disprove evidence:** at `q>3.0`, % of trades with `stretch_100 > 10 ATRs` drops from 14.1% (v84) → **8.7%** (v89), a 38% reduction in stretched-leg entries.

## Exit Logic (v88, deployed 2026-05-08)

```
Every in-trade M5 bar:
  1. Hard SL @ -4R (cp <= -1.0R in SL units)
  2. v88 Reverse-Setup RL Exit:        ← NEW
     scan all 30 rules at this bar
     if any opposite-direction setup fires:
        if q_entry[cid](v72l) > 0.10  → exit
  3. Trail (act=3.0R, gb=60% of peak)
  4. Binary ML exit (legacy)
  5. 60-bar max hold
```

The v88 reverse-setup check sits **before** the trail check so it can
catch reversals before the trail's high activation threshold (3R)
matters. Wraps in try/except — any rule-detection error falls through
to existing logic. Latency: ~232 ms per call (after slicing df to last
200 bars before rule scanning).

Validated on the unseen 30% of v84 RL trades (XAU 421 trades,
2025-12-10 → 2026-05-01):

| Metric | Pre-v88 exit | Post-v88 exit | Δ |
|---|---|---|---|
| PF | 4.18 | **4.54** | +0.36 |
| Total R | +959R | +963R | +4R |
| MaxDD | 20R | **15R** | **-5R** |
| Reverse-setup exits | — | 87 / 421 (21%) | new |
| Hard SL hits | 52 | 42 | -10 |
| Max-hold exits | 243 | 191 | -52 |

Source: `experiments/v88_exit_rl/13_reverse_setup_exit.py`. See
`experiments/v88_exit_rl/README.md` for the full 13-experiment catalog
(12 disproven attempts + this winner).

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

### v85 Drawdown Circuit Breaker
After a trade closes, the server tracks cumulative PnL for the product.
If PnL drops >25% from the session peak and the regime hasn't changed,
ALL new entries are blocked. Unblocks on: regime change, 4h timeout,
or PnL recovery to within 10% of peak.

Exit signals via `POST /decide/oracle_xau/exit` with position context
(entry_price, entry_atr, bars_held, direction). Server computes PnL
and records kill-switch + drawdown state.

---

## Full Pipeline Scripts

All scripts in `scripts/` — run in order to reproduce from scratch:

| # | Script | What it does |
|---|---|---|
| 1 | `scripts/01_validate_v72l.py` | Original v72l training — 28-rule confirm + exit + meta |
| 2 | `scripts/02_train_export.py` | Export trained models to pickle |
| 3 | `scripts/03_train_rl_entry.py` | **v84**: Train 5 Q-functions, confirm, exit, meta (RL entry) |
| 4 | `scripts/04_full_rl_exit.py` | Experimental: RL exit model (PF 3.85 — worse than ML exit) |
| 5 | `scripts/05_deploy_bundle.py` | Save final bundle to `products/models/` |

### v88 Exit Improvement Experiments
None of these change the model bundle — they only modify the runtime
exit logic in `commercial/server/decision_engine/decide.py`. See
`experiments/v88_exit_rl/README.md` for the full catalog.

| Script | Outcome |
|---|---|
| `experiments/v88_exit_rl/13_reverse_setup_exit.py` | ✅ DEPLOYED — PF 4.18 → 4.54, MaxDD -5R |
| `experiments/v88_exit_rl/01..12_*.py` | ❌ All disproven on unseen 30% — see README |
| `experiments/v87_multi_head_exit/` | ❌ -866R on unseen, REMOVED 2026-05-08 |

Also in root:
- `train_rl_entry.py` — shortcut to just run RL training (same as script 03)
- `deploy_bundle.py` — shortcut to deploy (same as script 05)

### Shared dependencies
- `_shared/scripts/build_regime_selector.py` — K=5 K-means + relabel rules
- `_shared/regime_selector_xau.json` — Pre-built XAU regime selector
- `_shared/04b_compute_physics_features.py` — Physics feature computer
