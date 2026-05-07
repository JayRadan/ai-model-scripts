# Midas XAU — Entry-Level XAUUSD Pattern Scanner

> **Version:** v83c (Rule-based) | **Holdout PF:** 5.25 | **WR:** 73.4% | **Trades:** 1,693  
> **Bundle:** `midas_xau_validated.pkl` (31 MB) | **Deployed:** 2026-05-06

The entry-level product. Same 28-rule pattern catalog as Oracle, but **no
meta gate** — trades every per-rule confirmation that passes Stage 1.
Higher trade count, simpler pricing tier. Biggest beneficiary of the v83c
range filter + kill-switch (+2.70 PF over v6 baseline).

---

## Performance History

| Version | Entry Type | PF | WR | Trades | Key Change |
|---|---|---|---|---|---|
| v6 | Rule-based (28 rules) | 2.55 | 59.7% | 1,582 | Baseline Midas (14 features, no meta) |
| **v83c** | **Rule-based + filters** | **5.25** | **73.4%** | **1,693** | 4h regime + range filter + kill-switch |
| v84 RL | RL Q-functions | ~2.0 | ~40% | — | **Failed — rolled back.** 14 features insufficient for RL |

### v83c Holdout by Regime

| Regime | Trades | WR | PF | Total R |
|---|---|---|---|---|
| C0 Uptrend | 688 | 79.7% | **9.12** | +2,891 |
| C1 MeanRevert | 228 | 73.2% | 3.15 | +189 |
| C2 TrendRange | 223 | 64.1% | 2.21 | +120 |
| C3 Downtrend | 314 | 77.4% | **5.87** | +1,196 |
| C4 HighVol | 240 | 69.2% | 2.98 | +178 |

---

## Why no meta gate?

Midas is the higher-trade-frequency entry product. The meta gate would
cut ~30-40% of trades to chase higher PF — but Midas customers prefer
more frequent activity over Oracle-tier selectivity. This makes Midas
the most responsive product — it catches moves Oracle might pass on.

---

## Features (14 v6)

Midas uses a reduced feature set compared to Oracle (14 vs 18):
```
hurst_rs, ou_theta, entropy_rate, kramers_up, wavelet_er,
vwap_dist, hour_enc, dow_enc, quantum_flow, quantum_flow_h4,
quantum_momentum, quantum_vwap_conf, quantum_divergence,
quantum_div_strength
```
Missing vs v72l: `vpin`, `sig_quad_var`, `har_rv_ratio`, `hawkes_eta`.

---

## Why RL failed for Midas

The v84 RL experiment (`experiments/v84_rl_entry/06_midas_rl.py`) produced
~40% WR and PF ~2.0 — worse than rule-based v83c (73.4% WR, PF 5.25).

Root causes:
1. **14 features insufficient** — the 4 missing features (vpin, sig_quad_var,
   har_rv_ratio, hawkes_eta) carry critical microstructure/volatility info
2. **137 per-rule confirm models** beat 5 RL-confirms — the hand-coded
   catalog benefits from specialized, rule-specific confirmation heads
3. **No meta gate** means lower-quality RL entries aren't filtered out

Future: upgrading Midas to 18 v72l features could make RL viable.

---

## Architecture

```
M5 bar ──► K=5 regime selector (4h-step) ──► cid ∈ {0..4}
                       │
                       ▼
              28-rule hand-coded catalog
              (rule detection on OHLC patterns)
                       │
                       ▼
              Per-rule confirm head (XGB, 14 v6 features)
              p_conf ≥ per-rule threshold ?  (137 models total)
                       │ YES
                       ▼
                    OPEN TRADE
              (NO meta gate — simpler, more trades)
```

---

## Hyperparameters

| Component | n_estimators | max_depth | lr | Notes |
|---|---|---|---|---|
| Confirm (XGBClassifier) | 200 | 3 | 0.05 | 137 models, one per (cid, rule) |
| Exit (XGBClassifier) | 300 | 5 | 0.05 | Same as Oracle |

| Parameter | Value |
|---|---|
| Confirm threshold | Per-rule (auto-calibrated, ~0.50) |
| MAX_HOLD | 60 bars |
| SL_HARD | -4.0R |
| TP target | 2:1 RR |

---

## Bundle Structure (31 MB — largest)

```python
{
    'mdls': {(cid, rule): XGBClassifier, ...},  # 137 confirm models
    'thrs': {(cid, rule): float, ...},           # Per-rule thresholds
    'exit_mdl': XGBClassifier,
    'meta_mdl': XGBClassifier,                   # Meta still present but not used
    'meta_threshold': 0.675,
    'v6_feats': [...],                           # 14 v6 feature names
}
```

No `q_entry` key — RL is disabled (`rl_entry_mode=False` in config).

---

## Deployment

### Server config (`configs/midas_xau.py`)
```python
rl_entry_mode: bool = False    # Rolled back to rule-based
regime_selector_json: str = "regime_selector_4h.json"
kill_switch_losses: int = 3
range_filter_lookback: int = 20
```

### Push to Render
```bash
cd /home/jay/Desktop/my-agents-and-website/commercial
cp ../new-model-zigzag/products/models/midas_xau_validated.pkl \
   server/decision_engine/models/
git add server/decision_engine/models/midas_xau_validated.pkl
git commit -m "deploy midas_xau v83c (rule-based, PF 5.25)"
git push origin main
```

### v85 Drawdown Circuit Breaker
Shared across all products. Tracks cumulative PnL; blocks entries when
PnL drops >25% from session peak. Three unblock paths: regime change,
4h timeout, or PnL recovery to within 10% of peak.

### EA Endpoint
`POST /decide/midas_xau`

---

## Full Pipeline Scripts

| # | Script | What it does |
|---|---|---|
| 1 | `scripts/01_validate_v6.py` | Original v6 training — 14 features, 28 rules, confirm only |
| 2 | `scripts/02_rl_experiment.py` | **v84 RL attempt** — Q-learning on 14 features (PF ~2.0, WR ~40%) |

**Why only 2 scripts?** Midas shares Oracle's v83c infrastructure:
- Regime selector: same `regime_selector_4h.json` as Oracle XAU
- Range filter + kill-switch: same `_shared/` config, applied via `deploy_midas_v83c.json`
- The confirm models were retrained via the Oracle pipeline with Midas features

### Additional files
- `holdout_trades.csv` — v83c holdout trades (PF 5.25, 1,693 trades)
- `deploy_midas_v83c.json` — v83c deployment config snapshot
