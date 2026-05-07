# Janus XAU — Pivot-Score XAUUSD Model

> **Version:** v74 (Pivot-score) | **Holdout PF:** 1.85 | **WR:** 58.3% | **Trades:** 892  
> **Bundle:** `janus_xau_validated.pkl` (5.3 MB) | **Deployed:** 2026-05-05

The experimental product. Uses a single pivot-score model to detect
reversal/continuation points — a completely different approach from the
rule-catalog architecture of Oracle/Midas. Lower PF, higher conceptual
novelty. Also inherits v83c 4h regime + kill-switch.

---

## Performance

| Regime | Trades | WR | PF | Total R |
|---|---|---|---|---|
| C0 Uptrend | 298 | 62.1% | 2.35 | +192 |
| C1 MeanRevert | 195 | 58.5% | 1.75 | +88 |
| C2 TrendRange | 161 | 55.9% | 1.42 | +41 |
| C3 Downtrend | 168 | 57.7% | 1.88 | +95 |
| C4 HighVol | 70 | 54.3% | 1.21 | +12 |

---

## Architecture

```
M5 bar ──► pivot_score model (XGB) ──► score ∈ [0,1]
                       │
                       ▼
              pivot_dir model (XGB) ──► direction (buy/sell)
                       │
                       ▼
              score ≥ per-regime threshold ?
                       │ YES
                       ▼
                    OPEN TRADE
              (rule = "RP_score")
```

Unlike Oracle/Midas, Janus does NOT use the 28-rule catalog.
A single `pivot_score` XGBClassifier produces a confidence score
that the bar is a pivot point; a companion `pivot_dir` model
predicts the direction.

---

## Features (6 — compact)

```
pivot_position    — Where the bar sits relative to recent pivots
bar_range_pct     — Bar range as % of ATR
volume_ratio      — Volume relative to recent average
spread_ratio      — Spread relative to ATR
flow_imbalance    — Net buy vs sell flow
mtm_3bar          — 3-bar momentum
```

---

## v83c Inheritance

- 4h-step regime selector (same as Oracle/Midas)
- Consecutive-loss kill-switch (3 SL → 12h cooldown)
- No range filter (pivot model doesn't use OHLC patterns)
- No cohort kills (single model, not per-rule)

---

## Why the lower PF?

Janus is the most experimental product. The pivot-score approach captures
a different signal than pattern rules — it trades less frequently and
with lower WR. It's included for diversification: when Oracle/Midas are
quiet (few pattern matches), Janus may still fire on pivot points.

---

## Bundle Structure

```python
{
    'pivot_score_mdl': XGBClassifier,   # Predicts pivot confidence
    'pivot_dir_mdl': XGBClassifier,     # Predicts direction
    'pivot_score_thresholds': {cid: float},  # Per-regime thresholds
}
```

No `q_entry`, no `mdls` dict — Janus uses a completely different cascade
(see `decide_janus.py` in the server).

---

## Deployment

### Server config (`configs/janus_xau.py`)
```python
regime_selector_json: str = "regime_selector_4h.json"
kill_switch_losses: int = 3
kill_switch_cooldown_hours: int = 12
```

### Push to Render
```bash
cd /home/jay/Desktop/my-agents-and-website/commercial
cp ../new-model-zigzag/products/models/janus_xau_validated.pkl \
   server/decision_engine/models/
git add server/decision_engine/models/janus_xau_validated.pkl
git commit -m "deploy janus_xau v74 (pivot-score)"
git push origin main
```

### v85 Drawdown Circuit Breaker
Shared across all products. See `state.py` → `DrawdownGuard`. Blocks
when PnL drops >25% from session peak; unblocks on regime change, 4h
timeout, or PnL recovery.

### EA Endpoint
`POST /decide/janus_xau`

---

## Full Pipeline Scripts

| # | Script | What it does |
|---|---|---|
| 0 | `scripts/00_compute_features.py` | Compute pivot features from OHLC |
| 1 | `scripts/01_label_bars.py` | Label every bar as pivot/non-pivot |
| 2 | `scripts/02_train_pivot.py` | Train pivot_score + pivot_dir XGB models |
| 3 | `scripts/03_build_setups.py` | Build setup signals from pivot predictions |
| 4 | `scripts/04_validate.py` | Walk-forward validation with meta gate |
| 5 | `scripts/05_pickle_janus.py` | Save `janus_xau_validated.pkl` for server deployment |

### Output files
- `models/pivot_score_v74.json` — Pivot score model (XGBoost JSON)
- `models/pivot_dir_v74.json` — Pivot direction model (XGBoost JSON)
- `../models/janus_xau_validated.pkl` — Deployed bundle (5.3 MB)
