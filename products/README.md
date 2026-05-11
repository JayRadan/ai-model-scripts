# Edge Predictor — Production Model Catalog

> **Last updated:** 2026-05-06  
> **GitHub:** `JayRadan/edge_predictor` (auto-deploys to Render on push to `main`)

Four production trading models for XAUUSD and BTCUSD (M5 timeframe).
All trained on Dukascopy tick data, validated on a strict 2024-12-12+
out-of-sample holdout.

---

## Quick Reference

| Product | Asset | Entry | Exit | PF (holdout) | WR | MaxDD | Bundle |
|---|---|---|---|---|---|---|---|
| **Oracle XAU** | XAUUSD | RL v89 (V72L+maturity) | trail + v88 reverse-setup | **6.44** | **77.4%** | **27R** | 5.6 MB |
| **Oracle BTC** | BTCUSD | RL v89 (V72L+maturity) | trail + v88 reverse-setup | **5.27** | **75.1%** | **36R** | 5.7 MB |

> Midas XAU and Janus XAU were retired 2026-05-11 — customers notified.
> Files in this folder and on the server were removed in the same commit
> sequence as the Oracle v89 deploy.

Evolution of Oracle PF/DD on the post-2024-12-12 holdout:

| Step | Date | XAU PF | XAU DD | BTC PF | BTC DD |
|---|---|---|---|---|---|
| v84 RL entry only | 2026-05-06 | 4.21 | 20R | 3.82 | 30R |
| + v88 reverse-setup exit | 2026-05-08 | 4.60 | 36R | 4.69 | 43R |
| + v89 maturity-aware q_entry | **2026-05-10** | **6.44** | **27R** | **5.27** | **36R** |

Each step is a clean improvement (commits `d1d22f1` → `308947c` → `7cb5a8f` on `JayRadan/edge_predictor:main`).

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

### v85 Equity Drawdown Circuit Breaker (all products)
Prevents giving back profits during sharp reversals. Tracks cumulative
PnL per product; if PnL drops >25% from session peak and regime hasn't
changed, blocks ALL new entries for that product. Auto-unblocks when
the regime classifier detects a regime change.

Backtested on live 2026-05-07 reversal: +244% PnL improvement (149.7R
vs 43.5R without guard). See `state.py` → `DrawdownGuard`.

### v89 Maturity-Aware q_entry (Oracle XAU + BTC)
Deployed 2026-05-10 (commit `7cb5a8f`). Adds 3 direction-signed trend-maturity
features to the q_entry input:

- `stretch_100` — ATRs above 100-bar low (long) / below 100-bar high (short)
- `stretch_200` — same on 200-bar window
- `pct_to_extreme_50` — position within last 50-bar range (0 = opposite extreme, 1 = at top of recent rally / bottom of selloff)

This prevents the RL from entering pullbacks at the *top of an extended rally*
or *bottom of an extended selloff* — trades that look like clean entries but
are actually at trend-exhaustion / reversal points.

Validated on the post-2024-12-12 holdout through full prod pipeline (q_entry
→ confirm → meta@0.775 → simulate w/ v88+trail+SL+max60 → kill-switch):

| | Oracle XAU | Oracle BTC |
|---|---|---|
| v84 + v88 (pre-v89): | PF 4.60, DD 36R, WR 71.4% | PF 4.69, DD 43R, WR 72.0% |
| **v89 + v88 (current):** | **PF 6.44, DD 27R, WR 77.4%** | **PF 5.27, DD 36R, WR 75.1%** |
| Δ | +1.84 PF, -9R DD, +6.0pp WR | +0.58 PF, -7R DD, +3.1pp WR |

Stretched-entry exposure cut **38% on XAU, 36% on BTC** — at `q>3.0` the
share of trades opened with `stretch_100 > 10 ATRs` drops from 14.1%→8.7%
(XAU) and 18.3%→11.8% (BTC).

q_entry input dim: 18 (V72L) → **21** (V72L + 3 maturity).
`MIN_Q` recalibrated: **0.3 → 3.0** (the new q_entry's distribution shifted).
Confirm + meta heads unchanged. See `experiments/v89_smart_exit/README.md`.

### v88 Reverse-Setup RL Exit (Oracle XAU + BTC)
Deployed 2026-05-08 (commit `d1d22f1`). At every in-trade bar, scans
all 30 rule detectors for any *opposite-direction* setup. If one fires
AND `q_entry[cid].predict(v72l_feats) > 0.10`, exit. The same RL that
picks good entries is now flagging a high-quality setup against the
open trade.

Validated on the unseen 30% of v84 RL trades:
- **Oracle XAU:** PF 4.18 → **4.54**, MaxDD 20R → **15R**
- **Oracle BTC:** PF 4.27 → **5.26**, MaxDD 30R → **25R**

This is the first positive result across 13 different exit-improvement
experiments. Why it works while every other angle failed: q_entry is
used as it was trained — only at bars where a rule pattern fires.
Naive q_entry-on-every-bar destroyed PF (-413R XAU). See
`experiments/v88_exit_rl/README.md` for the full disprove catalog.

Implementation: see `decide_exit` in
[`commercial/server/decision_engine/decide.py`](../../my-agents-and-website/commercial/server/decision_engine/decide.py).
Latency: ~232ms per call (200-bar slice before rule scanning).
Activates only when q_entry is loaded.

### Removed v87 Multi-Head Exit
v87's 4-head exit policy (giveback, upside, stop, new_high) was
deployed in commit `496e7be` claiming +265R, but a proper unseen-window
sweep (1470 combos) showed it destroyed -866R XAU / -674R BTC. Removed
2026-05-08 commit `350f7b8`. See `experiments/v87_multi_head_exit/README.md`.

---

## Folder Layout

```
products/
├── README.md                     ← THIS FILE
├── models/                       ← All .pkl bundles (gitignored, on disk)
│   ├── oracle_xau_validated.pkl
│   └── oracle_btc_validated.pkl
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
└── oracle_btc/                   ← BTC RL model
    ├── README.md
    ├── train_rl_entry.py
    └── scripts/
        ├── 01_validate_v72l.py
        ├── 02_train_export.py
        ├── 02b_build_selector.py       ← BTC K=5 regime selector
        ├── 03_v83c_pipeline.py         ← v83c range filter + kill-switch
        └── 04_train_rl_entry.py
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
| v84 RL Exit | `experiments/v84_rl_entry/02_full_rl.py` | PF 3.85 (worse than ML exit) |
| v84 Improved Exit | `experiments/v84_rl_entry/03_improved_exit.py` | PF 3.65 (worse) |
| v87 Multi-Head Exit | `experiments/v87_multi_head_exit/` | ❌ -866R XAU / -674R BTC on unseen, REMOVED |
| v88 Exit RL (12 angles) | `experiments/v88_exit_rl/` | 11/13 disproven; reverse-setup wins |
| **v88 Reverse-Setup RL Exit** | `experiments/v88_exit_rl/13_reverse_setup_exit.py` | ✅ XAU PF +0.36, BTC PF +0.99, both DD lower, DEPLOYED |
| **v89 Maturity-Aware q_entry** | `experiments/v89_smart_exit/11_retrain_q_with_maturity.py` | ✅ XAU PF 4.60→6.44 / DD 36→27R, BTC PF 4.69→5.27 / DD 43→36R, DEPLOYED 2026-05-10 |
| **Janus + Midas removal** | (this commit) | 2026-05-11: products retired entirely from server/products/website |
