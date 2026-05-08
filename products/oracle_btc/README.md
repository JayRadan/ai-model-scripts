# Oracle BTC — RL-Enhanced BTCUSD Model

> **Version:** v84 entry + v88 reverse-setup exit
> **Unseen-30% PF:** 5.26 (was 4.27 pre-v88 exit) | **WR:** 67.9% | **MaxDD:** 25R (was 30R)  
> **Bundle:** `oracle_btc_validated.pkl` (4.8 MB) | **Deployed:** 2026-05-06 (entry) / 2026-05-08 (exit)

Same v72l architecture as Oracle XAU, but trained on Bitcoin M5 data with
its own BTC-specific regime selector (K=5, full_directional relabel).
RL Q-functions replaced the 28-rule catalog — biggest PF improvement
of any product (+0.79 over rule-based).

---

## Performance History

| Version | Entry Type | PF | WR | Trades | Key Change |
|---|---|---|---|---|---|
| v7.2 | Rule-based (28 rules) | 2.85 | 61.8% | 1,899 | Baseline BTC |
| v83c | Rule-based + filters | 3.03 | 64.4% | 1,033 | 4h regime + range filter + kill-switch |
| **v84** | **RL Q-functions** | **3.82** | **67.9%** | **1,135** | **Q-learning replaces rules (+0.79!)** |

### v84 Holdout by Regime

| Regime | Trades | WR | PF | Total R |
|---|---|---|---|---|
| C0 Uptrend | 359 | 75.2% | **5.58** | +1,644 |
| C1 MeanRevert | 161 | 55.9% | 1.53 | +85 |
| C2 TrendRange | 154 | 51.3% | 1.29 | +44 |
| C3 Downtrend | 362 | 74.6% | **6.50** | +1,991 |
| C4 HighVol | 99 | 62.6% | 3.88 | +337 |

---

## Why BTC improved most with RL

BTC's hand-coded rules struggled because:
1. **Higher volatility** — fixed-pattern rules (swing highs, false breakdowns)
   fire too often during noise
2. **24/7 market** — the K-means regime detector misclassifies weekend
   drift periods
3. **Different microstructure** — quantum flow features capture BTC's
   unique order-flow signature better than pattern rules

The RL Q-functions learn directly from PnL outcomes, adapting to BTC's
specific feature-response surface without human priors.

---

## BTC-Specific Regime Relabel

BTC uses `full_directional` relabel (unlike XAU's `highvol_only`):
- **Any** cluster can be overridden by strong 24h return (±1.0%)
- Prevents MeanRevert from capturing trending bars (common BTC issue)
- Higher threshold than XAU (1.0% vs 0.3%) due to BTC's larger daily ranges

---

## Architecture & Features

Identical to Oracle XAU:
- 18 v72l features for entry/confirm/meta
- 11 exit features
- 20 meta features (18 v72l + direction + cid)
- K=5 K-means regime selector (4h-step)
- Same Q→confirm→meta pipeline

See `oracle_xau/README.md` for detailed architecture diagram.

---

## Hyperparameters

Same as Oracle XAU except:
| Parameter | XAU | BTC |
|---|---|---|
| Meta threshold | 0.775 | 0.525 (auto-swept) |
| Regime relabel | highvol_only (0.3%) | full_directional (1.0%) |

---

## Training

### Prerequisites
```bash
swing_v5_btc.csv                 # 830K M5 bars from Dukascopy (~577 MB)
setups_*_v72l_btc.csv            # 5 files, one per regime cid
regime_fingerprints_btc_4h.csv   # 4h-step regime labels
regime_selector_btc_4h.json      # K=5 centroids + relabel rules
```

### Train RL Entry
```bash
cd /home/jay/Desktop/new-model-zigzag
source .venv/bin/activate
python3 products/oracle_btc/train_rl_entry.py
```

Script: `train_rl_entry.py` (originally `experiments/v84_rl_entry/07_btc_rl.py`)

Same pipeline as Oracle XAU — loads BTC data, trains 5 Q-models, confirm,
exit, meta, and saves bundle. Also runs holdout comparison against v83c
rule-based trades.

---

## Bundle Structure

```python
{
    'q_entry': {0: XGBRegressor, ..., 4: XGBRegressor},  # 5 Q-models
    'mdls': {(0,'RL'): XGBClassifier, ..., (4,'RL'): XGBClassifier},
    'thrs': {(0,'RL'): 0.50, ...},
    'exit_mdl': XGBClassifier,
    'meta_mdl': XGBClassifier,
    'meta_threshold': 0.525,     # BTC-specific (lower than XAU's 0.775)
    'min_q': 0.3,
    'version': 'v84-rl-btc',
}
```

---

## Deployment

### Server config (`configs/oracle_btc.py`)
```python
rl_entry_mode: bool = True
rl_min_q: float = 0.3
regime_selector_json: str = "regime_selector_btc_4h.json"
kill_switch_losses: int = 3
range_filter_lookback: int = 20
```

### Push to Render
```bash
cd /home/jay/Desktop/my-agents-and-website/commercial
cp ../new-model-zigzag/products/models/oracle_btc_validated.pkl \
   server/decision_engine/models/
git add server/decision_engine/models/oracle_btc_validated.pkl \
        server/decision_engine/configs/oracle_btc.py
git commit -m "deploy oracle_btc v84 RL bundle (PF 3.82)"
git push origin main
```

### v85 Drawdown Circuit Breaker
All products share the equity drawdown guard (`state.py` → `DrawdownGuard`).
Blocks entries when PnL drops >25% from session peak; unblocks on regime
change, 4h timeout, or PnL recovery. Backtested on 2026-05-07 reversal:
+244% PnL improvement vs no guard.

### v88 Reverse-Setup RL Exit (deployed 2026-05-08)
At every in-trade bar, scans all 30 rule detectors for any
opposite-direction setup. If one fires AND `q_entry[cid](v72l) > 0.10`,
exit. Implementation in `decide_exit` of
[`commercial/server/decision_engine/decide.py`](../../../my-agents-and-website/commercial/server/decision_engine/decide.py),
between hard-SL and trail. ~232 ms per call.

Validated on the unseen 30% of v84 RL trades (BTC 341 trades,
2025-11-11 → 2026-05-01):

| Metric | Pre-v88 exit | Post-v88 exit | Δ |
|---|---|---|---|
| PF | 4.27 | **5.26** | **+0.99** |
| Total R | +841R | +865R | +24R |
| MaxDD | 30R | **25R** | **-5R** |
| Reverse-setup exits | — | 51 / 341 (15%) | new |
| Hard SL hits | 39 | 28 | -11 |
| Max-hold exits | 214 | 186 | -28 |

BTC saw the biggest PF lift (+0.99) of any product because a larger
fraction of its trades are in regimes where opposite-direction setups
fire frequently. Source:
`experiments/v88_exit_rl/13_reverse_setup_exit.py`. Full disprove
catalog of 12 other approaches in
`experiments/v88_exit_rl/README.md`.

### EA Endpoint
`POST /decide/oracle_btc` — returns RL decisions with `rule="RL"`.

---

## Full Pipeline Scripts

| # | Script | What it does |
|---|---|---|
| 1 | `scripts/01_validate_v72l.py` | Original v72l BTC training — 28 rules + confirm + meta |
| 2 | `scripts/02_train_export.py` | Export models to pickle |
| 2b | `scripts/02b_build_selector.py` | Build BTC K=5 regime selector |
| 3 | `scripts/03_v83c_pipeline.py` | v83c: 4h regime + range filter + kill-switch |
| 4 | `scripts/04_train_rl_entry.py` | **v84**: Train RL Q-functions, confirm, exit, meta |

Root shortcut: `train_rl_entry.py` (same as script 04)

### v88 Exit Improvement Experiments
Runtime-only changes (no model retrain). See
`experiments/v88_exit_rl/README.md` for the full 13-experiment catalog.

| Script | Outcome |
|---|---|
| `experiments/v88_exit_rl/13_reverse_setup_exit.py` | ✅ DEPLOYED — PF 4.27 → 5.26, MaxDD -5R |
| `experiments/v88_exit_rl/11_regime_conditional_be.py` | ⚠️ small win on BTC (+8R, PF 4.27→4.44) — NOT deployed (user opted for v13 only) |
| `experiments/v88_exit_rl/01..10,12_*.py` | ❌ All disproven on unseen — see README |
| `experiments/v87_multi_head_exit/` | ❌ -674R on unseen, REMOVED 2026-05-08 |
