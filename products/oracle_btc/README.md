# Oracle BTC — RL-Enhanced BTCUSD Model

> **Version:** v90 entry + v88 reverse-setup exit + **v97 wider hard SL (6 ATR)**
> **Holdout PF (q>3):** **5.43** at v97 (was 4.45 at SL=4) | **WR:** **73.4%** | **+15,264R**
> **Bundle:** `oracle_btc_validated.pkl` | **Deployed:** v84 2026-05-06 → v88 2026-05-08 → v89 2026-05-10 → v90 2026-05-12 → **v97 2026-05-13**

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
| v84 | RL Q-functions (V72L only) | 3.82 | 67.9% | 1,135 | Q-learning replaces rules |
| v84 + v88 reverse-setup exit | (same entry, smarter exit) | 4.69 | 72.0% | 1,127 | Symmetric RL exit when opposite setup fires |
| v89 + v88 | RL Q-functions (V72L + maturity) | 5.27 | 75.1% | 1,373 | 3 maturity features added; min_q 0.3→3.0 |
| v90 + v88 | RL Q-functions (V72L + maturity + 24h-mom) | 4.45 | 73.4% | 4,325 | 2 direction-signed 24h-return features added; +13.5% PF / +13.3% R at q>3 |
| **v97 (v90 + v88, SL widened 4→6 ATR)** | (same model, looser stop) | **5.43** ★ | **73.4%** | **4,325** | **+1,585R / +12% PF — 56% of 4-ATR stops were being hunted (recovered within 60 bars)** |

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

## Bundle Structure (v89)

```python
{
    'q_entry': {0..4: XGBRegressor},      # 5 Q-models, INPUT 21 feats (V72L+maturity)
    'q_entry_features': [...18 V72L..., 'stretch_100', 'stretch_200', 'pct_to_extreme_50'],
    'v89_maturity': True,                  # marker
    'mdls': {(0..4,'RL'): XGBClassifier},  # confirm heads, V72L only (unchanged)
    'thrs': {(0..4,'RL'): 0.50, ...},
    'exit_mdl': XGBClassifier,             # binary ML exit, EXIT_FEATS (unchanged)
    'meta_mdl': XGBClassifier,             # meta gate, V72L+dir+cid (unchanged)
    'meta_threshold': 0.775,
    'version': 'v89-mat-btc',
}
```

The v89 change is **only** to q_entry. Confirm and meta heads were validated to handle the new q_entry distribution natively without retraining (script 12/13 full-pipeline test).

---

## v89 Maturity Features (the v84→v89 change)

Three direction-signed features capture *trend extension*, computed from price history at entry time:
- `stretch_100` — ATRs above 100-bar low (long) / below 100-bar high (short)
- `stretch_200` — same on 200-bar window
- `pct_to_extreme_50` — position within last 50-bar range, 0 = opposite extreme, 1 = at top of recent rally / bottom of selloff

These prevent entries at *the top of an extended rally* / *bottom of an extended selloff* — entries that look like clean pullbacks but are actually at trend exhaustion / reversal points.

**Disprove evidence (BTC):** at `q>3.0`, % of trades with `stretch_100 > 10 ATRs` drops 18.3% (v84) → **11.8%** (v89), a 36% reduction in stretched-leg entries.

---

## Deployment

### Server config (`configs/oracle_btc.py`)
```python
rl_entry_mode: bool = True
rl_min_q: float = 3.0   # v89 calibrated; v84 was 0.3
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
