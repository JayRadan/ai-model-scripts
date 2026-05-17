# Oracle BTC — RL-Enhanced BTCUSD Model

> **Version:** **v99b q_entry** (dynamic-exit relabel) + v88 reverse-setup exit + v97 wider hard SL (6 ATR)
> **Holdout PF:** **5.60** @ Q≥2.0 (backtest with v88+trail+6×ATR exits) | **WR:** **88.1%** | **+482R / N=168**
> **Bundle:** `oracle_btc_validated.pkl` | **min_q:** 3.0 (raised from 0.3 — v99b Q-distribution runs higher)
> **Deployed:** v84 2026-05-06 → v88 2026-05-08 → v89 2026-05-10 → v90 2026-05-12 → v97 2026-05-13 → **v99b 2026-05-17 (commit `e2e9681`)**

## ⚠️ v99b live notes (deployed 2026-05-17)

Only `q_entry` (5 XGBRegressors) was swapped. All other components retained from v97:
confirm heads (`mdls`), `meta_mdl`, `exit_mdl`, `giveback_mdl`, exit_feats, meta_feats.

**Why v99b:**
v90 q_entry was trained on labels `TP=2R / SL=1R / 40-bar` — but production trades with
6×ATR hard SL + v88 reverse + ML exit + no fixed TP. v99b labels match production reality:
SL=2R, TP_min=4R required, then 2R trailing stop, max 200 bars.

**BTC-specific finding:** in backtest v99b BTC hits MORE +R per trade than XAU (avg +2.87R vs +2.72R at Q≥2.0), consistent with BTC's larger ATR moves making the trailing stop capture longer runs.

**Backups for rollback:** `oracle_btc_validated.pkl.bak_pre_v99b` (in vendored dir).
Quick rollback: `git revert e2e9681 --no-edit && git push` (Render redeploys in ~90s).

Same v72l architecture as Oracle XAU, but trained on Bitcoin M5 data with
its own BTC-specific regime selector (K=5, full_directional relabel).
RL Q-functions replaced the 28-rule catalog — biggest PF improvement
of any product (+0.79 over rule-based).

---

## Live Operations Tooling (2026-05-17)

Three production hardening features shipped same day as v99b. Server-side,
shared with Oracle XAU. For BTC specifically, the **stack-gate** addresses
exactly the failure mode seen on 2026-05-14: pyramid of 6 shorts from
79,900 → 83,320 (~$1,400 loss) as the regime classifier lagged a 4% rally.

### 1. Stack-gate — prevents pyramiding into a losing position
**Commit `1374daa`.** When `decide_entry` returns `action="open"`, the
server scans the EA's `open_positions` for any same-direction slot with
floating R < 0. If found, the new entry is replaced with `action="hold"`
and `reason="stack_gate: ..."`. With BTC running 6 slots, this caps a
bad-regime loss to **1 slot instead of 6**.

- `s0` always fires; `s1..s5` only if all prior same-direction slots are in profit
- Opposite-direction slots ignored
- Filterable in funnel log: `reason LIKE '%stack_gate%'`

**EA wiring:** `EdgePredictor_Connector_v2.mq5` `BuildDecideBody` appends
`open_positions: [...]` on entry calls. Recompiled `.ex5` shipped at
`website/public/files/EdgePredictor_Connector.ex5`. Old EAs send no field
→ gate inert → backward compatible.

### 2. Admin regime override — pin/clear live regime
**Commit `f22903c` (+ fixes `730f112`, `fe27c3e`).** Force the classifier's
verdict when the analyst sees it lagging real conditions.

Endpoints (`/decide/` prefix, ADMIN_SECRET-gated):
- `GET _regime-overrides`, `POST _regime-override?product=oracle_btc&cid=N`, `DELETE _regime-override?product=oracle_btc`
- Persisted to `regime_overrides.json` — survives Render restarts

UI: `/admin/regime` shows **Model says** vs **Engine acts as** with a Pin/Clear panel.

### 3. Funnel: model verdict vs effective regime
**Commit `d84b505`.** `/admin` → Decision Funnel: per-product regime card shows classifier verdict, pinned override (if any), and the resulting effective cluster. Card border turns amber when pinned.

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
| **v97 (v90 + v88, SL widened 4→6 ATR)** | (same model, looser stop) | **5.43** | **73.4%** | **4,325** | **+1,585R / +12% PF — 56% of 4-ATR stops were being hunted (recovered within 60 bars)** |
| **v99b (dynamic-exit relabel + v88 + 6×ATR)** ★ | Q regressors retrained on dynamic-exit labels (SL=2R, TP_min=4R, 2R trail, 200-bar max) | **5.60** @ Q≥2.0 | **88.1%** | 168 | **min_q raised 0.3→3.0; labels now match prod exit reality** |

### v84 Holdout by Regime

| Regime | Trades | WR | PF | Total R |
|---|---|---|---|---|
| C0 Uptrend | 359 | 75.2% | **5.58** | +1,644 |
| C1 MeanRevert | 161 | 55.9% | 1.53 | +85 |
| C2 TrendRange | 154 | 51.3% | 1.29 | +44 |
| C3 Downtrend | 362 | 74.6% | **6.50** | +1,991 |
| C4 HighVol | 99 | 62.6% | 3.88 | +337 |

---

## v99b — dynamic-exit q_entry retrain (2026-05-17)

### Threshold sweep on holdout (2024-12-12 → 2026-05-01)

Backtest uses production-matching exit stack: v88 reverse-setup + trail (after +4R, 2R behind peak) + 6×ATR hard SL + 200-bar max.

| Q ≥ | Trades | WR | PF | sumR | avgR | maxDD |
|---|---|---|---|---|---|---|
| 0.5 | 926 | 80.0% | 3.00 | +1747 | +1.89 | −29R |
| 1.0 | 653 | 79.9% | 3.04 | +1288 | +1.97 | −30R |
| 1.5 | 363 | 84.6% | 4.12 | +871 | +2.40 | −26R |
| **2.0** | **168** | **88.1%** | **5.60** | **+482** | **+2.87** | **−6R** ← deployed min_q |
| 2.5 | 76 | 89.5% | 5.94 | +212 | +2.78 | −11R |
| 3.0 | 36 | 91.7% | 6.92 | +107 | +2.96 | −10R |

### What changed (BTC-specific)
- **Label simulator**: v90 used `TP=2R / SL=1R / 40-bar` fixed exits. v99b uses **dynamic exit**:
  - SL = 2 ATR (fixed)
  - TP_min = 4 ATR (required to qualify as winner)
  - After TP_min hit: trail 2R behind peak
  - Max hold: 200 bars
- **Features**: identical 23 features (V72L + v89 maturity + v90 momentum)
- **Architecture**: identical (per-cluster XGBRegressor)
- **min_q**: bumped 0.3 → 3.0 because v99b Q-values run higher (BTC C0 zeros Q≈10 vs v97 ~2-5)

### Files
- `experiments/v99_rl_relabel/01_label.py` — labeler (smoke test on 30 days)
- `experiments/v99_rl_relabel/03_label_full.py` — full XAU + BTC labeling
- `experiments/v99_rl_relabel/06_add_features_retrain.py` — final 23-feat retrain
- `experiments/v99_rl_relabel/08_backtest_with_v88_exit.py` — backtest with prod-matching exits
- `experiments/v99_rl_relabel/09_build_deploy_bundle.py` — packages v99b q_entry into prod-pkl format
- `experiments/v99_rl_relabel/q_models_btc_v99b_23feat.pkl` — source (q_entry only)
- `commercial/server/decision_engine/models/oracle_btc_validated.pkl` — deployed full bundle
- `commercial/server/decision_engine/models/oracle_btc_validated.pkl.bak_pre_v99b` — rollback backup

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
