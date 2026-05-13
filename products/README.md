# Edge Predictor — Production Model Catalog

> **Last updated:** 2026-05-13  
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

Evolution of Oracle on the post-2024-12-12 holdout:

| Step | Date | XAU PF / R | BTC PF / R |
|---|---|---|---|
| v84 RL entry only | 2026-05-06 | 4.21 / +959R | 3.82 / +841R |
| + v88 reverse-setup exit | 2026-05-08 | 4.60 / +957R | 4.69 / +880R |
| + v89 maturity-aware q_entry | 2026-05-10 | 6.44 / +7,092R (q>3) | 5.27 / +12,070R (q>3) |
| + v90 24h-momentum q_entry | 2026-05-12 | 4.31 / +7,536R (q>3) | 4.45 / +13,680R (q>3) |
| **+ v97 wider hard SL (4→6 ATR)** | **2026-05-13** | **5.04 / +8,286R (q>3)** | **5.43 / +15,264R (q>3)** ★ |

PF dropped slightly on XAU at v90 because the model trades MORE setups (n=2,452 vs 2,178 at v89) and the marginal trades carry slightly lower PF — but total R rises. BTC sees clean PF + R wins across the board (+13.5% PF, +13.3% R).

v97 widens the hard SL after a stop-hunt diagnostic showed ~56% of 4-ATR stops would recover within 60 bars on the holdout. No model retraining; pure exit-policy change.

Commits: `d1d22f1` → `308947c` → `7cb5a8f` → `eec93e8` → `5816fbc` on `JayRadan/edge_predictor:main`.

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

### v97 Wider Hard SL (Oracle XAU + BTC)
Deployed 2026-05-13 (commit `5816fbc`). Pure config change — `sl_hard_atr` widened
from **4.0 → 6.0** in both `oracle_xau.py` and `oracle_btc.py`. **No model retrain,
no EA redistribution.**

**Root cause** (diagnostic at `experiments/v91_smart_regime/07_sl_hunt_check.py`):
on the holdout, **~56% of 4-ATR stops recovered the full 4R within 60 bars** — a
textbook stop-hunt / liquidity-sweep pattern. Median post-SL favorable move at
60 bars: +4.61R XAU / +4.62R BTC. 87% of stops saw at least +1R recovery.

**Validated** via full holdout PnL replay (`08_wider_sl_pnl.py`) — every q-threshold
on both products, no exception:

| Product | min_q | SL=-4 (was) | SL=-6 (deployed) | Δ R |
|---|---|---|---|---|
| Oracle XAU | 0.5 | PF 2.90, +19,342R | PF 3.19, +21,278R | **+1,936** |
| Oracle XAU | 1.0 | PF 3.23, +17,308R | PF 3.61, +19,022R | **+1,714** |
| Oracle XAU | 2.0 | PF 3.69, +12,206R | PF 4.17, +13,370R | **+1,164** |
| **Oracle XAU** | **3.0** | **PF 4.31, +7,536R** | **PF 5.04, +8,286R** | **+750** |
| Oracle BTC | 0.5 | PF 2.64, +29,473R | PF 2.99, +33,372R | **+3,899** |
| Oracle BTC | 1.0 | PF 2.96, +28,039R | PF 3.37, +31,437R | **+3,398** |
| Oracle BTC | 2.0 | PF 3.53, +20,730R | PF 4.12, +23,195R | **+2,465** |
| **Oracle BTC** | **3.0** | **PF 4.45, +13,680R** | **PF 5.43, +15,264R** | **+1,585** |

**Tradeoffs**:
- Per-trade max loss is 1.5× larger in absolute price terms. To keep dollar-risk
  per trade constant, halve lot size (XAU 0.05→0.033, BTC 0.10→0.067).
- Trail stop activation R-units now equal 18×ATR profit (was 12×ATR). Fewer trades
  reach trail-activation; backtest already accounts for this in totals.
- Exit head (`p_exit`) sees `unrealized_pnl_R` scaled 0.67× smaller → fires less
  often → trades run longer → behaviour converges toward backtest (which had no
  exit-head firing). Net PnL impact: positive.

EA: no change required. The connector reads `sl_atr_mult` from each `/decide`
response (server now sends 6.0). **Customer EAs are not redistributed.**

### Why other regime-fix paths were disproven (chart-staleness investigation)
A chart screenshot 2026-05-12 showed regime label = Uptrend during a clearly
bearish day — followed by a losing long. Five independent investigations sought
the cause; **none of the regime-routing fixes survived holdout validation**, but
the diagnostic surfaced the SL-hunt issue (above) which DID:

| Experiment | Approach | Holdout result | Verdict |
|---|---|---|---|
| `v92_supervised_regime/` | XGB trained to predict next-12h regime from current features | XAU: at conf>0.30, "Up" → mean fwd return **-0.20%** (anti-edge); BTC ~random at every threshold | ❌ Forward direction unpredictable |
| `v95_regime_repaint_ab/` | Bar-level A/B: block-aligned vs trailing-window K-means | XAU 40% disagree, BTC 65% disagree — staleness real | (informational only) |
| `v91/03_validate_trading_impact.py` | Naive flip to trailing cluster routing through current q_entry | XAU **-401R**, BTC **-1,030R** at q>3 | ❌ q_entry trained on block-aligned cids |
| `v91/04_trailing_trade_replay.py` | Subsample replay of (3) | XAU **-401R** confirmed | ❌ |
| `v91/05_full_pipeline_trailing.py` | Full retrain: q_entry refit on trailing cids | XAU **still -185R** vs prod at q>3 | ❌ Stale label is the right contract |
| `v91/06_momentum_veto.py` | Veto longs when regime=Up but 24h ret < -threshold | XAU **-1,569R**, BTC **-2,750R** at v=0.5%, q>3 | ❌ Kills the system's BEST trades (Up-fade-pullback pattern) |
| `v91/07_sl_hunt_check.py` | Diagnostic — what does price do AFTER SL hits? | **56% recover 4R in 60 bars** | ✅ Found the real issue |
| `v91/08_wider_sl_pnl.py` | Full PnL replay at SL = -4, -5, -6, -8, -10 | +750R XAU / +1,585R BTC at -6 | ✅ DEPLOYED as v97 |

**Lesson:** the "stale-looking" K-means regime label is part of how the system
makes money — block-trained q_entry has learned to handle it. Don't flip to
trailing without rebuilding the entire training pipeline (and even that lost
money on the 2024-12-12+ holdout). The actual cost of the user-reported losing
trade was the broker stop being hunted, not a regime mislabel.

**Live chart "repainting" (cosmetic only):** the chart endpoint
(`/admin/regime/_regime-chart`) recomputes historical labels each load,
which can shift past blocks if the live bar grid changes (Dukascopy
backfills). Anchoring via `funnel_log.persist_regime_for_bar` works
correctly but the SQLite DB on Render's default storage is wiped on
deploy → anchors lost → repainting visible again. Fix is one of:
mount a Render persistent disk, switch to Postgres, or fix the root
cause in `_extrapolated_boundaries` (none deployed yet — purely cosmetic,
production trading is unaffected).

### v90 24h-Momentum-Aware q_entry (Oracle XAU + BTC)
Deployed 2026-05-12 (commit `eec93e8`). Adds 2 direction-signed
24h-return features to q_entry:

- `ret_24h_signed` = direction × (close - close[t-288]) / close[t-288]
  (positive = trade direction matches recent 24h trend)
- `ret_24h_abs` = |close - close[t-288]| / close[t-288]
  (regime-stress magnitude proxy)

This addresses a user-reported issue where HighVol regime fired shorts
during periods of strong 24h-up movement (the weekly relabel threshold
wasn't crossed but intraday direction was clear). With `ret_24h_signed`,
the model learns to reject counter-trend trades within HighVol.

Validated on post-2024-12-12 holdout (q-only filter at q>3.0):
- Oracle XAU: PF 4.42 → 4.31, +444R (n grows 2,178 → 2,452, +0.5pp WR)
- Oracle BTC: PF 3.92 → **4.45** (+0.53), **+1,610R** (+13.3%), +1.4pp WR

Per-cluster wins at q>3.0:
- XAU Uptrend LONG: PF 4.41 → 5.04, +518R
- BTC Uptrend LONG: PF 3.92 → 4.78, +1,140R
- BTC TrendRange SHORT: PF 7.48 → 10.04, +149R

`ret_24h_signed` ranks #5 in HighVol q_entry feature importance on
both products — genuine signal, not noise. q_entry input dim grows
21 → 23. Confirm + meta heads unchanged.

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
| **Janus + Midas removal** | (commit `4e0ee0e` + `2ac70b5` + `180a0e5`) | 2026-05-11: products retired entirely from server/products/website |
| **v90 24h-Momentum q_entry** | `experiments/v90_fewer_clusters/03_24h_aware_q_entry.py` | ✅ BTC PF 3.92→4.45 (+13.5%) / +1,610R at q>3, XAU +444R, DEPLOYED 2026-05-12 |
| v90 K-cluster sweep (disprove) | `experiments/v90_fewer_clusters/02_k_sweep.py` | ❌ K=5 is optimal; K=6-10 marginal-or-worse, K=2-3 catastrophic |
| v92 supervised regime (disprove) | `experiments/v92_supervised_regime/02_train_classifier.py` | ❌ XAU "Up" conf>0.30 → mean fwd ret -0.20%; BTC random — forward direction unpredictable |
| v95 trailing K-means A/B | `commercial/experiments/v95_regime_repaint_ab/run_ab_fast.py` | (info) XAU 40% / BTC 65% bars disagree with block-aligned label |
| v91 trailing trade replay (disprove) | `experiments/v91_smart_regime/04_trailing_trade_replay.py` | ❌ XAU -401R / BTC -1,030R at q>3 — q_entry trained on block-aligned cids |
| v91 full pipeline retrain (disprove) | `experiments/v91_smart_regime/05_full_pipeline_trailing.py` | ❌ XAU still -185R after retraining q_entry on trailing labels |
| v91 momentum-disagreement veto (disprove) | `experiments/v91_smart_regime/06_momentum_veto.py` | ❌ XAU -1,569R / BTC -2,750R — veto kills system's best trades |
| **v91 SL-hunt diagnostic** | `experiments/v91_smart_regime/07_sl_hunt_check.py` | ✅ 56% of 4-ATR stops recover 4R within 60 bars — found root cause |
| **v97 wider hard SL** | `experiments/v91_smart_regime/08_wider_sl_pnl.py` | ✅ XAU PF 4.31→5.04 / +750R, BTC PF 4.45→5.43 / +1,585R at q>3, DEPLOYED 2026-05-13 |
