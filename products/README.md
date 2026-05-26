# Edge Predictor — Production Model Catalog

> **Last updated:** 2026-05-26
> **Strategy repo:** `JayRadan/ai-model-scripts` (this folder)
> **Production repo:** `JayRadan/edge_predictor` (auto-deploys to Render on push to `main`)
> **Website:** `edgepredictor.pro` (Vercel deploy from the same production repo)

Four live production trading models for XAUUSD and BTCUSD. All trained on Dukascopy data,
validated on out-of-sample holdouts, deployed server-side. EA Connector reads decisions
from `https://edge-predictor.onrender.com/decide/{product_slug}`.

---

## Quick Reference (current state)

| Product | Asset | TF | Architecture | Holdout PF | WR | Bundle | Slug |
|---|---|---|---|---|---|---|---|
| **Oracle XAU** | XAUUSD | M5 | v99b q_entry + v88 reverse exit + 6×ATR SL | **5.85** @ Q≥2.0 | 87.9% | `oracle_xau_validated.pkl` | `oracle_xau` |
| **Oracle BTC** | BTCUSD | M5 | v99b q_entry + v88 reverse exit + 6×ATR SL | **5.60** @ Q≥2.0 | 88.1% | `oracle_btc_validated.pkl` | `oracle_btc` |
| **Hermes XAU** | XAUUSD | M1 | TFK + 14 orderflow + combined-Q | **3.21** @ Q≥4.0 | 70.7% | `hermes_xau_validated.pkl` | `hermes_xau` |
| **Hermes BTC** | BTCUSD | M1 | TFK + 14 orderflow + Q | **3.87** @ Q≥2.5 | 74.9% | `hermes_btc_validated.pkl` | `hermes_btc` |

> **Retired:** Janus XAU and Midas XAU were retired 2026-05-11 — customers notified. Folders kept here for archaeology only.

### Trading style summary

- **Oracle** (XAU + BTC) — M5, slower, multi-slot Oracle XAU=10 slots / BTC=6, two-stage RL+confirm+meta gate, ~5 trades/day per product.
- **Hermes** (XAU + BTC) — M1, faster, 4-slot multi-position with switch + BE-on-new-entry, single XGBRegressor Q + TFK regime, ~26-34 trades/day.

---

## Folder Layout

```
products/
├── README.md                            ← THIS FILE (catalog index)
├── _shared/                             ← Cross-product infrastructure
│   ├── regime_selector_xau.json         (Oracle XAU K=5 K-means selector)
│   ├── regime_selector_btc.json         (Oracle BTC selector)
│   ├── regime_block_boundaries_*.json   (v83c 4h-step boundaries)
│   └── scripts/                         shared regime selector builder
│
├── oracle_xau/                          ← Oracle XAU model
│   ├── README.md                        full architecture + v99b deploy notes
│   ├── train_rl_entry.py                training entry point
│   ├── deploy_bundle.py                 pickle → server bundle
│   ├── deploy_v83c.json                 frozen hyperparams
│   └── scripts/
│       ├── 01_validate_v72l.py
│       ├── 02_train_export.py
│       ├── 03_train_rl_entry.py
│       ├── 04_full_rl_exit.py
│       └── 05_deploy_bundle.py
│
├── oracle_btc/                          ← Oracle BTC model (twin of XAU)
│   ├── README.md
│   ├── train_rl_entry.py
│   ├── deploy_btc_v83c.json
│   └── scripts/
│       ├── 01_validate_v72l.py
│       ├── 02_train_export.py
│       ├── 02b_build_selector.py        BTC-specific regime selector
│       ├── 03_v83c_pipeline.py
│       └── 04_train_rl_entry.py
│
├── hermes_xau/                          ← Hermes XAU (M1 TFK + orderflow)
│   ├── README.md                        v103 lineage + combined-Q upgrade
│   ├── tfk.py                           TFK indicator (Pine v6 port)
│   ├── train_bundle.py                  original v103 pullback-only trainer
│   ├── train_bundle_combined.py         ★ combined-Q (NEAR OR counter)
│   └── scripts/                         9-script reproduction recipe
│       ├── 01_download_m1_dukascopy.py
│       ├── 02_aggregate_orderflow_from_ticks.py
│       ├── 03_train_q_baseline_m1.py
│       ├── 04_train_q_with_orderflow.py
│       ├── 05_sweep_sl_threshold.py
│       ├── 06_sweep_near_threshold.py
│       ├── 07_meta_classifier_disproven.py
│       ├── 08_visualize_holdout.py
│       └── 09_build_website_backtest_json.py
│
├── hermes_btc/                          ← Hermes BTC (M1 TFK + orderflow, BTC)
│   ├── README.md
│   ├── tfk.py                           (mirror of hermes_xau/tfk.py)
│   ├── train_bundle.py                  BTC bundle trainer
│   ├── 01_download.log / 02_orderflow.log / train.log   ← timing records
│   └── scripts/                         9-script reproduction recipe (mirror)
│
└── janus_xau/                           ← RETIRED 2026-05-11
    └── models/                          (kept for historical reference)
```

**Note on `models/`** — production `.pkl` bundles live in the commercial repo at
`commercial/server/decision_engine/models/`, not here. This strategy repo only
contains the **training scripts** and **research artefacts**. The pickle files
in `products/models/` (if any) are local-only research copies.

---

## How each product is trained — at a glance

### Oracle XAU + Oracle BTC (same architecture, different bundles)

**Stack (most recent to oldest patch):**
1. **v99b q_entry** (2026-05-17) — 5 XGBRegressor Q-heads, one per K-means regime cluster. Labels: `SL=2R, TP_min=4R + 2R trailing, max 200 bars`. Min Q raised 0.3 → 3.0.
2. **v97 wider SL** (2026-05-13) — hard SL widened 4×ATR → 6×ATR. No retrain, pure exit config. Diagnostic: 56% of 4-ATR stops recovered within 60 bars.
3. **v90 24h-momentum features** (2026-05-12) — added `ret_24h_signed`, `ret_24h_abs` to q_entry input (21→23 features).
4. **v89 maturity-aware features** — added `stretch_100`, `stretch_200`, `pct_to_extreme_50` (18→21).
5. **v88 reverse-setup RL exit** (2026-05-08) — at every in-trade bar, scans 30 rule detectors for *opposite-direction* setups; if `q_entry > 0.10` on that setup, exits.
6. **v84 RL entry** — replaces the 28-rule hand-coded catalog with 5 Q-heads.
7. **v83c regime selector** — K=5 K-means on 8 fingerprint features, refreshed every 4h step (288-bar window).

**Live ops tooling (2026-05-17):**
- **Stack-gate** — server blocks new same-direction slot if any existing slot has floating R < 0
- **Admin regime override** — `/admin/regime` UI lets you pin a regime when classifier lags
- **v85 drawdown circuit breaker** — pause all new entries on a product if PnL drops >25% from session peak

### Hermes XAU (M1 high-frequency)

**Stack:**
1. **Combined-Q** (2026-05-26) — Q model retrained on union of candidate types:
   - **Pullback** bars: `|close − tfk_line| / atr ≤ 0.50`
   - **Counter** bars: `(dist_signed × committed_dir) ≤ -1.5` (price popped through to wrong side by ≥1.5 ATR)
   - Trade direction = `committed_dir` (always with-trend)
   - `q_thr` raised 1.0 → 4.0 to land at the holdout sweet spot
2. **v103 base** (2026-05-12) — TFK indicator + 43 features (29 std + 14 orderflow). Single XGBRegressor Q. Pullback-only entries at `q_thr=1.0`.

**Multi-position management:**
- 4 concurrent slots, magic base `421100..421103`
- `switch_delta=0.5` — new Q must beat worst active slot's Q by 0.5 to switch
- `cooldown_bars=5` — minimum 5 bars between same-direction entries
- **BE-on-new-entry** — when a new entry opens, slots already at favor ≥ +1R get SL moved to entry
- Hard SL = 4×ATR (broker-enforced), trail give-back = 3×ATR from peak

### Hermes BTC (M1 high-frequency, BTC twin)

Mirror of Hermes XAU architecture, BTC-specific bundle. Same TFK params, 43 features, multi-pos rules. Two config differences:
- `q_thr = 2.5` (XAU=4.0 after combined-Q; BTC kept on the **simpler pullback-only spec at q=2.5**)
- `spread_usd = 5.00` (XAU=0.30)
- Magic base `421150..421153`

> Note: Hermes BTC is NOT yet upgraded to combined-Q. The XAU result (counter trades have edge after retrain) should replicate on BTC but hasn't been validated yet. See the **TODO** section below.

---

## How each product was tested

All four products use **walk-forward holdout** validation — train on early data, test on
held-out tail. Holdout windows:

| Product | Train cutoff | Holdout window | Holdout days |
|---|---|---|---|
| Oracle XAU | 2024-12-12 | 2024-12-12 → 2026-05 | ~150 days |
| Oracle BTC | 2024-12-12 | 2024-12-12 → 2026-05 | ~150 days |
| Hermes XAU | 2025-09-01 | 2025-09-01 → 2026-05 | 242 days |
| Hermes BTC | 2026-01-15 | 2026-01-15 → 2026-05 | 107 days |

Hermes BTC's holdout is shorter only because the on-disk tick data starts 2024-12.
Plenty for stable stats (>1000 holdout trades at the chosen Q threshold).

**Metrics reported per product** (in each `README.md`):
- Profit factor (PF)
- Win rate (WR)
- Sum R (total R-multiples)
- Max drawdown in R-units AND in USD at the deployed lot size
- Trades per day
- Per-cluster / per-setup-type breakdown
- A multi-position simulation (not just per-candidate) where applicable

---

## Deployment Architecture

```
MT5 EA (EdgePredictor_Connector_v2.mq5)
                    │
                    │  M{1|5} bar close → POST /decide/{slug}
                    │  body: bars[] + position + open_positions[] + account + license
                    ▼
Render — FastAPI decision_engine
  ├─ loader.py        loads {product}_validated.pkl per slug
  ├─ api.py           dispatch:
  │                     engine_type == "hermes" → decide_hermes.py
  │                     else                     → decide.py (Oracle pipeline)
  ├─ decide.py        Oracle: RL Q-entry + confirm + meta + v88 exit
  ├─ decide_hermes.py Hermes: TFK gate + Q-regressor + multi-pos rules
  ├─ tick_source.py   live Dukascopy ticks → orderflow features (Hermes only)
  ├─ stack_gate       blocks pyramiding into losing slots
  └─ funnel_log.py    SQLite log of every decision (admin funnel UI reads this)
                    │
                    │  JSON response: {action, direction, sl_atr_mult, ...}
                    ▼
EA executes (open / hold / exit, with broker-side hard SL + trail)
```

**Per-product dispatch** is automatic via `engine_type` on the config:
- Oracle XAU/BTC → `engine_type="oracle"` → routes through `decide.py`
- Hermes XAU/BTC → `engine_type="hermes"` → routes through `decide_hermes.py`

---

## Live Operations Cheat-sheet

```bash
# 1. Inspect live regime state for any product
curl https://edge-predictor.onrender.com/decide/{slug}/_regime-debug \
  -H "X-Admin-Secret: $ADMIN_SECRET" | jq

# 2. Pull last 24h funnel
admin → Decision Funnel → product=<slug> → 24h → Download CSV

# 3. Pin a regime (when classifier lags)
admin → /admin/regime → product=<slug> → cid=<n> → Set

# 4. Watch live admin chart (Oracle only — Hermes uses TFK direction)
admin → /admin/regime → product=<oracle_xau|oracle_btc> → 168h

# 5. Rollback a deploy
cd /home/jay/Desktop/my-agents-and-website/commercial
git revert <commit>
# If pickle was changed: restore .bak file
cd server/decision_engine/models
mv {product}_validated.pkl.bak_<tag> {product}_validated.pkl
git add -A && git commit && git push  # Render redeploys
```

---

## Reproduce a product from scratch

### Oracle XAU / Oracle BTC

```bash
# 1. Build regime selector
python3 experiments/v83_range_position_filter/01_build_selector_4h.py
# → data/regime_selector_4h.json + regime_fingerprints_4h.csv

# 2. Train RL entry
python3 products/oracle_xau/train_rl_entry.py
python3 products/oracle_btc/train_rl_entry.py

# 3. Deploy
cp products/models/{oracle_xau,oracle_btc}_validated.pkl \
   /home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine/models/
cd /home/jay/Desktop/my-agents-and-website/commercial
git add server/ && git commit -m "deploy oracle bundle" && git push
```

### Hermes XAU / Hermes BTC

```bash
# Pre-req for both: 8y M1 + 14-month tick parquets in data/

# Hermes XAU — original pullback-only spec:
python3 products/hermes_xau/train_bundle.py
# OR — combined-Q (PRODUCTION as of 2026-05-26):
python3 products/hermes_xau/train_bundle_combined.py

# Hermes BTC:
python3 products/hermes_btc/train_bundle.py

# Bundles auto-write to commercial/server/decision_engine/models/.
# Then: git commit + push.
```

Each product folder's `README.md` has the step-by-step recipe (data download,
features, sweep, train, deploy). All numbered scripts in `scripts/` are
self-contained and runnable individually.

---

## Why the architectures differ

| | Oracle | Hermes |
|---|---|---|
| Bar | M5 | M1 |
| Regime | K=5 K-means on 8 features | TFK indicator (committed_dir ±1) |
| Entry signal | 5 RL Q-heads per regime + confirm head + meta gate | Single XGBRegressor Q-regressor + TFK distance gate |
| Features | 21-23 inputs to q_entry | 43 (29 std + 14 orderflow) |
| Position mgmt | 6-10 slots, slow stacking | 4 slots, fast switch+BE rule |
| Trades/day | ~5 | ~26-34 |
| Best for | trending sessions, slower swings | intraday, pullback/counter patterns |

They're **complementary** — designed to run together on different charts under one license.
The Apex bundle (Oracle XAU + Oracle BTC) is sold for cross-asset diversification; future
"Hermes bundle" could mirror this for the M1 products.

---

## Historical experiment log (active references)

Production-relevant only. Older entries kept for context.

| Experiment | Result | Status |
|---|---|---|
| v83c range filter + kill-switch | +0.5–2.7 PF across products | active |
| v84 RL Entry (XAU + BTC) | PF 4.21 / 3.82 | superseded by v89/v90/v99b |
| **v88 reverse-setup RL exit** | XAU PF +0.36, BTC PF +0.99 | **active** |
| v87 multi-head exit | -866R XAU / -674R BTC unseen | REMOVED |
| v89 maturity-aware q_entry | XAU PF 4.60→6.44 | superseded by v99b |
| **v90 24h-momentum q_entry** | BTC PF +0.53, +13.3% R | **active** |
| v92 supervised regime classifier | XAU "Up" conf>0.30 → fwd ret -0.20% | DISPROVEN |
| v95/v96 4h regime forecast | All variants worse than persistence | DISPROVEN (5×) |
| v97 forecast unblock | +5.35pp BTC precision lift on C1/C2 | OPT-IN flag (off default) |
| **v97 wider hard SL (4→6 ATR)** | XAU +750R, BTC +1,585R at q>3 | **active** |
| **v99b dynamic-exit relabel** | XAU PF 5.04→5.85, BTC PF 5.43→5.60 | **active** |
| **v103 TFK Hermes XAU** | PF 2.09 holdout | superseded by combined-Q |
| **Hermes XAU combined-Q** | PF 2.09 → **3.21** @ Q≥4.0 | **active (2026-05-26)** |
| **Hermes BTC** | PF 3.87 @ Q≥2.5 holdout | **active (2026-05-25)** |

Full disprove catalogs live in each `experiments/v{N}_*/README.md`. Memory of which avenues
have been blocked off is the most valuable artifact this repo has — re-litigating disproven
ideas is the most expensive way to burn research time.

---

## Open TODOs (research direction)

1. **Hermes BTC → combined-Q port.** The counter-pullback hypothesis was validated on XAU
   (PF 2.09 → 3.21). It should replicate on BTC but hasn't been tested. ~30 min of work:
   adapt `train_bundle_combined.py` for BTC, run sweep, decide q_thr.
2. **Hermes weekend gap risk.** Both Hermes products use brokers that quote BTC over the
   weekend, but Dukascopy doesn't. Positions held over Sat/Sun could blow past 4×ATR SL
   on Monday open. Worth a "close on Friday" rule for slots held > N bars approaching weekend.
3. **Hermes bundle pricing.** Oracle has Apex (XAU+BTC for $149). Hermes should have a
   similar bundle ($149?). Just a website + entitlements change.
4. **Per-cluster Q for Hermes.** Combined-Q learned both pullback and counter equally. Could
   we also segment by `regime_age` buckets (fresh trend vs mature trend) for further uplift?
5. **Postgres or persistent disk on Render.** SQLite funnel_log gets wiped on deploy → live
   regime anchors lost. Pure infra fix.
