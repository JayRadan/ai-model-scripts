# Edge Predictor — Production Model Catalog

> **Last updated:** 2026-06-30 (edge_pullback deploy + server cache full-refresh)
> **Strategy repo:** `JayRadan/ai-model-scripts` (this folder)
> **Production repo:** `JayRadan/edge_predictor` (auto-deploys to Render on push to `main`)
> **Website:** `edgepredictor.pro`
> **Decision API:** `https://edge-predictor.onrender.com/decide/{product_slug}`

**8 live products** across XAUUSD, BTCUSD, DJIUSD (US30) + 1 retired (Janus). The server
fetches its own canonical Dukascopy bars (`dukascopy_source.py`), computes features +
ML decisions, and the EA Connector executes. Each product is one slug; the server routes
by `engine_type` (oracle / hermes / atlas) and then by the bundle's `version` string.

---

## 🟢 Live deployment snapshot (source of truth: server `configs/` + bundle `version`)

| Product | Asset | TF | Slots | Engine / strategy (deployed now) | Bundle version |
|---|---|---|---|---|---|
| **oracle_xau** | XAUUSD | M5 | 6 | Oracle smart-pipeline: rule cascade + q_entry + RL reverse-setup exit + BE-lock | `v84-rl+v87-giveback+v1-smart-upside…` |
| **oracle_btc** | BTCUSD | M5 | 2 | Oracle smart-pipeline (v99b dynamic-exit relabel, min_q 3) + BE-lock | `v99b_dynamic_exit_relabel_minq3…` |
| **hermes_xau** | XAUUSD | M1 | 1 | TFK combined-Q (pullback OR counter), q≥1.0 — reverted to first-deploy 2026-06-25 | *(no version)* |
| **hermes_btc** | BTCUSD | M1 | 1 | TFK combined-Q, q≥2.5 | *(no version)* |
| **hermes_dji** | DJIUSD | M1 | 1 | 🆕 **edge_pullback** (pullback + XGB-expected-R, SL6/trail2) — deployed 2026-06-30 | `edge_pullback_v1_hermes_dji` |
| **atlas_xau** | XAUUSD | M1 | 1 | 🆕 **edge_pullback** (same as hermes_dji) — deployed 2026-06-30 | `edge_pullback_v1_atlas_xau` |
| **atlas_btc** | BTCUSD | M1 | 1 | Atlas strict-candle reversal (Kalman + TFK confluence), 8y deep retrain | *(no version)* |
| **atlas_dji** | DJIUSD | M1 | 1 | Atlas strict-candle reversal, q≥3.0 — the one historically robust edge | *(no version)* |
| ~~janus_xau~~ | XAUUSD | — | — | **RETIRED 2026-05-11** (not in server registry) | — |

> Bundle `version` drives routing: `edge_pullback*` → `edge_pullback.py`; `hermes_band_pullback*`
> → `hermes_band_pullback.py`; *(no version)* → the standard oracle/hermes/atlas decide path.

---

## Per-product summary

### Oracle XAU / Oracle BTC — M5, the flagship pair
Multi-rule cascade + maturity-aware `q_entry` (XGB on stretch/trend/orderflow features) + RL
reverse-setup exit + **breakeven-lock** (peak ≥5R → floor locks +2R, deployed 2026-06-25). XAU
6 slots, BTC 2 slots. Oracle BTC was redeployed on a deep retrain; **Oracle XAU's edge is the
most scrutinised** — multiple look-ahead corrections over the months (see memory). Smart-exit is
server-side (`smart_pipeline.py`), not a simple trail.

### Hermes XAU / Hermes BTC — M1, TFK combined-Q
TFK `committed_dir` regime + combined-Q entry (pullback ≤0.5 ATR **OR** counter ≥1.5 ATR), single
XGB Q. XAU uses orderflow; BTC mirrors it. **Both reverted to first-deploy on 2026-06-25** (the
2026-06-23 band-pullback experiment was rolled back). Exit: config SL 4×ATR / trail 3×ATR, 1 slot.
⚠️ Honest causal edge is weak (HTF look-ahead removed; see `hermes_xau` README).

### Hermes DJI / Atlas XAU — 🆕 edge_pullback (deployed 2026-06-30)
Both run the **same** new engine `decision_engine/edge_pullback.py`:
- **Entry:** `committed_dir != 0` AND `|close−tfk_line|/ATR ≤ 1.0` (with-trend pullback to the TFK
  line); XGBRegressor predicts gross R; take if `pred_R ≥ threshold` (0.559 dji / 0.573 xau).
- **Exit:** server-side **2×ATR trailing give-back** + max_hold 300; hard SL 6×ATR is EA-owned.
- **Sizing:** 1 slot, cooldown 5, ~11 trades/day.
- **8-yr walk-forward (causal, 1pt spread):** DJI +4811R, 12/12 windows, Sharpe ~3.0; XAU
  generalises (+3890R); DJI+XAU portfolio Sharpe 3.6, daily correlation 0.04.
- ⚠️ **Thin edge (+0.27R/trade), spread-sensitive (breakeven ~3pt), never forward-tested.** XAU is
  the weaker side (contradicts prior "XAU dead"; 0.15R spread assumption may flatter it).
- Rollback: `models/{hermes_dji,atlas_xau}_validated.pkl.bak_pre_edge_pullback_2026-06-30`.
- Detail: `hermes_dji/README.md`, `atlas_xau/README.md`. Research: `experiments/_hermes_retrain/`.

### Atlas BTC / Atlas DJI — M1 strict-candle reversal
2-bar strict reversal: TFK regime + M1 Kalman confluence + strong prev-bar body, enter on
follow-through. Exit SL 6×ATR / trail 2×ATR (server-side trail) / BE@0.5R, 1 slot. **Atlas DJI is
the historically robust edge** (WF 10/10 windows). Atlas BTC redeployed on an 8-year deep retrain
(2026-06-26) after the original short-window train overfit.

### Janus XAU — RETIRED
Pivot-score XAU model, killed 2026-05-11 (redundant with Oracle's stretch features). Not in the
server registry. Folder kept for reference only.

---

## ⚠️ Missing / stale / needs attention

- **Server cache (fixed 2026-06-30):** `dukascopy_source.get_bars` now does a **full fresh fetch +
  replace** each refresh (commit 95ebab1, env `DUKASCOPY_FULL_REFRESH`) so live decisions are
  reproducible offline. Watch for EA timeouts (bigger fetch); revert via the env var if needed.
- **hermes_dji / atlas_xau config vs bundle:** their `configs/*.py` still carry the OLD exit params
  (hermes_dji sl=4/trail=3; atlas_xau sl=6/trail=2) and `q_thr`, but the **edge_pullback bundle
  overrides all exit/threshold params** (sl_R=6, trail_R=2, threshold in payload). Config exit
  values are now **unused** for these two — harmless, but clean up the config docstrings when convenient.
- **edge_pullback never forward-tested** — deployed straight from backtest. The honest plan was
  demo-first; it went live by request. Monitor the funnel; a losing fortnight is normal variance.
- **Individual product READMEs lag** — several (`oracle_*`, `hermes_xau/btc`, `atlas_*`) still
  describe pre-revert / pre-2026-06-30 strategies in their bodies and quote **inflated PFs** (HTF
  look-ahead). This catalog table is the authoritative current state; treat per-product README
  *bodies* as historical until refreshed.
- **Thin script coverage:** `atlas_dji/scripts` (4) and `janus_xau/scripts` (0) — no `sim_today`
  monitor for atlas_dji; janus has none (retired).
- **No live `sim_edge` monitor script** for the new edge_pullback products yet — use
  `experiments/hermes_dji_edge_last2weeks.py` (parametrise the window) to monitor DJI/XAU forward.

---

## Bar sourcing (Dukascopy `SYMBOL_MAP`)

| symbol_base | Dukascopy instrument | dukascopy-python constant |
|---|---|---|
| `XAUUSD` | `XAU/USD` | `INSTRUMENT_FX_METALS_XAU_USD` |
| `BTCUSD` | `BTC/USD` | `INSTRUMENT_VCCY_BTC_USD` |
| `DJIUSD` | `E_D&J-Ind` | `INSTRUMENT_IDX_AMERICA_E_D_J_IND` |

Server fetches `TARGET_BARS=8700` per symbol; per-symbol coalesced; `_market_is_open` gates
refresh outside session hours (DJI 13:30–21:00 UTC; XAU/BTC ~24h).

## Deploy / rollback

- **Deploy:** push to `JayRadan/edge_predictor` `main` → Render auto-deploys (~3–5 min).
- **Rollback a product:** restore its `models/*_validated.pkl.bak_*`, commit, push. Version-routing
  falls back automatically.
- **Confirm live:** `/decide/_health` (server up) + `/decide/_log-dump?product=…&hours=1` with
  `x-admin-secret` (check `reason` strings reflect the expected engine).
