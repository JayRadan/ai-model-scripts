"""
RL ZIGZAG ENTRY/EXIT — incremental curriculum learning WITHOUT catastrophic forgetting.
DJI M1 (the documented robust edge; XAU M1 end-to-end RL proven breakeven 2026-06-29).

Agent: PPO (MlpPolicy 64x64), actions {flat, long, short} on every M1 bar close.
Observations (causal only): 11 market features + 3 position-state features.
Reward: mark-to-market R/bar - spread on entry + lambda * zigzag-leg alignment bonus.
  The zigzag (4*ATR reversal) is computed in HINDSIGHT but used ONLY in the reward
  during training (legitimate shaping signal) — the policy never observes it.

CURRICULUM (per phase): lambda 0.3 -> 0, spread 0 -> full (phase 1 only; full after).
INCREMENTAL PHASES: P1 2018-19, P2 2020-21, P3 2022-23, P4 2024.
ANTI-FORGETTING: rehearsal episode sampling — 70% current chunk / 30% uniform over
  ALL previous chunks + reduced LR (3e-4 -> 1e-4) in later phases.
VARIANTS: REH (the method) | NOREH (no rehearsal — forgetting control) | JOINT (upper
  baseline: uniform over all train data, same total steps).
EVAL: deterministic policy exported to numpy, numba rollout bar-by-bar (fast, exact).
  Forgetting matrix after every phase (eval on all past chunks) + final table incl.
  UNTOUCHED HOLDOUT 2025-01 -> end. Net at 1.5pt spread throughout eval.
"""
import sys, time, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import gymnasium as gym
from gymnasium import spaces
from numba import njit
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path("/home/jay/Desktop/new-model-zigzag")
sys.path.insert(0, str(ROOT / "products/hermes_xau"))
from tfk import compute_tfk
OUT = Path(__file__).parent
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

SPREAD_PT = 1.5; ZZ_MULT = 4.0; SHAPE = 0.05; EP_LEN = 1440
PHASE_STEPS = 400_000; N_ENVS = 16; SEED = 0
REHEARSAL = 0.30

# ---------------------------------------------------------------- features (causal)
def emanp(x, n):
    a = 2 / (n + 1); o = np.empty_like(x); o[0] = x[0]
    for i in range(1, len(x)): o[i] = a * x[i] + (1 - a) * o[i - 1]
    return o

@njit(cache=True)
def eff_ratio(c, n):
    m = len(c); er = np.ones(m)
    for i in range(n, m):
        net = abs(c[i] - c[i - n]); s = 0.0
        for j in range(i - n + 1, i + 1): s += abs(c[j] - c[j - 1])
        er[i] = net / s if s > 0 else 1.0
    return er

@njit(cache=True)
def rsi_nb(c, n):
    m = len(c); o = np.full(m, 50.0); au = 0.0; ad = 0.0
    for i in range(1, m):
        ch = c[i] - c[i - 1]; u = ch if ch > 0 else 0.0; dn = -ch if ch < 0 else 0.0
        if i <= n: au += u / n; ad += dn / n
        else: au = (au * (n - 1) + u) / n; ad = (ad * (n - 1) + dn) / n
        o[i] = 100 - 100 / (1 + au / ad) if ad > 0 else 100.0
    return o

@njit(cache=True)
def zigzag_dir(H, L, atr, mult):
    """hindsight zigzag leg direction per bar (+1 up-leg / -1 down-leg). REWARD ONLY."""
    n = len(H); zdir = np.zeros(n, np.int8)
    up = True; ext = H[0]; exti = 0; lastp = 0
    for i in range(1, n):
        a = atr[i]; thr = mult * a if a > 0 else 1e18
        if up:
            if H[i] >= ext: ext = H[i]; exti = i
            if ext - L[i] >= thr:
                for j in range(lastp, exti + 1): zdir[j] = 1
                lastp = exti + 1; up = False; ext = L[i]; exti = i
        else:
            if L[i] <= ext: ext = L[i]; exti = i
            if H[i] - ext >= thr:
                for j in range(lastp, exti + 1): zdir[j] = -1
                lastp = exti + 1; up = True; ext = H[i]; exti = i
    for j in range(lastp, n): zdir[j] = 1 if up else -1
    return zdir

log("loading DJI M1...")
m1 = pd.read_parquet(ROOT / "data/m1_dji_full.parquet")
m1 = m1.rename(columns={[c for c in m1.columns if "time" in c.lower()][0]: "time"})
m1["time"] = pd.to_datetime(m1["time"]); m1 = m1.sort_values("time").drop_duplicates("time").reset_index(drop=True)
C = m1.close.to_numpy(float); H = m1.high.to_numpy(float); L = m1.low.to_numpy(float)
n = len(C); times = m1["time"]
pc = np.concatenate([[C[0]], C[:-1]])
trr = np.maximum(H - L, np.maximum(np.abs(H - pc), np.abs(L - pc)))
atr = pd.Series(trr).rolling(14, min_periods=14).mean().to_numpy()
ema50 = emanp(C, 50); ema200 = emanp(C, 200); er = eff_ratio(C, 240); rs = rsi_nb(C, 14)
arat = atr / pd.Series(atr).rolling(100, min_periods=20).mean().to_numpy()
def rh(h):
    r = np.zeros(n); r[h:] = (C[h:] - C[:-h]) / np.maximum(atr[h:], 1e-9); return r
r5, r15, r60 = rh(5), rh(15), rh(60)
s = m1.set_index("time")
m30 = pd.DataFrame({"open": s.open.resample("30min").first(), "high": s.high.resample("30min").max(),
                    "low": s.low.resample("30min").min(), "close": s.close.resample("30min").last(),
                    "tick_volume": (s.close * 0 + 1).resample("30min").sum()}).dropna(subset=["close"]).reset_index()
m30["d"] = compute_tfk(m30, flip_bars=5, color_confirm=8)["committed_dir"].to_numpy()
tfkd = pd.merge_asof(m1[["time"]], m30[["time", "d"]].assign(time=m30["time"] + pd.Timedelta("30min")),
                     on="time")["d"].fillna(0).to_numpy()
tod = times.dt.hour * 60 + times.dt.minute
tod_sin = np.sin(2 * np.pi * tod / 1440).to_numpy(); tod_cos = np.cos(2 * np.pi * tod / 1440).to_numpy()
cl = np.clip
F = np.nan_to_num(np.stack([
    cl((C - ema50) / np.maximum(atr, 1e-9), -6, 6), cl((C - ema200) / np.maximum(atr, 1e-9), -10, 10),
    er, rs / 100, cl(arat, 0, 3), cl(r5, -6, 6), cl(r15, -6, 6), cl(r60, -6, 6),
    tfkd, tod_sin, tod_cos], 1).astype(np.float32))
RET1 = np.zeros(n, np.float32); RET1[:-1] = ((C[1:] - C[:-1]) / np.maximum(atr[:-1], 1e-9)).astype(np.float32)
RET1 = np.clip(RET1, -15, 15)
SPR = (SPREAD_PT / np.maximum(atr, 1e-9)).astype(np.float32)
ZDIR = zigzag_dir(H, L, atr, ZZ_MULT).astype(np.float32)
ATRv = np.maximum(atr, 1e-9).astype(np.float32)
valid = np.isfinite(atr) & (atr > 0); valid[:400] = False; valid[-(EP_LEN + 2):] = False
log(f"{n:,} bars, zigzag legs mean len {np.mean(np.diff(np.where(np.diff(ZDIR) != 0)[0])):.0f} bars")

CHUNKS = [("P1_2018-19", "2018-01-01", "2020-01-01"), ("P2_2020-21", "2020-01-01", "2022-01-01"),
          ("P3_2022-23", "2022-01-01", "2024-01-01"), ("P4_2024", "2024-01-01", "2025-01-01")]
HOLD = ("HOLDOUT_2025+", "2025-01-01", "2027-01-01")
def rng_idx(a, b):
    return np.searchsorted(times.values, np.datetime64(a)), np.searchsorted(times.values, np.datetime64(b))
CH_IX = {nm: rng_idx(a, b) for nm, a, b in CHUNKS}; HO_IX = rng_idx(HOLD[1], HOLD[2])
def starts_of(nm):
    cs, ce = CH_IX[nm]
    cand = np.arange(cs, ce - EP_LEN - 2, 5)
    return cand[valid[cand]]
CH_STARTS = {nm: starts_of(nm) for nm, _, _ in CHUNKS}

# ---------------------------------------------------------------- env
class ZZEnv(gym.Env):
    def __init__(self, seed=0):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-12, 12, (14,), np.float32)
        self.rng = np.random.RandomState(seed)
        self.cur_starts = CH_STARTS["P1_2018-19"]; self.prev_starts = None
        self.mix = 0.0; self.lam = 0.0; self.spread_on = 1.0
    def set_phase(self, cur, prev, mix): self.cur_starts, self.prev_starts, self.mix = cur, prev, mix
    def set_shape(self, lam, spread_on): self.lam, self.spread_on = lam, spread_on
    def _obs(self):
        o = np.empty(14, np.float32); o[:11] = F[self.t]
        o[11] = self.pos
        o[12] = 0.0 if self.pos == 0 else cl(self.pos * (C[self.t] - self.entry) / ATRv[self.t], -8, 8)
        o[13] = min(self.held / 300.0, 2.0)
        return o
    def reset(self, *, seed=None, options=None):
        arr = self.prev_starts if (self.prev_starts is not None and len(self.prev_starts)
                                   and self.rng.rand() < self.mix) else self.cur_starts
        self.t = int(arr[self.rng.randint(len(arr))]); self.end = self.t + EP_LEN
        self.pos = 0; self.entry = 0.0; self.held = 0
        return self._obs(), {}
    def step(self, a):
        tgt = 0 if a == 0 else (1 if a == 1 else -1)
        r = 0.0
        if tgt != self.pos:
            if tgt != 0: r -= SPR[self.t] * self.spread_on
            self.pos = tgt; self.entry = C[self.t]; self.held = 0
        if self.pos != 0:
            r += self.pos * RET1[self.t]
            r += self.lam * SHAPE * self.pos * ZDIR[self.t]
            self.held += 1
        self.t += 1
        return self._obs(), float(r), self.t >= self.end, False, {}

# ---------------------------------------------------------------- fast deterministic eval
@njit(cache=True)
def rollout(W1, b1, W2, b2, W3, b3, s, e, F, C, RET1, SPR, ATRv, valid):
    pos = 0; entry = 0.0; held = 0.0; cur = 0.0
    ntr = 0; pfw = 0.0; pfl = 0.0; sumR = 0.0
    eq = np.zeros(e - s, np.float32)
    obs = np.empty(14, np.float64)
    for t in range(s, e):
        if not valid[t]:
            eq[t - s] = sumR; continue
        for j in range(11): obs[j] = F[t, j]
        obs[11] = pos
        obs[12] = 0.0 if pos == 0 else min(max(pos * (C[t] - entry) / ATRv[t], -8.0), 8.0)
        obs[13] = min(held / 300.0, 2.0)
        h1 = np.tanh(W1 @ obs + b1); h2 = np.tanh(W2 @ h1 + b2); lg = W3 @ h2 + b3
        a = 0
        if lg[1] > lg[0] and lg[1] >= lg[2]: a = 1
        elif lg[2] > lg[0] and lg[2] > lg[1]: a = 2
        tgt = 0 if a == 0 else (1 if a == 1 else -1)
        if tgt != pos:
            if pos != 0:
                ntr += 1
                if cur > 0: pfw += cur
                else: pfl -= cur
            cur = 0.0
            if tgt != 0: cur -= SPR[t]; sumR -= SPR[t]; entry = C[t]; held = 0.0
            pos = tgt
        if pos != 0:
            cur += pos * RET1[t]; sumR += pos * RET1[t]; held += 1
        eq[t - s] = sumR
    if pos != 0:
        ntr += 1
        if cur > 0: pfw += cur
        else: pfl -= cur
    return sumR, ntr, pfw, pfl, eq

def eval_policy(model, s, e):
    p = model.policy
    pn = p.mlp_extractor.policy_net
    W1 = pn[0].weight.detach().numpy().astype(np.float64); b1 = pn[0].bias.detach().numpy().astype(np.float64)
    W2 = pn[2].weight.detach().numpy().astype(np.float64); b2 = pn[2].bias.detach().numpy().astype(np.float64)
    W3 = p.action_net.weight.detach().numpy().astype(np.float64); b3 = p.action_net.bias.detach().numpy().astype(np.float64)
    sumR, ntr, pfw, pfl, eq = rollout(W1, b1, W2, b2, W3, b3, s, e,
                                      F.astype(np.float64), C, RET1.astype(np.float64),
                                      SPR.astype(np.float64), ATRv.astype(np.float64), valid)
    days = max((e - s) / 1380, 1)
    return dict(sumR=float(sumR), n=int(ntr), perday=ntr / days,
                pf=float(pfw / max(pfl, 1e-9)), eq=eq)

# ---------------------------------------------------------------- training
def make_env(i): return lambda: ZZEnv(seed=SEED * 100 + i)

def train_variant(name, rehearsal, joint=False):
    log(f"=== VARIANT {name} ===")
    venv = DummyVecEnv([make_env(i) for i in range(N_ENVS)])
    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=256, batch_size=1024,
                gamma=0.997, gae_lambda=0.95, ent_coef=0.01, seed=SEED, verbose=0,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])))
    fmat = {}
    phases = [("JOINT_ALL", None)] if joint else [(nm, i) for i, (nm, _, _) in enumerate(CHUNKS)]
    for pi_, (pname, ci) in enumerate(phases):
        if joint:
            cur = np.concatenate([CH_STARTS[nm] for nm, _, _ in CHUNKS]); prev = None; mix = 0.0
            steps = PHASE_STEPS * len(CHUNKS)
        else:
            cur = CH_STARTS[pname]
            prevs = [CH_STARTS[nm] for nm, _, _ in CHUNKS[:ci]]
            prev = np.concatenate(prevs) if (rehearsal and prevs) else None
            mix = REHEARSAL if prev is not None else 0.0
            steps = PHASE_STEPS
        lr = 3e-4 if pi_ == 0 else 1e-4
        model.learning_rate = lr; model._setup_lr_schedule()
        for env in venv.envs: env.set_phase(cur, prev, mix)
        first_phase = pi_ == 0
        # curriculum inside the phase: shaped+cheap -> pure net PnL
        for env in venv.envs: env.set_shape(0.3, 0.0 if first_phase else 1.0)
        model.learn(total_timesteps=int(steps * 0.4), reset_num_timesteps=False, progress_bar=False)
        for env in venv.envs: env.set_shape(0.0, 1.0)
        model.learn(total_timesteps=int(steps * 0.6), reset_num_timesteps=False, progress_bar=False)
        # forgetting matrix row: eval on ALL chunks seen so far
        row = {}
        upto = len(CHUNKS) if joint else ci + 1
        for nm, _, _ in CHUNKS[:upto]:
            cs, ce = CH_IX[nm]; r = eval_policy(model, cs, ce)
            row[nm] = (r["pf"], r["sumR"], r["perday"])
        fmat[pname] = row
        log(f"  phase {pname} done | " + " | ".join(f"{k}: PF {v[0]:.2f} {v[1]:+.0f}R" for k, v in row.items()))
    res = {}
    for nm, _, _ in CHUNKS:
        cs, ce = CH_IX[nm]; res[nm] = eval_policy(model, cs, ce)
    res[HOLD[0]] = eval_policy(model, *HO_IX)
    return model, fmat, res

RES = {}; FM = {}
for name, reh, joint in [("REH", True, False), ("NOREH", False, False), ("JOINT", False, True)]:
    _, fmat, res = train_variant(name, reh, joint)
    RES[name] = res; FM[name] = fmat

# ---------------------------------------------------------------- report
print(f"\n{'='*90}\nFORGETTING MATRIX (PF on each chunk after each phase) — REH vs NOREH\n{'='*90}")
for vn in ("REH", "NOREH"):
    print(f"\n[{vn}]")
    for pname, row in FM[vn].items():
        print(f"  after {pname:<12} " + "  ".join(f"{k}: {v[0]:5.2f}" for k, v in row.items()))

print(f"\n{'='*90}\nFINAL EVAL (net @ {SPREAD_PT}pt spread) — all chunks + UNTOUCHED HOLDOUT\n{'='*90}")
print(f"{'chunk':<16}" + "".join(f"{vn:>26}" for vn in RES))
for nm in list(CH_IX) + [HOLD[0]]:
    line = f"{nm:<16}"
    for vn in RES:
        r = RES[vn][nm]
        line += f"  PF {r['pf']:5.2f} {r['sumR']:>+8.0f}R {r['perday']:4.1f}/d"
    print(line)

fig, ax = plt.subplots(figsize=(13, 5))
cs, ce = HO_IX
for vn, col in zip(RES, ["#16a34a", "#dc2626", "#2563eb"]):
    eq = RES[vn][HOLD[0]]["eq"]
    ax.plot(times.iloc[cs:ce].values, eq, lw=1.1, color=col,
            label=f"{vn} ({RES[vn][HOLD[0]]['sumR']:+.0f}R, PF {RES[vn][HOLD[0]]['pf']:.2f})")
ax.set_title(f"RL zigzag-curriculum DJI M1 — HOLDOUT 2025+ equity, net @ {SPREAD_PT}pt")
ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "rl_zigzag_holdout.png", dpi=110)
json.dump({vn: {k: {kk: vv for kk, vv in r.items() if kk != "eq"} for k, r in RES[vn].items()} for vn in RES},
          open(OUT / "rl_zigzag_results.json", "w"), indent=1)
log(f"done -> {OUT/'rl_zigzag_holdout.png'}")
