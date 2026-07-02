"""
v3 — SETUP-GATED RL EXIT (the v88 lesson: rule-gating is where RL worked).
DJI M1. The agent does NOT free-run: episodes are gate-approved edge_pullback
trades (|dist_tfk|<=1.0, dir=committed_dir, XGB pred_R gate ~11/day). It manages
each trade bar-by-bar: {hold, exit}; hard SL 7*ATR + maxhold 300 enforced.

Training method = everything validated in v1/v2:
  - incremental phases P1 2018-19 .. P4 2024, rehearsal 70/30 (no forgetting)
  - curriculum: zigzag-leg alignment shaping (lam 0.3 -> 0, reward-only)
  - 2x spread during training (selectivity margin), real spread at eval
  - P5 recency fine-tune (P3+P4, LR 5e-5)
  - 2 seeds (0, 1)

EVAL (honest): gate model + threshold fitted on TRAIN period only (2018-2024);
holdout 2025+ candidates gated with the same frozen model. RL exit vs deployed
tt30/0.75 vs SL7/T2 on the SAME candidates, 1-slot cooldown5 portfolio,
net @ 1.0/1.5/2.0 pt.
"""
import sys, time, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import gymnasium as gym
from gymnasium import spaces
from numba import njit
from xgboost import XGBRegressor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path("/home/jay/Desktop/new-model-zigzag")
SRV = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(SRV)); sys.path.insert(0, str(ROOT / "products/hermes_xau"))
import decision_engine.edge_pullback as ep
import pickle
OUT = Path(__file__).parent
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

SPREAD_PT = 1.5; SL = 7.0; MAXH = 300; ZZ_MULT = 4.0; SHAPE = 0.05
PHASE_STEPS = 1_200_000; N_ENVS = 16; REHEARSAL = 0.30; SEEDS = [0, 1]
TARGET_PER_DAY = 11.0; COOLDOWN = 5
FC = pickle.load(open(SRV / "decision_engine/models/hermes_dji_validated.pkl", "rb"))["feat_cols"]

# ---------------------------------------------------------------- data + features
df = pd.read_parquet(ROOT / "data/m1_dji_full.parquet")
df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
feat = ep.compute_edge_features(df); log("edge features done")
atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
da = np.abs(feat["dist_at_signal"].to_numpy(float))
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"]; n = len(df)
XFULL = np.nan_to_num(feat[FC].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

# compact 11-dim RL observation features (same family as v1/v2)
def emanp(x, m_):
    a = 2 / (m_ + 1); o = np.empty_like(x); o[0] = x[0]
    for i in range(1, len(x)): o[i] = a * x[i] + (1 - a) * o[i - 1]
    return o
ema50 = emanp(C, 50); ema200 = emanp(C, 200)
rs = feat["rsi14"].to_numpy(float)
arat = np.nan_to_num(feat["atr_ratio"].to_numpy(float), nan=1.0)
def rh(h):
    r = np.zeros(n); r[h:] = (C[h:] - C[:-h]) / np.maximum(atr[h:], 1e-9); return r
r5, r15, r60 = rh(5), rh(15), rh(60)
tod = times.dt.hour * 60 + times.dt.minute
cl = np.clip
F = np.nan_to_num(np.stack([
    cl((C - ema50) / np.maximum(atr, 1e-9), -6, 6), cl((C - ema200) / np.maximum(atr, 1e-9), -10, 10),
    cl(np.nan_to_num(rs, nan=50.0) / 100, 0, 1), cl(arat, 0, 3),
    cl(r5, -6, 6), cl(r15, -6, 6), cl(r60, -6, 6), cdir.astype(float),
    cl(np.nan_to_num(feat["dist_at_signal"].to_numpy(float), nan=0.0), -4, 4),
    np.sin(2 * np.pi * tod / 1440).to_numpy(), np.cos(2 * np.pi * tod / 1440).to_numpy()], 1).astype(np.float32))
RET1 = np.zeros(n, np.float32); RET1[:-1] = ((C[1:] - C[:-1]) / np.maximum(atr[:-1], 1e-9)).astype(np.float32)
RET1 = np.clip(RET1, -15, 15)
SPR = (SPREAD_PT / np.maximum(atr, 1e-9)).astype(np.float32)
ATRv = np.maximum(atr, 1e-9)

@njit(cache=True)
def zigzag_dir(H, L, atr, mult):
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
ZDIR = zigzag_dir(H, L, atr, ZZ_MULT).astype(np.float32)

ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-(MAXH + 2):] = False
cand = np.where((da <= 1.0) & ok)[0]; cdirs = cdir[cand].astype(np.int64)
ct = times.values[cand]; sig_atr = atr[cand]
log(f"candidates {len(cand):,}")

# ---------------------------------------------------------------- reference exits (baselines)
@njit(cache=True)
def sim_exit(idxs, dirs, O, H, L, C, atr, n, SL, TRAIL, MAXH, ta, tt):
    m = len(idxs); pnl = np.zeros(m); ebar = np.full(m, -1, np.int64); xit = np.full(m, -1, np.int64)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]
        if i + 1 >= n or not (a > 0): continue
        st = i + 1; epr = O[st]; hard = SL * a; mf = 0.0
        end = min(st + MAXH, n - 1); done = False
        for jx in range(st, end + 1):
            adv = (epr - L[jx]) if d == 1 else (H[jx] - epr)
            if adv >= hard: pnl[k] = -SL; xit[k] = jx; done = True; break
            fav = d * (C[jx] - epr)
            if fav > mf: mf = fav
            trd = TRAIL * a
            if ta > 0 and (jx - st) >= ta:
                w = tt * a
                if w < trd: trd = w
            if mf >= trd and (mf - fav) >= trd: pnl[k] = (mf - trd) / a; xit[k] = jx; done = True; break
        if not done: pnl[k] = d * (C[end] - epr) / a; xit[k] = end
        ebar[k] = st
    return pnl, ebar, xit

@njit(cache=True)
def take(order_idx, ebar, xit, cd):
    busy = -1; m = len(order_idx); out = np.empty(m, np.int64); c = 0
    for t in range(m):
        k = order_idx[t]
        if ebar[k] <= busy: continue
        out[c] = k; busy = xit[k] + cd; c += 1
    return out[:c]

pnl_base, ebar, xit_base = sim_exit(cand, cdirs, O, H, L, C, atr, n, SL, 2.0, MAXH, 0, 0.0)
pnl_tt, _, xit_tt = sim_exit(cand, cdirs, O, H, L, C, atr, n, SL, 2.0, MAXH, 30, 0.75)

# ---------------------------------------------------------------- gate (train-period only)
TRAIN_END = np.datetime64("2025-01-01")
trm = ct < TRAIN_END
rng = np.random.RandomState(0)
tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 250_000 else rng.choice(tix, 250_000, replace=False)
gate = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
                    colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
gate.fit(XFULL[cand][tix_f], pnl_base[tix_f]); preds = gate.predict(XFULL[cand]).astype(np.float64)
tr_days = max((pd.Timestamp(ct[tix].max()) - pd.Timestamp(ct[tix].min())).days * 5 / 7, 1)
best = 1e18; thr = 0.0
for th in np.quantile(preds[tix], np.linspace(0.30, 0.97, 24)):
    kk = tix[preds[tix] >= th]
    if len(kk) < 5: continue
    tk = take(kk[np.argsort(ebar[kk])].astype(np.int64), ebar, xit_base, COOLDOWN)
    gap = abs(len(tk) / tr_days - TARGET_PER_DAY)
    if gap < best: best = gap; thr = th
gated = preds >= thr
log(f"gate thr {thr:.3f}, gated {gated.sum():,} candidates ({gated[trm].sum():,} train / {gated[~trm].sum():,} holdout)")

CHUNKS = [("P1", "2018-01-01", "2020-01-01"), ("P2", "2020-01-01", "2022-01-01"),
          ("P3", "2022-01-01", "2024-01-01"), ("P4", "2024-01-01", "2025-01-01")]
POOL = {nm: np.where(gated & (ct >= np.datetime64(a)) & (ct < np.datetime64(b)))[0]
        for nm, a, b in CHUNKS}
for nm in POOL: log(f"  pool {nm}: {len(POOL[nm]):,} episodes")

# ---------------------------------------------------------------- env: manage ONE gated trade
class ExitEnv(gym.Env):
    def __init__(self, seed=0):
        super().__init__()
        self.action_space = spaces.Discrete(2)       # 0 hold, 1 exit
        self.observation_space = spaces.Box(-12, 12, (15,), np.float32)
        self.rng = np.random.RandomState(seed)
        self.pool = POOL["P1"]; self.prev = None; self.mix = 0.0
        self.lam = 0.0; self.spread_mult = 2.0
    def set_phase(self, pool, prev, mix): self.pool, self.prev, self.mix = pool, prev, mix
    def set_shape(self, lam, sm): self.lam, self.spread_mult = lam, sm
    def _obs(self):
        t = self.t; o = np.empty(15, np.float32); o[:11] = F[t]
        unreal = self.d * (C[t] - self.epr) / ATRv[self.sig]
        o[11] = cl(unreal, -8, 8); o[12] = min(self.held / 300.0, 2.0)
        o[13] = cl(self.peak, 0, 12); o[14] = cl(self.peak - unreal, 0, 12)
        return o
    def reset(self, *, seed=None, options=None):
        arr = self.prev if (self.prev is not None and len(self.prev) and self.rng.rand() < self.mix) else self.pool
        k = int(arr[self.rng.randint(len(arr))])
        self.sig = cand[k]; self.d = int(cdirs[k]); self.t = self.sig + 1
        self.epr = O[self.t]; self.held = 0; self.peak = 0.0; self.acc = 0.0
        self.end = min(self.t + MAXH, n - 1)
        self.first = True
        return self._obs(), {}
    def step(self, a):
        t = self.t; d = self.d; asig = ATRv[self.sig]
        r = 0.0
        if self.first:
            r -= SPR[self.sig] * self.spread_mult; self.first = False
        if a == 1:   # exit at this bar close (credit this bar's mark move too)
            r += d * (C[t] - self.epr) / asig - self.acc
            return self._obs(), float(r), True, False, {}
        adv = (self.epr - L[t]) if d == 1 else (H[t] - self.epr)
        if adv >= SL * asig:   # hard SL: true-up to exactly -SL total
            r += (-SL) - self.acc
            return self._obs(), float(r), True, False, {}
        step_r = d * (C[t] - self.epr) / asig - self.acc   # mark-to-market delta
        self.acc += step_r; r += step_r
        unreal = self.acc
        if unreal > self.peak: self.peak = unreal
        r += self.lam * SHAPE * d * ZDIR[t]
        self.t += 1; self.held += 1
        done = self.t > self.end
        return self._obs(), float(r), done, False, {}

# ---------------------------------------------------------------- deterministic RL-exit rollout
@njit(cache=True)
def rl_exit_all(W1, b1, W2, b2, W3, b3, cand, cdirs, O, H, L, C, atrv, n, SL, MAXH, F):
    m = len(cand); pnl = np.zeros(m); ebar = np.full(m, -1, np.int64); xit = np.full(m, -1, np.int64)
    obs = np.empty(15, np.float64)
    for k in range(m):
        i = cand[k]; d = cdirs[k]; a = atrv[i]
        st = i + 1
        if st >= n or not (a > 0): continue
        epr = O[st]; peak = 0.0; acc = 0.0
        end = min(st + MAXH, n - 1); done = False
        for t in range(st, end + 1):
            for j in range(11): obs[j] = F[t, j]
            unreal = d * (C[t] - epr) / a
            obs[11] = min(max(unreal, -8.0), 8.0)
            obs[12] = min((t - st) / 300.0, 2.0)
            obs[13] = min(max(peak, 0.0), 12.0)
            obs[14] = min(max(peak - unreal, 0.0), 12.0)
            h1 = np.tanh(W1 @ obs + b1); h2 = np.tanh(W2 @ h1 + b2); lg = W3 @ h2 + b3
            if lg[1] > lg[0]:   # exit at close
                pnl[k] = d * (C[t] - epr) / a
                xit[k] = t; done = True; break
            adv = (epr - L[t]) if d == 1 else (H[t] - epr)
            if adv >= SL * a:
                pnl[k] = -SL; xit[k] = t; done = True; break
            acc = d * (C[t] - epr) / a
            if acc > peak: peak = acc
        if not done: pnl[k] = d * (C[end] - epr) / a; xit[k] = end
        ebar[k] = st
    return pnl, ebar, xit

def eval_variant(pnl_, xit_, mask, spreads=(1.0, 1.5, 2.0)):
    kk = np.where(mask & gated)[0]
    tk = take(kk[np.argsort(ebar[kk])].astype(np.int64), ebar, xit_, COOLDOWN)
    R = pnl_[tk]; cost = 1.0 / sig_atr[tk]
    out = {}
    for sp in spreads:
        net = R - sp * cost
        eqd = np.cumsum(net); dd = float((np.maximum.accumulate(eqd) - eqd).max()) if len(eqd) else 0.0
        out[sp] = dict(n=len(tk), sumR=float(net.sum()), wr=float((net > 0).mean() * 100),
                       pf=float(net[net > 0].sum() / max(-net[net <= 0].sum(), 1e-9)), dd=dd)
    out["hold"] = float(np.mean(xit_[tk] - ebar[tk])) if len(tk) else 0.0
    return out

# ---------------------------------------------------------------- train
def train_seed(seed):
    log(f"=== RL gated-exit seed {seed} ===")
    venv = DummyVecEnv([(lambda s=seed * 100 + i: ExitEnv(seed=s)) for i in range(N_ENVS)])
    model = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=256, batch_size=1024,
                gamma=0.999, gae_lambda=0.95, ent_coef=0.005, seed=seed, verbose=0,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])))
    phases = [(nm, i) for i, (nm, _, _) in enumerate(CHUNKS)] + [("P5_recency", -1)]
    for pi_, (pname, ci) in enumerate(phases):
        if pname == "P5_recency":
            pool = np.concatenate([POOL["P3"], POOL["P4"]]); prev = None; mix = 0.0
            steps = 400_000; lr = 5e-5
        else:
            pool = POOL[pname]
            prevs = [POOL[nm] for nm, _, _ in CHUNKS[:ci]]
            prev = np.concatenate(prevs) if prevs else None
            mix = REHEARSAL if prev is not None else 0.0
            steps = PHASE_STEPS; lr = 3e-4 if pi_ == 0 else 1e-4
        model.learning_rate = lr; model._setup_lr_schedule()
        for env in venv.envs: env.set_phase(pool, prev, mix)
        for env in venv.envs: env.set_shape(0.3, 2.0)
        model.learn(total_timesteps=int(steps * 0.4), reset_num_timesteps=False)
        for env in venv.envs: env.set_shape(0.0, 2.0)
        model.learn(total_timesteps=int(steps * 0.6), reset_num_timesteps=False)
        log(f"  seed {seed} phase {pname} done")
    p = model.policy; pn = p.mlp_extractor.policy_net
    W1 = pn[0].weight.detach().numpy().astype(np.float64); b1 = pn[0].bias.detach().numpy().astype(np.float64)
    W2 = pn[2].weight.detach().numpy().astype(np.float64); b2 = pn[2].bias.detach().numpy().astype(np.float64)
    W3 = p.action_net.weight.detach().numpy().astype(np.float64); b3 = p.action_net.bias.detach().numpy().astype(np.float64)
    return rl_exit_all(W1, b1, W2, b2, W3, b3, cand, cdirs, O, H, L, C, ATRv, n, SL, MAXH, F.astype(np.float64))

VAR = {"SL7/T2 base": (pnl_base, xit_base), "tt30/0.75 (deployed)": (pnl_tt, xit_tt)}
for seed in SEEDS:
    pnl_r, _, xit_r = train_seed(seed)
    VAR[f"RL_exit seed{seed}"] = (pnl_r, xit_r)

# ---------------------------------------------------------------- report
hom = ~trm
print(f"\n{'='*100}\nGATED RL EXIT vs deployed exits — same candidates, same gate, 1-slot (DJI)\n{'='*100}")
for label, mask in [("TRAIN 2018-2024", trm), ("HOLDOUT 2025+ (untouched)", hom)]:
    print(f"\n--- {label} ---")
    print(f"{'exit policy':<24}{'n':>7}{'net@1pt':>10}{'net@1.5':>10}{'net@2pt':>10}{'PF@1.5':>8}{'WR%':>6}{'DD@1.5':>8}{'hold':>6}")
    for vn, (pnl_, xit_) in VAR.items():
        r = eval_variant(pnl_, xit_, mask)
        print(f"{vn:<24}{r[1.5]['n']:>7}{r[1.0]['sumR']:>+10.0f}{r[1.5]['sumR']:>+10.0f}{r[2.0]['sumR']:>+10.0f}"
              f"{r[1.5]['pf']:>8.2f}{r[1.5]['wr']:>6.0f}{r[1.5]['dd']:>8.0f}{r['hold']:>6.0f}")
json.dump({vn: {lab: eval_variant(p_, x_, m_)[1.5] for lab, m_ in [("train", trm), ("holdout", hom)]}
           for vn, (p_, x_) in VAR.items()}, open(OUT / "rl_gated_exit_results.json", "w"), indent=1, default=str)
log("gated RL exit lab done")
