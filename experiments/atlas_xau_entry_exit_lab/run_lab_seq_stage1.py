"""
SEQUENCE-MODEL STAGE 1 (proof of signal) — Jay 2026-07-15: "train RL/LSTM/CNN,
anything". Genuinely untried angle: a 1D-CNN+GRU reads the RAW last-64-bars
window (returns, range, close-position, ATR-rel) + the 29 engineered feats, and
scores edge_pullback candidates with pinball(0.1) loss — the deep q10.

STAGED design (no holdout touched):
  train 2020-07..2023-12, evaluate 2024-01..2024-12 (dev year)
  baseline = XGB q10 on 29 feats, SAME train rows, SAME eval trades
  metrics: portfolio net@$0.20 at ~11/day (1-slot cd5) + top-decile mean R
Only if the seq model beats XGB here does it earn the full 12-window WF +
holdout. CPU torch; ~200k train sequences.
"""
import sys, pickle, time, json
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from xgboost import XGBRegressor
import torch, torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
SRV = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(SRV))
import decision_engine.edge_pullback as ep
OUT = Path(__file__).parent
FC = pickle.load(open(SRV / "decision_engine/models/hermes_dji_validated.pkl", "rb"))["feat_cols"]
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
SEQ = 64; SP = 0.20; COOLDOWN = 5; TA, TT = 30, 0.75
TR_S, TR_E = pd.Timestamp("2020-07-01"), pd.Timestamp("2024-01-01")
TE_S, TE_E = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")

@njit(cache=True)
def sim_tt(idxs, dirs, O, H, L, C, atr, n, SL, TRAIL, MAXH, ta, tt):
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

@njit(cache=True)
def build_seq(sel_idx, C, H, L, O, atr, seq):
    m = len(sel_idx); X = np.zeros((m, 4, seq), np.float32)
    for q in range(m):
        i = sel_idx[q]
        for s in range(seq):
            j = i - seq + 1 + s
            a = atr[i]
            if a <= 0: continue
            X[q, 0, s] = (C[j] - C[j - 1]) / a                 # return in ATR
            X[q, 1, s] = (H[j] - L[j]) / a                     # bar range
            rng_ = H[j] - L[j]
            X[q, 2, s] = ((C[j] - L[j]) / rng_ - 0.5) if rng_ > 0 else 0.0
            X[q, 3, s] = (C[j] - C[i]) / a                     # dist from decision close
    return X

class SeqNet(nn.Module):
    def __init__(self, nstatic):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2, stride=2), nn.ReLU(),
            nn.Conv1d(32, 48, 3, padding=1, stride=2), nn.ReLU())
        self.gru = nn.GRU(48, 48, batch_first=True)
        self.head = nn.Sequential(nn.Linear(48 + nstatic, 64), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(64, 1))
    def forward(self, xs, xf):
        h = self.conv(xs).transpose(1, 2)
        _, hn = self.gru(h)
        return self.head(torch.cat([hn[-1], xf], dim=1)).squeeze(-1)

def pinball(pred, y, a=0.10):
    d = y - pred
    return torch.mean(torch.maximum(a * d, (a - 1) * d))

log("loading XAU M1...")
df = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet")
df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
feat = ep.compute_edge_features(df); log("features done")
atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
da = np.abs(feat["dist_at_signal"].to_numpy(float))
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df)
ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64)
ct = pd.to_datetime(times[idx]); sig_atr = atr[idx]
X29 = np.nan_to_num(feat[FC].to_numpy(np.float32)[idx], nan=0.0, posinf=0.0, neginf=0.0)
del feat
pnl, eb, xt = sim_tt(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, 300, TA, TT)

trm = (ct >= TR_S) & (ct < TR_E); tem = (ct >= TE_S) & (ct < TE_E)
tix = np.where(trm)[0]; tex = np.where(tem)[0]
rng = np.random.RandomState(0)
tix_f = tix if len(tix) <= 200_000 else rng.choice(tix, 200_000, replace=False)
tr_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1)
log(f"train {len(tix_f):,} rows, eval {len(tex):,} candidates (2024)")

# static feats standardized on train
mu, sd = X29[tix_f].mean(0), X29[tix_f].std(0) + 1e-9
XF = (X29 - mu) / sd

def portfolio(p, label):
    cand = np.quantile(p[tix], np.linspace(0.30, 0.97, 24)); thr = cand[-1]; best = 1e18
    for th in cand:
        kk = tix[p[tix] >= th]
        if len(kk) < 5: continue
        tk = take(kk[np.argsort(eb[kk])].astype(np.int64), eb, xt, COOLDOWN)
        gap = abs(len(tk) / tr_days - 11.0)
        if gap < best: best = gap; thr = th
    kk = tex[p[tex] >= thr]
    tk = take(kk[np.argsort(eb[kk])].astype(np.int64), eb, xt, COOLDOWN)
    netv = pnl[tk] - SP / sig_atr[tk]
    dec = np.quantile(p[tex], 0.9)
    top = pnl[tex][p[tex] >= dec]
    print(f"  {label:<10} 2024: n={len(tk):>5} net@20c {netv.sum():+8.0f}R "
          f"per-trade {netv.mean() if len(tk) else 0:+.3f}  WR {(netv>0).mean()*100 if len(tk) else 0:4.1f}%  "
          f"top-decile mean {top.mean():+.3f}R (n={len(top)})")
    return float(netv.sum())

log("XGB q10 baseline...")
mq = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
                  colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0,
                  objective="reg:quantileerror", quantile_alpha=0.10)
mq.fit(X29[tix_f], pnl[tix_f])
pq = mq.predict(X29).astype(np.float64)
net_xgb = portfolio(pq, "xgb_q10")

log("building sequences...")
XS_tr = build_seq(idx[tix_f], C, H, L, O, atr, SEQ)
y_tr = pnl[tix_f].astype(np.float32)
XF_tr = XF[tix_f]
log("training seq net (CPU)...")
net = SeqNet(X29.shape[1])
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
BS = 2048
ord_ = np.arange(len(y_tr))
for epoch in range(4):
    rng.shuffle(ord_); tot = 0.0; nb = 0
    for b in range(0, len(ord_), BS):
        sel = ord_[b:b + BS]
        xs = torch.from_numpy(XS_tr[sel]); xf = torch.from_numpy(XF_tr[sel])
        yy = torch.from_numpy(y_tr[sel])
        opt.zero_grad()
        loss = pinball(net(xs, xf), yy)
        loss.backward(); opt.step()
        tot += float(loss); nb += 1
    log(f"  epoch {epoch + 1}: pinball {tot / nb:.4f}")

log("scoring all candidates with seq net...")
net.eval()
ps = np.full(len(idx), -9e9)
with torch.no_grad():
    for src in (tix, tex):
        for b in range(0, len(src), 8192):
            sel = src[b:b + 8192]
            xs = torch.from_numpy(build_seq(idx[sel], C, H, L, O, atr, SEQ))
            xf = torch.from_numpy(XF[sel])
            ps[sel] = net(xs, xf).numpy()
net_seq = portfolio(ps, "seq_cnn")

print(f"\nVERDICT stage-1: xgb_q10 {net_xgb:+.0f}R vs seq_cnn {net_seq:+.0f}R on 2024 "
      f"-> {'SEQ EARNS FULL WF' if net_seq > net_xgb * 1.1 else 'NO ADVANTAGE — stop here'}")
log("seq stage-1 done")
