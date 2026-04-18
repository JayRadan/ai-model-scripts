"""
STRUCTURAL-REWARD CURRICULUM RL
================================
Per-cluster recurrent DQN that learns to recognize LOCAL PIVOTS in price,
not to memorize bar-index → PnL mappings.

Reward philosophy
-----------------
Agent is rewarded ONLY at entry and exit, and ONLY when the action touches a
real local pivot:

  entry_LONG  at bar i → reward = W_ENTRY · prox(i, nearest_low) · strength_low(i)
  entry_SHORT at bar i → reward = W_ENTRY · prox(i, nearest_high)· strength_high(i)
  exit_LONG   at bar i → reward = W_EXIT  · prox(i, nearest_high)· strength_high(i)
  exit_SHORT  at bar i → reward = W_EXIT  · prox(i, nearest_low) · strength_low(i)

where
  prox(i, p)      = exp(-|i - p_idx| / TOL)
  strength_low(i) = (high_after - low_i) / atr_i,   clamped [0, 3]
  strength_high(i)= (high_i - low_after) / atr_i,   clamped [0, 3]

Pivot search window (±L) is PRE-COMPUTED on training data (non-causal, which
is fine for training labels — the agent still sees only causal features at
runtime). No mark-to-market, no PnL reward, no penalty for holding.

Anti-memorization:
  1. Random episode starts across the whole training pool every reset
  2. Short episodes (150 bars)
  3. Inner 90/10 train/val split for early-stopping
  4. Smaller LSTM (hidden=64) + dropout + weight decay
  5. Curriculum on PIVOT STRENGTH THRESHOLD (big swings first, then small)
  6. Per-cluster action masks (C1 short-only, C3 long-only)

Honest final eval on the LAST 20% of each cluster that the agent has never
seen. Reports both the structural score AND raw PnL (for comparison with
previous runs).

Usage: python3 train_rl_structural.py
"""
from __future__ import annotations
import json, random, time
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Features ────────────────────────────────────────────────────────────────
BASE_FEATURES = [
    "f01_CPR","f02_WickAsym","f03_BEF","f04_TCS","f05_SPI",
    "f06_LRSlope","f07_RECR","f08_SCM","f09_HLER","f10_EP",
    "f11_KE","f12_MCS","f13_Work","f14_EDR","f15_AI",
    "f16_PPShigh","f16_PPSlow","f17_SCR","f18_RVD","f19_WBER","f20_NCDE",
]
TECH_FEATURES = [
    "rsi14","rsi6","stoch_k","stoch_d","bb_pct",
    "mom5","mom10","mom20",
    "ll_dist10","hh_dist10",
    "vol_accel","atr_ratio","spread_norm",
    "hour_enc","dow_enc",
]
FEATURE_COLS = BASE_FEATURES + TECH_FEATURES  # 36
FEAT_DIM       = len(FEATURE_COLS)
POS_DIM        = 4
WINDOW_K       = 20

CLUSTER_NAMES = {0:"Ranging", 1:"Downtrend", 2:"Shock_News", 3:"Uptrend"}
CLUSTERS      = [0, 1, 2, 3]

# action encoding: 0 FLAT, 1 LONG, 2 SHORT
CLUSTER_ACTIONS = {
    0: [0, 1, 2],
    1: [0, 2],
    2: [0, 1, 2],
    3: [0, 1],
}

# ── Pivot / reward config ───────────────────────────────────────────────────
PIVOT_LOOKBACK  = 8          # WIDER local-extremum window — fewer, more meaningful pivots
PIVOT_FWD_LOOK  = 20         # longer forward look — measure bigger swings
HARD_PROX_BARS  = 2          # HARD: pivot reward only if within ±N bars of actual pivot
W_ENTRY         = 1.0
W_EXIT          = 1.0
W_PNL           = 0.5         # NEW: realised PnL contribution to close reward (in R units)
MIN_HOLD_BARS   = 5
MIN_PIVOT_DIST  = 6           # entry pivot vs exit pivot must differ by ≥ this many bars
SHORT_HOLD_PENALTY_BARS = 3
SHORT_HOLD_PENALTY      = 1.0
IDLE_BONUS_PER_BAR      = 0.01  # tiny per-bar reward while flat → encourages selectivity
MAX_HOLD_BARS           = 50    # FORCE close after this many bars — kills "hold forever" exploit
HARD_SL_ATR             = 1.5   # force close if loss exceeds this many ATRs (v5 used 1.0 — too tight)
HARD_TP_ATR             = 3.0   # take profit at this many ATRs

# Hard floor on pivot strength — pivots below this give zero reward at any
# curriculum phase. Forces the agent to learn ECONOMICALLY MEANINGFUL pivots.
ABSOLUTE_MIN_PIVOT_STRENGTH = 0.5

# Curriculum — pivot strength threshold. Phase 1 only counts huge swings;
# phases relax until all pivots count (but never below ABSOLUTE_MIN).
PIVOT_CURRICULUM = [2.5, 2.0, 1.5, 1.0, 0.5]

# Spread still charged on open
SPREAD_USD = 0.40

# ── Episode / training ──────────────────────────────────────────────────────
EPISODE_LEN            = 150
VAL_FRAC               = 0.10
STEPS_PER_PHASE_CAP    = 20_000
EVAL_EVERY             = 500
MASTERY_VAL_REWARD     = 0.3     # avg reward per trade on the val set
MASTERY_STABILITY      = 3
MIN_TRADES_FOR_EVAL    = 10

# ── DQN ─────────────────────────────────────────────────────────────────────
GAMMA          = 0.95
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
BATCH          = 64
BUFFER_SIZE    = 20_000
TARGET_SYNC    = 500
EPS_START      = 1.0
EPS_END        = 0.05
EPS_DECAY_STEPS = 15_000
WARMUP         = 800
LSTM_HIDDEN    = 64
HEAD_HIDDEN    = 64
DROPOUT        = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42); random.seed(42)


# ── Utils ───────────────────────────────────────────────────────────────────
def compute_atr(high, low, close, period=14):
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low,
                            np.abs(high - prev_close),
                            np.abs(low  - prev_close)])
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()
    return np.clip(atr, 1e-10, None)


def precompute_pivots(highs: np.ndarray, lows: np.ndarray, atr: np.ndarray,
                       L: int = PIVOT_LOOKBACK, fwd: int = PIVOT_FWD_LOOK):
    """
    For each bar i, find whether it's a local high / low and how strong that
    pivot is (based on the forward swing size normalized by ATR).

    Returns dict with arrays:
      is_low[i]       : 1 if low[i] is min in [i-L, i+L]
      is_high[i]      : 1 if high[i] is max in [i-L, i+L]
      nearest_low[i]  : index of nearest bar with is_low=1  (search both directions)
      nearest_high[i] : same for highs
      low_strength[i] : (max(high[i:i+fwd]) - low[i]) / atr[i]     clipped [0, 3]
      high_strength[i]: (high[i] - min(low[i:i+fwd])) / atr[i]     clipped [0, 3]
    """
    n = len(highs)
    is_low  = np.zeros(n, dtype=bool)
    is_high = np.zeros(n, dtype=bool)

    # Local extremum detection (non-causal — OK for training labels)
    # Using np stride view would be faster; the plain loop is readable and
    # only runs once per cluster, so it's fine.
    for i in range(n):
        lo_win = lows [max(0, i - L) : min(n, i + L + 1)]
        hi_win = highs[max(0, i - L) : min(n, i + L + 1)]
        if lows [i] == lo_win.min(): is_low [i] = True
        if highs[i] == hi_win.max(): is_high[i] = True

    # Forward swing strength — bar i's "quality" as a bottom: how far does
    # price go UP in the next `fwd` bars relative to atr[i]?
    low_strength  = np.zeros(n, dtype=np.float32)
    high_strength = np.zeros(n, dtype=np.float32)
    for i in range(n):
        j = min(n, i + fwd + 1)
        if is_low[i]:
            move_up = highs[i:j].max() - lows[i]
            low_strength[i] = np.clip(move_up / max(atr[i], 1e-6), 0, 3)
        if is_high[i]:
            move_down = highs[i] - lows[i:j].min()
            high_strength[i] = np.clip(move_down / max(atr[i], 1e-6), 0, 3)

    # Distance to nearest low/high (two-sided search)
    low_idx  = np.where(is_low)[0]
    high_idx = np.where(is_high)[0]

    def nearest_distance(n: int, sorted_idx: np.ndarray) -> np.ndarray:
        out = np.full(n, np.inf, dtype=np.float32)
        if len(sorted_idx) == 0:
            return out
        pos = 0
        for i in range(n):
            while pos < len(sorted_idx) - 1 and abs(sorted_idx[pos+1] - i) <= abs(sorted_idx[pos] - i):
                pos += 1
            # pos now holds the closest from the left; also check right neighbour
            best = abs(sorted_idx[pos] - i)
            if pos + 1 < len(sorted_idx):
                alt = abs(sorted_idx[pos+1] - i)
                if alt < best: best = alt
            out[i] = best
        return out

    dist_to_low  = nearest_distance(n, low_idx)
    dist_to_high = nearest_distance(n, high_idx)

    # For each bar, record BOTH the index and strength of its nearest pivot.
    # Knowing the index lets us enforce "exit pivot != entry pivot" at runtime.
    def nearest_info(n: int, sorted_idx: np.ndarray, strength: np.ndarray):
        out_idx = np.full(n, -1, dtype=np.int32)
        out_str = np.zeros(n, dtype=np.float32)
        if len(sorted_idx) == 0:
            return out_idx, out_str
        pos = 0
        for i in range(n):
            while pos < len(sorted_idx) - 1 and abs(sorted_idx[pos+1] - i) <= abs(sorted_idx[pos] - i):
                pos += 1
            best_i = sorted_idx[pos]; best_d = abs(best_i - i)
            if pos + 1 < len(sorted_idx):
                alt_i = sorted_idx[pos+1]; alt_d = abs(alt_i - i)
                if alt_d < best_d: best_i, best_d = alt_i, alt_d
            out_idx[i] = best_i
            out_str[i] = strength[best_i]
        return out_idx, out_str

    nearest_low_idx,  nearest_low_strength  = nearest_info(n, low_idx,  low_strength)
    nearest_high_idx, nearest_high_strength = nearest_info(n, high_idx, high_strength)

    return {
        "is_low": is_low, "is_high": is_high,
        "dist_to_low": dist_to_low, "dist_to_high": dist_to_high,
        "nearest_low_idx":       nearest_low_idx,
        "nearest_high_idx":      nearest_high_idx,
        "nearest_low_strength":  nearest_low_strength,
        "nearest_high_strength": nearest_high_strength,
    }


# ── Environment ─────────────────────────────────────────────────────────────
class StructuralEnv:
    """Windowed state, free-exit actions, STRUCTURAL reward at entry/exit only.

    Reward at entry LONG:
       W_ENTRY · exp(-dist_to_low[i]/TOL) · max(0, nearest_low_strength[i] - phase_threshold)
    At entry SHORT:
       W_ENTRY · exp(-dist_to_high[i]/TOL) · max(0, nearest_high_strength[i] - phase_threshold)
    At exit of LONG (close):
       W_EXIT · exp(-dist_to_high[i]/TOL) · max(0, nearest_high_strength[i] - phase_threshold)
    At exit of SHORT:
       W_EXIT · exp(-dist_to_low[i]/TOL) · max(0, nearest_low_strength[i] - phase_threshold)

    A small spread cost is still applied on open to discourage spam flipping,
    but it enters the reward as a flat -$0.04 normalized term (not in pivot space)
    so it doesn't dominate.
    """
    POS_FLAT, POS_LONG, POS_SHORT = 0, 1, 2

    def __init__(self, df, feature_norm, pivots, phase_threshold: float):
        feats = df[FEATURE_COLS].fillna(0).to_numpy(dtype=np.float32)
        self.closes = df["close"].to_numpy(dtype=np.float64)
        self.highs  = df["high"].to_numpy(dtype=np.float64)
        self.lows   = df["low"].to_numpy(dtype=np.float64)
        self.atr    = compute_atr(self.highs, self.lows, self.closes, 14)

        mu, sigma = feature_norm
        self.features = ((feats - mu) / sigma).astype(np.float32)

        self.pivots = pivots
        self.phase_threshold = phase_threshold

        self.n = len(df)
        self.reset_full()

    def reset_full(self):
        self.i = WINDOW_K
        self.episode_end = self.n - 1
        self._reset_position()
        self.trades = []

    def _reset_position(self):
        self.current_pos = self.POS_FLAT
        self.entry_price = 0.0
        self.entry_atr   = 1.0
        self.pos_bars    = 0
        self.entry_pivot_idx = -1   # index of the pivot that entry was rewarded for

    def reset(self, start=None, episode_len=EPISODE_LEN):
        min_start = WINDOW_K
        max_start = max(min_start, self.n - episode_len - 2)
        if start is None:
            start = random.randint(min_start, max_start)
        self.i = int(start)
        self.episode_end = min(self.i + episode_len, self.n - 1)
        self._reset_position()
        self.trades = []
        return self._state()

    def _window(self):
        start = self.i - WINDOW_K + 1
        if start >= 0:
            return self.features[start:self.i + 1]
        pad = np.zeros((-start, FEAT_DIM), dtype=np.float32)
        return np.concatenate([pad, self.features[0:self.i + 1]], axis=0)

    def _pos_feats(self):
        in_long  = 1.0 if self.current_pos == self.POS_LONG  else 0.0
        in_short = 1.0 if self.current_pos == self.POS_SHORT else 0.0
        if self.current_pos == self.POS_FLAT:
            age, unreal = 0.0, 0.0
        else:
            age = min(self.pos_bars / 40.0, 3.0)
            sign = 1.0 if self.current_pos == self.POS_LONG else -1.0
            unreal = sign * (self.closes[self.i] - self.entry_price) / (self.entry_atr + 1e-9)
        return np.array([in_long, in_short, age, unreal], dtype=np.float32)

    def _state(self):
        return (self._window(), self._pos_feats())

    # ── Pivot reward helpers ─────────────────────────────────────────────
    def _proximity_low(self, i):
        # HARD cutoff — only pay if within ±HARD_PROX_BARS of an actual pivot
        d = self.pivots["dist_to_low"][i]
        return 1.0 if d <= HARD_PROX_BARS else 0.0

    def _proximity_high(self, i):
        d = self.pivots["dist_to_high"][i]
        return 1.0 if d <= HARD_PROX_BARS else 0.0

    def _strength_low(self, i):
        s = float(self.pivots["nearest_low_strength"][i])
        # Hard floor: any pivot below ABSOLUTE_MIN earns nothing, period
        if s < ABSOLUTE_MIN_PIVOT_STRENGTH:
            return 0.0
        # Curriculum: phase threshold further filters down
        return max(0.0, s - self.phase_threshold)

    def _strength_high(self, i):
        s = float(self.pivots["nearest_high_strength"][i])
        if s < ABSOLUTE_MIN_PIVOT_STRENGTH:
            return 0.0
        return max(0.0, s - self.phase_threshold)

    # ── Close helpers ────────────────────────────────────────────────────
    def _exit_reward(self, i_exit: int) -> float:
        """Reward for closing current position at bar i_exit.

        Exit of LONG must be near a LOCAL HIGH that is DIFFERENT from the
        entry pivot. Exit of SHORT must be near a LOCAL LOW that is
        DIFFERENT from the entry pivot. Closes under MIN_HOLD_BARS pay
        zero exit reward AND receive a SHORT_HOLD_PENALTY.
        """
        # Mandatory minimum hold — structurally no reward + penalty for flips
        if self.pos_bars < MIN_HOLD_BARS:
            return -SHORT_HOLD_PENALTY if self.pos_bars < SHORT_HOLD_PENALTY_BARS else 0.0

        if self.current_pos == self.POS_LONG:
            pivot_idx = int(self.pivots["nearest_high_idx"][i_exit])
            if pivot_idx < 0:                       return 0.0
            if abs(pivot_idx - self.entry_pivot_idx) < MIN_PIVOT_DIST:
                return 0.0                           # same / near same pivot → no reward
            return W_EXIT * self._proximity_high(i_exit) * self._strength_high(i_exit)
        else:  # SHORT
            pivot_idx = int(self.pivots["nearest_low_idx"][i_exit])
            if pivot_idx < 0:                       return 0.0
            if abs(pivot_idx - self.entry_pivot_idx) < MIN_PIVOT_DIST:
                return 0.0
            return W_EXIT * self._proximity_low(i_exit) * self._strength_low(i_exit)

    # ── Step ─────────────────────────────────────────────────────────────
    def step(self, desired: int):
        reward = 0.0
        i = self.i

        # ── Idle bonus while flat ──
        # If the agent stays flat OR is currently flat AND chooses to remain
        # flat, accumulate a tiny per-bar reward. This makes "do nothing on a
        # mediocre setup" actively good instead of just neutral.
        if self.current_pos == self.POS_FLAT and desired == self.POS_FLAT:
            reward += IDLE_BONUS_PER_BAR

        # Transition handling — at start of current bar
        if desired != self.current_pos:
            # ── Close existing position ──
            if self.current_pos != self.POS_FLAT:
                # Block manual close under MIN_HOLD_BARS: agent's request is ignored
                # (the agent loses its turn but position stays). This prevents 1-bar flips.
                if self.pos_bars < MIN_HOLD_BARS:
                    reward -= SHORT_HOLD_PENALTY * 0.5   # small penalty for trying
                    desired = self.current_pos            # force hold current pos
                else:
                    exit_price = self.closes[i]
                    if self.current_pos == self.POS_LONG:
                        pnl = exit_price - self.entry_price
                    else:
                        pnl = self.entry_price - exit_price
                    net = pnl - SPREAD_USD
                    # Structural exit reward
                    reward += self._exit_reward(i)
                    # NEW: realized-PnL component (in R-units)
                    reward += W_PNL * (net / max(self.entry_atr, 1e-6))
                    self.trades.append({
                        "pnl": float(net),
                        "bars": self.pos_bars,
                        "dir":  "long" if self.current_pos==self.POS_LONG else "short",
                    })

            # ── Open new position ──
            if desired != self.current_pos and desired != self.POS_FLAT:
                self.entry_price = self.closes[i]
                self.entry_atr   = max(self.atr[i], 1e-6)
                self.pos_bars    = 0
                if desired == self.POS_LONG:
                    reward += W_ENTRY * self._proximity_low(i) * self._strength_low(i)
                    self.entry_pivot_idx = int(self.pivots["nearest_low_idx"][i])
                else:
                    reward += W_ENTRY * self._proximity_high(i) * self._strength_high(i)
                    self.entry_pivot_idx = int(self.pivots["nearest_high_idx"][i])
                reward -= SPREAD_USD * 0.05
                self.current_pos = desired
            elif desired == self.POS_FLAT and self.current_pos == self.POS_FLAT:
                pass  # no-op
            elif desired != self.current_pos:
                # Transitioned to FLAT via the close branch above — nothing to open
                self.current_pos = desired
                self.entry_pivot_idx = -1

        # Advance bar
        self.i += 1
        if self.i >= self.n: self.i = self.n - 1
        if self.current_pos != self.POS_FLAT:
            self.pos_bars += 1

        # ── Hard SL / TP check on the bar we just moved into ──
        # Uses bar high/low to detect intraday hits. Applied BEFORE MAX_HOLD
        # so a stop-out is recorded accurately.
        if self.current_pos != self.POS_FLAT:
            hi_now = self.highs[self.i]
            lo_now = self.lows[self.i]
            sl_dist = HARD_SL_ATR * self.entry_atr
            tp_dist = HARD_TP_ATR * self.entry_atr
            sl_hit = False
            tp_hit = False
            exit_px = self.closes[self.i]
            if self.current_pos == self.POS_LONG:
                if lo_now <= self.entry_price - sl_dist:
                    sl_hit = True
                    exit_px = self.entry_price - sl_dist
                elif hi_now >= self.entry_price + tp_dist:
                    tp_hit = True
                    exit_px = self.entry_price + tp_dist
            else:  # SHORT
                if hi_now >= self.entry_price + sl_dist:
                    sl_hit = True
                    exit_px = self.entry_price + sl_dist
                elif lo_now <= self.entry_price - tp_dist:
                    tp_hit = True
                    exit_px = self.entry_price - tp_dist

            if sl_hit or tp_hit:
                if self.current_pos == self.POS_LONG:
                    pnl = exit_px - self.entry_price
                else:
                    pnl = self.entry_price - exit_px
                net = pnl - SPREAD_USD
                # Structural exit reward only if above min hold AND pivot conditions
                if self.pos_bars >= MIN_HOLD_BARS:
                    reward += self._exit_reward(self.i)
                reward += W_PNL * (net / max(self.entry_atr, 1e-6))
                # Small penalty for hitting SL, small bonus for hitting TP cleanly
                if sl_hit:
                    reward -= 0.1
                else:
                    reward += 0.2
                self.trades.append({
                    "pnl": float(net),
                    "bars": self.pos_bars,
                    "dir": "long" if self.current_pos==self.POS_LONG else "short",
                })
                self._reset_position()

        # ── Force close on MAX_HOLD ──
        # Kills "hold forever" exploit. The agent gets the structural exit
        # reward (if applicable) plus a small penalty for letting it time out.
        forced_close = False
        if self.current_pos != self.POS_FLAT and self.pos_bars >= MAX_HOLD_BARS:
            i2 = self.i
            if self.pos_bars >= MIN_HOLD_BARS:
                reward += self._exit_reward(i2)
            if self.current_pos == self.POS_LONG:
                pnl = self.closes[i2] - self.entry_price
            else:
                pnl = self.entry_price - self.closes[i2]
            net = pnl - SPREAD_USD
            reward += W_PNL * (net / max(self.entry_atr, 1e-6))
            reward -= 0.2  # small penalty for needing to be force-closed
            self.trades.append({
                "pnl": float(net),
                "bars": self.pos_bars,
                "dir": "long" if self.current_pos==self.POS_LONG else "short",
            })
            self._reset_position()
            forced_close = True

        done = self.i >= self.episode_end
        if done and self.current_pos != self.POS_FLAT:
            i2 = self.i
            if self.pos_bars >= MIN_HOLD_BARS:
                reward += self._exit_reward(i2)
            if self.current_pos == self.POS_LONG:
                pnl = self.closes[i2] - self.entry_price
            else:
                pnl = self.entry_price - self.closes[i2]
            net = pnl - SPREAD_USD
            reward += W_PNL * (net / max(self.entry_atr, 1e-6))
            self.trades.append({
                "pnl": float(net),
                "bars": self.pos_bars,
                "dir": "long" if self.current_pos==self.POS_LONG else "short",
            })
            self._reset_position()

        return self._state(), float(reward), done, {}


# ── LSTM Q-network (smaller + dropout) ─────────────────────────────────────
class LSTMQNet(nn.Module):
    def __init__(self, feat_dim, pos_dim, n_actions,
                 lstm_hidden=LSTM_HIDDEN, head_hidden=HEAD_HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, lstm_hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + pos_dim, head_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_actions),
        )

    def forward(self, w, p):
        out, _ = self.lstm(w)
        last = self.drop(out[:, -1, :])
        return self.head(torch.cat([last, p], dim=-1))


@dataclass
class Transition:
    w: np.ndarray; p: np.ndarray; a: int; r: float
    w1: np.ndarray; p1: np.ndarray; done: bool

class ReplayBuffer:
    def __init__(self, sz): self.buf = deque(maxlen=sz)
    def push(self, t): self.buf.append(t)
    def __len__(self): return len(self.buf)
    def sample(self, n):
        ts = random.sample(self.buf, n)
        w  = torch.from_numpy(np.stack([t.w  for t in ts])).to(DEVICE)
        p  = torch.from_numpy(np.stack([t.p  for t in ts])).to(DEVICE)
        a  = torch.tensor([t.a for t in ts], dtype=torch.long, device=DEVICE)
        r  = torch.tensor([t.r for t in ts], dtype=torch.float32, device=DEVICE)
        w1 = torch.from_numpy(np.stack([t.w1 for t in ts])).to(DEVICE)
        p1 = torch.from_numpy(np.stack([t.p1 for t in ts])).to(DEVICE)
        d  = torch.tensor([t.done for t in ts], dtype=torch.float32, device=DEVICE)
        return w, p, a, r, w1, p1, d


class RDQNAgent:
    def __init__(self, allowed):
        self.q_net      = LSTMQNet(FEAT_DIM, POS_DIM, 3).to(DEVICE)
        self.target_net = LSTMQNet(FEAT_DIM, POS_DIM, 3).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optim  = torch.optim.AdamW(self.q_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.allowed = list(allowed)
        mask = torch.full((3,), float("-inf"), device=DEVICE)
        for a in self.allowed: mask[a] = 0.0
        self.action_mask = mask
        self.step_count = 0

    def _q(self, w: np.ndarray, p: np.ndarray):
        W = torch.from_numpy(w).unsqueeze(0).to(DEVICE)
        P = torch.from_numpy(p).unsqueeze(0).to(DEVICE)
        return self.q_net(W, P)

    def act(self, state, eps):
        if random.random() < eps:
            return random.choice(self.allowed)
        self.q_net.eval()
        with torch.no_grad():
            q = self._q(*state) + self.action_mask
        self.q_net.train()
        return int(torch.argmax(q, dim=1).item())

    def train_step(self):
        if len(self.buffer) < max(BATCH, WARMUP): return None
        w, p, a, r, w1, p1, d = self.buffer.sample(BATCH)
        with torch.no_grad():
            online_next = self.q_net(w1, p1) + self.action_mask
            next_a = torch.argmax(online_next, dim=1)
            next_q = self.target_net(w1, p1).gather(1, next_a.unsqueeze(1)).squeeze(1)
            target = r + GAMMA * next_q * (1.0 - d)
        q_sa = self.q_net(w, p).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_sa, target)
        self.optim.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optim.step()
        self.step_count += 1
        if self.step_count % TARGET_SYNC == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())


# ── Deterministic rollout evaluation ───────────────────────────────────────
def eval_policy(agent: RDQNAgent, df: pd.DataFrame, norm, pivots, phase_threshold):
    env = StructuralEnv(df, norm, pivots, phase_threshold)
    env.reset_full()
    env.i = WINDOW_K
    env.episode_end = env.n - 1
    state = env._state()
    total_r = 0.0
    while env.i < env.episode_end:
        agent.q_net.eval()
        with torch.no_grad():
            q = agent._q(*state) + agent.action_mask
        agent.q_net.train()
        action = int(torch.argmax(q, dim=1).item())
        state, r, done, _ = env.step(action)
        total_r += r
        if done: break

    trades = env.trades
    if not trades:
        return {"n":0,"pnl":0,"winrate":0,"pf":0,"avg_bars":0,
                "mean_r_per_trade":0,"total_structural_r":float(total_r)}
    pnl = np.array([t["pnl"] for t in trades])
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 999.0
    return {
        "n": int(len(pnl)),
        "pnl": float(pnl.sum()),
        "winrate": float((pnl > 0).mean()),
        "pf": float(pf),
        "avg_bars": float(np.mean([t["bars"] for t in trades])),
        "mean_r_per_trade": float(total_r / max(1, len(trades))),
        "total_structural_r": float(total_r),
    }


# ── Per-cluster driver ─────────────────────────────────────────────────────
def train_cluster(cid: int):
    name = CLUSTER_NAMES[cid]
    print(f"\n{'═'*66}\nC{cid} {name} — STRUCTURAL CURRICULUM\n{'═'*66}")

    df = (pd.read_csv(f"cluster_{cid}_data.csv", parse_dates=["time"])
            .sort_values("time").reset_index(drop=True))
    cutoff_ho = int(len(df) * 0.80)
    train_df  = df.iloc[:cutoff_ho].reset_index(drop=True)
    holdout_df = df.iloc[cutoff_ho:].reset_index(drop=True)

    val_cut = int(len(train_df) * (1 - VAL_FRAC))
    inner_train = train_df.iloc[:val_cut].reset_index(drop=True)
    inner_val   = train_df.iloc[val_cut:].reset_index(drop=True)

    print(f"  sizes: inner_train={len(inner_train):,}  val={len(inner_val):,}  "
          f"holdout={len(holdout_df):,}")

    # Normalization stats on inner_train only (no leak)
    feats = inner_train[FEATURE_COLS].fillna(0).to_numpy(dtype=np.float32)
    norm  = (feats.mean(axis=0), feats.std(axis=0) + 1e-6)

    # Pre-compute pivots on each slice (one-shot per slice)
    print("  pre-computing pivots on inner_train ...")
    piv_train   = precompute_pivots(
        inner_train["high"].to_numpy(np.float64),
        inner_train["low"] .to_numpy(np.float64),
        compute_atr(
            inner_train["high"].to_numpy(np.float64),
            inner_train["low"] .to_numpy(np.float64),
            inner_train["close"].to_numpy(np.float64), 14),
    )
    print("  pre-computing pivots on val ...")
    piv_val     = precompute_pivots(
        inner_val["high"].to_numpy(np.float64),
        inner_val["low"] .to_numpy(np.float64),
        compute_atr(
            inner_val["high"].to_numpy(np.float64),
            inner_val["low"] .to_numpy(np.float64),
            inner_val["close"].to_numpy(np.float64), 14),
    )
    print("  pre-computing pivots on holdout ...")
    piv_holdout = precompute_pivots(
        holdout_df["high"].to_numpy(np.float64),
        holdout_df["low"] .to_numpy(np.float64),
        compute_atr(
            holdout_df["high"].to_numpy(np.float64),
            holdout_df["low"] .to_numpy(np.float64),
            holdout_df["close"].to_numpy(np.float64), 14),
    )

    allowed = CLUSTER_ACTIONS[cid]
    names_map = {0:"FLAT",1:"LONG",2:"SHORT"}
    print(f"  allowed actions: {[names_map[a] for a in allowed]}")
    agent = RDQNAgent(allowed)
    best_val_r = -1e9
    best_state = None
    phase_log = []
    start_time = time.time()

    for phase_idx, phase_threshold in enumerate(PIVOT_CURRICULUM, start=1):
        print(f"\n  ▶ PHASE {phase_idx}/{len(PIVOT_CURRICULUM)}  "
              f"pivot_strength ≥ {phase_threshold}")

        env = StructuralEnv(inner_train, norm, piv_train, phase_threshold)
        state = env.reset()
        steps = 0
        stability = 0

        while True:
            eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * steps / EPS_DECAY_STEPS)
            a = agent.act(state, eps)
            s1, r, done, _ = env.step(a)
            agent.buffer.push(Transition(
                w=state[0].copy(), p=state[1].copy(),
                a=a, r=r,
                w1=s1[0].copy(), p1=s1[1].copy(), done=done,
            ))
            state = s1
            if done:
                state = env.reset()
            agent.train_step()
            steps += 1

            if steps % EVAL_EVERY == 0:
                val = eval_policy(agent, inner_val, norm, piv_val, phase_threshold)
                # Composite checkpoint metric — rewards real trades with reasonable
                # holds AND positive PnL. Penalises 1-bar flip exploits.
                bars_factor = min(val["avg_bars"] / 10.0, 1.0)       # saturates at 10 bars
                pnl_factor  = np.tanh(val["pnl"] / 500.0)            # ~+1 once PnL>500
                composite   = val["mean_r_per_trade"] * bars_factor + 0.5 * pnl_factor
                val_r = val["mean_r_per_trade"]

                ok = (val["avg_bars"] >= MIN_HOLD_BARS
                      and val["pnl"] > 0
                      and val["pf"] >= 1.1
                      and val["n"] >= MIN_TRADES_FOR_EVAL)
                stability = stability + 1 if ok else 0
                marker = "  ★ mastered" if stability >= MASTERY_STABILITY else ""
                print(f"    step {steps:>6}  eps={eps:.2f}  "
                      f"VAL meanR={val_r:+.3f}  PnL={val['pnl']:+6.0f}  "
                      f"PF={val['pf']:.2f}  WR={val['winrate']:.0%}  "
                      f"n={val['n']:>4}  bars={val['avg_bars']:.1f}  "
                      f"comp={composite:+.3f}  "
                      f"stab={stability}/{MASTERY_STABILITY}{marker}")

                # Save best by composite — not by mean_r_per_trade alone
                if composite > best_val_r:
                    best_val_r = composite
                    best_state = {k: v.detach().clone() for k, v in agent.q_net.state_dict().items()}

                if stability >= MASTERY_STABILITY:
                    break

            if steps >= STEPS_PER_PHASE_CAP:
                print(f"    ⚠ phase cap reached ({STEPS_PER_PHASE_CAP}), moving on")
                break

        phase_log.append({
            "phase": phase_idx,
            "threshold": phase_threshold,
            "steps": steps,
            "final_val": val,
            "mastered": stability >= MASTERY_STABILITY,
        })
        print(f"    [cluster elapsed: {(time.time()-start_time)/60:.1f} min]")

    # Load best val-checkpoint back before holdout eval
    if best_state is not None:
        agent.q_net.load_state_dict(best_state)
        print(f"\n  restored best-val checkpoint (val meanR = {best_val_r:+.3f})")

    # Honest holdout eval — uses full-difficulty reward (phase_threshold=0)
    print(f"\n  ◆ Honest holdout eval (unseen last 20%):")
    ho = eval_policy(agent, holdout_df, norm, piv_holdout, 0.0)
    mark = "✅" if ho["pnl"] > 0 and ho["pf"] >= 1.2 else "❌"
    print(f"    {mark}  n={ho['n']}  PnL={ho['pnl']:+.1f}  PF={ho['pf']:.2f}  "
          f"WR={ho['winrate']:.0%}  bars={ho['avg_bars']:.1f}  "
          f"structR/trade={ho['mean_r_per_trade']:+.3f}")

    # Save
    path = f"models/rdqn_struct_{cid}_{name}.pt"
    torch.save({
        "q_net": agent.q_net.state_dict(),
        "feature_cols": FEATURE_COLS,
        "norm_mu": norm[0].tolist(),
        "norm_sigma": norm[1].tolist(),
        "window_k": WINDOW_K,
        "allowed_actions": allowed,
        "best_val_reward_per_trade": best_val_r,
        "holdout": ho,
        "phase_log": phase_log,
    }, path)
    print(f"  saved {path}")
    return ho, phase_log


def main():
    print(f"Using device: {DEVICE}")
    print(f"LSTM hidden={LSTM_HIDDEN}  window_K={WINDOW_K}  episode_len={EPISODE_LEN}")
    print(f"Pivot curriculum: {PIVOT_CURRICULUM}")
    all_ho = {}
    for cid in CLUSTERS:
        ho, log = train_cluster(cid)
        all_ho[cid] = ho

    print(f"\n{'═'*66}\nALL CLUSTERS — HONEST HOLDOUT\n{'═'*66}")
    for cid, ho in all_ho.items():
        mark = "✅" if ho["pnl"] > 0 and ho["pf"] >= 1.2 else "❌"
        print(f"  {mark} C{cid} {CLUSTER_NAMES[cid]:<12}  "
              f"n={ho['n']:>5}  PnL={ho['pnl']:+8.1f}  "
              f"PF={ho['pf']:6.2f}  WR={ho['winrate']:.0%}  "
              f"structR/trade={ho['mean_r_per_trade']:+.3f}")

    with open("models/rdqn_struct_summary.json", "w") as f:
        json.dump({str(k): v for k, v in all_ho.items()}, f, indent=2, default=float)

if __name__ == "__main__":
    main()
