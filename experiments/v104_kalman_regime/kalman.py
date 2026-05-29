"""
Kalman Regime Features (Kinematic State-Space) — Python port of the Pine v6
indicator "Kalman Regime Features (Kinematic State-Space)".

Faithful 1:1 port of the constant-velocity Kalman filter on price. On an M1
series we feed `close` directly each bar (the Pine `useLTF` lower-timeframe
path is OFF — useless on a 1m chart per the indicator's own tooltip).

Outputs (all causal / no look-ahead):
  kf_line        : filtered price  (Pine kf.p)           — price-space "trend line"
  kf_v           : velocity        (Pine kf.v)
  committed_dir  : regime color, +1 if kf_v >= 0 else -1 (Pine `trendUp = kf.v >= 0`)
  f_velPct       : kf.v / max(|kf.p|,eps) * 100          — trend speed, % of price/bar
  f_velSignif    : kf.v / sqrt(max(P11,eps))             — velocity t-stat
  f_innovZ       : innov / sqrt(max(S,eps))              — standardized surprise
  f_volState     : sqrt(max(R,0)) / max(kf.p,eps)        — scale-free vol state
  f_accel        : change(kf.v)                          — curvature / momentum
  f_velRaw       : kf.v                                  — raw velocity

The trade logic that consumes this (see 01_train_test_flip.py) is COLOR-FLIP:
  red -> green (committed_dir -1 -> +1)  => BUY
  green -> red (committed_dir +1 -> -1)  => SELL
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    """Pine ta.ema: alpha = 2/(n+1), seeded with the first value."""
    a = 2.0 / (n + 1.0)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1.0 - a) * out[i - 1]
    return out


def compute_kalman(
    df: pd.DataFrame,
    *,
    q: float = 0.01,        # process noise (responsiveness)
    r_mult: float = 1.0,    # measurement-noise multiplier
    r_len: int = 50,        # noise estimation length
    dt: float = 1.0,        # time step
    mintick: float = 0.01,  # XAU min tick (mintick^2 noise floor)
    src_col: str = "close",
) -> pd.DataFrame:
    """Run the kinematic-state-space Kalman filter over df[src_col].

    df must have columns: time, open, high, low, close (tick_volume optional).
    Returns a copy with the kf_* / committed_dir / f_* columns added.
    """
    z = df[src_col].to_numpy(dtype=np.float64)
    n = len(z)

    # ── Adaptive measurement noise R ──────────────────────────────────────
    # ret = ta.change(src); varR = ta.ema(ret^2, rLen); R = rMult*max(varR, mintick^2)
    ret = np.empty(n, dtype=np.float64)
    ret[0] = 0.0                      # ta.change is na on bar 0 -> nz() = 0
    ret[1:] = np.diff(z)
    varR = _ema(ret * ret, r_len)
    floor = mintick * mintick
    R = r_mult * np.maximum(varR, floor)

    # ── State (KFState.new(na, 0, 1e6, 0, 0, 1e6, 0, 1)) ──────────────────
    p = np.nan
    v = 0.0
    P00, P01, P10, P11 = 1e6, 0.0, 0.0, 1e6

    kf_line = np.empty(n, dtype=np.float64)
    kf_v = np.empty(n, dtype=np.float64)
    P11_arr = np.empty(n, dtype=np.float64)
    innov_arr = np.empty(n, dtype=np.float64)
    S_arr = np.empty(n, dtype=np.float64)

    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt

    for i in range(n):
        zi = z[i]
        Ri = R[i]
        if np.isnan(p):
            p = zi

        # ── Predict (constant-velocity kinematics) ──
        pPred = p + dt * v
        vPred = v
        q00 = q * dt4 / 4.0
        q01 = q * dt3 / 2.0
        q11 = q * dt2
        M00 = P00 + dt * P10
        M01 = P01 + dt * P11
        M10 = P10
        M11 = P11
        Pp00 = M00 + dt * M01 + q00
        Pp01 = M01 + q01
        Pp10 = M10 + dt * M11 + q01
        Pp11 = M11 + q11

        # ── Update ──
        Sden = Pp00 + Ri
        K0 = Pp00 / Sden
        K1 = Pp10 / Sden
        y = zi - pPred
        p = pPred + K0 * y
        v = vPred + K1 * y
        P00 = (1.0 - K0) * Pp00
        P01 = (1.0 - K0) * Pp01
        P10 = Pp10 - K1 * Pp00
        P11 = Pp11 - K1 * Pp01

        kf_line[i] = p
        kf_v[i] = v
        P11_arr[i] = P11
        innov_arr[i] = y
        S_arr[i] = Sden

    # ── Derived features (scale-free, causal) ─────────────────────────────
    eps = 1e-10
    f_velPct = kf_v / np.maximum(np.abs(kf_line), eps) * 100.0
    f_velSignif = kf_v / np.sqrt(np.maximum(P11_arr, eps))
    f_innovZ = innov_arr / np.sqrt(np.maximum(S_arr, eps))
    f_volState = np.sqrt(np.maximum(R, 0.0)) / np.maximum(kf_line, eps)
    f_accel = np.empty(n, dtype=np.float64)
    f_accel[0] = 0.0
    f_accel[1:] = np.diff(kf_v)
    f_velRaw = kf_v

    committed = np.where(kf_v >= 0.0, 1, -1).astype(np.int8)

    out = df.copy()
    out["kf_line"] = kf_line
    out["kf_v"] = kf_v
    out["committed_dir"] = committed
    out["f_velPct"] = f_velPct
    out["f_velSignif"] = f_velSignif
    out["f_innovZ"] = f_innovZ
    out["f_volState"] = f_volState
    out["f_accel"] = f_accel
    out["f_velRaw"] = f_velRaw
    return out


# Kalman-native feature columns (analogous to TFK's force/velocity/x_est/...).
KALMAN_FEATS = ["f_velPct", "f_velSignif", "f_innovZ", "f_volState", "f_accel", "f_velRaw"]
