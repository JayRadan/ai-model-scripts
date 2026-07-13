"""One-window debug of hedge v3: PH distribution + train nets per grid point."""
src = open("run_lab_hedge_v3.py").read().split("WINS = []")[0].replace("cache=True", "cache=False")
exec(src)
tr_s, te_s = pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01")
trm = (ct >= tr_s) & (ct < te_s)
tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
mg = XGBRegressor(**XGBQ); mg.fit(Xc[tix_f], pnlB[tix_f])
p = mg.predict(Xc).astype(np.float64)
thr = np.quantile(p[tix], 0.9)
tkk = np.where(trm & (p >= thr))[0]
tk_tr = take(tkk[np.argsort(ebB[tkk])].astype(np.int64), ebB, xitB, COOLDOWN)
Xtr, ytr = [], []
for SN, HH, XSs in ((SN2, H2, XS2), (SN30, H30, XS30)):
    sb = SN[0]; hp, ha, hv = HH
    s = tk_tr[(sb[tk_tr] >= 0) & (hv[tk_tr] == 1)]
    Xtr.append(XSs[s]); ytr.append(hp[s] - 2.0 / np.maximum(ha[s], 1e-9))
mh = XGBRegressor(**XGBH); mh.fit(np.concatenate(Xtr), np.concatenate(ytr))
print("train snapshot label mean:", round(float(np.concatenate(ytr).mean()), 3),
      "frac>0:", round(float((np.concatenate(ytr) > 0).mean()), 3))
TK, BR, ST = perbar_stats(tk_tr.astype(np.int64), ebB, xitB, idx, dirs, O, C, atr, MIN_HELD)
Xpb = np.concatenate([FEAT_ALL[BR], ST.astype(np.float32)], axis=1)
PH = mh.predict(np.nan_to_num(Xpb, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.float64)
print("PH pct 50/90/99/99.9:", np.round(np.percentile(PH, [50, 90, 99, 99.9]), 3),
      " frac>=0.1:", round(float((PH >= 0.1).mean()), 4))
rs = np.zeros(len(tk_tr), np.int64); re = np.zeros(len(tk_tr), np.int64); pos = 0
for q, k in enumerate(tk_tr):
    rs[q] = pos
    while pos < len(TK) and TK[pos] == k: pos += 1
    re[q] = pos
for bm in (0, 1):
    for ton, toff in GRID:
        nt, ne = hedge_machine(tk_tr.astype(np.int64), rs, re, BR, PH, ebB, xitB,
                               idx, dirs, O, H, L, C, atr, n, ton, toff, bm, 2.0)
        print(f"bm={bm} tau_on={ton} tau_off={toff}: episodes={ne} trainNet={nt:+.0f}R")
