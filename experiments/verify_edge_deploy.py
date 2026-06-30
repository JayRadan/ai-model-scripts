"""Pre-push verification: load the new bundles via the real loader and simulate
live decide_entry / decide_exit for hermes_dji (engine hermes) and atlas_xau
(engine atlas). Confirms routing -> edge_pullback and clean responses. No network."""
import sys, pickle
from pathlib import Path
import pandas as pd
SRV = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(SRV))
from decision_engine import loader, decide_hermes, decide_atlas
from decision_engine.configs import REGISTRY

class Pos:  # minimal position stub for decide_exit
    direction = 1; bars_held = 10

def load_bars(pq, ncol="time"):
    d = pd.read_parquet(pq).rename(columns={"timestamp": "time"})
    if "time" not in d.columns: d = d.rename(columns={[c for c in d.columns if "time" in c.lower()][0]: "time"})
    d["time"] = pd.to_datetime(d["time"]); d = d.sort_values("time").drop_duplicates("time")
    if "tick_volume" not in d.columns: d["tick_volume"] = d.get("volume", 0)
    return d.tail(9000).reset_index(drop=True)

for prod, pq, mod in [
    ("hermes_dji", "/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet", decide_hermes),
    ("atlas_xau",  "/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet",  decide_atlas),
]:
    print("="*70); print(prod)
    cfg = REGISTRY[prod] if not callable(REGISTRY[prod]) else REGISTRY[prod]
    bundle = loader.load_bundle(cfg)
    print(f"  loaded bundle: version={bundle.payload.get('version')} thr={bundle.payload.get('threshold')} "
          f"engine={getattr(cfg,'engine_type',None)} max_conc={getattr(cfg,'max_concurrent',None)}")
    bars = load_bars(pq)
    print(f"  bars: {len(bars)}  {bars.time.iloc[0]} -> {bars.time.iloc[-1]}")
    r = mod.decide_entry(bars, bundle, account="verify", open_positions=[])
    print(f"  ENTRY action={r.get('action')} dir={r.get('direction')} reason={r.get('reason')}")
    print(f"        sl={r.get('sl_atr_mult')} trail={r.get('trail_atr_mult')} maxh={r.get('max_hold_bars')} "
          f"engine={r.get('trace',{}).get('engine')}")
    rx = mod.decide_exit(bars, Pos(), bundle, account="verify")
    print(f"  EXIT  action={rx.get('action')} reason={rx.get('reason')} engine={rx.get('trace',{}).get('engine')}")
    assert r.get('trace', {}).get('engine') == 'edge_pullback', "ROUTING FAILED — not edge_pullback!"
    assert r.get('action') in ('open', 'hold'), "bad action"
    assert rx.get('trace', {}).get('engine') == 'edge_pullback', "EXIT routing failed!"
    print("  ✓ routed to edge_pullback, responses valid")
print("="*70); print("ALL VERIFICATIONS PASSED")
