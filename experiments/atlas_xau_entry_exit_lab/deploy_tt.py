"""Deploy time-boxed-patience trail to the atlas_xau bundle: add tight_after=30,
tight_trail_R=0.75 (conservative plateau pick), bump version. Model/threshold/labels
UNCHANGED (label-matched retrain added nothing). Backs up the current bundle."""
import pickle, shutil
from pathlib import Path

MODELS = Path("/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine/models")
f = MODELS / "atlas_xau_validated.pkl"
bak = MODELS / "atlas_xau_validated.pkl.bak_pre_tt_2026-07-02"
p = pickle.load(open(f, "rb"))
print("before:", p["version"], {k: p.get(k) for k in ("sl_R", "trail_R", "maxh", "threshold")})
if not bak.exists():
    shutil.copy(f, bak); print("backup ->", bak.name)
p["tight_after"] = 30
p["tight_trail_R"] = 0.75
p["version"] = "edge_pullback_v3_tt30_atlas_xau"
p["recipe"] = (p.get("recipe", "") +
               " | tt-trail 2026-07-02: server trail tightens 2->0.75*ATR after 30 bars"
               " (experiments/atlas_xau_entry_exit_lab, dev 8/9 holdout 3/3)")
pickle.dump(p, open(f, "wb"))
q = pickle.load(open(f, "rb"))
print("after: ", q["version"], "tight_after", q["tight_after"], "tight_trail_R", q["tight_trail_R"])
