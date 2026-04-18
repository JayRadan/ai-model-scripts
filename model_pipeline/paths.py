"""Centralised data + output paths so the pipeline scripts can run from
anywhere without breaking when files are moved.

All data lives in <project_root>/data/ and all model artifacts in
<project_root>/models/.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
LIVE_DIR     = PROJECT_ROOT / "live_deployment"

# Make sure callers can still chdir into PROJECT_ROOT and use these
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def data(name: str) -> str:
    return str(DATA_DIR / name)

def model(name: str) -> str:
    return str(MODELS_DIR / name)
