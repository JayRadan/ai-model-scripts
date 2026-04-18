from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LIVE_DIR = PROJECT_ROOT / "live_deployment"

def data(name): return str(DATA_DIR / name)
def model(name): return str(MODELS_DIR / name)
