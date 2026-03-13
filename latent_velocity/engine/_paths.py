"""
Centralized path resolution for the LAVA pipeline.

All directories are computed relative to this file's location,
so scripts work regardless of the caller's working directory.
"""
from pathlib import Path

# latent_velocity/engine/_paths.py  →  ROOT = latent_velocity/
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"
PLOTS_DIR  = ROOT / "plots"
ENGINE_DIR = ROOT / "engine"
