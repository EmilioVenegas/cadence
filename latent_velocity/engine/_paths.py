"""
Centralized path resolution for the CADENCE pipeline.

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
ODE_DIR    = ROOT / "ode-digitaltwin"

# Plot Subdirectories
TSNE_DIR    = PLOTS_DIR / "tSNE"
RANKING_DIR = PLOTS_DIR / "intervention_ranking"
TWIN_DIR    = PLOTS_DIR / "digital_twin"
LATENT_DIR  = PLOTS_DIR / "latent_space"
STREAM_DIR  = PLOTS_DIR / "streamplots"
HEATMAP_DIR = PLOTS_DIR / "heatmaps"
GP_DIR      = PLOTS_DIR / "gp_trajectories"
