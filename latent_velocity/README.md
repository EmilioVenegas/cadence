# CADENCE: Module Overview

This directory contains the core implementation of the CADENCE project. The architecture is split into two primary functional layers: the **Foundation Engine** and the **Longitudinal Dynamics Engine**.

## 📂 Subdirectories

### 1. [⚙️ engine/](engine/)
The **Foundation Engine** handles the mapping of clinical snapshots into the latent manifold and the initial extraction of individual velocities.
- **Role**: Data preprocessing, $\beta$-VAE training, Gaussian Process smoothing, and ground-truth clinical validation.
- **Key Resources**: [Engine Technical Documentation](engine/README.md)

### 2. [🤖 ode-digitaltwin/](ode-digitaltwin/)
The **Longitudinal Dynamics Engine** implements the predictive and counterfactual features of CADENCE using Neural ODEs.
- **Role**: Learning the continuous vector field, simulating clinical interventions (Digital Twins), and ranking recommended treatments.
- **Key Resources**: [Digital Twin Technical Documentation](ode-digitaltwin/README.md)

### 3. [📊 plots/](plots/)
Categorized visualization suite for interpreting manifold flow and patient outcomes.
- Includes subfolders for `tSNE`, `intervention_ranking`, `digital_twin`, `latent_space`, `streamplots`, and `gp_trajectories`.

### 4. [🧠 models/](models/)
Storage for trained network weights (`beta_vae_model.pth`, `neural_ode_model.pth`) and high-resolution trajectory datasets.

### 5. [💾 data/](data/)
Curated datasets, including the fractional Frailty Index matrix used for training.

---

## 🔗 Navigation
- **Top-Level**: See the main [Project README](../README.md) for clinical context and installation.
- **Mathematics**: Consult the READMEs within `engine/` and `ode-digitaltwin/` for detailed formulations.
