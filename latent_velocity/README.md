# CADENCE: Module Overview

This directory contains the full implementation of CADENCE. The system is built around a single unified **Latent ODE-VAE** trained end-to-end — there is no separate VAE pretraining or GP smoothing step.

## Architecture at a Glance

```
MHAS visits {(x_t, s, t)}
        │
        ▼
  RecognitionRNN          ← backward masked GRU; handles irregular spacing
  (37D input per wave)
        │
        ▼ q(z₀ | x) = N(μ, σ²) ∈ R⁸
        │
        ▼
  LatentODEFunc           ← dz/dt = f_θ(z, 10u), 4-layer MLP with SiLU
  (RK4 integration)
        │
        ├─────────────────────────────────────┐
        ▼                                     ▼
  Decoder g_φ                          RiskHead r_ψ
  z(t) → x̂(t) ∈ [0,1]³⁴              μ → scalar Cox risk
  (reconstruction)                     (survival supervision)
        │
        ▼
  Velocity v(t) = f_θ(z(t), 10u)      ← ODE right-hand side; native output
```

The joint training objective is ELBO + Cox partial-likelihood:

```
L = L_recon  +  β(e) · L_KL  +  λ_cox · L_Cox
```

with β annealed 0 → 0.1 over epochs 20–80, per-dimension free bits (δ = 0.5 nats), and λ_cox = 0.15.

## Subdirectories

### `engine/`

End-to-end pipeline from raw MHAS data to trained model and validation results.

| Script | Role |
|---|---|
| `prepare_frailty_data.py` | MHAS preprocessing → `frailty_index_data.csv` (34 deficits, MICE imputation) |
| `train_latent_ode.py` | Train the unified Latent ODE-VAE |
| `extract_latent_ode_velocity.py` | 30-sample MC ODE integration → dense velocity trajectories |
| `clinical_validation.py` | Cox PH, LMM, Kaplan–Meier curves, velocity-domain heatmap |
| `benchmark.py` | B1–B4 vs CADENCE Harrell C-index table |
| `server.py` | FastAPI backend (per-patient inference, Digital Twin, LLM action plans) |

### `ode-digitaltwin/`

Counterfactual simulation module.

| Script | Role |
|---|---|
| `digital_twin.py` | Loads the trained Latent ODE-VAE, simulates control trajectories with biological washout, ranks interventions by 5-year AUC velocity reduction. Includes Ghost Twin Guardrail (Mahalanobis distance, threshold D_M > 3.0). |

### `app_ui/`

React 19 + Vite dashboard. Calls the FastAPI backend for per-patient inference and renders intervention trajectories, the intervention ranking bar chart, and an LLM-generated action plan.

### `models/`

Frozen model weights and trajectory CSVs (not committed). Expected files:
- `latent_ode_model.pt` — trained Latent ODE-VAE
- `latent_velocity_trajectory_128.csv` — MC velocity dataset

### `data/`

Curated MHAS datasets. Key file: `frailty_index_data.csv`.

### `plots/`

Visualization outputs organized by type: `tSNE/`, `intervention_ranking/`, `digital_twin/`, `latent_space/`, `streamplots/`, `heatmaps/`.

### `paper/`

Quarto manuscript (`paper_jbi.qmd`), bibliography, and generated figures.

---

## Navigation

- **Top-level**: See the main [Project README](../README.md) for clinical context, installation, and usage commands.
