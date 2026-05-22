# CADENCE Digital Twin: Counterfactual Simulation & Intervention Ranking

This module implements the **Longitudinal Simulation Engine** using the Latent ODE-VAE model. It encodes a patient's full clinical history into a latent initial condition, then integrates the ODE forward under counterfactual lifestyle controls to rank interventions.

---

## Architecture Overview

The digital twin pipeline runs in three stages:

1. **Encode** — The patient's full observation sequence `{(x_t, t)}` is passed through the RecognitionRNN to produce `z₀ ~ q(z | x)`. All available visits are used, not just the latest, giving the best posterior estimate.

2. **Integrate** — The latent ODE `dz/dt = f_θ(z, u)` is solved forward from `z₀` under a given control vector `u(t)`. Baseline and each counterfactual intervention are solved as separate forward passes.

3. **Rank** — Interventions are ranked by their 5-year AUC velocity reduction relative to the baseline twin.

`digital_twin.py` auto-detects the model type at load time: if `models/latent_ode_model.pth` exists it uses the Latent ODE-VAE; otherwise it falls back to the legacy β-VAE + standalone Neural ODE.

---

## 1. ODE Dynamics

The same `LatentODEFunc` used during training drives all simulations:

```math
\frac{dz}{dt} = f_\theta(z,\, u)
```

`f_θ` is a 4-layer MLP (SiLU activations): `(8 + 7) → 128 → 256 → 128 → 8`.

The control vector `u ∈ ℝ⁷` encodes:

| Index | Variable | Intervention target |
|---|---|---|
| 0 | `tabaco` | Quit smoking → 0 |
| 1 | `bmi_imp` | Normalise BMI → 0 |
| 2 | `ejer_3_por_sem` | Add exercise → 0 |
| 3 | `hipertension` | Control hypertension → 0 |
| 4 | `diabetes` | Manage diabetes → 0 |
| 5 | `alcohol` | Reduce alcohol → 0 |
| 6 | `social_isolation` | Social engagement → 0 |

`social_isolation` is derived at inference time as `1 − max(asiste_club, voluntario)`.

---

## 2. Biological Washout (Temporal Realism)

Lifestyle changes do not produce instantaneous physiological effects. Transitions are modelled with an exponential washout:

```math
u(t) = u_{target} + (u_{start} - u_{target}) \cdot e^{-k \cdot t}
```

- At `t = 0`: `u(0) = u_start` — the baseline and twin start with identical velocity.
- As `t → ∞`: `u(t) → u_target` — the intervention gradually takes over.
- Default washout rate: `k = 0.5` yr⁻¹ (half-life ≈ 1.4 years).

This prevents biologically impossible "instantaneous healing" artefacts in counterfactual trajectories.

---

## 3. Intervention Ranking

Each scenario is scored by the 5-year AUC of the velocity magnitude:

```math
AUC = \int_{0}^{5} \|v(t)\|_2\, dt \approx \text{trapz}(\|v\|, t)
```

The improvement metric relative to the no-intervention baseline:

```math
\Delta\% = \frac{AUC_{baseline} - AUC_{twin}}{AUC_{baseline}} \times 100
```

Interventions are ranked from highest to lowest `Δ%`.

---

## 4. Ghost Twin Guardrail (Clinical Safety)

To prevent the model from simulating trajectories in regions of the latent manifold with no real patient support, each counterfactual is checked against the empirical cohort distribution.

**Mahalanobis distance** between the patient's encoded `z₀` and the sub-cohort of patients sharing the target clinical profile `u_twin`:

```math
D_M(z_0) = \sqrt{(z_0 - \bar{z})^\top S^{-1} (z_0 - \bar{z})}
```

where `z̄` and `S` are the empirical mean and covariance of matching patients in the encoded cohort.

- `D_M > 3.0` → simulation flagged as low-confidence / out-of-distribution.
- The guardrail operates in the Latent ODE-VAE latent space (posterior mean `μ`), computed via the full-sequence encoder for both the patient and each cohort member.

---

## 5. RiskHead (Training Only)

During training, a dedicated `RiskHead` MLP (`8 → 32 → 1`) predicts scalar mortality risk from `μ` and receives the Cox partial-likelihood gradient. This decouples survival supervision from the latent geometry:

- **`μ`** encodes clinical state faithfully — good for ODE trajectory fidelity and counterfactual realism.
- **`RiskHead(μ)`** encodes mortality risk — used only for training-time Cox loss, not at inference.

At inference (digital twin simulation), only the ODE dynamics `f_θ(z, u)` and the decoder `g_φ(z)` are used. The RiskHead is ignored.

---

## 6. Server API Endpoints

The FastAPI backend (`engine/server.py`) exposes:

| Endpoint | Method | Description |
|---|---|---|
| `/api/patients` | GET | List all patient IDs |
| `/api/rank/{cunicah}/{np}` | GET | Rank interventions for a stored patient |
| `/api/rank/live` | POST | Rank interventions for a manually specified patient |
| `/api/summary` | POST | LLM-generated clinical action plan |

All ranking endpoints call `rank_interventions()` in `digital_twin.py`, which runs the full encode → integrate → rank pipeline.
