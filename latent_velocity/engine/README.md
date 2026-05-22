# CADENCE Engine: Mathematical Foundations

This document details the mathematical framework of the Latent ODE-VAE — the unified model that replaced the original β-VAE + Gaussian Process + Neural ODE three-stage pipeline.

---

## 1. Data Representation

### Clinical Frailty Index
Raw MHAS survey responses are encoded into a 34-dimensional deficit vector per visit.

```math
FI = \frac{1}{N} \sum_{i=1}^{N} d_i
```

Each patient's observation sequence is a set of `(x_t, t)` pairs at irregular survey waves. MHAS wave years (relative to 2001): `[0, 2, 11, 14, 17, 20]`.

The full input per frame is 37D: 34 deficit features + `[edad, sexo, educacion]`.

---

## 2. Encoder: RecognitionRNN

A backward masked GRU that reads the observation sequence in reverse chronological order, skipping unobserved waves.

```math
h_t = \text{GRUCell}([x_t \| t_{norm}],\ h_{t+1}) \cdot \mathbf{1}_{obs} + h_{t+1} \cdot (1 - \mathbf{1}_{obs})
```

- `t_norm = t / T_max` normalises times to `[0, 1]`
- The masked update ensures unobserved waves leave the hidden state unchanged
- Final hidden state `h₀` is projected to the posterior parameters:

```math
\mu = W_\mu h_0, \quad \log\sigma^2 = W_{\log\sigma} h_0
```

### Reparameterisation
```math
z_0 = \mu + \epsilon \odot \exp\!\left(\tfrac{1}{2}\log\sigma^2\right), \quad \epsilon \sim \mathcal{N}(0, I)
```

---

## 3. ODE Dynamics: LatentODEFunc

The latent state evolves according to a controlled Neural ODE:

```math
\frac{dz}{dt} = f_\theta(z,\, u)
```

`f_θ` is a 4-layer MLP with SiLU activations: `(8 + 7) → 128 → 256 → 128 → 8`.

The control vector `u` (7D lifestyle factors) is scaled by 10 before concatenation to amplify actionable signal:

```math
u_{scaled} = 10 \cdot u
```

The ODE is solved using RK4 on the fixed MHAS wave grid (fully batched across all patients in a mini-batch), replacing the per-patient GP fits of the legacy pipeline.

---

## 4. Decoder

Maps each latent state `z(t)` back to the 34-dimensional deficit space:

```math
\hat{x}(t) = g_\phi(z(t)) \in [0, 1]^{34}
```

`g_φ` uses LayerNorm + LeakyReLU and a final Sigmoid to keep outputs in the clinical deficit range.

---

## 5. Training Objective

### 5a. ELBO with Free Bits and β-Annealing

```math
\mathcal{L}_{ELBO} = \mathcal{L}_{recon} + \beta(t) \cdot \mathcal{L}_{KL}
```

**Reconstruction** (weighted MSE, observed frames only):

```math
\mathcal{L}_{recon} = \frac{1}{|\mathcal{T}_{obs}|} \sum_{t \in \mathcal{T}_{obs}} \sum_{j=1}^{34} w_j \left(\hat{x}_j(t) - x_j(t)\right)^2
```

**Inverse-variance feature weights** (same as the legacy β-VAE):

```math
w_j = \frac{1/\sigma_j^2}{\sum_{d=1}^{D} 1/\sigma_d^2} \times D
```

**KL divergence with free bits** (prevents posterior collapse):

```math
\mathcal{L}_{KL} = \sum_{k=1}^{8} \max\!\left(\delta,\ -\tfrac{1}{2}\left(1 + \log\sigma_k^2 - \mu_k^2 - \sigma_k^2\right)\right)
```

where `δ = 0.5` nats is the free bits threshold. All 8 latent dimensions are active when their KL exceeds `δ`.

**β-annealing** ramps from 0 → β over training epochs 20–80, allowing reconstruction to converge before regularisation pressure is applied:

```math
\beta(e) = \begin{cases} 0 & e < 20 \\ \beta_{target} \cdot \frac{e - 20}{60} & 20 \le e < 80 \\ \beta_{target} & e \ge 80 \end{cases}
```

### 5b. Cox Partial-Likelihood via RiskHead

A dedicated RiskHead MLP (`8 → 32 → 1`) predicts a scalar mortality risk score from `μ`. The Cox loss flows through the head, leaving the latent geometry free to encode clinical state:

```math
r_i = h_\psi(\mu_i) \in \mathbb{R}
```

```math
\mathcal{L}_{Cox} = -\frac{1}{N_{ev}} \sum_{i:\, ev_i=1} \left( r_i - \log \sum_{j:\, t_j \ge t_i} e^{r_j} \right)
```

### 5c. Total Loss

```math
\mathcal{L} = \mathcal{L}_{ELBO} + \lambda_{cox} \cdot \mathcal{L}_{Cox}
```

Default: `λ_cox = 0.15`, `β_target = 0.1`.

---

## 6. Velocity Extraction (MC Sampling)

Unlike the legacy GP pipeline, velocity is the ODE's native output — no derivative computation required.

For each patient:

1. Encode `{(x_t, t)}` → `μ, σ²`
2. Draw `n_mc = 30` samples: `z₀^{(s)} = μ + ε^{(s)} ⊙ σ`
3. Integrate each sample on a dense grid (step `Δt = 0.5` yr)
4. Velocity at each grid point: `v^{(s)}(t) = f_θ(z^{(s)}(t), u₀)`
5. Aggregate:

```math
\bar{v}(t) = \frac{1}{S} \sum_{s=1}^{S} v^{(s)}(t), \qquad \sigma_v(t) = \text{std}_s\!\left(\|v^{(s)}(t)\|\right)
```

The output CSV (`latent_velocity_trajectory_128.csv`) is backward-compatible with `clinical_validation.py`.

---

## 7. Clinical Validation

### Survival Analysis
Patients are phenotyped as Fast/Slow Agers via the signed frailty-velocity projection `v_frailty` (Ridge regression projecting `v(t)` onto the empirical FI gradient). Cox PH is fit with:
- `Fast_Ager_Flag` (Q1 vs Q4 phenotype)
- `baseline_age` (age confound)
- `mean_unc_z` (standardised MC uncertainty)

Current results: **HR = 4.77** for Fast vs Slow Ager (p < 0.001).

### Latent Velocity Magnitude

```math
\|v\|_2 = \sqrt{\sum_{k=1}^{8} v_k^2}
```

### Hazard Ratio

```math
HR = \exp(\hat{\beta}_{Fast\_Ager})
```

