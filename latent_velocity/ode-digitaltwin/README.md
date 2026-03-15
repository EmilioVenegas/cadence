# 🤖 LAVA Digital Twin: Longitudinal Dynamics & Interventions

This module implements the **Longitudinal Simulation Engine** using Exogenous Control Neural ODEs.

## 📊 Performance Benchmarks
The High-Momentum Neural ODE is now the **default production model** in the LAVA pipeline.
- **Trajectory Accuracy**: **79.0% R2** (3-year forecasts).
- **Directional Alignment**: **0.46 Velocity Magnitude Correlation** and **0.39 Cosine Similarity**.
- **Inference Speed**: < 2ms per patient simulation.

## 1. Exogenous Control Neural ODE
Instead of a simple autonomous system, LAVA's dynamics are governed by an **Exogenous Control ODE**. The velocity of biological decline depends on the current state $z$ and an external control vector $u$ (lifestyle/clinical factors).

- **The Flow Function**:
```math
\frac{dz}{dt} = f_\theta(z, t, u)
```
Where $f_\theta$ is a high-capacity MLP (SiLU activations) trained to map the 8D latent state $z$ and 7D control vector $u$ to the instantaneous velocity vector.

- **Feature Scaling**: To ensure the model is sensitive to actionable lifestyle changes, the control vector $u$ is scaled by a factor of 10 before entering the network.
```math
u_{scaled} = 10 \cdot u
```

## 2. Biological Washout (Temporal Realism)
Clinically, lifestyle changes (e.g., quitting smoking) do not result in "instantaneous healing." The physiological system has momentum. We model this using a **Biological Washout** exponential decay for the control vector $u(t)$.

```math
u(t) = u_{target} + (u_{start} - u_{target}) \cdot e^{-k \cdot t}
```

- **Mathematical Depth**: At $t=0$, $u(0) = u_{start}$, which ensures that the Digital Twin and the Baseline trajectory start with the **exact same velocity** at the moment of intervention.
- **Divergence**: As $t \to \infty$, $u(t) \to u_{target}$, allowing the new "cured" vector field to gradually take over.

## 3. Training Objective (Multi-Objective ODE)
The ODE is trained to minimize the error between the predicted latent state at time $T$ and the true observed state, while constrained by ground-truth analytical velocities.

- **Terminal State Loss**:
```math
\mathcal{L}_{mse} = \|z_{pred}(T) - z_{observed}(T)\|^2
```

- **Velocity Regularization (Momentum Penalty)**: To ensure the ODE follows the actual biological "currents," we penalize the difference between the network's predicted $t=0$ velocity and the analytical GP velocity $v_0$.
```math
\mathcal{L}_{reg} = \|f_\theta(z_0, 0, u_0) - v_0\|^2
```

- **Total Loss**:
```math
\mathcal{L} = \mathcal{L}_{mse} + \lambda \mathcal{L}_{reg}
```

## 4. Intervention Ranking Metrics
Scenarios are ranked using the **Area Under the Curve (AUC)** of the velocity magnitude over a 5-year forecast window.

```math
AUC = \int_{0}^{5} \|v(t)\|_2 \, dt \approx \text{trapz}(\text{velocity\_magnitude}, t)
```

- **Improvement Metric**:
```math
\Delta\% = \frac{AUC_{baseline} - AUC_{twin}}{AUC_{baseline}} \times 100
```

## 5. Clinical Safety (Ghost Twin Guardrail)
To prevent the model from generating "Ghost Twins" (hallucinating trajectories in regions of the manifold where no similar patients exist), we calculate the **Mahalanobis Distance** between the simulated patient and the empirical cohort.

```math
D_M(z, u) = \sqrt{(z - \mu)^T S^{-1} (z - \mu)}
```
Where $\mu$ and $S$ are the mean and covariance of the latent states of patients who share the target clinical profile $u_{twin}$. Scenarios with $D_M > 3.0$ are flagged as low-confidence/OOD.
