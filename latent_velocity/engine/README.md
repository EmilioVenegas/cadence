# 🌋 LAVA Engine: Mathematical Foundations

This document details the mathematical framework underpinning the Latent Aging Velocity Atlas (LAVA).

## 1. Clinical Indexing (The Frailty Index)
LAVA begins by encoding raw longitudinal survey data into a high-dimensional deficit space.
- **Frailty Index (FI)**: The unweighted average of $N$ accumulated health deficits.
```math
  FI = \frac{1}{N} \sum_{i=1}^{N} d_i
```

## 2. Generative Manifold Learning ($\beta$-VAE)
We map the deficit space into a low-dimensional latent manifold using a $\beta$-Variational Autoencoder.

- **Reparameterization Trick**: Allows backpropagation through stochastic nodes.
```math
  z = \mu + \epsilon \odot \exp\left(\frac{1}{2} \log \sigma^2\right)
```

- **Feature-Weighted $\beta$-VAE Loss**:
```math
\mathcal{L} = \sum_{j=1}^{D} \left( w_j (x_j - \hat{x}_j)^2 \right) - \frac{\beta}{2} \sum_{k=1}^{K} \left( 1 + \log(\sigma_k^2) - \mu_k^2 - \sigma_k^2 \right)
```

- **Inverse-Variance Feature Weights**: Used to prioritize subtle signals (e.g., cognition) over high-variance comorbidity counts.
```math
w_j = \frac{\frac{1}{\sigma_j^2}}{\sum_{d=1}^{D} \frac{1}{\sigma_d^2}} \times D
```

## 3. Longitudinal Velocity Inference (Gaussian Processes)
We fit a Gaussian Process to each patient's latent history to extract continuous-time derivatives.

- **GP Kernel (Constant $\times$ RBF + White Noise)**:
```math
k(t, t') = \sigma_f^2 \exp\left(-\frac{(t - t')^2}{2l^2}\right) + \sigma_n^2 \delta_{t, t'}
```

- **Posterior Mean Trajectory**:
```math
\bar{z}(t^*) = K(t^*, t_{obs}) (K(t_{obs}, t_{obs}) + \sigma_n^2 I)^{-1} y_{obs}
```

- **RBF Kernel Derivative**:
```math
K'(t^*, t_{obs}) = -\frac{t^* - t_{obs}}{l^2} k(t^*, t_{obs})
```

- **Analytic Latent Velocity**: The exact temporal derivative of the latent state.
```math
v(t^*) = \frac{\partial \bar{z}(t^*)}{\partial t^*} = K'(t^*, t_{obs}) \alpha
```
Where $\alpha = (K(t_{obs}, t_{obs}) + \sigma_n^2 I)^{-1} y_{obs}$.

## 4. Clinical Validation & Survival
- **Latent Velocity Magnitude**:
```math
v_{mag} = \sqrt{\sum_{k} v_k^2}
```
- **Empirical Clinical Velocity**:
```math
v_{emp} = \frac{FI_{t+1} - FI_t}{t_{t+1} - t_t}
```
- **Hazard Ratio (Cox Proportional Hazards)**:
```math
HR = \exp(\beta_{coef})
```

## 5. Vector Field Kinematics
- **Discrete Acceleration**:
```math
a_k(t) = \frac{v_k(t) - v_k(t-\Delta t)}{\Delta t}
```
- **Gaussian Weighted Spatial Interpolation**:
```math
\vec{V}_{interp} = \frac{\sum_{i} \exp\left(-\frac{d_i^2}{2\sigma^2}\right) \vec{v}_i}{\sum_{i} \exp\left(-\frac{d_i^2}{2\sigma^2}\right)}
```
