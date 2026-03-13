# 🌋 LAVA: Latent Aging Velocity Atlas
## Continuous-Time Mapping of Multi-Domain Biological Decline

LAVA is a production-grade diagnostic framework designed to disentangle the complex, non-linear trajectories of human aging. By projecting multi-domain clinical deficits into a continuous latent manifold, LAVA moves beyond static "snapshots" of health to measure the instantaneous velocity of biological aging.

## 🚀 Pipeline Overview

The LAVA framework operates through five distinct modular stages:

### 1. Data Preparation (`prepare_frailty_data.py`)
Encodes raw longitudinal survey data (MHAS) into a high-dimensional deficit space.
- **Domains**: Clinical, Functionality, Mental Health, Cognition, and Biometrics.
- **Processing**: Iterative imputation of missing values and standardization of chronological time.

**Mathematical Foundation (Clinical Indexing):**
- **Frailty Index (FI)**: The unweighted average of $N$ accumulated health deficits.
```math
  FI = \frac{1}{N} \sum_{i=1}^{N} d_i
```


### 2. Generative Manifold Learning (`train_vae.py`)
Maps the deficit space into a low-dimensional latent manifold using a $\beta$-Variational Autoencoder ($\beta$-VAE).
- **Disentanglement**: Optimized $\beta$-annealing to separate generic frailty from domain-specific signals.
- **Architecture**: Deep MLP with feature-weighted reconstruction losses.

**Mathematical Foundation ($\beta$-VAE Architecture):**
- **Reparameterization Trick**: Sampling from the latent space allowing for backpropagation.
```math
  z = \mu + \epsilon \odot \exp\left(\frac{1}{2} \log \sigma^2\right)
```
- **Feature-Weighted $\beta$-VAE Loss**: Combining Inverse-Variance weighted Mean Squared Error (MSE) and $\beta$-scaled Kullback-Leibler (KL) Divergence.
```math
\mathcal{L} = \sum_{j=1}^{D} \left( w_j (x_j - \hat{x}_j)^2 \right) - \frac{\beta}{2} \sum_{k=1}^{K} \left( 1 + \log(\sigma_k^2) - \mu_k^2 - \sigma_k^2 \right)
```
- **Inverse-Variance Feature Weights**: Normalized weights to penalize features with low variance.
```math
w_j = \frac{\frac{1}{\sigma_j^2}}{\sum_{d=1}^{D} \frac{1}{\sigma_d^2}} \times D
```


### 3. Longitudinal Velocity Inference (`extract_velocity.py`)
Infers continuous-time trajectories and their derivatives for each individual.
- **GP Smoothing**: Gaussian Process regression over longitudinal latent states.
- **Analytic Velocity**: Extraction of exact temporal derivatives ($\frac{dz}{dt}$) for precise measurement of decline speed.

**Mathematical Foundation (Gaussian Process & Analytic Derivatives):**
- **GP Kernel (Constant $\times$ RBF + White Noise)**:
```math
k(t, t') = \sigma_f^2 \exp\left(-\frac{(t - t')^2}{2l^2}\right) + \sigma_n^2 \delta_{t, t'}
```
- **Posterior Mean Trajectory**: The smoothed latent state at a dense grid of points $t^*$.
```math
\bar{z}(t^*) = K(t^*, t_{obs}) (K(t_{obs}, t_{obs}) + \sigma_n^2 I)^{-1} y_{obs}
```
- **GP Weights ($\alpha$)**:
```math
\alpha = (K(t_{obs}, t_{obs}) + \sigma_n^2 I)^{-1} y_{obs}
```
- **RBF Kernel Derivative**: The exact partial derivative of the cross-covariance matrix with respect to the prediction time $t^*$.
```math
K'(t^*, t_{obs}) = -\frac{t^* - t_{obs}}{l^2} k(t^*, t_{obs})
```
- **Analytic Latent Velocity**: The exact derivative of the posterior mean trajectory.
```math
v(t^*) = \frac{\partial \bar{z}(t^*)}{\partial t^*} = K'(t^*, t_{obs}) \alpha
```

### 4. Clinical Validation 
`clinical_validation.py`

Rigorous statistical verification of latent velocities.
- **Survival Analysis**: Cox Proportional Hazards modeling to test velocity as a predictor of mortality.
- **Mixed Models**: Linear Mixed Models to correlate latent derivatives with longitudinal clinical domain progression.

**Mathematical Foundation (Validation & Survival):**
- **Latent Velocity Magnitude**: Calculating the Euclidean norm of the latent velocity vector.
```math
v_{mag} = \sqrt{\sum_{k} v_k^2}
```
- **Empirical Clinical Velocity**: Forward finite difference of the Frailty Index over time.
```math
v_{emp} = \frac{FI_{t+1} - FI_t}{t_{t+1} - t_t}
```
- **Z-Score Normalization**: Used for standardizing age and education.
```math
z = \frac{x - \mu}{\sigma}
```
- **Hazard Ratio (Cox Proportional Hazards)**: Exponentiating the coefficient for the "Fast Ager" flag.
```math
HR = \exp(\beta_{Fast\_Ager})
```

### 5. Visual Diagnostics (`visualize_*.py`)
Generates high-resolution visualizations for model interpretability.
- **Vector Fields**: Latent flow streamplots (e.g., Physical vs. Cognitive disentanglement).
- **Phase Portraits**: Visualizing biological "momentum" by plotting state vs. velocity.

**Mathematical Foundation (Vector Field Kinematics):**
- **Discrete Acceleration** (in `visualize_streamplot.py`): Backward finite difference of the analytic velocity.
```math
a_k(t) = \frac{v_k(t) - v_k(t-\Delta t)}{\Delta t}
```
- **Gaussian Weighted Spatial Interpolation**: Computing localized vector streams using nearest neighbors in the 2D plane based on distance $d_i$.
```math
\vec{V}_{interp} = \frac{\sum_{i} \exp\left(-\frac{d_i^2}{2\sigma^2}\right) \vec{v}_i}{\sum_{i} \exp\left(-\frac{d_i^2}{2\sigma^2}\right)}
```

## 📂 Project Structure

```
latent_velocity/
│
├── data/
│   ├── simpleMHAS.sav               # Raw MHAS data (Ignored in .gitignore)
│   └── frailty_index_data.csv       # Phase 1: Curated 34-item fractional FI matrix
│
├── engine/
│   ├── _paths.py                    # Centralized path resolution
│   ├── prepare_frailty_data.py      # Phase 1: MICE imputation & explicit filtering
│   ├── train_vae.py                 # Phase 2 & 7: Inverse-Variance Weighted β-VAE
│   ├── extract_velocity.py          # Phase 3 & 4: GP Interpolation & Analytical Derivative
│   ├── clinical_validation.py       # Phase 5: LMM Validation & Cox PH Survival Models
│   ├── vector_field_inference.py    # Phase 9: KD-Tree Construction & Single-Point Inference
│   └── diagnostic_tests.py          # Per-domain reconstruction & variance diagnostics
│
├── models/
│   ├── beta_vae_model.pth           # Frozen encoder/decoder weights
│   └── latent_velocity_trajectory.csv # The 5-million point historical manifold
│
├── plots/                           # Phase 6 & 8 outputs (Streamplots, Heatmaps, KM Curves)
│   ├── visualize_streamplot.py
│   ├── visualize_tsne.py
│   ├── visualize_gp.py
│   └── analyze_heatmap.py
│
└── README.md
```

## 🛠 Usage

To execute the full LAVA pipeline:

```bash
cd latent_velocity/engine
python prepare_frailty_data.py
python train_vae.py
python extract_velocity.py
python clinical_validation.py

cd ../plots
python visualize_streamplot.py
```

## 🏗 Requirements
- torch
- pandas
- numpy
- pyreadstat
- scikit-learn
- joblib
- statsmodels
- lifelines
- matplotlib
- seaborn
- scipy

