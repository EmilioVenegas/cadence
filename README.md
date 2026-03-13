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
  $$FI = \frac{1}{N} \sum_{i=1}^{N} d_i$$

### 2. Generative Manifold Learning (`train_vae.py`)
Maps the deficit space into a low-dimensional latent manifold using a $\beta$-Variational Autoencoder ($\beta$-VAE).
- **Disentanglement**: Optimized $\beta$-annealing to separate generic frailty from domain-specific signals.
- **Architecture**: Deep MLP with feature-weighted reconstruction losses.

**Mathematical Foundation ($\beta$-VAE Architecture):**
- **Reparameterization Trick**: Sampling from the latent space allowing for backpropagation.
  $$z = \mu + \epsilon \odot \exp\left(\frac{1}{2} \log \sigma^2\right)$$
- **Feature-Weighted $\beta$-VAE Loss**: Combining Inverse-Variance weighted Mean Squared Error (MSE) and $\beta$-scaled Kullback-Leibler (KL) Divergence.
  $$\mathcal{L} = \sum_{j=1}^{D} \left( w_j (x_j - \hat{x}_j)^2 \right) - \frac{\beta}{2} \sum_{k=1}^{K} \left( 1 + \log(\sigma_k^2) - \mu_k^2 - \sigma_k^2 \right)$$
- **Inverse-Variance Feature Weights**: Normalized weights to penalize features with low variance.
  $$w_j = \frac{\frac{1}{\sigma_j^2}}{\sum_{d=1}^{D} \frac{1}{\sigma_d^2}} \times D$$

### 3. Longitudinal Velocity Inference (`extract_velocity.py`)
Infers continuous-time trajectories and their derivatives for each individual.
- **GP Smoothing**: Gaussian Process regression over longitudinal latent states.
- **Analytic Velocity**: Extraction of exact temporal derivatives ($\frac{dz}{dt}$) for precise measurement of decline speed.

**Mathematical Foundation (Gaussian Process & Analytic Derivatives):**
- **GP Kernel (Constant $\times$ RBF + White Noise)**:
  $$k(t, t') = \sigma_f^2 \exp\left(-\frac{(t - t')^2}{2l^2}\right) + \sigma_n^2 \delta_{t, t'}$$
- **Posterior Mean Trajectory**: The smoothed latent state at a dense grid of points $t^*$.
  $$\bar{z}(t^*) = K(t^*, t_{obs}) (K(t_{obs}, t_{obs}) + \sigma_n^2 I)^{-1} y_{obs}$$
- **GP Weights ($\alpha$)**:
  $$\alpha = (K(t_{obs}, t_{obs}) + \sigma_n^2 I)^{-1} y_{obs}$$
- **RBF Kernel Derivative**: The exact partial derivative of the cross-covariance matrix with respect to the prediction time $t^*$.
  $$K'(t^*, t_{obs}) = -\frac{t^* - t_{obs}}{l^2} k(t^*, t_{obs})$$
- **Analytic Latent Velocity**: The exact derivative of the posterior mean trajectory.
  $$v(t^*) = \frac{\partial \bar{z}(t^*)}{\partial t^*} = K'(t^*, t_{obs}) \alpha$$

### 4. Clinical Validation (`clinical_validation.py`)
Rigorous statistical verification of latent velocities.
- **Survival Analysis**: Cox Proportional Hazards modeling to test velocity as a predictor of mortality.
- **Mixed Models**: Linear Mixed Models to correlate latent derivatives with longitudinal clinical domain progression.

**Mathematical Foundation (Validation & Survival):**
- **Latent Velocity Magnitude**: Calculating the Euclidean norm of the latent velocity vector.
  $$v_{mag} = \sqrt{\sum_{k} v_k^2}$$
- **Empirical Clinical Velocity**: Forward finite difference of the Frailty Index over time.
  $$v_{emp} = \frac{FI_{t+1} - FI_t}{t_{t+1} - t_t}$$
- **Z-Score Normalization**: Used for standardizing age and education.
  $$z = \frac{x - \mu}{\sigma}$$
- **Hazard Ratio (Cox Proportional Hazards)**: Exponentiating the coefficient for the "Fast Ager" flag.
  $$HR = \exp(\beta_{Fast\_Ager})$$

### 5. Visual Diagnostics (`visualize_*.py`)
Generates high-resolution visualizations for model interpretability.
- **Vector Fields**: Latent flow streamplots (e.g., Physical vs. Cognitive disentanglement).
- **Phase Portraits**: Visualizing biological "momentum" by plotting state vs. velocity.

**Mathematical Foundation (Vector Field Kinematics):**
- **Discrete Acceleration** (in `visualize_streamplot.py`): Backward finite difference of the analytic velocity.
  $$a_k(t) = \frac{v_k(t) - v_k(t-\Delta t)}{\Delta t}$$
- **Gaussian Weighted Spatial Interpolation**: Computing localized vector streams using nearest neighbors in the 2D plane based on distance $d_i$.
  $$\vec{V}_{interp} = \frac{\sum_{i} \exp\left(-\frac{d_i^2}{2\sigma^2}\right) \vec{v}_i}{\sum_{i} \exp\left(-\frac{d_i^2}{2\sigma^2}\right)}$$

## 🛠 Usage

To execute the full LAVA pipeline:

```bash
cd latent_velocity
python prepare_frailty_data.py
python train_vae.py
python extract_velocity.py
python clinical_validation.py
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
    