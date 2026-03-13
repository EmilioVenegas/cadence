# 🌋 LAVA: Latent Aging Velocity Atlas
## Continuous-Time Mapping of Multi-Domain Biological Decline

LAVA is a production-grade diagnostic framework designed to disentangle the complex, non-linear trajectories of human aging. By projecting multi-domain clinical deficits into a continuous latent manifold, LAVA moves beyond static "snapshots" of health to measure the instantaneous velocity of biological aging.

## 🚀 Pipeline Overview

The LAVA framework operates through five distinct modular stages:

### 1. Data Preparation (`prepare_frailty_data.py`)
Encodes raw longitudinal survey data (MHAS) into a high-dimensional deficit space.
- **Domains**: Clinical, Functionality, Mental Health, Cognition, and Biometrics.
- **Processing**: Iterative imputation of missing values and standardization of chronological time.

### 2. Generative Manifold Learning (`train_vae.py`)
Maps the deficit space into a low-dimensional latent manifold using a $\beta$-Variational Autoencoder ($\beta$-VAE).
- **Disentanglement**: Optimized $\beta$-annealing to separate generic frailty from domain-specific signals (e.g., physical vs. cognitive).
- **Architecture**: Deep MLP with feature-weighted reconstruction losses.

### 3. Longitudinal Velocity Inference (`extract_velocity.py`)
Infers continuous-time trajectories and their derivatives for each individual.
- **GP Smoothing**: Gaussian Process regression over longitudinal latent states.
- **Analytic Velocity**: Extraction of exact temporal derivatives ($\frac{dz}{dt}$) for precise measurement of decline speed.

### 4. Clinical Validation (`clinical_validation.py`)
Rigorous statistical verification of latent velocities.
- **Survival Analysis**: Cox Proportional Hazards modeling to test velocity as a predictor of mortality.
- **Mixed Models**: Linear Mixed Models to correlate latent derivatives with longitudinal clinical domain progression.

### 5. Visual Diagnostics (`visualize_*.py`)
Generates high-resolution visualizations for model interpretability.
- **Vector Fields**: Latent flow streamplots (e.g., Physical vs. Cognitive disentanglement).
- **Phase Portraits**: Visualizing biological "momentum" by plotting state vs. velocity.
- **t-SNE Embeddings**: Global topology of aging states.

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
- PyTorch (DL Framework)
- Scikit-learn (GP & Imputation)
- Lifelines (Survival Analysis)
- Statsmodels (Mixed Effects Models)
- Matplotlib/Seaborn (Visualization)
