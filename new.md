What was implemented
  
  3 files created/updated:

  engine/train_latent_ode.py (new)

  Full Latent ODE-VAE — replaces both train_vae.py and train_ode.py:
  - RecognitionRNN: backward masked GRU encoder — processes all of a patient's survey waves in reverse, skipping unobserved ones
  - LatentODEFunc: identical interface to the old ODEFunc — washout simulation in digital_twin.py unchanged
  - Decoder: MLP with Sigmoid — reconstructs the 34 deficit features
  - LatentODEDataset: snaps raw MHAS rows to the 6 fixed survey wave times; 22,729 patients loaded vs ~18k before
  - Cox partial-likelihood loss (λ_cox=0.1): pulls the latent space toward survival discrimination during training — directly addresses the low HR

  engine/extract_latent_ode_velocity.py (new)

  Replaces extract_velocity.py (β-VAE + GP):
  - Encodes each patient's full sequence → z₀ distribution, integrates ODE on a dense time grid
  - MC sampling (default 30 samples) gives principled velocity uncertainty — no GP needed
  - Outputs the same CSV format as before → clinical_validation.py needs zero changes

  ode-digitaltwin/digital_twin.py (updated)

  - load_models() auto-detects latent_ode_model.pth; falls back to β-VAE if not present
  - rank_interventions() uses the full patient sequence for encoding when the new model is loaded
  - _compute_cohort_mahalanobis() handles the new single-visit encoding path for the guardrail

  ---
  New pipeline order

  # 1. unchanged
  python latent_velocity/engine/prepare_frailty_data.py

  # 2. replaces train_vae + train_ode (single command)
  python latent_velocity/engine/train_latent_ode.py --epochs 150

  # 3. replaces extract_velocity + prepare_ode_data
  python latent_velocity/engine/extract_latent_ode_velocity.py

  # 4. unchanged — reads same CSV format
  python latent_velocity/engine/clinical_validation.py
