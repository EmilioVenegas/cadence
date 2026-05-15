import torch
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from joblib import Parallel, delayed
from train_vae import BetaVAE, FrailtyDataset
from torch.utils.data import DataLoader
import warnings
from _paths import DATA_DIR, MODELS_DIR

def extract_latent_vectors(model_path, data_path, device='cpu'):
    print("Loading Trained β-VAE...")
    vae = BetaVAE(latent_dim=8).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    vae.eval()
    
    print("Processing full dataset through Encoder...")
    dataset = FrailtyDataset(data_path, device=device)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    mu_list = []
    with torch.no_grad():
        for b_deficits, b_static in dataloader:
            # We strictly extract the core Mean vector mu_i(t)
            mu, _ = vae.encode(b_deficits, b_static)
            mu_list.append(mu.cpu().numpy())
            
    mu_array = np.vstack(mu_list)
    print(f"Latent Vectors Extracted. Shape: {mu_array.shape}")
    print(f"Empirical Latent Mean: {np.mean(mu_array, axis=0)}")
    print(f"Empirical Latent Std: {np.std(mu_array, axis=0)}")
    print(f"Overall Latent Expansion (Mean Std): {np.mean(np.std(mu_array, axis=0))}")
    
    # Attach latent dimensions back to the patient identifiers and time
    # Time is standardized: t = Year - 2001
    
    # Reset index to ensure alignment with mu_array (0 to N-1)
    df_meta = dataset.data[['cunicah', 'np', 'a_o_ent']].reset_index(drop=True).copy()
    
    # Drop rows with missing interview years before calculating time t
    missing_years = df_meta['a_o_ent'].isna()
    if missing_years.any():
        print(f"Dropping {missing_years.sum()} rows with missing interview year ('a_o_ent')")
        
    df_meta['t'] = df_meta['a_o_ent'] - 2001
    
    dim_cols = [f'z_{k}' for k in range(mu_array.shape[1])]
    df_latent = pd.DataFrame(mu_array, columns=dim_cols)
    
    # Concat and strictly drop any row that didn't have a valid time t
    df_final = pd.concat([df_meta, df_latent], axis=1)
    df_final = df_final.dropna(subset=['t'])
    
    return df_final, dim_cols

def fit_predict_gp(patient_data, z_cols, t_grid_step=0.1):
    """
    Fits K independent GPs for a single patient's longitudinal trajectory.
    Returns the dense time array, predicted posterior means, and exact analytic derivatives.
    """
    patient_id = (patient_data['cunicah'].iloc[0], patient_data['np'].iloc[0])
    
    # Sort chronologically
    patient_data = patient_data.sort_values(by='t')
    
    # Observed times and values
    t_obs = patient_data['t'].values.reshape(-1, 1)
    
    # Create the dense grid across their observation window
    t_min = t_obs.min()
    t_max = t_obs.max()
    t_dense = np.arange(t_min, t_max + t_grid_step, t_grid_step).reshape(-1, 1)
    
    # If a patient only has one observation, GP smoothing is impossible/meaningless.
    if len(t_obs) < 2:
        return None
        
    # Kernel definition: 1.0 * RBF + WhiteKernel
    # Length scale bounds: 2.0 to 15.0 years (Biological smoothness constraint)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=5.0, length_scale_bounds=(2.0, 15.0)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
    
    results = {
        'cunicah': patient_id[0],
        'np': patient_id[1],
        't': t_dense.flatten()
    }
    
    # Fit independent GP for each K latent dimension
    for k, col in enumerate(z_cols):
        y_obs = patient_data[col].values
        
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gp.fit(t_obs, y_obs)
            
        # 1. Predict Posterior Mean Trajectory: \bar{z}(t)
        z_mean, _ = gp.predict(t_dense, return_std=True)
        results[f'z_mean_{k}'] = z_mean
        
        # 2. Extract Exact Analytic Derivative (The Velvet Hammer)
        # For the kernel k(x, x') = sigma_f^2 * exp(-0.5 * d^2 / l^2)
        # where d = x - x', the derivative w.r.t dense time t* is:
        # \partial \bar{z}(t*) / \partial t* = K'(t*, t_obs) @ (K(t_obs, t_obs) + sigma_n^2 I)^{-1} @ y_obs
        
        # Extract optimized parameters
        opt_len_scale = gp.kernel_.k1.k2.length_scale
        
        # We need the weights: alpha_ = (K + sigma_n I)^-1 * y
        alpha = gp.alpha_
        
        # Compute exact cross-covariance derivative matrix: K'(t*, t_obs)
        K_cross = gp.kernel_.k1(t_dense, t_obs)
        
        diff_matrix = t_dense - t_obs.T  # (N_dense, N_obs), (t* - t_obs)
        
        # Derivative of RBF kernel k(t, t') is:  -(t - t') / l^2 * k(t, t')
        K_prime = -(diff_matrix / (opt_len_scale ** 2)) * K_cross
        
        # 3. Exact Analytic Velocity: v(t*) = sum_obs [ K_prime(t*, t_obs) * alpha ]
        v_analytic = K_prime @ alpha
        
        results[f'v_{k}'] = v_analytic
        
    return pd.DataFrame(results)

def extract_velocity(model_path, data_path, output_path, n_jobs=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Phase 3 Extraction on: {device}")
    
    # 1. Get sparse latent coordinates
    df_latent, z_cols = extract_latent_vectors(model_path, data_path, device)
    
    # 2. Group by patient for parallel GP fitting
    print("Parallelizing Gaussian Process smoothing across all unique patients...")
    patients = [group for _, group in df_latent.groupby(['cunicah', 'np'])]
    
    # Extract dense continuous trajectories and analytic velocities
    dense_dfs = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(fit_predict_gp)(p_data, z_cols) for p_data in patients
    )
    
    # Filter out None results (patients with < 2 observations)
    dense_dfs = [df for df in dense_dfs if df is not None]
    
    print("Compiling global high-resolution trajectory grid...")
    df_trajectory = pd.concat(dense_dfs, ignore_index=True)
    
    print(f"Saving high-resolution Latent Velocity dataset to {output_path}...")
    df_trajectory.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    model_path = str(MODELS_DIR / 'beta_vae_model_128.pth')
    data_path  = str(DATA_DIR / 'frailty_index_data.csv')
    output_path = str(MODELS_DIR / 'latent_velocity_trajectory_128.csv')
    
    extract_velocity(model_path, data_path, output_path)
