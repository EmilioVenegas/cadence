import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import warnings
import sys
import argparse
import os
from pathlib import Path

# Add the engine directory to the Python path for imports
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "engine"))

from train_vae import BetaVAE, FrailtyDataset
from _paths import DATA_DIR, MODELS_DIR, GP_DIR

def get_patient_data(cunicah, np_id, dataset):
    """Filter dataset for a specific patient."""
    mask = (dataset.data['cunicah'] == cunicah) & (dataset.data['np'] == np_id)
    patient_df = dataset.data[mask].copy()
    
    if len(patient_df) == 0:
        return None, None, None
        
    indices = patient_df.index.tolist()
    x_deficits = dataset.x_deficits[indices]
    x_static = dataset.x_static[indices]
    
    return patient_df, x_deficits, x_static

def visualize_patient_gp(cunicah=None, np_id=None, save_dir=GP_DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print("Loading Trained β-VAE...")
    vae = BetaVAE(latent_dim=8).to(device)
    model_path = str(MODELS_DIR / 'beta_vae_model.pth')
    vae.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    vae.eval()
    
    # 2. Load Dataset
    print("Loading Dataset...")
    data_path = str(DATA_DIR / 'frailty_index_data.csv')
    dataset = FrailtyDataset(data_path, device=device)
    
    # 3. Select Patient
    if cunicah is None or np_id is None:
        # Pick a random patient with at least 3 observations for better GP visualization
        counts = dataset.data.groupby(['cunicah', 'np']).size()
        multi_visit_patients = counts[counts >= 3].index.tolist()
        if not multi_visit_patients:
            multi_visit_patients = counts[counts >= 2].index.tolist()
        
        idx = np.random.choice(len(multi_visit_patients))
        cunicah, np_id = multi_visit_patients[idx]
        print(f"Randomly selected Patient: cunicah={cunicah}, np={np_id}")
    else:
        print(f"Visualizing Patient: cunicah={cunicah}, np={np_id}")
        
    patient_df, x_deficits, x_static = get_patient_data(cunicah, np_id, dataset)
    if patient_df is None:
        print(f"Error: Patient {cunicah}-{np_id} not found in dataset.")
        return

    # 4. Encode Latent Vectors
    with torch.no_grad():
        mu, _ = vae.encode(x_deficits.to(device), x_static.to(device))
        mu_array = mu.cpu().numpy()
        
    # Standardize time: t = Year - 2001 (matching extract_velocity.py)
    patient_df['t'] = patient_df['a_o_ent'] - 2001
    patient_df = patient_df.sort_values('t')
    t_obs = patient_df['t'].values.reshape(-1, 1)
    
    # 5. Fit GP for each dimension
    t_min, t_max = t_obs.min(), t_obs.max()
    t_grid = np.linspace(t_min - 1, t_max + 1, 200).reshape(-1, 1)
    
    # Kernel matching extract_velocity.py
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=5.0, length_scale_bounds=(2.0, 15.0)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
    axes = axes.flatten()
    
    for k in range(8):
        y_obs = mu_array[:, k]
        
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gp.fit(t_obs, y_obs)
            
        # Diagnosis: Print optimized kernel parameters
        opt_kernel = gp.kernel_
        noise_level = opt_kernel.k2.noise_level if hasattr(opt_kernel, 'k2') else "N/A"
        # The kernel is (Constant * RBF) + White
        # Access: k1 (Constant*RBF) -> k1.k1 (Constant), k1.k2 (RBF)
        # k2 (White)
        try:
            l_scale = opt_kernel.k1.k2.length_scale
            c_val = opt_kernel.k1.k1.constant_value
            n_level = opt_kernel.k2.noise_level
            print(f"Dim {k} | L-Scale: {l_scale:.2f} | Constant: {c_val:.2f} | Noise: {n_level:.4f}")
        except:
            print(f"Dim {k} | Kernel: {opt_kernel}")

        y_mean, y_std = gp.predict(t_grid, return_std=True)
        
        ax = axes[k]
        ax.plot(t_grid + 2001, y_mean, label='GP Mean', color='blue', lw=2)
        ax.fill_between((t_grid + 2001).flatten(), 
                        y_mean - 1.96 * y_std, 
                        y_mean + 1.96 * y_std, 
                        alpha=0.2, color='blue', label='95% CI')
        ax.scatter(t_obs + 2001, y_obs, color='red', s=50, edgecolors='black', label='Observations', zorder=5)
        
        ax.set_title(f"Latent Dim {k}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if k >= 4:
            ax.set_xlabel("Year (Age Proxy)", fontsize=10)
        if k % 4 == 0:
            ax.set_ylabel("Latent Value", fontsize=10)
            
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = f"patient_{cunicah}_{np_id}_gp_trajectory.png"
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a patient's GP fitted curve.")
    parser.add_argument("--cunicah", type=int, help="Patient CUNICAH ID")
    parser.add_argument("--np", type=int, help="Patient NP ID")
    args = parser.parse_args()
    
    visualize_patient_gp(args.cunicah, args.np)
