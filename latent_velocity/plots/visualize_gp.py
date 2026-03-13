import sys
from pathlib import Path

# Add the engine directory to the Python path for imports
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "engine"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from extract_velocity import extract_latent_vectors, fit_predict_gp
from _paths import DATA_DIR, MODELS_DIR, PLOTS_DIR
import warnings

warnings.filterwarnings('ignore')

def plot_patient_gp(patient_data, gp_results, z_cols, patient_id):
    k_dims = len(z_cols)
    fig, axes = plt.subplots(k_dims, 1, figsize=(10, 2.5 * k_dims), sharex=True)
    if k_dims == 1:
        axes = [axes]
    
    for k, col in enumerate(z_cols):
        ax = axes[k]
        
        # Raw data
        ax.scatter(patient_data['t'], patient_data[col], color='red', label='Observed $z_{}$'.format(k), zorder=5)
        
        # GP fit and Velocity
        if gp_results is not None:
            line1 = ax.plot(gp_results['t'], gp_results[f'z_mean_{k}'], color='blue', label='Posterior Mean', lw=2)
            
            # Twin axis for velocity
            ax_v = ax.twinx()
            line2 = ax_v.plot(gp_results['t'], gp_results[f'v_{k}'], color='green', linestyle='--', label='Velocity $v_{}$'.format(k), lw=1.5)
            ax_v.set_ylabel('Velocity', color='green')
            ax_v.tick_params(axis='y', labelcolor='green')
            
            # Combine legends
            lines = line1 + line2 + [ax.collections[0]]
            labels = ['Posterior Mean', 'Velocity $v_{}$'.format(k), 'Observed $z_{}$'.format(k)]
            ax.legend(lines, labels, loc='upper left')
        else:
            ax.legend(loc='upper left')
            
        ax.set_ylabel(f'Latent Dim {k}')
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.set_title(f'GP Trajectories and Velocities for Patient {patient_id}')
            
    axes[-1].set_xlabel('Time (Years since 2001)')
    plt.tight_layout()
    
    out_path = str(PLOTS_DIR / f'patient_{int(patient_id[0])}_{int(patient_id[1])}_gp.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved GP visualization to: {out_path}")
    plt.close()

def main():
    model_path = str(MODELS_DIR / 'beta_vae_model.pth')
    data_path = str(DATA_DIR / 'frailty_index_data.csv')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Extracting latent space...")
    
    df_latent, z_cols = extract_latent_vectors(model_path, data_path, device)
    
    # Find patients with many observations for a good visualization
    counts = df_latent.groupby(['cunicah', 'np']).size()
    rich_patients = counts[counts >= 5].index.tolist()
    
    if not rich_patients:
        rich_patients = counts[counts >= 4].index.tolist()
        
    if not rich_patients:
        rich_patients = counts[counts >= 3].index.tolist()
        
    if not rich_patients:
        print("No patients with >= 3 observations found.")
        return
        
    # Pick a specific patient to visualize
    target_patient = rich_patients[10]
    print(f"Selected patient: {target_patient} with {counts[target_patient]} observations.")
    
    patient_data = df_latent[(df_latent['cunicah'] == target_patient[0]) & (df_latent['np'] == target_patient[1])].copy()
    
    print("Fitting independent GPs and extracting velocities...")
    gp_results = fit_predict_gp(patient_data, z_cols)
    
    print("Plotting results...")
    plot_patient_gp(patient_data, gp_results, z_cols, target_patient)

if __name__ == "__main__":
    main()
