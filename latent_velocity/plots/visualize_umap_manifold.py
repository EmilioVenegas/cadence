import sys
from pathlib import Path

# Add the engine directory to the Python path for imports
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "engine"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import torch
import matplotlib.cm as cm
from extract_velocity import extract_latent_vectors
from _paths import DATA_DIR, MODELS_DIR, PLOTS_DIR, LATENT_DIR

from sklearn.preprocessing import StandardScaler

def plot_all_umap(df_merged, z_cols):
    print("Computing UMAP embedding. This is faster and more topologically faithful than t-SNE...")
    
    # 1. Standardize the latent space
    z_data = df_merged[z_cols].values
    z_scaled = StandardScaler().fit_transform(z_data)
    
    # 2. UMAP Configuration - REFINED for Global Connectivity
    # n_neighbors=200 to force UMAP to look at the big picture (bridge islands)
    # min_dist=0.5 to let points spread out for better legibility
    reducer = umap.UMAP(
        n_neighbors=200, 
        min_dist=0.5, 
        n_components=2, 
        random_state=42,
        metric='euclidean'
    )
    z_embedded = reducer.fit_transform(z_scaled)
    
    # Helper to calculate color limits to prevent outlier washout
    def get_limits(series):
        return series.quantile(0.05), series.quantile(0.95)

    print("Plotting 1: Ground-Truth Frailty Index (FI)")
    plt.figure(figsize=(10, 8))
    vmin, vmax = get_limits(df_merged['FI'])
    scatter = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], 
                          c=df_merged['FI'], cmap='inferno', s=10, alpha=0.6,
                          vmin=vmin, vmax=vmax)
    plt.colorbar(scatter, label='Frailty Index (FI)')
    plt.title('UMAP Projection by Frailty Index')
    plt.grid(True, alpha=0.3)
    plt.savefig(str(LATENT_DIR / 'umap_frailty_index.png'), dpi=150)
    plt.close()
    
    print("Plotting 2: Chronological Age")
    plt.figure(figsize=(10, 8))
    vmin, vmax = get_limits(df_merged['edad'])
    scatter = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], 
                          c=df_merged['edad'], cmap='viridis', s=10, alpha=0.6,
                          vmin=vmin, vmax=vmax)
    plt.colorbar(scatter, label='Chronological Age (Years)')
    plt.title('UMAP Projection by Chronological Age')
    plt.grid(True, alpha=0.3)
    plt.savefig(str(LATENT_DIR / 'umap_age.png'), dpi=150)
    plt.close()

    print("Plotting 3: Mortality / Terminal State")
    plt.figure(figsize=(10, 8))
    # Plot alive first as background
    mask_terminal = df_merged['is_terminal'] == 1
    mask_alive = ~mask_terminal
    
    plt.scatter(z_embedded[mask_alive, 0], z_embedded[mask_alive, 1], 
                c='lightgray', s=10, alpha=0.3, label='Survivors / Stable')
    # Plot terminal events (last known state before death)
    plt.scatter(z_embedded[mask_terminal, 0], z_embedded[mask_terminal, 1], 
                c='red', marker='x', s=25, alpha=0.9, label='Terminal State (Died)')
    plt.legend()
    plt.title('Mortality "Event Horizon" (Last Known Alive State)')
    plt.grid(True, alpha=0.3)
    plt.savefig(str(LATENT_DIR / 'umap_mortality.png'), dpi=150)
    plt.close()
    
    print("Plotting 4: Domain-Specific Deficits (Cognitive vs Physical)")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
    
    # Cognitive
    vmin_c, vmax_c = get_limits(df_merged['recuerdo1'])
    sc1 = axes[0].scatter(z_embedded[:, 0], z_embedded[:, 1], 
                          c=df_merged['recuerdo1'], cmap='coolwarm', s=10, alpha=0.6,
                          vmin=vmin_c, vmax=vmax_c)
    fig.colorbar(sc1, ax=axes[0], label='Word Recall Score')
    axes[0].set_title('Cognitive Function Projection (UMAP)')
    axes[0].grid(True, alpha=0.3)
    
    # Physical
    vmin_p, vmax_p = get_limits(df_merged['n_abvd'])
    sc2 = axes[1].scatter(z_embedded[:, 0], z_embedded[:, 1], 
                          c=df_merged['n_abvd'], cmap='coolwarm', s=10, alpha=0.6,
                          vmin=vmin_p, vmax=vmax_p)
    fig.colorbar(sc2, ax=axes[1], label='# ADL Difficulties')
    axes[1].set_title('Physical Independence Projection (UMAP)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(LATENT_DIR / 'umap_domains.png'), dpi=150)
    plt.close()
    
    print("Successfully generated all 4 UMAP projections!")
    return z_embedded

def plot_categorical_islands(df_merged, z_embedded):
    print("Plotting categorical variables to identify UMAP islands...")
    
    categorical_cols = [
        'sexo', 'hipertension', 'diabetes', 'tabaco', 
        'alcohol', 'ejer_3_por_sem', 'social_isolation'
    ]
    
    for col in categorical_cols:
        if col not in df_merged.columns:
            print(f"  [Skip] {col} not found in dataframe.")
            continue
            
        plt.figure(figsize=(10, 8))
        
        # Clean up imputed fractional values by rounding to strict 0s and 1s
        temp_series = df_merged[col].dropna().round()
        categories = sorted(temp_series.unique())
        cmap = plt.get_cmap('Set1') # High contrast for binary/discrete data
        
        for i, cat in enumerate(categories):
            mask = df_merged[col].round() == cat
            plt.scatter(z_embedded[mask, 0], z_embedded[mask, 1], 
                        color=cmap(i % 9), s=15, alpha=0.7, label=f'{col} = {int(cat)}')
            
        plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f'UMAP Island Analysis: {col} (Rounded Imputation)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = LATENT_DIR / f'umap_island_{col}.png'
        plt.savefig(str(save_path), dpi=150)
        plt.close()
        print(f"  -> Saved {save_path.name}")

def plot_aging_trajectories(df_merged, z_embedded, n_patients_to_plot=30):
    print(f"Plotting longitudinal aging trajectories for {n_patients_to_plot} patients...")
    
    # 1. Attach the 2D UMAP coordinates to our dataframe for easy grouping
    df_merged = df_merged.copy()
    df_merged['umap_x'] = z_embedded[:, 0]
    df_merged['umap_y'] = z_embedded[:, 1]
    
    plt.figure(figsize=(12, 10))
    
    # 2. Plot the background "Geography" (All events, faint)
    plt.scatter(df_merged['umap_x'], df_merged['umap_y'], 
                c='lightgray', s=10, alpha=0.1, zorder=1)
    
    # 3. Find patients with a rich history (at least 3 visits)
    visit_counts = df_merged.groupby(['cunicah', 'np']).size()
    multi_visit_patients = visit_counts[visit_counts >= 3].index.tolist()
    
    if len(multi_visit_patients) == 0:
        print("Not enough patients with 3+ visits to plot trajectories.")
        return
        
    # 4. Randomly sample patients to avoid a clustered hairball
    np.random.seed(42)
    sampled_patients = np.random.choice(
        len(multi_visit_patients), 
        size=min(n_patients_to_plot, len(multi_visit_patients)), 
        replace=False
    )
    
    cmap = plt.get_cmap('tab20')
    
    # 5. Draw the trajectories!
    for i, patient_idx in enumerate(sampled_patients):
        patient_id = multi_visit_patients[patient_idx]
        
        # Get patient data and sort chronologically by visit date/year
        p_data = df_merged[
            (df_merged['cunicah'] == patient_id[0]) & 
            (df_merged['np'] == patient_id[1])
        ].sort_values('a_o_ent')
        
        x_coords = p_data['umap_x'].values
        y_coords = p_data['umap_y'].values
        
        color = cmap(i % 20)
        
        # Plot the individual visits as distinct solid dots
        plt.scatter(x_coords, y_coords, color=color, s=30, zorder=3, edgecolors='black', linewidth=0.5)
        
        # Draw arrows connecting visit T to visit T+1
        for j in range(len(x_coords) - 1):
            dx = x_coords[j+1] - x_coords[j]
            dy = y_coords[j+1] - y_coords[j]
            
            # Using quiver to draw arrows
            plt.quiver(x_coords[j], y_coords[j], dx, dy, 
                       angles='xy', scale_units='xy', scale=1, 
                       color=color, alpha=0.8, width=0.003, 
                       headwidth=5, headlength=6, zorder=2)
            
            # Highlight catastrophic shifts (Jumps across islands)
            distance = np.sqrt(dx**2 + dy**2)
            if distance > 3.0: # Arbitrary threshold for an inter-island jump
                plt.annotate('!', 
                             xy=(x_coords[j] + dx/2, y_coords[j] + dy/2),
                             color='red', weight='bold', fontsize=12, zorder=4)

    plt.title(f'Longitudinal Aging Phenotypes (Trajectories of {n_patients_to_plot} Patients)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = LATENT_DIR / 'umap_aging_trajectories.png'
    plt.savefig(str(save_path), dpi=200)
    plt.close()
    print(f"  -> Saved {save_path.name}")

def main():
    model_path = str(MODELS_DIR / 'beta_vae_model.pth')
    data_path = str(DATA_DIR / 'frailty_index_data.csv')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Extracting latent space...")
    
    df_latent, z_cols = extract_latent_vectors(model_path, data_path, device)
    
    df_orig = pd.read_csv(data_path)
    
    # NEW: adding categorical columns to the merge list
    columns_to_keep = [
        'cunicah', 'np', 'a_o_ent', 'FI', 'edad', 'fallecido', 'recuerdo1', 'n_abvd',
        'sexo', 'hipertension', 'diabetes', 'tabaco', 'alcohol', 'ejer_3_por_sem',
        'asiste_club', 'voluntario'
    ]
    
    # Safety check
    valid_cols = [c for c in columns_to_keep if c in df_orig.columns]
    
    # Propagate mortality signal: Identify patients who EVER died in the original data
    dead_ids = df_orig[df_orig['fallecido'] == 1]['cunicah'].unique()
    
    # Create the merge
    df_merged = pd.merge(df_latent, df_orig[valid_cols], 
                         on=['cunicah', 'np', 'a_o_ent'], how='left')
    
    # Mark the LAST observation in df_merged for those who eventually died as the "Terminal State"
    # (Since fallecido=1 rows were dropped by VAE, the max(a_o_ent) in df_merged is their last alive state)
    df_merged['is_terminal'] = 0
    last_wave_idx = df_merged.groupby('cunicah')['a_o_ent'].idxmax()
    
    # Correctly filter last_wave_idx using boolean mask
    is_dead_mask = df_merged.loc[last_wave_idx, 'cunicah'].isin(dead_ids)
    dead_terminal_indices = last_wave_idx.values[is_dead_mask.values]
    
    df_merged.loc[dead_terminal_indices, 'is_terminal'] = 1
    
    # Derive social isolation in the same way as digital_twin.py
    if 'asiste_club' in df_merged.columns and 'voluntario' in df_merged.columns:
        df_merged['social_isolation'] = 1.0 - df_merged[['asiste_club', 'voluntario']].max(axis=1)
    
    # NEW: Combo Profile (Island Fingerprinting)
    # Creates labels like "Exer:1_Diab:0" to see exactly which combo defines which island
    if 'ejer_3_por_sem' in df_merged.columns and 'diabetes' in df_merged.columns:
        df_merged['combo_profile'] = "Exer:" + df_merged['ejer_3_por_sem'].round().astype(int).astype(str) + \
                                     "_Diab:" + df_merged['diabetes'].round().astype(int).astype(str)
    
    z_embedded = plot_all_umap(df_merged, z_cols)
    plot_categorical_islands(df_merged, z_embedded)
    
    # NEW: Patient Trajectories (Longitudinal Flow)
    plot_aging_trajectories(df_merged, z_embedded, n_patients_to_plot=10)
    
    # Plot the Fingerprint Combo if it exists
    if 'combo_profile' in df_merged.columns:
        print("Plotting Combo Profile (Island Fingerprinting)...")
        plt.figure(figsize=(12, 8))
        categories = sorted(df_merged['combo_profile'].unique())
        cmap = plt.get_cmap('Set1')
        for i, cat in enumerate(categories):
            mask = df_merged['combo_profile'] == cat
            plt.scatter(z_embedded[mask, 0], z_embedded[mask, 1], 
                        color=cmap(i % 9), s=15, alpha=0.7, label=cat)
        plt.legend(title="Clinical Profile", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('UMAP Island Fingerprinting (Exercise/Diabetes Combinations)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = LATENT_DIR / 'umap_fingerprint_combo.png'
        plt.savefig(str(save_path), dpi=150)
        plt.close()
        print(f"  -> Saved {save_path.name}")

if __name__ == "__main__":
    main()
