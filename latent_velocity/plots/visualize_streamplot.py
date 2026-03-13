import sys
from pathlib import Path

# Add the engine directory to the Python path for imports
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "engine"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import os
import time
from _paths import DATA_DIR, MODELS_DIR, PLOTS_DIR

def load_data_advanced(traj_file):
    print("Loading trajectory vector field...")
    df = pd.read_csv(traj_file)
    
    # Pre-sort to ensure diff() works for acceleration
    df = df.sort_values(by=['cunicah', 'np', 't'])
    
    # Calculate Acceleration (for Phase Portrait)
    print("Calculating local accelerations for Phase Portrait...")
    # dt = 0.1 based on extract_velocity.py
    dt = 0.1
    for k in range(8):
        df[f'a_{k}'] = df.groupby(['cunicah', 'np'])[f'v_{k}'].diff() / dt
        # Fill first point with 0 or backfill
        df[f'a_{k}'] = df[f'a_{k}'].ffill().bfill()
        
    z_cols = [col for col in df.columns if col.startswith('z_mean_')]
    v_cols = [col for col in df.columns if col.startswith('v_')]
    a_cols = [col for col in df.columns if col.startswith('a_')]
    
    # Mortality coordinates
    print("Extracting terminal event coordinates from master simpleMHAS.sav...")
    import pyreadstat
    sav_path = str(DATA_DIR / 'simpleMHAS.sav')
    df_sav, _ = pyreadstat.read_sav(sav_path, usecols=['cunicah', 'np', 'fallecido'])
    death_ids = df_sav[df_sav['fallecido'] == 1][['cunicah', 'np']].drop_duplicates()
    
    df_dead = pd.merge(df, death_ids, on=['cunicah', 'np'], how='inner')
    death_coords_df = df_dead.sort_values('t').groupby(['cunicah', 'np']).tail(1)
    
    return df, death_coords_df, z_cols, v_cols, a_cols

def build_interpolated_grid(Z_data, V_data, grid_res=60, k_neighbors=50, sigma=0.5, mask_threshold=0.0005):
    """
    Generalized grid interpolator for any 2D coordinate system and 2D vector field.
    """
    x_min, x_max = Z_data[:, 0].min(), Z_data[:, 0].max()
    y_min, y_max = Z_data[:, 1].min(), Z_data[:, 1].max()
    
    # Margin
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    
    XX, YY = np.meshgrid(np.linspace(x_min - x_margin, x_max + x_margin, grid_res),
                         np.linspace(y_min - y_margin, y_max + y_margin, grid_res))
    
    grid_points = np.c_[XX.ravel(), YY.ravel()]
    tree = KDTree(Z_data)
    
    U = np.zeros(grid_points.shape[0])
    W = np.zeros(grid_points.shape[0])
    
    for i, pt in enumerate(grid_points):
        dist, ind = tree.query(pt.reshape(1, -1), k=k_neighbors)
        
        # Masking
        if np.mean(dist[0]) > mask_threshold:
            U[i], W[i] = np.nan, np.nan
            continue
            
        weights = np.exp(-(dist[0]**2) / (2 * (sigma**2)))
        v_interp = np.average(V_data[ind[0]], axis=0, weights=weights)
        U[i], W[i] = v_interp[0], v_interp[1]
        
    return XX, YY, U.reshape(grid_res, grid_res), W.reshape(grid_res, grid_res)

def plot_custom_streamplot(XX, YY, U, W, Z_pts, D_pts, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 8))
    speed = np.sqrt(U**2 + W**2)
    
    # Background points
    plt.scatter(Z_pts[:, 0], Z_pts[:, 1], color='gray', s=1, alpha=0.05, label='States')
    
    # Mortality markers
    if len(D_pts) > 0:
        plt.scatter(D_pts[:, 0], D_pts[:, 1], color='red', marker='o', s=1, alpha=0.05, label='Death')
        
    # Streams
    strm = plt.streamplot(XX, YY, U, W, color=speed, linewidth=1.5, cmap='plasma', density=1.5)
    plt.colorbar(strm.lines, label='Velocity Magnitude')
    
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.2)
    plt.legend(loc='upper right')
    
    out_path = str(PLOTS_DIR / filename)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    plt.close()

def main():
    traj_file = str(MODELS_DIR / 'latent_velocity_trajectory.csv')
        
    df, df_death, z_cols, v_cols, a_cols = load_data_advanced(traj_file)
    
    # Subsample for tree construction speed
    df_sub = df.sample(n=min(100000, len(df)), random_state=42)
    
    # --- Option 1: True Disentanglement Plot (z3: Physical vs z7: Cognitive) ---
    print("\nGenerating Option 1: Disentanglement Plot (z3 vs z7)...")
    Z1 = df_sub[['z_mean_3', 'z_mean_7']].values
    V1 = df_sub[['v_3', 'v_7']].values
    D1 = df_death[['z_mean_3', 'z_mean_7']].values
    
    XX1, YY1, U1, W1 = build_interpolated_grid(Z1, V1, mask_threshold=0.01)
    plot_custom_streamplot(XX1, YY1, U1, W1, Z1, D1, 
                           "Latent Disentanglement Flow: Physical vs Cognitive",
                           "Latent Dimension 3 (Physical/General Frailty)",
                           "Latent Dimension 7 (Cognitive State)",
                           "streamplot_disentanglement.png")
    
    # --- Option 2: Phase Portrait (z3 Position vs v3 Velocity) ---
    print("\nGenerating Option 2: Phase Portrait (z3 vs v3)...")
    Z2 = df_sub[['z_mean_3', 'v_3']].values
    V2 = df_sub[['v_3', 'a_3']].values
    D2 = df_death[['z_mean_3', 'v_3']].values
    
    # Scale correction for Phase Portrait interpolation
    scale_y = 1000
    Z2_scaled = Z2 * np.array([1, scale_y])
    
    XX2, YY2, U2, W2 = build_interpolated_grid(Z2_scaled, V2, mask_threshold=0.02)
    # Unscale YY for plotting
    YY2_unscaled = YY2 / scale_y
    
    plot_custom_streamplot(XX2, YY2_unscaled, U2, W2, Z2, D2,
                           "Biological Phase Portrait: State vs Momentum",
                           "Latent Position (z3: Physical State)",
                           "Latent Velocity (v3: Decline Speed)",
                           "streamplot_phase_portrait.png")

if __name__ == "__main__":
    main()
