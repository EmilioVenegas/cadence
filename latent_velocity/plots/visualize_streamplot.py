import matplotlib
matplotlib.use('Agg')
import sys
from pathlib import Path

# Add the engine directory to the Python path for imports
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "engine"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import umap
import os
import time
from _paths import DATA_DIR, MODELS_DIR, PLOTS_DIR, STREAM_DIR

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

def build_interpolated_grid(Z_data, V_data, grid_res=80, k_neighbors=50,
                            sigma=None, mask_quantile=0.90,
                            clip_pct=(2, 98)):
    """
    Generalized grid interpolator for any 2D coordinate system and 2D vector field.
    
    Parameters
    ----------
    sigma : float or None
        Gaussian kernel bandwidth.  If None, auto-set to the median NN distance.
    mask_quantile : float  (0-1)
        Grid points whose mean-k-NN distance exceeds this quantile of the
        *data-to-data* distances are masked out (no data support).
    clip_pct : tuple (lo, hi)
        Percentiles used to define grid bounds, avoiding sparse tails.
    """
    x_lo, x_hi = np.percentile(Z_data[:, 0], clip_pct)
    y_lo, y_hi = np.percentile(Z_data[:, 1], clip_pct)
    
    # Small margin
    x_margin = (x_hi - x_lo) * 0.05
    y_margin = (y_hi - y_lo) * 0.05
    
    XX, YY = np.meshgrid(np.linspace(x_lo - x_margin, x_hi + x_margin, grid_res),
                         np.linspace(y_lo - y_margin, y_hi + y_margin, grid_res))
    
    grid_points = np.c_[XX.ravel(), YY.ravel()]
    tree = KDTree(Z_data)
    
    # --- Auto-calibrate mask_threshold from the data itself ---
    data_dists, _ = tree.query(Z_data, k=k_neighbors)
    data_mean_dists = data_dists.mean(axis=1)
    mask_threshold = np.percentile(data_mean_dists, mask_quantile * 100)
    
    if sigma is None:
        sigma = np.median(data_mean_dists)
    
    print(f"  Grid interpolation: sigma={sigma:.4f}, "
          f"mask_threshold={mask_threshold:.4f} "
          f"(q{mask_quantile*100:.0f} of data NN dists)")
    
    U = np.zeros(grid_points.shape[0])
    W = np.zeros(grid_points.shape[0])
    
    for i, pt in enumerate(grid_points):
        dist, ind = tree.query(pt.reshape(1, -1), k=k_neighbors)
        
        # Masking: skip grid cells far from any data
        if np.mean(dist[0]) > mask_threshold:
            U[i], W[i] = np.nan, np.nan
            continue
            
        weights = np.exp(-(dist[0]**2) / (2 * (sigma**2)))
        v_interp = np.average(V_data[ind[0]], axis=0, weights=weights)
        U[i], W[i] = v_interp[0], v_interp[1]
        
    n_valid = np.isfinite(U).sum()
    n_total = len(U)
    print(f"  Grid coverage: {n_valid}/{n_total} cells "
          f"({100*n_valid/n_total:.1f}%)")
        
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
    
    # Zoom axes to the grid region (where streamlines live)
    plt.xlim(XX.min(), XX.max())
    plt.ylim(YY.min(), YY.max())
    
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.2)
    plt.legend(loc='upper right')
    
    out_path = str(STREAM_DIR / filename)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    plt.close()

def plot_speed_heatmap(Z_pts, speed, D_pts, title, xlabel, ylabel, filename,
                       grid_res=80, clip_pct=(2, 98)):
    """Plot a 2D heatmap of scalar speed (velocity magnitude) using hex-binning."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x_lo, x_hi = np.percentile(Z_pts[:, 0], clip_pct)
    y_lo, y_hi = np.percentile(Z_pts[:, 1], clip_pct)
    mask = ((Z_pts[:, 0] >= x_lo) & (Z_pts[:, 0] <= x_hi) &
            (Z_pts[:, 1] >= y_lo) & (Z_pts[:, 1] <= y_hi))
    
    hb = ax.hexbin(Z_pts[mask, 0], Z_pts[mask, 1], C=speed[mask],
                   gridsize=40, cmap='inferno', reduce_C_function=np.median,
                   mincnt=5)
    plt.colorbar(hb, ax=ax, label='Median Aging Speed')
    
    if len(D_pts) > 0:
        dm = ((D_pts[:, 0] >= x_lo) & (D_pts[:, 0] <= x_hi) &
              (D_pts[:, 1] >= y_lo) & (D_pts[:, 1] <= y_hi))
        ax.scatter(D_pts[dm, 0], D_pts[dm, 1], color='cyan', marker='x',
                   s=4, alpha=0.15, label='Death')
        ax.legend(loc='upper right')
    
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.15)
    
    out_path = str(STREAM_DIR / filename)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    plt.close()


def main():
    traj_file = str(MODELS_DIR / 'latent_velocity_trajectory.csv')
        
    df, df_death, z_cols, v_cols, a_cols = load_data_advanced(traj_file)
    
    # Subsample for tree construction speed
    df_sub = df.sample(n=min(100000, len(df)), random_state=42)
    
    # ---------------------------------------------------------------
    # Plot 1 — Disentanglement Flow (z3: Physical vs z7: Cognitive)
    #   Interpretation: reveals whether physical frailty and cognitive
    #   decline proceed in tandem or on independent trajectories.
    # ---------------------------------------------------------------
    print("\n[1/8] Disentanglement Flow (z3 Physical vs z0 Metabolic)...")
    Z1 = df_sub[['z_mean_3', 'z_mean_0']].values
    V1 = df_sub[['v_3', 'v_0']].values
    D1 = df_death[['z_mean_3', 'z_mean_0']].values
    
    XX1, YY1, U1, W1 = build_interpolated_grid(Z1, V1)
    plot_custom_streamplot(XX1, YY1, U1, W1, Z1, D1, 
                           "Latent Disentanglement Flow: Physical vs Cardiometabolic",
                           "Latent Dimension 3 (Master Frailty Axis)",
                           "Latent Dimension 0 (Cardiometabolic Axis)",
                           "streamplot_disentanglement.png")
    
    # ---------------------------------------------------------------
    # Plot 2 — Phase Portrait (z3 Position vs v3 Velocity)
    #   Interpretation: classical dynamical-systems portrait; spirals
    #   indicate oscillation, fixed points indicate equilibria, saddle
    #   separatrices indicate tipping points.
    # ---------------------------------------------------------------
    print("\n[2/8] Phase Portrait (z3 vs v3)...")
    Z2 = df_sub[['z_mean_3', 'v_3']].values
    V2 = df_sub[['v_3', 'a_3']].values
    D2 = df_death[['z_mean_3', 'v_3']].values
    
    XX2, YY2, U2, W2 = build_interpolated_grid(Z2, V2)
    plot_custom_streamplot(XX2, YY2, U2, W2, Z2, D2,
                           "Biological Phase Portrait: State vs Momentum",
                           "Latent Position (z3: Physical State)",
                           "Latent Velocity (v3: Decline Speed)",
                           "streamplot_phase_portrait.png")
    
    # ---------------------------------------------------------------
    # Plot 3 — PCA-Projected Global Velocity Field
    #   Interpretation: projects all 8 latent dims and all 8 velocity
    #   components into the top-2 principal components, giving a
    #   single summary field of overall aging flow.
    # ---------------------------------------------------------------
    print("\n[3/8] UMAP-Projected Global Velocity Field...")
    
    Z_full = df_sub[[f'z_mean_{k}' for k in range(8)]].values
    V_full = df_sub[[f'v_{k}' for k in range(8)]].values
    
    # 1. Fit UMAP on the positions
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.1, random_state=42)
    Z_umap = reducer.fit_transform(Z_full)
    
    # 2. Project mortality points
    Z_death_full = df_death[[f'z_mean_{k}' for k in range(8)]].values
    D_umap = reducer.transform(Z_death_full)
    
    # 3. For velocity in UMAP, we visualize "Total Aging Speed" 
    total_speed = np.sqrt(np.sum(V_full**2, axis=1))
    
    # We use build_interpolated_grid but pass total_speed as both U and W to 
    # generate a heatmap-like streamplot or just use plot_speed_heatmap directly.
    # The prompt suggests: "Let's visualize the 'Total Aging Speed' flowing over the UMAP islands"
    # Using total_speed as U and W in build_interpolated_grid might not be ideal for streamplot,
    # but let's follow the spirit of "Total Aging Speed" for the global field.
    # Actually, projecting V into UMAP space is tricky. A common hack is to use 
    # specific V dimensions or the speed.
    
    # For Plot 3 (streamplot), let's use a dummy velocity or a specific one if projecting V is hard.
    # The prompt mentions: "Just pass the high-D V magnitudes or a specific V component to interpolate"
    # Let's use the first two V components scaled or something similar for the "flow".
    # Or just use the speed heatmap for Plot 7 and a speed-colored streamplot for 3.
    # Let's try to project V using the Jacobian if we had it, but we don't.
    # Alternative: interpolate 8D V over the 2D UMAP grid?
    
    # Actually, the user suggested:
    # "You can pass total_speed as both U and W just to generate a heatmap over UMAP"
    # Let's do that for the grid-based flow visualization if they want a streamplot.
    
    V_umap_dummy = np.column_stack([total_speed, total_speed])
    
    XX3, YY3, U3, W3 = build_interpolated_grid(Z_umap, V_umap_dummy)
    plot_custom_streamplot(XX3, YY3, U3, W3, Z_umap, D_umap,
                           "Global Aging Flow over UMAP Islands",
                           "UMAP Dim 1",
                           "UMAP Dim 2",
                           "streamplot_umap_global.png")
    
    # ---------------------------------------------------------------
    # Plot 4 — Dominant Decline Axis Phase Portrait (z7)
    #   Dim 7 has the strongest velocity signal and a negative mean
    #   velocity.  Its phase portrait reveals whether cognitive/global
    #   decline is a one-way slide or has restoring dynamics.
    # ---------------------------------------------------------------
    print("\n[4/8] Dominant Decline Phase Portrait (z7 vs v7)...")
    Z4 = df_sub[['z_mean_7', 'v_7']].values
    V4 = df_sub[['v_7', 'a_7']].values
    D4 = df_death[['z_mean_7', 'v_7']].values
    
    XX4, YY4, U4, W4 = build_interpolated_grid(Z4, V4)
    plot_custom_streamplot(XX4, YY4, U4, W4, Z4, D4,
                           "Dominant Decline Phase Portrait (Dim 7)",
                           "Latent Position (z7: Global Decline Axis)",
                           "Latent Velocity (v7: Decline Rate)",
                           "streamplot_phase_dim7.png")
    
    # ---------------------------------------------------------------
    # Plot 5 — Competing Dynamics (z3 vs z6)
    #   v_3 and v_6 are the most anti-correlated velocity pair (r=-0.10),
    #   suggesting they capture opposing aging processes — perhaps
    #   compensatory reserve vs. accumulated damage.
    # ---------------------------------------------------------------
    print("\n[5/8] Competing Dynamics (z3 vs z6)...")
    Z5 = df_sub[['z_mean_3', 'z_mean_6']].values
    V5 = df_sub[['v_3', 'v_6']].values
    D5 = df_death[['z_mean_3', 'z_mean_6']].values
    
    XX5, YY5, U5, W5 = build_interpolated_grid(Z5, V5)
    plot_custom_streamplot(XX5, YY5, U5, W5, Z5, D5,
                           "Competing Dynamics: Dim 3 vs Dim 6",
                           "Latent Dimension 3 (Physical/General Frailty)",
                           "Latent Dimension 6 (Compensatory/Adaptive)",
                           "streamplot_competing.png")
    
    # ---------------------------------------------------------------
    # Plot 6 — Functional Spread vs Biological Axis (z5 vs z0)
    #   Dim 5 has the 2nd-widest spread; dim 0 is another high-spread
    #   axis.  Together they show functional–biological interplay.
    # ---------------------------------------------------------------
    print("\n[6/8] Functional vs Biological (z5 vs z0)...")
    Z6 = df_sub[['z_mean_5', 'z_mean_0']].values
    V6 = df_sub[['v_5', 'v_0']].values
    D6 = df_death[['z_mean_5', 'z_mean_0']].values
    
    XX6, YY6, U6, W6 = build_interpolated_grid(Z6, V6)
    plot_custom_streamplot(XX6, YY6, U6, W6, Z6, D6,
                           "Functional–Biological Interplay (Dim 5 vs Dim 0)",
                           "Latent Dimension 5 (Functional Spread)",
                           "Latent Dimension 0 (Biological Axis)",
                           "streamplot_functional_bio.png")
    
    # ---------------------------------------------------------------
    # Plot 7 — Aging Speed Heatmap over PCA space
    #   Interpretation: colour = median total velocity magnitude at
    #   each location.  Hot zones = regions of rapid aging; cold zones
    #   = regions of stasis or resilience.
    # ---------------------------------------------------------------
    print("\n[7/8] Aging Speed Heatmap over UMAP Islands...")
    # total_speed already calculated in [3/8]
    
    plot_speed_heatmap(Z_umap, total_speed, D_umap,
                       "Aging Speed Landscape over UMAP Islands",
                       "UMAP Dim 1",
                       "UMAP Dim 2",
                       "heatmap_aging_speed_umap.png")
    
    # ---------------------------------------------------------------
    # Plot 8 — Cross-Domain Phase Plane (Physical z3 vs Mental z1)
    #   Dim 1 captures a separate axis (different from cognitive dim7).
    #   This shows how mental/affective state co-evolves with physical
    #   frailty — potential to identify psychosomatic coupling.
    # ---------------------------------------------------------------
    print("\n[8/8] Cross-Domain Phase Plane (z4 Cognitive vs z1 Mental)...")
    Z8 = df_sub[['z_mean_4', 'z_mean_1']].values
    V8 = df_sub[['v_4', 'v_1']].values
    D8 = df_death[['z_mean_4', 'z_mean_1']].values
    
    XX8, YY8, U8, W8 = build_interpolated_grid(Z8, V8)
    plot_custom_streamplot(XX8, YY8, U8, W8, Z8, D8,
                           "Cross-Domain Flow: Cognitive vs Mental State",
                           "Latent Dimension 4 (Cognitive Disentanglement)",
                           "Latent Dimension 1 (Mental/Affective Axis)",
                           "streamplot_cognitive_mental.png")
    
    print("\n✓ All 8 plots generated.")

if __name__ == "__main__":
    main()
