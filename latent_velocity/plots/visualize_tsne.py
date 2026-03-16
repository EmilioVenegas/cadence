import sys
from pathlib import Path

# Add the engine directory to the Python path for imports
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "engine"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from extract_velocity import extract_latent_vectors
from _paths import DATA_DIR, MODELS_DIR, PLOTS_DIR, TSNE_DIR

from sklearn.preprocessing import StandardScaler

def plot_all_tsne(df_merged, z_cols):
    print("Computing t-SNE embedding. This may take a minute...")
    
    # 1. Standardize the latent space before feeding to t-SNE
    z_data = df_merged[z_cols].values
    z_scaled = StandardScaler().fit_transform(z_data)
    
    # 2. Use PCA initialization and higher perplexity for global structure
    tsne = TSNE(
        n_components=2, 
        perplexity=80,       # Increased from 30
        init='pca',          # CRITICAL: Preserves global gradients like aging
        learning_rate='auto',
        max_iter=1000, 
        random_state=42, 
        n_jobs=-1
    )
    z_embedded = tsne.fit_transform(z_scaled)
    
    # Helper to calculate color limits to prevent outlier washout
    def get_limits(series):
        return series.quantile(0.05), series.quantile(0.95)

    print("Plotting 1: Ground-Truth Frailty Index (FI)")
    plt.figure(figsize=(10, 8))
    vmin, vmax = get_limits(df_merged['FI'])
    scatter = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], 
                          c=df_merged['FI'], cmap='inferno', s=10, alpha=0.6,
                          vmin=vmin, vmax=vmax) # Clamp colors
    plt.colorbar(scatter, label='Frailty Index (FI)')
    plt.title('Latent Space by Frailty Index')
    plt.grid(True, alpha=0.3)
    plt.savefig(str(TSNE_DIR / 'tsne_frailty_index.png'), dpi=150)
    plt.close()
    
    print("Plotting 2: Chronological Age")
    plt.figure(figsize=(10, 8))
    vmin, vmax = get_limits(df_merged['edad'])
    scatter = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], 
                          c=df_merged['edad'], cmap='viridis', s=10, alpha=0.6,
                          vmin=vmin, vmax=vmax) # Clamp colors
    plt.colorbar(scatter, label='Chronological Age (Years)')
    plt.title('Latent Space by Chronological Age')
    plt.grid(True, alpha=0.3)
    plt.savefig(str(TSNE_DIR / 'tsne_age.png'), dpi=150)
    plt.close()

    print("Plotting 3: Mortality / Terminal State")
    plt.figure(figsize=(10, 8))
    # Plot alive first as background
    mask_alive = df_merged['fallecido'] == 0
    plt.scatter(z_embedded[mask_alive, 0], z_embedded[mask_alive, 1], 
                c='lightgray', s=10, alpha=0.3, label='Alive')
    # Plot dead on top with higher opacity
    mask_dead = df_merged['fallecido'] == 1
    plt.scatter(z_embedded[mask_dead, 0], z_embedded[mask_dead, 1], 
                c='red', marker='x', s=25, alpha=0.9, label='Terminal Event (Died)')
    plt.legend()
    plt.title('Mortality "Event Horizon" in Latent Space')
    plt.grid(True, alpha=0.3)
    plt.savefig(str(TSNE_DIR / 'tsne_mortality.png'), dpi=150)
    plt.close()
    
    print("Plotting 4: Domain-Specific Deficits (Cognitive vs Physical)")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
    
    # Cognitive
    vmin_c, vmax_c = get_limits(df_merged['recuerdo1'])
    sc1 = axes[0].scatter(z_embedded[:, 0], z_embedded[:, 1], 
                          c=df_merged['recuerdo1'], cmap='coolwarm', s=10, alpha=0.6,
                          vmin=vmin_c, vmax=vmax_c)
    fig.colorbar(sc1, ax=axes[0], label='Word Recall Score')
    axes[0].set_title('Cognitive Function Projection')
    axes[0].grid(True, alpha=0.3)
    
    # Physical
    vmin_p, vmax_p = get_limits(df_merged['n_abvd'])
    sc2 = axes[1].scatter(z_embedded[:, 0], z_embedded[:, 1], 
                          c=df_merged['n_abvd'], cmap='coolwarm', s=10, alpha=0.6,
                          vmin=vmin_p, vmax=vmax_p)
    fig.colorbar(sc2, ax=axes[1], label='# ADL Difficulties')
    axes[1].set_title('Physical Independence Projection')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(TSNE_DIR / 'tsne_domains.png'), dpi=150)
    plt.close()
    
    print("Successfully generated all 4 t-SNE projections!")


def main():
    model_path = str(MODELS_DIR / 'beta_vae_model.pth')
    data_path = str(DATA_DIR / 'frailty_index_data.csv')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Extracting latent space for the entire dataset...")
    
    df_latent, z_cols = extract_latent_vectors(model_path, data_path, device)
    
    print("Merging latent embeddings with ground-truth clinical features...")
    df_orig = pd.read_csv(data_path)
    
    df_merged = pd.merge(df_latent, df_orig[['cunicah', 'np', 'a_o_ent', 'FI', 'edad', 'fallecido', 'recuerdo1', 'n_abvd']], 
                         on=['cunicah', 'np', 'a_o_ent'], how='left')
    
    plot_all_tsne(df_merged, z_cols)

if __name__ == "__main__":
    main()
