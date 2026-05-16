import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "engine"))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from _paths import DATA_DIR, MODELS_DIR, TSNE_DIR


def extract_latent_vectors_ode():
    """
    Encode every patient in frailty_index_data.csv with the Latent ODE-VAE encoder.
    Returns a DataFrame with columns [cunicah, np, z_0..z_7] — one row per patient.
    """
    from train_latent_ode import LatentODE, LatentODEDataset, MHAS_WAVES, T_MAX, LATENT_DIM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Latent ODE-VAE encoder | device: {device}")

    ckpt  = torch.load(str(MODELS_DIR / 'latent_ode_model.pth'),
                       map_location=device, weights_only=False)
    model = LatentODE().to(device)
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval()

    dataset = LatentODEDataset(
        str(DATA_DIR / 'frailty_index_data.csv'),
        edad_mean=ckpt['edad_mean'], edad_std=ckpt['edad_std'],
        edu_mean=ckpt['edu_mean'],   edu_std=ckpt['edu_std'],
    )

    t_norm_base = torch.tensor([w / T_MAX for w in MHAS_WAVES], dtype=torch.float32).to(device)

    records = []
    with torch.no_grad():
        for s in dataset.samples:
            x    = s['x'].unsqueeze(0).to(device)
            mask = s['mask'].unsqueeze(0).to(device)
            t_norm = t_norm_base.unsqueeze(0)
            mu, _ = model.encode(x, t_norm, mask)
            row = {'cunicah': s['cunicah'], 'np': s['np']}
            for k in range(LATENT_DIM):
                row[f'z_{k}'] = float(mu[0, k].cpu())
            records.append(row)

    df = pd.DataFrame(records)
    z_cols = [f'z_{k}' for k in range(LATENT_DIM)]
    print(f"Encoded {len(df):,} patients. Latent std: "
          f"{df[z_cols].std().mean():.3f}")
    return df, z_cols


def plot_all_tsne(df_merged, z_cols):
    print("Computing t-SNE embedding...")
    z_scaled  = StandardScaler().fit_transform(df_merged[z_cols].values)
    tsne      = TSNE(n_components=2, perplexity=80, init='pca',
                     learning_rate='auto', max_iter=1000,
                     random_state=42, n_jobs=-1)
    z_emb = tsne.fit_transform(z_scaled)

    def clamp(series):
        return series.quantile(0.05), series.quantile(0.95)

    TSNE_DIR.mkdir(parents=True, exist_ok=True)

    print("Plotting 1: Frailty Index")
    plt.figure(figsize=(10, 8))
    vmin, vmax = clamp(df_merged['FI'])
    sc = plt.scatter(z_emb[:, 0], z_emb[:, 1],
                     c=df_merged['FI'], cmap='inferno', s=10, alpha=0.6,
                     vmin=vmin, vmax=vmax)
    plt.colorbar(sc, label='Frailty Index (FI)')
    plt.title('Latent ODE-VAE — t-SNE by Frailty Index')
    plt.grid(True, alpha=0.3)
    plt.savefig(str(TSNE_DIR / 'tsne_frailty_index.png'), dpi=150)
    plt.close()

    print("Plotting 2: Chronological Age")
    plt.figure(figsize=(10, 8))
    vmin, vmax = clamp(df_merged['edad'])
    sc = plt.scatter(z_emb[:, 0], z_emb[:, 1],
                     c=df_merged['edad'], cmap='viridis', s=10, alpha=0.6,
                     vmin=vmin, vmax=vmax)
    plt.colorbar(sc, label='Chronological Age (Years)')
    plt.title('Latent ODE-VAE — t-SNE by Chronological Age')
    plt.grid(True, alpha=0.3)
    plt.savefig(str(TSNE_DIR / 'tsne_age.png'), dpi=150)
    plt.close()

    print("Plotting 3: Mortality")
    plt.figure(figsize=(10, 8))
    alive = df_merged['fallecido'] == 0
    plt.scatter(z_emb[alive, 0],  z_emb[alive, 1],
                c='lightgray', s=10, alpha=0.3, label='Alive')
    plt.scatter(z_emb[~alive, 0], z_emb[~alive, 1],
                c='red', marker='x', s=25, alpha=0.9, label='Deceased')
    plt.legend()
    plt.title('Latent ODE-VAE — t-SNE Mortality Event Horizon')
    plt.grid(True, alpha=0.3)
    plt.savefig(str(TSNE_DIR / 'tsne_mortality.png'), dpi=150)
    plt.close()

    print("Plotting 4: Cognitive vs Physical domains")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
    for ax, col, label, title in [
        (axes[0], 'recuerdo1', 'Word Recall Score',    'Cognitive Function'),
        (axes[1], 'n_abvd',   '# ADL Difficulties',   'Physical Independence'),
    ]:
        vmin, vmax = clamp(df_merged[col])
        sc = ax.scatter(z_emb[:, 0], z_emb[:, 1],
                        c=df_merged[col], cmap='coolwarm', s=10, alpha=0.6,
                        vmin=vmin, vmax=vmax)
        fig.colorbar(sc, ax=ax, label=label)
        ax.set_title(f'{title} Projection')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(TSNE_DIR / 'tsne_domains.png'), dpi=150)
    plt.close()

    print(f"Saved 4 t-SNE plots → {TSNE_DIR}")


def main():
    df_latent, z_cols = extract_latent_vectors_ode()

    df_orig   = pd.read_csv(DATA_DIR / 'frailty_index_data.csv')
    keep_cols = ['cunicah', 'np', 'FI', 'edad', 'fallecido', 'recuerdo1', 'n_abvd']
    # Patient-level aggregates (one row per patient from the trajectory CSV)
    df_agg = (df_orig[keep_cols]
              .groupby(['cunicah', 'np'])
              .agg(FI=('FI', 'mean'), edad=('edad', 'mean'),
                   fallecido=('fallecido', 'max'),
                   recuerdo1=('recuerdo1', 'mean'),
                   n_abvd=('n_abvd', 'mean'))
              .reset_index())

    df_merged = df_latent.merge(df_agg, on=['cunicah', 'np'], how='left')
    df_merged = df_merged.dropna(subset=['FI', 'edad', 'fallecido'])

    plot_all_tsne(df_merged, z_cols)


if __name__ == "__main__":
    main()
