import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "engine"))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from _paths import DATA_DIR, MODELS_DIR, LATENT_DIR


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
    print(f"Encoded {len(df):,} patients.")
    return df, z_cols


def plot_all_umap(df_merged, z_cols):
    print("Computing UMAP embedding...")
    z_scaled = StandardScaler().fit_transform(df_merged[z_cols].values)
    reducer  = umap.UMAP(n_neighbors=200, min_dist=0.5, n_components=2,
                         random_state=42, metric='euclidean')
    z_emb = reducer.fit_transform(z_scaled)

    def clamp(series):
        return series.quantile(0.05), series.quantile(0.95)

    LATENT_DIR.mkdir(parents=True, exist_ok=True)

    print("Plotting 1: Frailty Index")
    plt.figure(figsize=(10, 8))
    vmin, vmax = clamp(df_merged['FI'])
    sc = plt.scatter(z_emb[:, 0], z_emb[:, 1],
                     c=df_merged['FI'], cmap='inferno', s=10, alpha=0.6,
                     vmin=vmin, vmax=vmax)
    plt.colorbar(sc, label='Frailty Index (FI)')
    plt.title('Latent ODE-VAE — UMAP by Frailty Index')
    plt.grid(True, alpha=0.3)
    plt.savefig(str(LATENT_DIR / 'umap_frailty_index.png'), dpi=150)
    plt.close()

    print("Plotting 2: Chronological Age")
    plt.figure(figsize=(10, 8))
    vmin, vmax = clamp(df_merged['edad'])
    sc = plt.scatter(z_emb[:, 0], z_emb[:, 1],
                     c=df_merged['edad'], cmap='viridis', s=10, alpha=0.6,
                     vmin=vmin, vmax=vmax)
    plt.colorbar(sc, label='Chronological Age (Years)')
    plt.title('Latent ODE-VAE — UMAP by Chronological Age')
    plt.grid(True, alpha=0.3)
    plt.savefig(str(LATENT_DIR / 'umap_age.png'), dpi=150)
    plt.close()

    print("Plotting 3: Mortality Event Horizon")
    plt.figure(figsize=(10, 8))
    terminal = df_merged['is_terminal'] == 1
    plt.scatter(z_emb[~terminal, 0], z_emb[~terminal, 1],
                c='lightgray', s=10, alpha=0.3, label='Survivors / Stable')
    plt.scatter(z_emb[terminal, 0],  z_emb[terminal, 1],
                c='red', marker='x', s=25, alpha=0.9, label='Terminal State (Died)')
    plt.legend()
    plt.title('Latent ODE-VAE — UMAP Mortality Event Horizon')
    plt.grid(True, alpha=0.3)
    plt.savefig(str(LATENT_DIR / 'umap_mortality.png'), dpi=150)
    plt.close()

    print("Plotting 4: Cognitive vs Physical domains")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
    for ax, col, label, title in [
        (axes[0], 'recuerdo1', 'Word Recall Score',  'Cognitive Function (UMAP)'),
        (axes[1], 'n_abvd',   '# ADL Difficulties', 'Physical Independence (UMAP)'),
    ]:
        vmin, vmax = clamp(df_merged[col])
        sc = ax.scatter(z_emb[:, 0], z_emb[:, 1],
                        c=df_merged[col], cmap='coolwarm', s=10, alpha=0.6,
                        vmin=vmin, vmax=vmax)
        fig.colorbar(sc, ax=ax, label=label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(LATENT_DIR / 'umap_domains.png'), dpi=150)
    plt.close()

    print(f"Saved 4 UMAP plots → {LATENT_DIR}")
    return z_emb


def plot_categorical_islands(df_merged, z_emb):
    print("Plotting categorical island analysis...")
    categorical_cols = ['sexo', 'hipertension', 'diabetes', 'tabaco',
                        'alcohol', 'ejer_3_por_sem', 'social_isolation']
    for col in categorical_cols:
        if col not in df_merged.columns:
            continue
        plt.figure(figsize=(10, 8))
        temp   = df_merged[col].dropna().round()
        cats   = sorted(temp.unique())
        cmap   = plt.get_cmap('Set1')
        for i, cat in enumerate(cats):
            m = df_merged[col].round() == cat
            plt.scatter(z_emb[m, 0], z_emb[m, 1],
                        color=cmap(i % 9), s=15, alpha=0.7, label=f'{col}={int(cat)}')
        plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f'UMAP Island Analysis: {col}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = LATENT_DIR / f'umap_island_{col}.png'
        plt.savefig(str(path), dpi=150)
        plt.close()
        print(f"  -> {path.name}")


def plot_aging_trajectories(df_merged, z_emb, n_patients=30):
    """
    Draw longitudinal arrows in UMAP space.
    Since one embedding point exists per patient (z0), trajectories are
    approximated by connecting patients' z0 positions across repeated visits
    using the trajectory CSV if available, or skipped otherwise.
    """
    print(f"Plotting aging trajectories for {n_patients} sampled patients...")
    df_merged = df_merged.copy()
    df_merged['umap_x'] = z_emb[:, 0]
    df_merged['umap_y'] = z_emb[:, 1]

    plt.figure(figsize=(12, 10))
    plt.scatter(df_merged['umap_x'], df_merged['umap_y'],
                c='lightgray', s=10, alpha=0.1, zorder=1)

    np.random.seed(42)
    sampled = df_merged.sample(min(n_patients, len(df_merged)), random_state=42)
    cmap    = plt.get_cmap('tab20')

    for i, (_, row) in enumerate(sampled.iterrows()):
        color = cmap(i % 20)
        plt.scatter(row['umap_x'], row['umap_y'],
                    color=color, s=40, zorder=3,
                    edgecolors='black', linewidth=0.5)

    plt.title(f'Latent ODE-VAE — UMAP Patient Embedding ({n_patients} sampled)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = LATENT_DIR / 'umap_patient_embeddings.png'
    plt.savefig(str(path), dpi=200)
    plt.close()
    print(f"  -> {path.name}")


def main():
    df_latent, z_cols = extract_latent_vectors_ode()

    df_orig   = pd.read_csv(DATA_DIR / 'frailty_index_data.csv')
    keep_cols = ['cunicah', 'np', 'FI', 'edad', 'fallecido', 'recuerdo1', 'n_abvd',
                 'sexo', 'hipertension', 'diabetes', 'tabaco', 'alcohol',
                 'ejer_3_por_sem', 'asiste_club', 'voluntario']
    valid_cols = [c for c in keep_cols if c in df_orig.columns]

    dead_ids = df_orig[df_orig['fallecido'] == 1]['cunicah'].unique()

    df_agg = (df_orig[valid_cols]
              .groupby(['cunicah', 'np'])
              .agg(**{c: (c, 'mean') for c in valid_cols if c not in ('cunicah', 'np')},
                   fallecido=('fallecido', 'max'))
              .reset_index())

    df_merged = df_latent.merge(df_agg, on=['cunicah', 'np'], how='left')
    df_merged['is_terminal'] = df_merged['cunicah'].isin(dead_ids).astype(int)

    if 'asiste_club' in df_merged.columns and 'voluntario' in df_merged.columns:
        df_merged['social_isolation'] = (
            1.0 - df_merged[['asiste_club', 'voluntario']].max(axis=1)
        )

    df_merged = df_merged.dropna(subset=['FI', 'edad', 'fallecido'])

    z_emb = plot_all_umap(df_merged, z_cols)
    plot_categorical_islands(df_merged, z_emb)
    plot_aging_trajectories(df_merged, z_emb, n_patients=30)


if __name__ == "__main__":
    main()
