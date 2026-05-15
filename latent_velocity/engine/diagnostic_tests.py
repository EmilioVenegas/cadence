import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_vae import BetaVAE, FrailtyDataset
from _paths import DATA_DIR, MODELS_DIR

# Define Domain Indices mapping to the 34 deficit variables used by the VAE.
# tabaco and ejer_3_por_sem were removed from the VAE input space (they are
# behavioral controls, not biological states) and now live only in ODE U_COLS.
deficit_cols = [
    'hipertension', 'diabetes', 'enf_pulm', 'artritis', 'infarto', 'embolia', 'cancer', 'salud_glob', # 8 Clinical (0-7)
    'n_abvd', 'n_aivd', 'n_mov', 'n_img', 'motoras_gruesas', 'motoras_finas', # 6 Phys (8-13)
    'deprimido', 'esfuerzo', 'intranquilo', 'triste', 'cansado', 'solo', 'feliz', 'disf_vida', 'energia', # 9 Mental (14-22)
    'recuerdo1', 'recuerdo2', 'copiafiguras1', 'copiafiguras2', 'orientacion', 'serial7', 'visualscan', 'memoria', # 8 Cog (23-30)
    'bmi_imp', # 1 Bio (31)
    'hospitalizacion', 'visita_medica' # 2 Health (32-33)
]

domain_indices = {
    'Clinical': list(range(0, 8)),
    'Physical': list(range(8, 14)),
    'Mental': list(range(14, 23)),
    'Cognitive': list(range(23, 31))
}

def test_1_per_domain_loss(model_path, data_path, device):
    print("\n--- TEST 1: Per-Domain Reconstruction Loss ---")
    vae = BetaVAE(input_dim=34, latent_dim=8).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    vae.eval()
    
    dataset = FrailtyDataset(data_path, device=device)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    domain_mse_sums = {dom: 0.0 for dom in domain_indices.keys()}
    total_elements = {dom: 0 for dom in domain_indices.keys()}
    
    with torch.no_grad():
        for b_deficits, b_static in dataloader:
            recon, _, _ = vae(b_deficits, b_static)
            
            for dom, indices in domain_indices.items():
                x_sub = b_deficits[:, indices]
                recon_sub = recon[:, indices]
                
                mse_sum = F.mse_loss(recon_sub, x_sub, reduction='sum').item()
                domain_mse_sums[dom] += mse_sum
                total_elements[dom] += (b_deficits.size(0) * len(indices))
                
    print("Average MSE Per-Domain:")
    for dom in domain_indices.keys():
        avg_mse = domain_mse_sums[dom] / total_elements[dom]
        print(f"  {dom:<10}: {avg_mse:.5f}")

def test_2_input_variance(data_path):
    print("\n--- TEST 2: Input Variance Sanity Check ---")
    df = pd.read_csv(data_path)
    
    # Calculate variance for each of the 34 features
    variances = df[deficit_cols].var()
    
    print("Mean Empirical Variance Per-Domain:")
    for dom, indices in domain_indices.items():
        dom_cols = [deficit_cols[i] for i in indices]
        mean_var = variances[dom_cols].mean()
        max_var = variances[dom_cols].max()
        min_var = variances[dom_cols].min()
        print(f"  {dom:<10}: Mean={mean_var:.5f} | Range=[{min_var:.5f}, {max_var:.5f}]")
        
    print("\nVariance Warning Analysis:")
    mental_cog_ratio = variances[[deficit_cols[i] for i in domain_indices['Mental']]].mean() / \
                       variances[[deficit_cols[i] for i in domain_indices['Cognitive']]].mean()
    print(f"  Mental-to-Cognitive Variance Ratio: {mental_cog_ratio:.2f}x")
    if mental_cog_ratio > 3.0:
        print("  => VAE ALERT: Mental Health variance is heavily dominating Cognitive variance. Disentanglement collapse likely.")

def test_3_reconstruction_accuracy_r2(model_path, data_path, device):
    print("\n--- TEST 3: Explained Variance (R2 Accuracy) ---")
    vae = BetaVAE(input_dim=34, latent_dim=8).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    vae.eval()
    
    dataset = FrailtyDataset(data_path, device=device)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # 1. Get MSE per domain
    domain_mse_sums = {dom: 0.0 for dom in domain_indices.keys()}
    total_elements = {dom: 0 for dom in domain_indices.keys()}
    
    with torch.no_grad():
        for b_deficits, b_static in dataloader:
            recon, _, _ = vae(b_deficits, b_static)
            for dom, indices in domain_indices.items():
                x_sub = b_deficits[:, indices]
                recon_sub = recon[:, indices]
                domain_mse_sums[dom] += F.mse_loss(recon_sub, x_sub, reduction='sum').item()
                total_elements[dom] += (b_deficits.size(0) * len(indices))
                
    # 2. Get Variance per domain (from the same cleaned dataset)
    # We use the dataset's cleaned data for consistency
    df_clean = dataset.data[deficit_cols]
    variances = df_clean.var()
    
    # 3. Calculate R2 = 1 - (MSE / Variance)
    print("Explained Variance (R2) Per-Domain:")
    for dom, indices in domain_indices.items():
        avg_mse = domain_mse_sums[dom] / total_elements[dom]
        
        dom_cols = [deficit_cols[i] for i in indices]
        mean_var = variances[dom_cols].mean()
        
        r2 = 1 - (avg_mse / mean_var)
        print(f"  {dom:<10}: R2={r2:.4f} ({r2*100:4.1f}% accurate)")

def test_4_latent_statistics(model_path, data_path, device):
    print("\n--- TEST 4: Latent Space Statistics ---")
    vae = BetaVAE(input_dim=34, latent_dim=8).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    vae.eval()
    
    dataset = FrailtyDataset(data_path, device=device)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_mus = []
    all_logvars = []
    
    with torch.no_grad():
        for b_deficits, b_static in dataloader:
            mu, log_var = vae.encode(b_deficits, b_static)
            all_mus.append(mu.cpu().numpy())
            all_logvars.append(log_var.cpu().numpy())
            
    mus = np.vstack(all_mus)
    logvars = np.vstack(all_logvars)
    stds = np.exp(0.5 * logvars)
    
    mu_means = mus.mean(axis=0)
    mu_stds = mus.std(axis=0)
    avg_posterior_std = stds.mean(axis=0)
    
    active_dims = np.sum(mu_stds > 0.1) # Dimension is active if it encodes variation
    
    print(f"Active Dimensions (>0.1 std): {active_dims} / {vae.latent_dim}")
    print("\nPer-Dimension Stats (mu_mean | mu_std | avg_posterior_std):")
    for i in range(vae.latent_dim):
        print(f"  Z{i}: mean={mu_means[i]:.3f} | std={mu_stds[i]:.3f} | post_std={avg_posterior_std[i]:.3f}")

def test_5_feature_level_r2(model_path, data_path, device):
    print("\n--- TEST 5: Feature-Level Reconstruction Analysis ---")
    vae = BetaVAE(input_dim=34, latent_dim=8).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    vae.eval()
    
    dataset = FrailtyDataset(data_path, device=device)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_recons = []
    all_originals = []
    
    with torch.no_grad():
        for b_deficits, b_static in dataloader:
            recon, _, _ = vae(b_deficits, b_static)
            all_recons.append(recon.cpu().numpy())
            all_originals.append(b_deficits.cpu().numpy())
            
    recons = np.vstack(all_recons)
    originals = np.vstack(all_originals)
    
    # Calculate R2 per feature
    feature_r2 = []
    for i in range(len(deficit_cols)):
        mse = np.mean((recons[:, i] - originals[:, i])**2)
        var = np.var(originals[:, i])
        r2 = 1 - (mse / var) if var > 1e-6 else 0.0
        feature_r2.append((deficit_cols[i], r2))
        
    # Sort features by R2
    feature_r2.sort(key=lambda x: x[1], reverse=True)
    
    # Overall Reconstruction R2 (Unweighted Average)
    overall_r2 = np.mean([r[1] for r in feature_r2])
    print(f"OVERALL RECONSTRUCTION R2: {overall_r2:.4f}")
    
    print("\nTop 5 Best Reconstructed Features:")
    for name, r2 in feature_r2[:5]:
        print(f"  {name:<20}: R2={r2:.4f}")
        
    print("\nTop 5 Worst Reconstructed Features:")
    for name, r2 in feature_r2[-5:]:
        print(f"  {name:<20}: R2={r2:.4f}")

def test_6_save_reconstruction_examples(model_path, data_path, device, num_samples=3):
    print(f"\n--- TEST 6: Saving Reconstruction Examples (n={num_samples}) ---")
    vae = BetaVAE(input_dim=34, latent_dim=8).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    vae.eval()
    
    dataset = FrailtyDataset(data_path, device=device)
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    
    b_deficits, b_static = next(iter(dataloader))
    
    with torch.no_grad():
        recon, _, _ = vae(b_deficits, b_static)
        
    recon = recon.cpu().numpy()
    orig = b_deficits.cpu().numpy()
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4 * num_samples))
    if num_samples == 1: axes = [axes]
    
    x = np.arange(len(deficit_cols))
    
    for i in range(num_samples):
        axes[i].bar(x - 0.2, orig[i], width=0.4, label='Original', color='blue', alpha=0.6)
        axes[i].bar(x + 0.2, recon[i], width=0.4, label='Reconstructed', color='red', alpha=0.6)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(deficit_cols, rotation=90, fontsize=8)
        axes[i].set_title(f"Sample Patient {i+1} Reconstruction")
        axes[i].legend()
        axes[i].set_ylim(0, 1.1)
        
    plt.tight_layout()
    save_path = str(MODELS_DIR / 'vae_reconstruction_samples.png')
    plt.savefig(save_path)
    print(f"Reconstruction comparison plot saved to: {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = str(MODELS_DIR / 'beta_vae_model_128.pth')
    data_path = str(DATA_DIR / 'frailty_index_data.csv')
    
    test_1_per_domain_loss(model_path, data_path, device)
    test_2_input_variance(data_path)
    test_3_reconstruction_accuracy_r2(model_path, data_path, device)
    test_4_latent_statistics(model_path, data_path, device)
    test_5_feature_level_r2(model_path, data_path, device)
    test_6_save_reconstruction_examples(model_path, data_path, device)
