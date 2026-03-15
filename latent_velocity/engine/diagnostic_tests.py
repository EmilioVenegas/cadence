import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_vae import BetaVAE, FrailtyDataset
from _paths import DATA_DIR, MODELS_DIR

# Define Domain Indices mapping to the 34 deficit variables
deficit_cols = [
    'hipertension', 'diabetes', 'enf_pulm', 'artritis', 'infarto', 'embolia', 'cancer', 'salud_glob', # 8 Clinical (0-7)
    'n_abvd', 'n_aivd', 'n_mov', 'n_img', 'motoras_gruesas', 'motoras_finas', # 6 Phys (8-13)
    'deprimido', 'esfuerzo', 'intranquilo', 'triste', 'cansado', 'solo', 'feliz', 'disf_vida', 'energia', # 9 Mental (14-22)
    'recuerdo1', 'recuerdo2', 'copiafiguras1', 'orientacion', 'serial7', 'memoria', # 6 Cog (23-28)
    'bmi_imp', 'ejer_3_por_sem', 'tabaco', # 3 Bio (29-31)
    'hospitalizacion', 'visita_medica' # 2 Health (32-33)
]

domain_indices = {
    'Clinical': list(range(0, 8)),
    'Physical': list(range(8, 14)),
    'Mental': list(range(14, 23)),
    'Cognitive': list(range(23, 29))
}

def test_1_per_domain_loss(model_path, data_path, device):
    print("\n--- TEST 1: Per-Domain Reconstruction Loss ---")
    vae = BetaVAE(latent_dim=8).to(device)
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
    vae = BetaVAE(latent_dim=8).to(device)
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = str(MODELS_DIR / 'beta_vae_model.pth')
    data_path = str(DATA_DIR / 'frailty_index_data.csv')
    
    test_1_per_domain_loss(model_path, data_path, device)
    test_2_input_variance(data_path)
    test_3_reconstruction_accuracy_r2(model_path, data_path, device)
