import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from _paths import DATA_DIR, MODELS_DIR

class FrailtyDataset(Dataset):
    def __init__(self, csv_file, device='cpu'):
        """
        Args:
            csv_file (string): Path to the processed frailty_index_data.csv.
        """
        self.data = pd.read_csv(csv_file)
        self.device = device
        
        # Define columns
        self.static_cols = ['edad', 'sexo', 'educacion']
        
        # 36 Deficit columns based on prepare_frailty_data.py
        self.deficit_cols = [
            'hipertension', 'diabetes', 'enf_pulm', 'artritis', 'infarto', 'embolia', 'cancer', 'salud_glob', # 8 Clinical
            'n_abvd', 'n_aivd', 'n_mov', 'n_img', 'motoras_gruesas', 'motoras_finas', # 6 Func
            'deprimido', 'esfuerzo', 'intranquilo', 'triste', 'cansado', 'solo', 'feliz', 'disf_vida', 'energia', # 9 Mental
            'recuerdo1', 'recuerdo2', 'copiafiguras1', 'copiafiguras2', 'orientacion', 'serial7', 'visualscan', 'memoria', # 8 Cog
            'bmi_imp', 'ejer_3_por_sem', 'tabaco', # 3 Bio
            'hospitalizacion', 'visita_medica' # 2 Health
        ]
        
        # 1b. Filter for valid clinical data only (Drop fallecido rows/mask where clinical items are NaN)
        initial_len = len(self.data)
        self.data = self.data.dropna(subset=['FI']).copy()
        if len(self.data) < initial_len:
            print(f"Dropped {initial_len - len(self.data)} rows with missing clinical data (e.g., post-mortem waves).")

        # 1c. Preprocess static covariates (normalization and basic imputation)
        self.data['edad'] = self.data['edad'].fillna(self.data['edad'].median())
        self.data['educacion'] = self.data['educacion'].fillna(self.data['educacion'].median())
        
        self.data['edad'] = (self.data['edad'] - self.data['edad'].mean()) / self.data['edad'].std()
        self.data['educacion'] = (self.data['educacion'] - self.data['educacion'].mean()) / self.data['educacion'].std()
        self.data['sexo'] = self.data['sexo'] - 1.0 # 1=Male, 2=Female -> 0=Male, 1=Female
        
        # Convert to tensors
        self.x_static = torch.tensor(self.data[self.static_cols].values, dtype=torch.float32)
        self.x_deficits = torch.tensor(self.data[self.deficit_cols].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x_deficits[idx].to(self.device), self.x_static[idx].to(self.device)

class BetaVAE(nn.Module):
    def __init__(self, input_dim=36, static_dim=3, hidden_dims=[64, 32], latent_dim=8):
        super(BetaVAE, self).__init__()
        
        self.latent_dim = latent_dim
        encoder_input_dim = input_dim + static_dim
        
        # Encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.LeakyReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[1], latent_dim)
        
        # Decoder
        # Decodes K back to 34 (it does NOT reconstruct static covariates)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid() # Bounds output to [0, 1] mapped deficit space
        )

    def encode(self, x_deficits, x_static):
        # Concatenate x(t) and static covariates for the encoder
        x_in = torch.cat([x_deficits, x_static], dim=1)
        hidden = self.encoder_mlp(x_in)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder_mlp(z)

    def forward(self, x_deficits, x_static):
        mu, log_var = self.encode(x_deficits, x_static)
        z = self.reparameterize(mu, log_var)
        recon_batch = self.decode(z)
        return recon_batch, mu, log_var

def beta_vae_loss(recon_x, x, mu, log_var, beta=4.0, feature_weights=None):
    """
    Custom Loss Function for beta-VAE with Feature-Weighted MSE.
    L = sum_j( w_j * MSE_j(x, x^) ) + beta * D_KL
    """
    if feature_weights is not None:
        # Element-wise MSE (B, D) then mean across batch (D,) then weighted sum
        mse_unreduced = nn.functional.mse_loss(recon_x, x, reduction='none')
        mse_dim = mse_unreduced.mean(dim=0) # Average over batch
        MSE = torch.sum(feature_weights * mse_dim) * x.size(0) # Scale back up to match reduction='sum' format
    else:
        MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total Beta-weighted Loss
    return MSE + beta * KLD, MSE, KLD

def train_model(model, dataloader, dataset, epochs=100, learning_rate=1e-3, target_beta=4.0, device='cpu'):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # PHASE 8: INVERSE-VARIANCE WEIGHTING
    print("\n[Phase 8] Calculating Empirical Variance Weights...")
    all_deficits = dataset.x_deficits.numpy()
    feature_vars = np.var(all_deficits, axis=0) # Variance per feature
    feature_vars = np.clip(feature_vars, a_min=1e-5, a_max=None)
    inv_vars = (1.0 / feature_vars) ** 2 # Fix C: Squared inverse weights to prioritize subtle signals
    weights_normalized = inv_vars * (len(feature_vars) / np.sum(inv_vars))
    tensor_weights = torch.tensor(weights_normalized, dtype=torch.float32).to(device)
    
    print(f"Inverse-Weights Calculated. Min Weight: {weights_normalized.min():.3f} | Max Weight: {weights_normalized.max():.3f}")
    print(f"\nStarting Pipeline Reset: Epochs={epochs}, LR={learning_rate}, Target_Beta={target_beta}")
    
    # Monitoring variables
    anneal_start = 10
    anneal_end = 60
    
    for epoch in range(epochs):
        # Calculate Annealed Beta
        if epoch < anneal_start:
            beta = 0.0  # Pure Autoencoder phase
        elif epoch < anneal_end:
            # Linear ramp from 0 to target_beta
            beta = target_beta * (epoch - anneal_start) / (anneal_end - anneal_start)
        else:
            beta = target_beta # Stay at target_beta
            
        train_loss = 0
        train_mse = 0
        train_kld = 0
        all_mu = []
        
        for batch_idx, (b_deficits, b_static) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, log_var = model(b_deficits, b_static)
            all_mu.append(mu.detach().cpu().numpy())
            
            # Calculate feature-weighted loss with annealed beta
            loss, mse, kld = beta_vae_loss(recon_batch, b_deficits, mu, log_var, beta=beta, feature_weights=tensor_weights)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mse += mse.item()
            train_kld += kld.item()
            
        # Calculate Empirical Latent Expansion (Standard Deviation of Means)
        latent_mu = np.vstack(all_mu)
        latent_std = np.mean(np.std(latent_mu, axis=0))
        
        print(f"Epoch [{epoch+1}/{epochs}] | Beta: {beta:.2f} | Latent Std: {latent_std:.4f} "
              f"| MSE: {train_mse/len(dataloader.dataset):.4f} | KLD: {train_kld/len(dataloader.dataset):.4f}")
              
    return model

if __name__ == "__main__":
    print("Testing Architecture & Starting Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    vae = BetaVAE(latent_dim=8).to(device)
    
    try:
        data_path = str(DATA_DIR / 'frailty_index_data.csv')
            
        dataset = FrailtyDataset(data_path, device=device)
        print(f"Dataset successfully loaded from {data_path}. Total records: {len(dataset)}")
        
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        # Fix B: Relaxed bottleneck (target_beta=0.01) over more epochs for better reconstruction
        trained_vae = train_model(vae, dataloader, dataset, epochs=100, learning_rate=5e-4, target_beta=0.01, device=device)
        
        # Save the model
        model_path = str(MODELS_DIR / 'beta_vae_model.pth')
        torch.save(trained_vae.state_dict(), model_path)
        print(f"Trained model saved to {model_path}")
        
    except FileNotFoundError:
        print("frailty_index_data.csv not found. Run prepare_frailty_data.py first.")
