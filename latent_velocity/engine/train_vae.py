import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from _paths import DATA_DIR, MODELS_DIR

class FrailtyDataset(Dataset):
    def __init__(self, csv_file, device='cpu'):
        self.data = pd.read_csv(csv_file)
        self.device = device
        
        self.static_cols = ['edad', 'sexo', 'educacion']
        self.deficit_cols = [
            'hipertension', 'diabetes', 'enf_pulm', 'artritis', 'infarto', 'embolia', 'cancer', 'salud_glob',
            'n_abvd', 'n_aivd', 'n_mov', 'n_img', 'motoras_gruesas', 'motoras_finas',
            'deprimido', 'esfuerzo', 'intranquilo', 'triste', 'cansado', 'solo', 'feliz', 'disf_vida', 'energia',
            'recuerdo1', 'recuerdo2', 'copiafiguras1', 'copiafiguras2', 'orientacion', 'serial7', 'visualscan', 'memoria',
            'bmi_imp',
            'hospitalizacion', 'visita_medica'
        ]
        # tabaco and ejer_3_por_sem are excluded: they are direct behavioral choices, not
        # biological states. Their causal effects are captured by the 32 remaining deficit
        # features (e.g. enf_pulm, infarto for smoking; mobility scores for inactivity).
        # Both variables remain in frailty_index_data.csv for use as ODE control inputs.
        
        initial_len = len(self.data)
        self.data = self.data.dropna(subset=['FI']).copy()
        if len(self.data) < initial_len:
            print(f"Dropped {initial_len - len(self.data)} rows with missing clinical data.")

        self.data['edad'] = self.data['edad'].fillna(self.data['edad'].median())
        self.data['educacion'] = self.data['educacion'].fillna(self.data['educacion'].median())
        
        self.data['edad'] = (self.data['edad'] - self.data['edad'].mean()) / self.data['edad'].std()
        self.data['educacion'] = (self.data['educacion'] - self.data['educacion'].mean()) / self.data['educacion'].std()
        self.data['sexo'] = self.data['sexo'] - 1.0 
        
        self.x_static = torch.tensor(self.data[self.static_cols].values, dtype=torch.float32)
        self.x_deficits = torch.tensor(self.data[self.deficit_cols].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x_deficits[idx].to(self.device), self.x_static[idx].to(self.device)

class BetaVAE(nn.Module):
    def __init__(self, input_dim=36, static_dim=3, hidden_dims=[128, 64, 32], latent_dim=8):
        super(BetaVAE, self).__init__()
        
        self.latent_dim = latent_dim
        encoder_input_dim = input_dim + static_dim
        
        # Expanded Encoder with gentle Dropout
        self.encoder_mlp = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.LeakyReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dims[2], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[2], latent_dim)
        
        # Expanded Decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid() 
        )

    def encode(self, x_deficits, x_static):
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

def beta_vae_loss(recon_x, x, mu, log_var, beta=0.1, feature_weights=None, free_bits=0.5):
    """Beta-VAE Loss with Free Bits."""
    if feature_weights is not None:
        mse_unreduced = nn.functional.mse_loss(recon_x, x, reduction='none')
        mse_dim = mse_unreduced.mean(dim=0) 
        MSE = torch.sum(feature_weights * mse_dim) * x.size(0)
    else:
        MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # Calculate KLD per dimension
    kld_unreduced = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kld_per_latent = kld_unreduced.mean(dim=0)
    
    # Apply Free Bits minimum threshold
    if free_bits > 0.0:
        free_bits_tensor = torch.full_like(kld_per_latent, fill_value=free_bits)
        kld_per_latent = torch.max(kld_per_latent, free_bits_tensor)
        
    KLD = torch.sum(kld_per_latent) * x.size(0)
    
    return MSE + beta * KLD, MSE, KLD

def train_model(model, dataloader, dataset, epochs=120, learning_rate=5e-4, target_beta=0.1, device='cpu'):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("\nCalculating Empirical Variance Weights...")
    all_deficits = dataset.x_deficits.numpy()
    feature_vars = np.var(all_deficits, axis=0)
    feature_vars = np.clip(feature_vars, a_min=1e-5, a_max=None)
    
    # FIXED: Standard inverse variance, no squaring
    inv_vars = 1.0 / feature_vars 
    weights_normalized = inv_vars * (len(feature_vars) / np.sum(inv_vars))
    tensor_weights = torch.tensor(weights_normalized, dtype=torch.float32).to(device)
    
    print(f"Starting Training: Epochs={epochs}, Target_Beta={target_beta}, Free Bits=0.5")
    
    # Extended annealing window
    anneal_start = 20
    anneal_end = 80
    
    for epoch in range(epochs):
        if epoch < anneal_start:
            beta = 0.0  
        elif epoch < anneal_end:
            beta = target_beta * (epoch - anneal_start) / (anneal_end - anneal_start)
        else:
            beta = target_beta 
            
        train_loss, train_mse, train_kld = 0, 0, 0
        all_mu = []
        
        for batch_idx, (b_deficits, b_static) in enumerate(dataloader):
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(b_deficits, b_static)
            all_mu.append(mu.detach().cpu().numpy())
            
            # Pass free_bits=0.5
            loss, mse, kld = beta_vae_loss(recon_batch, b_deficits, mu, log_var, beta=beta, feature_weights=tensor_weights, free_bits=0.5)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mse += mse.item()
            train_kld += kld.item()
            
        latent_mu = np.vstack(all_mu)
        latent_std = np.mean(np.std(latent_mu, axis=0))
        
        print(f"Epoch [{epoch+1}/{epochs}] | Beta: {beta:.3f} | Latent Std: {latent_std:.4f} "
              f"| MSE: {train_mse/len(dataloader.dataset):.4f} | KLD: {train_kld/len(dataloader.dataset):.4f}")
              
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    vae = BetaVAE(input_dim=34, latent_dim=8).to(device)
    
    data_path = str(DATA_DIR / 'frailty_index_data.csv')
    dataset = FrailtyDataset(data_path, device=device)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Target beta set to 0.1 for micro-scale MSE
    trained_vae = train_model(vae, dataloader, dataset, epochs=120, learning_rate=5e-4, target_beta=0.1, device=device)
    
    model_path = str(MODELS_DIR / 'beta_vae_model_128.pth')
    torch.save(trained_vae.state_dict(), model_path)
    print(f"Trained model saved to {model_path}")