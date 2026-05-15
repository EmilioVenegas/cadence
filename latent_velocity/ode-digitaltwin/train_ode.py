import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import os
from _paths import MODELS_DIR

# 1. High-Capacity ODE Function for Exogenous Control
class ODEFunc(nn.Module):
    def __init__(self, latent_dim=8, control_dim=7, hidden_dim=128):
        super(ODEFunc, self).__init__()
        self.control_dim = control_dim
        # Using SiLU (Swish) for better gradient flow in deep networks
        self.net = nn.Sequential(
            nn.Linear(latent_dim + control_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.current_u = None
        self.target_u = None
        self.washout_k = 0.0  # k=0 means constant u = current_u
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        # Calculate u(t) to simulate biological washout
        # u(t) = target_u + (start_u - target_u) * exp(-k * t)
        if self.current_u is None:
            u = torch.zeros(z.size(-2) if z.dim() >= 2 else 1, self.control_dim).to(z.device)
        elif self.target_u is not None and self.washout_k > 0:
            # Shift u(t) from current (baseline) to target (cured)
            u = self.target_u + (self.current_u - self.target_u) * torch.exp(-self.washout_k * t)
        else:
            u = self.current_u
            
        if z.dim() == 3:
            u = u.unsqueeze(0).expand(z.size(0), -1, -1)
            
        # Scale U for higher sensitivity relative to Z
        u_scaled = u * 10.0
        x = torch.cat([z, u_scaled], dim=-1)
        return self.net(x)

def train_ode(data_path, model_path, epochs=100, lr=1e-3, batch_size=4096, solver='rk4'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hyper ODE Training: {device} | Batch: {batch_size} | Solver: {solver}")
    
    data = torch.load(data_path, map_location=device, weights_only=True)
    z0 = data['z_0'].to(device)
    zT = data['z_T'].to(device)
    v0 = data['v_0'].to(device)
    u0 = data['u_0'].to(device)
    dt = data['dt'][0].item()
    
    t_span = torch.tensor([0.0, dt], device=device)
    dataset_size = z0.size(0)
    control_dim = u0.size(1)
    
    func = ODEFunc(control_dim=control_dim).to(device)
    optimizer = optim.Adam(func.parameters(), lr=lr, weight_decay=1e-5)
    
    # Scheduler to help hit that 10^-2 target
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    lambda_reg = 0.5 
    
    print(f"Aggressive convergence strategy for {dataset_size} pairs...")
    
    try:
        for epoch in range(epochs):
            perm = torch.randperm(dataset_size)
            epoch_loss = 0
            epoch_mse = 0
            
            for i in range(0, dataset_size, batch_size):
                optimizer.zero_grad()
                indices = perm[i:i+batch_size]
                
                b_z0 = z0[indices]
                b_zT = zT[indices]
                b_v0 = v0[indices]
                b_u0 = u0[indices]
                
                func.current_u = b_u0
                
                # Predict
                b_z_pred = odeint(func, b_z0, t_span, method=solver)
                z_pred_final = b_z_pred[1]
                mse_loss = criterion(z_pred_final, b_zT)
                
                # Regularize
                v_pred_start = func(0, b_z0)
                reg_loss = criterion(v_pred_start, b_v0)
                
                loss = mse_loss + lambda_reg * reg_loss
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * (len(indices) / dataset_size)
                epoch_mse += mse_loss.item() * (len(indices) / dataset_size)
            
            scheduler.step(epoch_mse)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Obj: {epoch_loss:.6f} | MSE: {epoch_mse:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    except KeyboardInterrupt:
        print("Training interrupted. Saving current state...")
    finally:
        torch.save(func.state_dict(), model_path)
        print(f"Optimized ODE saved to {model_path}")

if __name__ == "__main__":
    data_path = str(MODELS_DIR / 'ode_training_pairs.pth')
    model_path = str(MODELS_DIR / 'neural_ode_model.pth')
    train_ode(data_path, model_path, epochs=100)
