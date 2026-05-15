import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import os
from _paths import MODELS_DIR
from train_ode import ODEFunc

def train_ode_high_momentum(data_path, model_path, epochs=100, lr=1e-3, batch_size=4096, solver='dopri5'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- High-Momentum ODE Training ---")
    print(f"Device: {device} | Batch: {batch_size} | Solver: {solver}")
    
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    # Aggressive Regularization
    lambda_reg = 15.0 
    lambda_dir = 5.0
    lambda_mag = 10.0
    
    print(f"Priority: Velocity Correlation (lambda_reg={lambda_reg}, lambda_dir={lambda_dir}, lambda_mag={lambda_mag})")
    print(f"Aggressive convergence strategy for {dataset_size} pairs...")
    
    try:
        for epoch in range(epochs):
            # Curriculum Learning: Euler for speed in early epochs, adaptive for refinement
            if epoch < 20: 
                current_solver = 'euler'
                solver_opts = {}
            else:
                current_solver = solver # Default dopri5
                solver_opts = {'atol': 1e-3, 'rtol': 1e-3}
            
            func.nfe = 0 # Reset function evaluation counter
            perm = torch.randperm(dataset_size)
            epoch_loss = 0
            epoch_mse = 0
            epoch_reg = 0
            epoch_dir = 0
            epoch_mag = 0
            
            for i in range(0, dataset_size, batch_size):
                optimizer.zero_grad()
                indices = perm[i:i+batch_size]
                
                b_z0 = z0[indices]
                b_zT = zT[indices]
                b_v0 = v0[indices]
                b_u0 = u0[indices]
                
                func.current_u = b_u0
                
                # 1. Prediction (Final State)
                b_z_pred = odeint(func, b_z0, t_span, method=current_solver, **solver_opts)
                z_pred_final = b_z_pred[1]
                mse_loss = criterion(z_pred_final, b_zT)
                
                # 2. Velocity Prediction (Initial State)
                v_pred_start = func(0, b_z0)
                reg_loss = criterion(v_pred_start, b_v0)
                
                # 3. Explicit Directional Loss (Cosine Similarity)
                # Penalize bad angles explicitly: 1 - cosine_similarity
                dir_loss = 1 - torch.nn.functional.cosine_similarity(v_pred_start, b_v0).mean()
                
                # 4. Explicit Magnitude Loss
                # Penalize the difference in vector norms
                v_pred_norm = torch.norm(v_pred_start, dim=1)
                v_true_norm = torch.norm(b_v0, dim=1)
                mag_loss = criterion(v_pred_norm, v_true_norm)
                
                # Balanced Loss: State MSE + Velocity MSE + Directional + Magnitude
                loss = mse_loss + lambda_reg * reg_loss + lambda_dir * dir_loss + lambda_mag * mag_loss
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * (len(indices) / dataset_size)
                epoch_mse += mse_loss.item() * (len(indices) / dataset_size)
                epoch_reg += reg_loss.item() * (len(indices) / dataset_size)
                epoch_dir += dir_loss.item() * (len(indices) / dataset_size)
                epoch_mag += mag_loss.item() * (len(indices) / dataset_size)
            
            scheduler.step(epoch_mse)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Obj: {epoch_loss:.6f} | MSE: {epoch_mse:.6f} | V-MSE: {epoch_reg:.6f} | Dir: {epoch_dir:.4f} | Mag: {epoch_mag:.6f} | NFE: {func.nfe} | LR: {optimizer.param_groups[0]['lr']:.2e}")
                
    except KeyboardInterrupt:
        print("Training interrupted. Saving current state...")
    finally:
        torch.save(func.state_dict(), model_path)
        print(f"High-Momentum ODE saved to {model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    data_path = str(MODELS_DIR / 'ode_training_pairs_128.pth')
    model_path = str(MODELS_DIR / 'neural_ode_high_momentum_128.pth')
    train_ode_high_momentum(data_path, model_path, epochs=args.epochs)
