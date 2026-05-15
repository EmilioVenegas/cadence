import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from _paths import MODELS_DIR
from train_ode import ODEFunc

def evaluate_ode(data_path, model_path, solver='dopri5'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Neural ODE Performance Evaluation ---")
    print(f"Device: {device} | Solver: {solver}")
    
    # Load Data
    data = torch.load(data_path, map_location=device, weights_only=True)
    z0 = data['z_0'].to(device)
    zT = data['z_T'].to(device)
    v0 = data['v_0'].to(device)
    u0 = data['u_0'].to(device)
    dt = data['dt'][0].item()
    
    t_span = torch.tensor([0.0, dt], device=device)
    control_dim = u0.size(1)
    
    # Load Model
    func = ODEFunc(control_dim=control_dim).to(device)
    func.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    func.eval()
    
    print(f"Evaluating on {z0.size(0)} temporal pairs...")
    
    with torch.no_grad():
        # 1. Prediction (Terminal State)
        func.current_u = u0
        # Use relaxed tolerances consistent with refined training
        z_pred_traj = odeint(func, z0, t_span, method=solver, atol=1e-3, rtol=1e-3)
        z_pred_final = z_pred_traj[1]
        
        # 2. Velocity Prediction (Baseline Correspondence)
        v_pred_start = func(0, z0)
        
    # Calculate Metrics
    zT_np = zT.cpu().numpy()
    z_pred_np = z_pred_final.cpu().numpy()
    v0_np = v0.cpu().numpy()
    v_pred_np = v_pred_start.cpu().numpy()
    
    # MSE
    mse_z = np.mean((zT_np - z_pred_np)**2)
    mse_v = np.mean((v0_np - v_pred_np)**2)
    
    # R2 for each latent dimension
    r2_dims = []
    for i in range(zT_np.shape[1]):
        r2_dims.append(r2_score(zT_np[:, i], z_pred_np[:, i]))
    
    mean_r2 = np.mean(r2_dims)
    
    # Magnitude Comparison (Clinical Consistency)
    v_mag_true = np.sqrt(np.sum(v0_np**2, axis=1))
    v_mag_pred = np.sqrt(np.sum(v_pred_np**2, axis=1))
    mse_mag = np.mean((v_mag_true - v_mag_pred)**2)
    
    print("\n[METRICS SUMMARY]")
    print(f"  Terminal Latent State MSE: {mse_z:.6f}")
    print(f"  Initial Velocity MSE:     {mse_v:.6f}")
    print(f"  Initial Velocity Mag MSE: {mse_mag:.6f}")
    print(f"  Mean Explained Variance (R2): {mean_r2:.4f} ({mean_r2*100:.1f}%)")
    
    print("\nPer-Dimension R2 Scores:")
    for i, r2 in enumerate(r2_dims):
        print(f"  z_mean_{i:<2}: {r2:.4f}")
        
    v_mag_corr = np.corrcoef(v_mag_true, v_mag_pred)[0, 1]
    
    # Calculate Velocity Component Correlations
    print("\nPer-Component Velocity Correlations ($v_k$):")
    v_corrs = []
    for i in range(v0_np.shape[1]):
        corr = np.corrcoef(v0_np[:, i], v_pred_np[:, i])[0, 1]
        v_corrs.append(corr)
        print(f"  v_{i:<2}: {corr:.4f}")
    
    v_mag_corr = np.corrcoef(v_mag_true, v_mag_pred)[0, 1]
    
    print(f"\nOverall Velocity Magnitude Correlation: {v_mag_corr:.4f}")
    
    # Calculate Velocity Directional Alignment (Cosine Similarity)
    cos_sim = nn.functional.cosine_similarity(
        torch.tensor(v0_np), 
        torch.tensor(v_pred_np), 
        dim=1
    )
    mean_cos_sim = cos_sim.mean().item()
    
    print(f"Overall Velocity Directional Alignment (Cosine Sim): {mean_cos_sim:.4f}")
    
    # Scaling Sanity Check
    print(f"\nVelocity Scale Check:")
    print(f"  True v_mag Mean:  {np.mean(v_mag_true):.4f} | Std: {np.std(v_mag_true):.4f}")
    print(f"  Pred v_mag Mean:  {np.mean(v_mag_pred):.4f} | Std: {np.std(v_mag_pred):.4f}")

    return {
        'mse_z': mse_z,
        'mse_v': mse_v,
        'mse_mag': mse_mag,
        'mean_r2': mean_r2,
        'r2_dims': r2_dims,
        'v_corr_mag': v_mag_corr,
        'v_corrs': v_corrs,
        'mean_cos_sim': mean_cos_sim
    }

if __name__ == "__main__":
    data_path = str(MODELS_DIR / 'ode_training_pairs_128.pth')
    model_path = str(MODELS_DIR / 'neural_ode_high_momentum_128.pth')
    evaluate_ode(data_path, model_path)
