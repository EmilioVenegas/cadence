import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "engine"))
sys.path.insert(0, str(_ROOT / "ode-digitaltwin"))

from _paths import DATA_DIR, PLOTS_DIR, TWIN_DIR
from digital_twin import load_models, _extract_patient_u, U_COLS, TARGET_MAP, LABEL_MAP
from train_vae import FrailtyDataset
from torchdiffeq import odeint

def simulate_trajectories(cunicah, np_val, target_intervention='ejer_3_por_sem', years=5.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, ode_func = load_models(device)
    
    # Load patient data
    dataset = FrailtyDataset(DATA_DIR / 'frailty_index_data.csv', device=device)
    df_raw = dataset.data
    p_data = df_raw[(df_raw['cunicah'] == cunicah) & (df_raw['np'] == np_val)]
    
    if p_data.empty:
        print(f"Patient {cunicah}/{np_val} not found.")
        return
        
    latest_visit = p_data.sort_values(by='a_o_ent').iloc[-1]
    
    # Extract Initial Latent State (z0)
    x_def = torch.tensor(latest_visit[dataset.deficit_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
    x_sta = torch.tensor(latest_visit[dataset.static_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        z0, _ = vae.encode(x_def, x_sta)
        
    # Extract Baseline Control Vector (u)
    u_dict = _extract_patient_u(latest_visit)
    u_baseline = torch.tensor([[u_dict[c] for c in U_COLS]], dtype=torch.float32).to(device)
    
    # Create Intervention Twin Vector
    u_twin = u_baseline.clone()
    idx = U_COLS.index(target_intervention)
    u_twin[0, idx] = TARGET_MAP[target_intervention]
    
    # Time span (5 years, 100 steps for smooth plotting)
    t_span = torch.linspace(0, years, 100).to(device)
    
    with torch.no_grad():
        # --- 1. BASELINE SIMULATION ---
        ode_func.current_u = u_baseline
        ode_func.target_u = None
        ode_func.washout_k = 0.0
        z_baseline = odeint(ode_func, z0, t_span, method='rk4').squeeze().cpu().numpy()
        
        # --- 2. TWIN SIMULATION (With Washout) ---
        ode_func.current_u = u_baseline
        ode_func.target_u = u_twin
        ode_func.washout_k = 2.0  # Biological momentum
        z_twin = odeint(ode_func, z0, t_span, method='rk4').squeeze().cpu().numpy()
        
    t_np = t_span.cpu().numpy()
    
    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.suptitle(f"Digital Twin Counterfactual: Patient {int(cunicah)}\nIntervention: {LABEL_MAP[target_intervention]}", 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: The Master Frailty Axis over Time
    axes[0].plot(t_np, z_baseline[:, 3], color='red', linewidth=3, label='Baseline (Do Nothing)')
    axes[0].plot(t_np, z_twin[:, 3], color='green', linewidth=3, linestyle='--', label=f'Twin ({LABEL_MAP[target_intervention]})')
    axes[0].set_title("Master Frailty Trajectory (z3) over 5 Years")
    axes[0].set_xlabel("Years from Baseline")
    axes[0].set_ylabel("Systemic Frailty Severity (z_mean_3)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: 2D Phase Plane (Cardiometabolic vs Physical Frailty)
    axes[1].plot(z_baseline[:, 0], z_baseline[:, 3], color='red', linewidth=2.5, label='Baseline Path')
    axes[1].plot(z_twin[:, 0], z_twin[:, 3], color='green', linewidth=2.5, linestyle='--', label='Twin Path')
    
    # Mark the start point
    axes[1].scatter(z0[0, 0].item(), z0[0, 3].item(), color='black', s=100, zorder=5, label='Current State (t=0)')
    
    axes[1].set_title("Biological Phase Shift (z0 vs z3)")
    axes[1].set_xlabel("Cardiometabolic Axis (z_mean_0)")
    axes[1].set_ylabel("Systemic Frailty Axis (z_mean_3)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    TWIN_DIR.mkdir(exist_ok=True, parents=True)
    out_path = TWIN_DIR / f'counterfactual_{int(cunicah)}.png'
    plt.savefig(str(out_path), dpi=200)
    print(f"Saved Digital Twin Counterfactual to {out_path}")

if __name__ == "__main__":
    # Feel free to change this cunicah ID to any patient ID from your dataset!
    simulate_trajectories(cunicah=7226.0, np_val=10.0, target_intervention='ejer_3_por_sem')
