import sys
from pathlib import Path

# Add ode/ and engine/ to path for imports
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "ode-digitaltwin"))
sys.path.insert(0, str(_ROOT / "engine"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _paths import PLOTS_DIR, MODELS_DIR, TWIN_DIR
from digital_twin import run_digital_twin_intervention

def plot_digital_twin_comparison(cunicah, np_val):
    print(f"Running Digital Twin Simulation for Patient {cunicah}/{np_val}...")
    results = run_digital_twin_intervention(cunicah, np_val)
    
    if not results:
        return
        
    t = results['t']
    v_baseline = results['v_mag_baseline']
    v_twin = results['v_mag_twin']
    
    # Calculate Phenotype Thresholds (for clinical context)
    # We load a sample of the trajectory to get the distribution of velocities
    df_traj = pd.read_csv(MODELS_DIR / 'latent_velocity_trajectory.csv', usecols=['v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7'])
    # Sample velocity magnitude for threshold calc
    v_cols = [c for c in df_traj.columns if c.startswith('v_')]
    v_mags = np.sqrt((df_traj[v_cols]**2).sum(axis=1))
    q1_threshold = v_mags.quantile(0.25)
    q3_threshold = v_mags.quantile(0.75)
    
    plt.figure(figsize=(10, 6))
    
    # Plot curves
    plt.plot(t, v_baseline, color='#d62728', linewidth=3, label='Baseline (Current Trajectory)')
    plt.plot(t, v_twin, color='#1f77b4', linewidth=3, label='Digital Twin (Intervention: Cured Smoking/BMI)')
    
    # Thresholds
    plt.axhline(y=q3_threshold, color='gray', linestyle='--', alpha=0.5, label='Fast Ager Threshold (Q3)')
    plt.axhline(y=q1_threshold, color='gray', linestyle=':', alpha=0.5, label='Slow Ager Threshold (Q1)')
    
    # Labeling
    plt.fill_between(t, q3_threshold, max(v_baseline.max(), v_twin.max())*1.1, color='#d62728', alpha=0.05)
    plt.text(0.1, q3_threshold * 1.05, "HIGH RISK ZONE (Fast Aging)", color='#d62728', fontweight='bold', alpha=0.7)
    
    plt.title(f"LAVA Digital Twin: Longitudinal Risk Mitigation\nPatient {int(cunicah)} Wave Comparison", fontsize=15, fontweight='bold')
    plt.xlabel("Forecast Horizon (Years)", fontsize=12)
    plt.ylabel("Systemic Biological Aging Velocity ($||v||$)", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = str(TWIN_DIR / 'digital_twin_intervention.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Saved {output_path}")

if __name__ == "__main__":
    # Test for Patient X (cunicah=10, np=10)
    plot_digital_twin_comparison(cunicah=7088.0, np_val=24.0)
