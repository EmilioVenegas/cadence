import torch
import numpy as np
import pandas as pd
from _paths import MODELS_DIR
from evaluate_ode import evaluate_ode

def run_comparison():
    data_path = str(MODELS_DIR / 'ode_training_pairs.pth')
    baseline_path = str(MODELS_DIR / 'neural_ode_model.pth')
    high_mom_path = str(MODELS_DIR / 'neural_ode_high_momentum.pth')
    
    results = {}
    
    print("\n" + "="*50)
    print(" BENCKMARKING: BASELINE vs HIGH-MOMENTUM")
    print("="*50)
    
    print("\n--- Model A: Baseline ---")
    results['Baseline'] = evaluate_ode(data_path, baseline_path)
    
    print("\n--- Model B: High-Momentum ---")
    if os.path.exists(high_mom_path):
        results['High-Momentum'] = evaluate_ode(data_path, high_mom_path)
    else:
        print(f"High-Momentum model at {high_mom_path} not found yet. Run training first.")
        return

    # Comparative Summary
    print("\n" + "="*50)
    print(" FINAL COMPARISON")
    print("="*50)
    print(f"{'Metric':<30} {'Baseline':>12} {'High-Mom':>12}")
    print(f"{'-'*30} {'-'*12} {'-'*12}")
    
    metrics = [
        ('Terminal State MSE (Z)', 'mse_z', '{:.6f}'),
        ('Velocity MSE (V)', 'mse_v', '{:.6f}'),
        ('Mean R2 (Latent State)', 'mean_r2', '{:.4f}'),
        ('Velocity Mag Correlation', 'v_corr_mag', '{:.4f}'),
        ('Velocity Directional Cosine', 'mean_cos_sim', '{:.4f}') # New Metric
    ]
    
    for label, key, fmt in metrics:
        b_val = results['Baseline'][key]
        h_val = results['High-Momentum'][key]
        print(f"{label:<30} {fmt.format(b_val)} {fmt.format(h_val)}")
    
    print("="*50)

if __name__ == "__main__":
    import os
    run_comparison()
