import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "ode-digitaltwin"))
sys.path.insert(0, str(_ROOT / "engine"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _paths import PLOTS_DIR, MODELS_DIR, RANKING_DIR
from digital_twin import rank_interventions


def plot_intervention_ranking(cunicah, np_val):
    """Generate the Automated Intervention Ranking plot and console report."""
    ranking = rank_interventions(cunicah, np_val)
    
    if not ranking:
        return
    
    t = ranking['t']
    v_baseline = ranking['v_mag_baseline']
    interventions = ranking['ranked_interventions']
    patient_id = ranking['patient_id']
    
    # Phenotype thresholds
    df_traj = pd.read_csv(MODELS_DIR / 'latent_velocity_trajectory_128.csv',
                          usecols=['v_0','v_1','v_2','v_3','v_4','v_5','v_6','v_7'])
    v_cols = [c for c in df_traj.columns if c.startswith('v_')]
    v_mags = np.sqrt((df_traj[v_cols]**2).sum(axis=1))
    q1 = v_mags.quantile(0.25)
    q3 = v_mags.quantile(0.75)
    
    # Color palette
    palette = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf',
               '#e377c2', '#bcbd22', '#d62728', '#8c564b', '#7f7f7f',
               '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173']
    
    fig, ax = plt.subplots(figsize=(13, 7))
    
    # High-risk zone
    y_max = max(v_baseline.max(), max(r['v_mag'].max() for r in interventions)) * 1.1
    ax.fill_between(t, q3, y_max, color='#d62728', alpha=0.05)
    ax.text(0.1, q3 * 1.03, "HIGH RISK ZONE (Fast Aging)", color='#d62728',
            fontweight='bold', alpha=0.7, fontsize=9)
    
    # Baseline
    ax.plot(t, v_baseline, color='#d62728', linewidth=3.5,
            label=f'Baseline (AUC: {ranking["auc_baseline"]:.2f})', zorder=10)
    
    for i, r in enumerate(interventions):
        color = palette[i % len(palette)]
        is_high = r['confidence'] == "High"
        conf_marker = "" if is_high else " ⚠"
        linestyle = '-' if is_high else '--'
        alpha = 0.85 if is_high else 0.3
        
        ax.plot(t, r['v_mag'], color=color, linewidth=2, alpha=alpha,
                linestyle=linestyle,
                label=f'{r["label"]} ({r["auc_reduction_pct"]:+.1f}%){conf_marker}',
                zorder=9 - i)
    
    # Thresholds
    ax.axhline(y=q3, color='gray', linestyle='--', alpha=0.4, label='Fast Ager (Q3)')
    ax.axhline(y=q1, color='gray', linestyle=':', alpha=0.4, label='Slow Ager (Q1)')
    
    ax.set_title(f"LAVA Digital Twin: Automated Intervention Ranking\nPatient {patient_id}",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Forecast Horizon (Years)", fontsize=12)
    ax.set_ylabel("Systemic Biological Aging Velocity ($||v||$)", fontsize=12)
    
    # Legend — shrink font if many entries
    n_entries = len(interventions) + 3  # baseline + 2 thresholds
    legend_size = max(7, 10 - n_entries * 0.2)
    ax.legend(loc='upper right', fontsize=legend_size, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = str(RANKING_DIR / 'intervention_ranking.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n -> Saved {output_path}")
    
    # Console report
    print(f"\n{'='*85}")
    print(f"  INTERVENTION RANKING — Patient {patient_id}")
    print(f"  Baseline 5-Year Velocity AUC: {ranking['auc_baseline']:.3f}")
    print(f"{'='*85}")
    print(f"  {'#':<4} {'Intervention':<35} {'AUC':>7} {'Δ%':>8}  {'Conf':>12}  {'Maha':>5}  {'n':>5}")
    print(f"  {'-'*4} {'-'*35} {'-'*7} {'-'*8}  {'-'*12}  {'-'*5}  {'-'*5}")
    for i, r in enumerate(interventions, 1):
        print(f"  {i:<4} {r['label']:<35} {r['auc']:>7.3f} {r['auc_reduction_pct']:>+7.1f}%"
              f"  {r['confidence']:>12s}  {r['mahalanobis']:>5.1f}  {r['n_cohort_match']:>5d}")
    print(f"{'='*85}")
    
    best = interventions[0]
    print(f"\n  >>> PRIMARY CLINICAL TARGET: {best['label']}")
    print(f"      Velocity Reduction: {best['auc_reduction_pct']:+.1f}% | "
          f"Confidence: {best['confidence']} (Mahalanobis: {best['mahalanobis']:.2f})")
    
    return ranking


if __name__ == "__main__":
    plot_intervention_ranking(cunicah=7226.0, np_val=10.0)

