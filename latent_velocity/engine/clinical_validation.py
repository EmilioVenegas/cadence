import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings
from _paths import DATA_DIR, MODELS_DIR, PLOTS_DIR

def calculate_velocity_magnitude(traj_path=None):
    if traj_path is None:
        traj_path = str(MODELS_DIR / 'latent_velocity_trajectory.csv')
    print(f"Loading high-resolution Latent Velocity dataset from {traj_path}...")
        
    df_traj = pd.read_csv(traj_path)
    velocity_cols = [col for col in df_traj.columns if col.startswith('v_')]
    df_traj['v_mag'] = np.sqrt((df_traj[velocity_cols] ** 2).sum(axis=1))
    return df_traj, velocity_cols

def execute_lmm_clinical_validation(df_traj, empirical_fi_path=None):
    print("\n--- Phase 5.2: Ground-Truth Clinical Validation (LMM) ---")
    if empirical_fi_path is None:
        empirical_fi_path = str(DATA_DIR / 'frailty_index_data.csv')
    df_fi = pd.read_csv(empirical_fi_path)
    df_fi = df_fi.sort_values(by=['cunicah', 'np', 'a_o_ent'])
    
    df_fi['t'] = df_fi['a_o_ent'] - 2001
    df_fi['next_t'] = df_fi.groupby(['cunicah', 'np'])['t'].shift(-1)
    df_fi['next_FI'] = df_fi.groupby(['cunicah', 'np'])['FI'].shift(-1)
    
    df_fi_valid = df_fi.dropna(subset=['next_t', 'next_FI']).copy()
    df_fi_valid['empirical_velocity'] = (df_fi_valid['next_FI'] - df_fi_valid['FI']) / (df_fi_valid['next_t'] - df_fi_valid['t'])
    
    df_traj['t_round'] = df_traj['t'].round(2)
    df_fi_valid['t_round'] = df_fi_valid['t'].round(2)
    
    merged_data = pd.merge(df_fi_valid, df_traj[['cunicah', 'np', 't_round', 'v_mag']], 
                           on=['cunicah', 'np', 't_round'], how='inner')
    
    if len(merged_data) == 0:
        print("CRITICAL MERGE ERROR: Time matching failed.")
        return
        
    merged_data['edad_z'] = (merged_data['edad'] - merged_data['edad'].mean()) / merged_data['edad'].std()
    merged_data['educ_z'] = (merged_data['educacion'] - merged_data['educacion'].mean()) / merged_data['educacion'].std()
    merged_data = merged_data.dropna(subset=['empirical_velocity', 'v_mag', 'edad_z', 'sexo', 'educ_z', 'cunicah']).reset_index(drop=True)
    
    print("Fitting Linear Mixed-Effects Model (LMM)...")
    model = smf.mixedlm("empirical_velocity ~ v_mag + edad_z + sexo + educ_z", 
                        merged_data, groups=merged_data["cunicah"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = model.fit(method='lbfgs', maxiter=1000)
        except Exception as e:
            print(f"LBFGS failed: {e}. Falling back to default CG...")
            result = model.fit()
    
    print("\n[LMM SUMMARY]")
    print(result.summary().tables[1])

def generate_visualizations(df_traj_full, surv_grouped, cph, hr, velocity_cols):
    print("\n--- Phase 6: Generating Final Visualizations ---")
    
    # 1. Kaplan-Meier Survival Curves
    print("Generating Kaplan-Meier plots...")
    plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)
    
    fast_data = surv_grouped[surv_grouped['Phenotype'] == 'Fast_Ager']
    slow_data = surv_grouped[surv_grouped['Phenotype'] == 'Slow_Ager']
    
    kmf_fast = KaplanMeierFitter()
    kmf_slow = KaplanMeierFitter()
    
    kmf_fast.fit(fast_data['time_to_event'], event_observed=fast_data['fallecido'], label='Fast Agers (Q4)')
    kmf_slow.fit(slow_data['time_to_event'], event_observed=slow_data['fallecido'], label='Slow Agers (Q1)')
    
    kmf_fast.plot_survival_function(ax=ax, linewidth=2.5, color='#d62728') # Red
    kmf_slow.plot_survival_function(ax=ax, linewidth=2.5, color='#1f77b4') # Blue
    
    plt.title("Kaplan-Meier Survival Curves:\nLatent Velocity Phenotypes", fontsize=14, fontweight='bold')
    plt.xlabel("Years of Follow-Up", fontsize=12)
    plt.ylabel("Survival Probability", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    add_at_risk_counts(kmf_fast, kmf_slow, ax=ax, xticks=[0, 5, 10, 15, 20])
    
    results = logrank_test(fast_data['time_to_event'], slow_data['time_to_event'], 
                           fast_data['fallecido'], slow_data['fallecido'])
    
    textstr = f"Cox Hazard Ratio: {hr:.2f}\nLog-Rank p-value: {results.p_value:.1e}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.05, 0.15, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props)
            
    plt.tight_layout()
    output_path = str(PLOTS_DIR / 'km_survival_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved {output_path}")
    
    # 2. Archetypal Latent Trajectories
    print("Generating Archetypal Trajectories...")
    df_early = df_traj_full.groupby(['cunicah', 'np']).head(10)
    patient_vmag = df_early.groupby(['cunicah', 'np'])['v_mag'].mean().reset_index()
    patient_vmag = patient_vmag.sort_values(by='v_mag')
    
    slow_p = patient_vmag.iloc[0]
    fast_p = patient_vmag.iloc[-1]
    
    from extract_velocity import extract_latent_vectors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    m_path = str(MODELS_DIR / 'beta_vae_model.pth')
    d_path = str(DATA_DIR / 'frailty_index_data.csv')
    
    df_obs, _ = extract_latent_vectors(m_path, d_path, device)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(r"Archetypal Latent Trajectories ($\bar{z}_k(t)$): Fast vs Slow Ager", fontsize=18, fontweight='bold')
    axes = axes.flatten()
    
    for k in range(8):
        ax = axes[k]
        for i, p in enumerate([slow_p, fast_p]):
            label = "Slow Ager" if i == 0 else "Fast Ager"
            color = "#1f77b4" if i == 0 else "#d62728"
            
            p_dense = df_traj_full[(df_traj_full['cunicah'] == p['cunicah']) & (df_traj_full['np'] == p['np'])]
            p_obs = df_obs[(df_obs['cunicah'] == p['cunicah']) & (df_obs['np'] == p['np'])]
            
            ax.plot(p_dense['t'], p_dense[f'z_mean_{k}'], color=color, linewidth=3, label=f'{label} (GP)')
            ax.scatter(p_obs['t'], p_obs[f'z_{k}'], color=color, marker='o', s=80, edgecolor='white', zorder=5, label=f'{label} (Obs)')
            
        ax.set_title(f"Latent Dimension $z_{k}$", fontsize=14)
        ax.set_xlabel("Years since baseline (t)", fontsize=11)
        ax.grid(True, alpha=0.3)
        if k == 0: ax.legend(fontsize=10)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = str(PLOTS_DIR / 'archetypal_trajectories.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Saved {output_path}")
    
    # 3. Velocity Heatmaps
    print("Generating Velocity-Domain Heatmaps...")
    fi_path = str(DATA_DIR / 'frailty_index_data.csv')
    df_fi = pd.read_csv(fi_path)
    df_fi = df_fi.sort_values(by=['cunicah', 'np', 'a_o_ent'])
    
    cog_cols = ['recuerdo1', 'recuerdo2', 'copiafiguras1', 'copiafiguras2', 'orientacion', 'serial7', 'visualscan', 'memoria']
    phys_cols = ['n_abvd', 'n_aivd', 'n_mov', 'n_img', 'motoras_gruesas', 'motoras_finas', 
                 'hipertension', 'diabetes', 'enf_pulm', 'artritis', 'infarto', 'embolia', 'cancer', 'salud_glob']
    ment_cols = ['deprimido', 'esfuerzo', 'intranquilo', 'triste', 'cansado', 'solo', 'feliz', 'disf_vida', 'energia']
    
    df_fi['Cog_FI'] = df_fi[cog_cols].mean(axis=1)
    df_fi['Phys_FI'] = df_fi[phys_cols].mean(axis=1)
    df_fi['Ment_FI'] = df_fi[ment_cols].mean(axis=1)
    df_fi['t'] = df_fi['a_o_ent'] - 2001
    
    for col in ['t', 'Cog_FI', 'Phys_FI', 'Ment_FI']:
        df_fi[f'next_{col}'] = df_fi.groupby(['cunicah', 'np'])[col].shift(-1)
        
    df_fi_valid = df_fi.dropna(subset=['next_t']).copy()
    for domain in ['Cog_FI', 'Phys_FI', 'Ment_FI']:
        df_fi_valid[f'delta_{domain}'] = (df_fi_valid[f'next_{domain}'] - df_fi_valid[domain]) / (df_fi_valid['next_t'] - df_fi_valid['t'])
        
    df_traj_full['t_round'] = df_traj_full['t'].round(2)
    df_fi_valid['t_round'] = df_fi_valid['t'].round(2)
    
    merged = pd.merge(df_fi_valid, df_traj_full, on=['cunicah', 'np', 't_round'], how='inner')
    
    delta_cols = ['delta_Cog_FI', 'delta_Phys_FI', 'delta_Ment_FI']
    corr_matrix = merged[velocity_cols + delta_cols].corr().loc[velocity_cols, delta_cols]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                cbar_kws={'label': 'Pearson Correlation', 'pad': 0.02}, 
                annot_kws={"size": 12},
                linewidths=1, linecolor='white')
                
    plt.title(r'$\beta$-VAE Disentanglement:' + '\nLatent Velocity vs Clinical Domain Velocity', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel(r"Latent Velocity Component ($v_k$)", fontsize=14)
    plt.xlabel("Empirical Domain Degradation Rate", fontsize=14)
    plt.xticks([0.5, 1.5, 2.5], ['Cognitive Decline', 'Physical Decline', 'Mental Health Decline'], fontsize=12)
    
    plt.tight_layout()
    output_path = str(PLOTS_DIR / 'velocity_domain_heatmap.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Saved {output_path}")

def execute_survival_analysis(df_traj_full, velocity_cols=None):
    print("\n--- Phase 5.3: Phenotyping & Survival Analysis ---")
    
    df_early = df_traj_full.groupby(['cunicah', 'np']).head(10)
    patient_vmag = df_early.groupby(['cunicah', 'np'])['v_mag'].mean().reset_index()
    
    q1 = patient_vmag['v_mag'].quantile(0.25)
    q3 = patient_vmag['v_mag'].quantile(0.75)
    
    conditions = [
        (patient_vmag['v_mag'] <= q1),
        (patient_vmag['v_mag'] >= q3)
    ]
    choice_list = ['Slow_Ager', 'Fast_Ager']
    patient_vmag['Phenotype'] = np.select(conditions, choice_list, default='Normal')
    patient_vmag = patient_vmag[patient_vmag['Phenotype'] != 'Normal']
    
    import pyreadstat
    master_data_path = str(DATA_DIR / 'simpleMHAS.sav')
    df_true_outcomes, _ = pyreadstat.read_sav(master_data_path, usecols=['cunicah', 'np', 'fallecido', 'a_o_ent', 'edad'])
    surv_data = pd.merge(patient_vmag, df_true_outcomes, on=['cunicah', 'np'], how='left')
    
    surv_grouped = surv_data.groupby(['cunicah', 'np']).agg(
        Phenotype=('Phenotype', 'first'),
        fallecido=('fallecido', 'max'),
        start_year=('a_o_ent', 'min'),
        end_year=('a_o_ent', 'max'),
        baseline_age=('edad', 'min')
    ).reset_index()
    
    surv_grouped['time_to_event'] = surv_grouped['end_year'] - surv_grouped['start_year'] + 1
    surv_grouped = surv_grouped.dropna(subset=['time_to_event', 'fallecido'])
    
    surv_grouped['Fast_Ager_Flag'] = (surv_grouped['Phenotype'] == 'Fast_Ager').astype(float)
    cph_data = surv_grouped[['time_to_event', 'fallecido', 'Fast_Ager_Flag', 'baseline_age']].copy()
    
    cph = CoxPHFitter()
    cph.fit(cph_data, duration_col='time_to_event', event_col='fallecido')
    hr = np.exp(cph.params_['Fast_Ager_Flag'])
    
    generate_visualizations(df_traj_full, surv_grouped, cph, hr, velocity_cols)

if __name__ == "__main__":
    df_magnitude, velocity_cols = calculate_velocity_magnitude()
    execute_lmm_clinical_validation(df_magnitude)
    execute_survival_analysis(df_magnitude, velocity_cols=velocity_cols)
