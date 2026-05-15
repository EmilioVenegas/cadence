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
import pyreadstat # Moved to top

from _paths import DATA_DIR, MODELS_DIR, PLOTS_DIR
from extract_velocity import extract_latent_vectors # Moved to top

# Global configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FI_PATH = str(DATA_DIR / 'frailty_index_data.csv')

def calculate_velocity_magnitude(traj_path=None):
    if traj_path is None:
        traj_path = str(MODELS_DIR / 'latent_velocity_trajectory_128.csv')
    print(f"Loading high-resolution Latent Velocity dataset from {traj_path}...")
        
    df_traj = pd.read_csv(traj_path)
    velocity_cols = [col for col in df_traj.columns if col.startswith('v_')]
    df_traj['v_mag'] = np.sqrt((df_traj[velocity_cols] ** 2).sum(axis=1))
    return df_traj, velocity_cols

def execute_lmm_clinical_validation(df_traj, df_fi):
    print("\n--- Phase 5.2: Ground-Truth Clinical Validation (LMM) ---")
    
    df_fi = df_fi.sort_values(by=['cunicah', 'np', 'a_o_ent']).copy()
    
    df_fi['t'] = df_fi['a_o_ent'] - 2001
    df_fi['next_t'] = df_fi.groupby(['cunicah', 'np'])['t'].shift(-1)
    df_fi['next_FI'] = df_fi.groupby(['cunicah', 'np'])['FI'].shift(-1)
    
    df_fi_valid = df_fi.dropna(subset=['next_t', 'next_FI']).copy()
    df_fi_valid['empirical_velocity'] = (df_fi_valid['next_FI'] - df_fi_valid['FI']) / (df_fi_valid['next_t'] - df_fi_valid['t'])
    
    df_traj['a_o_ent'] = (df_traj['t'] + 2001).round(0).astype(int)
    
    merged_data = pd.merge(df_fi_valid, df_traj[['cunicah', 'np', 'a_o_ent', 'v_mag']], 
                           on=['cunicah', 'np', 'a_o_ent'], how='inner')
    
    if len(merged_data) == 0:
        raise ValueError("CRITICAL MERGE ERROR: Time matching failed. Check exact integer alignments.")
        
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

# --- Refactored Plotting Functions ---

def plot_survival_curves(surv_grouped, hr):
    print("Generating Kaplan-Meier plots...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    fast_data = surv_grouped[surv_grouped['Phenotype'] == 'Fast_Ager']
    slow_data = surv_grouped[surv_grouped['Phenotype'] == 'Slow_Ager']
    
    kmf_fast, kmf_slow = KaplanMeierFitter(), KaplanMeierFitter()
    
    kmf_fast.fit(fast_data['time_to_event'], event_observed=fast_data['fallecido'], label='Fast Agers (Q4)')
    kmf_slow.fit(slow_data['time_to_event'], event_observed=slow_data['fallecido'], label='Slow Agers (Q1)')
    
    kmf_fast.plot_survival_function(ax=ax, linewidth=2.5, color='#d62728')
    kmf_slow.plot_survival_function(ax=ax, linewidth=2.5, color='#1f77b4')
    
    plt.title("Kaplan-Meier Survival Curves:\nLatent Velocity Phenotypes", fontsize=14, fontweight='bold')
    plt.xlabel("Years of Follow-Up", fontsize=12)
    plt.ylabel("Survival Probability", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    add_at_risk_counts(kmf_fast, kmf_slow, ax=ax, xticks=[0, 5, 10, 15, 20])
    
    results = logrank_test(fast_data['time_to_event'], slow_data['time_to_event'], 
                           fast_data['fallecido'], slow_data['fallecido'])
    
    textstr = f"Cox Hazard Ratio: {hr:.2f}\nLog-Rank p-value: {results.p_value:.1e}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.05, 0.15, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', bbox=props)
            
    plt.tight_layout()
    output_path = str(PLOTS_DIR / 'km_survival_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved {output_path}")

def plot_archetypal_trajectories(df_traj_full):
    print("Generating Archetypal Trajectories...")
    df_early = df_traj_full.groupby(['cunicah', 'np']).head(10)
    patient_vmag = df_early.groupby(['cunicah', 'np'])['v_mag'].mean().reset_index().sort_values(by='v_mag')
    
    slow_p = patient_vmag.iloc[0]
    fast_p = patient_vmag.iloc[-1]
    
    m_path = str(MODELS_DIR / 'beta_vae_model_128.pth')
    df_obs, _ = extract_latent_vectors(m_path, FI_PATH, DEVICE)
    
    # Optimization: Extract patient data once before the loop
    slow_p_dense = df_traj_full[(df_traj_full['cunicah'] == slow_p['cunicah']) & (df_traj_full['np'] == slow_p['np'])]
    fast_p_dense = df_traj_full[(df_traj_full['cunicah'] == fast_p['cunicah']) & (df_traj_full['np'] == fast_p['np'])]
    slow_p_obs = df_obs[(df_obs['cunicah'] == slow_p['cunicah']) & (df_obs['np'] == slow_p['np'])]
    fast_p_obs = df_obs[(df_obs['cunicah'] == fast_p['cunicah']) & (df_obs['np'] == fast_p['np'])]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(r"Archetypal Latent Trajectories ($\bar{z}_k(t)$): Fast vs Slow Ager", fontsize=18, fontweight='bold')
    axes = axes.flatten()
    
    for k in range(8):
        ax = axes[k]
        
        # Plot Slow
        ax.plot(slow_p_dense['t'], slow_p_dense[f'z_mean_{k}'], color="#1f77b4", linewidth=3, label='Slow Ager (GP)')
        ax.scatter(slow_p_obs['t'], slow_p_obs[f'z_{k}'], color="#1f77b4", marker='o', s=80, edgecolor='white', zorder=5, label='Slow Ager (Obs)')
        
        # Plot Fast
        ax.plot(fast_p_dense['t'], fast_p_dense[f'z_mean_{k}'], color="#d62728", linewidth=3, label='Fast Ager (GP)')
        ax.scatter(fast_p_obs['t'], fast_p_obs[f'z_{k}'], color="#d62728", marker='o', s=80, edgecolor='white', zorder=5, label='Fast Ager (Obs)')
            
        ax.set_title(f"Latent Dimension $z_{k}$", fontsize=14)
        ax.set_xlabel("Years since baseline (t)", fontsize=11)
        ax.grid(True, alpha=0.3)
        if k == 0: ax.legend(fontsize=10)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = str(PLOTS_DIR / 'archetypal_trajectories.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Saved {output_path}")

def plot_velocity_heatmaps(df_traj_full, df_fi, velocity_cols):
    print("Generating Velocity-Domain Heatmaps...")
    
    # --- 9 Granular Clinical Domains (Optimized for Geriatric Phenotyping) ---

    # 1. Clinical (Split into Vascular/Metabolic vs General Chronic)
    metabolic_cols = ['hipertension', 'diabetes', 'infarto', 'embolia']
    chronic_cols   = ['enf_pulm', 'artritis', 'cancer']

    # 2. Functional (Strictly physical limitations)
    func_cols      = ['n_abvd', 'n_aivd', 'n_mov', 'n_img', 'motoras_gruesas', 'motoras_finas']

    # 3. Cognitive (Standard Battery)
    cog_cols       = ['recuerdo1', 'recuerdo2', 'copiafiguras1', 'copiafiguras2', 'orientacion', 'serial7', 'visualscan', 'memoria']

    # 4. Mental / Psychological (Split into Emotional vs Physical Fatigue)
    affective_cols = ['deprimido', 'intranquilo', 'triste', 'solo', 'feliz', 'disf_vida']
    somatic_cols   = ['esfuerzo', 'cansado', 'energia'] # The "Exhaustion" phenotype

    # 5. Subjective Overall Health
    health_cols    = ['salud_glob']

    # 6. Lifestyle & Utilization
    life_cols      = ['bmi_imp', 'ejer_3_por_sem', 'tabaco']
    util_cols      = ['hospitalizacion', 'visita_medica']

    domains = {
        'Metabolic': metabolic_cols,
        'Chronic': chronic_cols,
        'Functional': func_cols, 
        'Cognitive': cog_cols, 
        'Affective_Mood': affective_cols,
        'Somatic_Fatigue': somatic_cols,
        'Self_Rated_Health': health_cols,
        'Lifestyle': life_cols,
        'Utilization': util_cols
    }

    df_fi = df_fi.sort_values(by=['cunicah', 'np', 'a_o_ent']).copy()
    df_fi['t'] = df_fi['a_o_ent'] - 2001
    df_fi['next_t'] = df_fi.groupby(['cunicah', 'np'])['t'].shift(-1)
    
    delta_cols = []
    for d_name, d_vars in domains.items():
        df_fi[f'{d_name}_FI'] = df_fi[d_vars].mean(axis=1)
        df_fi[f'next_{d_name}_FI'] = df_fi.groupby(['cunicah', 'np'])[f'{d_name}_FI'].shift(-1)
        df_fi[f'delta_{d_name}'] = (df_fi[f'next_{d_name}_FI'] - df_fi[f'{d_name}_FI']) / (df_fi['next_t'] - df_fi['t'])
        delta_cols.append(f'delta_{d_name}')
        
    df_fi_valid = df_fi.dropna(subset=['next_t']).copy()
    df_traj_full['a_o_ent'] = (df_traj_full['t'] + 2001).round(0).astype(int)
    
    merged = pd.merge(df_fi_valid, df_traj_full, on=['cunicah', 'np', 'a_o_ent'], how='inner')
    corr_matrix = merged[velocity_cols + delta_cols].corr().loc[velocity_cols, delta_cols]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                cbar_kws={'label': 'Pearson Correlation', 'pad': 0.02}, 
                annot_kws={"size": 10}, linewidths=1, linecolor='white')
                
    plt.title(r'$\beta$-VAE Disentanglement:' + '\nLatent Velocity Component Correlation', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel(r"Latent Velocity Component ($v_k$)", fontsize=14)
    plt.xlabel("Empirical Domain Degradation Rate", fontsize=14)
    plt.xticks(np.arange(len(domains.keys())) + 0.5, list(domains.keys()), fontsize=11, rotation=45)
    
    plt.tight_layout()
    output_path = str(PLOTS_DIR / 'velocity_domain_heatmap.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" -> Saved {output_path}")
    
    return corr_matrix

def execute_survival_analysis(df_traj_full, df_fi, velocity_cols):
    print("\n--- Phase 5.3: Phenotyping & Survival Analysis ---")
    
    df_early = df_traj_full.groupby(['cunicah', 'np']).head(10)
    patient_vmag = df_early.groupby(['cunicah', 'np'])['v_mag'].mean().reset_index()
    
    q1 = patient_vmag['v_mag'].quantile(0.25)
    q3 = patient_vmag['v_mag'].quantile(0.75)
    
    conditions = [(patient_vmag['v_mag'] <= q1), (patient_vmag['v_mag'] >= q3)]
    patient_vmag['Phenotype'] = np.select(conditions, ['Slow_Ager', 'Fast_Ager'], default='Normal')
    patient_vmag = patient_vmag[patient_vmag['Phenotype'] != 'Normal']
    
    master_data_path = str(DATA_DIR / 'simpleMHAS.sav')
    df_true_outcomes, _ = pyreadstat.read_sav(master_data_path, usecols=['cunicah', 'np', 'fallecido', 'a_o_ent', 'edad'])
    df_true_outcomes['fallecido'] = (df_true_outcomes['fallecido'] == 1).astype(int)
    
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
    cph = CoxPHFitter().fit(cph_data, duration_col='time_to_event', event_col='fallecido')
    hr = np.exp(cph.params_['Fast_Ager_Flag'])
    
    # Call the refactored visualization functions
    print("\n--- Phase 6: Generating Final Visualizations ---")
    plot_survival_curves(surv_grouped, hr)
    plot_archetypal_trajectories(df_traj_full)
    
    # Save or utilize the correlation matrix properly
    corr_matrix = plot_velocity_heatmaps(df_traj_full, df_fi, velocity_cols)
    return cph, corr_matrix

if __name__ == "__main__":
    # Load foundational data ONCE
    df_fi_global = pd.read_csv(FI_PATH)
    
    df_magnitude, velocity_cols = calculate_velocity_magnitude()
    
    try:
        execute_lmm_clinical_validation(df_magnitude, df_fi_global)
        cph_model, corr_mat = execute_survival_analysis(df_magnitude, df_fi_global, velocity_cols)
    except ValueError as e:
        print(f"Execution aborted: {e}")