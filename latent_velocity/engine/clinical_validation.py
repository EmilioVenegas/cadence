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

# Global configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FI_PATH = str(DATA_DIR / 'frailty_index_data.csv')

def calculate_velocity_magnitude(traj_path=None):
    if traj_path is None:
        traj_path = str(MODELS_DIR / 'latent_velocity_trajectory_128.csv')
    print(f"Loading high-resolution Latent Velocity dataset from {traj_path}...")

    df_traj = pd.read_csv(traj_path)
    # Infer latent dim from z_mean columns to avoid picking up v_uncertainty
    n_latent = sum(1 for c in df_traj.columns if c.startswith('z_mean_'))
    velocity_cols = [f'v_{k}' for k in range(n_latent)]
    df_traj['v_mag'] = np.sqrt((df_traj[velocity_cols] ** 2).sum(axis=1))
    return df_traj, velocity_cols


def compute_frailty_velocity(df_traj, df_fi, velocity_cols):
    """
    Project latent velocity onto the direction that best predicts empirical FI change.
    Adds a signed 'v_frailty' column: positive = deteriorating, negative = improving.

    Two design choices over the naive instantaneous approach:
    1. Uses 34-feature FI (matching the VAE training set) rather than the CSV FI column,
       which may still include tabaco/ejer removed from the VAE deficit space.
    2. Averages the dense trajectory velocity over each clinical interval [t_start, t_end]
       before regressing — this matches the temporal scale of fi_vel = dFI/dt, which is
       itself an average over the same interval.
    """
    from sklearn.linear_model import Ridge

    deficit_cols_34 = [
        'hipertension', 'diabetes', 'enf_pulm', 'artritis', 'infarto', 'embolia', 'cancer', 'salud_glob',
        'n_abvd', 'n_aivd', 'n_mov', 'n_img', 'motoras_gruesas', 'motoras_finas',
        'deprimido', 'esfuerzo', 'intranquilo', 'triste', 'cansado', 'solo', 'feliz', 'disf_vida', 'energia',
        'recuerdo1', 'recuerdo2', 'copiafiguras1', 'copiafiguras2', 'orientacion', 'serial7', 'visualscan', 'memoria',
        'bmi_imp', 'hospitalizacion', 'visita_medica',
    ]

    df_fi = df_fi.copy()
    df_fi['t'] = df_fi['a_o_ent'] - 2001
    available = [c for c in deficit_cols_34 if c in df_fi.columns]
    df_fi['fi_34'] = df_fi[available].mean(axis=1)

    df_fi = df_fi.sort_values(['cunicah', 'np', 't'])
    df_fi['next_fi_34'] = df_fi.groupby(['cunicah', 'np'])['fi_34'].shift(-1)
    df_fi['next_t']     = df_fi.groupby(['cunicah', 'np'])['t'].shift(-1)
    df_fi_pairs = df_fi.dropna(subset=['fi_34', 'next_fi_34', 't', 'next_t']).copy()
    df_fi_pairs = df_fi_pairs[df_fi_pairs['next_t'] > df_fi_pairs['t']]
    df_fi_pairs['fi_vel'] = ((df_fi_pairs['next_fi_34'] - df_fi_pairs['fi_34']) /
                             (df_fi_pairs['next_t']     - df_fi_pairs['t']))

    intervals = df_fi_pairs[['cunicah', 'np', 't', 'next_t', 'fi_vel']].reset_index(drop=True)

    # For each dense trajectory point, assign it to the clinical interval it falls in:
    # merge_asof backward finds the most recent interval start ≤ t_traj,
    # then we keep only points where t_traj ≤ interval end.
    traj_v = (df_traj[['cunicah', 'np', 't'] + velocity_cols]
              .dropna(subset=['t'])
              .sort_values('t'))
    intervals_sorted = (intervals
                        .rename(columns={'t': 't_start'})
                        .sort_values('t_start'))

    traj_assigned = pd.merge_asof(
        traj_v,
        intervals_sorted[['cunicah', 'np', 't_start', 'next_t', 'fi_vel']],
        left_on='t', right_on='t_start',
        by=['cunicah', 'np'],
        direction='backward',
    )
    traj_in_interval = traj_assigned[
        traj_assigned['t_start'].notna() & (traj_assigned['t'] <= traj_assigned['next_t'])
    ]
    interval_means = (traj_in_interval
                      .groupby(['cunicah', 'np', 't_start', 'fi_vel'])[velocity_cols]
                      .mean()
                      .reset_index())

    valid = interval_means[velocity_cols + ['fi_vel']].notna().all(axis=1)
    ridge = Ridge(alpha=1.0)
    ridge.fit(interval_means.loc[valid, velocity_cols], interval_means.loc[valid, 'fi_vel'])
    r2 = ridge.score(interval_means.loc[valid, velocity_cols], interval_means.loc[valid, 'fi_vel'])
    w_fi = ridge.coef_
    w_fi_norm = w_fi / np.linalg.norm(w_fi)

    df_traj = df_traj.copy()
    df_traj['v_frailty'] = df_traj[velocity_cols].values @ w_fi_norm
    print(f"  FI-velocity direction fitted: {valid.sum()} intervals, R²={r2:.4f}, "
          f"||w_fi||={np.linalg.norm(w_fi):.4f}")
    return df_traj

def execute_lmm_clinical_validation(df_traj, df_fi):
    print("\n--- Phase 5.2: Ground-Truth Clinical Validation (LMM) ---")
    
    df_fi = df_fi.sort_values(by=['cunicah', 'np', 'a_o_ent']).copy()
    
    df_fi['t'] = df_fi['a_o_ent'] - 2001
    df_fi['next_t'] = df_fi.groupby(['cunicah', 'np'])['t'].shift(-1)
    df_fi['next_FI'] = df_fi.groupby(['cunicah', 'np'])['FI'].shift(-1)
    
    df_fi_valid = df_fi.dropna(subset=['next_t', 'next_FI']).copy()
    df_fi_valid['empirical_velocity'] = (df_fi_valid['next_FI'] - df_fi_valid['FI']) / (df_fi_valid['next_t'] - df_fi_valid['t'])

    # Snap each sparse FI observation to the nearest dense trajectory point.
    # A round-to-integer-year merge would produce ~10 matches per observation
    # (one per 0.1-step grid point in that year), inflating LMM sample size 10×.
    traj_snap = df_traj[['cunicah', 'np', 't', 'v_mag']].sort_values('t')
    merged_data = pd.merge_asof(
        df_fi_valid.sort_values('t'),
        traj_snap,
        on='t',
        by=['cunicah', 'np'],
        direction='nearest',
    )

    if merged_data['v_mag'].isna().all():
        raise ValueError("CRITICAL MERGE ERROR: No trajectory matches found. Check patient ID alignment.")
        
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
    df_early = df_traj_full.groupby(['cunicah', 'np']).head(30)
    # Use signed frailty velocity so fast = deteriorating, slow = improving
    patient_vf = (
        df_early.groupby(['cunicah', 'np'])['v_frailty'].mean()
        .reset_index().sort_values(by='v_frailty')
    )

    slow_p = patient_vf.iloc[0]   # most negative = improving fastest
    fast_p = patient_vf.iloc[-1]  # most positive = deteriorating fastest
    
    slow_p_dense = df_traj_full[(df_traj_full['cunicah'] == slow_p['cunicah']) & (df_traj_full['np'] == slow_p['np'])]
    fast_p_dense = df_traj_full[(df_traj_full['cunicah'] == fast_p['cunicah']) & (df_traj_full['np'] == fast_p['np'])]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(r"Archetypal Latent Trajectories ($\bar{z}_k(t)$): Fast vs Slow Ager", fontsize=18, fontweight='bold')
    axes = axes.flatten()

    for k in range(8):
        ax = axes[k]

        ax.plot(slow_p_dense['t'], slow_p_dense[f'z_mean_{k}'], color="#1f77b4", linewidth=3, label='Slow Ager (ODE)')
        ax.plot(fast_p_dense['t'], fast_p_dense[f'z_mean_{k}'], color="#d62728", linewidth=3, label='Fast Ager (ODE)')
            
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

    traj_snap = df_traj_full[['cunicah', 'np', 't'] + velocity_cols].sort_values('t')
    merged = pd.merge_asof(
        df_fi_valid.sort_values('t'),
        traj_snap,
        on='t',
        by=['cunicah', 'np'],
        direction='nearest',
    )
    corr_matrix = merged[velocity_cols + delta_cols].corr().loc[velocity_cols, delta_cols]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                cbar_kws={'label': 'Pearson Correlation', 'pad': 0.02}, 
                annot_kws={"size": 10}, linewidths=1, linecolor='white')
                
    # Title omitted per BMC Bioinformatics (lives in the manuscript caption).
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

    # Use v_frailty (signed projection onto FI gradient) so that Fast_Ager means
    # "moving toward worse health", not just "moving fast in any direction".
    df_early = df_traj_full.groupby(['cunicah', 'np']).head(30)
    patient_vf = (
        df_early.groupby(['cunicah', 'np'])['v_frailty'].mean()
        .reset_index().rename(columns={'v_frailty': 'v_frailty_mean'})
    )

    # Mean GP uncertainty over the early trajectory window. Patients with sparse
    # observations (who died early or dropped out) have higher v_uncertainty.
    # Including it as an explicit Cox covariate separates the "observation completeness"
    # signal from the biological velocity signal, preventing the two from conflating.
    if 'v_uncertainty' in df_traj_full.columns:
        patient_unc = (
            df_early.groupby(['cunicah', 'np'])['v_uncertainty'].mean()
            .reset_index().rename(columns={'v_uncertainty': 'mean_unc'})
        )
        patient_vf = patient_vf.merge(patient_unc, on=['cunicah', 'np'], how='left')
    else:
        patient_vf['mean_unc'] = np.nan

    # Baseline age is kept for Cox covariate adjustment — do not residualize v_frailty
    # before phenotyping. Prior residualization + Cox age covariate = double adjustment,
    # which compresses the phenotype signal. Cox handles the age confound on its own.
    patient_ages = df_fi.groupby(['cunicah', 'np'])['edad'].min().reset_index(name='baseline_age')
    patient_vf = patient_vf.merge(patient_ages, on=['cunicah', 'np'], how='left')

    patient_vf['v_frailty_adj'] = patient_vf['v_frailty_mean']  # raw, no age residualization

    q1 = patient_vf['v_frailty_adj'].quantile(0.25)
    q3 = patient_vf['v_frailty_adj'].quantile(0.75)

    conditions = [
        patient_vf['v_frailty_adj'] <= q1,
        patient_vf['v_frailty_adj'] >= q3,
    ]
    patient_vf['Phenotype'] = np.select(conditions, ['Slow_Ager', 'Fast_Ager'], default='Normal')

    # Survivorship bias: single-obs deceased patients couldn't have GP velocity
    # computed and are silently excluded. They are the fastest agers (died within
    # one wave interval). Assign them Fast_Ager to avoid attenuating the Cox HR.
    obs_counts = df_fi.groupby(['cunicah', 'np']).size().reset_index(name='n_obs')
    single_obs = obs_counts[obs_counts['n_obs'] == 1]
    single_obs_dead = single_obs.merge(
        df_fi[['cunicah', 'np', 'fallecido']].query('fallecido == 1').drop_duplicates(),
        on=['cunicah', 'np'], how='inner'
    )
    if not single_obs_dead.empty:
        n_rescued = len(single_obs_dead)
        print(f"  Survivorship fix: adding {n_rescued} single-obs deceased patients as Fast_Agers.")
        q3_vf = patient_vf['v_frailty_mean'].quantile(0.75)
        rescued = single_obs_dead[['cunicah', 'np']].copy()
        rescued['v_frailty_mean'] = q3_vf + 1e-3
        rescued['v_frailty_adj']  = q3_vf + 1e-3
        rescued['baseline_age']   = single_obs_dead.merge(
            patient_ages, on=['cunicah', 'np'], how='left'
        )['baseline_age'].values
        # Single-obs patients have no dense trajectory → highest uncertainty by definition
        rescued['mean_unc'] = patient_vf['mean_unc'].quantile(0.75)
        rescued['Phenotype'] = 'Fast_Ager'
        patient_vf = pd.concat([patient_vf, rescued], ignore_index=True)

    patient_vf = patient_vf[patient_vf['Phenotype'] != 'Normal']

    # Re-use already-loaded df_fi for outcomes instead of re-reading the raw SAV
    df_true_outcomes = df_fi[['cunicah', 'np', 'fallecido', 'a_o_ent', 'edad']].copy()
    df_true_outcomes['fallecido'] = (df_true_outcomes['fallecido'] == 1).astype(int)

    # ── Issue 8: Death year precision ────────────────────────────────────────
    # MHAS reports fallecido=1 in the wave AFTER death — end_year is the
    # death-report wave, not the actual death year (up to ~3 years late).
    # For deceased patients, use last_alive_year (last wave where fallecido=0)
    # as the event time, which is a conservative lower bound on survival duration.
    last_alive = (
        df_true_outcomes[df_true_outcomes['fallecido'] == 0]
        .groupby(['cunicah', 'np'])['a_o_ent'].max()
        .reset_index(name='last_alive_year')
    )

    surv_data = pd.merge(patient_vf, df_true_outcomes, on=['cunicah', 'np'], how='left')
    surv_grouped = surv_data.groupby(['cunicah', 'np']).agg(
        Phenotype=('Phenotype', 'first'),
        fallecido=('fallecido', 'max'),
        start_year=('a_o_ent', 'min'),
        end_year=('a_o_ent', 'max'),
        baseline_age=('edad', 'min'),
        mean_unc=('mean_unc', 'first'),
    ).reset_index()

    surv_grouped = surv_grouped.merge(last_alive, on=['cunicah', 'np'], how='left')
    deceased_mask = surv_grouped['fallecido'] == 1
    surv_grouped.loc[deceased_mask, 'end_year'] = (
        surv_grouped.loc[deceased_mask, 'last_alive_year']
        .fillna(surv_grouped.loc[deceased_mask, 'end_year'])
    )

    surv_grouped['time_to_event'] = surv_grouped['end_year'] - surv_grouped['start_year']
    surv_grouped = surv_grouped.dropna(subset=['time_to_event', 'fallecido'])
    surv_grouped = surv_grouped[surv_grouped['time_to_event'] > 0]
    surv_grouped['Fast_Ager_Flag'] = (surv_grouped['Phenotype'] == 'Fast_Ager').astype(float)

    # Standardize mean_unc before entering Cox (puts it on the same scale as other covariates)
    unc_mean = surv_grouped['mean_unc'].mean()
    unc_std  = surv_grouped['mean_unc'].std()
    surv_grouped['mean_unc_z'] = (surv_grouped['mean_unc'] - unc_mean) / unc_std

    cph_data = surv_grouped[['time_to_event', 'fallecido', 'Fast_Ager_Flag',
                              'baseline_age', 'mean_unc_z']].dropna()
    cph = CoxPHFitter().fit(cph_data, duration_col='time_to_event', event_col='fallecido')
    print("\n[COX MODEL]")
    print(cph.summary[['coef', 'exp(coef)', 'p']].round(4))
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
    df_magnitude = compute_frailty_velocity(df_magnitude, df_fi_global, velocity_cols)

    try:
        execute_lmm_clinical_validation(df_magnitude, df_fi_global)
        cph_model, corr_mat = execute_survival_analysis(df_magnitude, df_fi_global, velocity_cols)
    except ValueError as e:
        print(f"Execution aborted: {e}")