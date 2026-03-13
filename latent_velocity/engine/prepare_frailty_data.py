import pandas as pd
import numpy as np
import pyreadstat
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
from _paths import DATA_DIR

def load_data(filepath):
    print("Loading simpleMHAS.sav...")
    df, meta = pyreadstat.read_sav(filepath)
    return df

def map_clinical(df):
    print("Mapping Clinical & Health Conditions...")
    cols = ['hipertension', 'diabetes', 'enf_pulm', 'artritis', 'infarto', 'embolia', 'cancer']
    # These are already 0/1, just copy them
    df_mapped = df[cols].copy()
    
    # salud_glob: 1 (Excellent) to 5 (Poor)
    if 'salud_glob' in df.columns:
        df_mapped['salud_glob'] = (df['salud_glob'] - 1) / 4.0
    
    return df_mapped

def map_functionality(df):
    print("Mapping Functionality & Mobility...")
    df_mapped = pd.DataFrame(index=df.index)
    if 'n_abvd' in df.columns: df_mapped['n_abvd'] = df['n_abvd'] / 5.0
    if 'n_aivd' in df.columns: df_mapped['n_aivd'] = df['n_aivd'] / 4.0
    if 'n_mov' in df.columns: df_mapped['n_mov'] = df['n_mov'] / 5.0
    if 'n_img' in df.columns: df_mapped['n_img'] = df['n_img'] / 4.0
    if 'motoras_gruesas' in df.columns: df_mapped['motoras_gruesas'] = df['motoras_gruesas'] / 5.0
    if 'motoras_finas' in df.columns: df_mapped['motoras_finas'] = df['motoras_finas'] / 3.0
    return df_mapped

def map_mental_health(df):
    print("Mapping Mental Health & Psychosocial...")
    df_mapped = pd.DataFrame(index=df.index)
    
    # Direct binary
    direct_cols = ['deprimido', 'esfuerzo', 'intranquilo', 'triste', 'cansado', 'solo']
    for col in direct_cols:
        if col in df.columns:
            df_mapped[col] = df[col]
            
    # Reverse code binary
    reverse_cols = ['feliz', 'disf_vida', 'energia']
    for col in reverse_cols:
        if col in df.columns:
            df_mapped[col] = 1.0 - df[col]
            
    return df_mapped

def map_cognition(df):
    print("Mapping Cognition...")
    df_mapped = pd.DataFrame(index=df.index)
    if 'recuerdo1' in df.columns: df_mapped['recuerdo1'] = (8.0 - df['recuerdo1']) / 8.0
    if 'recuerdo2' in df.columns: df_mapped['recuerdo2'] = (8.0 - df['recuerdo2']) / 8.0
    if 'copiafiguras1' in df.columns: df_mapped['copiafiguras1'] = (2.0 - df['copiafiguras1']) / 2.0
    if 'copiafiguras2' in df.columns: df_mapped['copiafiguras2'] = (2.0 - df['copiafiguras2']) / 2.0
    if 'orientacion' in df.columns: df_mapped['orientacion'] = (3.0 - df['orientacion']) / 3.0
    if 'serial7' in df.columns: df_mapped['serial7'] = (5.0 - df['serial7']) / 5.0
    
    if 'visualscan' in df.columns:
        df_mapped['visualscan'] = (60.0 - df['visualscan']) / 60.0
        
    if 'memoria' in df.columns:
        df_mapped['memoria'] = (df['memoria'] - 1) / 4.0
        
    return df_mapped

def map_biometrics(df):
    print("Mapping Biometrics & Physical Factors...")
    df_mapped = pd.DataFrame(index=df.index)
    
    if 'bmi_imp' in df.columns:
        # BMI < 18.5 or BMI > 30 = 1, else 0
        df_mapped['bmi_imp'] = np.where((df['bmi_imp'] < 18.5) | (df['bmi_imp'] > 30.0), 1.0, 0.0)
        # Preserve NaNs
        df_mapped.loc[df['bmi_imp'].isna(), 'bmi_imp'] = np.nan
        
    if 'ejer_3_por_sem' in df.columns:
        df_mapped['ejer_3_por_sem'] = 1.0 - df['ejer_3_por_sem']
        
    if 'tabaco' in df.columns:
        df_mapped['tabaco'] = df['tabaco']
        
    return df_mapped

def map_healthcare(df):
    print("Mapping Healthcare Utilization...")
    df_mapped = pd.DataFrame(index=df.index)
    if 'hospitalizacion' in df.columns: df_mapped['hospitalizacion'] = df['hospitalizacion']
    if 'visita_medica' in df.columns: df_mapped['visita_medica'] = df['visita_medica']
    return df_mapped

def prepare_data(filepath, output_path):
    df_raw = load_data(filepath)
    
    # 0. Global Missing Code Cleanup (MHAS specific 8/9, 88/99 codes)
    print("Converting MHAS missing codes (8, 9, 88, 99) to NaNs...")
    
    # Category A: Binary/Categorical (where 8/9 are missing codes)
    binary_cols = [
        'hipertension', 'diabetes', 'enf_pulm', 'artritis', 'infarto', 'embolia', 'cancer', 'salud_glob',
        'deprimido', 'esfuerzo', 'intranquilo', 'triste', 'cansado', 'solo', 'feliz', 'disf_vida', 'energia',
        'asiste_club', 'voluntario', 'lee', 'cruci_rompe', 'ejer_3_por_sem', 'hospitalizacion', 'visita_medica'
    ]
    
    # Category B: Continuous Cognitive (where 8 is a VALID score, only 88/98/99 are missing)
    continuous_cog_cols = [
        'recuerdo1', 'recuerdo2', 'copiafiguras1', 'copiafiguras2', 'orientacion', 'serial7', 'visualscan'
    ]
    
    for col in binary_cols:
        if col in df_raw.columns:
            df_raw.loc[df_raw[col].isin([8, 9]), col] = np.nan
            
    for col in continuous_cog_cols:
        if col in df_raw.columns:
            df_raw.loc[df_raw[col].isin([88, 98, 99]), col] = np.nan

    # Static covariates to keep
    static_vars = [
        'cunicah', 'np', 'ronda', 'a_o_ent', 'edad', 'sexo', 'educacion', 'urbano', 'fallecido', 'a_o_nac',
        'imp_neto', 'est_conyugal', 'n_hijos_vivos', 'alcohol', 'asiste_club', 'voluntario', 'lee', 'cruci_rompe'
    ]
    # Check which ones exist
    existing_static = [v for v in static_vars if v in df_raw.columns]
    df_static = df_raw[existing_static].copy()
    
    # 1. Map all deficit variables to [0, 1]
    df_clinical = map_clinical(df_raw)
    df_func = map_functionality(df_raw)
    df_mental = map_mental_health(df_raw)
    df_cog = map_cognition(df_raw)
    df_bio = map_biometrics(df_raw)
    df_health = map_healthcare(df_raw)
    
    # Combine all mapped deficits
    df_deficits = pd.concat([df_clinical, df_func, df_mental, df_cog, df_bio, df_health], axis=1)
    
    # 2. Check variance of visita_medica
    if 'visita_medica' in df_deficits.columns:
        mean_val = df_deficits['visita_medica'].mean()
        print(f"'visita_medica' mean value: {mean_val:.4f}")
        if mean_val > 0.95 or mean_val < 0.05:
            print("Dropping 'visita_medica' due to critically low variance.")
            df_deficits.drop(columns=['visita_medica'], inplace=True)
            
    items = df_deficits.columns.tolist()
    total_items = len(items)
    print(f"Total deficit items selected: {total_items}")
    
    # Add grouping variables back for imputation by wave
    df_combined = pd.concat([df_static, df_deficits], axis=1)
    
    # 2b. Explicitly filter out proxy interviews (Deceased NEXT-OF-KIN are KEPT for survival markers)
    print("Filtering out proxy interviews (retaining Exit/Next-of-kin for deaths)...")
    initial_count = len(df_combined)
    
    # Keep direct interviews (1, 2) and Next-of-kin Exit interviews (5)
    valid_tipent = getattr(df_raw, 'tipent', pd.Series([1]*len(df_raw)))
    df_combined = df_combined[valid_tipent.isin([1, 2, 5])]
    
    print(f"Dropped {initial_count - len(df_combined)} rows based on explicit proxy criteria.")
    
    # Update deficits dataframe to match the filtered combined dataframe
    df_deficits = df_combined[items]
    
    # 3. The 20% Missingness Rule (Searle Standard)
    print("Applying 20% Missingness Rule (Exempting terminal Exit interviews)...")
    max_missing = int(total_items * 0.20)
    missing_counts = df_deficits.isna().sum(axis=1)
    
    # Keep if:
    # A) Missingness is low (<= 20%)
    # B) Record represents a death event (even if 100% missing items)
    valid_mask = (missing_counts <= max_missing) | (df_combined['fallecido'] == 1)
    
    print(f"Dropping {sum(~valid_mask)} additional patient-waves with >{max_missing} missing items.")
    df_valid = df_combined[valid_mask].copy()
    
    # 4. Perform Multiple Imputation by Chained Equations (MICE) within each wave
    print("Performing MICE imputation for remaining missing items (per wave)...")
    impute_covariates = ['edad', 'sexo', 'educacion']
    features_to_impute = items + impute_covariates
    
    imputer = IterativeImputer(random_state=42, max_iter=10, keep_empty_features=True)
    
    imputed_dfs = []
    waves = df_valid['ronda'].unique()
    for wave in sorted(waves):
        print(f"  Imputing wave {wave}...")
        wave_data = df_valid[df_valid['ronda'] == wave].copy()
        
        if len(wave_data) > 0:
            # Subset features including covariates
            active_cols = [c for c in features_to_impute if c in wave_data.columns]
            wave_subset = wave_data[active_cols]
            
            # MICE
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                imputed_arrays = imputer.fit_transform(wave_subset)
            
            # Reconstruct
            imputed_df = pd.DataFrame(imputed_arrays, columns=active_cols, index=wave_data.index)
            # Clip items (not covariates) to [0, 1]
            for col in items:
                if col in imputed_df.columns:
                    imputed_df[col] = imputed_df[col].clip(0, 1)
            
            for col in active_cols:
                wave_data[col] = imputed_df[col]
            
            imputed_dfs.append(wave_data)
            
    df_imputed = pd.concat(imputed_dfs)
    
    # 4b. Perform Cross-wave MICE for completely empty columns in specific waves
    print("Checking for remaining NaNs (items completely missing in a wave)...")
    if df_imputed[items].isna().sum().sum() > 0:
        print("Performing cross-wave MICE for items missing across entire waves...")
        cross_imputer = IterativeImputer(random_state=42, max_iter=10)
        
        cross_subset = df_imputed[[c for c in features_to_impute if c in df_imputed.columns]]
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputed_arrays_cross = cross_imputer.fit_transform(cross_subset)
        
        df_imputed_cross = pd.DataFrame(imputed_arrays_cross, columns=cross_subset.columns, index=df_imputed.index)
        
        for col in items:
            if col in df_imputed_cross.columns:
                df_imputed[col] = df_imputed_cross[col].clip(0, 1)
    
    # 5. Calculate Frailty Index (FI)
    print("Calculating Frailty Index (FI)...")
    df_imputed['FI'] = df_imputed[items].sum(axis=1) / total_items
    
    # 6. Mask out post-mortem waves
    print("Masking out post-mortem waves...")
    post_mortem_mask = (df_imputed['fallecido'] == 1)
    print(f"Masking FI/items for {sum(post_mortem_mask)} records where fallecido=1 (death identified since last wave).")
    df_imputed.loc[post_mortem_mask, 'FI'] = np.nan
    df_imputed.loc[post_mortem_mask, items] = np.nan
    
    # Save the final dataset
    print(f"Saving to {output_path}...")
    df_imputed.to_csv(output_path, index=False)
    print("Done!")
    
if __name__ == "__main__":
    input_file = str(DATA_DIR / 'simpleMHAS.sav')
    output_file = str(DATA_DIR / 'frailty_index_data.csv')
    prepare_data(input_file, output_file)
