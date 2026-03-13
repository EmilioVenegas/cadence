import pandas as pd
import numpy as np
import os

# Load data
# Based on ls output, the directory is nested
data_dir = '/home/emiliovenegas/Documents/mendel/latent_velocity'
fi_path = os.path.join(data_dir, 'frailty_index_data.csv')
traj_path = os.path.join(data_dir, 'latent_velocity', 'latent_velocity_trajectory.csv')

if not os.path.exists(traj_path):
    print(f"Error: {traj_path} not found.")
    # Try local search
    traj_path = os.path.join(data_dir, 'latent_velocity_trajectory.csv')
    print(f"Trying: {traj_path}")

df_fi = pd.read_csv(fi_path)
df_traj = pd.read_csv(traj_path)

# Compute domains exactly like clinical_validation.py does
cog_cols = ['recuerdo1', 'recuerdo2', 'copiafiguras1', 'orientacion', 'serial7', 'memoria']
phys_cols = ['n_abvd', 'n_aivd', 'n_mov', 'n_img', 'motoras_gruesas', 'motoras_finas']
ment_cols = ['deprimido', 'esfuerzo', 'intranquilo', 'triste', 'cansado', 'solo', 'feliz', 'disf_vida', 'energia']

df_fi = df_fi.sort_values(by=['cunicah', 'np', 'a_o_ent'])
df_fi['Cog_FI'] = df_fi[cog_cols].mean(axis=1)
df_fi['Phys_FI'] = df_fi[phys_cols].mean(axis=1)
df_fi['Ment_FI'] = df_fi[ment_cols].mean(axis=1)
df_fi['t'] = df_fi['a_o_ent'] - 2001

for col in ['t', 'Cog_FI', 'Phys_FI', 'Ment_FI']:
    df_fi[f'next_{col}'] = df_fi.groupby(['cunicah', 'np'])[col].shift(-1)
    
df_fi_valid = df_fi.dropna(subset=['next_t']).copy()
for domain in ['Cog_FI', 'Phys_FI', 'Ment_FI']:
    df_fi_valid[f'delta_{domain}'] = (df_fi_valid[f'next_{domain}'] - df_fi_valid[domain]) / (df_fi_valid['next_t'] - df_fi_valid['t'])
    
df_traj['t_round'] = df_traj['t'].round(2)
df_fi_valid['t_round'] = df_fi_valid['t'].round(2)

merged = pd.merge(df_fi_valid[['cunicah', 'np', 't_round', 'Cog_FI', 'Phys_FI', 'Ment_FI', 'delta_Cog_FI', 'delta_Phys_FI', 'delta_Ment_FI']], 
                  df_traj, on=['cunicah', 'np', 't_round'], how='inner')

velocity_cols = [f'v_{k}' for k in range(8)]
delta_cols = ['delta_Cog_FI', 'delta_Phys_FI', 'delta_Ment_FI']

corr_matrix = merged[velocity_cols + delta_cols].corr().loc[velocity_cols, delta_cols]
print("\n--- Correlation Matrix (Velocity vs clinical Domain Decline) ---")
print(corr_matrix)

# Also check absolute position correlations
z_cols = [f'z_mean_{k}' for k in range(8)]
pos_cols = ['Cog_FI', 'Phys_FI', 'Ment_FI']
pos_corr = merged[z_cols + pos_cols].corr().loc[z_cols, pos_cols]
print("\n--- Correlation Matrix (Latent Position vs clinical Domain State) ---")
print(pos_corr)
