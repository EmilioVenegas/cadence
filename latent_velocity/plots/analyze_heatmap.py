import sys
from pathlib import Path

# Add the engine directory to the Python path for imports
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "engine"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from _paths import DATA_DIR, MODELS_DIR, HEATMAP_DIR

# Load data
fi_path = str(DATA_DIR / 'frailty_index_data.csv')
traj_path = str(MODELS_DIR / 'latent_velocity_trajectory.csv')

df_fi = pd.read_csv(fi_path)
df_traj = pd.read_csv(traj_path)

# 1. Expand Domains & Categories
cog_cols = ['recuerdo1', 'recuerdo2', 'copiafiguras1', 'orientacion', 'serial7', 'memoria']
phys_cols = ['n_abvd', 'n_aivd', 'n_mov', 'n_img', 'motoras_gruesas', 'motoras_finas']
ment_cols = ['deprimido', 'esfuerzo', 'intranquilo', 'triste', 'cansado', 'solo', 'feliz', 'disf_vida', 'energia']

# 2. Derive enriched clinical categories
# Derive social_isolation (1.0 - max(club, volunteer))
if 'asiste_club' in df_fi.columns and 'voluntario' in df_fi.columns:
    df_fi['social_isolation'] = 1.0 - df_fi[['asiste_club', 'voluntario']].max(axis=1)

category_cols = ['sexo', 'hipertension', 'diabetes', 'tabaco', 'alcohol', 'ejer_3_por_sem', 'social_isolation']

df_fi = df_fi.sort_values(by=['cunicah', 'np', 'a_o_ent'])
df_fi['Cog_FI'] = df_fi[cog_cols].mean(axis=1)
df_fi['Phys_FI'] = df_fi[phys_cols].mean(axis=1)
df_fi['Ment_FI'] = df_fi[ment_cols].mean(axis=1)
df_fi['t'] = df_fi['a_o_ent'] - 2001

# 3. Calculate Deltas (Velocity) for clinical domains
for col in ['t', 'Cog_FI', 'Phys_FI', 'Ment_FI']:
    df_fi[f'next_{col}'] = df_fi.groupby(['cunicah', 'np'])[col].shift(-1)
    
df_fi_valid = df_fi.dropna(subset=['next_t']).copy()
for domain in ['Cog_FI', 'Phys_FI', 'Ment_FI']:
    df_fi_valid[f'delta_{domain}'] = (df_fi_valid[f'next_{domain}'] - df_fi_valid[domain]) / (df_fi_valid['next_t'] - df_fi_valid['t'])
    
df_traj['t_round'] = df_traj['t'].round(2)
df_fi_valid['t_round'] = df_fi_valid['t'].round(2)

# 4. Merge Latent and Clinical
merge_cols = ['cunicah', 'np', 't_round', 'Cog_FI', 'Phys_FI', 'Ment_FI', 
              'delta_Cog_FI', 'delta_Phys_FI', 'delta_Ment_FI', 'FI', 'edad'] + category_cols
merged = pd.merge(df_fi_valid[merge_cols], df_traj, on=['cunicah', 'np', 't_round'], how='inner')

# 5. Correlation Matrices
velocity_cols = [f'v_{k}' for k in range(8)]
delta_cols = ['delta_Cog_FI', 'delta_Phys_FI', 'delta_Ment_FI']
z_cols = [f'z_mean_{k}' for k in range(8)]
state_cols = ['FI', 'edad', 'Cog_FI', 'Phys_FI', 'Ment_FI'] + category_cols

# Latent Position (State) vs Clinical State
pos_corr = merged[z_cols + state_cols].corr().loc[z_cols, state_cols]
print("\n--- Correlation Matrix (Latent Position vs Clinical State) ---")
print(pos_corr)

# 6. Visualization: Seaborn Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(pos_corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f', linewidths=0.5)
plt.title('LAVA Manifold Alignment: Latent Dimensions (Z) vs Clinical Features', fontsize=15)
plt.ylabel('8D Latent Dimensions')
plt.xlabel('Clinical / Demographic Features')
plt.tight_layout()

save_path = HEATMAP_DIR / 'latent_clinical_heatmap.png'
plt.savefig(str(save_path), dpi=150)
plt.close()
print(f"\nSuccessfully generated Heatmap: {save_path.name}")

# Velocity vs Decline
vel_corr = merged[velocity_cols + delta_cols].corr().loc[velocity_cols, delta_cols]
print("\n--- Correlation Matrix (Latent Velocity vs Clinical Decline) ---")
print(vel_corr)
