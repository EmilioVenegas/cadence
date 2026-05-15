import pandas as pd
import torch
import numpy as np
from _paths import MODELS_DIR

def prepare_ode_pairs(input_path, output_path, dt=3.0):
    print(f"Loading high-resolution trajectories from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Identify columns
    z_cols = [col for col in df.columns if col.startswith('z_mean_')]
    # Derive v_cols explicitly from latent dim count to avoid picking up v_uncertainty
    v_cols = [f'v_{k}' for k in range(len(z_cols))]
    
    # Join with raw data to get lifestyle/clinical features using merge_asof
    from _paths import DATA_DIR
    df_raw = pd.read_csv(DATA_DIR / 'frailty_index_data.csv')
    
    # Derive social_isolation: 1 = isolated (bad), 0 = engaged (good)
    # Polarity: asiste_club=1 and voluntario=1 mean participation (good),
    # so we invert: social_isolation = 1 - max(asiste_club, voluntario)
    df_raw['social_isolation'] = 1.0 - df_raw[['asiste_club', 'voluntario']].max(axis=1)
    
    u_cols = ['tabaco', 'bmi_imp', 'ejer_3_por_sem', 'hipertension', 'diabetes', 'alcohol', 'social_isolation']
    # Note: ejer_3_por_sem is stored inverted in frailty_index_data.csv (1=sedentary=bad, 0=exercises=good),
    # because prepare_frailty_data.py maps it as 1 - raw. TARGET_MAP targets 0.0 (exercises).
    
    df_u = df_raw[['cunicah', 'np', 'a_o_ent'] + u_cols].rename(columns={'a_o_ent': 't'})
    
    # Sort both for merge_asof (crucial: must be sorted by the 'on' key)
    # Also drop NaNs from the merge key
    df = df.dropna(subset=['t']).sort_values(by='t')
    df_u = df_u.dropna(subset=['t']).sort_values(by='t')
    
    # Merge using merge_asof to handle high-res time mapping
    # Note: merge_asof with 'by' requires the 'on' key to be sorted globally in some pandas versions
    df = pd.merge_asof(
        df, df_u, 
        on='t', 
        by=['cunicah', 'np'], 
        direction='nearest'
    )
    
    # Ensure no NaNs remain after merge (should be very few if any)
    df[u_cols] = df.groupby(['cunicah', 'np'])[u_cols].ffill().bfill().fillna(0.0)

    # Filter out the 10% most uncertain trajectory points (prior-dominated interpolation
    # in the middle of long observation gaps contributes noise, not signal, to ODE training).
    if 'v_uncertainty' in df.columns:
        uncertainty_threshold = df['v_uncertainty'].quantile(0.90)
        n_before = len(df)
        df = df[df['v_uncertainty'] <= uncertainty_threshold].copy()
        print(f"Uncertainty filter: removed {n_before - len(df)} high-uncertainty points "
              f"(threshold={uncertainty_threshold:.4f}).")

    pairs = []

    # Non-overlapping stride: consecutive pair windows are separated by exactly dt.
    # Previously every 0.1-year step generated a pair, creating ~30 correlated pairs
    # per non-overlapping window. Stride = dt / grid_step = 3.0 / 0.1 = 30.
    T_GRID_STEP = 0.1
    stride = max(1, int(dt / T_GRID_STEP))

    print(f"Slicing trajectories with non-overlapping stride={stride} (dt={dt}y, step={T_GRID_STEP}y)...")
    grouped = df.groupby(['cunicah', 'np'])

    for (cunicah, np_val), p_data in grouped:
        p_data = p_data.sort_values(by='t')
        times = p_data['t'].values
        z_vals = p_data[z_cols].values
        v_vals = p_data[v_cols].values
        u_vals = p_data[u_cols].values

        t_max = times.max()

        for i in range(0, len(times), stride):
            t_start = times[i]
            t_target = t_start + dt
            if t_target > t_max:
                continue

            idx_end = np.where(np.abs(times - t_target) < 1e-4)[0]

            if len(idx_end) > 0:
                idx_end = idx_end[0]
                z_0 = z_vals[i]
                z_T = z_vals[idx_end]
                v_0 = v_vals[i]
                u_0 = u_vals[i]  # lifestyle at the start of the interval

                pairs.append({
                    'z_0': z_0,
                    'z_T': z_T,
                    'v_0': v_0,
                    'u_0': u_0,
                    'dt': dt
                })
                
    if not pairs:
        print("Error: No valid pairs found.")
        return

    # Convert to Tensors
    train_z0 = torch.tensor([p['z_0'] for p in pairs], dtype=torch.float32)
    train_zT = torch.tensor([p['z_T'] for p in pairs], dtype=torch.float32)
    train_v0 = torch.tensor([p['v_0'] for p in pairs], dtype=torch.float32)
    train_u0 = torch.tensor([p['u_0'] for p in pairs], dtype=torch.float32)
    train_dt = torch.tensor([p['dt'] for p in pairs], dtype=torch.float32)
    
    data_dict = {
        'z_0': train_z0,
        'z_T': train_zT,
        'v_0': train_v0,
        'u_0': train_u0,
        'dt': train_dt
    }
    
    torch.save(data_dict, output_path)
    print(f"Successfully generated {len(pairs)} training pairs.")
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    traj_path = str(MODELS_DIR / 'latent_velocity_trajectory_128.csv')
    out_path = str(MODELS_DIR / 'ode_training_pairs_128.pth')
    prepare_ode_pairs(traj_path, out_path)
