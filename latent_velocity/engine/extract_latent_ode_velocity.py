"""
Velocity extraction for the Latent ODE-VAE model.

Replaces extract_velocity.py (β-VAE + GP pipeline).
For each patient, encodes the full observation sequence → z0 distribution,
then integrates the ODE on a dense time grid using MC sampling to propagate
uncertainty. Outputs a CSV in the same format as latent_velocity_trajectory_128.csv
so clinical_validation.py requires no changes.

Usage:
    python latent_velocity/engine/extract_latent_ode_velocity.py
    python latent_velocity/engine/extract_latent_ode_velocity.py --t_step 0.25 --n_mc 50
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchdiffeq import odeint
from _paths import DATA_DIR, MODELS_DIR
from train_latent_ode import (
    LatentODE, LatentODEDataset,
    MHAS_WAVES, T_MAX, LATENT_DIM, N_WAVES,
)


def extract_velocity(model_path=None, csv_path=None, output_path=None,
                     n_mc=30, t_step=0.5):
    if model_path is None: model_path = str(MODELS_DIR / 'latent_ode_model.pth')
    if csv_path   is None: csv_path   = str(DATA_DIR   / 'frailty_index_data.csv')
    if output_path is None:
        output_path = str(MODELS_DIR / 'latent_velocity_trajectory_128.csv')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Latent ODE-VAE velocity extraction | Device: {device} | "
          f"MC samples: {n_mc} | t_step: {t_step}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = LatentODE().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    dataset = LatentODEDataset(
        csv_path,
        edad_mean=ckpt['edad_mean'], edad_std=ckpt['edad_std'],
        edu_mean=ckpt['edu_mean'],   edu_std=ckpt['edu_std'],
    )

    waves_arr  = np.array(MHAS_WAVES, dtype=np.float32)
    t_grid_ten = torch.tensor(MHAS_WAVES, dtype=torch.float32).to(device)
    t_norm_ten = t_grid_ten / T_MAX

    records = []
    processed_keys = set()
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path, usecols=['cunicah', 'np'])
            processed_keys = set(zip(existing_df['cunicah'], existing_df['np']))
            print(f"Found {len(processed_keys)} already processed patients in {output_path}. Resuming...")
        except Exception as e:
            print(f"Could not read existing output file: {e}")

    try:
        with torch.no_grad():
            for s in tqdm(dataset.samples, desc="Extracting velocity"):
                cunicah = s['cunicah']
                np_val  = s['np']

                if (cunicah, np_val) in processed_keys:
                    continue

            x    = s['x'].unsqueeze(0).to(device)     # (1, N_WAVES, INPUT_DIM)
            u    = s['u'].unsqueeze(0).to(device)     # (1, N_WAVES, CONTROL_DIM)
            mask = s['mask'].unsqueeze(0).to(device)  # (1, N_WAVES)

            t_norm = t_norm_ten.unsqueeze(0)           # (1, N_WAVES)

            # Baseline control: first observed wave
            first_wi = mask[0].float().argmax().item()
            u0 = u[0, first_wi, :].unsqueeze(0)        # (1, 7)

            # Posterior distribution of z0
            mu, logvar = model.encode(x, t_norm, mask)  # (1, LATENT_DIM)
            std = (0.5 * logvar).exp()                   # (1, LATENT_DIM)

            # Dense time grid over patient's observed window
            obs_mask   = mask[0].cpu().numpy()
            t_obs      = waves_arr[obs_mask]
            t_dense    = np.arange(t_obs.min(), t_obs.max() + t_step, t_step)
            t_dense_ten = torch.tensor(t_dense, dtype=torch.float32).to(device)
            T_dense    = len(t_dense)

            # MC sampling: shape (n_mc, LATENT_DIM)
            eps      = torch.randn(n_mc, LATENT_DIM, device=device)
            z0_mc    = mu + eps * std              # (n_mc, LATENT_DIM)

            all_z = np.zeros((n_mc, T_dense, LATENT_DIM), dtype=np.float32)
            all_v = np.zeros((n_mc, T_dense, LATENT_DIM), dtype=np.float32)

            model.ode_func.current_u = u0          # broadcast over any batch dim

            for mc_i in range(n_mc):
                z0_i  = z0_mc[mc_i:mc_i+1]        # (1, LATENT_DIM)
                z_traj = odeint(model.ode_func, z0_i, t_dense_ten, method='rk4')
                # z_traj: (T_dense, 1, LATENT_DIM)
                z_traj = z_traj.squeeze(1)          # (T_dense, LATENT_DIM)

                # Velocity = ODE right-hand side at each time step
                v_traj = torch.stack([
                    model.ode_func(t_dense_ten[ti], z_traj[ti:ti+1]).squeeze(0)
                    for ti in range(T_dense)
                ])                                  # (T_dense, LATENT_DIM)

                all_z[mc_i] = z_traj.cpu().numpy()
                all_v[mc_i] = v_traj.cpu().numpy()

            z_mean = all_z.mean(0)                  # (T_dense, 8)
            z_std  = all_z.std(0)                   # (T_dense, 8)
            v_mean = all_v.mean(0)                  # (T_dense, 8)
            v_unc  = all_v.std(0).mean(-1)          # (T_dense,) — cross-dim velocity std

            for ti, t_val in enumerate(t_dense):
                row = {'cunicah': cunicah, 'np': np_val, 't': round(float(t_val), 3)}
                for k in range(LATENT_DIM):
                    row[f'z_mean_{k}'] = float(z_mean[ti, k])
                    row[f'z_std_{k}']  = float(z_std[ti, k])
                    row[f'v_{k}']      = float(v_mean[ti, k])
                row['v_uncertainty'] = float(v_unc[ti])
                records.append(row)

    except KeyboardInterrupt:
        print("\nInterrupted by user! Saving progress...")
    finally:
        if records:
            df_out = pd.DataFrame(records)
            write_header = not os.path.exists(output_path)
            df_out.to_csv(output_path, mode='a', index=False, header=write_header)
            n_patients = df_out[['cunicah', 'np']].drop_duplicates().shape[0]
            print(f"Saved {len(records):,} new rows across {n_patients:,} patients → {output_path}")
        else:
            print("No new records to save.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_mc',   type=int,   default=30,  help='MC samples for uncertainty')
    ap.add_argument('--t_step', type=float, default=0.5, help='Dense grid step in years')
    args = ap.parse_args()
    extract_velocity(n_mc=args.n_mc, t_step=args.t_step)
