"""
Latent ODE-VAE  (Chen et al., 2019 — "Latent ODEs for Irregularly-Sampled Time Series")

Replaces the β-VAE + GP + Neural ODE three-stage pipeline with a single end-to-end model:

  RecognitionRNN  →  z0 distribution  →  ODEFunc  →  z(t)  →  Decoder  →  x̂(t)

Velocity dz/dt = f_θ(z, u) is the model's native output — no GP step required.
Cox partial-likelihood loss (λ_cox=0.1) directly optimises the latent space for
survival discrimination, addressing the low HR in the previous pipeline.

Usage:
    python latent_velocity/engine/train_latent_ode.py
    python latent_velocity/engine/train_latent_ode.py --epochs 200 --lambda_cox 0.2
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
import pandas as pd
import numpy as np
from _paths import DATA_DIR, MODELS_DIR

# ─── Column Definitions ─────────────────────────────────────────────────────

DEFICIT_COLS = [
    'hipertension', 'diabetes', 'enf_pulm', 'artritis', 'infarto', 'embolia',
    'cancer', 'salud_glob',
    'n_abvd', 'n_aivd', 'n_mov', 'n_img', 'motoras_gruesas', 'motoras_finas',
    'deprimido', 'esfuerzo', 'intranquilo', 'triste', 'cansado', 'solo',
    'feliz', 'disf_vida', 'energia',
    'recuerdo1', 'recuerdo2', 'copiafiguras1', 'copiafiguras2',
    'orientacion', 'serial7', 'visualscan', 'memoria',
    'bmi_imp', 'hospitalizacion', 'visita_medica',
]
STATIC_COLS = ['edad', 'sexo', 'educacion']
U_COLS      = [
    'tabaco', 'bmi_imp', 'ejer_3_por_sem',
    'hipertension', 'diabetes', 'alcohol', 'social_isolation',
]

# MHAS survey wave years relative to 2001
MHAS_WAVES  = [0.0, 2.0, 11.0, 14.0, 17.0, 20.0]
N_WAVES     = len(MHAS_WAVES)
T_MAX       = 20.0           # used to normalise times → [0, 1] for encoder input

LATENT_DIM  = 8
CONTROL_DIM = len(U_COLS)   # 7
INPUT_DIM   = len(DEFICIT_COLS) + len(STATIC_COLS)  # 37


# ─── Architecture ───────────────────────────────────────────────────────────

class RecognitionRNN(nn.Module):
    """
    Backward GRU encoder for irregular-time clinical sequences.

    Processes MHAS observations in reverse chronological order using a GRU cell.
    Masked updates ensure unobserved survey waves don't corrupt the hidden state.
    Concatenates the (normalised) observation time to each input frame.
    """
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=64, latent_dim=LATENT_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell   = nn.GRUCell(input_dim + 1, hidden_dim)  # +1 for time
        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar  = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, t_norm, mask):
        # x:      (B, T, INPUT_DIM)
        # t_norm: (B, T)  — times in [0, 1]
        # mask:   (B, T)  — True where patient was observed
        B = x.size(0)
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        for i in range(x.size(1) - 1, -1, -1):
            inp   = torch.cat([x[:, i, :], t_norm[:, i:i+1]], dim=-1)
            h_new = self.gru_cell(inp, h)
            upd   = mask[:, i].float().unsqueeze(-1)   # (B, 1)
            h     = h_new * upd + h * (1.0 - upd)     # skip unobserved
        return self.fc_mu(h), self.fc_logvar(h)


class LatentODEFunc(nn.Module):
    """
    dz/dt = f_θ(z, u).

    Interface-compatible with ODEFunc in train_ode.py so digital_twin.py
    can use washout simulation unchanged.
    """
    def __init__(self, latent_dim=LATENT_DIM, control_dim=CONTROL_DIM, hidden_dim=128):
        super().__init__()
        self.control_dim = control_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + control_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.current_u = None
        self.target_u  = None
        self.washout_k = 0.0
        self.nfe       = 0

    def forward(self, t, z):
        self.nfe += 1
        if self.current_u is None:
            u = torch.zeros(*z.shape[:-1], self.control_dim, device=z.device)
        elif self.target_u is not None and self.washout_k > 0:
            u = self.target_u + (self.current_u - self.target_u) * torch.exp(-self.washout_k * t)
        else:
            u = self.current_u
        return self.net(torch.cat([z, u * 10.0], dim=-1))


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, output_dim=len(DEFICIT_COLS), hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class RiskHead(nn.Module):
    """
    Predicts scalar mortality risk from the posterior mean μ.

    Decouples survival discrimination from latent geometry: Cox loss flows
    through this head, leaving μ free to encode clinical state faithfully.
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, mu):
        return self.net(mu).squeeze(-1)   # (B,)


class LatentODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = RecognitionRNN()
        self.ode_func  = LatentODEFunc()
        self.decoder   = Decoder()
        self.risk_head = RiskHead()

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def encode(self, x, t_norm, mask):
        """Returns (mu, logvar) of shape (B, LATENT_DIM)."""
        return self.encoder(x, t_norm, mask)

    def forward(self, x, t_norm, mask, t_grid, u0):
        """
        x:      (B, N_WAVES, INPUT_DIM)
        t_norm: (B, N_WAVES)
        mask:   (B, N_WAVES) bool
        t_grid: (N_WAVES,)   integration time points
        u0:     (B, CONTROL_DIM) baseline control

        Returns: x_hat (B, N_WAVES, N_DEFICITS), mu, logvar, z_traj (N_WAVES, B, 8)
        """
        mu, logvar = self.encoder(x, t_norm, mask)
        z0 = self.reparameterize(mu, logvar)

        self.ode_func.current_u = u0
        # Fully batched ODE solve: (N_WAVES, B, LATENT_DIM)
        z_traj = odeint(self.ode_func, z0, t_grid, method='rk4')

        T, B = z_traj.shape[:2]
        x_hat = self.decoder(z_traj.reshape(T * B, LATENT_DIM)).reshape(T, B, -1)
        return x_hat.permute(1, 0, 2), mu, logvar, z_traj  # x_hat: (B, T, N_DEFICITS)


# ─── Dataset ────────────────────────────────────────────────────────────────

class LatentODEDataset(Dataset):
    """
    Converts frailty_index_data.csv into fixed-grid patient sequences.

    Each patient occupies one or more rows in the CSV (one per wave).
    Observations are snapped to the nearest MHAS wave (within ±1.5 yr).
    Missing waves are zero-padded; mask=False marks padded positions.
    """
    def __init__(self, csv_path, wave_tolerance=1.5,
                 edad_mean=None, edad_std=None,
                 edu_mean=None,  edu_std=None):
        df = pd.read_csv(csv_path)
        df['t'] = df['a_o_ent'] - 2001

        if 'social_isolation' not in df.columns:
            df['social_isolation'] = (
                1.0 - df[['asiste_club', 'voluntario']].fillna(0).max(axis=1)
            )

        df = df.dropna(subset=['FI']).copy()

        # Normalise static features; store stats for inference
        if edad_mean is None:
            self.edad_mean = float(df['edad'].mean())
            self.edad_std  = float(df['edad'].std())
            self.edu_mean  = float(df['educacion'].mean())
            self.edu_std   = float(df['educacion'].std())
        else:
            self.edad_mean, self.edad_std = edad_mean, edad_std
            self.edu_mean,  self.edu_std  = edu_mean,  edu_std

        df['edad']      = (df['edad']      - self.edad_mean) / self.edad_std
        df['educacion'] = (df['educacion'] - self.edu_mean)  / self.edu_std
        df['sexo']      = df['sexo'] - 1.0

        # Fill NaN deficits with column medians
        for col in DEFICIT_COLS + U_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median()).fillna(0.0)

        # Patient-level survival outcomes
        surv = df.groupby(['cunicah', 'np']).agg(
            event=('fallecido', 'max'),
            first_year=('a_o_ent', 'min'),
            last_year=('a_o_ent', 'max'),
        ).reset_index()
        surv['tte'] = (surv['last_year'] - surv['first_year']).clip(lower=0.0)
        surv_map = surv.set_index(['cunicah', 'np'])[['tte', 'event']]

        waves_arr = np.array(MHAS_WAVES, dtype=np.float32)
        n_def     = len(DEFICIT_COLS)
        n_static  = len(STATIC_COLS)
        n_u       = len(U_COLS)

        self.samples = []

        for (cunicah, np_val), grp in df.groupby(['cunicah', 'np']):
            grp = grp.sort_values('t')

            x_grid = np.zeros((N_WAVES, n_def + n_static), dtype=np.float32)
            u_grid = np.zeros((N_WAVES, n_u),               dtype=np.float32)
            mask   = np.zeros(N_WAVES,                       dtype=bool)

            for _, row in grp.iterrows():
                t_val = row.get('t', np.nan)
                if pd.isna(t_val):
                    continue
                dists = np.abs(waves_arr - t_val)
                wi    = int(np.argmin(dists))
                if dists[wi] > wave_tolerance:
                    continue

                x_row = np.array([row.get(c, 0.0) for c in DEFICIT_COLS], dtype=np.float32)
                s_row = np.array([row.get(c, 0.0) for c in STATIC_COLS],  dtype=np.float32)
                u_row = np.array([row.get(c, 0.0) for c in U_COLS],       dtype=np.float32)

                x_grid[wi] = np.concatenate([x_row, s_row])
                u_grid[wi] = u_row
                mask[wi]   = True

            if mask.sum() < 2:
                continue

            try:
                tte_val = float(surv_map.loc[(cunicah, np_val), 'tte'])
                ev_val  = float(surv_map.loc[(cunicah, np_val), 'event'])
            except KeyError:
                tte_val, ev_val = 0.0, 0.0

            self.samples.append({
                'cunicah': cunicah,
                'np':      np_val,
                'x':       torch.tensor(x_grid, dtype=torch.float32),
                'u':       torch.tensor(u_grid, dtype=torch.float32),
                'mask':    torch.tensor(mask,   dtype=torch.bool),
                'tte':     torch.tensor(tte_val, dtype=torch.float32),
                'ev':      torch.tensor(ev_val,  dtype=torch.float32),
            })

        print(f"LatentODEDataset: {len(self.samples)} patients with ≥2 observations.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {k: v for k, v in s.items() if k not in ('cunicah', 'np')}


# ─── Loss Functions ─────────────────────────────────────────────────────────

def elbo_loss(x_hat, x_true, mask, mu, logvar, feat_weights, beta, free_bits=0.5):
    """
    ELBO = weighted reconstruction MSE (observed frames only) + β·KLD with free bits.

    x_hat, x_true: (B, T, N_DEFICITS)
    mask:          (B, T)
    feat_weights:  (N_DEFICITS,)  inverse-variance weights
    """
    mse_w    = ((x_hat - x_true) ** 2) * feat_weights           # (B, T, D)
    mse_obs  = mse_w.sum(-1) * mask.float()                     # (B, T)
    recon    = mse_obs.sum() / (mask.float().sum() + 1e-8)

    kld_dim  = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # (B, D)
    kld_dim  = kld_dim.mean(0)                                   # (D,)
    kld_dim  = torch.max(kld_dim, kld_dim.new_full(kld_dim.shape, free_bits))
    kld      = kld_dim.sum()

    return recon + beta * kld, recon, kld


def cox_partial_loss(risk_scores, tte, ev):
    """
    Differentiable Cox partial log-likelihood.
    risk_scores: (B,) scalar output of RiskHead — decoupled from latent geometry.
    """
    risk         = risk_scores                     # (B,)
    order        = torch.argsort(tte, descending=True)
    risk_sorted  = risk[order]
    ev_sorted    = ev[order]
    log_cumsum   = torch.logcumsumexp(risk_sorted, dim=0)
    nll          = -(risk_sorted - log_cumsum) * ev_sorted
    n_ev         = ev_sorted.sum()
    return nll.sum() / (n_ev + 1e-8)


# ─── Training ───────────────────────────────────────────────────────────────

def train(csv_path=None, model_path=None, epochs=150, lr=3e-4,
          batch_size=256, target_beta=0.1, lambda_cox=0.15):
    if csv_path   is None: csv_path   = str(DATA_DIR   / 'frailty_index_data.csv')
    if model_path is None: model_path = str(MODELS_DIR / 'latent_ode_model.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Latent ODE-VAE | Device: {device} | Epochs: {epochs} | "
          f"β={target_beta} | λ_cox={lambda_cox}")

    dataset = LatentODEDataset(csv_path)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0, drop_last=True)

    # Inverse-variance feature weights for reconstruction loss
    df_raw = pd.read_csv(csv_path).dropna(subset=['FI'])
    for col in DEFICIT_COLS:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].fillna(df_raw[col].median()).fillna(0.0)
    feat_var     = np.clip(df_raw[DEFICIT_COLS].values.astype(np.float32).var(0), 1e-5, None)
    inv_var      = 1.0 / feat_var
    feat_weights = torch.tensor(inv_var * len(inv_var) / inv_var.sum(),
                                dtype=torch.float32).to(device)

    t_grid      = torch.tensor(MHAS_WAVES, dtype=torch.float32).to(device)
    t_norm_base = t_grid / T_MAX   # (N_WAVES,) in [0, 1]

    model     = LatentODE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    anneal_start, anneal_end = 20, 80

    history = {'recon': [], 'kld': [], 'cox': [], 'loss': []}

    for epoch in range(epochs):
        if epoch < anneal_start:
            beta = 0.0
        elif epoch < anneal_end:
            beta = target_beta * (epoch - anneal_start) / (anneal_end - anneal_start)
        else:
            beta = target_beta

        model.train()
        model.ode_func.nfe = 0
        tot_loss = tot_recon = tot_kld = tot_cox = 0.0
        n_b = 0

        for batch in loader:
            x    = batch['x'].to(device)     # (B, N_WAVES, INPUT_DIM)
            u    = batch['u'].to(device)     # (B, N_WAVES, CONTROL_DIM)
            mask = batch['mask'].to(device)  # (B, N_WAVES)
            tte  = batch['tte'].to(device)   # (B,)
            ev   = batch['ev'].to(device)    # (B,)

            B = x.size(0)
            t_norm = t_norm_base.unsqueeze(0).expand(B, -1)  # (B, N_WAVES)

            # First observed wave's control vector as ODE baseline
            first_wi = mask.float().argmax(dim=1)            # (B,)
            u0 = u[torch.arange(B), first_wi, :]             # (B, 7)

            x_deficits = x[:, :, :len(DEFICIT_COLS)]         # (B, T, 34)

            optimizer.zero_grad()
            x_hat, mu, logvar, _ = model(x, t_norm, mask, t_grid, u0)

            loss_elbo, recon, kld = elbo_loss(
                x_hat, x_deficits, mask, mu, logvar, feat_weights, beta)

            loss_cox = torch.tensor(0.0, device=device)
            if lambda_cox > 0.0 and ev.sum() > 0:
                risk_scores = model.risk_head(mu)
                loss_cox = cox_partial_loss(risk_scores, tte, ev)

            loss = loss_elbo + lambda_cox * loss_cox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot_loss  += loss.item()
            tot_recon += recon.item()
            tot_kld   += kld.item()
            tot_cox   += loss_cox.item()
            n_b       += 1

        avg_recon = tot_recon / n_b
        avg_kld   = tot_kld   / n_b
        avg_cox   = tot_cox   / n_b
        avg_loss  = tot_loss  / n_b
        scheduler.step(avg_recon)

        history['recon'].append(avg_recon)
        history['kld'].append(avg_kld)
        history['cox'].append(avg_cox)
        history['loss'].append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[{epoch+1:3d}/{epochs}] β={beta:.3f} "
                  f"| Loss={avg_loss:.4f} "
                  f"| Recon={avg_recon:.4f} "
                  f"| KLD={avg_kld:.4f} "
                  f"| Cox={avg_cox:.4f} "
                  f"| NFE={model.ode_func.nfe} "
                  f"| LR={optimizer.param_groups[0]['lr']:.1e}")

    ckpt = {
        'model_state': model.state_dict(),
        'edad_mean':   dataset.edad_mean,
        'edad_std':    dataset.edad_std,
        'edu_mean':    dataset.edu_mean,
        'edu_std':     dataset.edu_std,
        'history':     history,
        'hparams':     {'epochs': epochs, 'beta': target_beta,
                        'lambda_cox': lambda_cox, 'lr': lr},
    }
    torch.save(ckpt, model_path)
    print(f"\nLatent ODE-VAE saved → {model_path}")
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs',      type=int,   default=150)
    ap.add_argument('--lr',          type=float, default=3e-4)
    ap.add_argument('--batch_size',  type=int,   default=256)
    ap.add_argument('--lambda_cox',  type=float, default=0.1,
                    help='Cox partial-likelihood weight. 0 to disable.')
    args = ap.parse_args()
    train(epochs=args.epochs, lr=args.lr,
          batch_size=args.batch_size, lambda_cox=args.lambda_cox)
