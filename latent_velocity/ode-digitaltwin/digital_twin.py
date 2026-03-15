import sys
from pathlib import Path

_ODE = Path(__file__).resolve().parent
_ROOT = _ODE.parent
sys.path.insert(0, str(_ODE))
sys.path.insert(0, str(_ROOT / "engine"))

import torch
import pandas as pd
import numpy as np
from torchdiffeq import odeint
from _paths import DATA_DIR, MODELS_DIR
from train_vae import BetaVAE, FrailtyDataset
from train_ode import ODEFunc


# ── Canonical 7D Control Vector ──────────────────────────────────────
U_COLS = ['tabaco', 'bmi_imp', 'ejer_3_por_sem', 'hipertension',
          'diabetes', 'alcohol', 'social_isolation']

# What "cured" means for each variable (1=bad/deficit, 0=good/healthy)
# Exception: ejer_3_por_sem where 1=good (exercises), 0=bad (sedentary)
TARGET_MAP = {
    'tabaco':          0.0,  # Quit smoking
    'bmi_imp':         0.0,  # Normalize BMI
    'ejer_3_por_sem':  1.0,  # Start exercising
    'hipertension':    0.0,  # Control blood pressure
    'diabetes':        0.0,  # Manage diabetes
    'alcohol':         0.0,  # Reduce alcohol
    'social_isolation':0.0,  # Increase social engagement
}

LABEL_MAP = {
    'tabaco':           'Quit Smoking',
    'bmi_imp':          'Normalize BMI',
    'ejer_3_por_sem':   'Add Exercise',
    'hipertension':     'Control Hypertension',
    'diabetes':         'Manage Diabetes',
    'alcohol':          'Reduce Alcohol',
    'social_isolation': 'Social Engagement',
}


def _extract_patient_u(latest_visit):
    """Extract 7D control vector from a patient row, deriving social_isolation."""
    u_raw = {}
    for col in U_COLS:
        if col == 'social_isolation':
            club = latest_visit.get('asiste_club', 0.0)
            vol  = latest_visit.get('voluntario', 0.0)
            club = 0.0 if pd.isna(club) else float(club)
            vol  = 0.0 if pd.isna(vol)  else float(vol)
            u_raw[col] = 1.0 - max(club, vol)
        else:
            val = latest_visit.get(col, 0.0)
            u_raw[col] = 0.0 if pd.isna(val) else float(val)
    return u_raw


def load_models(device):
    vae = BetaVAE(latent_dim=8).to(device)
    vae.load_state_dict(torch.load(MODELS_DIR / 'beta_vae_model.pth',
                                   map_location=device, weights_only=True))
    vae.eval()

    ode_func = ODEFunc(control_dim=7).to(device)
    ode_func.load_state_dict(torch.load(MODELS_DIR / 'neural_ode_high_momentum.pth',
                                        map_location=device, weights_only=True))
    ode_func.eval()
    return vae, ode_func


# ── Single Twin Simulation ───────────────────────────────────────────

def _simulate_single_twin(ode_func, z0, u_baseline, u_twin, t_span, washout_k=2.0):
    """Run one twin simulation with biological washout; returns v_mag array."""
    ode_func.current_u = u_baseline
    ode_func.target_u  = u_twin
    ode_func.washout_k = washout_k

    with torch.no_grad():
        z_traj = odeint(ode_func, z0, t_span, method='rk4')
        v_mag = []
        for i, t in enumerate(t_span):
            v_t = ode_func(t, z_traj[i])
            v_mag.append(torch.norm(v_t, dim=-1).item())
    return np.array(v_mag)


# ── Ghost Twin Guardrail (Mahalanobis) ───────────────────────────────

def _compute_cohort_mahalanobis(z_now_np, u_twin_np, df_raw, deficit_cols, static_cols, vae, device):
    """
    Compute the Mahalanobis distance of (z_now, u_twin) from the empirical
    distribution of patients who already have the proposed control profile.
    
    Returns (distance, n_matching) where n_matching is the count of patients
    with control profile within 0.5 tolerance on each variable.
    """
    # Find patients with a similar control profile (tolerance for continuous vars)
    mask = pd.Series(True, index=df_raw.index)
    for i, col in enumerate(U_COLS):
        target = u_twin_np[i]
        if col == 'social_isolation':
            # Derive on the fly
            club = df_raw.get('asiste_club', pd.Series(0.0, index=df_raw.index)).fillna(0.0)
            vol  = df_raw.get('voluntario', pd.Series(0.0, index=df_raw.index)).fillna(0.0)
            col_vals = 1.0 - np.maximum(club.values, vol.values)
        else:
            col_vals = df_raw[col].fillna(0.0).values
        mask = mask & (np.abs(col_vals - target) < 0.5)

    n_matching = mask.sum()
    
    if n_matching < 5:
        # Not enough data to compute Mahalanobis — flag immediately
        return float('inf'), int(n_matching)

    # Encode the matching cohort through VAE to get their Z distribution
    matching = df_raw.loc[mask]
    x_def = torch.tensor(matching[deficit_cols].values, dtype=torch.float32).to(device)
    x_sta = torch.tensor(matching[static_cols].values, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        z_cohort, _ = vae.encode(x_def, x_sta)
    z_cohort = z_cohort.cpu().numpy()
    
    # Mahalanobis distance from the cohort centroid
    mu = z_cohort.mean(axis=0)
    diff = z_now_np - mu
    
    if n_matching < 15:
        # Too few samples for a reliable covariance → use diagonal (variance only)
        var = z_cohort.var(axis=0) + 1e-6
        maha = np.sqrt(np.sum(diff**2 / var))
    else:
        cov = np.cov(z_cohort.T) + np.eye(z_cohort.shape[1]) * 1e-6
        cov_inv = np.linalg.inv(cov)
        maha = np.sqrt(diff @ cov_inv @ diff)

    return float(maha), int(n_matching)


# ── Original Intervention Runner (backward compat) ───────────────────

def run_digital_twin_intervention(cunicah, np_val, years=5.0,
                                   intervention_targets=None):
    if intervention_targets is None:
        intervention_targets = list(U_COLS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, ode_func = load_models(device)

    dataset = FrailtyDataset(DATA_DIR / 'frailty_index_data.csv', device=device)
    df_raw = dataset.data

    p_data = df_raw[(df_raw['cunicah'] == cunicah) & (df_raw['np'] == np_val)]
    if p_data.empty:
        print(f"Patient {cunicah}/{np_val} not found.")
        return None

    latest_visit = p_data.sort_values(by='a_o_ent').iloc[-1]

    deficit_cols = dataset.deficit_cols
    static_cols  = dataset.static_cols
    x_def = torch.tensor(latest_visit[deficit_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
    x_sta = torch.tensor(latest_visit[static_cols].values, dtype=torch.float32).unsqueeze(0).to(device)

    u_dict = _extract_patient_u(latest_visit)
    u_baseline = torch.tensor([[u_dict[c] for c in U_COLS]], dtype=torch.float32).to(device)

    # Build twin u
    u_twin = u_baseline.clone()
    for target in intervention_targets:
        if target in U_COLS:
            idx = U_COLS.index(target)
            current_val = u_baseline[0, idx].item()
            new_val = TARGET_MAP[target]
            if current_val == new_val:
                print(f" [Warning] Already at target for {target} ({current_val}).")
            else:
                print(f" [Intervention] Flipping {target}: {current_val} -> {new_val}")
                u_twin[0, idx] = new_val

    with torch.no_grad():
        z0, _ = vae.encode(x_def, x_sta)
        t_span = torch.linspace(0, years, 50).to(device)

    # Baseline
    ode_func.current_u = u_baseline
    ode_func.target_u  = None
    ode_func.washout_k = 0.0
    with torch.no_grad():
        z_traj_base = odeint(ode_func, z0, t_span, method='rk4')

    # Twin (washout)
    v_mag_twin = _simulate_single_twin(ode_func, z0, u_baseline, u_twin, t_span, 2.0)

    # Baseline velocities
    v_mag_base = []
    ode_func.current_u = u_baseline
    ode_func.target_u  = None
    ode_func.washout_k = 0.0
    with torch.no_grad():
        for i, t in enumerate(t_span):
            v_t = ode_func(t, z_traj_base[i])
            v_mag_base.append(torch.norm(v_t, dim=-1).item())

    return {
        't': t_span.cpu().numpy(),
        'v_mag_baseline': np.array(v_mag_base),
        'v_mag_twin': v_mag_twin,
        'z_baseline': z_traj_base.squeeze().detach().cpu().numpy(),
    }


# ── Automated Intervention Ranking Engine ─────────────────────────────

def rank_interventions(cunicah, np_val, years=5.0, washout_k=2.0):
    """
    Generates all meaningful intervention scenarios, simulates each as a
    Digital Twin with biological washout, computes Velocity AUC, applies
    Ghost Twin guardrail (Mahalanobis), and returns a ranked list.
    """
    from itertools import combinations

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, ode_func = load_models(device)

    dataset = FrailtyDataset(DATA_DIR / 'frailty_index_data.csv', device=device)
    df_raw = dataset.data

    p_data = df_raw[(df_raw['cunicah'] == cunicah) & (df_raw['np'] == np_val)]
    if p_data.empty:
        print(f"Patient {cunicah}/{np_val} not found.")
        return None

    latest_visit = p_data.sort_values(by='a_o_ent').iloc[-1]

    deficit_cols = dataset.deficit_cols
    static_cols  = dataset.static_cols
    x_def = torch.tensor(latest_visit[deficit_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
    x_sta = torch.tensor(latest_visit[static_cols].values, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        z0, _ = vae.encode(x_def, x_sta)
    z0_np = z0.squeeze().cpu().numpy()

    u_dict = _extract_patient_u(latest_visit)
    u_baseline = torch.tensor([[u_dict[c] for c in U_COLS]], dtype=torch.float32).to(device)

    # ── Phase 8: Actionability Filter ──
    actionable = []
    for col in U_COLS:
        idx = U_COLS.index(col)
        current = u_baseline[0, idx].item()
        target  = TARGET_MAP[col]
        if current != target:
            actionable.append(col)

    if not actionable:
        print(f"Patient {int(cunicah)}/{int(np_val)} already meets all targets.")
        return None

    print(f"\nActionable for Patient {int(cunicah)}/{int(np_val)}: "
          f"{[LABEL_MAP[a] for a in actionable]}")

    # ── Phase 9: Combinatorial Constraint ──
    N = len(actionable)
    max_r = N if N <= 3 else 2  # singles+pairs only when N>3
    scenarios = []
    for r in range(1, max_r + 1):
        for combo in combinations(actionable, r):
            scenarios.append(list(combo))

    print(f"  Generating {len(scenarios)} intervention scenarios "
          f"(max combo size = {max_r})...")

    t_span = torch.linspace(0, years, 50).to(device)
    t_np   = t_span.cpu().numpy()

    # ── Baseline AUC ──
    ode_func.current_u = u_baseline
    ode_func.target_u  = None
    ode_func.washout_k = 0.0
    with torch.no_grad():
        z_traj_base = odeint(ode_func, z0, t_span, method='rk4')
        v_mag_base = []
        for i, t in enumerate(t_span):
            v_t = ode_func(t, z_traj_base[i])
            v_mag_base.append(torch.norm(v_t, dim=-1).item())
    v_mag_base = np.array(v_mag_base)
    auc_baseline = np.trapz(v_mag_base, t_np)

    # ── Simulate each scenario ──
    results = []
    for scenario in scenarios:
        # Build twin u
        u_twin = u_baseline.clone()
        for col in scenario:
            idx = U_COLS.index(col)
            u_twin[0, idx] = TARGET_MAP[col]

        label = " + ".join([LABEL_MAP[c] for c in scenario])

        # Phase 10: Ghost Twin Guardrail (Mahalanobis)
        u_twin_np = u_twin.squeeze().cpu().numpy()
        maha_dist, n_match = _compute_cohort_mahalanobis(
            z0_np, u_twin_np, df_raw, deficit_cols, static_cols, vae, device
        )
        confidence = "High" if maha_dist <= 3.0 else "Low (OOD)"

        # Simulate
        v_mag_twin = _simulate_single_twin(ode_func, z0, u_baseline, u_twin,
                                            t_span, washout_k)
        auc_twin = np.trapz(v_mag_twin, t_np)
        reduction_pct = (auc_baseline - auc_twin) / auc_baseline * 100

        results.append({
            'label': label,
            'targets': scenario,
            'v_mag': v_mag_twin,
            'auc': auc_twin,
            'auc_reduction_pct': reduction_pct,
            'mahalanobis': maha_dist,
            'n_cohort_match': n_match,
            'confidence': confidence,
        })

    results.sort(key=lambda x: x['auc_reduction_pct'], reverse=True)

    # Clean up
    ode_func.current_u = None
    ode_func.target_u  = None
    ode_func.washout_k = 0.0

    return {
        't': t_np,
        'v_mag_baseline': v_mag_base,
        'auc_baseline': auc_baseline,
        'ranked_interventions': results,
        'patient_id': f"{int(cunicah)}/{int(np_val)}",
    }


if __name__ == "__main__":
    ranking = rank_interventions(cunicah=7226.0, np_val=10.0)
    if ranking:
        print(f"\n{'='*80}")
        print(f" INTERVENTION RANKING — Patient {ranking['patient_id']}")
        print(f" Baseline 5-Year Velocity AUC: {ranking['auc_baseline']:.3f}")
        print(f"{'='*80}")
        print(f"  {'#':<4} {'Intervention':<35} {'AUC':>7} {'Δ%':>8}  {'Conf':>12}  {'Maha':>5}  {'n':>5}")
        print(f"  {'-'*4} {'-'*35} {'-'*7} {'-'*8}  {'-'*12}  {'-'*5}  {'-'*5}")
        for i, r in enumerate(ranking['ranked_interventions'], 1):
            print(f"  {i:<4} {r['label']:<35} {r['auc']:>7.3f} {r['auc_reduction_pct']:>+7.1f}%"
                  f"  {r['confidence']:>12s}  {r['mahalanobis']:>5.1f}  {r['n_cohort_match']:>5d}")
        print(f"{'='*80}")
        best = ranking['ranked_interventions'][0]
        print(f"\n  >>> PRIMARY CLINICAL TARGET: {best['label']}")
        print(f"      Velocity Reduction: {best['auc_reduction_pct']:+.1f}% | "
              f"Confidence: {best['confidence']} (Mahalanobis: {best['mahalanobis']:.2f})")
