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


# ── Canonical 7D Control Vector ──────────────────────────────────────
U_COLS = ['tabaco', 'bmi_imp', 'ejer_3_por_sem', 'hipertension',
          'diabetes', 'alcohol', 'social_isolation']

TARGET_MAP = {
    'tabaco':          0.0,
    'bmi_imp':         0.0,
    'ejer_3_por_sem':  0.0,
    'hipertension':    0.0,
    'diabetes':        0.0,
    'alcohol':         0.0,
    'social_isolation':0.0,
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


# ── Model Loading ────────────────────────────────────────────────────

def load_models(device):
    """Load the Latent ODE-VAE model. Returns (model, ode_func)."""
    from train_latent_ode import LatentODE
    ckpt  = torch.load(MODELS_DIR / 'latent_ode_model.pth',
                       map_location=device, weights_only=False)
    model = LatentODE().to(device)
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval()
    model._ckpt = ckpt
    return model, model.ode_func


# ── Patient Encoding ─────────────────────────────────────────────────

def _build_latent_ode_z0(model, ckpt, p_data, device):
    """
    Encode a patient's full observation sequence → z0 posterior mean.
    All available visits are used for the best estimate.
    """
    from train_latent_ode import (
        DEFICIT_COLS, STATIC_COLS, U_COLS as _U_COLS,
        MHAS_WAVES, N_WAVES, T_MAX,
    )

    waves_arr = np.array(MHAS_WAVES, dtype=np.float32)
    n_def     = len(DEFICIT_COLS)
    n_static  = len(STATIC_COLS)

    p_data = p_data.copy()
    p_data['edad']      = (p_data['edad']      - ckpt['edad_mean']) / ckpt['edad_std']
    p_data['educacion'] = (p_data['educacion'] - ckpt['edu_mean'])  / ckpt['edu_std']
    p_data['sexo']      = p_data['sexo'] - 1.0
    if 'social_isolation' not in p_data.columns:
        p_data['social_isolation'] = (
            1.0 - p_data[['asiste_club', 'voluntario']].fillna(0).max(axis=1)
        )
    p_data = p_data.sort_values('a_o_ent')
    p_data['t'] = p_data['a_o_ent'] - 2001

    x_grid = np.zeros((N_WAVES, n_def + n_static), dtype=np.float32)
    u_grid = np.zeros((N_WAVES, len(_U_COLS)), dtype=np.float32)
    mask   = np.zeros(N_WAVES, dtype=bool)

    for _, row in p_data.iterrows():
        t_val = row.get('t', np.nan)
        if pd.isna(t_val):
            continue
        dists = np.abs(waves_arr - t_val)
        wi    = int(np.argmin(dists))
        if dists[wi] > 1.5:
            continue
        x_row = np.array([row.get(c, 0.0) for c in DEFICIT_COLS], dtype=np.float32)
        s_row = np.array([row.get(c, 0.0) for c in STATIC_COLS],  dtype=np.float32)
        u_row = np.array([row.get(c, 0.0) for c in _U_COLS],      dtype=np.float32)
        np.nan_to_num(x_row, copy=False)
        np.nan_to_num(u_row, copy=False)
        x_grid[wi] = np.concatenate([x_row, s_row])
        u_grid[wi] = u_row
        mask[wi]   = True

    if not mask.any():
        row = p_data.iloc[-1]
        x_row = np.array([row.get(c, 0.0) for c in DEFICIT_COLS], dtype=np.float32)
        s_row = np.array([row.get(c, 0.0) for c in STATIC_COLS],  dtype=np.float32)
        x_grid[0] = np.concatenate([x_row, s_row])
        mask[0]   = True

    x_t    = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(0).to(device)
    mask_t = torch.tensor(mask,   dtype=torch.bool).unsqueeze(0).to(device)
    t_norm = torch.tensor(
        [w / T_MAX for w in MHAS_WAVES], dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        mu, _ = model.encode(x_t, t_norm, mask_t)
    return mu   # (1, 8)


def _build_single_visit_z0(model, ckpt, patient_dict, device):
    """
    Encode a single-visit patient dict (e.g. from live inference form) → z0.
    Wraps the visit in a length-1 sequence on the first MHAS wave slot.
    """
    from train_latent_ode import DEFICIT_COLS, STATIC_COLS, MHAS_WAVES, T_MAX, N_WAVES

    x_row  = np.array([float(patient_dict.get(c, 0.0)) for c in DEFICIT_COLS], dtype=np.float32)
    edad_n = (float(patient_dict.get('edad', 65.0)) - ckpt['edad_mean']) / ckpt['edad_std']
    edu_n  = (float(patient_dict.get('educacion', 6.0)) - ckpt['edu_mean']) / ckpt['edu_std']
    sexo_n = float(patient_dict.get('sexo', 1.0)) - 1.0
    s_row  = np.array([edad_n, sexo_n, edu_n], dtype=np.float32)

    n_def    = len(DEFICIT_COLS)
    n_static = len(STATIC_COLS)
    x_grid   = np.zeros((N_WAVES, n_def + n_static), dtype=np.float32)
    mask     = np.zeros(N_WAVES, dtype=bool)
    x_grid[0] = np.concatenate([x_row, s_row])
    mask[0]   = True

    x_t    = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(0).to(device)
    mask_t = torch.tensor(mask,   dtype=torch.bool).unsqueeze(0).to(device)
    t_norm = torch.tensor(
        [w / T_MAX for w in MHAS_WAVES], dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        mu, _ = model.encode(x_t, t_norm, mask_t)
    return mu   # (1, 8)


# ── ODE State Management ─────────────────────────────────────────────

def _configure_ode(ode_func, current_u, target_u=None, washout_k=0.0):
    ode_func.current_u = current_u
    ode_func.target_u  = target_u
    ode_func.washout_k = washout_k


# ── Single Twin Simulation ───────────────────────────────────────────

def _simulate_single_twin(ode_func, z0, u_baseline, u_twin, t_span, washout_k=2.0):
    """Run one twin simulation with biological washout; returns v_mag array."""
    _configure_ode(ode_func, u_baseline, target_u=u_twin, washout_k=washout_k)
    with torch.no_grad():
        z_traj = odeint(ode_func, z0, t_span, method='rk4')
        v_mag  = [torch.norm(ode_func(t, z_traj[i]), dim=-1).item()
                  for i, t in enumerate(t_span)]
    return np.array(v_mag)


# ── Ghost Twin Guardrail (Mahalanobis) ───────────────────────────────

def _compute_cohort_mahalanobis(z_now_np, u_twin_np, df_raw_csv, model, ckpt, device):
    """
    Mahalanobis distance of z_now from the sub-cohort matching u_twin.
    Encodes cohort patients with the Latent ODE encoder (single-visit path).
    """
    from train_latent_ode import DEFICIT_COLS, STATIC_COLS, MHAS_WAVES, T_MAX, N_WAVES

    col_mask = pd.Series(True, index=df_raw_csv.index)
    for i, col in enumerate(U_COLS):
        target = u_twin_np[i]
        if col == 'social_isolation':
            club     = df_raw_csv.get('asiste_club', pd.Series(0.0, index=df_raw_csv.index)).fillna(0.0)
            vol      = df_raw_csv.get('voluntario',  pd.Series(0.0, index=df_raw_csv.index)).fillna(0.0)
            col_vals = 1.0 - np.maximum(club.values, vol.values)
        else:
            col_vals = df_raw_csv[col].fillna(0.0).values if col in df_raw_csv.columns else np.zeros(len(df_raw_csv))
        col_mask = col_mask & (np.abs(col_vals - target) < 0.5)

    n_matching = col_mask.sum()
    if n_matching < 5:
        return float('inf'), int(n_matching)

    matching = df_raw_csv.loc[col_mask].copy()
    matching['edad']      = (matching['edad']      - ckpt['edad_mean']) / ckpt['edad_std']
    matching['educacion'] = (matching['educacion'] - ckpt['edu_mean'])  / ckpt['edu_std']
    matching['sexo']      = matching['sexo'] - 1.0
    for col in DEFICIT_COLS:
        if col in matching.columns:
            matching[col] = matching[col].fillna(matching[col].median()).fillna(0.0)

    n      = len(matching)
    n_def  = len(DEFICIT_COLS)
    n_sta  = len(STATIC_COLS)
    x_grid = np.zeros((n, N_WAVES, n_def + n_sta), dtype=np.float32)
    for j, (_, row) in enumerate(matching.iterrows()):
        x_row = np.array([row.get(c, 0.0) for c in DEFICIT_COLS], dtype=np.float32)
        s_row = np.array([row.get(c, 0.0) for c in STATIC_COLS],  dtype=np.float32)
        np.nan_to_num(x_row, copy=False)
        np.nan_to_num(s_row, copy=False)
        x_grid[j, 0] = np.concatenate([x_row, s_row])

    mask_seq = np.zeros((n, N_WAVES), dtype=bool)
    mask_seq[:, 0] = True

    x_t    = torch.tensor(x_grid,    dtype=torch.float32).to(device)
    mask_t = torch.tensor(mask_seq,  dtype=torch.bool).to(device)
    t_norm = torch.tensor([w / T_MAX for w in MHAS_WAVES],
                          dtype=torch.float32, device=device).unsqueeze(0).expand(n, -1)

    with torch.no_grad():
        z_cohort, _ = model.encode(x_t, t_norm, mask_t)
    z_cohort = z_cohort.cpu().numpy()

    mu   = z_cohort.mean(axis=0)
    diff = z_now_np - mu

    if n_matching < 15:
        var  = z_cohort.var(axis=0) + 1e-6
        maha = np.sqrt(np.sum(diff**2 / var))
    else:
        cov     = np.cov(z_cohort.T) + np.eye(z_cohort.shape[1]) * 1e-6
        cov_inv = np.linalg.inv(cov)
        maha    = np.sqrt(diff @ cov_inv @ diff)

    return float(maha), int(n_matching)


# ── LLM Summary ──────────────────────────────────────────────────────

def generate_llm_summary(actionable_deficits, best_intervention, base_auc, new_auc, patient_context=""):
    import os
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "AI summary generation skipped: GOOGLE_API_KEY not found in .env."

        llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0.0)
        prompt = f"""
You are a clinical AI assistant. Analyze the patient data below to write a highly concise clinical summary.
Contextualize the findings around the patient's demographics and history.

Patient Context: {patient_context}
Current Issues (Actionable Deficits): {actionable_deficits}

Predicted Digital Twin Timeline:
- Baseline 5-Year Velocity Area Under Curve (AUC): {base_auc:.2f}
- Recommended Action Plan: {best_intervention['label']}
- New Predicted 5-Year Velocity AUC: {new_auc:.2f}
- Impact: {best_intervention['auc_reduction_pct']:.1f}% reduction in biological aging velocity.

Provide a professional 3-sentence clinical summary of the patient's current state and your recommended action plan. Provide only the core summary.
"""
        response = llm.invoke(prompt)
        content  = response.content
        if isinstance(content, list):
            content = " ".join(c["text"] if isinstance(c, dict) else c for c in content)
        return str(content).strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return "AI summary generation currently unavailable."


# ── Core Ranking Logic (shared) ──────────────────────────────────────

def _run_ranking(z0, u_baseline, df_raw_csv, model, ode_func, ckpt, device,
                 years=5.0, washout_k=2.0):
    """
    Given encoded z0 and baseline control, simulate all intervention scenarios
    and return the ranked results dict.
    """
    from itertools import combinations

    z0_np = z0.squeeze().cpu().numpy()
    actionable = [
        col for col in U_COLS
        if abs(u_baseline[0, U_COLS.index(col)].item() - TARGET_MAP[col]) > 0.05
    ]

    if not actionable:
        return None

    N     = len(actionable)
    max_r = N if N <= 3 else 2
    scenarios = [list(combo)
                 for r in range(1, max_r + 1)
                 for combo in combinations(actionable, r)]

    t_span = torch.linspace(0, years, 50).to(device)
    t_np   = t_span.cpu().numpy()

    _configure_ode(ode_func, u_baseline)
    with torch.no_grad():
        z_traj_base = odeint(ode_func, z0, t_span, method='rk4')
        v_mag_base  = [torch.norm(ode_func(t, z_traj_base[i]), dim=-1).item()
                       for i, t in enumerate(t_span)]
    v_mag_base   = np.array(v_mag_base)
    auc_baseline = np.trapz(v_mag_base, t_np)

    results = []
    for scenario in scenarios:
        u_twin = u_baseline.clone()
        for col in scenario:
            u_twin[0, U_COLS.index(col)] = TARGET_MAP[col]

        u_twin_np  = u_twin.squeeze().cpu().numpy()
        maha_dist, n_match = _compute_cohort_mahalanobis(
            z0_np, u_twin_np, df_raw_csv, model, ckpt, device
        )

        v_mag_twin   = _simulate_single_twin(ode_func, z0, u_baseline, u_twin, t_span, washout_k)
        auc_twin     = np.trapz(v_mag_twin, t_np)
        reduction_pct = (auc_baseline - auc_twin) / auc_baseline * 100 if auc_baseline > 0 else 0.0

        results.append({
            'label':             " + ".join(LABEL_MAP[c] for c in scenario),
            'targets':           scenario,
            'v_mag':             v_mag_twin,
            'auc':               auc_twin,
            'auc_reduction_pct': reduction_pct,
            'mahalanobis':       maha_dist,
            'n_cohort_match':    n_match,
            'confidence':        "High" if maha_dist <= 3.0 else "Low (OOD)",
        })

    results.sort(key=lambda x: x['auc_reduction_pct'], reverse=True)
    return t_np, v_mag_base, auc_baseline, results, [LABEL_MAP[a] for a in actionable]


# ── Public API ────────────────────────────────────────────────────────

def rank_interventions(cunicah, np_val, years=5.0, washout_k=2.0):
    """Rank interventions for a stored patient (identified by cunicah/np_val)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ode_func = load_models(device)
    ckpt = model._ckpt

    df_raw_csv = pd.read_csv(DATA_DIR / 'frailty_index_data.csv')
    p_data = df_raw_csv[(df_raw_csv['cunicah'] == cunicah) & (df_raw_csv['np'] == np_val)]
    if p_data.empty:
        print(f"Patient {cunicah}/{np_val} not found.")
        return None

    z0           = _build_latent_ode_z0(model, ckpt, p_data, device)
    latest_visit = p_data.sort_values(by='a_o_ent').iloc[-1]
    u_dict       = _extract_patient_u(latest_visit)
    u_baseline   = torch.tensor([[u_dict[c] for c in U_COLS]], dtype=torch.float32).to(device)

    print(f"\nActionable for Patient {int(cunicah)}/{int(np_val)}: "
          f"{[LABEL_MAP[c] for c in U_COLS if abs(u_baseline[0, U_COLS.index(c)].item() - TARGET_MAP[c]) > 0.05]}")

    out = _run_ranking(z0, u_baseline, df_raw_csv, model, ode_func, ckpt, device, years, washout_k)
    if out is None:
        print(f"Patient {int(cunicah)}/{int(np_val)} already meets all targets.")
        return None
    t_np, v_mag_base, auc_baseline, results, actionable_labels = out

    p_row = latest_visit
    age   = p_row.get('edad', 65.0)
    sex   = "Male" if p_row.get('sexo', 1.0) == 1.0 else "Female"
    hist  = [c for c in ['hipertension', 'diabetes', 'enf_pulm', 'artritis',
                          'infarto', 'embolia', 'cancer'] if p_row.get(c) == 1.0]
    patient_context = (f"{int(age)}-year-old {sex} with "
                       f"{'a history of ' + ', '.join(hist) if hist else 'no major comorbidities'}.")

    return {
        't':                   t_np,
        'v_mag_baseline':      v_mag_base,
        'auc_baseline':        auc_baseline,
        'ranked_interventions':results,
        'patient_id':          f"{int(cunicah)}/{int(np_val)}",
        'actionable_deficits': actionable_labels,
        'patient_context':     patient_context,
    }


def rank_custom_patient(patient_data: dict, years=5.0, washout_k=2.0):
    """Rank interventions for a custom patient submitted via the live inference form."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ode_func = load_models(device)
    ckpt = model._ckpt

    df_raw_csv = pd.read_csv(DATA_DIR / 'frailty_index_data.csv')

    z0         = _build_single_visit_z0(model, ckpt, patient_data, device)
    u_dict     = _extract_patient_u(pd.Series(patient_data))
    u_baseline = torch.tensor([[u_dict[c] for c in U_COLS]], dtype=torch.float32).to(device)

    out = _run_ranking(z0, u_baseline, df_raw_csv, model, ode_func, ckpt, device, years, washout_k)
    if out is None:
        print("Custom patient already meets all targets.")
        return None
    t_np, v_mag_base, auc_baseline, results, actionable_labels = out

    try:
        age  = float(patient_data.get('edad', 65.0))
        sex  = "Male" if float(patient_data.get('sexo', 1.0)) == 1.0 else "Female"
        hist = [c for c in ['hipertension', 'diabetes', 'enf_pulm', 'artritis',
                             'infarto', 'embolia', 'cancer']
                if float(patient_data.get(c, 0.0)) == 1.0]
        patient_context = (f"{int(age)}-year-old {sex} with "
                           f"{'a history of ' + ', '.join(hist) if hist else 'no major comorbidities'}.")
    except Exception:
        patient_context = "Custom patient."

    return {
        't':                   t_np,
        'v_mag_baseline':      v_mag_base,
        'auc_baseline':        auc_baseline,
        'ranked_interventions':results,
        'patient_id':          "Live Inference",
        'actionable_deficits': actionable_labels,
        'patient_context':     patient_context,
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
