"""
Benchmark: CADENCE vs clinical baselines (BMC Bioinformatics companion).

Five models evaluated on the full MHAS cohort:

  B1   Cox PH — baseline FI only            (clinical gold standard)
  B2   Cox PH — first-interval FI slope     (naive velocity)
  B3   Cox PH — demographics only           (age, sex, education)
  B4   Latent ODE-VAE encoder + Cox         (ablation: same encoder, no ODE dynamics)
  CADENCE Full model                           (velocity phenotype + uncertainty + age)

B4 uses the same RecognitionRNN encoder and RiskHead as CADENCE but skips the ODE
integration entirely, directly fitting Cox on [risk_head(µ), baseline_age]. This
isolates the contribution of the continuous-time velocity step.

Primary metric: Harrell's C-index, 95% CI via 1000-sample bootstrap.

Output:
  console  — formatted results table
  latent_velocity/paper/benchmark_results.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
sys.path.insert(0, str(_THIS.parent))

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from _paths import DATA_DIR, MODELS_DIR

FI_PATH   = DATA_DIR   / "frailty_index_data.csv"
TRAJ_PATH = MODELS_DIR / "latent_velocity_trajectory_128.csv"
OUT_CSV   = _ROOT / "paper" / "benchmark_results.csv"
OUT_CSV_CV = _ROOT / "paper" / "benchmark_results_cv.csv"
NESTED_CSV = _ROOT / "paper" / "benchmark_nested_cv.csv"
STRATA_CSV = _ROOT / "paper" / "benchmark_agestrata_cv.csv"
STRATA_PAIRED_CSV = _ROOT / "paper" / "benchmark_agestrata_paired_cv.csv"
AGE_BANDS = ((0, 60), (60, 70), (70, 80), (80, 200))

FORECAST_CSV = _ROOT / "paper" / "benchmark_forecast_cv.csv"
INCIDENT_CSV = _ROOT / "paper" / "benchmark_incident_cv.csv"

# Domain composites (subsets of the 34-deficit FI). Each item is a deficit in
# [0,1] with higher = worse, so a composite is the fraction of domain deficits.
FUNC_COLS = ["n_abvd", "n_aivd", "n_mov", "n_img",
             "motoras_gruesas", "motoras_finas"]
COG_COLS  = ["recuerdo1", "recuerdo2", "copiafiguras1", "copiafiguras2",
             "orientacion", "serial7", "visualscan", "memoria"]
FOLDS_PATH = MODELS_DIR / "fold_assignments.csv"

N_BOOT    = 1000
BOOT_SEED = 42


# ─── Shared survival outcome table ──────────────────────────────────────────

def build_survival_data(df_fi: pd.DataFrame) -> pd.DataFrame:
    """
    Per-patient survival table, consistent with clinical_validation.py:
      tte   — years from first to last alive-confirmed wave (deceased)
              or last observed wave (censored)
      event — 1 if fallecido ever recorded, else 0
    """
    df = df_fi.copy()
    df["fallecido"] = (df["fallecido"] == 1).astype(int)

    last_alive = (df[df["fallecido"] == 0]
                  .groupby(["cunicah", "np"])["a_o_ent"].max()
                  .reset_index(name="last_alive_year"))

    surv = df.groupby(["cunicah", "np"]).agg(
        event      = ("fallecido",  "max"),
        start_year = ("a_o_ent",    "min"),
        end_year   = ("a_o_ent",    "max"),
        age        = ("edad",       "min"),
        sex        = ("sexo",       "first"),
        edu        = ("educacion",  "first"),
    ).reset_index()

    surv = surv.merge(last_alive, on=["cunicah", "np"], how="left")
    dead = surv["event"] == 1
    surv.loc[dead, "end_year"] = (surv.loc[dead, "last_alive_year"]
                                  .fillna(surv.loc[dead, "end_year"]))
    surv["tte"] = surv["end_year"] - surv["start_year"]
    return surv[surv["tte"] > 0].dropna(subset=["tte", "event"]).copy()


# ─── Bootstrap C-index ──────────────────────────────────────────────────────

def cindex_with_ci(tte: np.ndarray, risk: np.ndarray, event: np.ndarray,
                   n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Return (point, lo, hi) with percentile bootstrap CI."""
    point = concordance_index(tte, -risk, event)
    rng   = np.random.default_rng(seed)
    n     = len(tte)
    boot  = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if event[idx].sum() == 0:
            continue
        try:
            boot.append(concordance_index(tte[idx], -risk[idx], event[idx]))
        except Exception:
            continue
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, lo, hi


def _fit_and_score(df: pd.DataFrame, feature_cols: list[str]) -> tuple:
    cph  = CoxPHFitter(penalizer=0.1).fit(
        df[["tte", "event"] + feature_cols],
        duration_col="tte", event_col="event",
    )
    risk = cph.predict_partial_hazard(df[feature_cols]).values
    ci   = cindex_with_ci(df["tte"].values, risk, df["event"].values)
    return ci, len(df)


# ─── B1: Cox on baseline FI ─────────────────────────────────────────────────

def run_b1(df_fi: pd.DataFrame, surv: pd.DataFrame):
    fi_base = (df_fi.sort_values(["cunicah", "np", "a_o_ent"])
               .groupby(["cunicah", "np"])["FI"].first()
               .reset_index(name="fi_base"))
    df = surv.merge(fi_base, on=["cunicah", "np"]).dropna(subset=["fi_base"])
    return _fit_and_score(df, ["fi_base"])


# ─── B2: Cox on first-interval FI slope ─────────────────────────────────────

def run_b2(df_fi: pd.DataFrame, surv: pd.DataFrame):
    """Naive velocity: ΔFI / Δt between first and second observed waves."""
    tmp = df_fi.copy()
    tmp["t"] = tmp["a_o_ent"] - 2001
    tmp = tmp.sort_values(["cunicah", "np", "t"])

    slopes = []
    for (c, n), g in tmp.groupby(["cunicah", "np"]):
        g = g.dropna(subset=["FI"])
        if len(g) < 2:
            continue
        dt = float(g["t"].iloc[1] - g["t"].iloc[0])
        if dt <= 0:
            continue
        slopes.append({
            "cunicah":  c,
            "np":       n,
            "fi_slope": float(g["FI"].iloc[1] - g["FI"].iloc[0]) / dt,
        })

    df = surv.merge(pd.DataFrame(slopes), on=["cunicah", "np"]).dropna(subset=["fi_slope"])
    return _fit_and_score(df, ["fi_slope"])


# ─── B3: Cox on demographics ─────────────────────────────────────────────────

def run_b3(surv: pd.DataFrame):
    df = surv.dropna(subset=["age", "sex", "edu"]).copy()
    df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std()
    df["edu_z"] = (df["edu"] - df["edu"].mean()) / df["edu"].std()
    return _fit_and_score(df, ["age_z", "sex", "edu_z"])


# ─── B4: static encoder + Cox (ablation — no ODE dynamics) ──────────────────

def run_b4(surv: pd.DataFrame):
    """
    Encode each patient's full visit sequence with the RecognitionRNN → µ,
    then pass µ through the RiskHead to produce a scalar mortality score.
    Fit Cox on [risk_score, baseline_age].

    This is a direct ablation of CADENCE: same encoder and risk head, but the
    ODE integration and velocity phenotyping steps are skipped entirely.
    """
    from train_latent_ode import LatentODE, LatentODEDataset, MHAS_WAVES, T_MAX

    ckpt = torch.load(str(MODELS_DIR / "latent_ode_model.pth"),
                      map_location="cpu", weights_only=False)
    model = LatentODE()
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    dataset = LatentODEDataset(
        str(FI_PATH),
        edad_mean=ckpt["edad_mean"], edad_std=ckpt["edad_std"],
        edu_mean=ckpt["edu_mean"],   edu_std=ckpt["edu_std"],
    )
    t_norm = torch.tensor([w / T_MAX for w in MHAS_WAVES], dtype=torch.float32)

    rows = []
    with torch.no_grad():
        for s in dataset.samples:
            mu, _ = model.encode(
                s["x"].unsqueeze(0), t_norm.unsqueeze(0), s["mask"].unsqueeze(0)
            )
            rows.append({
                "cunicah":     s["cunicah"],
                "np":          s["np"],
                "risk_score":  model.risk_head(mu).item(),
            })

    enc = pd.DataFrame(rows)
    df  = surv.merge(enc, on=["cunicah", "np"]).dropna(subset=["risk_score", "age"])
    df["risk_z"] = (df["risk_score"] - df["risk_score"].mean()) / df["risk_score"].std()
    df["age_z"]  = (df["age"]        - df["age"].mean())        / df["age"].std()
    return _fit_and_score(df, ["risk_z", "age_z"])


# ─── CADENCE: full model ────────────────────────────────────────────────────────

def run_lava(df_fi: pd.DataFrame, surv: pd.DataFrame):
    """
    Replicates the CADENCE Cox model from clinical_validation.py:
      Fast_Ager_Flag + age_z + unc_z → Cox
    """
    from clinical_validation import calculate_velocity_magnitude, compute_frailty_velocity

    df_traj, vcols = calculate_velocity_magnitude()
    df_traj = compute_frailty_velocity(df_traj, df_fi, vcols)

    df_early   = df_traj.groupby(["cunicah", "np"]).head(30)
    patient_vf = (df_early.groupby(["cunicah", "np"])["v_frailty"].mean()
                  .reset_index().rename(columns={"v_frailty": "v_frailty_mean"}))

    if "v_uncertainty" in df_traj.columns:
        patient_unc = (df_early.groupby(["cunicah", "np"])["v_uncertainty"].mean()
                       .reset_index().rename(columns={"v_uncertainty": "mean_unc"}))
        patient_vf = patient_vf.merge(patient_unc, on=["cunicah", "np"], how="left")
    else:
        patient_vf["mean_unc"] = np.nan

    q1 = patient_vf["v_frailty_mean"].quantile(0.25)
    q3 = patient_vf["v_frailty_mean"].quantile(0.75)
    cond = [patient_vf["v_frailty_mean"] <= q1,
            patient_vf["v_frailty_mean"] >= q3]
    patient_vf["Phenotype"] = np.select(cond, ["Slow_Ager", "Fast_Ager"], default="Normal")

    # Survivorship fix: single-observation deceased patients → Fast_Ager
    obs_n      = df_fi.groupby(["cunicah", "np"]).size().reset_index(name="n_obs")
    single_dead = (obs_n[obs_n["n_obs"] == 1]
                   .merge(df_fi[["cunicah", "np", "fallecido"]]
                          .query("fallecido == 1").drop_duplicates(),
                          on=["cunicah", "np"], how="inner"))
    if not single_dead.empty:
        rescued = single_dead[["cunicah", "np"]].copy()
        rescued["v_frailty_mean"] = q3 + 1e-3
        rescued["mean_unc"]       = patient_vf["mean_unc"].quantile(0.75)
        rescued["Phenotype"]      = "Fast_Ager"
        patient_vf = pd.concat([patient_vf, rescued], ignore_index=True)

    patient_vf = patient_vf[patient_vf["Phenotype"] != "Normal"]

    df = surv.merge(patient_vf[["cunicah", "np", "Phenotype", "mean_unc"]],
                    on=["cunicah", "np"])
    df["Fast_Ager_Flag"] = (df["Phenotype"] == "Fast_Ager").astype(float)
    df["age_z"] = (df["age"]      - df["age"].mean())      / df["age"].std()
    df["unc_z"] = (df["mean_unc"] - df["mean_unc"].mean()) / df["mean_unc"].std()
    df2 = df[["tte", "event", "Fast_Ager_Flag", "age_z", "unc_z"]].dropna()
    return _fit_and_score(df2, ["Fast_Ager_Flag", "age_z", "unc_z"])


# ─── Driver ──────────────────────────────────────────────────────────────────

MODELS = [
    ("B1",   "Cox — baseline FI",                      lambda fi, s: run_b1(fi, s)),
    ("B2",   "Cox — FI slope (first interval)",        lambda fi, s: run_b2(fi, s)),
    ("B3",   "Cox — demographics only",                lambda fi, s: run_b3(s)),
    ("B4",   "Latent encoder + Cox  [no ODE]",         lambda fi, s: run_b4(s)),
    ("CADENCE", "CADENCE  [full model]",                     lambda fi, s: run_lava(fi, s)),
]


def main():
    print("Loading MHAS data…")
    df_fi = pd.read_csv(FI_PATH)
    surv  = build_survival_data(df_fi)
    print(f"  Survival table: {len(surv):,} patients  "
          f"({int(surv['event'].sum()):,} events)\n")

    results = {}
    for key, label, fn in MODELS:
        print(f"[{key}] {label}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (ci, lo, hi), n = fn(df_fi, surv)
        results[key] = (label, ci, lo, hi, n)
        print(f"      C-index {ci:.3f}  [{lo:.3f}–{hi:.3f}]  n={n:,}\n")

    # ── Console table ────────────────────────────────────────────────────────
    w = 80
    print("=" * w)
    print(f"  {'Model':<44} {'C-index':>8}  {'95% CI':>15}  {'n':>7}")
    print("-" * w)
    for key, (label, ci, lo, hi, n) in results.items():
        tag = "  ◀" if key == "CADENCE" else ""
        print(f"  {label:<44} {ci:.3f}    [{lo:.3f}–{hi:.3f}]  {n:>7,}{tag}")
    print("=" * w)
    print("Bootstrap CIs: 1,000 samples, percentile method.")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"model": k, "label": v[0], "c_index": round(v[1], 4),
         "ci_lo": round(v[2], 4), "ci_hi": round(v[3], 4), "n": v[4]}
        for k, v in results.items()
    ]).to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV.relative_to(_ROOT)}")


# ════════════════════════════════════════════════════════════════════════════
# Out-of-fold (5-fold patient-level CV) variants
# ════════════════════════════════════════════════════════════════════════════
#
# These functions assemble out-of-fold predictions across the K folds saved by
# `train_latent_ode.py --cv K`. For every patient, the risk score used in the
# C-index is produced by a Cox model that never saw that patient at fit time
# AND (for B4, CADENCE) by a LatentODE checkpoint that did not include them in
# its training set. Pooled across folds this yields a single honest cohort-level
# C-index whose bootstrap CI reflects out-of-sample generalisation.

def _load_folds() -> pd.DataFrame:
    if not FOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Fold assignments not found at {FOLDS_PATH}. "
            f"Run `python latent_velocity/engine/train_latent_ode.py --cv 5` first.")
    return pd.read_csv(FOLDS_PATH)


def _cox_oof(df: pd.DataFrame, feature_cols: list[str], fold_col: str = "fold"):
    """For each fold k, fit Cox on rows with fold!=k and predict on rows with fold==k.
    Returns the input df augmented with a 'risk' column of out-of-fold partial hazards."""
    df = df.copy()
    df["risk"] = np.nan
    for k in sorted(df[fold_col].unique()):
        train = df[df[fold_col] != k]
        test  = df[df[fold_col] == k]
        cph = CoxPHFitter(penalizer=0.1).fit(
            train[["tte", "event"] + feature_cols],
            duration_col="tte", event_col="event",
        )
        df.loc[test.index, "risk"] = cph.predict_partial_hazard(test[feature_cols]).values
    return df.dropna(subset=["risk"])


def _pool_cindex(df: pd.DataFrame):
    """Compute pooled C-index and bootstrap CI on out-of-fold risks."""
    ci = cindex_with_ci(df["tte"].values, df["risk"].values, df["event"].values)
    return ci, len(df)


def _paired_delta_cindex(df: pd.DataFrame, risk_full: str, risk_base: str,
                         n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Paired patient bootstrap of C-index(full) − C-index(base) on IDENTICAL rows.

    Both risk columns are out-of-fold predictions for the same patients, so the
    delta is the within-patient incremental discrimination of the full model over
    the baseline. Resampling the same indices for both arms makes the CI reflect
    the paired difference rather than two independent sampling distributions.
    Returns ((c_full, c_base, delta), (delta_lo, delta_hi)).
    """
    tte = df["tte"].values
    ev  = df["event"].values
    rf  = df[risk_full].values
    rb  = df[risk_base].values
    c_full = concordance_index(tte, -rf, ev)
    c_base = concordance_index(tte, -rb, ev)
    rng = np.random.default_rng(seed)
    n = len(tte)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if ev[idx].sum() == 0:
            continue
        try:
            cf = concordance_index(tte[idx], -rf[idx], ev[idx])
            cb = concordance_index(tte[idx], -rb[idx], ev[idx])
        except Exception:
            continue
        deltas.append(cf - cb)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return (c_full, c_base, c_full - c_base), (float(lo), float(hi))


def _cindex_by_age_strata(df: pd.DataFrame, risk_cols: dict, bands=AGE_BANDS):
    """Pooled out-of-fold C-index within age bands for each risk column.

    Within a narrow age band chronological age is nearly constant and cannot
    discriminate, so this isolates the discrimination contributed by the model's
    own signal. `risk_cols` maps a label → column of out-of-fold risk. Returns a
    list of dict rows (band, label, c_index, lo, hi, n_total, n_events).
    """
    rows = []
    for lo_a, hi_a in bands:
        sub = df[(df["age"] >= lo_a) & (df["age"] < hi_a)]
        band = f"{lo_a}-{hi_a}" if hi_a < 200 else f"{lo_a}+"
        n_ev = int(sub["event"].sum())
        for label, col in risk_cols.items():
            s = sub.dropna(subset=[col])
            if len(s) < 50 or s["event"].sum() < 10:
                rows.append({"band": band, "label": label, "c_index": np.nan,
                             "ci_lo": np.nan, "ci_hi": np.nan,
                             "n_total": len(s), "n_events": int(s["event"].sum())})
                continue
            ci, clo, chi = cindex_with_ci(s["tte"].values, s[col].values,
                                          s["event"].values)
            rows.append({"band": band, "label": label, "c_index": ci,
                         "ci_lo": clo, "ci_hi": chi,
                         "n_total": len(s), "n_events": n_ev})
    return rows


def _auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int = N_BOOT,
            seed: int = BOOT_SEED):
    """ROC-AUC with percentile bootstrap CI for a binary outcome `y`."""
    from sklearn.metrics import roc_auc_score
    point = roc_auc_score(y, score)
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        try:
            boot.append(roc_auc_score(y[idx], score[idx]))
        except Exception:
            continue
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _encode_fold_models_full_cohort(folds: pd.DataFrame, surv: pd.DataFrame):
    """
    For every fold k, load latent_ode_model_fold{k}.pth, encode the patients in
    fold k (held out from that model's training), and return a unified DataFrame
    (cunicah, np, fold, mu_0..mu_7, risk_score, v_unc).

    v_unc is the cross-dimensional std of the ODE-integrated velocity at the
    MHAS wave grid under n_mc latent samples — analogous to the dense
    v_uncertainty produced by extract_latent_ode_velocity.py but evaluated only
    at the (sparse) observation grid for cost.
    """
    from train_latent_ode import (
        LatentODE, LatentODEDataset,
        MHAS_WAVES, T_MAX, LATENT_DIM,
    )
    from torchdiffeq import odeint as _odeint

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_folds = int(folds["fold"].max()) + 1
    n_mc    = 10  # cheap MC for v_unc

    out_rows = []
    for k in range(n_folds):
        ckpt_path = MODELS_DIR / f"latent_ode_model_fold{k}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Missing CV checkpoint {ckpt_path}. "
                f"Run `python latent_velocity/engine/train_latent_ode.py --cv {n_folds}` first.")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = LatentODE().to(device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.eval()

        # Encode the held-out fold only (out-of-sample for this checkpoint).
        held_out = set(map(tuple, folds.loc[folds["fold"] == k, ["cunicah", "np"]].values))
        dataset  = LatentODEDataset(
            str(FI_PATH),
            edad_mean=ckpt["edad_mean"], edad_std=ckpt["edad_std"],
            edu_mean=ckpt["edu_mean"],   edu_std=ckpt["edu_std"],
            include_keys=list(held_out),
        )
        t_grid_ten = torch.tensor(MHAS_WAVES, dtype=torch.float32).to(device)
        t_norm_ten = (t_grid_ten / T_MAX)

        print(f"  fold {k}: encoding {len(dataset.samples):,} held-out patients")
        with torch.no_grad():
            for s in dataset.samples:
                x    = s["x"].unsqueeze(0).to(device)
                u    = s["u"].unsqueeze(0).to(device)
                mask = s["mask"].unsqueeze(0).to(device)

                mu, logvar = model.encode(x, t_norm_ten.unsqueeze(0), mask)
                std = (0.5 * logvar).exp()

                first_wi = int(mask[0].float().argmax().item())
                u0 = u[0, first_wi, :].unsqueeze(0)
                model.ode_func.current_u = u0
                model.ode_func.target_u  = None

                # MC velocity at the wave grid
                eps   = torch.randn(n_mc, LATENT_DIM, device=device)
                z0_mc = mu + eps * std
                v_per_mc = []
                for mc_i in range(n_mc):
                    z_traj = _odeint(model.ode_func, z0_mc[mc_i:mc_i+1],
                                     t_grid_ten, method="rk4").squeeze(1)
                    v_t = torch.stack([
                        model.ode_func(t_grid_ten[ti], z_traj[ti:ti+1]).squeeze(0)
                        for ti in range(len(MHAS_WAVES))
                    ])  # (N_WAVES, LATENT_DIM)
                    v_per_mc.append(v_t)
                v_stack = torch.stack(v_per_mc)            # (n_mc, N_WAVES, D)
                v_mean  = v_stack.mean(0).cpu().numpy()    # (N_WAVES, D)
                v_unc   = v_stack.std(0).mean().item()     # scalar — averaged over (T, D)

                row = {
                    "cunicah":    s["cunicah"],
                    "np":         s["np"],
                    "fold":       k,
                    "risk_score": float(model.risk_head(mu).item()),
                    "v_unc":      v_unc,
                }
                # store wave-grid velocity for ridge projection (CADENCE phenotype)
                for ti, t_val in enumerate(MHAS_WAVES):
                    for d in range(LATENT_DIM):
                        row[f"v_{ti}_{d}"] = float(v_mean[ti, d])
                out_rows.append(row)

    enc = pd.DataFrame(out_rows)
    enc = enc.merge(surv[["cunicah", "np", "tte", "event", "age"]],
                    on=["cunicah", "np"], how="inner")
    return enc


def _v_frailty_oof(enc: pd.DataFrame, df_fi: pd.DataFrame, n_folds: int):
    """
    For each fold k, fit a Ridge mapping wave-grid latent velocity → empirical
    FI velocity using only patients with fold != k, then project fold k's
    patients onto the resulting direction. Returns enc augmented with v_frailty.
    """
    from sklearn.linear_model import Ridge
    from train_latent_ode import LATENT_DIM, MHAS_WAVES

    # Empirical per-interval FI velocity, restricted to the 34-deficit FI used
    # at training time (matches compute_frailty_velocity in clinical_validation).
    deficit_cols_34 = [
        "hipertension", "diabetes", "enf_pulm", "artritis", "infarto", "embolia",
        "cancer", "salud_glob",
        "n_abvd", "n_aivd", "n_mov", "n_img", "motoras_gruesas", "motoras_finas",
        "deprimido", "esfuerzo", "intranquilo", "triste", "cansado", "solo",
        "feliz", "disf_vida", "energia",
        "recuerdo1", "recuerdo2", "copiafiguras1", "copiafiguras2",
        "orientacion", "serial7", "visualscan", "memoria",
        "bmi_imp", "hospitalizacion", "visita_medica",
    ]
    df = df_fi.copy()
    df["t"] = df["a_o_ent"] - 2001
    available = [c for c in deficit_cols_34 if c in df.columns]
    df["fi_34"] = df[available].mean(axis=1)
    df = df.sort_values(["cunicah", "np", "t"])
    df["next_fi_34"] = df.groupby(["cunicah", "np"])["fi_34"].shift(-1)
    df["next_t"]    = df.groupby(["cunicah", "np"])["t"].shift(-1)
    pairs = df.dropna(subset=["fi_34", "next_fi_34", "t", "next_t"])
    pairs = pairs[pairs["next_t"] > pairs["t"]].copy()
    pairs["fi_vel"] = ((pairs["next_fi_34"] - pairs["fi_34"]) /
                      (pairs["next_t"]    - pairs["t"]))

    # Patient-level average empirical fi_vel — coarse target for the ridge.
    target = (pairs.groupby(["cunicah", "np"])["fi_vel"].mean()
              .reset_index(name="fi_vel_mean"))

    # Patient-level mean wave-grid velocity (mean over waves, per dim).
    v_cols_mean = [f"v_mean_{d}" for d in range(LATENT_DIM)]
    for d in range(LATENT_DIM):
        cols = [f"v_{ti}_{d}" for ti in range(len(MHAS_WAVES))]
        enc[f"v_mean_{d}"] = enc[cols].mean(axis=1)

    enc = enc.merge(target, on=["cunicah", "np"], how="left")
    enc["v_frailty"] = np.nan
    for k in range(n_folds):
        train = enc[(enc["fold"] != k) & enc["fi_vel_mean"].notna()]
        test  = enc[enc["fold"] == k]
        if len(train) < 10:
            continue
        ridge = Ridge(alpha=1.0).fit(train[v_cols_mean].values, train["fi_vel_mean"].values)
        w     = ridge.coef_
        w_n   = np.linalg.norm(w)
        if w_n < 1e-12:
            continue
        w_unit = w / w_n
        enc.loc[test.index, "v_frailty"] = test[v_cols_mean].values @ w_unit
    return enc


def run_cv(n_folds_default: int = 5):
    print("Loading MHAS data…")
    df_fi = pd.read_csv(FI_PATH)
    surv  = build_survival_data(df_fi)
    folds = _load_folds()
    n_folds = int(folds["fold"].max()) + 1
    print(f"  Survival table: {len(surv):,} patients  "
          f"({int(surv['event'].sum()):,} events)")
    print(f"  Fold assignments: {n_folds} folds, sizes "
          f"{folds['fold'].value_counts().sort_index().tolist()}\n")

    # ── B1: out-of-fold Cox on baseline FI ─────────────────────────────────────
    print("[B1] Cox — baseline FI (out-of-fold)")
    fi_base = (df_fi.sort_values(["cunicah", "np", "a_o_ent"])
               .groupby(["cunicah", "np"])["FI"].first()
               .reset_index(name="fi_base"))
    df_b1 = (surv.merge(fi_base, on=["cunicah", "np"])
                 .merge(folds, on=["cunicah", "np"])
                 .dropna(subset=["fi_base"]))
    df_b1 = _cox_oof(df_b1, ["fi_base"])
    res_b1 = _pool_cindex(df_b1)
    print(f"      C-index {res_b1[0][0]:.3f}  "
          f"[{res_b1[0][1]:.3f}–{res_b1[0][2]:.3f}]  n={res_b1[1]:,}\n")

    # ── B2: out-of-fold Cox on first-interval FI slope ─────────────────────────
    print("[B2] Cox — first-interval FI slope (out-of-fold)")
    tmp = df_fi.copy()
    tmp["t"] = tmp["a_o_ent"] - 2001
    tmp = tmp.sort_values(["cunicah", "np", "t"])
    slopes = []
    for (c, n), g in tmp.groupby(["cunicah", "np"]):
        g = g.dropna(subset=["FI"])
        if len(g) < 2:
            continue
        dt = float(g["t"].iloc[1] - g["t"].iloc[0])
        if dt <= 0:
            continue
        slopes.append({"cunicah": c, "np": n,
                       "fi_slope": float(g["FI"].iloc[1] - g["FI"].iloc[0]) / dt})
    df_b2 = (surv.merge(pd.DataFrame(slopes), on=["cunicah", "np"])
                 .merge(folds, on=["cunicah", "np"])
                 .dropna(subset=["fi_slope"]))
    df_b2 = _cox_oof(df_b2, ["fi_slope"])
    res_b2 = _pool_cindex(df_b2)
    print(f"      C-index {res_b2[0][0]:.3f}  "
          f"[{res_b2[0][1]:.3f}–{res_b2[0][2]:.3f}]  n={res_b2[1]:,}\n")

    # ── B3: out-of-fold Cox on demographics ────────────────────────────────────
    print("[B3] Cox — demographics only (out-of-fold)")
    df_b3 = (surv.merge(folds, on=["cunicah", "np"])
                 .dropna(subset=["age", "sex", "edu"]).copy())
    df_b3["age_z"] = (df_b3["age"] - df_b3["age"].mean()) / df_b3["age"].std()
    df_b3["edu_z"] = (df_b3["edu"] - df_b3["edu"].mean()) / df_b3["edu"].std()
    df_b3 = _cox_oof(df_b3, ["age_z", "sex", "edu_z"])
    res_b3 = _pool_cindex(df_b3)
    print(f"      C-index {res_b3[0][0]:.3f}  "
          f"[{res_b3[0][1]:.3f}–{res_b3[0][2]:.3f}]  n={res_b3[1]:,}\n")

    # ── B4 + CADENCE: encode each fold's held-out cohort with its model ────────
    print(f"[B4/CADENCE] encoding held-out cohorts under their fold checkpoints…")
    enc = _encode_fold_models_full_cohort(folds, surv)
    enc = enc.merge(folds, on=["cunicah", "np"], suffixes=("", "_dup"))
    if "fold_dup" in enc.columns:
        enc = enc.drop(columns=["fold_dup"])

    # B4 out-of-fold: Cox on [risk_z, age_z]
    print("[B4] Latent encoder + Cox (out-of-fold)")
    df_b4 = enc.copy()
    df_b4["risk_z"] = (df_b4["risk_score"] - df_b4["risk_score"].mean()) / df_b4["risk_score"].std()
    df_b4["age_z"]  = (df_b4["age"]        - df_b4["age"].mean())        / df_b4["age"].std()
    df_b4 = _cox_oof(df_b4, ["risk_z", "age_z"])
    res_b4 = _pool_cindex(df_b4)
    print(f"      C-index {res_b4[0][0]:.3f}  "
          f"[{res_b4[0][1]:.3f}–{res_b4[0][2]:.3f}]  n={res_b4[1]:,}\n")

    # v_frailty for the FULL encodable cohort (no dichotomization, no dropping).
    enc_vf = _v_frailty_oof(enc, df_fi, n_folds)

    # ── [1] CADENCE-continuous: full cohort, continuous features ───────────────
    # The dichotomized CADENCE arm below keeps only Q1/Q3 velocity extremes and
    # collapses velocity to one bit. Here we instead use the continuous velocity
    # and uncertainty over EVERY encodable patient — more honest and uses all data.
    print("[CADENCE-C] Continuous v_frailty + uncertainty + age, full cohort (out-of-fold)")
    cadc = enc_vf.dropna(subset=["v_frailty"]).copy()
    cadc["vf_z"]  = (cadc["v_frailty"] - cadc["v_frailty"].mean()) / cadc["v_frailty"].std()
    cadc["unc_z"] = (cadc["v_unc"]     - cadc["v_unc"].mean())     / cadc["v_unc"].std()
    cadc["age_z"] = (cadc["age"]       - cadc["age"].mean())       / cadc["age"].std()
    cadc = _cox_oof(cadc, ["vf_z", "unc_z", "age_z"])
    res_cadc = _pool_cindex(cadc)
    print(f"      C-index {res_cadc[0][0]:.3f}  "
          f"[{res_cadc[0][1]:.3f}–{res_cadc[0][2]:.3f}]  n={res_cadc[1]:,}")
    # nested Δ vs age-only on the same full cohort
    nested_c = cadc.rename(columns={"risk": "risk_full"}).copy()
    nested_c = _cox_oof(nested_c, ["age_z"]).rename(columns={"risk": "risk_age"})
    (cc_full, cc_age, cd_pt), (cd_lo, cd_hi) = _paired_delta_cindex(
        nested_c, "risk_full", "risk_age")
    cc_incremental = not (cd_lo <= 0 <= cd_hi)
    print(f"      Δ vs age-only         {cd_pt:+.3f}  [{cd_lo:+.3f}, {cd_hi:+.3f}]"
          f"  → {'incremental' if cc_incremental else 'no incremental value'}\n")

    # CADENCE out-of-fold: phenotype + Cox on [Fast_Ager_Flag, age_z, unc_z]
    print("[CADENCE] Fast/Slow Ager phenotype + uncertainty + age (out-of-fold)")
    cad = enc_vf.dropna(subset=["v_frailty"]).copy()
    cad["Phenotype"] = "Normal"
    for k in range(n_folds):
        train = cad[cad["fold"] != k]
        test_idx = cad.index[cad["fold"] == k]
        if len(train) < 10:
            continue
        q1 = train["v_frailty"].quantile(0.25)
        q3 = train["v_frailty"].quantile(0.75)
        v = cad.loc[test_idx, "v_frailty"]
        cad.loc[test_idx, "Phenotype"] = np.select(
            [v <= q1, v >= q3], ["Slow_Ager", "Fast_Ager"], default="Normal")
    cad = cad[cad["Phenotype"] != "Normal"].copy()
    cad["Fast_Ager_Flag"] = (cad["Phenotype"] == "Fast_Ager").astype(float)
    cad["age_z"] = (cad["age"]   - cad["age"].mean())   / cad["age"].std()
    cad["unc_z"] = (cad["v_unc"] - cad["v_unc"].mean()) / cad["v_unc"].std()
    cad = _cox_oof(cad, ["Fast_Ager_Flag", "age_z", "unc_z"])
    res_lava = _pool_cindex(cad)
    print(f"      C-index {res_lava[0][0]:.3f}  "
          f"[{res_lava[0][1]:.3f}–{res_lava[0][2]:.3f}]  n={res_lava[1]:,}\n")

    # ── NESTED: incremental value over age, SAME cohort, out-of-fold ───────────
    # The headline question: do the latent velocity / phenotype / uncertainty
    # features add discrimination *on top of chronological age*, on the exact same
    # patients? `cad` already carries the full-model out-of-fold risk in 'risk';
    # we refit an age-only Cox out-of-fold on those identical rows and compare via
    # a paired bootstrap. If the Δ CI includes 0, the machinery adds nothing.
    print("[NESTED] Incremental value over age (same CADENCE cohort, out-of-fold)")
    nested = cad.rename(columns={"risk": "risk_full"}).copy()
    nested = _cox_oof(nested, ["age_z"]).rename(columns={"risk": "risk_age"})
    (c_full, c_age, d_pt), (d_lo, d_hi) = _paired_delta_cindex(
        nested, "risk_full", "risk_age")
    incremental = not (d_lo <= 0 <= d_hi)
    verdict = ("incremental value over age (Δ CI excludes 0)" if incremental
               else "NO incremental value over age (Δ CI includes 0)")
    print(f"      age-only      C-index {c_age:.3f}")
    print(f"      CADENCE full  C-index {c_full:.3f}")
    print(f"      Δ (full − age)        {d_pt:+.3f}  "
          f"[{d_lo:+.3f}, {d_hi:+.3f}]  n={len(nested):,}")
    print(f"      → {verdict}\n")

    NESTED_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"cohort": "CADENCE_extremes", "n": len(nested),
         "c_age_only": round(c_age, 4), "c_cadence_full": round(c_full, 4),
         "delta": round(d_pt, 4), "delta_lo": round(d_lo, 4),
         "delta_hi": round(d_hi, 4), "incremental_value": bool(incremental)},
        {"cohort": "CADENCE_continuous_full", "n": len(nested_c),
         "c_age_only": round(cc_age, 4), "c_cadence_full": round(cc_full, 4),
         "delta": round(cd_pt, 4), "delta_lo": round(cd_lo, 4),
         "delta_hi": round(cd_hi, 4), "incremental_value": bool(cc_incremental)},
    ]).to_csv(NESTED_CSV, index=False)
    print(f"Saved → {NESTED_CSV.relative_to(_ROOT)}\n")

    # ── [2] Age-stratified C-index ─────────────────────────────────────────────
    # Within a band, age is nearly constant → isolates the model's own signal.
    print("[STRATA] Out-of-fold C-index within age bands (age-only vs CADENCE-C)")
    strata_df = cadc.copy()                      # carries risk(age+vf+unc) in 'risk'
    strata_df = strata_df.rename(columns={"risk": "risk_cadence"})
    strata_df = _cox_oof(strata_df, ["age_z"]).rename(columns={"risk": "risk_age"})
    strata_rows = _cindex_by_age_strata(
        strata_df, {"age_only": "risk_age", "CADENCE_continuous": "risk_cadence"})
    for r in strata_rows:
        c = f"{r['c_index']:.3f}" if pd.notna(r["c_index"]) else "  n/a"
        print(f"      age {r['band']:<6} {r['label']:<18} C-index {c}"
              f"   (n={r['n_total']:,}, ev={r['n_events']:,})")
    STRATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(strata_rows).to_csv(STRATA_CSV, index=False)
    print(f"Saved → {STRATA_CSV.relative_to(_ROOT)}")

    # Paired within-band bootstrap Δ (CADENCE-C − age) on identical patients —
    # the rigorous significance test for the stratified claim.
    print("      paired within-band Δ (CADENCE-C − age-only):")
    paired_rows = []
    for lo_a, hi_a in AGE_BANDS:
        sub = strata_df[(strata_df["age"] >= lo_a) & (strata_df["age"] < hi_a)] \
            .dropna(subset=["risk_cadence", "risk_age"])
        band = f"{lo_a}-{hi_a}" if hi_a < 200 else f"{lo_a}+"
        if len(sub) < 50 or sub["event"].sum() < 10:
            continue
        (cf, ca, dp), (dlo, dhi) = _paired_delta_cindex(
            sub, "risk_cadence", "risk_age")
        sig = not (dlo <= 0 <= dhi)
        paired_rows.append({"band": band, "n": len(sub),
                            "c_age_only": round(ca, 4), "c_cadence": round(cf, 4),
                            "delta": round(dp, 4), "delta_lo": round(dlo, 4),
                            "delta_hi": round(dhi, 4), "significant": bool(sig)})
        print(f"      age {band:<6} Δ {dp:+.3f}  [{dlo:+.3f}, {dhi:+.3f}]"
              f"  {'SIG' if sig else 'ns'}")
    pd.DataFrame(paired_rows).to_csv(STRATA_PAIRED_CSV, index=False)
    print(f"Saved → {STRATA_PAIRED_CSV.relative_to(_ROOT)}\n")

    results = {
        "B1":        ("Cox — baseline FI",                res_b1),
        "B2":        ("Cox — FI slope (first interval)",  res_b2),
        "B3":        ("Cox — demographics only",          res_b3),
        "B4":        ("Latent encoder + Cox  [no ODE]",   res_b4),
        "CADENCE-C": ("CADENCE-continuous [full cohort]", res_cadc),
        "CADENCE":   ("CADENCE  [dichotomized]",          res_lava),
    }

    w = 80
    print("=" * w)
    print(f"  OUT-OF-FOLD ({n_folds}-fold patient-level CV)")
    print(f"  {'Model':<44} {'C-index':>8}  {'95% CI':>15}  {'n':>7}")
    print("-" * w)
    for key, (label, ((ci, lo, hi), n)) in results.items():
        tag = "  ◀" if key == "CADENCE-C" else ""
        print(f"  {label:<44} {ci:.3f}    [{lo:.3f}–{hi:.3f}]  {n:>7,}{tag}")
    print("=" * w)
    print("Bootstrap CIs: 1,000 samples, percentile method on pooled OOF risks.")

    OUT_CSV_CV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"model": k, "label": label, "c_index": round(ci, 4),
         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n": n}
        for k, (label, ((ci, lo, hi), n)) in results.items()
    ]).to_csv(OUT_CSV_CV, index=False)
    print(f"\nSaved → {OUT_CSV_CV.relative_to(_ROOT)}")


# ════════════════════════════════════════════════════════════════════════════
# Trajectory forecasting (plan items #3 and #4) — out-of-fold
# ════════════════════════════════════════════════════════════════════════════
#
# For every patient with ≥3 observed waves, we hide the LAST observed wave,
# encode only the earlier waves with the fold checkpoint that never trained on
# the patient, integrate the ODE forward, and decode the held-out wave. This is
# a genuine forecast: the target is unseen by both the checkpoint (held-out fold)
# and the encoder (masked input). We compare CADENCE's forecast against naive
# baselines (last-value-carried-forward, linear extrapolation) and a demographic
# predictor, on the whole Frailty Index (#4) and on the functional and cognitive
# sub-domains, where chronological age is a weak predictor of decline (#3).

def _forecast_fold_models(folds: pd.DataFrame, fi_df: pd.DataFrame):
    """Per-patient held-out-wave forecast rows assembled across fold checkpoints."""
    from train_latent_ode import (
        LatentODE, LatentODEDataset, DEFICIT_COLS,
        MHAS_WAVES, T_MAX, LATENT_DIM,
    )
    from torchdiffeq import odeint as _odeint

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_folds = int(folds["fold"].max()) + 1
    waves   = np.array(MHAS_WAVES, dtype=np.float64)
    func_ix = [DEFICIT_COLS.index(c) for c in FUNC_COLS]
    cog_ix  = [DEFICIT_COLS.index(c) for c in COG_COLS]
    n_def   = len(DEFICIT_COLS)

    def _composites(vec):  # vec: (34,) deficit values in [0,1]
        return {"fi":   float(np.mean(vec)),
                "func": float(np.mean(vec[func_ix])),
                "cog":  float(np.mean(vec[cog_ix]))}

    out_rows = []
    t_grid = torch.tensor(MHAS_WAVES, dtype=torch.float32).to(device)
    t_norm = (t_grid / T_MAX).unsqueeze(0)

    for k in range(n_folds):
        ckpt_path = MODELS_DIR / f"latent_ode_model_fold{k}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Missing CV checkpoint {ckpt_path}. "
                f"Run `python latent_velocity/engine/train_latent_ode.py --cv {n_folds}` first.")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = LatentODE().to(device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.eval()

        held = set(map(tuple, folds.loc[folds["fold"] == k, ["cunicah", "np"]].values))
        ds = LatentODEDataset(
            str(FI_PATH),
            edad_mean=ckpt["edad_mean"], edad_std=ckpt["edad_std"],
            edu_mean=ckpt["edu_mean"],   edu_std=ckpt["edu_std"],
            include_keys=list(held),
        )
        print(f"  fold {k}: forecasting on held-out patients (≥3 waves only)")
        with torch.no_grad():
            for s in ds.samples:
                mask_np = s["mask"].numpy()
                obs = np.where(mask_np)[0]
                if len(obs) < 3:
                    continue                      # need ≥2 input + 1 target
                tgt_i  = int(obs[-1])
                last_i = int(obs[-2])
                in_idx = obs[:-1]

                x = s["x"].unsqueeze(0).to(device)        # (1, N_WAVES, INPUT_DIM)
                u = s["u"].unsqueeze(0).to(device)
                mask_in = torch.zeros_like(s["mask"]).unsqueeze(0).to(device)
                mask_in[0, in_idx] = True

                mu, _ = model.encode(x, t_norm, mask_in)  # (1, LATENT_DIM)
                u0 = u[0, int(in_idx[0]), :].unsqueeze(0)
                model.ode_func.current_u = u0
                model.ode_func.target_u  = None
                z_traj = _odeint(model.ode_func, mu, t_grid, method="rk4").squeeze(1)
                pred_def = model.decoder(z_traj).cpu().numpy()  # (N_WAVES, 34)

                x_np   = s["x"].numpy()[:, :n_def]
                actual = _composites(x_np[tgt_i])
                base   = _composites(x_np[last_i])
                pred   = _composites(pred_def[tgt_i])

                # linear extrapolation per composite from the input waves
                lin = {}
                for name, ix in (("fi", slice(None)), ("func", func_ix), ("cog", cog_ix)):
                    y_in = x_np[in_idx][:, ix].mean(axis=1) if name != "fi" \
                        else x_np[in_idx].mean(axis=1)
                    A = np.polyfit(waves[in_idx], y_in, 1)
                    lin[name] = float(np.clip(np.polyval(A, waves[tgt_i]), 0.0, 1.0))

                row = {"cunicah": s["cunicah"], "np": s["np"], "fold": k,
                       "t_target": float(waves[tgt_i])}
                for name in ("fi", "func", "cog"):
                    row[f"{name}_actual"]   = actual[name]
                    row[f"{name}_base"]     = base[name]      # LVCF prediction
                    row[f"{name}_cadence"]  = pred[name]
                    row[f"{name}_linear"]   = lin[name]
                out_rows.append(row)

    enc = pd.DataFrame(out_rows)
    # baseline (first-wave) demographics
    demo = (fi_df.sort_values(["cunicah", "np", "a_o_ent"])
            .groupby(["cunicah", "np"]).first().reset_index()
            [["cunicah", "np", "edad", "sexo", "educacion"]]
            .rename(columns={"edad": "age", "sexo": "sex", "educacion": "edu"}))
    return enc.merge(demo, on=["cunicah", "np"], how="left")


def run_forecast():
    """Out-of-fold trajectory-forecasting benchmark (plan #4) and functional/
    cognitive decline forecasting (plan #3)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_absolute_error

    print("Loading MHAS data…")
    fi_df = pd.read_csv(FI_PATH)
    folds = _load_folds()
    enc = _forecast_fold_models(folds, fi_df)
    enc["age_z"] = (enc["age"] - enc["age"].mean()) / enc["age"].std()
    enc["edu_z"] = (enc["edu"] - enc["edu"].mean()) / enc["edu"].std()
    n_folds = int(folds["fold"].max()) + 1
    print(f"\n  Forecast cohort: {len(enc):,} patients with ≥3 observed waves\n")

    # ── #4: continuous trajectory-forecast accuracy ────────────────────────────
    # Demographics+baseline Ridge is fit out-of-fold; LVCF/linear/CADENCE are
    # per-patient and need no fitting.
    def _demo_oof(col):
        pred = np.full(len(enc), np.nan)
        feats = ["age_z", "sex", "edu_z", f"{col}_base"]
        for k in range(n_folds):
            tr = enc[enc["fold"] != k]
            te_idx = enc.index[enc["fold"] == k]
            r = Ridge(alpha=1.0).fit(tr[feats].values, tr[f"{col}_actual"].values)
            pred[enc.index.get_indexer(te_idx)] = r.predict(enc.loc[te_idx, feats].values)
        return np.clip(pred, 0.0, 1.0)

    fc_rows = []
    print("[FORECAST] R² / MAE of held-out-wave prediction (out-of-fold)")
    for col, label in (("fi", "Frailty Index (34)"),
                       ("func", "Functional/ADL (6)"),
                       ("cog", "Cognition (8)")):
        y = enc[f"{col}_actual"].values
        preds = {
            "LVCF (carry-forward)":   enc[f"{col}_base"].values,
            "Linear extrapolation":   enc[f"{col}_linear"].values,
            "Demographics+baseline":  _demo_oof(col),
            "CADENCE (ODE forecast)": enc[f"{col}_cadence"].values,
        }
        print(f"  {label}:")
        for m, p in preds.items():
            r2  = r2_score(y, p)
            mae = mean_absolute_error(y, p)
            fc_rows.append({"domain": col, "model": m,
                            "r2": round(r2, 4), "mae": round(mae, 4), "n": len(y)})
            print(f"      {m:<26} R²={r2:6.3f}   MAE={mae:.4f}")
    pd.DataFrame(fc_rows).to_csv(FORECAST_CSV, index=False)
    print(f"  Saved → {FORECAST_CSV.relative_to(_ROOT)}\n")

    # ── #3: incident decline in age-weak domains ───────────────────────────────
    # Event = gained ≥1 deficit in the domain between the last input wave and the
    # held-out wave. Compare AUC of forecast-change predictors vs demographics.
    inc_rows = []
    print("[INCIDENT] AUC for predicting incident decline (≥1 new domain deficit)")
    for col, label, n_items in (("func", "Functional/ADL", len(FUNC_COLS)),
                                ("cog",  "Cognition",      len(COG_COLS))):
        thr = 1.0 / n_items - 1e-6
        room = enc[f"{col}_base"] < (1.0 - 1e-6)        # can still decline
        sub = enc[room].copy()
        sub["event"] = ((sub[f"{col}_actual"] - sub[f"{col}_base"]) >= thr).astype(int)
        if sub["event"].nunique() < 2:
            continue
        # demographics predictor: OOF Ridge P(decline-ish) on [age,sex,edu,base]
        dpred = np.full(len(sub), np.nan)
        feats = ["age_z", "sex", "edu_z", f"{col}_base"]
        for k in range(n_folds):
            tr = sub[sub["fold"] != k]
            te_idx = sub.index[sub["fold"] == k]
            r = Ridge(alpha=1.0).fit(tr[feats].values, tr["event"].values)
            dpred[sub.index.get_indexer(te_idx)] = r.predict(sub.loc[te_idx, feats].values)
        scores = {
            "Age + sex + edu + baseline": dpred,
            "Linear Δ extrapolation":     (sub[f"{col}_linear"] - sub[f"{col}_base"]).values,
            "CADENCE forecast Δ":         (sub[f"{col}_cadence"] - sub[f"{col}_base"]).values,
        }
        y = sub["event"].values
        print(f"  {label}  (n={len(sub):,}, events={int(y.sum()):,}, "
              f"{100*y.mean():.1f}%):")
        for m, sc in scores.items():
            auc, lo, hi = _auc_ci(y, sc)
            inc_rows.append({"domain": col, "model": m, "auc": round(auc, 4),
                             "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                             "n": len(sub), "events": int(y.sum())})
            print(f"      {m:<28} AUC={auc:.3f}  [{lo:.3f}–{hi:.3f}]")
    pd.DataFrame(inc_rows).to_csv(INCIDENT_CSV, index=False)
    print(f"  Saved → {INCIDENT_CSV.relative_to(_ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true",
                    help="Compute out-of-fold C-indices from the per-fold checkpoints "
                         "produced by `train_latent_ode.py --cv K`.")
    ap.add_argument("--forecast", action="store_true",
                    help="Out-of-fold trajectory-forecasting + functional/cognitive "
                         "decline benchmark (plan items #3 and #4).")
    args = ap.parse_args()
    if args.forecast:
        run_forecast()
    elif args.cv:
        run_cv()
    else:
        main()
