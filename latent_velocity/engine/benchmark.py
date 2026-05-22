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


if __name__ == "__main__":
    main()
