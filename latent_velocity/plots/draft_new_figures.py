"""
Draft figures for the CADENCE manuscript — kept separate until approved for paper_figures.py.

  fig_kl_dims  : Per-dimension KL divergence bar chart (supports '8/8 active' claim)
  fig_fi_vel   : Baseline FI vs latent velocity magnitude, coloured by mortality

Usage:
    python latent_velocity/plots/draft_new_figures.py
    python latent_velocity/plots/draft_new_figures.py --only kl
    python latent_velocity/plots/draft_new_figures.py --only fi_vel
"""

import sys
import argparse
import warnings
from pathlib import Path

_THIS  = Path(__file__).resolve()
_PLOTS = _THIS.parent
_ROOT  = _PLOTS.parent
sys.path.insert(0, str(_ROOT / "engine"))

import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from _paths import MODELS_DIR, DATA_DIR

FIG_DIR    = _ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LATENT_DIM = 8
FREE_BITS  = 0.5   # δ in nats

PALETTE = {
    "fast":    "#c0392b",
    "slow":    "#1f4e79",
    "accent":  "#117a65",
    "thresh":  "#e67e22",
}


def apply_style():
    mpl.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linestyle":    "--",
        "legend.fontsize":   9,
        "legend.frameon":    False,
        "figure.dpi":        110,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })


def _load_model_and_dataset():
    """Load the current Latent ODE-VAE checkpoint and the full MHAS dataset."""
    from train_latent_ode import LatentODE, LatentODEDataset, MHAS_WAVES, T_MAX

    ckpt  = torch.load(str(MODELS_DIR / "latent_ode_model.pth"),
                       map_location="cpu", weights_only=False)
    model = LatentODE()
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    dataset = LatentODEDataset(
        str(DATA_DIR / "frailty_index_data.csv"),
        edad_mean=ckpt["edad_mean"], edad_std=ckpt["edad_std"],
        edu_mean=ckpt["edu_mean"],   edu_std=ckpt["edu_std"],
    )
    t_norm = torch.tensor([w / T_MAX for w in MHAS_WAVES], dtype=torch.float32)
    return model, dataset, t_norm


# ─── KL per-dimension bar chart ─────────────────────────────────────────────

def fig_kl_dims():
    """
    Mean per-dimension KL divergence across the full cohort, sorted descending.
    Bars that exceed the free-bits threshold δ are coloured green (active);
    any that fall below are coloured red (collapsed — none expected).
    """
    model, dataset, t_norm = _load_model_and_dataset()
    print(f"  Computing KL over {len(dataset.samples):,} patients…")

    kl_accum = np.zeros(LATENT_DIM)
    with torch.no_grad():
        for s in dataset.samples:
            x    = s["x"].unsqueeze(0)
            mask = s["mask"].unsqueeze(0)
            mu, logvar = model.encode(x, t_norm.unsqueeze(0), mask)
            # KL_k = -0.5 * (1 + logvar_k - mu_k^2 - exp(logvar_k))
            kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
            kl_accum += kl.squeeze(0).numpy()
    kl_mean = kl_accum / len(dataset.samples)

    order  = np.argsort(kl_mean)[::-1]          # sort descending for readability
    kl_s   = kl_mean[order]
    labels = [f"$z_{{{i}}}$" for i in order]
    colors = [PALETTE["accent"] if v > FREE_BITS else PALETTE["fast"] for v in kl_s]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(np.arange(LATENT_DIM), kl_s, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.65)

    ax.axvline(FREE_BITS, color=PALETTE["thresh"], linewidth=1.8, linestyle="--",
               label=rf"Free-bits threshold $\delta = {FREE_BITS}$ nats")

    for bar, val in zip(bars, kl_s):
        ax.text(val + max(kl_s) * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left", fontsize=9)

    ax.set_yticks(np.arange(LATENT_DIM))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Mean KL divergence (nats)")
    ax.set_xlim(0, kl_s.max() * 1.22)
    ax.legend(loc="lower right")
    plt.tight_layout()

    out = FIG_DIR / "fig_kl_dims.png"
    plt.savefig(out)
    plt.close()
    print(f"  -> {out.relative_to(_ROOT)}")
    print(f"     Active dims (KL > {FREE_BITS}): {(kl_mean > FREE_BITS).sum()}/{LATENT_DIM}")


# ─── FI vs velocity scatter ─────────────────────────────────────────────────

def fig_fi_vel():
    """
    Baseline Frailty Index vs baseline latent aging velocity magnitude,
    coloured by eventual mortality.  KDE contours overlay each group.

    Illustrates the paper's core claim: two patients at the same FI level
    can occupy very different velocity regimes, and velocity captures
    additional mortality risk beyond FI alone.
    """
    df_fi   = pd.read_csv(DATA_DIR / "frailty_index_data.csv")
    df_traj = pd.read_csv(MODELS_DIR / "latent_velocity_trajectory_128.csv")

    v_cols = [f"v_{k}" for k in range(LATENT_DIM)]
    df_traj["v_mag"] = np.sqrt((df_traj[v_cols] ** 2).sum(axis=1))

    # Baseline values: first observed wave per patient
    fi_base = (df_fi.sort_values(["cunicah", "np", "a_o_ent"])
               .groupby(["cunicah", "np"]).first()
               .reset_index()[["cunicah", "np", "FI"]])

    v_base = (df_traj[df_traj["t"] == 0.0]
              .groupby(["cunicah", "np"])["v_mag"].first()
              .reset_index())
    if v_base.empty:
        # Fall back to the earliest time point if t=0 is absent for some patients
        v_base = (df_traj.sort_values(["cunicah", "np", "t"])
                  .groupby(["cunicah", "np"]).first()
                  .reset_index()[["cunicah", "np", "v_mag"]])

    # Mortality: ever observed as deceased
    outcome = (df_fi.groupby(["cunicah", "np"])["fallecido"]
               .max().reset_index())
    outcome["died"] = (outcome["fallecido"] == 1).astype(int)

    df = (fi_base
          .merge(v_base,   on=["cunicah", "np"])
          .merge(outcome[["cunicah", "np", "died"]], on=["cunicah", "np"]))
    df = df.dropna(subset=["FI", "v_mag"])

    # Clip top 1% of velocity (avoids sparse extreme outliers dominating the axis)
    v_cap = df["v_mag"].quantile(0.99)
    df    = df[df["v_mag"] <= v_cap].copy()

    alive = df[df["died"] == 0]
    dead  = df[df["died"] == 1]

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    # Points: alive first (behind) then deceased (on top)
    ax.scatter(alive["FI"], alive["v_mag"], c=PALETTE["slow"],
               alpha=0.15, s=9, linewidths=0,
               label=f"Survived / censored  (n={len(alive):,})")
    ax.scatter(dead["FI"],  dead["v_mag"],  c=PALETTE["fast"],
               alpha=0.40, s=13, linewidths=0,
               label=f"Deceased  (n={len(dead):,})")

    # KDE contours for each group
    fi_range  = np.linspace(df["FI"].min(),    df["FI"].max(),    150)
    vm_range  = np.linspace(df["v_mag"].min(), df["v_mag"].max(), 150)
    FI_g, VM_g = np.meshgrid(fi_range, vm_range)
    grid_pts   = np.vstack([FI_g.ravel(), VM_g.ravel()])

    for grp, color, bw in [(alive, PALETTE["slow"], 0.20), (dead, PALETTE["fast"], 0.28)]:
        if len(grp) < 60:
            continue
        kde = gaussian_kde(np.vstack([grp["FI"], grp["v_mag"]]), bw_method=bw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = kde(grid_pts).reshape(FI_g.shape)
        levels = np.quantile(Z[Z > 0], [0.50, 0.82])
        ax.contour(FI_g, VM_g, Z, levels=levels, colors=[color],
                   linewidths=[0.9, 1.6], alpha=0.80)

    # Vertical lines at FI quartiles to make within-FI-band velocity spread visible
    for q in np.quantile(df["FI"], [0.25, 0.50, 0.75]):
        ax.axvline(q, color="#aaaaaa", linewidth=0.7, linestyle=":", alpha=0.6)

    ax.set_xlabel("Baseline Frailty Index")
    ax.set_ylabel(r"Baseline latent aging velocity $\|v\|_2$")
    ax.legend(loc="upper left", markerscale=2.2)
    plt.tight_layout()

    out = FIG_DIR / "fig_fi_vel.png"
    plt.savefig(out)
    plt.close()
    print(f"  -> {out.relative_to(_ROOT)}")
    print(f"     Patients: alive={len(alive):,}  deceased={len(dead):,}")


# ─── Driver ─────────────────────────────────────────────────────────────────

FIGURES = {
    "kl":     fig_kl_dims,
    "fi_vel": fig_fi_vel,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="Comma-separated keys: kl, fi_vel. Default: all.")
    args = ap.parse_args()
    apply_style()
    keys = ([k.strip() for k in args.only.split(",") if k.strip()]
            if args.only else list(FIGURES))
    for k in keys:
        if k not in FIGURES:
            print(f"  ! unknown figure '{k}'")
            continue
        print(f"Generating '{k}':")
        FIGURES[k]()
    print(f"\nOutputs under {FIG_DIR.relative_to(_ROOT)}/")


if __name__ == "__main__":
    main()
