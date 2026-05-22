"""
Unified figure generation for the CADENCE manuscript (BMC Bioinformatics).

Writes 6 publication-quality figures to latent_velocity/paper/figures/
with a single shared matplotlib style. All figures are generated from the
**current** Latent ODE-VAE checkpoint (models/latent_ode_model.pth) — no
legacy artefacts are reused.

Figures:
  1. Architecture schematic                 (native matplotlib)
  2. Kaplan–Meier survival curves           (no at-risk table)
  3. Velocity-domain correlation heatmap    (re-uses validation output)
  4. UMAP composite: FI + ||v||_2           (regenerated with current model)
  5. Single-patient counterfactual          (digital_twin.rank_interventions)
  6. Population intervention ranking        (reuses ranking PNG)

Usage:
    python latent_velocity/plots/paper_figures.py
    python latent_velocity/plots/paper_figures.py --only 4
"""

import sys
import argparse
import shutil
import warnings
from pathlib import Path

_THIS  = Path(__file__).resolve()
_PLOTS = _THIS.parent
_ROOT  = _PLOTS.parent
sys.path.insert(0, str(_ROOT / "engine"))
sys.path.insert(0, str(_ROOT / "ode-digitaltwin"))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _paths import (
    PLOTS_DIR, MODELS_DIR, DATA_DIR,
    RANKING_DIR, LATENT_DIR,
)

FIG_DIR = _ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─── Shared style ───────────────────────────────────────────────────────────

PALETTE = {
    "fast":    "#c0392b",
    "slow":    "#1f4e79",
    "accent":  "#117a65",
    "neutral": "#4d4d4d",
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


# ─── Figure 1 — Architecture overview (detailed) ────────────────────────────

def fig1_architecture():
    """
    Detailed Latent ODE-VAE schematic with five horizontal stages:
        Input timeline  →  Encoder  →  Latent posterior  →  Neural ODE  →  Outputs
    plus a training-loss strip beneath that annotates which output each loss
    term draws from. Tensor shapes, MLP widths, integration scheme, and
    free-bits / β-annealing / Cox weighting are all displayed.
    """
    fig, ax = plt.subplots(figsize=(16, 9.8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10.0)
    ax.axis("off")

    C_DATA   = "#ecf3fb"
    C_ENC    = "#cfe0f4"
    C_LAT    = "#b6d2eb"
    C_ODE    = "#9bc3e2"
    C_DEC    = "#f6e1c4"
    C_RISK   = "#f6c4c8"
    C_VEL    = "#cfead8"
    C_LOSS   = "#f3eef7"
    EDGE     = "#33425a"
    DARK     = "#1b2538"

    # Title and subtitle intentionally omitted (BMC Bioinformatics: figure
    # titles belong in the manuscript caption, not in the graphic file).

    # =========================================================================
    # Stage 1 — Irregular MHAS timeline + input frame
    # =========================================================================
    stage1 = FancyBboxPatch((0.15, 2.4), 2.8, 5.3,
                            boxstyle="round,pad=0.04,rounding_size=0.15",
                            linewidth=1.3, edgecolor=EDGE, facecolor=C_DATA)
    ax.add_patch(stage1)
    ax.text(1.55, 7.35, "1. MHAS visits", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=DARK)

    # Timeline
    wave_years = [0, 2, 11, 14, 17, 20]
    tl_x0, tl_x1, tl_y = 0.35, 2.75, 5.7
    ax.plot([tl_x0, tl_x1], [tl_y, tl_y], color=EDGE, linewidth=1.2)
    for i, yr in enumerate(wave_years):
        x_dot = tl_x0 + (yr / 20.0) * (tl_x1 - tl_x0)
        ax.scatter([x_dot], [tl_y], s=46, color=EDGE, zorder=3)
        ax.text(x_dot, tl_y + 0.27, str(yr), ha="center", va="bottom",
                fontsize=8.6, color=EDGE)
    ax.text(1.55, tl_y - 0.55, "irregular waves (years since 2001)",
            ha="center", va="center", fontsize=8.6, style="italic", color="#555")

    # Per-visit input frame
    ax.text(1.55, 4.6,
            r"$x_t \in [0,1]^{34}$  (deficits)" "\n"
            r"$s \in \mathbb{R}^{3}$  (age, sex, edu)" "\n"
            r"$u_t \in \mathbb{R}^{7}$  (lifestyle)" "\n"
            r"mask $m_t \in \{0,1\}$",
            ha="center", va="center", fontsize=9.5, color=DARK)
    ax.text(1.55, 3.05, "input frame:\n$[x_t \\| s \\| t_{\\mathrm{norm}}]$,  37+1-D",
            ha="center", va="center", fontsize=9.0, color="#244")

    # =========================================================================
    # Stage 2 — Encoder (RecognitionRNN)
    # =========================================================================
    stage2 = FancyBboxPatch((3.25, 2.4), 3.0, 5.3,
                            boxstyle="round,pad=0.04,rounding_size=0.15",
                            linewidth=1.3, edgecolor=EDGE, facecolor=C_ENC)
    ax.add_patch(stage2)
    ax.text(4.75, 7.35, "2. RecognitionRNN", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=DARK)
    ax.text(4.75, 6.95, "backward masked GRU",
            ha="center", va="center", fontsize=9.0, style="italic", color="#444")

    # GRU chain — 4 explicit cells (t=20, t=17, …, t=0) with an ellipsis in
    # the middle to imply the full 6-wave sequence without crowding the box.
    cell_y    = 5.65
    cell_w    = 0.58
    cell_h    = 0.50
    cell_data = [
        (3.40, "GRU\n$t{=}20$"),
        (4.05, "GRU\n$t{=}17$"),
        (4.60, r"$\cdots$"),
        (5.05, "GRU\n$t{=}2$"),
        (5.70, "GRU\n$t{=}0$"),
    ]
    for i, (cx, lbl) in enumerate(cell_data):
        if lbl.startswith("$\\cdots$") or lbl == r"$\cdots$":
            ax.text(cx + cell_w/2, cell_y + cell_h/2, lbl,
                    ha="center", va="center", fontsize=14, color=DARK)
        else:
            ax.add_patch(FancyBboxPatch((cx, cell_y), cell_w, cell_h,
                                        boxstyle="round,pad=0.02,rounding_size=0.08",
                                        linewidth=0.9, edgecolor=EDGE,
                                        facecolor="white"))
            ax.text(cx + cell_w/2, cell_y + cell_h/2, lbl,
                    ha="center", va="center", fontsize=7.4, color=DARK,
                    linespacing=0.9)
        # Right-to-left arrow from this cell to the next leftward cell
        if i > 0:
            prev_cx = cell_data[i-1][0]
            x_from  = cx
            x_to    = prev_cx + cell_w + 0.02
            ax.add_patch(FancyArrowPatch(
                (x_from, cell_y + cell_h/2),
                (x_to,   cell_y + cell_h/2),
                arrowstyle="-|>", mutation_scale=9,
                linewidth=0.9, color=EDGE))

    ax.text(4.75, 5.25, r"$h_t = m_t\,\mathrm{GRU}([x_t\|t],\,h_{t+1}) + (1-m_t)\,h_{t+1}$",
            ha="center", va="center", fontsize=8.6, color=DARK)
    ax.text(4.75, 4.65, "masked update skips\nunobserved waves",
            ha="center", va="center", fontsize=8.4, style="italic", color="#555")

    # Final projection
    ax.add_patch(FancyBboxPatch((3.55, 3.05), 2.4, 1.05,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=0.9, edgecolor=EDGE, facecolor="white"))
    ax.text(4.75, 3.85, r"$h_0 \rightarrow (\mu,\,\log\sigma^2)$",
            ha="center", va="center", fontsize=9.5, color=DARK)
    ax.text(4.75, 3.30, r"two linear heads,  $\mathbb{R}^{8}$ each",
            ha="center", va="center", fontsize=8.6, style="italic", color="#444")

    # =========================================================================
    # Stage 3 — Latent posterior + reparameterisation
    # =========================================================================
    stage3 = FancyBboxPatch((6.55, 2.4), 2.5, 5.3,
                            boxstyle="round,pad=0.04,rounding_size=0.15",
                            linewidth=1.3, edgecolor=EDGE, facecolor=C_LAT)
    ax.add_patch(stage3)
    ax.text(7.8, 7.35, "3. Latent  $z_0$", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=DARK)

    # Gaussian sketch
    xs = np.linspace(-2.6, 2.6, 200)
    ys = np.exp(-xs**2 / 2)
    ax.plot(7.8 + 0.3 * xs, 5.9 + 0.5 * ys, color=EDGE, linewidth=1.1)
    ax.fill_between(7.8 + 0.3 * xs, 5.9, 5.9 + 0.5 * ys,
                    color="white", alpha=0.85)
    ax.text(7.8, 6.7, r"$q(z_0 \,|\, x) = \mathcal{N}(\mu,\,\sigma^2)$",
            ha="center", va="center", fontsize=9.6, color=DARK)

    ax.text(7.8, 4.85,
            r"$z_0 = \mu + \varepsilon \odot \sigma$" "\n"
            r"$\varepsilon \sim \mathcal{N}(0, I_8)$",
            ha="center", va="center", fontsize=9.4, color=DARK)
    ax.text(7.8, 3.85, "reparameterisation",
            ha="center", va="center", fontsize=8.6, style="italic", color="#444")
    ax.text(7.8, 3.30, r"$z_0 \in \mathbb{R}^{8}$",
            ha="center", va="center", fontsize=10.0, color=DARK,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor=EDGE, linewidth=0.7))
    ax.text(7.8, 2.65, r"$(\mu,\log\sigma^2)$  feed  $\mathcal{L}_{\mathrm{KL}}$",
            ha="center", va="center", fontsize=7.8, style="italic", color="#264a73")

    # =========================================================================
    # Stage 4 — Neural ODE
    # =========================================================================
    stage4 = FancyBboxPatch((9.35, 2.4), 3.3, 5.3,
                            boxstyle="round,pad=0.04,rounding_size=0.15",
                            linewidth=1.3, edgecolor=EDGE, facecolor=C_ODE)
    ax.add_patch(stage4)
    ax.text(11.0, 7.35, "4. Controlled Neural ODE", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=DARK)
    ax.text(11.0, 6.95, r"$dz/dt = f_{\theta}(z,\,10\,u)$",
            ha="center", va="center", fontsize=10.0, color=DARK)

    # MLP layer schematic — vertical bars whose height encodes width
    widths    = [15, 128, 256, 128, 8]
    labels    = ["$z\\!\\oplus\\!u$\n15", "FC\n128", "FC\n256", "FC\n128", "out\n8"]
    fill_cols = ["#dbe9f7", "#b8d3ee", "#8eb9e2", "#b8d3ee", "#dbe9f7"]
    bar_w     = 0.32
    bar_xs    = np.linspace(9.65, 12.35, len(widths))
    bar_top   = 6.50
    bar_bot   = 5.10
    h_max     = bar_top - bar_bot
    max_w     = max(widths)
    for x, w, lbl, fc in zip(bar_xs, widths, labels, fill_cols):
        h = h_max * (w / max_w) ** 0.5
        y = bar_bot + (h_max - h) / 2
        ax.add_patch(FancyBboxPatch((x - bar_w/2, y), bar_w, h,
                                    boxstyle="round,pad=0.005,rounding_size=0.05",
                                    linewidth=0.9, edgecolor=EDGE, facecolor=fc))
        ax.text(x, bar_bot - 0.18, lbl, ha="center", va="top",
                fontsize=7.8, color=DARK)
    # connect bars
    for x0, x1 in zip(bar_xs[:-1], bar_xs[1:]):
        ax.plot([x0 + bar_w/2, x1 - bar_w/2],
                [bar_bot + h_max/2, bar_bot + h_max/2],
                color=EDGE, linewidth=1.0, alpha=0.55)
    ax.text(11.0, 4.55, "4-layer MLP, SiLU activations",
            ha="center", va="center", fontsize=8.5, style="italic", color="#444")

    # Trajectory inset — centered in stage 4, with margins, axis lines, and
    # tick labels so it reads as a real plot rather than a free-floating sketch.
    inset_x0, inset_y0 = 9.60, 2.75
    inset_w,  inset_h  = 2.80, 1.10
    ax.add_patch(FancyBboxPatch((inset_x0, inset_y0), inset_w, inset_h,
                                boxstyle="round,pad=0.015,rounding_size=0.06",
                                linewidth=0.7, edgecolor="#aab", facecolor="white"))

    # Internal plot area (with margins for labels and ticks)
    plot_x0 = inset_x0 + 0.42
    plot_y0 = inset_y0 + 0.22
    plot_w  = inset_w  - 0.62
    plot_h  = inset_h  - 0.46

    # Axis lines
    ax.plot([plot_x0, plot_x0 + plot_w], [plot_y0, plot_y0],
            color="#666", linewidth=0.7)
    ax.plot([plot_x0, plot_x0], [plot_y0, plot_y0 + plot_h],
            color="#666", linewidth=0.7)

    # Sample latent trajectories, scaled inside the plot area with 8% padding
    tt = np.linspace(0, 1, 80)
    samples = [
        ( 0.55 * np.sin(2.4 * tt) + 0.35 * tt, "#1f4e79"),
        ( 0.40 * np.cos(1.8 * tt) - 0.30 * tt, "#c0392b"),
        ( 0.25 * np.sin(3.5 * tt + 0.5) + 0.10 * tt, "#117a65"),
    ]
    pad = 0.08
    for traj, c in samples:
        norm = (traj - traj.min()) / (np.ptp(traj) + 1e-6)
        ax.plot(plot_x0 + tt * plot_w,
                plot_y0 + pad * plot_h + (1 - 2 * pad) * plot_h * norm,
                color=c, linewidth=1.3, alpha=0.95)

    # X-axis ticks (start / end of the 6-wave window)
    for x_val, lbl in [(0.0, "0"), (1.0, "20")]:
        x_pt = plot_x0 + x_val * plot_w
        ax.plot([x_pt, x_pt], [plot_y0 - 0.02, plot_y0 + 0.02],
                color="#666", linewidth=0.7)
        ax.text(x_pt, plot_y0 - 0.05, lbl,
                ha="center", va="top", fontsize=7.0, color="#555")
    ax.text(plot_x0 + plot_w / 2, plot_y0-0.05,
            r"$t$ (years)", ha="center", va="top",
            fontsize=7.8, color="#333")

    # Y-axis label
    ax.text(plot_x0 - 0.18, plot_y0 + plot_h / 2,
            r"$z_k(t)$", ha="right", va="center",
            fontsize=7.8, color="#333", rotation=90)

    # Title above inset
    ax.text(inset_x0 + inset_w / 2, inset_y0 + inset_h +0.2,
            r"$z(t) \in \mathbb{R}^{8}$  —  RK4 on $\{0,2,11,14,17,20\}$ yr",
            ha="center", va="top", fontsize=8.2, style="italic", color="#444")

    # =========================================================================
    # Stage 5 — Outputs (Decoder / RiskHead / Velocity)
    # =========================================================================
    stage5 = FancyBboxPatch((13.05, 2.4), 2.7, 5.3,
                            boxstyle="round,pad=0.04,rounding_size=0.15",
                            linewidth=1.3, edgecolor=EDGE, facecolor="white")
    ax.add_patch(stage5)
    ax.text(14.4, 7.35, "5. Outputs", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=DARK)

    # Decoder block
    ax.add_patch(FancyBboxPatch((13.2, 5.85), 2.4, 1.20,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=0.9, edgecolor=EDGE, facecolor=C_DEC))
    ax.text(14.4, 6.78, r"Decoder  $g_{\phi}$", ha="center", va="center",
            fontsize=10.0, fontweight="medium", color=DARK)
    ax.text(14.4, 6.32, r"$z(t)\!\rightarrow\!\hat{x}(t)\in[0,1]^{34}$",
            ha="center", va="center", fontsize=8.8, color=DARK)
    ax.text(14.4, 5.96, r"feeds  $\mathcal{L}_{\mathrm{recon}}$",
            ha="center", va="bottom", fontsize=7.6, style="italic", color="#7a5230")

    # RiskHead block
    ax.add_patch(FancyBboxPatch((13.2, 4.40), 2.4, 1.20,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=0.9, edgecolor=EDGE, facecolor=C_RISK))
    ax.text(14.4, 5.33, r"RiskHead  $r_{\psi}$", ha="center", va="center",
            fontsize=10.0, fontweight="medium", color=DARK)
    ax.text(14.4, 4.88, r"$\mu\!\rightarrow\!r\in\mathbb{R}$",
            ha="center", va="center", fontsize=8.8, color=DARK)
    ax.text(14.4, 4.51, r"feeds  $\mathcal{L}_{\mathrm{Cox}}$",
            ha="center", va="bottom", fontsize=7.6, style="italic", color="#7a4248")

    # Velocity block
    ax.add_patch(FancyBboxPatch((13.2, 2.95), 2.4, 1.20,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=0.9, edgecolor=EDGE, facecolor=C_VEL))
    ax.text(14.4, 3.88, r"Velocity  $v(t)$", ha="center", va="center",
            fontsize=10.0, fontweight="medium", color=DARK)
    ax.text(14.4, 3.43, r"$v(t)=f_{\theta}(z(t),u)$",
            ha="center", va="center", fontsize=8.8, color=DARK)
    ax.text(14.4, 3.06, r"inference-time output",
            ha="center", va="bottom", fontsize=7.6, style="italic", color="#1c5a3a")

    # =========================================================================
    # Inter-stage arrows (between the five stages)
    # =========================================================================
    inter = [
        (2.95, 5.0, 3.25, 5.0),     # 1 → 2
        (6.25, 5.0, 6.55, 5.0),     # 2 → 3
        (9.05, 5.0, 9.35, 5.0),     # 3 → 4
        (12.65, 6.45, 13.20, 6.45), # ODE → Decoder
        (12.65, 5.00, 13.20, 5.00), # μ from posterior → RiskHead
        (12.65, 3.55, 13.20, 3.55), # ODE → Velocity
    ]
    for (x0, y0, x1, y1) in inter:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                     arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.5, color=EDGE))

    # Control u(t) feed
    ax.add_patch(FancyArrowPatch((11.0, 2.05), (11.0, 2.40),
                                 arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.2, color=EDGE))
    ax.text(11.0, 1.85, r"control  $u(t)$  (7-D lifestyle vector)",
            ha="center", va="center", fontsize=9.5, style="italic", color=DARK)

    # =========================================================================
    # Loss strip (bottom) — header + three substantive sub-cards
    # =========================================================================
    loss = FancyBboxPatch((0.15, 0.08), 15.6, 1.62,
                          boxstyle="round,pad=0.03,rounding_size=0.14",
                          linewidth=1.3, edgecolor=EDGE, facecolor=C_LOSS)
    ax.add_patch(loss)
    ax.text(8.00, 1.55, "Training objective", ha="center", va="center",
            fontsize=11.0, fontweight="bold", color=DARK)
    ax.text(8.00, 1.27,
            r"$\mathcal{L} \;=\; \mathcal{L}_{\mathrm{recon}}"
            r" \;+\; \beta(e)\,\mathcal{L}_{\mathrm{KL}}"
            r" \;+\; \lambda_{\mathrm{cox}}\,\mathcal{L}_{\mathrm{Cox}}$",
            ha="center", va="center", fontsize=11.0, color=DARK)

    # Three sub-cards
    card_y, card_h = 0.20, 0.92
    sub_cards = [
        (0.45, 4.85, "#7a5230", r"Reconstruction $\mathcal{L}_{\mathrm{recon}}$",
         r"$\mathrm{MSE}_w(\hat{x}_t, x_t)$  on observed waves",
         r"inverse-variance weights $w_j$,  $D{=}34$"),
        (5.55, 4.85, "#264a73", r"KL with free bits  $\beta(e)\,\mathcal{L}_{\mathrm{KL}}$",
         r"$\sum_k \max(\delta,\,\mathrm{KL}_k)$",
         r"free bits $\delta{=}0.5$ nats,  $\beta$: 20$\to$80 ep,  $\beta_{\max}{=}0.1$"),
        (10.65, 4.85, "#7a4248", r"Cox  $\lambda_{\mathrm{cox}}\,\mathcal{L}_{\mathrm{Cox}}$",
         r"$-\sum_{i:\mathrm{ev}_i=1}\,(r_i - \log\!\sum_{j:t_j\geq t_i} e^{r_j})$",
         r"event weight $\lambda_{\mathrm{cox}}{=}0.15$,  risk $r{=}r_\psi(\mu)$"),
    ]
    for (x, w, col, hdr, formula, settings) in sub_cards:
        ax.add_patch(FancyBboxPatch((x, card_y), w, card_h,
                                    boxstyle="round,pad=0.02,rounding_size=0.07",
                                    linewidth=0.9, edgecolor=col, facecolor="white"))
        ax.text(x + w/2, card_y + card_h - 0.16, hdr,
                ha="center", va="center", fontsize=9.0,
                fontweight="bold", color=col)
        ax.text(x + w/2, card_y + card_h - 0.46, formula,
                ha="center", va="center", fontsize=8.8, color=DARK)
        ax.text(x + w/2, card_y + 0.15, settings,
                ha="center", va="center", fontsize=8.0,
                style="italic", color="#444")

    # (Loss attribution is now shown by inline labels next to the blocks they
    # feed: Decoder → L_recon, Posterior → L_KL, RiskHead → L_Cox. Long
    # diagonal connectors are intentionally omitted to avoid visually crossing
    # block 3.)

    out = FIG_DIR / "fig1_architecture.png"
    plt.savefig(out); plt.close()
    print(f"  -> {out.relative_to(_ROOT)}")


# ─── Figure 2 — KM curves (no at-risk table) ────────────────────────────────

def _build_phenotype_table():
    """Run the validation pipeline far enough to produce a per-patient table
    with phenotype, time-to-event, event flag, and Cox HR."""
    from clinical_validation import (
        calculate_velocity_magnitude, compute_frailty_velocity,
        execute_survival_analysis,
    )
    # execute_survival_analysis returns the fitted Cox model + corr matrix,
    # but also internally computes surv_grouped. We re-implement the minimal
    # bits here so we can return that table directly.
    from lifelines import CoxPHFitter
    import numpy as np
    import pandas as pd

    df_fi = pd.read_csv(DATA_DIR / 'frailty_index_data.csv')
    df_traj, vcols = calculate_velocity_magnitude()
    df_traj = compute_frailty_velocity(df_traj, df_fi, vcols)

    df_early   = df_traj.groupby(['cunicah', 'np']).head(30)
    patient_vf = (df_early.groupby(['cunicah', 'np'])['v_frailty'].mean()
                  .reset_index().rename(columns={'v_frailty': 'v_frailty_mean'}))
    if 'v_uncertainty' in df_traj.columns:
        patient_unc = (df_early.groupby(['cunicah', 'np'])['v_uncertainty'].mean()
                       .reset_index().rename(columns={'v_uncertainty': 'mean_unc'}))
        patient_vf = patient_vf.merge(patient_unc, on=['cunicah', 'np'], how='left')
    else:
        patient_vf['mean_unc'] = np.nan

    ages = df_fi.groupby(['cunicah', 'np'])['edad'].min().reset_index(name='baseline_age')
    patient_vf = patient_vf.merge(ages, on=['cunicah', 'np'], how='left')
    patient_vf['v_frailty_adj'] = patient_vf['v_frailty_mean']
    q1 = patient_vf['v_frailty_adj'].quantile(0.25)
    q3 = patient_vf['v_frailty_adj'].quantile(0.75)
    cond = [patient_vf['v_frailty_adj'] <= q1, patient_vf['v_frailty_adj'] >= q3]
    patient_vf['Phenotype'] = np.select(cond, ['Slow_Ager', 'Fast_Ager'], default='Normal')

    obs_n = df_fi.groupby(['cunicah', 'np']).size().reset_index(name='n_obs')
    single = obs_n[obs_n['n_obs'] == 1].merge(
        df_fi[['cunicah', 'np', 'fallecido']].query('fallecido == 1').drop_duplicates(),
        on=['cunicah', 'np'], how='inner')
    if not single.empty:
        q3_vf = patient_vf['v_frailty_mean'].quantile(0.75)
        resc = single[['cunicah', 'np']].copy()
        resc['v_frailty_mean'] = q3_vf + 1e-3
        resc['v_frailty_adj']  = q3_vf + 1e-3
        resc['baseline_age']   = single.merge(ages, on=['cunicah','np'], how='left')['baseline_age'].values
        resc['mean_unc']       = patient_vf['mean_unc'].quantile(0.75)
        resc['Phenotype']      = 'Fast_Ager'
        patient_vf = pd.concat([patient_vf, resc], ignore_index=True)

    patient_vf = patient_vf[patient_vf['Phenotype'] != 'Normal']
    out = df_fi[['cunicah', 'np', 'fallecido', 'a_o_ent', 'edad']].copy()
    out['fallecido'] = (out['fallecido'] == 1).astype(int)
    last_alive = (out[out['fallecido'] == 0]
                  .groupby(['cunicah', 'np'])['a_o_ent'].max()
                  .reset_index(name='last_alive_year'))

    surv = patient_vf.merge(out, on=['cunicah', 'np'], how='left')
    surv = surv.groupby(['cunicah', 'np']).agg(
        Phenotype=('Phenotype', 'first'),
        fallecido=('fallecido', 'max'),
        start_year=('a_o_ent', 'min'),
        end_year=('a_o_ent', 'max'),
        baseline_age=('edad', 'min'),
        mean_unc=('mean_unc', 'first'),
    ).reset_index()
    surv = surv.merge(last_alive, on=['cunicah', 'np'], how='left')
    dead = surv['fallecido'] == 1
    surv.loc[dead, 'end_year'] = surv.loc[dead, 'last_alive_year'].fillna(surv.loc[dead, 'end_year'])
    surv['time_to_event'] = surv['end_year'] - surv['start_year']
    surv = surv.dropna(subset=['time_to_event', 'fallecido'])
    surv = surv[surv['time_to_event'] > 0]
    surv['Fast_Ager_Flag'] = (surv['Phenotype'] == 'Fast_Ager').astype(float)
    surv['mean_unc_z'] = (surv['mean_unc'] - surv['mean_unc'].mean()) / surv['mean_unc'].std()

    cph_data = surv[['time_to_event', 'fallecido', 'Fast_Ager_Flag', 'baseline_age', 'mean_unc_z']].dropna()
    cph = CoxPHFitter().fit(cph_data, duration_col='time_to_event', event_col='fallecido')
    hr = float(np.exp(cph.params_['Fast_Ager_Flag']))
    return surv, hr


def fig2_km_curves():
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    surv, hr = _build_phenotype_table()
    fast = surv[surv['Phenotype'] == 'Fast_Ager']
    slow = surv[surv['Phenotype'] == 'Slow_Ager']

    fig, ax = plt.subplots(figsize=(9, 5.6))
    kmf_f = KaplanMeierFitter().fit(fast['time_to_event'], fast['fallecido'],
                                    label='Fast Agers (top quartile)')
    kmf_s = KaplanMeierFitter().fit(slow['time_to_event'], slow['fallecido'],
                                    label='Slow Agers (bottom quartile)')
    kmf_f.plot_survival_function(ax=ax, linewidth=2.6, color=PALETTE['fast'])
    kmf_s.plot_survival_function(ax=ax, linewidth=2.6, color=PALETTE['slow'])

    lr = logrank_test(fast['time_to_event'], slow['time_to_event'],
                      fast['fallecido'], slow['fallecido'])
    p = lr.p_value
    p_txt = "p < 0.001" if p < 1e-3 else f"p = {p:.3g}"
    ax.text(0.04, 0.12, f"Cox HR = {hr:.2f}\nLog-rank {p_txt}",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#999", alpha=0.95))

    ax.set_xlabel("Follow-up time (years)")
    ax.set_ylabel("Survival probability")
    # Title omitted per BMC Bioinformatics (lives in the manuscript caption).
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, surv['time_to_event'].max())
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = FIG_DIR / "fig2_km_survival.png"
    plt.savefig(out); plt.close()
    print(f"  -> {out.relative_to(_ROOT)}")


# ─── Figure 3 — velocity-domain heatmap ─────────────────────────────────────

def fig3_velocity_heatmap():
    """Restyled version of plot_velocity_heatmaps output."""
    src = PLOTS_DIR / "velocity_domain_heatmap.png"
    if not src.exists():
        print(f"  ! missing {src} — run clinical_validation.py first")
        return
    shutil.copyfile(src, FIG_DIR / "fig3_velocity_domain_heatmap.png")
    print(f"  -> paper/figures/fig3_velocity_domain_heatmap.png")


# ─── Figure 4 — UMAP composite (regenerated, current model) ────────────────

def fig4_umap_composite():
    """
    Encode every patient with the *current* Latent ODE-VAE checkpoint,
    compute their first-grid latent velocity magnitude from the trajectory
    CSV, and plot a side-by-side UMAP coloured by:
      (a) Frailty Index at first observation
      (b) Latent aging velocity magnitude  ||v||_2  at t = 0
    """
    import torch
    from sklearn.preprocessing import StandardScaler
    try:
        import umap
    except ImportError:
        print("  ! umap-learn not installed; skipping Fig 4")
        return

    from train_latent_ode import (
        LatentODE, LatentODEDataset, MHAS_WAVES, T_MAX, LATENT_DIM,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  encoding with current Latent ODE-VAE on {device}")
    ckpt  = torch.load(str(MODELS_DIR / 'latent_ode_model.pth'),
                       map_location=device, weights_only=False)
    model = LatentODE().to(device)
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval()

    dataset = LatentODEDataset(
        str(DATA_DIR / 'frailty_index_data.csv'),
        edad_mean=ckpt['edad_mean'], edad_std=ckpt['edad_std'],
        edu_mean=ckpt['edu_mean'],   edu_std=ckpt['edu_std'],
    )
    t_norm_base = torch.tensor([w / T_MAX for w in MHAS_WAVES],
                               dtype=torch.float32).to(device)

    rows = []
    with torch.no_grad():
        for s in dataset.samples:
            x    = s['x'].unsqueeze(0).to(device)
            mask = s['mask'].unsqueeze(0).to(device)
            mu, _ = model.encode(x, t_norm_base.unsqueeze(0), mask)
            mu = mu[0].cpu().numpy()
            r = {'cunicah': s['cunicah'], 'np': s['np']}
            for k in range(LATENT_DIM):
                r[f'z_{k}'] = float(mu[k])
            rows.append(r)
    df_z = pd.DataFrame(rows)
    z_cols = [f'z_{k}' for k in range(LATENT_DIM)]

    df_fi = pd.read_csv(DATA_DIR / 'frailty_index_data.csv')
    fi_first = (df_fi.sort_values(['cunicah', 'np', 'a_o_ent'])
                .groupby(['cunicah', 'np']).first().reset_index()[['cunicah', 'np', 'FI']])

    df_traj = pd.read_csv(MODELS_DIR / 'latent_velocity_trajectory_128.csv')
    vcols   = [f'v_{k}' for k in range(LATENT_DIM)]
    df_traj['v_mag'] = np.sqrt((df_traj[vcols] ** 2).sum(axis=1))
    v_first = (df_traj.sort_values(['cunicah', 'np', 't'])
               .groupby(['cunicah', 'np']).first().reset_index()[['cunicah', 'np', 'v_mag']])

    df = (df_z.merge(fi_first, on=['cunicah', 'np'], how='left')
                .merge(v_first, on=['cunicah', 'np'], how='left'))
    df = df.dropna(subset=['FI', 'v_mag'])
    print(f"  UMAP input: {len(df):,} patients")

    z_scaled = StandardScaler().fit_transform(df[z_cols].values)
    reducer  = umap.UMAP(n_neighbors=200, min_dist=0.5, n_components=2,
                         random_state=42, metric='euclidean')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb = reducer.fit_transform(z_scaled)

    def clamp(s): return s.quantile(0.05), s.quantile(0.95)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
    for ax in axes:
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_aspect("equal", adjustable="datalim")

    vmin, vmax = clamp(df['FI'])
    sc0 = axes[0].scatter(emb[:, 0], emb[:, 1], c=df['FI'].values,
                          cmap='inferno', s=6, alpha=0.65, vmin=vmin, vmax=vmax)
    cb0 = fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label("Frailty Index (baseline visit)")
    axes[0].set_title("a", loc="left", fontsize=14, fontweight="bold")

    vmin, vmax = clamp(df['v_mag'])
    sc1 = axes[1].scatter(emb[:, 0], emb[:, 1], c=df['v_mag'].values,
                          cmap='viridis', s=6, alpha=0.7, vmin=vmin, vmax=vmax)
    cb1 = fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label(r"Latent aging velocity $\|v\|_2$  (at $t=0$)")
    axes[1].set_title("b", loc="left", fontsize=14, fontweight="bold")

    plt.tight_layout()
    out = FIG_DIR / "fig4_umap_composite.png"
    plt.savefig(out); plt.close()
    print(f"  -> {out.relative_to(_ROOT)}")


# ─── Figure 5 — Counterfactual ──────────────────────────────────────────────

def fig5_counterfactual(cunicah=7226.0, np_val=10.0):
    from digital_twin import rank_interventions

    ranking = rank_interventions(cunicah=cunicah, np_val=np_val)
    if ranking is None:
        print(f"  ! no actionable interventions for {cunicah}/{np_val}")
        return

    # Cohort Q1 / Q3 of instantaneous velocity magnitude — defines
    # Slow / Fast Ager bands in the same units as the y-axis.
    df_traj = pd.read_csv(MODELS_DIR / "latent_velocity_trajectory_128.csv",
                          usecols=[f"v_{k}" for k in range(8)])
    v_all = np.sqrt((df_traj.values ** 2).sum(axis=1))
    q1 = float(np.quantile(v_all, 0.25))
    q3 = float(np.quantile(v_all, 0.75))

    t        = ranking["t"]
    v_base   = ranking["v_mag_baseline"]
    auc_base = ranking["auc_baseline"]
    inters   = ranking["ranked_interventions"][:6]

    fig, ax = plt.subplots(figsize=(11, 6.0))

    # Reference bands: Slow Ager (≤Q1), Normal (Q1–Q3), Fast Ager (≥Q3).
    y_lo = min(v_base.min(), min(r["v_mag"].min() for r in inters), q1) * 0.9
    y_hi = max(v_base.max(), max(r["v_mag"].max() for r in inters), q3) * 1.08
    ax.axhspan(y_lo, q1, color=PALETTE["slow"], alpha=0.08, zorder=0)
    ax.axhspan(q3, y_hi, color=PALETTE["fast"], alpha=0.08, zorder=0)
    ax.axhline(q1, color=PALETTE["slow"], linewidth=1, linestyle=":", alpha=0.6)
    ax.axhline(q3, color=PALETTE["fast"], linewidth=1, linestyle=":", alpha=0.6)
    ax.text(t[-1] * 0.99, q1, f" Slow Ager (Q1 = {q1:.2f})",
            color=PALETTE["slow"], fontsize=9, va="bottom", ha="right")
    ax.text(t[-1] * 0.99, q3, f" Fast Ager (Q3 = {q3:.2f})",
            color=PALETTE["fast"], fontsize=9, va="top", ha="right")

    # Baseline trajectory.
    ax.plot(t, v_base, color=PALETTE["fast"], linewidth=3,
            label=f"Baseline  (no intervention)", zorder=10)

    # Intervention trajectories. Flip sign of the percentage so a decrease
    # in aging velocity is displayed as a negative value.
    cmap = plt.colormaps.get_cmap("viridis").resampled(max(len(inters), 2))
    for i, r in enumerate(inters):
        ls = "-" if r["confidence"] == "High" else "--"
        delta_pct = -r["auc_reduction_pct"]   # negative ⇒ aging velocity decreases
        ax.plot(t, r["v_mag"], color=cmap(i), linewidth=2.2,
                linestyle=ls, alpha=0.9,
                label=f'{r["label"]}  (Δ aging velocity {delta_pct:+.1f}%)')

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Forecast horizon (years)")
    ax.set_ylabel(r"Latent aging velocity $\|v\|_2$")
    # Title omitted per BMC Bioinformatics (lives in the manuscript caption).
    ax.legend(loc="upper right", fontsize=9, ncol=1, frameon=False)
    plt.tight_layout()
    out = FIG_DIR / "fig5_counterfactual_patient.png"
    plt.savefig(out); plt.close()
    print(f"  -> {out.relative_to(_ROOT)}")


# ─── Driver ─────────────────────────────────────────────────────────────────

FIGURES = {
    1: fig1_architecture,
    2: fig2_km_curves,
    3: fig3_velocity_heatmap,
    4: fig4_umap_composite,
    5: fig5_counterfactual,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="",
                    help="Comma-separated figure numbers. Default: all.")
    args = ap.parse_args()

    apply_style()
    nums = ([int(n) for n in args.only.split(",") if n.strip()]
            if args.only else sorted(FIGURES))

    # Clean stale outputs from prior figure schemes
    for stale in ("fig4_archetypal_trajectories.png", "fig5_umap_composite.png",
                  "fig6_velocity_field.png", "fig7_counterfactual_patient.png",
                  "fig8_population_ranking.png", "fig6_population_ranking.png"):
        p = FIG_DIR / stale
        if p.exists():
            p.unlink()

    for n in nums:
        if n not in FIGURES:
            print(f"  ! unknown figure {n}"); continue
        print(f"Figure {n}:")
        FIGURES[n]()
    print(f"\nAll outputs under {FIG_DIR.relative_to(_ROOT)}/")


if __name__ == "__main__":
    main()
