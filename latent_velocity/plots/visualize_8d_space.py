"""
visualize_8d_space.py
=====================
Three complementary views of the 8-dimensional LAVA latent space designed
for clinicians and reviewers.

  1. Radar / Spider Charts
     -- Per-patient 8D polygon showing position + velocity magnitude.
        Overlay healthy vs. fast-ager phenotypes.

  2. Parallel Coordinates Plot
     -- All patients as poly-lines across z0..z7, coloured by total
        velocity magnitude, revealing the "highways" of terminal decline.

  3. Interactive 3D UMAP Projection (Plotly)
     -- UMAP 8D → 3D, coloured by velocity magnitude.  Saved as
        interactive HTML so reviewers can rotate the manifold.

Run from any directory:
    python latent_velocity/plots/visualize_8d_space.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "engine"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import warnings

from _paths import DATA_DIR, MODELS_DIR, PLOTS_DIR, LATENT_DIR

# ---------------------------------------------------------------------------
# Shared labels for the 8 latent dimensions
# ---------------------------------------------------------------------------
DIM_LABELS = [
    "z0\nBiological",
    "z1\nMental",
    "z2\nReserve",
    "z3\nPhysical",
    "z4\nMetabolic",
    "z5\nFunctional",
    "z6\nAdaptive",
    "z7\nCognitive",
]

# ============================================================================
# DATA LOADING
# ============================================================================

def load_trajectory_data():
    """Load the full GP-smoothed trajectory CSV."""
    traj_file = MODELS_DIR / "latent_velocity_trajectory.csv"
    print(f"Loading trajectory data from {traj_file} …")
    df = pd.read_csv(traj_file)
    df = df.sort_values(["cunicah", "np", "t"])

    z_cols = [f"z_mean_{k}" for k in range(8)]
    v_cols = [f"v_{k}" for k in range(8)]

    # Total velocity magnitude per row
    df["speed"] = np.sqrt((df[v_cols].values ** 2).sum(axis=1))
    return df, z_cols, v_cols


def load_mortality_ids():
    """Return a set of (cunicah, np) tuples for deceased patients."""
    try:
        import pyreadstat
        sav_path = str(DATA_DIR / "simpleMHAS.sav")
        df_sav, _ = pyreadstat.read_sav(sav_path, usecols=["cunicah", "np", "fallecido"])
        dead = df_sav[df_sav["fallecido"] == 1][["cunicah", "np"]].drop_duplicates()
        return set(zip(dead["cunicah"], dead["np"]))
    except Exception as e:
        print(f"  [warn] Could not load mortality data: {e}")
        return set()


# ============================================================================
# 1. RADAR / SPIDER CHARTS
# ============================================================================

def _radar_axes(n):
    """Return evenly-spaced angles for n axes, closing the polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.append(angles, angles[0])


def _draw_radar_polygon(ax, values, angles, color, alpha=0.25, lw=1.5, label=None):
    vals = np.append(values, values[0])
    ax.plot(angles, vals, color=color, lw=lw, label=label)
    ax.fill(angles, vals, color=color, alpha=alpha)


def plot_radar_charts(df, z_cols, v_cols, death_ids):
    """
    Panel A – Phenotype comparison: terminal states for extreme agers (deciles).
    Panel B – Individual trajectory: one patient's polygons at early / late time-points.
    """
    print("\n[1/3] Generating Radar / Spider Charts …")

    # ── Population Statistics for Normalization (Z-score approach) ───────
    pop_means = df[z_cols].mean().values
    pop_stds  = df[z_cols].std().values
    pop_stds = np.where(pop_stds > 1e-5, pop_stds, 1.0)

    def norm_z(v):
        """Map Z-scores to [0, 1] for radar. 0 is -2.5SD, 0.5 is mean, 1 is +2.5SD."""
        z = (v - pop_means) / pop_stds
        return np.clip((z + 2.5) / 5.0, 0, 1)

    # ── Classify Deciles of Speed ────────────────────────────────────────
    patient_stats = df.groupby(["cunicah", "np"]).agg({"speed": "median"}).reset_index()
    q_low, q_high = patient_stats["speed"].quantile([0.1, 0.9])
    
    fast_ids = patient_stats[patient_stats["speed"] >= q_high][["cunicah", "np"]]
    slow_ids = patient_stats[patient_stats["speed"] <= q_low][["cunicah", "np"]]

    # Get Terminal States (last visit) for these groups
    df_term = df.sort_values("t").groupby(["cunicah", "np"]).tail(1)
    
    fast_term_mean = df_term.merge(fast_ids, on=["cunicah", "np"])[z_cols].mean().values
    slow_term_mean = df_term.merge(slow_ids, on=["cunicah", "np"])[z_cols].mean().values

    fast_n = norm_z(fast_term_mean)
    slow_n = norm_z(slow_term_mean)

    # ── Pick one illustrative patient with ≥ 10 observations ──────────────
    counts = df.groupby(["cunicah", "np"])["t"].count().reset_index()
    counts.columns = ["cunicah", "np", "n_obs"]
    rich = counts[counts["n_obs"] >= 10]

    # Prefer a fast-ager from top 10%
    candidates = rich.merge(fast_ids, on=["cunicah", "np"])
    if len(candidates) > 0:
        pid = candidates.sample(1, random_state=7)[["cunicah", "np"]].iloc[0]
    else:
        pid = rich.sample(1, random_state=7)[["cunicah", "np"]].iloc[0]

    pat_df = df[(df["cunicah"] == pid["cunicah"]) & (df["np"] == pid["np"])].sort_values("t")
    early = pat_df.iloc[0][z_cols].values
    late  = pat_df.iloc[-1][z_cols].values
    early_n = norm_z(early)
    late_n  = norm_z(late)

    # ── Build figure ─────────────────────────────────────────────────────
    n_dims = 8
    angles = _radar_axes(n_dims)

    fig = plt.figure(figsize=(18, 8), facecolor="#0d0d1a")
    fig.suptitle(
        "LAVA 8D Phenotype Radar — Maximum Physiological Divergence\n"
        "Center = -2.5 SD | Mid-ring = Population Mean | Edge = +2.5 SD",
        color="white", fontsize=15, fontweight="bold", y=0.97
    )

    label_positions = np.linspace(0, 2 * np.pi, n_dims, endpoint=False)

    # ── Panel A: Phenotype comparison ────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1, polar=True, facecolor="#0d0d1a")
    ax1.set_facecolor("#0d0d1a")

    for r in [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        color = "#555577" if r == 0.5 else "#222244"
        lw = 1.0 if r == 0.5 else 0.5
        ax1.plot(angles, [r] * (n_dims + 1), color=color, lw=lw, zorder=0)

    _draw_radar_polygon(ax1, fast_n, angles,
                        color="#ff4d6d", alpha=0.35, lw=2.5,
                        label="Fast Ager (Decile 10, Terminal)")
    _draw_radar_polygon(ax1, slow_n, angles,
                        color="#00f5ff", alpha=0.35, lw=2.5,
                        label="Slow Ager (Decile 1, Terminal)")

    ax1.set_xticks(label_positions)
    ax1.set_xticklabels(DIM_LABELS, color="white", fontsize=8)
    ax1.set_yticks([0.2, 0.5, 0.8, 1.0])
    ax1.set_yticklabels(["-1.5σ", "Mean", "+1.5σ", "+2.5σ"], color="#888888", fontsize=6)

    ax1.tick_params(colors="white")
    ax1.spines["polar"].set_color("#333355")
    ax1.grid(color="#222244", lw=0.4)
    ax1.set_title("Phenotype Comparison", color="white", fontsize=12,
                  fontweight="bold", pad=20)
    leg = ax1.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
                     facecolor="#1a1a2e", edgecolor="#444466",
                     labelcolor="white", fontsize=9)

    # ── Panel B: Individual trajectory ───────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2, polar=True, facecolor="#0d0d1a")
    ax2.set_facecolor("#0d0d1a")

    for r in np.linspace(0, 1, 5)[1:]:
        ax2.plot(angles, [r] * (n_dims + 1), color="#333355", lw=0.5, zorder=0)

    n_frames = min(8, len(pat_df))
    indices = np.linspace(0, len(pat_df) - 1, n_frames, dtype=int)
    cmap_traj = matplotlib.colormaps["plasma"]
    for fi, idx in enumerate(indices):
        frame = pat_df.iloc[idx][z_cols].values
        frame_n = norm_z(frame)
        c = cmap_traj(fi / (n_frames - 1))
        alpha_v = 0.10 + 0.70 * (fi / (n_frames - 1))
        _draw_radar_polygon(ax2, frame_n, angles, color=c,
                            alpha=alpha_v * 0.4, lw=1.2 + fi * 0.15)

    # Highlight first and last
    _draw_radar_polygon(ax2, early_n, angles,
                        color="#4cc9f0", alpha=0.3, lw=2.2, label="Early (baseline)")
    _draw_radar_polygon(ax2, late_n, angles,
                        color="#e63946", alpha=0.3, lw=2.2, label="Late (terminal)")

    ax2.set_xticks(label_positions)
    ax2.set_xticklabels(DIM_LABELS, color="white", fontsize=8)
    ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["25%", "50%", "75%", "100%"], color="#888888", fontsize=6)
    ax2.tick_params(colors="white")
    ax2.spines["polar"].set_color("#333355")
    ax2.grid(color="#222244", lw=0.4)
    t_early = pat_df["t"].iloc[0] + 2001
    t_late  = pat_df["t"].iloc[-1] + 2001
    ax2.set_title(
        f"Individual Trajectory\n(Patient {int(pid['cunicah'])}, {t_early:.0f}→{t_late:.0f})",
        color="white", fontsize=12, fontweight="bold", pad=20
    )
    leg2 = ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
                      facecolor="#1a1a2e", edgecolor="#444466",
                      labelcolor="white", fontsize=9)

    # Colorbar for trajectory panel
    sm = cm.ScalarMappable(cmap="plasma",
                           norm=mcolors.Normalize(vmin=t_early, vmax=t_late))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, orientation="vertical",
                        fraction=0.035, pad=0.15, shrink=0.6)
    cbar.set_label("Year", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = LATENT_DIR / "radar_8d_lava.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="#0d0d1a")
    print(f"  ✓ Saved → {out}")
    plt.close()


# ============================================================================
# 2. PARALLEL COORDINATES PLOT
# ============================================================================

def plot_parallel_coordinates(df, z_cols, v_cols, death_ids, n_sample=4000):
    """
    Each patient observation is a poly-line across z0..z7.
    Lines are coloured by velocity magnitude (plasma cmap).
    """
    print("\n[2/4] Generating Parallel Coordinates Plot …")
    _plot_parcoords_base(df, z_cols, "parallel_coords_8d_lava.png", 
                         "LAVA 8D Parallel Coordinates — Manifold States\n"
                         "Each line = one patient observation; colour = aging velocity |v|",
                         death_ids, n_sample)


def plot_velocity_parallel_coordinates(df, z_cols, v_cols, death_ids, n_sample=4000):
    """
    Each patient observation is a poly-line across v0..v7 (rates of change).
    This shows which specific domains are accelerating.
    """
    print("\n[3/4] Generating Velocity Parallel Coordinates Plot …")
    _plot_parcoords_base(df, v_cols, "velocity_parcoords_8d_lava.png",
                         "LAVA 8D Velocity Highways — Rates of Change\n"
                         "Each line = one patient; colour = total aging speed |v|",
                         death_ids, n_sample, is_velocity=True)


def _plot_parcoords_base(df, cols, filename, title, death_ids, n_sample, is_velocity=False):
    """Shared logic for Parallel Coordinates plots."""
    df_plot = df.sample(n=min(n_sample, len(df)), random_state=42).copy()

    # Normalise per-dimension to [0, 1] for display
    # Use Robust Scaling (5-95 percentile) to avoid outlier squashing
    vals = df[cols].values
    mins = np.percentile(vals, 2, axis=0)
    maxs = np.percentile(vals, 98, axis=0)
    rngs = np.where(maxs - mins > 0, maxs - mins, 1.0)
    
    C_norm = (df_plot[cols].values - mins) / rngs
    speeds = df_plot["speed"].values

    # Normalise speeds onto [0, 1] for colour mapping
    spd_lo, spd_hi = np.percentile(df["speed"].values, [2, 98])
    speeds_c = np.clip((speeds - spd_lo) / (spd_hi - spd_lo), 0, 1)

    n_dims = 8
    fig, ax = plt.subplots(figsize=(16, 8), facecolor="#0d0d1a")
    ax.set_facecolor("#0d0d1a")
    fig.suptitle(title, color="white", fontsize=14, fontweight="bold")

    cmap = matplotlib.colormaps["plasma"]
    order = np.argsort(speeds_c)
    C_norm = C_norm[order]
    speeds_c = speeds_c[order]

    x_positions = np.arange(n_dims)

    for i in range(len(C_norm)):
        color = cmap(speeds_c[i])
        alpha = 0.02 + 0.30 * speeds_c[i]
        lw    = 0.3 + 0.6 * speeds_c[i]
        ax.plot(x_positions, C_norm[i], color=color, alpha=alpha,
                lw=lw, solid_capstyle="round")

    # Labels
    labels = [d.replace("\n", " ") for d in DIM_LABELS]
    if is_velocity:
        labels = [f"Δ {l}" for l in labels]

    for j in range(n_dims):
        ax.axvline(j, color="#334466", lw=1.2, zorder=5)
        ax.text(j, 1.05, labels[j],
                ha="center", va="bottom", color="white",
                fontsize=9, fontweight="bold", transform=ax.get_xaxis_transform())
        
        # Reference ticks
        for tick in [0.0, 0.5, 1.0]:
            val = mins[j] + tick * rngs[j]
            ax.text(j - 0.07, tick, f"{val:.2f}",
                    ha="right", va="center", color="#8899bb", fontsize=6.5)

    ax.set_xlim(-0.3, n_dims - 0.7)
    ax.set_ylim(-0.1, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = cm.ScalarMappable(cmap="plasma", norm=mcolors.Normalize(vmin=spd_lo, vmax=spd_hi))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.01, shrink=0.8)
    cbar.set_label("Total Velocity Magnitude |v|", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    out = LATENT_DIR / filename
    plt.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="#0d0d1a")
    print(f"  ✓ Saved → {out}")
    plt.close()


# ============================================================================
# 4. INTERACTIVE 3D UMAP PROJECTION (Plotly)
# ============================================================================

def plot_3d_umap(df, z_cols, v_cols, death_ids, n_sample=15000):
    """
    Compress 8D → 3D with UMAP and render an interactive Plotly scatter.
    Colour = velocity magnitude; shape distinguishes survivors vs. deceased.
    Saves as both HTML (interactive) and a static PNG snapshot.
    """
    print("\n[4/4] Generating 3D UMAP Projection …")

    try:
        import umap
    except ImportError:
        print("  [warn] umap-learn not installed. Run:  pip install umap-learn")
        print("         Skipping 3D UMAP plot.")
        return

    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        print("  [warn] plotly not installed. Run:  pip install plotly")
        print("         Skipping 3D UMAP plot.")
        return

    # Subsample for UMAP speed
    df_umap = df.sample(n=min(n_sample, len(df)), random_state=42).copy()
    Z_full  = df_umap[z_cols].values
    speeds  = df_umap["speed"].values

    # Mortality flag per row
    is_dead = df_umap.apply(
        lambda r: (r["cunicah"], r["np"]) in death_ids, axis=1
    ).values

    print(f"  Fitting UMAP on {len(df_umap):,} points (8D → 3D) …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
            verbose=False
        )
        Z3 = reducer.fit_transform(Z_full)

    print("  UMAP done.  Building Plotly figure …")

    spd_lo, spd_hi = np.percentile(speeds, [2, 98])
    speeds_c = np.clip(speeds, spd_lo, spd_hi)

    # ── Two traces: alive vs. deceased ───────────────────────────────────
    def make_trace(mask, name, symbol, size, opacity):
        return go.Scatter3d(
            x=Z3[mask, 0], y=Z3[mask, 1], z=Z3[mask, 2],
            mode="markers",
            name=name,
            marker=dict(
                size=size,
                symbol=symbol,
                color=speeds_c[mask],
                colorscale="Plasma",
                cmin=spd_lo,
                cmax=spd_hi,
                opacity=opacity,
                showscale=False,
            ),
        )

    alive_mask = ~is_dead
    dead_mask  = is_dead

    traces = [make_trace(alive_mask, "Survivor", "circle", 2.5, 0.55)]
    if dead_mask.sum() > 0:
        traces.append(make_trace(dead_mask, "Deceased", "cross", 4, 0.90))

    # Colorbar trace (invisible, just to show scale)
    colorbar_trace = go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(
            size=0,
            color=[spd_lo, spd_hi],
            colorscale="Plasma",
            cmin=spd_lo,
            cmax=spd_hi,
            showscale=True,
            colorbar=dict(
                title=dict(text="Aging Speed |v|", font=dict(color="white")),
                tickfont=dict(color="white"),
                bgcolor="rgba(0,0,0,0)",
                len=0.6,
                thickness=15,
            ),
        ),
        showlegend=False,
    )
    traces.append(colorbar_trace)

    layout = go.Layout(
        title=dict(
            text="LAVA Aging Manifold — 3D UMAP Projection<br>"
                 "<sup>Rotate to explore distinct pathways of biological decline · "
                 "Colour = velocity magnitude |v|</sup>",
            x=0.5, xanchor="center",
            font=dict(color="white", size=16),
        ),
        scene=dict(
            xaxis=dict(title="UMAP-1", backgroundcolor="#0d0d1a",
                       gridcolor="#223", color="white"),
            yaxis=dict(title="UMAP-2", backgroundcolor="#0d0d1a",
                       gridcolor="#223", color="white"),
            zaxis=dict(title="UMAP-3", backgroundcolor="#0d0d1a",
                       gridcolor="#223", color="white"),
            bgcolor="#0d0d1a",
        ),
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#0d0d1a",
        font=dict(color="white"),
        legend=dict(
            font=dict(color="white"),
            bgcolor="rgba(13,13,26,0.8)",
            bordercolor="#334466",
        ),
        margin=dict(l=0, r=0, t=80, b=0),
    )

    fig = go.Figure(data=traces, layout=layout)

    # Save interactive HTML
    html_out = LATENT_DIR / "umap_3d_lava.html"
    fig.write_html(str(html_out), include_plotlyjs="cdn")
    print(f"  ✓ Interactive HTML → {html_out}")

    # Static PNG snapshot (requires kaleido)
    try:
        png_out = LATENT_DIR / "umap_3d_lava.png"
        fig.write_image(str(png_out), width=1400, height=900, scale=2)
        print(f"  ✓ Static PNG     → {png_out}")
    except Exception as e:
        print(f"  [warn] Could not save static PNG ({e}). "
              f"Install kaleido:  pip install kaleido")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  LAVA 8D Space Visualizations")
    print("=" * 60)

    df, z_cols, v_cols = load_trajectory_data()
    death_ids           = load_mortality_ids()

    print(f"\nDataset: {len(df):,} rows · "
          f"{df.groupby(['cunicah','np']).ngroups:,} patient-visits")

    plot_radar_charts(df, z_cols, v_cols, death_ids)
    plot_parallel_coordinates(df, z_cols, v_cols, death_ids)
    plot_velocity_parallel_coordinates(df, z_cols, v_cols, death_ids)
    plot_3d_umap(df, z_cols, v_cols, death_ids)

    print("\n" + "=" * 60)
    print("  ✓ All 3 visualizations complete.")
    print(f"  Output directory: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
