"""
Plotting utilities for EV Stag Hunt experiments.

All functions accept pandas DataFrames produced by ev_experiments
and save figures to disk, returning the output path.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _default_plot_path(filename: str) -> str:
    plots_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return os.path.join(plots_dir, filename)


def plot_fanchart(traces_df: pd.DataFrame, out_path: Optional[str] = None) -> str:
    """Plot fan charts (quantile bands) for baseline vs subsidy using traces DF.

    traces_df columns: ['group', 'trial', 'time', 'X'] where group in {'baseline','subsidy'}.
    """
    if traces_df.empty:
        raise ValueError("traces_df is empty")

    groups = ["baseline", "subsidy"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    for j, group in enumerate(groups):
        gdf = traces_df[traces_df["group"] == group]

        # Compute quantiles by time across trials
        q = gdf.groupby("time")["X"].quantile([0.10, 0.25, 0.75, 0.90]).unstack(level=1)
        mean = gdf.groupby("time")["X"].mean()
        t = mean.index.to_numpy()

        ax = axes[0, j]
        ax.fill_between(t, q[0.10], q[0.90], color=("steelblue" if group == "baseline" else "darkorange"), alpha=0.15, label="10–90%")
        ax.fill_between(t, q[0.25], q[0.75], color=("steelblue" if group == "baseline" else "darkorange"), alpha=0.30, label="25–75%")

        # Overlay some traces for context (sample up to 100 trials)
        trial_ids = gdf["trial"].unique()
        rng = np.random.default_rng(123)
        sample = rng.choice(trial_ids, size=min(100, len(trial_ids)), replace=False)
        for tr in sample:
            tr_df = gdf[gdf["trial"] == tr]
            ax.plot(tr_df["time"], tr_df["X"], color=("steelblue" if group == "baseline" else "darkorange"), alpha=0.1, linewidth=0.8)

        ax.plot(t, mean, color=("steelblue" if group == "baseline" else "darkorange"), linewidth=2, label="mean")
        ax.set_title(f"{group.capitalize()} adoption")
        ax.set_xlabel("Time")
        ax.set_ylabel("X(t)")
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")

        # Final X(T) histogram
        t_max = int(gdf["time"].max())
        final_vals = gdf[gdf["time"] == t_max].groupby("trial")["X"].mean().to_numpy()
        axes[1, j].hist(final_vals, bins=20, color=("steelblue" if group == "baseline" else "darkorange"), alpha=0.8)
        axes[1, j].set_title(f"{group.capitalize()} final X(T)")
        axes[1, j].set_xlabel("X(T)")
        axes[1, j].set_ylabel("Count")

    if out_path is None:
        out_path = _default_plot_path("ev_intervention_fanchart.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_spaghetti(traces_df: pd.DataFrame, *, max_traces: int = 100, alpha: float = 0.15, out_path: Optional[str] = None) -> str:
    """Spaghetti plot from traces DF for baseline vs subsidy."""
    groups = ["baseline", "subsidy"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    rng = np.random.default_rng(123)

    for j, group in enumerate(groups):
        gdf = traces_df[traces_df["group"] == group]
        trial_ids = gdf["trial"].unique()
        sample = rng.choice(trial_ids, size=min(max_traces, len(trial_ids)), replace=False)
        ax = axes[j]
        for tr in sample:
            tr_df = gdf[gdf["trial"] == tr]
            ax.plot(tr_df["time"], tr_df["X"], color=("steelblue" if group == "baseline" else "darkorange"), alpha=alpha, linewidth=0.8)
        ax.set_title(f"{group.capitalize()} traces")
        ax.set_xlabel("Time")
        ax.set_ylabel("X(t)")
        ax.set_ylim(0, 1)

    if out_path is None:
        out_path = _default_plot_path("ev_spaghetti.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_density(traces_df: pd.DataFrame, *, x_bins: int = 50, time_bins: Optional[int] = None, out_path: Optional[str] = None) -> str:
    """Time-evolving density plot (2D histogram) from traces DF."""
    groups = ["baseline", "subsidy"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    for j, group in enumerate(groups):
        gdf = traces_df[traces_df["group"] == group]
        T = int(gdf["time"].max()) + 1
        if time_bins is None:
            bins_time = T
        else:
            bins_time = time_bins
        hb = axes[j].hist2d(gdf["time"].to_numpy(), gdf["X"].to_numpy(), bins=[bins_time, x_bins], range=[[0, T - 1], [0.0, 1.0]], cmap="magma")
        axes[j].set_title(f"{group.capitalize()} density: time vs X(t)")
        axes[j].set_xlabel("Time")
        axes[j].set_ylabel("X(t)")
        fig.colorbar(hb[3], ax=axes[j], label="count")

    if out_path is None:
        out_path = _default_plot_path("ev_density.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_ratio_sweep(sweep_df: pd.DataFrame, out_path: Optional[str] = None) -> str:
    """Plot X* vs ratio from a DataFrame with columns ['ratio','X_mean']."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep_df["ratio"], sweep_df["X_mean"], color="C0", lw=2)
    ax.set_xlabel("a_I / b (ratio)")
    ax.set_ylabel("Final adoption X*")
    ax.set_title("X* vs ratio")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    if out_path is None:
        out_path = _default_plot_path("ev_ratio_sweep.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_phase_plot(phase_df: pd.DataFrame, out_path: Optional[str] = None) -> str:
    """Plot heatmap from tidy DataFrame with columns ['X0','ratio','X_final']."""
    # Pivot to matrix for imshow
    pivot = phase_df.pivot(index="ratio", columns="X0", values="X_final").sort_index().sort_index(axis=1)
    ratios = pivot.index.to_numpy()
    X0s = pivot.columns.to_numpy()

    plt.figure(figsize=(7, 4))
    im = plt.imshow(
        pivot.to_numpy(),
        origin="lower",
        extent=[X0s[0], X0s[-1], ratios[0], ratios[-1]],
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="plasma",
    )
    plt.colorbar(im, label="Final adopters X*")
    plt.xlabel("X0 (initial adoption)")
    plt.ylabel("a_I / b (initial payoff ratio)")
    plt.title("Network phase plot: X* over X0 and a_I/b")

    # Overlay threshold X = 1/ratio
    X_thresh = 1.0 / ratios
    X_thresh_clipped = np.clip(X_thresh, 0.0, 1.0)
    plt.plot(X_thresh_clipped, ratios, color="white", linestyle="--", linewidth=1.5, label="X = b / a_I (initial)")
    plt.legend(loc="upper right")

    if out_path is None:
        out_path = _default_plot_path("ev_phase_plot.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    return out_path

# ------------------------------------------------------
#   PART 1 ADDITIONS
# ------------------------------------------------------

def plot_phase_plot_X0_I0(
    df: pd.DataFrame, out_path: Optional[str] = None
) -> str:
    """Plot heatmap from tidy DataFrame with columns ['X0','I0','X_final']."""

    pivot = (
        df.pivot(index="I0", columns="X0", values="X_final")
        .sort_index()
        .sort_index(axis=1)
    )

    X0s = pivot.columns.to_numpy()
    I0s = pivot.index.to_numpy()

    plt.figure(figsize=(7, 4))
    im = plt.imshow(
        pivot.to_numpy(),
        origin="lower",
        extent=[X0s[0], X0s[-1], I0s[0], I0s[-1]],
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="plasma",
    )
    plt.colorbar(im, label="Final adoption X*")
    plt.xlabel("X0 (initial adoption)")
    plt.ylabel("I0 (initial infrastructure)")
    plt.title("Phase plot: X* over X0 and I0")

    if out_path is None:
        out_path = "phase_X0_I0.png"

    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()

    return out_path

def plot_phase_trajectories(
    df: pd.DataFrame,
    *,
    out_path: str,
    title: str = "Phase trajectories for multiple initial conditions",
) -> str:
    """Plot phase-plane trajectories (X(t) vs I(t)) from `phase_trajectories_grid_df` output."""

    fig, ax = plt.subplots(figsize=(6, 5))

    for run_id, g in df.groupby("run_id"):
        ax.plot(g["X"].to_numpy(), g["I"].to_numpy(), linewidth=1.0, alpha=0.8)

    ax.set_xlabel("X(t)  (adoption fraction)")
    ax.set_ylabel("I(t)  (infrastructure level)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_beta_sensitivity(
    df: pd.DataFrame,
    *,
    out_path: str,
    title: str = "Sensitivity of Final Adoption to βI",
) -> str:
    """Plot mean final adoption vs beta_I from `beta_sensitivity_df` output."""

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        df["beta_I"].to_numpy(),
        df["mean_X_final"].to_numpy(),
        marker="o",
        markersize=4,
        linewidth=1.5,
    )

    ax.set_xlabel("βI (infrastructure feedback strength)")
    ax.set_ylabel("Final adoption X*")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path

# ------------------------------------------------------
#   PART 1 ADDITIONS --- END
# ------------------------------------------------------

# ------------------------------------------------------
#   PART 2 ADDITIONS
# ------------------------------------------------------

def plot_part2_heatmaps_1d(
    final_means_df: pd.DataFrame,
    *,
    out_path: str,
    cmap: str = "plasma",
) -> str:
    """1D heatmaps of mean final adoption vs X0 (rows=seeding, cols=networks).

    Expects a tidy DF from `part2_final_means_df` with columns:
      ['network_type','seeding_strategy','X0','mean_X_final']
    """
    df = final_means_df.copy()

    networks = sorted(df["network_type"].unique().tolist())
    seeding_strategies = sorted(df["seeding_strategy"].unique().tolist())

    n_rows = len(seeding_strategies)
    n_cols = len(networks)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * n_cols, 1.9 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    vmin, vmax = 0.0, 1.0
    im = None

    for i, seed_strat in enumerate(seeding_strategies):
        for j, net in enumerate(networks):
            ax = axes[i, j]

            sub = df[(df["network_type"] == net) & (df["seeding_strategy"] == seed_strat)].copy()
            sub = sub.sort_values("X0")
            X0_vals = sub["X0"].to_numpy()
            X_final_mean = sub["mean_X_final"].to_numpy()
            data = X_final_mean[np.newaxis, :]

            im = ax.imshow(
                data,
                aspect="auto",
                origin="lower",
                extent=[float(X0_vals[0]), float(X0_vals[-1]), 0, 1],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )

            major_ticks = np.arange(0.0, 1.01, 0.2)
            ax.set_xticks(major_ticks)
            ax.set_xticklabels([f"{x:.1f}" for x in major_ticks])
            ax.set_yticks([])

            display_net = "ER" if net == "random" else net
            ax.set_title(f"{display_net} – {seed_strat} seeding")

            if i == n_rows - 1:
                ax.set_xlabel("Initial adoption $X_0$")

    if im is not None:
        cbar = fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            shrink=0.8,
            location="left",
            pad=0.02,
        )
        cbar.set_label("Mean final adoption X(T)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150) 
    plt.close(fig)
    return out_path


def plot_part2_mean_trajectories_grid(
    mean_traj_df: pd.DataFrame,
    *,
    out_path: str,
    X0_by_seeding: Optional[dict] = None,
) -> str:
    """2×3 grid of mean X(t) trajectories.

    mean_traj_df must have columns:
      ['network_type','seeding_strategy','X0','t','X'] where X0 is rounded.
    """
    df = mean_traj_df.copy()

    if X0_by_seeding is None:
        X0_by_seeding = {
            "degree": [0.05, 0.10, 0.15, 0.20],
            "random": [0.10, 0.20, 0.30, 0.40, 0.50],
        }

    seeding_order = ["degree", "random"]
    networks = ["BA", "WS", "random"]
    net_display = {"BA": "BA", "WS": "WS", "random": "ER"}

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 2.6 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    t_vals = sorted(df["t"].unique())

    for i, seed_strat in enumerate(seeding_order):
        X0_vals = X0_by_seeding.get(seed_strat, [])
        cmap = plt.get_cmap("plasma")
        colors = {x0: cmap(k / max(len(X0_vals) - 1, 1)) for k, x0 in enumerate(X0_vals)}

        df_seed = df[df["seeding_strategy"] == seed_strat]

        for j, net in enumerate(networks):
            ax = axes[i, j]
            for x0 in X0_vals:
                x0r = round(float(x0), 3)
                sub = df_seed[(df_seed["network_type"] == net) & (df_seed["X0"] == x0r)].sort_values("t")
                if sub.empty:
                    continue
                ax.plot(
                    sub["t"],
                    sub["X"],
                    label=f"$X_0={x0:.2f}$",
                    color=colors[x0],
                    linewidth=1.6,
                )

            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(min(t_vals), max(t_vals))
            ax.set_yticks([0.0, 0.5, 1.0])

            if j == 0:
                ax.set_ylabel("Adoption $X(t)$")
            if i == n_rows - 1:
                ax.set_xlabel("Time $t$")

            ax.set_title(f"{net_display[net]} – {seed_strat} seeding")
            ax.grid(True, alpha=0.2)

        axes[i, -1].legend(
            title="$X_0$",
            fontsize=8,
            title_fontsize=9,
            loc="lower right",
            frameon=False,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_part2_prob_high_adoption_grid(
    prob_df: pd.DataFrame,
    *,
    out_path: str,
    threshold: float = 0.8,
) -> str:
    """2×3 grid of probability of high adoption vs X0.

    prob_df expected from `part2_prob_high_df` with columns:
      ['network_type','seeding_strategy','X0_round','n_runs','n_high','prob_high']
    """
    df = prob_df.copy()
    seeding_order = ["degree", "random"]
    networks = ["BA", "WS", "random"]
    net_display = {"BA": "BA", "WS": "WS", "random": "ER"}

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 2.6 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for i, seed_strat in enumerate(seeding_order):
        for j, net in enumerate(networks):
            ax = axes[i, j]
            sub = df[(df["seeding_strategy"] == seed_strat) & (df["network_type"] == net)].copy()
            sub = sub.sort_values("X0_round")

            ax.plot(
                sub["X0_round"],
                sub["prob_high"],
                marker="o",
                linewidth=1.6,
            )

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(-0.05, 1.05)

            major_xticks = np.arange(0.0, 1.01, 0.2)
            ax.set_xticks(major_xticks)
            ax.set_xticklabels([f"{x:.1f}" for x in major_xticks])

            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

            if j == 0:
                ax.set_ylabel(f"$P(X(T) \geq {threshold:.2f})$")
            if i == n_rows - 1:
                ax.set_xlabel("$X_0$")

            ax.set_title(f"{net_display[net]} – {seed_strat} seeding")
            ax.grid(True, alpha=0.2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

# ------------------------------------------------------
#   PART 2 ADDITIONS --- END
# ------------------------------------------------------

# ------------------------------------------------------
#   PART 3 ADDITIONS
# ------------------------------------------------------

def plot_part3_targeted_sweep_heatmap(
    df: pd.DataFrame,
    *,
    x_col: str = "X0_hubs",
    y_col: str = "delta_a0",
    value_col: str = "mean_final_X",
    title: str,
    out_path: str,
    cmap: str = "plasma",
) -> str:
    """Heatmap: mean final adoption over (X0_hubs, delta_a0) for one network."""
    pivot = (
        df.pivot(index=y_col, columns=x_col, values=value_col)
        .sort_index()
        .sort_index(axis=1)
    )

    x_vals = pivot.columns.to_numpy(dtype=float)
    y_vals = pivot.index.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)

    im = ax.imshow(
        pivot.to_numpy(),
        origin="lower",
        aspect="auto",
        extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
    )

    ax.set_xticks(x_vals)
    ax.set_xticklabels([f"{x:.2f}" for x in x_vals])
    ax.set_yticks(y_vals)
    ax.set_yticklabels([f"{y:.1f}" for y in y_vals])

    ax.set_xlabel("Targeted giveaway X_hubs")
    ax.set_ylabel("Subsidy strength deta_a0")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, location="left", pad=0.02)
    cbar.set_label("Mean final adoption X(T)$")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_part3_timing_sweep(
    df: pd.DataFrame,
    *,
    out_path: str,
    title: str = "Effect of subsidy timing window on adoption (mean final X)",
) -> str:
    """Line plot: mean final adoption vs subsidy_end, one line per network_label."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)

    for label in sorted(df["network_label"].unique().tolist()):
        sub = df[df["network_label"] == label].sort_values("subsidy_end")
        ax.plot(sub["subsidy_end"], sub["mean_final_X"], marker="o", label=label)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Subsidy end time")
    ax.set_ylabel("Mean final adoption  X(T)")
    ax.set_title(title)
    ax.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_part3_tipping_probability_over_time(
    baseline_X,
    policy_X,
    *,
    threshold: float = 0.8,
    subsidy_end: int | None = None,
    out_path: str,
    title: str = "High-adoption probability over time",
) -> str:
    """Plot P(X(t) >= threshold) over time for baseline vs policy."""
    T = len(baseline_X[0])

    base_prob = np.zeros(T)
    pol_prob = np.zeros(T)

    for t in range(T):
        base_prob[t] = np.mean([x[t] >= threshold for x in baseline_X])
        pol_prob[t] = np.mean([x[t] >= threshold for x in policy_X])

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(base_prob, label="Baseline")
    ax.plot(pol_prob, label="Policy")

    if subsidy_end is not None:
        ax.axvline(int(subsidy_end), linestyle="--", alpha=0.6, label="Subsidy end")

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"P(X(t) ≥ {threshold})")
    ax.set_title(title)
    ax.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

# ------------------------------------------------------
#   PART 3 ADDITIONS --- END
# ------------------------------------------------------