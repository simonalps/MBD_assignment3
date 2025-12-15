import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_part2_data():
    """Load the CSV produced by p2_run.py."""
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "results")
    csv_path = os.path.join(results_dir, "part2_all_timeseries.csv")
    return pd.read_csv(csv_path)


def plot_prob_high_adoption_grid(
    df,
    threshold: float = 0.8,
    out_name: str = "p2_prob_high_adoption.png",
):
    """
    Make a 2 x 3 grid of P[X(T) >= threshold] vs X0.

    Rows:
        0: degree seeding
        1: random seeding
    Columns:
        BA, WS, ER

    Uses all X0 values present in the dataset.
    """

    # Work on a copy and stabilise X0 with rounding
    df = df.copy()
    df["X0_round"] = df["X0"].round(3)

    # Final time step
    t_final = df["t"].max()
    df_final = df[df["t"] == t_final]

    # Compute probability per (network_type, seeding_strategy, X0_round)
    grouped = (
        df_final
        .groupby(["network_type", "seeding_strategy", "X0_round"])["X"]
        .agg(
            n_runs="count",
            n_high=lambda x: np.sum(x >= threshold),
        )
        .reset_index()
    )
    grouped["prob_high"] = grouped["n_high"] / grouped["n_runs"]

    # Orderings
    seeding_order = ["degree", "random"]     # degree on top
    networks = ["BA", "WS", "random"]        # BA, WS, ER (random=ER)
    net_display = {"BA": "BA", "WS": "WS", "random": "ER"}

    # Unique X0 values (sorted)
    X0_vals = sorted(grouped["X0_round"].unique())

    # Layout
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

            sub = grouped[
                (grouped["seeding_strategy"] == seed_strat)
                & (grouped["network_type"] == net)
            ]

            # Ensure in X0 order
            sub = sub.sort_values("X0_round")

            ax.plot(
                sub["X0_round"],
                sub["prob_high"],
                marker="o",
                linewidth=1.6,
            )

            # Axes styles
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(-0.05, 1.05)

            # X ticks at 0.0, 0.2, ...
            major_xticks = np.arange(0.0, 1.01, 0.2)
            ax.set_xticks(major_xticks)
            ax.set_xticklabels([f"{x:.1f}" for x in major_xticks])

            # Y ticks 0, 0.25, 0.5, 0.75, 1
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

            if j == 0:
                ax.set_ylabel(f"$P(X(T) \\geq {threshold:.2f})$")

            if i == n_rows - 1:
                ax.set_xlabel("$X_0$")

            ax.set_title(f"{net_display[net]} – {seed_strat} seeding")

            # optional: light grid, comment out if you prefer no grid at all
            ax.grid(True, alpha=0.2)

    # No suptitle (to stay consistent with your style)

    # Save
    base_dir = os.path.dirname(__file__)
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, out_name)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved probability-of-high-adoption figure to: {out_path}")


def main():
    df = load_part2_data()

    # New mean trajectories figure:
    plot_prob_high_adoption_grid(df)


if __name__ == "__main__":
    main()
