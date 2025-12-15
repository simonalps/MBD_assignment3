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


def plot_mean_trajectories_grid(df, out_name="p2_mean_trajectories.png"):
    """
    Make a 2 x 3 grid of mean X(t) trajectories.

    Row 0 = degree seeding
    Row 1 = random seeding

    Columns left→right = BA, WS, ER

    Degree seeding X0 values:  0.05, 0.10, 0.15, 0.20
    Random seeding X0 values:  0.10, 0.20, 0.30, 0.40, 0.50
    """

    # X0 values to use
    X0_by_seeding = {
        "degree": [0.05, 0.10, 0.15, 0.20],
        "random": [0.10, 0.20, 0.30, 0.40, 0.50],
    }

    # Make X0 rounding consistent
    df = df.copy()
    df["X0_round"] = df["X0"].round(3)

    # ---- ORDERING: degree on top, random on bottom ----
    seeding_order = ["degree", "random"]

    # ---- ORDERING: BA, WS, ER ----
    networks = ["BA", "WS", "random"]  # 'random' = ER in display
    net_display = {"BA": "BA", "WS": "WS", "random": "ER"}

    # Layout
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 2.6 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    # Time axis
    t_vals = sorted(df["t"].unique())

    for i, seed_strat in enumerate(seeding_order):
        X0_vals = X0_by_seeding[seed_strat]

        # Colormap for X0 curves
        cmap = plt.get_cmap("plasma")
        colors = {
            x0: cmap(k / max(len(X0_vals) - 1, 1))
            for k, x0 in enumerate(X0_vals)
        }

        # Filter once by seeding strategy
        df_seed = df[df["seeding_strategy"] == seed_strat]

        # Precompute mean trajectories
        mean_traj = (
            df_seed.groupby(["network_type", "X0_round", "t"])["X"]
            .mean()
            .reset_index()
        )

        for j, net in enumerate(networks):
            ax = axes[i, j]

            for x0 in X0_vals:
                x0r = round(x0, 3)
                sub = mean_traj[
                    (mean_traj["network_type"] == net)
                    & (mean_traj["X0_round"] == x0r)
                ]
                if sub.empty:
                    continue

                sub = sub.sort_values("t")

                ax.plot(
                    sub["t"],
                    sub["X"],
                    label=f"$X_0={x0:.2f}$",
                    color=colors[x0],
                    linewidth=1.6,
                )

            # Axis styling
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(min(t_vals), max(t_vals))
            ax.set_yticks([0.0, 0.5, 1.0])

            if j == 0:
                ax.set_ylabel("Adoption $X(t)$")

            if i == n_rows - 1:
                ax.set_xlabel("Time $t$")

            ax.set_title(f"{net_display[net]} – {seed_strat} seeding")

            ax.grid(True, alpha=0.2)

        # Put a legend on the rightmost column of each row
        axes[i, -1].legend(
            title="$X_0$",
            fontsize=8,
            title_fontsize=9,
            loc="lower right",
            frameon=False,
        )

    # No suptitle — consistent with heatmaps

    # Save
    base_dir = os.path.dirname(__file__)
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, out_name)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved mean trajectories figure to: {out_path}")


def main():
    df = load_part2_data()

    # New mean trajectories figure:
    plot_mean_trajectories_grid(df)


if __name__ == "__main__":
    main()
