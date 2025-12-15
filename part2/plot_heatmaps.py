import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_part2_data():
    """Load the CSV produced by p2_run.py."""
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "results")
    csv_path = os.path.join(results_dir, "part2_all_timeseries.csv")

    df = pd.read_csv(csv_path)
    return df


def compute_final_means(df):
    """
    From the full timeseries, compute mean final adoption X(T)
    as a function of X0, separately for each network_type and seeding_strategy.

    Returns:
        grouped: dict[(network_type, seeding_strategy)] -> (X0_values, mean_X_final_array)
    """
    # Use the last recorded time step as "final"
    t_final = df["t"].max()

    grouped = {}
    for (net, seed_strat), sub in df[df["t"] == t_final].groupby(
        ["network_type", "seeding_strategy"]
    ):
        # Average across runs for each X0
        means = (
            sub.groupby("X0")["X"]
            .mean()
            .sort_index()
        )
        X0_vals = means.index.to_numpy()
        X_final_mean = means.to_numpy()
        grouped[(net, seed_strat)] = (X0_vals, X_final_mean)

    return grouped


def plot_1d_heatmaps(grouped, cmap="plasma", out_name="p2_heatmap_1d.png"):
    """
    1D heatmaps:
    rows = seeding strategies, cols = network types.
    Colorbar on the LEFT. No suptitle. Clean axes. Tick labels on ALL subplots.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    networks = sorted({k[0] for k in grouped.keys()})
    seeding_strategies = sorted({k[1] for k in grouped.keys()})

    n_rows = len(seeding_strategies)
    n_cols = len(networks)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 1.8 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    vmin, vmax = 0.0, 1.0

    for i, seed_strat in enumerate(seeding_strategies):
        for j, net in enumerate(networks):
            ax = axes[i, j]

            X0_vals, X_final_mean = grouped[(net, seed_strat)]
            data = X_final_mean[np.newaxis, :]

            im = ax.imshow(
                data,
                aspect="auto",
                origin="lower",
                extent=[X0_vals[0], X0_vals[-1], 0, 1],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )

            # --- ticks on ALL rows ---
            major_ticks = np.arange(0.0, 1.01, 0.2)
            ax.set_xticks(major_ticks)
            ax.set_xticklabels([f"{x:.1f}" for x in major_ticks])

            ax.set_yticks([])

            # Rename "random" → "ER" for display purposes
            display_net = "ER" if net == "random" else net
            ax.set_title(f"{display_net} – {seed_strat} seeding")

            # bottom row gets xlabel
            if i == n_rows - 1:
                ax.set_xlabel("Initial adoption $X_0$")

    # ---- Colorbar on LEFT ----
    cbar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.8,
        location="left",
        pad=0.02
    )
    cbar.set_label("Mean final adoption $\\bar{X}(T)$")

    # No suptitle requested

    # Save
    base_dir = os.path.dirname(__file__)
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, out_name)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 1D heatmap figure to: {out_path}")


def main():
    df = load_part2_data()
    grouped = compute_final_means(df)
    plot_1d_heatmaps(grouped)


if __name__ == "__main__":
    main()