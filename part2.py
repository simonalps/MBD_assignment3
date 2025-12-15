import os
import numpy as np

from utils.ev_experiments import (
    part2_run_timeseries_df,
    part2_save_timeseries_csv,
    part2_load_timeseries_csv,
    part2_final_means_df,
    part2_mean_trajectories_df,
    part2_prob_high_df,
)
from utils.ev_plotting import (
    plot_part2_heatmaps_1d,
    plot_part2_mean_trajectories_grid,
    plot_part2_prob_high_adoption_grid,
)


def main():
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "results", "part2")
    plots_dir = os.path.join(base_dir, "plots", "part2")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # -----------------------------
    # Global experiment settings
    # -----------------------------
    network_types = ["random", "BA", "WS"]
    seeding_strategies = ["random", "degree"]
    X0_values = np.linspace(0.0, 1.0, 21)

    n_runs_per_setting = 20
    T = 50
    seed_base = 42

    # Common model parameters across all networks
    base_params = dict(
        ratio=2.3,
        beta_I=2.0,
        b=1.0,
        g_I=0.05,
        I0=0.05,
        n_nodes=400,
    )

    csv_path = os.path.join(results_dir, "part2_all_timeseries.csv")

    # # 1) One run -> CSV
    # print("Part 2: running full sweep and writing CSV")
    # df_all = part2_run_timeseries_df(
    #     network_types=network_types,
    #     seeding_strategies=seeding_strategies,
    #     X0_values=X0_values,
    #     n_runs_per_setting=n_runs_per_setting,
    #     T=T,
    #     seed_base=seed_base,
    #     base_params=base_params,
    #     k_target=6.0,
    #     ws_rewire_p=0.1,
    #     strategy_choice_func="imitate",
    #     tau=1.0,
    #     progress=True,
    # )
    # part2_save_timeseries_csv(df_all, csv_path)
    # print(f"Saved Part 2 data to: {csv_path}")

    # Load back data
    df = part2_load_timeseries_csv(csv_path)

    # 2) Heatmaps
    print("Part 2: plotting heatmaps")
    final_means = part2_final_means_df(df)
    heatmap_path = os.path.join(plots_dir, "part2_heatmap_1d.png")
    plot_part2_heatmaps_1d(final_means, out_path=heatmap_path, cmap="plasma")
    print(f"Saved: {heatmap_path}")

    # 3) Trajectories
    print("Part 2: plotting mean trajectories")
    mean_traj = part2_mean_trajectories_df(df)
    traj_path = os.path.join(plots_dir, "part2_mean_trajectories.png")
    plot_part2_mean_trajectories_grid(mean_traj, out_path=traj_path)
    print(f"Saved: {traj_path}")

    # 4) Probability of high adoption
    print("Part 2: plotting probability of high adoption")
    prob_df = part2_prob_high_df(df, threshold=0.8)
    prob_path = os.path.join(plots_dir, "part2_prob_high_adoption.png")
    plot_part2_prob_high_adoption_grid(prob_df, out_path=prob_path, threshold=0.8)
    print(f"Saved: {prob_path}")


if __name__ == "__main__":
    main()