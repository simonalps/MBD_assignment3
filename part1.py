"""Part 1 script.

Generates:
  1) Heatmap over X0 × I0
  2) Heatmap over X0 × ratio
  3) Phase-plane trajectories (X(t) vs I(t))
  4) βI sensitivity curve

All outputs are written under:
  - results/part1/
  - plots/part1/
"""

from __future__ import annotations

import os
import numpy as np

from utils.ev_experiments import (
    phase_sweep_X0_I0_df,
    phase_sweep_df,
    phase_trajectories_grid_df,
    beta_sensitivity_df,
)
from utils.ev_plotting import (
    plot_phase_plot_X0_I0,
    plot_phase_plot,
    plot_phase_trajectories,
    plot_beta_sensitivity,
)


def _make_dirs(base_dir: str) -> tuple[str, str]:
    plots_dir = os.path.join(base_dir, "plots", "part1")
    results_dir = os.path.join(base_dir, "results", "part1")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return plots_dir, results_dir


def main():
    base_dir = os.path.dirname(__file__)
    plots_dir, results_dir = _make_dirs(base_dir)

    # -----------------------------
    # Scenario parameters
    # -----------------------------
    scenario_kwargs = dict(
        ratio=2.3,
        beta_I=2.0,
        b=1.0,
        g_I=0.05,
        I0=0.05,
        n_nodes=400,
        network_type="random",  # ER
        p=0.015,                # mean degree ~6 when n_nodes=400
        m=3,                    # mean degree 6
        init_method="random",
    )

    T = 50
    batch_size = 20
    seed_base = 42

    # =========================================================
    # 1) Heatmap: X0 × I0 for ER and BA
    # =========================================================
    print("1. X0 × I0 heatmap")
    X0_values = np.linspace(0.0, 1.0, 21)
    I0_values = np.linspace(0.0, 1.0, 21)

    heatmap_networks = [
        ("random", "ER"), 
        ("BA", "BA"),
    ]

    for net_type, net_label in heatmap_networks:
        print(f"\n--- Heatmap for network: {net_label} ({net_type}) ---")

        scenario_hm = dict(scenario_kwargs)
        scenario_hm["network_type"] = net_type

        print("Sweeping over X0 and I0...")
        df_x0_i0 = phase_sweep_X0_I0_df(
            max_workers=8,
            backend="process",
            X0_values=X0_values,
            I0_values=I0_values,
            scenario_kwargs=scenario_hm,
            batch_size=batch_size,
            T=T,
            strategy_choice_func="imitate",
            tau=1.0,
        )
        print("Sweep ended!")

        csv_path = os.path.join(results_dir, f"part1_heatmap_X0_I0_{net_label}.csv")
        df_x0_i0.to_csv(csv_path, index=False)
        print(f"Saved csv under: {csv_path}.")

        fig_path = os.path.join(plots_dir, f"part1_heatmap_X0_I0_{net_label}.png")
        plot_phase_plot_X0_I0(df_x0_i0, out_path=fig_path)
        print(f"Saved X0 × I0 heatmap under: {fig_path}.")

    # =========================================================
    # 2) Heatmap: X0 × ratio for ER and BA
    # =========================================================
    print("\n2. X0 × ratio heatmap")
    ratio_values = np.linspace(0.8, 3.5, 31)

    for net_type, net_label in heatmap_networks:
        print(f"\n--- Heatmap for network: {net_label} ({net_type}) ---")

        scenario_hm = dict(scenario_kwargs)
        scenario_hm["network_type"] = net_type

        print("Sweeping over X0 and ratio...")
        df_x0_ratio = phase_sweep_df(
            max_workers=8,
            backend="process",
            X0_values=X0_values,
            ratio_values=ratio_values,
            scenario_kwargs=scenario_hm,
            batch_size=batch_size,
            T=T,
            strategy_choice_func="imitate",
            tau=1.0,
        )
        print("Sweep ended!")

        csv_path = os.path.join(results_dir, f"part1_heatmap_X0_ratio_{net_label}.csv")
        df_x0_ratio.to_csv(csv_path, index=False)
        print(f"Saved csv under: {csv_path}.")

        fig_path = os.path.join(plots_dir, f"part1_heatmap_X0_ratio_{net_label}.png")
        plot_phase_plot(df_x0_ratio, out_path=fig_path)
        print(f"Saved X0 × ratio heatmap under: {fig_path}.")

    # =========================================================
    # 3) Trajectories: X(t) vs I(t)
    # =========================================================
    print("3. Trajectory plot.")
    X0_traj = np.linspace(0.1, 1.0, 10)
    I0_traj = np.linspace(0.0, 1.0, 11)

    print("Running...")
    df_traj = phase_trajectories_grid_df(
        X0_values=X0_traj,
        I0_values=I0_traj,
        T=T,
        scenario_kwargs=scenario_kwargs,
        seed_base=seed_base,
        strategy_choice_func="imitate",
        tau=1.0,
    )
    print("Trajectories completed!")

    csv_path = os.path.join(results_dir, "part1_phase_trajectories_XI.csv")
    df_traj.to_csv(csv_path, index=False)
    print(f"Saved csv under: {csv_path}.")

    fig_path = os.path.join(plots_dir, "part1_phase_trajectories_XI.png")
    plot_phase_trajectories(df_traj, out_path=fig_path)
    print(f"Saved trajectory plot under: {fig_path}.")

    # =========================================================
    # 4) βI sensitivity
    # =========================================================
    print("4. β_I sensitivity analysis.")
    beta_values = np.linspace(0.0, 5.0, 51)

    print("Sweeping over β_I...")
    df_beta = beta_sensitivity_df(
        beta_values=beta_values,
        batch_size=batch_size,
        X0_frac=0.30,
        ratio=2.3,
        scenario_kwargs=dict(
            I0=0.05,
            b=1.0,
            g_I=0.05,
            T=T,
            network_type="random",
            n_nodes=400,
            p=0.015,
            m=3,
            init_method="random",
        ),
    )
    print("Sweep ended!")

    csv_path = os.path.join(results_dir, "part1_beta_sensitivity.csv")
    df_beta.to_csv(csv_path, index=False)
    print(f"Saved csv under: {csv_path}.")

    fig_path = os.path.join(plots_dir, "part1_beta_sensitivity.png")
    plot_beta_sensitivity(df_beta, out_path=fig_path)
    print(f"Saved β_I sensitivity analysis plot under: {fig_path}.")


if __name__ == "__main__":
    main()