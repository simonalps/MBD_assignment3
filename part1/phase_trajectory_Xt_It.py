import os
import numpy as np
import matplotlib.pyplot as plt

from utils.ev_experiments import run_timeseries_trial


def main():
    base_dir = os.path.dirname(__file__)
    plots_dir = os.path.join(base_dir, "plots")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Grids of initial conditions
    X0_values = np.linspace(0.1, 0.5, 9)  # 0.0, 0.1, ..., 1.0
    I0_values = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0

    T = 100  

    fig, ax = plt.subplots(figsize=(7, 5))

    for I0 in I0_values:
        for X0 in X0_values:
            scenario = dict(
                ratio=2.3,
                beta_I=2.0,
                b=1.0,
                g_I=0.05,
                I0=float(I0),
                network_type="random",
                n_nodes=300,
                p=0.05,
                m=2,
                collect=True,
                X0_frac=float(X0),
                init_method="random",
            )

            # Different seed each time so trajectories differ
            seed = np.random.randint(0, 2**31 - 1)

            X_series, I_series, _ = run_timeseries_trial(
                T=T,
                scenario_kwargs=scenario,
                seed=seed,
                strategy_choice_func="imitate",
                tau=1.0,
            )

            # Plot this trajectory as a faint line
            ax.plot(X_series, I_series, linewidth=0.7, alpha=0.4)

    ax.set_xlabel("X(t)  (adoption fraction)")
    ax.set_ylabel("I(t)  (infrastructure level)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Phase trajectories for multiple initial conditions")

    fig_path = os.path.join(plots_dir, "phase_plot_grid_XI.png")
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid phase plot to: {fig_path}")


if __name__ == "__main__":
    main()
