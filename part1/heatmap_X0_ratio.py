import os
import numpy as np
from utils.ev_experiments import phase_sweep_df
from utils.ev_plotting import plot_phase_plot


def main():
    base_dir = os.path.dirname(__file__)
    plots_dir = os.path.join(base_dir, "plots")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Grids
    X0_values = np.linspace(0.0, 1.0, 21)
    ratio_values = np.linspace(0.0, 3.5, 41)

    # Scenario settings (these override the defaults inside phase_sweep_df)
    scenario_kwargs = dict(
        I0=0.05,
        beta_I=2.0,
        b=1.0,
        g_I=0.05,
        network_type="random", 
        n_nodes=300,
        p=0.05,
        m=2,
    )

    print("Sweeping...")
    df = phase_sweep_df(
        max_workers=8,                # use up to 8 processes
        backend="process",            # process-based parallelism
        X0_values=X0_values,
        ratio_values=ratio_values,
        scenario_kwargs=scenario_kwargs,
        batch_size=16,                # trials per cell
        init_noise_I=0.04,
        T=200,
        strategy_choice_func="imitate",  
        tau=1.0,
    )
    print("Sweep completed")

    # Save CSV
    csv_path = os.path.join(results_dir, "phase_X0_ratio_ER.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results to: {csv_path}")

    # Save plot
    fig_path = os.path.join(plots_dir, "phase_X0_ratio_ER.png")
    plot_phase_plot(df, out_path=fig_path)
    print(f"Saved heatmap to: {fig_path}")

    # For BA network
    # Scenario settings (these override the defaults inside phase_sweep_df)
    scenario_kwargs = dict(
        I0=0.05,
        beta_I=2.0,
        b=1.0,
        g_I=0.05,
        network_type="BA", 
        n_nodes=300,
        p=0.05,
        m=2,
    )

    print("Sweeping...")
    df = phase_sweep_df(
        max_workers=8,                # use up to 8 processes
        backend="process",            # process-based parallelism
        X0_values=X0_values,
        ratio_values=ratio_values,
        scenario_kwargs=scenario_kwargs,
        batch_size=16,                # trials per cell
        init_noise_I=0.04,
        T=200,
        strategy_choice_func="imitate",  
        tau=1.0,
    )
    print("Sweep completed")

    # Save CSV
    csv_path = os.path.join(results_dir, "phase_X0_ratio_BA.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results to: {csv_path}")

    # Save plot
    fig_path = os.path.join(plots_dir, "phase_X0_ratio_BA.png")
    plot_phase_plot(df, out_path=fig_path)
    print(f"Saved heatmap to: {fig_path}")


if __name__ == "__main__":
    main()