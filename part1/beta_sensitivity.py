import os
import numpy as np
import matplotlib.pyplot as plt

from utils.ev_core import run_network_trial


def main():

    base_dir = os.path.dirname(__file__)
    plots_dir = os.path.join(base_dir, "plots")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # --- Beta sweep grid ---
    beta_values = np.linspace(0.0, 5.0, 51)    # 0.0, 0.1, ..., 5.0
    batch_size = 16                            # number of runs per beta

    # --- Store results ---
    mean_final_adoption = []

    print("Running βI sensitivity sweep...")

    for beta_I in beta_values:
        finals = []

        for _ in range(batch_size):

            seed = np.random.randint(0, 2**31 - 1)

            x_final = run_network_trial(
                X0_frac=0.30,                 
                ratio=2.3,
                I0=0.05,
                beta_I=beta_I,                
                b=1.0,
                g_I=0.05,
                T=200,
                network_type="random",
                n_nodes=300,
                p=0.05,
                m=2,
                seed=seed,
                collect=False,
                strategy_choice_func="imitate",
                tau=1.0,
            )

            finals.append(x_final)

        mean_final_adoption.append(np.mean(finals))

    mean_final_adoption = np.array(mean_final_adoption)

    # --- Save results CSV ---
    csv_path = os.path.join(results_dir, "sensitivity_betaI.csv")
    np.savetxt(
        csv_path,
        np.column_stack([beta_values, mean_final_adoption]),
        delimiter=",",
        header="beta_I,mean_X_final",
        comments="",
    )
    print(f"Saved results to: {csv_path}")

    # --- Plot sensitivity curve ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(beta_values, mean_final_adoption, "-o", markersize=4)

    ax.set_xlabel("βI (infrastructure feedback strength)")
    ax.set_ylabel("Final adoption X*")
    ax.set_title("Sensitivity of Final Adoption to βI")
    ax.set_ylim(0, 1.0)

    fig_path = os.path.join(plots_dir, "sensitivity_betaI.png")
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {fig_path}")


if __name__ == "__main__":
    main()