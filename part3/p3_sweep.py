# part3_sweep_heatmap.py
# Sweep X0_hubs (x-axis) and delta_a0 (y-axis) for each network type,
# and plot a heatmap of mean final adoption X(T).

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.ev_experiments import run_timeseries_trial, policy_subsidy_factory

NETWORK_LABELS = {"random": "ER", "BA": "BA", "WS": "WS"}

def network_params_like_part2(network_type: str, n_nodes: int) -> Dict:
    """Match Part 2: target mean degree ~6 for ER/BA/WS."""
    k_target = 6.0
    if network_type == "random":
        return dict(p=k_target / (n_nodes - 1), m=3)   # m unused for ER
    if network_type == "BA":
        return dict(m=3, p=0.0)                        # p unused for BA
    if network_type == "WS":
        return dict(m=3, p=0.1)                        # WS: k=2m=6, rewiring p=0.1
    raise ValueError(f"Unknown network_type: {network_type}")


def run_cell_mean_final_X(
    *,
    scenario_base: Dict,
    T: int,
    n_trials: int,
    seed_base: int,
    subsidy_start: int,
    subsidy_end: int,
    delta_a0: float,
) -> float:
    """Run n_trials and return mean final adoption X(T-1)."""
    policy = policy_subsidy_factory(
        start=subsidy_start,
        end=subsidy_end,
        delta_a0=delta_a0,
        delta_beta_I=0.0,  # keep infra lever off for now
    )

    finals: List[float] = []
    for i in range(n_trials):
        seed = seed_base + i
        X, I, df = run_timeseries_trial(
            T=T,
            scenario_kwargs=scenario_base,
            seed=seed,
            policy=policy,
            strategy_choice_func="imitate",
            tau=1.0,
        )
        finals.append(float(X[-1]))
    return float(np.mean(finals))


def plot_heatmap(
    Z: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    title: str,
    out_path: str,
    cmap: str = "plasma",
):
    """Heatmap with X0_hubs on x-axis and delta_a0 on y-axis."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)

    extent = [x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()]

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
    )

    # ticks similar cleanliness to plot_heatmaps.py
    ax.set_xticks(x_vals)
    ax.set_xticklabels([f"{x:.2f}" for x in x_vals])
    ax.set_yticks(y_vals)
    ax.set_yticklabels([f"{y:.1f}" for y in y_vals])

    ax.set_xlabel("Targeted giveaway $X_{0,\\,hubs}$")
    ax.set_ylabel("Subsidy strength $\\Delta a_0$")
    ax.set_title(title)

    # Colorbar on LEFT, same style choice as plot_heatmaps.py
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, location="left", pad=0.02)
    cbar.set_label("Mean final adoption $\\bar{X}(T)$")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # --- Sweep settings ---
    x0_random = 0.10  # your background random adoption level (adjust if you want)
    x0_hubs_vals = np.linspace(0.01, 0.05, 5)   # 0.01, 0.02, ..., 0.05
    delta_a0_vals = np.linspace(0.1, 0.6, 6)     # 0.1, 0.2, ..., 0.6

    # --- Runtime settings ---
    T = 50              
    n_trials = 40        # increase later if needed
    seed_base = 42

    # subsidy window (early)
    subsidy_start = 0
    subsidy_end = 15

    # --- Base model params (match Part 2) ---
    base_params = dict(
        ratio=2.3,
        beta_I=2.0,
        b=1.0,
        g_I=0.05,
        I0=0.05,
        n_nodes=400,
        collect=True,
    )

    networks = ["random", "BA", "WS"]

    all_rows = []

    for net in networks:
        net_kwargs = network_params_like_part2(net, n_nodes=base_params["n_nodes"])

        # Z will be shaped (len(delta_a0_vals), len(x0_hubs_vals))
        Z = np.zeros((len(delta_a0_vals), len(x0_hubs_vals)), dtype=float)

        for yi, da0 in enumerate(delta_a0_vals):
            for xi, x0h in enumerate(x0_hubs_vals):
                # IMPORTANT: keep hybrid init consistent with your current utilities:
                # - random part uses X0_frac
                # - hubs part uses X0_hubs
                scenario = dict(
                    base_params,
                    network_type=net,
                    **net_kwargs,
                    init_method="hybrid",
                    X0_frac=x0_random,
                    X0_hubs=float(x0h),
                    high=True,
                )

                mean_final = run_cell_mean_final_X(
                    scenario_base=scenario,
                    T=T,
                    n_trials=n_trials,
                    seed_base=seed_base,
                    subsidy_start=subsidy_start,
                    subsidy_end=subsidy_end,
                    delta_a0=float(da0),
                )
                Z[yi, xi] = mean_final

                all_rows.append(
                    dict(
                        network_type=net,
                        X0_random=x0_random,
                        X0_hubs=float(x0h),
                        delta_a0=float(da0),
                        T=T,
                        n_trials=n_trials,
                        subsidy_start=subsidy_start,
                        subsidy_end=subsidy_end,
                        mean_final_X=mean_final,
                    )
                )

                print(f"[{net}] X0_hubs={x0h:.2f} delta_a0={da0:.1f} -> mean X(T)={mean_final:.3f}")

        # save heatmap per network
        net_label = NETWORK_LABELS[net]

        tag = f"net={net_label}_X0r={x0_random:.2f}_T={T}_trials={n_trials}_sub=0-{subsidy_end}"
        out_png = os.path.join("plots", f"part3_heatmap_mean_final_{tag}.png")

        plot_heatmap(
            Z=Z,
            x_vals=x0_hubs_vals,
            y_vals=delta_a0_vals,
            title=f"Mean final adoption $\\bar{{X}}(T)$ — {net_label}",
            out_path=out_png,
        )

        print("Saved:", out_png)

    # save raw sweep table
    df = pd.DataFrame(all_rows)
    out_csv = os.path.join("results", "part3_sweep_mean_final_heatmap_data.csv")
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()