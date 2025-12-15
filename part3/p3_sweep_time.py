# p3_timing_sweep.py
# Sweep subsidy end time and plot mean final adoption per network.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.ev_experiments import run_timeseries_trial, policy_subsidy_factory


NETWORK_TYPES = ["random", "BA", "WS"]
NETWORK_LABELS = {"random": "ER", "BA": "BA", "WS": "WS"}


def set_network_params(network_type: str, n_nodes: int) -> dict:
    """Match Part 2: target mean degree ~6 for ER/BA/WS."""
    k_target = 6.0

    if network_type == "random":
        return dict(p=k_target / (n_nodes - 1), m=3)  # m unused for ER
    if network_type == "BA":
        return dict(m=3, p=0.0)  # p unused for BA
    if network_type == "WS":
        return dict(m=3, p=0.1)  # WS: k=2m=6, rewiring p=0.1

    raise ValueError(f"Unknown network_type: {network_type}")


def mean_final_adoption_for_setting(
    *,
    network_type: str,
    base_params: dict,
    X0_random: float,
    X0_hubs: float,
    delta_a0: float,
    subsidy_end: int,
    T: int,
    n_trials: int,
    seed_base: int,
) -> float:
    """Run n_trials and return mean final X(T-1)."""

    net_kwargs = set_network_params(network_type, n_nodes=base_params["n_nodes"])

    # Hybrid init: random background + hubs giveaway at t=0
    scenario = dict(
        base_params,
        network_type=network_type,
        **net_kwargs,
        init_method="hybrid",
        X0_frac=X0_random,   
        X0_hubs=X0_hubs,
        high=True,
    )

    policy = policy_subsidy_factory(
        start=0,
        end=subsidy_end,
        delta_a0=delta_a0,
        delta_beta_I=0.0,
    )

    finals = []
    for i in range(n_trials):
        seed = seed_base + i
        X, I, df = run_timeseries_trial(
            T=T,
            scenario_kwargs=scenario,
            seed=seed,
            policy=policy,
            strategy_choice_func="imitate",
            tau=1.0,
        )
        finals.append(float(X[-1]))

    return float(np.mean(finals))


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Fixed settings (Part 3 baseline choices)
    X0_random = 0.10
    X0_hubs = 0.03  
    delta_a0 = 0.40

    # Timing sweep
    subsidy_ends = [2, 3, 4, 5]

    # Runtime settings
    T = 100
    n_trials = 40
    seed_base = 42

    # Base params (match Part 2)
    base_params = dict(
        ratio=2.3,
        beta_I=2.0,
        b=1.0,
        g_I=0.05,
        I0=0.05,
        n_nodes=400,
        collect=True,
    )

    rows = []
    for net in NETWORK_TYPES:
        for end in subsidy_ends:
            mean_final = mean_final_adoption_for_setting(
                network_type=net,
                base_params=base_params,
                X0_random=X0_random,
                X0_hubs=X0_hubs,
                delta_a0=delta_a0,
                subsidy_end=end,
                T=T,
                n_trials=n_trials,
                seed_base=seed_base,
            )
            rows.append(
                dict(
                    network_type=net,
                    network_label=NETWORK_LABELS[net],
                    subsidy_end=end,
                    X0_random=X0_random,
                    X0_hubs=X0_hubs,
                    delta_a0=delta_a0,
                    T=T,
                    n_trials=n_trials,
                    mean_final_X=mean_final,
                )
            )
            print(f"[{NETWORK_LABELS[net]}] end={end:>2} -> mean X(T)={mean_final:.3f}")

    df = pd.DataFrame(rows)
    out_csv = "results/part3_timing_sweep.csv"
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    # --------- Plot: simple image ---------
    # One figure with 3 lines (ER/BA/WS) across subsidy end times.
    plt.figure(figsize=(7.5, 4.8))

    for net in NETWORK_TYPES:
        sub = df[df["network_type"] == net].sort_values("subsidy_end")
        plt.plot(sub["subsidy_end"], sub["mean_final_X"], marker="o", label=NETWORK_LABELS[net])

    plt.ylim(-0.02, 1.02)
    plt.xlabel("Subsidy end time")
    plt.ylabel("Mean final adoption  X(T)")
    plt.title("Effect of subsidy timing window on adoption (mean final X)")
    plt.legend()
    plt.tight_layout()

    out_png = "plots/part3_timing_sweep_mean_final.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved:", out_png)


if __name__ == "__main__":
    main()