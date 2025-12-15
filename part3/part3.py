"""Part 3 runner (kept intentionally small).

Runs baseline vs intervention for EACH network type, using the same network
parameter choices as Part 2 (mean degree ~6 for ER/BA/WS).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.ev_experiments import collect_intervention_trials, traces_to_long_df
from utils.ev_plotting import plot_fanchart

NETWORK_LABELS = {
    "random": "ER",
    "BA": "BA",
    "WS": "WS",
}

def set_network_params(network_type: str, n_nodes: int) -> dict:
    """Match the network parameter choices used in Part 2."""
    k_target = 6.0

    if network_type == "random":
        # ER: choose p so mean degree ≈ k_target
        return dict(p=k_target / (n_nodes - 1), m=3)  # m unused for ER
    if network_type == "BA":
        # BA: mean degree ≈ 2m -> choose m=3 to get ~6
        return dict(m=3, p=0.0)  # p unused for BA
    if network_type == "WS":
        # WS: k = 2m -> choose m=3 to get k=6; p is rewiring probability
        return dict(m=3, p=0.1)

    raise ValueError(f"Unknown network_type: {network_type}")

def plot_tipping_probability_over_time(
    baseline_X,
    policy_X,
    threshold=0.8,
    subsidy_end=None,
    out_path=None,
    title=None,
):
    """
    Plots P(X(t) >= threshold) over time for baseline vs policy.
    baseline_X / policy_X are lists of 1D arrays of length T.
    """
    T = len(baseline_X[0])

    base_prob = np.zeros(T)
    pol_prob = np.zeros(T)

    for t in range(T):
        base_prob[t] = np.mean([x[t] >= threshold for x in baseline_X])
        pol_prob[t] = np.mean([x[t] >= threshold for x in policy_X])

    plt.figure(figsize=(7, 4))
    plt.plot(base_prob, label="Baseline")
    plt.plot(pol_prob, label="Policy")

    if subsidy_end is not None:
        plt.axvline(subsidy_end, linestyle="--", alpha=0.6, label="Subsidy end")

    plt.ylim(-0.02, 1.02)
    plt.xlabel("Time")
    plt.ylabel(f"P(X(t) ≥ {threshold})")
    plt.title(title or "High-adoption probability over time")
    plt.legend()
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=200)
        plt.close()
    else:
        plt.show()

    return base_prob, pol_prob

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    T = 50
    n_trials = 40
    seed_base = 42

    # -----------------------------
    # Base parameters (match Part 2)
    # -----------------------------
    # Part 2 uses g_I=0.05, so we keep it identical for comparability.
    base = dict(
        ratio=2.3,
        beta_I=2.0,
        b=1.0,
        g_I=0.05,
        I0=0.05,
        n_nodes=400,
        collect=True,
    )

    # -----------------------------
    # Policy design choices
    # -----------------------------
    # Baseline init: "realistic" background adoption is random.
    X0_random = 0.1

    # Policy includes an additional T=0 giveaway to high-degree nodes
    # (interpreted as FRACTION: 0.03 = 3% of nodes).
    X0_hubs = 0.03

    # Timed subsidy in the early critical window (first ~15 steps)
    #
    # Recommended delta_a0:
    # Baseline a0 is computed (in the utilities) as:
    #   a0 = ratio*b - beta_I*I0 = 2.3*1 - 2.0*0.05 = 2.2
    # Choosing delta_a0=0.4 raises a0 to 2.6 during the window: +0.4 / 2.2 ≈ +18%.
    # That’s a strong early push without making EV universally dominant regardless of peers.
    subsidy = dict(start=0, end=15, delta_a0=0.4, delta_beta_I=0.0)

    network_types = ["random", "BA", "WS"]
    rows = []

    for net in network_types:
        net_kwargs = set_network_params(net, n_nodes=base["n_nodes"])

        # BASELINE: random seeds only
        scenario_baseline = dict(
            base,
            network_type=net,
            **net_kwargs,
            init_method="random",
            X0_frac=X0_random,
        )

        # POLICY: random background + high-degree giveaway at T=0
        scenario_intervention = dict(
            base,
            network_type=net,
            **net_kwargs,
            init_method="hybrid",
            X0_frac=X0_random,
            X0_hubs=X0_hubs,
            high=True,
        )

        baseline_X, baseline_I, policy_X, policy_I, df_base, df_pol = collect_intervention_trials(
            n_trials=n_trials,
            T=T,
            scenario_baseline=scenario_baseline,
            scenario_intervention=scenario_intervention,
            subsidy_params=subsidy,
            max_workers=8,  
            seed_base=seed_base,
            strategy_choice_func="imitate",
            tau=1.0,
        )

        net_label = NETWORK_LABELS[net]

        # Save per-network summary tables
        df_base_path = f"results/part3_baseline_summary_{net_label}.csv"
        df_pol_path  = f"results/part3_policy_summary_{net_label}.csv"
        df_base.to_csv(df_base_path, index=False)
        df_pol.to_csv(df_pol_path, index=False)

        # Plots (X(t)) per network
        traces = traces_to_long_df(baseline_X, policy_X)
        tag = f"net={net_label}_X0r={X0_random:.2f}_X0h={X0_hubs:.2f}_sub=0-15_da0={subsidy['delta_a0']:.2f}"
        fan_path = f"plots/part3_fanchart_{tag}.png"
        plot_fanchart(traces, out_path=fan_path)
        subsidy_end = subsidy["end"]
        tip_path = f"plots/part3_tipping_{tag}_thr=0.80.png"
        plot_tipping_probability_over_time(
            baseline_X,
            policy_X,
            threshold=0.8,
            subsidy_end=subsidy_end,
            out_path=tip_path,
            title=f"High-adoption probability (X(T) ≥ 0.8) — {net_label}",
        )

        # Record a small “index row” for a combined summary CSV
        rows.append(
            dict(
                network_type=net,
                X0_random=X0_random,
                X0_hubs=X0_hubs,
                sub_start=subsidy["start"],
                sub_end=subsidy["end"],
                delta_a0=subsidy["delta_a0"],
                baseline_final_mean=float(df_base["X_mean"].iloc[-1]),
                policy_final_mean=float(df_pol["X_mean"].iloc[-1]),
                baseline_final_p80=float(df_base["X_p80"].iloc[-1]) if "X_p80" in df_base.columns else np.nan,
                policy_final_p80=float(df_pol["X_p80"].iloc[-1]) if "X_p80" in df_pol.columns else np.nan,
                baseline_summary_csv=df_base_path,
                policy_summary_csv=df_pol_path,
                fanchart_png=fan_path,
                tipping_png=tip_path,
            )
        )

        print(f"[done] {net_label} | baseline mean X(T)={rows[-1]['baseline_final_mean']:.3f} "
              f"| policy mean X(T)={rows[-1]['policy_final_mean']:.3f}")

    pd.DataFrame(rows).to_csv("results/part3_overall_index.csv", index=False)
    print("Saved results/ and plots/ outputs for all networks.")


if __name__ == "__main__":
    main()