"""Part 3 runner.

Order:
1) baseline vs policy per network
2) targeted sweep heatmaps (X0_hubs × delta_a0) per network
3) timing window sweep (mean final adoption vs subsidy_end)
"""

import os
import numpy as np
import pandas as pd

from utils.ev_experiments import (
    collect_intervention_trials,
    traces_to_long_df,
    part3_targeted_sweep_mean_final_df,
    part3_timing_sweep_df,
    NETWORK_LABELS_PART3,
    part3_set_network_params,
)
from utils.ev_plotting import (
    plot_fanchart,
    plot_part3_tipping_probability_over_time,
    plot_part3_targeted_sweep_heatmap,
    plot_part3_timing_sweep,
)

def _make_dirs():
    os.makedirs(os.path.join("results", "part3"), exist_ok=True)
    os.makedirs(os.path.join("plots", "part3"), exist_ok=True)

def main():
    _make_dirs()

    # -----------------------------
    # Shared runtime settings
    # -----------------------------
    T = 50
    n_trials = 40
    seed_base = 42
    max_workers = 8
    strategy_choice_func = "imitate"
    tau = 1.0

    # -----------------------------
    # Base parameters (match Part 2)
    # -----------------------------
    base = dict(
        ratio=2.3,
        beta_I=2.0,
        b=1.0,
        g_I=0.05,
        I0=0.05,
        n_nodes=400,
        collect=True,
    )

    network_types = ["random", "BA", "WS"]

    # -----------------------------
    # Policy design choices (Part 3)
    # -----------------------------
    X0_random = 0.10
    X0_hubs = 0.03
    subsidy = dict(start=0, end=15, delta_a0=0.4, delta_beta_I=0.0)

    # ==================================================
    # 1) Baseline vs policy (per network)
    # ==================================================
    index_rows = []
    for net in network_types:
        net_label = NETWORK_LABELS_PART3.get(net, net)
        net_kwargs = part3_set_network_params(net, n_nodes=base["n_nodes"], k_target=6.0, ws_rewire_p=0.1)

        # baseline: random seeds only
        scenario_baseline = dict(
            base,
            network_type=net,
            **net_kwargs,
            init_method="random",
            X0_frac=X0_random,
        )

        # policy: hybrid init + subsidy window
        scenario_policy = dict(
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
            scenario_intervention=scenario_policy,
            subsidy_params=subsidy,
            max_workers=max_workers,
            seed_base=seed_base,
            strategy_choice_func=strategy_choice_func,
            tau=tau,
        )

        # Save summaries
        base_csv = os.path.join("results", "part3", f"baseline_summary_{net_label}.csv")
        pol_csv = os.path.join("results", "part3", f"policy_summary_{net_label}.csv")
        df_base.to_csv(base_csv, index=False)
        df_pol.to_csv(pol_csv, index=False)

        # Fan chart
        traces = traces_to_long_df(baseline_X, policy_X)
        fan_png = os.path.join(
            "plots",
            "part3",
            f"fanchart_net={net_label}_X0r={X0_random:.2f}_X0h={X0_hubs:.2f}_sub=0-{subsidy['end']}_da0={subsidy['delta_a0']:.2f}.png",
        )
        plot_fanchart(traces, out_path=fan_png)

        # Tipping probability over time
        tip_png = os.path.join(
            "plots",
            "part3",
            f"tipping_net={net_label}_X0r={X0_random:.2f}_X0h={X0_hubs:.2f}_sub=0-{subsidy['end']}_da0={subsidy['delta_a0']:.2f}_thr=0.80.png",
        )
        plot_part3_tipping_probability_over_time(
            baseline_X,
            policy_X,
            threshold=0.8,
            subsidy_end=subsidy["end"],
            out_path=tip_png,
            title=f"High-adoption probability (X(t) ≥ 0.8) — {net_label}",
        )

        index_rows.append(
            dict(
                network_type=net,
                network_label=net_label,
                X0_random=X0_random,
                X0_hubs=X0_hubs,
                sub_start=subsidy["start"],
                sub_end=subsidy["end"],
                delta_a0=subsidy["delta_a0"],
                baseline_final_mean=df_base["X_mean"].iloc[-1],
                policy_final_mean=df_pol["X_mean"].iloc[-1],
                baseline_summary_csv=base_csv,
                policy_summary_csv=pol_csv,
                fanchart_png=fan_png,
                tipping_png=tip_png,
            )
        )

        print(f"[done] {net_label} baseline mean X(T)={index_rows[-1]['baseline_final_mean']:.3f} | policy mean X(T)={index_rows[-1]['policy_final_mean']:.3f}")

    pd.DataFrame(index_rows).to_csv(os.path.join("results", "part3", "part3_overall_index.csv"), index=False)

    # ==================================================
    # 2) Targeted sweep heatmaps (per network)
    # ==================================================
    x0_hubs_vals = np.linspace(0.01, 0.05, 5)
    delta_a0_vals = np.linspace(0.1, 0.6, 6)

    sweep_df = part3_targeted_sweep_mean_final_df(
        network_types=network_types,
        base_params=base,
        X0_random=X0_random,
        x0_hubs_vals=x0_hubs_vals,
        delta_a0_vals=delta_a0_vals,
        subsidy_start=0,
        subsidy_end=15,
        T=T,
        n_trials=n_trials,
        seed_base=seed_base,
        progress=True,
    )
    sweep_csv = os.path.join("results", "part3", "part3_sweep_mean_final_heatmap_data.csv")
    sweep_df.to_csv(sweep_csv, index=False)

    for net in network_types:
        net_label = NETWORK_LABELS_PART3.get(net, net)
        net_kwargs = part3_set_network_params(net, n_nodes=base["n_nodes"], k_target=6.0, ws_rewire_p=0.1)
        sub = sweep_df[sweep_df["network_type"] == net].copy()

        out_png = os.path.join(
            "plots",
            "part3",
            f"heatmap_mean_final_net={net_label}_X0r={X0_random:.2f}_T={T}_trials={n_trials}_sub=0-15.png",
        )
        plot_part3_targeted_sweep_heatmap(
            sub,
            title=rf"Mean final adoption X(T) — {net_label}",
            out_path=out_png,
            cmap="plasma",
        )
        print("Saved:", out_png)

    # ==================================================
    # 3) Timing sweep (simple image)
    # ==================================================
    subsidy_ends = [2, 3, 4, 5] 
    timing_df = part3_timing_sweep_df(
        network_types=network_types,
        base_params=base,
        X0_random=X0_random,
        X0_hubs=0.03,
        delta_a0=0.40,
        subsidy_ends=subsidy_ends,
        T=T,
        n_trials=n_trials,
        seed_base=seed_base,
        progress=True,
    )
    timing_csv = os.path.join("results", "part3", "part3_timing_sweep.csv")
    timing_df.to_csv(timing_csv, index=False)

    timing_png = os.path.join("plots", "part3", "part3_timing_sweep_mean_final.png")
    plot_part3_timing_sweep(timing_df, out_path=timing_png)
    print("Saved:", timing_png)


if __name__ == "__main__":
    main()