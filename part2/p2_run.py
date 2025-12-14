import os
from typing import List, Dict

import numpy as np
import pandas as pd

from utils.ev_experiments import run_timeseries_trial


def main():
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # -----------------------------
    # Global experiment settings
    # -----------------------------
    network_types = ["random", "BA", "WS"]
    seeding_strategies = ["random", "degree"]  # maps to init_method
    X0_values = np.linspace(0.0, 1.0, 21) 

    n_runs_per_setting = 20
    T = 200
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

    rows: List[Dict] = []

    for network_type in network_types:
        for seeding_strategy in seeding_strategies:
            # Map seeding_strategy to init_method
            if seeding_strategy == "random":
                init_method = "random"
            elif seeding_strategy == "degree":
                init_method = "degree"
            else:
                raise ValueError(f"Unknown seeding_strategy: {seeding_strategy}")

            for X0 in X0_values:
                for run_id in range(n_runs_per_setting):

                    seed = seed_base + run_id

                    # Network-specific parameters
                    scenario = base_params.copy()
                    scenario.update(
                        dict(
                            network_type=network_type,
                            X0_frac=float(X0),
                            init_method=init_method,
                        )
                    )

                    n_nodes = scenario["n_nodes"]
                    k_target = 6.0

                    if network_type == "random":
                        # ER: choose p so mean degree ≈ k_target
                        scenario["p"] = k_target / (n_nodes - 1)   # ≈ 6/399 ≈ 0.015
                        scenario["m"] = 3  # unused by ER, but set for completeness
                    elif network_type == "BA":
                        # BA: mean degree ≈ 2m
                        scenario["m"] = 3
                        scenario["p"] = scenario.get("p", 0.0)  # unused by BA
                    elif network_type == "WS":
                        # WS: degree k = 2m, we want k_target → m = k_target / 2
                        scenario["m"] = 3         # → k = 6
                        scenario["p"] = 0.1       # rewiring probability
                    else:
                        raise ValueError(f"Unknown network_type: {network_type}")

                    # Run one simulation
                    X_series, I_series, _df = run_timeseries_trial(
                        T=T,
                        scenario_kwargs=scenario,
                        seed=seed,
                        policy=None,
                        strategy_choice_func="imitate",
                        tau=1.0,
                    )

                    # Store each time step as a row
                    for t, (x_val, i_val) in enumerate(zip(X_series, I_series)):
                        rows.append(
                            dict(
                                network_type=network_type,
                                seeding_strategy=seeding_strategy,
                                run_id=run_id,
                                seed=int(seed),
                                X0=float(X0),
                                t=int(t),
                                X=float(x_val),
                                I=float(i_val),
                            )
                        )

                    # Optional: print some progress so you see it’s running
                    print(
                        f"Done: net={network_type}, seed={seeding_strategy}, "
                        f"X0={X0:.2f}, run_id={run_id}"
                    )

    # Build DataFrame and save
    df_all = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "part2_all_timeseries.csv")
    df_all.to_csv(out_path, index=False)
    print(f"Saved Part 2 data to: {out_path}")


if __name__ == "__main__":
    main()