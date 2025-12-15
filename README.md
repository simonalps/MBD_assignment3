# Assignment 3 – Network-Based EV Adoption Model

This repository contains the code and experiments for **Assignment 3**. The project analyses baseline diffusion dynamics, network-dependent adoption behavior, and policy interventions using agent-based simulations on different network structures.

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/simonalps/MBD_assignment3
cd MBD_assignment3
```

### 2. Create and activate the virtual environment
```bash
python -m venv venv
```

Activate the environment:

On Windows:
```bash
venv\Scripts\activate
```

On macOS / Linux:
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
The project relies only on standard Python libraries and does not require any external datasets.

---

## Running the Experiments

All experiments should be run from the repository root directory.
Each script automatically creates the appropriate output folders if they do not already exist.

### Part 1 – Baseline Dynamics

This part investigates the intrinsic dynamics of the adoption model in the absence of policy interventions. It focuses on identifying tipping points, bistability regions, and sensitivity to initial conditions.

```bash
python part1.py
```

Outputs:

Heatmaps of final adoption over (X0, I0)

Heatmaps of final adoption over (X0, a(I)/b)

Phase trajectories (I(t), X(t))

Sensitivity analysis for infrastructure feedback strength βI​.



Results are saved to:

results/part1/

plots/part1/

### Part 2 – Network Structure Effects

This part examines how adoption dynamics depend on network topology. A single comprehensive simulation run is used to generate all figures, ensuring consistency across analyses.

```bash
python part2.py
```

Outputs:

Adoption heatmaps

Mean adoption trajectories

Probability of reaching a high-adoption equilibrium.


Results are saved to:

results/part2/

plots/part2/

### Part 3 – Policy Intervention Analysis

This part evaluates policy interventions derived from network vulnerabilities. Policy strength, timing windows, and network dependence are systematically explored.

```bash
python part3.py
```

Outputs:

Baseline vs intervention charts

Policy intensity comparisons

Sensitivity to intervention timing windows.


Results are saved to:

results/part3/

plots/part3/

---

## Notes

All simulations include stochastic elements arising from network generation and initial conditions.
Outcomes are averaged over multiple realizations to ensure robustness.
Network parameters are chosen to maintain comparable mean degree across topologies.
Core model logic, experiment orchestration, and plotting are modularized in the utils/ folder.
