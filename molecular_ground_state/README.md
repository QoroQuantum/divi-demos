# Molecular Ground State via VQE using `divi`

This example computes the **potential energy surface (PES)** of the hydrogen molecule (H₂) — ground-state energy as a function of bond length — using the `divi` quantum programming framework.

It showcases Divi's VQE features:

- **`VQEHyperparameterSweep`** — runs a grid search over ansätze × Hamiltonians in parallel via `ProgramBatch`
- **`VQE` with molecular Hamiltonians** — finds the ground-state energy of a qubit Hamiltonian
- **Multiple ansätze** — compare UCCSD (chemistry gold standard) vs. HardwareEfficient (gate-efficient)

## Project Structure

```text
.
├── molecular_ground_state.py       # The entry point
├── molecular_ground_state.ipynb    # Interactive notebook tutorial
└── README.md                       # This file
```

## Prerequisites

```bash
pip install qoro-divi matplotlib
```

## Usage

### 1. Run

```bash
python molecular_ground_state.py
```

### 2. Expected Output

1. The terminal will show VQE progress for each (ansatz, bond_length) combination.
2. A summary table will print the equilibrium bond length and energy for each ansatz.
3. A Matplotlib plot will display the H₂ potential energy surface:

```
  🔬 Potential Energy Surface — Results
======================================================================

  UCCSDAnsatz:
    Equilibrium bond length : 0.740 Å
    Ground-state energy     : -1.136189 Ha

  HardwareEfficient:
    Equilibrium bond length : 0.740 Å
    Ground-state energy     : -1.130422 Ha

  🏆 Best overall: UCCSDAnsatz at scale=1.00  →  E = -1.136189 Ha
======================================================================
```

## Configuration

| Parameter           | Description                                                                      |
|---------------------|----------------------------------------------------------------------------------|
| BASE_BOND_LENGTH    | Equilibrium H–H distance in Ångströms (default: 0.74).                           |
| bond_scale_factors  | Multipliers for the bond length scan (default: 15 points from 0.5× to 3.0×).    |
| ansatze             | Ansatz circuits to compare: UCCSD, HardwareEfficient, etc.                       |
| max_iterations      | Optimizer iterations per VQE run (default: 15).                                  |
| shots               | Measurement samples per circuit (default: 10,000).                               |

## Remote Execution (QoroService)

1. Obtain an API key from [dash.qoroquantum.net](https://dash.qoroquantum.net).
2. Set the environment variable:
    ```bash
    export QORO_API_KEY="your_api_key_here"
    ```
3. Change `USE_CLOUD = True` in `molecular_ground_state.py`.
