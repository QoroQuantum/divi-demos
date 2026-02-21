# Economic Load Dispatch with Prohibited Operating Zones

Quantum-classical tutorial that solves the Economic Load Dispatch (ELD)
problem using **Pauli Correlation Encoding (PCE)** on the [Divi](https://pypi.org/project/divi/) quantum SDK.

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# 2. Install dependencies
pip install qoro-divi 

# 3. Run the script
python economic_load_dispatch.py
```

Or open `economic_load_dispatch.ipynb` in Jupyter:

```bash
pip install jupyter
jupyter notebook economic_load_dispatch.ipynb
```

## Requirements

Python **3.10+** is required. The `qoro-divi` package bundles all needed
dependencies (`dimod`, `pennylane`, `numpy`, `pymoo`).

## What It Does

Dispatches power across **3 generators** to meet a **195 MW demand** at
minimum fuel cost, while avoiding Prohibited Operating Zones (mechanical
vibration bands).

1. Encodes the dispatch as a QUBO with **12 binary variables** (4 qubits per generator).
2. Solves it with PCE-VQE using **polynomial encoding** (compresses 12 variables → 5 qubits).
3. Applies a **feasibility repair** heuristic to convert near-optimal quantum candidates into fully valid dispatches.
4. Compares the quantum result against a classical brute-force optimum.

## Files

| File | Description |
|------|-------------|
| `economic_load_dispatch.py` | Standalone script with full comments |
| `economic_load_dispatch.ipynb` | Jupyter notebook with markdown explanations |
| `README.md` | This file |

## Tuning Parameters

Edit these at the top of the script to experiment:

- **`DEMAND`** — target load in MW
- **`GENERATORS`** — add/remove generators, change cost curves or POZ ranges
- **`PENALTY_LAMBDA`** / **`POZ_MU`** — constraint penalty weights
- **`n_layers`** — PCE ansatz depth
- **`max_iterations`** — DE optimizer generations
- **`encoding_type`** — `"poly"` (fewer qubits) or `"dense"` (even fewer qubits)
