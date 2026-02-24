# Travelling Salesman Problem via QUBO + QAOA

Quantum solution for the **Travelling Salesman Problem (TSP)** — finding the
shortest route that visits every city exactly once and returns to the start.
Uses a standard **QUBO encoding** solved with **QAOA** on the
[Divi](https://dash.qoroquantum.net) quantum SDK.

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# 2. Install dependencies
pip install qoro-divi

# 3. Run the script
python travelling_salesman.py
```

Or open `travelling_salesman.ipynb` in Jupyter:

```bash
pip install jupyter
jupyter notebook travelling_salesman.ipynb
```

## Requirements

Python **3.10+** is required. The `qoro-divi` package bundles all needed
dependencies (`dimod`, `numpy`, `matplotlib`).

## What It Does

Finds the shortest cyclic tour through **N randomly placed cities**:

1. Encodes the TSP as a QUBO with **n² binary variables** using one-hot position encoding
   (`x_{i,t} = 1` ⟺ city *i* is at position *t* in the tour).
2. Enforces constraints via penalty terms:
   - **One city per time step** — exactly one city is visited at each position.
   - **One time step per city** — each city appears exactly once in the tour.
3. Solves the QUBO with **QAOA** using Divi's `MonteCarloOptimizer`.
4. Decodes the best feasible bitstring into a tour and compares against a
   classical brute-force optimum.

## Files

| File | Description |
|------|-------------|
| `travelling_salesman.py` | Standalone script with full comments |
| `travelling_salesman.ipynb` | Jupyter notebook with markdown explanations |
| `README.md` | This file |

## Configuration

Edit these constants at the bottom of the script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_CITIES` | 4 | Number of cities (keep ≤ 5 for local simulation; n² qubits) |
| `SEED` | 42 | Random seed for city placement |
| `N_LAYERS` | 3 | QAOA circuit depth |
| `MAX_ITERATIONS` | 20 | Optimizer iterations |
| `SHOTS` | 20 000 | Measurement samples per circuit evaluation |

### Scaling Note

The QUBO uses n² binary variables (qubits), so:

| Cities | Qubits | Brute-force permutations |
|--------|--------|--------------------------|
| 3 | 9 | 2 |
| 4 | 16 | 6 |
| 5 | 25 | 24 |
| 6 | 36 | 120 |

For **≤ 5 cities** the problem runs comfortably on a local simulator.
For larger instances, use the QoroService cloud backend.

## Expected Output

1. A plot of the randomly generated city locations.
2. Terminal output showing QAOA progress and the top bitstring analysis.
3. A side-by-side comparison plot of the classical and quantum tours.
4. A summary comparing tour distances:

```
🗺️  Travelling Salesman — Classical vs. Quantum
======================================================================
   Classical optimum:  tour = [0, 2, 3, 1]
                       distance = 2.1547
   QAOA result:        tour = [0, 2, 3, 1]
                       distance = 2.1547
   🎉 QAOA found the optimal tour!
======================================================================
```

## Remote Execution (QoroService)

To run on Qoro's cloud backend:

1. Obtain an API key from [dash.qoroquantum.net](https://dash.qoroquantum.net).
2. Set the environment variable:
    ```bash
    export QORO_API_KEY="your_api_key_here"
    ```
3. Set `USE_CLOUD = True` in the script.
