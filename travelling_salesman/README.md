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
3. Solves the TSP using three unique quantum approaches:
   - **Direct QAOA:** Solves exactly the uncompressed QUBO mapping for small problems (uses $n^2$ qubits).
   - **Partitioned QAOA (`QUBOPartitioningQAOA`):** Enables solving classically prohibitive QUBO problems by automatically decomposing the interaction graph into smaller manageable limits, solving them in parallel via Divi, and merging results natively.
   - **PCE (Pauli Correlation Encoding):** Compress QUBO variables to a base-2 smaller representation ($O(\log(n))$ scaling), reducing the qubit requirements significantly for NISQ devices.
4. Decodes the quantum outputs, performs a greedy heuristic repair, and compares the relative accuracy versus classical brute-force for all scenarios.


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
| Parameter | Description |
|-----------|-------------|
| `N_CITIES_SMALL` | Number of cities for Direct QAOA and PCE (default 4) |
| `N_CITIES_LARGE` | Number of cities for Partitioned QAOA (default 8) |
| `USE_CLOUD` | Uses `QoroService` instead of local simulator if set to `True` |

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

```text
  � Summary — Three Divi Approaches to TSP
======================================================================

  Method                              Cities   Qubits   Distance    Ratio
  ───────────────────────────────────────────────────────────────────
  Classical (brute force)                  4        —     1.9633    1.000
  A: Direct QAOA                           4       16     1.9633    1.000
  C: PCE (poly encoding)                   4        6     1.9633    1.000

  Classical (brute force)                  8        —     2.4049    1.000
  B: Partitioned QAOA                      8      ≤15     4.1914    1.743
```

## Remote Execution (QoroService)

To run on Qoro's cloud backend:

1. Obtain an API key from [dash.qoroquantum.net](https://dash.qoroquantum.net).
2. Set the environment variable in a .env file:
    ```bash
    QORO_API_KEY="your_api_key_here"
    ```
3. Set `USE_CLOUD = True` in the script.
