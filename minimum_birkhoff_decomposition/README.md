# Minimum Birkhoff Decomposition

> 🚀 **Skip the local bottleneck.** Qoro is giving away **$100 in free cloud compute credits.**
> Get your API key at **[dash.qoroquantum.net](https://dash.qoroquantum.net)** to run this at scale.

## Why Cloud?

Each iteration evaluates parameterized circuits, then a multi-threaded classical `black_box_optimizer` (CPLEX) decodes measured bitstrings into the best convex combination of permutations. As matrix dimensions grow, the number of permutations explodes. QoroService offloads the **circuit evaluations** so the classical optimizer never waits for quantum results.

## Step 0: Set Your API Key

```bash
pip install qoro-divi docplex cplex
```

Create a `.env` file in the repo root:

```
QORO_API_KEY="your_api_key_here"
```

👉 **[Get your free API key →](https://dash.qoroquantum.net)**

## What It Does

Birkhoff decomposition is a *non-VQE* problem — its cost function isn't a Hamiltonian expectation value, but a CPLEX integer program over measured bitstrings. This demo shows how to wire that into Divi by composing two primitives directly: a `CircuitPipeline` that returns raw shot histograms, and a Divi optimizer that drives any Python `cost_fn(params) -> float` you give it.

If your problem fits the shape *"sample circuits, then do classical work on the histograms,"* it fits this template — Hamiltonian or not.

Under the hood:

1. A standalone **`CircuitPipeline`** (`PennyLaneSpecStage → MeasurementStage(COUNTS) → ParameterBindingStage`) maps parameter sets to raw shot histograms.
2. A plain `cost_fn(params)` decodes each bitstring into a permutation combination, runs the multi-threaded **`black_box_optimizer`** (CPLEX) with caching, and returns a scalar loss.
3. A Divi optimizer (`ScipyOptimizer` or `MonteCarloOptimizer`) iterates `cost_fn` to convergence via its public `optimize(...)` API.

## Files

| File | Description |
|---|---|
| `birkhoff_decomposition.ipynb` | Walkthrough notebook — recommended starting point |
| `birkhoff.py` | Core logic: `run_birkhoff(...)` orchestrating pipeline + optimizer |
| `main.py` | CLI runner with data loading, argparse, visualization |
| `requirements.txt` | Demo-specific deps (qoro-divi, docplex, cplex) |

## Quick Start

```bash
# Default parameters (4×4 sparse matrix)
python main.py

# Specific experiment
python main.py -n 4 -k 2 -inst 5 -it 20

# View all options
python main.py --help
```

## Arguments

| Flag | Name | Description | Default |
|---|---|---|---|
| `-n` | `--dim` | Matrix dimension | `4` |
| `-k` | `--comb` | Number of permutations | `2` |
| `-m` | `--matrix_type` | `sparse` or `dense` | `sparse` |
| `-inst` | `--instance` | Problem instance (1-10) | `1` |
| `-it` | `--iterations` | Max optimizer iterations | `10` |
| `-opt` | `--optimizer` | Optimizer (`Cobyla` or `MonteCarlo`) | `Cobyla` |

## Citation

> G. S. Barron, et al., "Quantum Optimization Benchmarking Library: The Intractable Decathlon," arXiv:2504.03832 [quant-ph], (2025).

---

👉 **Ready for larger matrices?** [Get your API key](https://dash.qoroquantum.net) and scale with QoroService.
