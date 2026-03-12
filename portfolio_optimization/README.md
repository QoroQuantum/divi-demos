# Portfolio Optimization with Quantum Algorithms

> 🚀 **Skip the local bottleneck.** Qoro is giving away **$100 in free cloud compute credits.**
> Get your API key at **[dash.qoroquantum.net](https://dash.qoroquantum.net)** to run this at scale.

## Why Cloud?

A portfolio of **480 assets** gets partitioned into dozens of sub-problems, each requiring its own QAOA optimization with multiple circuit evaluations per iteration. Running these sequentially on a local simulator takes **hours**. QoroService runs every partition **in parallel** — all portfolios optimized simultaneously.

## Step 0: Set Your API Key

```bash
pip install qoro-divi
```

Create a `.env` file in the repo root:

```
QORO_API_KEY="your_api_key_here"
```

👉 **[Get your free API key →](https://dash.qoroquantum.net)**

## Overview

Portfolio optimization selects assets that maximize return while minimizing risk. This is formulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem, solved using QAOA.

For large portfolios, the problem is partitioned using **spectral graph partitioning** based on asset correlations. Each partition is solved independently, and solutions are aggregated into a global portfolio.

## Key Concepts

- **QUBO Formulation**: `Minimize: Risk - λ·Return`, where λ balances risk and return
- **Spectral Partitioning**: Groups assets by correlation structure
- **QAOA**: Solves each partition's QUBO problem using quantum circuits
- **Solution Aggregation**: Combines partition solutions into a global portfolio

## Files

| File | Description |
|------|-------------|
| `portfolio_optimization.ipynb` | Main QAOA workflow notebook |
| `portfolio_optimization_pce.ipynb` | PCE alternative (logarithmic qubit compression) |
| `utils.py` | QUBO building, solution aggregation, financial metrics |
| `visualization.py` | Correlation heatmaps, partition analysis |
| `qubo_batch.py` | Parallel QUBO solving via QAOA |
| `modularity_spectral_partitioning.py` | Spectral partitioning implementation |

## Quick Start

```bash
jupyter notebook portfolio_optimization.ipynb
```

Run cells sequentially to:
1. Load and scale financial data
2. Partition assets using spectral clustering
3. Build QUBO matrices per partition
4. Solve each partition with QAOA (quantum) and ExactSolver (classical)
5. Aggregate and compare solutions using financial metrics

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `MAX_PARTITION_SIZE` | Max assets per partition | 20 |
| `LAMBDA_PARAM` | Risk-return trade-off (higher = favor return) | 0.75 |
| `n_layers` | QAOA circuit depth | 2 |
| `max_iterations` | Optimization iterations | 15 |

## Results

The comparison between QAOA and ExactSolver provides insights into:
- **Solution Quality**: How close QAOA gets to optimal
- **Risk-Return Trade-off**: Whether QAOA finds better risk-adjusted returns
- **Asset Selection**: Which assets each method selects

---

👉 **Ready for real-world portfolios?** [Get your API key](https://dash.qoroquantum.net) and scale with QoroService.
