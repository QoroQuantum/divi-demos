# Portfolio Optimization with Quantum Algorithms

> 🚀 **Skip the local bottleneck.** Qoro is giving away **$100 in free cloud compute credits.**
> Get your API key at **[dash.qoroquantum.net](https://dash.qoroquantum.net)** to run this at scale.

## Why Cloud?

A 484-asset portfolio decomposes into dozens of QAOA sub-problems. QoroService runs every partition concurrently — locally they run sequentially.

## Step 0: Set Your API Key

```bash
pip install qoro-divi numpy pandas
```

Create a `.env` file in the repo root:

```
QORO_API_KEY="your_api_key_here"
```

👉 **[Get your free API key →](https://dash.qoroquantum.net)**

## Key Concepts

- **QUBO Formulation**: `Minimize: Risk - λ·Return`, where λ balances risk and return
- **Modularity-Spectral Partitioning**: Newman 2006 modularity-maximizing spectral bisection on the correlation matrix; mathematically distinct from divi's built-in Fiedler/METIS/KL partitioners
- **QAOA / PCE**: Solves each partition's QUBO; PCE compresses binary variables into logarithmically fewer qubits
- **Beam-search Aggregation**: `PartitioningProgramEnsemble.aggregate_results` stitches per-partition top-N candidates into a global solution

## Files

| File | Description |
|------|-------------|
| `portfolio_optimization.ipynb` | End-to-end demo: 8-asset QAOA, 484-asset modularity-partitioned QAOA, PCE variation. Defines `ModularityDecomposer` (a `hybrid.traits.ProblemDecomposer` adapter for `BinaryOptimizationProblem`) inline. |
| `modularity_spectral_partitioning.py` | Newman modularity-spectral partitioning algorithm |
| `utils.py` | Markowitz QUBO builder, financial metrics, evaluation helpers |
| `visualization.py` | Correlation heatmaps, partition analysis, λ guidance |

---

👉 **Ready for real-world portfolios?** [Get your API key](https://dash.qoroquantum.net) and scale with QoroService.
