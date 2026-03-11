# Partitioned QAOA for MaxCut using `divi`

> 🚀 **Don't choke your local machine.** Qoro is giving away **$100 in free cloud compute credits.**
> Get your API key at **[dash.qoroquantum.net](https://dash.qoroquantum.net)** to run this at scale.

## Why Cloud?

A 50-node MaxCut graph needs a **2⁵⁰ statevector** — that's over **8 petabytes of RAM.** Even with clever partitioning, running 10+ QAOA sub-circuits sequentially on your laptop means waiting for each one to finish before the next starts. Qoro's Maestro runs every partition **in parallel**, collapsing hours into minutes.

## Step 0: Set Your API Key

```bash
pip install qoro-divi networkx matplotlib
```

Create a `.env` file in the repo root:

```
QORO_API_KEY="your_api_key_here"
```

👉 **[Get your free API key →](https://dash.qoroquantum.net)**

## How It Works

This example uses **graph partitioning** to split a large graph into smaller, manageable sub-graphs using spectral clustering. Each sub-graph is solved independently with QAOA, then results are aggregated.

### Phase 1 — Local Toy Problem

Runs an 8-node graph locally to prove the algorithm works. This fits easily in your laptop's memory.

### Phase 2 — Scale Up with QoroService

Bumps to a **50-node graph**, which generates 10+ partitions. Each partition is dispatched to QoroService for parallel execution — the kind of workload that would take hours locally.

## Project Structure

```text
.
├── main.py           # Two-phase runner (local → cloud)
├── utils.py          # Graph generation, visualization, analysis
└── README.md         # This file
```

## Run

```bash
python main.py
```

## Expected Output

1. **Phase 1:** A small graph visualization and QAOA results completing in seconds.
2. **Phase 2:** Terminal output showing partitions being routed to Qoro Maestro, followed by a quantum vs. classical comparison.

```
Quantum Cut Size to Classical Cut Size Ratio = 0.98
```

## Configuration

| Parameter            | Description                                                                                     |
|----------------------|-------------------------------------------------------------------------------------------------|
| `n_qubits`           | Total number of nodes in the graph.                                                             |
| `n_clusters`         | Number of dense communities to generate.                                                        |
| `partitioning_config`| Controls how Divi splits the graph, using spectral clustering.                                  |
| `optimizer`          | Uses MonteCarloOptimizer to find optimal QAOA parameters.                                       |

---

👉 **Ready to go beyond 50 nodes?** [Get your API key](https://dash.qoroquantum.net) and scale with QoroService.
