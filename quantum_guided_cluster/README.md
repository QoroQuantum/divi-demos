# Quantum-Guided Cluster Algorithm for Max-Cut using `divi`

> 🚀 **Skip the local bottleneck.** Qoro is giving away **$100 in free cloud compute credits.**
> Get your API key at **[dash.qoroquantum.net](https://dash.qoroquantum.net)** to run this at scale.

## Why Cloud?

The paper's headline result is a **depth sweep** over 28-node, 10-regular graphs at p=1, 2, 3, 5 — each depth is a separate QAOA optimization, with its own DE inner loop running hundreds of circuit evaluations. QoroService runs the same Maestro engine you use locally, GPU-accelerated and dispatched in parallel, so the full sweep returns in minutes instead of hours.

## Step 0: Set Your API Key

```bash
pip install qoro-divi networkx matplotlib
```

Create a `.env` file in the repo root:

```
QORO_API_KEY="your_api_key_here"
```

👉 **[Get your free API key →](https://dash.qoroquantum.net)**

## What It Does

Implements the **Quantum-Guided Cluster Algorithm** ([arXiv:2508.10656](https://arxiv.org/abs/2508.10656), Eder et al., AWS Quantum Solutions Lab). Classical simulated annealing makes small, local moves that get trapped in rugged spin-glass landscapes. QGCA escapes those traps by using **quantum-derived two-point correlations** ⟨ZᵢZⱼ⟩ to guide *cluster* moves — spins that the quantum state thinks should align together get flipped together, enabling large coordinated jumps.

Divi features showcased:

- **`QAOA` with `MaxCutProblem`** — extracts ⟨ZᵢZⱼ⟩ from the optimized state
- **QWC observable grouping** — batches commuting ZZ terms into fewer circuits
- **`PCE`** (Pauli Correlation Encoding) — alternative quantum source compressing N variables into O(log₂N) qubits
- **`QoroService`** — cloud backend for 28+ qubit QAOA

### Phase 1 — Local Toy Problem (10 nodes)

Runs the full benchmark on a 10-node graph locally. Completes in seconds.

### Phase 2 — Scale Up with QoroService (28 nodes)

Paper benchmark size: **28-node, 10-regular graphs** at depths p=1, 2, 3, 5. Each QAOA run is dispatched to QoroService.

### Variation — Swap QAOA for PCE

Both extractors return the same `CorrelationResult`, so the cluster algorithm doesn't care which one produced its input matrix. Pass `pce_encodings=["dense", "poly"]` to `run_benchmark` to add PCE rows to the comparison.

## Project Structure

```text
.
├── main.py                       # run_benchmark orchestrator
├── algorithm.py                  # graph gen, QAOA/PCE extractors, cluster MC, SA
├── plotting.py                   # dark-themed visualization (4 plots)
├── quantum_guided_cluster.ipynb  # interactive walkthrough
└── plots/                        # generated visualizations
```

## Run

```bash
python main.py
```

Or open the interactive notebook:

```bash
jupyter notebook quantum_guided_cluster.ipynb
```

## Expected Output

```
  [SA                      ] best E =   -34.0 | mean r = 0.976 | best r = 1.000 | 0.4s
  [Cluster (J coupling)    ] best E =   -34.0 | mean r = 1.000 | best r = 1.000 | accept=5.1% | 0.5s
  [QAOA p=1-Guided         ] best E =   -34.0 | mean r = 1.000 | best r = 1.000 | accept=1.7% | circuits=84  | 1.2s
  [QAOA p=2-Guided         ] best E =   -34.0 | mean r = 0.992 | best r = 1.000 | accept=2.0% | circuits=110 | 1.6s
```

Four plots are saved to `plots/`: approximation ratios, correlation heatmaps, QWC circuit-efficiency, and energy distributions.

## Configuration

| Parameter            | Description                                              |
|----------------------|----------------------------------------------------------|
| `n_nodes`            | Graph nodes (= qubits for QAOA).                         |
| `degree`             | Graph regularity. Use 10+ for hard instances.            |
| `qaoa_depths`        | List of QAOA depths to compare.                          |
| `pce_encodings`      | Optional PCE encodings (`"dense"`, `"poly"`).            |
| `n_repetitions`      | Random restarts per method.                              |
| `use_cloud`          | Use QoroService (auto-enabled in Phase 2).               |

## Reference

```bibtex
@article{eder2025quantum,
  title   = {Quantum-Guided Cluster Algorithms for Combinatorial Optimization},
  author  = {Eder, Peter J. and Kerschbaumer, Aron and others},
  journal = {arXiv preprint arXiv:2508.10656},
  year    = {2025}
}
```

---

👉 **Ready for 100-node graphs?** [Get your API key](https://dash.qoroquantum.net) and scale with QoroService.
