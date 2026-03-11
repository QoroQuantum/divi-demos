# Quantum-Guided Cluster Algorithm for Max-Cut using `divi`

> 🚀 **Don't choke your local machine.** Qoro is giving away **$100 in free cloud compute credits.**
> Get your API key at **[dash.qoroquantum.net](https://dash.qoroquantum.net)** to run this at scale.

## Why Cloud?

The paper's key result uses **28-node, 10-regular graphs** at QAOA depth **p=5** — that's 28 qubits with deep variational circuits. Even the p=2 transition point (where quantum correlations start outperforming coupling constants) is infeasible on a local simulator for problem sizes beyond ~18 qubits. QoroService handles 28+ qubit QAOA with **Maestro's GPU-accelerated MPS simulation**.

## Step 0: Set Your API Key

```bash
pip install qoro-divi networkx matplotlib numpy
```

Create a `.env` file in the repo root:

```
QORO_API_KEY="your_api_key_here"
```

👉 **[Get your free API key →](https://dash.qoroquantum.net)**

## Why This Matters

Classical methods like simulated annealing make small, local moves that easily get trapped in rugged energy landscapes. The QGCA solves this by using **quantum-derived correlations** to guide cluster formation. QAOA is run *once* to extract pairwise correlations ⟨Z_i Z_j⟩, which encode how spins **tend to align in good solutions** — enabling large, targeted jumps through the search space.

This example implements the algorithm from [arXiv:2508.10656](https://arxiv.org/abs/2508.10656) (Eder et al., Amazon Quantum Solutions Lab). See also the [AWS Quantum Technologies Blog post](https://aws.amazon.com/blogs/quantum-computing/quantum-guided-cluster-algorithms-for-combinatorial-optimization/).

## Key Divi Features Showcased

| Feature | Role in QGCA |
|---------|-------------|
| **QAOA** with `GraphProblem.MAXCUT` | Prepares low-energy states and extracts two-point correlations |
| **QWC Observable Grouping** | Batches commuting ZZ observables — up to **60% circuit reduction** |
| **QDrift Trotterization** | Randomized Trotter method for scalable circuit depth |
| **QoroService** | Cloud backend for 28+ qubit QAOA |

## Two-Phase Structure

### Phase 1 — Local Toy Problem (10 nodes)

Runs the full benchmark on a 10-node graph locally. Proves the algorithm and generates all four plots. Completes in seconds.

### Phase 2 — Scale Up with QoroService (28 nodes)

Scales to the paper's benchmark size: **28-node, 10-regular graphs** at depths p=1,2,3,5. Each QAOA run is dispatched to QoroService — the kind of workload that would be infeasible locally.

## Project Structure

```text
.
├── main.py           # Two-phase benchmark runner
├── algorithm.py      # Core: graph gen, QAOA correlations, cluster MC, SA
├── plotting.py       # Dark-themed visualization (4 plots)
├── plots/            # Generated visualizations
└── README.md         # This file
```

## Run

```bash
python main.py
```

## Expected Output

```
  Method                         Best E   Mean r   Best r     σ(r)   Accept
  ────────────────────────────────────────────────────────────────────
  Simulated Annealing             -34.0    0.976    1.000    0.071        —
  Cluster (Coupling Const.)       -34.0    1.000    1.000    0.000    5.1%
  QAOA-Guided (p=1)               -34.0    1.000    1.000    0.000    1.7%
  QAOA-Guided (p=2)               -34.0    0.992    1.000    0.042    2.0%
  QAOA-Guided (p=5)               -34.0    0.992    1.000    0.042    2.3%
```

Four plots are saved to `plots/`:

| Plot | What It Shows |
|------|---------------|
| `1_approximation_ratios.png` | Mean approximation ratio with ±1σ error bars |
| `2_correlation_heatmaps.png` | Coupling constants vs. QAOA correlations at each depth |
| `3_circuit_efficiency.png` | QWC grouping savings vs. naive baseline |
| `4_energy_distributions.png` | Violin plots of energy distributions |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_nodes` | 16 | Graph nodes (= qubits). Phase 1: 10, Phase 2: 28 |
| `degree` | 10 | Graph regularity |
| `qaoa_depths` | [1,2,3,5] | QAOA circuit depths to compare |
| `n_repetitions` | 30 | Random restarts per method |
| `use_cloud` | False | Use QoroService (auto-enabled in Phase 2) |

## Algorithm Overview

1. **Phase 1 (Quantum — one-shot)**: QAOA extracts two-point correlations. QWC grouping reduces circuit count by up to 60%.
2. **Phase 2 (Classical)**: Cluster Monte Carlo uses correlations to build meaningful spin clusters, enabling coordinated multi-spin moves.

## Reference

```bibtex
@article{eder2025quantum,
  title   = {Quantum-Guided Cluster Algorithms for Combinatorial Optimization},
  author  = {Eder, Peter J. and Kerschbaumer, Aron and ...},
  journal = {arXiv preprint arXiv:2508.10656},
  year    = {2025}
}
```

---

👉 **Ready for 100-node graphs?** [Get your API key](https://dash.qoroquantum.net) and scale with QoroService.
