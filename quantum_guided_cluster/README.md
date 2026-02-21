# Quantum-Guided Cluster Algorithm for Max-Cut using `divi`

This example implements the **Quantum-Guided Cluster Algorithm (QGCA)** from
["Quantum-Guided Cluster Algorithms for Combinatorial Optimization" (arXiv:2508.10656)](https://arxiv.org/abs/2508.10656)
by Eder et al. (Amazon Quantum Solutions Lab), using the `divi` quantum programming framework.
See also the accompanying [AWS Quantum Technologies Blog post](https://aws.amazon.com/blogs/quantum-computing/quantum-guided-cluster-algorithms-for-combinatorial-optimization/).

## Why This Matters

Classical methods like simulated annealing make small, local moves (single spin
flips) that easily get trapped in rugged energy landscapes. Traditional cluster
algorithms (Swendsen-Wang, Wolff) can make larger moves, but they break down on
*frustrated* problems — the clusters either blow up to system-spanning size or
fail to capture useful structure.

The QGCA solves this by using **quantum-derived correlations** to guide cluster
formation. QAOA is run *once* to extract pairwise correlations ⟨Z_i Z_j⟩, which
encode how spins **tend to align in good solutions**. These correlations ensure
clusters reflect meaningful problem structure rather than random connectivity,
enabling large, targeted jumps through the search space even on hard, frustrated
instances.

The key result from the paper: on 10-regular graphs with 28 nodes, QAOA
correlations at depth p ≥ 2 outperform both coupling-constant-guided clusters
and simulated annealing, with performance improving monotonically with QAOA
depth.

## Key Divi Features Showcased

| Feature | Role in QGCA |
|---------|-------------|
| **QAOA** with `GraphProblem.MAXCUT` | Prepares low-energy states and extracts two-point correlations at varying circuit depths. |
| **QWC Observable Grouping** | Batches commuting ZZ observables into fewer measurement circuits — up to 60% reduction in circuit count. |
| **QDrift Trotterization** | Optional randomized Trotter method for scalable circuit depth on large instances. |
| **QoroService** | Cloud backend for scaling QAOA beyond 18 qubits to tackle larger problem instances. |

## Project Structure

```text
.
├── main.py           # Benchmark runner — orchestrates experiments and generates plots
├── algorithm.py      # Core algorithm: graph gen, QAOA correlations, cluster MC, SA baseline
├── plotting.py       # Dark-themed visualization utilities (4 plots)
├── plots/            # Generated visualizations (created on first run)
│   ├── 1_approximation_ratios.png
│   ├── 2_correlation_heatmaps.png
│   ├── 3_circuit_efficiency.png
│   └── 4_energy_distributions.png
└── README.md         # This file
```

## Prerequisites

```bash
pip install qoro-divi networkx matplotlib numpy
```

## Usage

### Run the benchmark

```bash
python main.py
```

This runs the full benchmark: generates a random regular graph with ±1 edge
weights (Ising spin glass), extracts QAOA correlations at multiple circuit
depths, and compares three methods:

1. **Simulated Annealing** — standard single-spin-flip baseline.
2. **Coupling-Constant Clusters** — clusters guided by raw edge weights J_ij
   (no quantum information, only the graph structure itself).
3. **QAOA-Guided Clusters** — clusters guided by QAOA two-point correlations at
   depths p = 1, 2, 3, 5 (p = 2 is the key transition point highlighted in the
   [AWS blog](https://aws.amazon.com/blogs/quantum-computing/quantum-guided-cluster-algorithms-for-combinatorial-optimization/)
   and the paper's Fig. 3).

### Expected Output

The terminal prints a summary table comparing all methods:

```
  Method                         Best E   Mean r   Best r     σ(r)   Accept
  ────────────────────────────────────────────────────────────────────
  Simulated Annealing             -34.0    0.976    1.000    0.071        —
  Cluster (Coupling Const.)       -34.0    1.000    1.000    0.000    5.1%
  QAOA-Guided (p=1)               -34.0    1.000    1.000    0.000    1.7%
  QAOA-Guided (p=2)               -34.0    0.992    1.000    0.042    2.0%
  QAOA-Guided (p=3)               -34.0    0.976    1.000    0.071    2.0%
  QAOA-Guided (p=5)               -34.0    0.992    1.000    0.042    2.3%
```

Four plots are saved to the `plots/` directory:

| Plot | What It Shows |
|------|---------------|
| `1_approximation_ratios.png` | Mean approximation ratio (r = E / E₀) with ±1σ error bars. The headline comparison across methods. |
| `2_correlation_heatmaps.png` | Side-by-side heatmaps of coupling constants vs. QAOA correlations at each depth — shows how QAOA captures increasingly structured spin-spin relationships. |
| `3_circuit_efficiency.png` | Total circuits executed vs. naive one-per-observable baseline, demonstrating Divi's QWC grouping savings. |
| `4_energy_distributions.png` | Violin plots of energy distributions across random restarts, revealing solution *reliability* (not just best-case). |

## Configuration

Edit the `run_benchmark()` call at the bottom of `main.py`:

| Parameter              | Default | Description                                                                 |
|------------------------|---------|-----------------------------------------------------------------------------|
| `n_nodes`              | 16      | Number of graph nodes (= qubits for QAOA).                                 |
| `degree`               | 10      | Graph regularity. Higher values create harder, more frustrated instances.   |
| `qaoa_depths`          | [1,2,3,5] | List of QAOA circuit depths to compare. p=2 is the transition point.  |
| `n_iterations_factor`  | 500     | Iteration budget per method: total iterations = factor × n_nodes.           |
| `n_repetitions`        | 30      | Number of independent random restarts per method.                           |
| `lambda_scale`         | 4       | Cluster link scaling (λ_scale in the paper's Eq. 3).                        |
| `seed`                 | 42      | Random seed for reproducibility.                                            |
| `use_cloud`            | False   | Use QoroService cloud backend (required for >18 qubits).                    |
| `shots`                | 10,000  | Number of measurement shots per QAOA circuit.                               |
| `output_dir`           | "plots" | Directory for saving generated plots.                                       |

## Remote Execution (QoroService)

To run on larger graphs (>18 qubits) — closer to the paper's 28-node, 100-node
benchmarks — use Qoro's cloud backend:

1. Obtain an API key from [dash.qoroquantum.net](https://dash.qoroquantum.net).
2. Set the environment variable:
   ```bash
   export QORO_API_KEY="your_api_key_here"
   ```
3. Set `use_cloud=True` and increase `n_nodes`:
   ```python
   results = run_benchmark(
       n_nodes=28,        # Paper's primary benchmark size
       degree=10,         # 10-regular graphs (paper's Fig. 3)
       qaoa_depths=[1, 2, 3, 5],
       use_cloud=True,
   )
   ```

## Algorithm Overview

The algorithm is a **hybrid quantum-classical workflow** with two distinct
phases. Crucially, the quantum computer is called only *once* — the rest is
efficient classical post-processing.

### Phase 1: Correlation Extraction (Quantum — one-shot)

QAOA is run on the Max-Cut Hamiltonian at depth *p*. From the optimized state
|ψ_opt⟩, two-point correlations Z_ij = ⟨ψ_opt| σ_i^z σ_j^z |ψ_opt⟩ are
measured for all qubit pairs. These correlations encode information about how
spins tend to align in low-energy (good) solutions.

Divi's **QWC grouping** batches qubit-wise commuting observables, reducing the
number of measurement circuits by up to 60% — critical for practical circuit
budgets on real hardware.

### Phase 2: Cluster Monte Carlo (Classical)

A simulated annealing loop uses the correlation matrix to build **clusters**
of correlated spins:

1. A random seed node is chosen.
2. Neighboring spins are added to the cluster with probability proportional to
   the correlation strength and alignment with the current spin configuration
   (Eq. 3 from the paper).
3. The entire cluster is flipped, and the move is accepted or rejected via the
   Metropolis criterion.

This enables coordinated multi-spin moves that escape local minima where
single-spin-flip SA gets stuck. Unlike Swendsen-Wang/Wolff algorithms, the
correlation-guided cluster formation **avoids percolation** — clusters remain
meaningful even on frustrated, dense graphs.

### Why It Works on Hard Instances

On frustrated problems (dense graphs, conflicting constraints), coupling
constants alone can provide misleading guidance. QAOA correlations incorporate
*global* structure from the quantum state, not just local edge weights. As shown
in the paper (Fig. 3), this advantage grows with QAOA depth:
- **p = 1**: Matches coupling-constant performance (local structure only).
- **p ≥ 2**: Begins outperforming — quantum correlations capture non-local
  relationships that improve cluster quality.

## Applications

The QGCA framework applies broadly to combinatorial optimization problems that
can be mapped to Max-Cut or QUBO formulations, including:

- **Scheduling & logistics** — job shop scheduling, vehicle routing
- **Network design** — graph partitioning, circuit layout
- **Portfolio optimization** — asset selection with constraints
- **Graph coloring & clustering** — community detection, resource allocation

## Reference

```bibtex
@article{eder2025quantum,
  title   = {Quantum-Guided Cluster Algorithms for Combinatorial Optimization},
  author  = {Eder, Peter J. and Kerschbaumer, Aron and Finžgar, Jernej Rudi
             and Medina, Raimel A. and Schuetz, Martin J. A. and
             Katzgraber, Helmut G. and Braun, Sarah and Mendl, Christian B.},
  journal = {arXiv preprint arXiv:2508.10656},
  year    = {2025}
}
```
