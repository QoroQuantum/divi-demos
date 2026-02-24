# Divi Examples

Example programs demonstrating quantum algorithms with the [Divi](https://github.com/QoroQuantum/divi) quantum programming framework.

## Table of Contents

1. [Cluster MaxCut](#1-cluster-maxcut) — Partitioned QAOA for large graphs
2. [Minimum Birkhoff Decomposition](#2-minimum-birkhoff-decomposition) — VQE for matrix decomposition
3. [Portfolio Optimization](#3-portfolio-optimization) — QAOA/PCE for financial optimization
4. [Quantum-Guided Cluster Algorithm](#4-quantum-guided-cluster-algorithm) — QAOA-guided Monte Carlo
5. [Economic Load Dispatch](#5-economic-load-dispatch) — PCE-VQE for power dispatch
6. [Molecular Ground State](#6-molecular-ground-state) — VQE for H₂ potential energy surface
7. [Travelling Salesman Problem](#7-travelling-salesman-problem) — QUBO + QAOA for routing
8. [Spin Dynamics (TFIM)](#8-spin-dynamics-tfim) — Time Evolution under Hamiltonian dynamics

## Getting Started

```bash
pip install -r requirements.txt
```

Or install the core dependency directly:

```bash
pip install qoro-divi
```

Each example also includes its own README with specific instructions.

## Examples

### 1. [Cluster MaxCut](./cluster_maxcut)

Partitioned QAOA for MaxCut using graph partitioning to solve the MaxCut problem on large, community-structured graphs. Splits large graphs (e.g., 50 qubits) into smaller sub-graphs using spectral clustering for parallel QAOA execution.

**Key Features:**
- Graph partitioning with spectral clustering
- Parallel QAOA execution on sub-graphs
- Comparison of quantum vs. classical results
- Support for both local simulation and remote execution via QoroService

📓 **[Interactive notebook](./cluster_maxcut/cluster_maxcut.ipynb)** — step-by-step tutorial

---

### 2. [Minimum Birkhoff Decomposition](./minimum_birkhoff_decomposition)

VQE-based approach to find the Birkhoff decomposition of doubly stochastic matrices. Showcases the modular design of Divi — a sophisticated application built by inheriting from the VQE class with minimal code.

**Key Features:**
- Custom VQE implementation for Birkhoff decomposition
- Multi-threaded classical optimization with caching
- Command-line interface for various problem configurations
- Support for sparse and dense matrix types

📓 **[Interactive notebook](./minimum_birkhoff_decomposition/birkhoff_decomposition.ipynb)** — step-by-step tutorial

---

### 3. [Portfolio Optimization](./portfolio_optimization)

Quantum portfolio optimization using QAOA and PCE combined with spectral partitioning to handle large-scale problems. Partitions the asset correlation graph into smaller sub-problems for efficient quantum solving.

**Key Features:**
- QUBO formulation for portfolio optimization
- Spectral partitioning based on asset correlations
- Interactive Jupyter notebook workflow
- Solution comparison between QAOA/PCE and exact solvers
- Comprehensive visualization and financial metrics analysis

📓 **[QAOA notebook](./portfolio_optimization/portfolio_optimization.ipynb)** — full QAOA workflow  
📓 **[PCE notebook](./portfolio_optimization/portfolio_optimization_pce.ipynb)** — same problem with logarithmic qubit compression

---

### 4. [Quantum-Guided Cluster Algorithm](./quantum_guided_cluster)

Implementation of the Quantum-Guided Cluster Algorithm from [arXiv:2508.10656](https://arxiv.org/abs/2508.10656). QAOA is run once to extract two-point correlations ⟨Z_i Z_j⟩, which then guide a classical cluster Monte Carlo for Max-Cut.

**Key Features:**
- QAOA as a one-shot correlation oracle for combinatorial optimization
- QWC observable grouping for efficient correlation measurement (up to 60% circuit reduction)
- Correlation-guided cluster Monte Carlo (Algorithm 1 from the paper)
- Benchmarks against simulated annealing and coupling-constant baselines
- Publication-quality dark-themed visualizations
- Support for cloud execution via QoroService for >18-qubit instances

📓 **[Interactive notebook](./quantum_guided_cluster/quantum_guided_cluster.ipynb)** — step-by-step tutorial

---

### 5. [Economic Load Dispatch](./economic_load_dispatch)

Quantum-classical solution for the **Economic Load Dispatch (ELD)** problem — dispatching power across generators to meet demand at minimum fuel cost while avoiding Prohibited Operating Zones. Uses **PCE-VQE** with polynomial encoding to compress 12 binary variables into just 5 qubits.

**Key Features:**
- QUBO formulation for power dispatch with demand and POZ constraints
- PCE with polynomial encoding (12 variables → 5 qubits)
- Classical feasibility repair heuristic for quantum candidates
- Comparison against brute-force classical optimum
- Support for both local simulation and QoroService cloud execution

📓 **[Interactive notebook](./economic_load_dispatch/economic_load_dispatch.ipynb)** — step-by-step tutorial

---

### 6. [Molecular Ground State](./molecular_ground_state)

Potential energy surface of **H₂** computed with VQE. Divi's `MoleculeTransformer` generates molecule variants at different bond lengths, and `VQEHyperparameterSweep` grid-searches over ansätze × geometries in parallel.

**Key Features:**
- Molecular Hamiltonian from PennyLane's `qml.qchem.Molecule`
- `MoleculeTransformer` for automated bond-length scanning
- `VQEHyperparameterSweep` for parallel ansatz × geometry grid search
- Comparison of UCCSD vs. hardware-efficient ansätze
- PES visualization
- Support for both local simulation and QoroService cloud execution

📓 **[Interactive notebook](./molecular_ground_state/molecular_ground_state.ipynb)** — step-by-step tutorial

---

### 7. [Travelling Salesman Problem](./travelling_salesman)

Quantum solution for the **Travelling Salesman Problem (TSP)** — finding the shortest route that visits every city exactly once and returns to the start. The TSP is encoded as a **QUBO** using one-hot position encoding and solved with **QAOA**.

**Key Features:**
- QUBO formulation with n² binary variables (one-hot encoding)
- Constraint penalties for one-city-per-step and one-step-per-city
- **Direct QAOA:** Simple implementation for small instances (≤4 cities).
- **Partitioned QAOA (`QUBOPartitioningQAOA`):** Enables solving classically prohibitive QUBO problems by automatically decomposing the graph into smaller partitions and solving them in parallel via Divi.
- **PCE (Pauli Correlation Encoding):** Compresses n² QUBO variables into fewer qubits via logarithmic-scale polynomial encoding.
- Greedy repair heuristic for near-feasible quantum bitstrings
- Classical brute-force comparison
- Side-by-side tour visualisation
- Support for both local simulation and QoroService cloud execution

📓 **[Interactive notebook](./travelling_salesman/travelling_salesman.ipynb)** — step-by-step tutorial

---

### 8. [Spin Dynamics (TFIM)](./spin_dynamics)

Quantum simulation of spin dynamics using Divi's **TimeEvolution** module. This example simulates a 1D chain of spins (qubits) evolving under the Transverse-Field Ising Model (TFIM) Hamiltonian.

**Key Features:**
- Demonstration of Divi's `TimeEvolution` API
- Simulation of **Ferromagnetic** and **Paramagnetic** physical phases
- Comparison of **Exact Trotterization** (deterministic) vs. **QDrift** (stochastic gate reduction)
- Beautiful visualization of $\langle Z_0 \rangle$ magnetization over time

📓 **[Interactive notebook](./spin_dynamics/spin_dynamics.ipynb)** — step-by-step tutorial

## License

See [LICENSE](./LICENSE) for details.
