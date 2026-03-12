# Divi Examples

> 🚀 **Skip the local bottleneck.** Qoro is giving away **$100 in free cloud compute credits.**
> Get your API key at **[dash.qoroquantum.net](https://dash.qoroquantum.net)** to run these tutorials at scale.

Example programs demonstrating quantum algorithms with the [Divi](https://github.com/QoroQuantum/divi) quantum programming framework — each structured as a **two-phase tutorial** showing the jump from local toy problems to cloud-scale execution on [QoroService](https://dash.qoroquantum.net).

## Why Qoro?

Running quantum simulations locally hits a wall *fast.* A 50-qubit statevector needs **8 PB of RAM.** Even clever MPS simulators grind to a halt on high-entanglement circuits. Qoro's **Divi + Maestro** stack solves this with:

- ⚡ **Parallelized circuit execution** — Divi automatically partitions large problems and dispatches sub-circuits in parallel
- 🧠 **Maestro MPS simulator** — GPU-accelerated simulation for deep circuits
- ☁️ **QoroService cloud backend** — swap one line of code (`backend=QoroService(...)`) and run at scale

Every example below follows the same pattern: **Phase 1** runs locally on a toy problem to prove the algorithm works. **Phase 2** scales up and offloads to QoroService — because your laptop shouldn't be the bottleneck.

---

## Step 0: Get Your API Key

```bash
pip install qoro-divi
```

Create a `.env` file in the repo root:

```
QORO_API_KEY="your_api_key_here"
```

```python
from divi.backends import QoroService, JobConfig
backend = QoroService(job_config=JobConfig(shots=10_000))
```

👉 **[Get your free API key →](https://dash.qoroquantum.net)**

---

## Examples

### 1. [Cluster MaxCut](./cluster_maxcut) ☁️

Partitioned QAOA for MaxCut on large, community-structured graphs. Splits large graphs (e.g., 50 qubits) into smaller sub-graphs via spectral clustering for parallel QAOA execution.

**The bottleneck:** A 50-node MaxCut graph has a 2⁵⁰ statevector. Your laptop can't touch it.
**The fix:** Divi partitions the graph and QoroService runs every partition in parallel.

📓 **[Interactive notebook](./cluster_maxcut/cluster_maxcut.ipynb)** — step-by-step tutorial

---

### 2. [Minimum Birkhoff Decomposition](./minimum_birkhoff_decomposition) ☁️

VQE-based approach to find the Birkhoff decomposition of doubly stochastic matrices. Showcases the modular design of Divi — a sophisticated application built by inheriting from the VQE class with minimal code.

**The bottleneck:** VQE optimizer iterations × multi-threaded classical post-processing compound quickly.
**The fix:** QoroService offloads the VQE circuit evaluations so the classical optimizer isn't starved.

📓 **[Interactive notebook](./minimum_birkhoff_decomposition/birkhoff_decomposition.ipynb)** — step-by-step tutorial

---

### 3. [Portfolio Optimization](./portfolio_optimization) ☁️

Quantum portfolio optimization using QAOA and PCE combined with spectral partitioning for large-scale financial problems. Partitions the asset correlation graph into smaller sub-problems.

**The bottleneck:** 480 assets → dozens of QAOA partitions. Running them sequentially on a local simulator takes hours.
**The fix:** QoroService runs every partition in parallel — all portfolios optimized simultaneously.

📓 **[QAOA notebook](./portfolio_optimization/portfolio_optimization.ipynb)** — full QAOA workflow  
📓 **[PCE notebook](./portfolio_optimization/portfolio_optimization_pce.ipynb)** — same problem with logarithmic qubit compression

---

### 4. [Quantum-Guided Cluster Algorithm](./quantum_guided_cluster) ☁️

Implementation of the Quantum-Guided Cluster Algorithm from [arXiv:2508.10656](https://arxiv.org/abs/2508.10656). QAOA extracts two-point correlations ⟨Z_i Z_j⟩ to guide a classical cluster Monte Carlo for Max-Cut.

**The bottleneck:** The paper's key result uses 28-node, 10-regular graphs at QAOA depth p=5 — 28 qubits with deep circuits. Infeasible locally.
**The fix:** QoroService handles 28+ qubit QAOA with Maestro's GPU-accelerated MPS simulation.

📓 **[Interactive notebook](./quantum_guided_cluster/quantum_guided_cluster.ipynb)** — step-by-step tutorial

---

### 5. [Economic Load Dispatch](./economic_load_dispatch) ☁️

Quantum-classical solution for the **Economic Load Dispatch (ELD)** problem using **PCE-VQE** with polynomial encoding to compress 12 binary variables into just 5 qubits.

**The bottleneck:** The PCE optimizer needs hundreds of circuit evaluations per iteration. Locally, each evaluation blocks the next.
**The fix:** QoroService evaluates circuits in parallel, collapsing wall-clock time.

📓 **[Interactive notebook](./economic_load_dispatch/economic_load_dispatch.ipynb)** — step-by-step tutorial

---

### 6. [Molecular Ground State](./molecular_ground_state) ☁️

Potential energy surface of **H₂** computed with VQE. Divi's `VQEHyperparameterSweep` grid-searches over ansätze × geometries in parallel.

**The bottleneck:** 2 ansätze × 12 bond lengths = 24 independent VQE runs. Sequentially, that's 24× the wait.
**The fix:** QoroService dispatches all 24 VQE instances in parallel — same result, fraction of the time.

📓 **[Interactive notebook](./molecular_ground_state/molecular_ground_state.ipynb)** — step-by-step tutorial

---

### 7. [Travelling Salesman Problem](./travelling_salesman) ☁️

Quantum solution for the **TSP** using QUBO + QAOA. Demonstrates Direct QAOA, Partitioned QAOA, and PCE encoding.

**The bottleneck:** 8 cities = 64-qubit QUBO. Even partitioned, each sub-problem is a full QAOA run.
**The fix:** QoroService parallelizes the partitioned sub-problems and handles deeper circuits with Maestro.

📓 **[Interactive notebook](./travelling_salesman/travelling_salesman.ipynb)** — step-by-step tutorial

---

### 8. [Spin Dynamics (TFIM)](./spin_dynamics) ☁️

Quantum simulation of spin dynamics using Divi's **TimeEvolution** module. Simulates a 1D spin chain under the Transverse-Field Ising Model Hamiltonian.

**The bottleneck:** Each time step generates a full Trotter circuit. 15-qubit chains with 15 time points = massive circuit volume.
**The fix:** QoroService runs every time-step circuit on Maestro's GPU-accelerated simulator.

📓 **[Interactive notebook](./spin_dynamics/spin_dynamics.ipynb)** — step-by-step tutorial

---

## Ready to Scale?

Every example in this repo works locally on small problems. But when you're ready to go beyond toy instances:

1. **[Get your free API key](https://dash.qoroquantum.net)** — $100 in credits, no credit card required
2. `pip install qoro-divi`
3. Set `QORO_API_KEY` in your environment
4. Change one line: `backend = QoroService(job_config=JobConfig(shots=10_000))`

**That's it.** Same code, cloud scale.

👉 **[dash.qoroquantum.net](https://dash.qoroquantum.net)**

## License

See [LICENSE](./LICENSE) for details.
