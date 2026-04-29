# Molecular Ground State via VQE using `divi`

> 🚀 **Skip the local bottleneck.** Qoro is giving away **$100 in free cloud compute credits.**
> Get your API key at **[dash.qoroquantum.net](https://dash.qoroquantum.net)** to run this at scale.

## Why Cloud?

Computing the H₂ potential energy surface means running **2 ansätze × 12 bond lengths = 24 independent VQE runs.** Each VQE has its own optimizer loop with hundreds of circuit evaluations. Sequentially, that's 24× the wait. QoroService dispatches **all 24 VQE instances in parallel** — same result, a fraction of the time.

## Step 0: Set Your API Key

```bash
pip install qoro-divi matplotlib
```

Create a `.env` file in the repo root:

```
QORO_API_KEY="your_api_key_here"
```

👉 **[Get your free API key →](https://dash.qoroquantum.net)**

## What It Does

Computes the **potential energy surface (PES)** of hydrogen (H₂) — ground-state energy as a function of bond length — using Divi's VQE features:

- **`VQEHyperparameterSweep`** — grid search over ansätze × Hamiltonians in parallel
- **`VQE` with molecular Hamiltonians** — finds the ground-state energy of a qubit Hamiltonian
- **Multiple ansätze** — compare UCCSD (chemistry gold standard) vs. GenericLayer (RY+RZ / CNOT, gate-efficient)

### Phase 1 — Local PES (5 bond lengths)

5 bond lengths × 2 ansätze = 10 VQE runs. Fast enough locally to prove the algorithm works.

### Phase 2 — High-Resolution PES with QoroService (12 bond lengths)

12 bond lengths × 2 ansätze = 24 VQE runs dispatched to QoroService in parallel. Higher resolution PES in a fraction of the time.

## Quick Start

```bash
python molecular_ground_state.py
```

Or explore interactively:

```bash
jupyter notebook molecular_ground_state.ipynb
```

## Expected Output

```
  🔬 Potential Energy Surface — Results
======================================================================

  UCCSDAnsatz:
    Equilibrium bond length : 0.740 Å
    Ground-state energy     : -1.136189 Ha
    Points computed         : 12

  GenericLayerAnsatz:
    Equilibrium bond length : 0.740 Å
    Ground-state energy     : -1.130422 Ha
    Points computed         : 12

  🏆 Best overall: UCCSDAnsatz at r=0.74 Å  →  E = -1.136189 Ha
======================================================================
```

## Configuration

| Parameter           | Description                                                                      |
|---------------------|----------------------------------------------------------------------------------|
| `BOND_LENGTHS_LOCAL`| Local bond length scan (default: 5 points, 0.3–2.5 Å).                          |
| `BOND_LENGTHS_CLOUD`| Cloud bond length scan (default: 12 points, 0.3–2.5 Å).                         |
| `ansatze`           | Ansatz circuits to compare: UCCSD, GenericLayer, etc.                            |
| `max_iterations`    | Optimizer iterations per VQE run (15 local, 25 cloud).                           |
| `shots`             | Measurement samples per circuit (5k local, 10k cloud).                           |

---

👉 **Ready for larger molecules?** [Get your API key](https://dash.qoroquantum.net) and scale with QoroService.
