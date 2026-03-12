# Economic Load Dispatch with Prohibited Operating Zones

> 🚀 **Skip the local bottleneck.** Qoro is giving away **$100 in free cloud compute credits.**
> Get your API key at **[dash.qoroquantum.net](https://dash.qoroquantum.net)** to run this at scale.

## Why Cloud?

The PCE-VQE optimizer needs **hundreds of circuit evaluations** per iteration — each computing expectation values across multiple qubits. Running these sequentially on a local simulator means every evaluation blocks the next. QoroService evaluates circuits **in parallel**, collapsing wall-clock time dramatically.

## Step 0: Set Your API Key

```bash
pip install qoro-divi
```

Create a `.env` file in the repo root:

```
QORO_API_KEY="your_api_key_here"
```

👉 **[Get your free API key →](https://dash.qoroquantum.net)**

## What It Does

Dispatches power across generators to meet demand at minimum fuel cost, while avoiding Prohibited Operating Zones (mechanical vibration bands).

1. Encodes the dispatch as a QUBO with **4 bits per generator**.
2. Solves it with PCE-VQE using **polynomial encoding** (logarithmic qubit compression).
3. Applies a **feasibility repair** heuristic for constraint-valid dispatches.
4. Compares quantum results against a classical baseline.

### Phase 1 — Local Toy Problem (3 generators)

| Parameter | Value |
|-----------|-------|
| Generators | 3 |
| Binary variables | 12 |
| PCE qubits | ~5 |
| Demand | 195 MW |

Runs locally to prove the algorithm works.

### Phase 2 — Scale Up with QoroService (6 generators)

| Parameter | Value |
|-----------|-------|
| Generators | 6 |
| Binary variables | 24 |
| PCE qubits | ~8 |
| Demand | 390 MW |

Doubles the grid size. QoroService parallelises the circuit evaluations on Maestro.

## Quick Start

```bash
python economic_load_dispatch.py
```

Or open the interactive notebook:

```bash
jupyter notebook economic_load_dispatch.ipynb
```

## Files

| File | Description |
|------|-------------|
| `economic_load_dispatch.py` | Two-phase script (local → cloud) |
| `economic_load_dispatch.ipynb` | Jupyter notebook with markdown explanations |
| `README.md` | This file |

## Tuning Parameters

- **`DEMAND`** — target load in MW
- **`GENERATORS`** — add/remove generators, change cost curves or POZ ranges
- **`PENALTY_LAMBDA`** / **`POZ_MU`** — constraint penalty weights
- **`n_layers`** — PCE ansatz depth
- **`max_iterations`** — DE optimizer generations
- **`encoding_type`** — `"poly"` (fewer qubits) or `"dense"` (even fewer)

---

👉 **Ready for production-scale dispatch?** [Get your API key](https://dash.qoroquantum.net) and scale with QoroService.
