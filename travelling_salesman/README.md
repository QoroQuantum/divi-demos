# Travelling Salesman Problem via QUBO + QAOA

> 🚀 **Don't choke your local machine.** Qoro is giving away **$100 in free cloud compute credits.**
> Get your API key at **[dash.qoroquantum.net](https://dash.qoroquantum.net)** to run this at scale.

## Why Cloud?

The TSP uses **n² qubits** — 8 cities means a **64-qubit QUBO**. Even with Divi's automatic graph partitioning, each sub-problem is a full QAOA optimization. Running them sequentially on a local simulator takes ages. QoroService **parallelizes the partitioned sub-problems** and handles deeper circuits with Maestro's GPU-accelerated MPS simulator.

| Cities | Qubits | Brute-force permutations |
|--------|--------|--------------------------|
| 3 | 9 | 2 |
| 4 | 16 | 6 |
| 5 | 25 | 24 |
| 8 | 64 | 5,040 |

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

Finds the shortest cyclic tour through **N randomly placed cities** using three quantum approaches:

1. **Direct QAOA** — Solves the uncompressed QUBO (uses n² qubits). Good for small instances.
2. **Partitioned QAOA** (`QUBOPartitioningQAOA`) — Divi decomposes the QUBO into smaller sub-problems, solves in parallel, and merges results. Scales to large instances.
3. **PCE** (Pauli Correlation Encoding) — Compresses QUBO variables logarithmically. Fewer qubits for NISQ devices.

### Phase 1 — Local Toy Problem

3-4 cities (9-16 qubits). Runs direct QAOA and PCE locally. Completes in seconds.

### Phase 2 — Scale Up with QoroService

8 cities (64 qubits). Partitioned QAOA dispatches sub-problems to QoroService in parallel.

## Quick Start

```bash
python travelling_salesman.py
```

Or open the interactive notebook:

```bash
jupyter notebook travelling_salesman.ipynb
```

## Expected Output

```text
  📊 Summary — Three Divi Approaches to TSP
======================================================================

  Method                              Cities   Qubits   Distance    Ratio
  ───────────────────────────────────────────────────────────────────
  Classical (brute force)                  4        —     1.9633    1.000
  A: Direct QAOA                          4       16     1.9633    1.000
  C: PCE (poly encoding)                  4        6     1.9633    1.000

  Classical (brute force)                  8        —     2.4049    1.000
  B: Partitioned QAOA                      8      ≤15     4.1914    1.743
```

## Files

| File | Description |
|------|-------------|
| `travelling_salesman.py` | Two-phase script (local → cloud) |
| `travelling_salesman.ipynb` | Jupyter notebook with markdown explanations |
| `README.md` | This file |

## Configuration

| Parameter | Description |
|-----------|-------------|
| `N_CITIES_SMALL` | Cities for Direct QAOA and PCE (default 4) |
| `N_CITIES_LARGE` | Cities for Partitioned QAOA (default 8) |

---

👉 **Ready for 20+ cities?** [Get your API key](https://dash.qoroquantum.net) and scale with QoroService.
