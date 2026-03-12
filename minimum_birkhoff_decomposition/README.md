# VQE-based Minimum Birkhoff Decomposition

> 🚀 **Skip the local bottleneck.** Qoro is giving away **$100 in free cloud compute credits.**
> Get your API key at **[dash.qoroquantum.net](https://dash.qoroquantum.net)** to run this at scale.

## Why Cloud?

VQE optimizer iterations compound quickly — each iteration evaluates circuits, then the classical `black_box_optimizer` runs multi-threaded post-processing with caching. As matrix dimensions grow, the number of permutations explodes. QoroService offloads the **circuit evaluations** so the classical optimizer never waits for quantum results.

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

Finds the Birkhoff decomposition of doubly stochastic matrices using VQE. Built by **inheriting from Divi's VQE class** with only two changes:

1. **Override post-processing** — connects quantum measurements to the problem-specific `black_box_optimizer`
2. **Implement the classical routine** — multi-threaded optimizer with caching

All circuit execution, backend management, and optimization orchestration is inherited from Divi.

## Files

| File | Description |
|---|---|
| `birkhoff_vqe.py` | Core logic: `BirkhoffDecomposition` class extending VQE |
| `main.py` | Executable script with data loading, CLI args, visualization |

## Quick Start

```bash
# Default parameters (4×4 sparse matrix)
python main.py

# Specific experiment
python main.py -n 4 -k 2 -inst 5 -it 20

# View all options
python main.py --help
```

## Arguments

| Flag | Name | Description | Default |
|---|---|---|---|
| `-n` | `--dim` | Matrix dimension | `4` |
| `-k` | `--comb` | Number of permutations | `2` |
| `-m` | `--matrix_type` | `sparse` or `dense` | `sparse` |
| `-inst` | `--instance` | Problem instance (1-10) | `1` |
| `-it` | `--iterations` | Max VQE optimizer iterations | `10` |
| `-opt` | `--optimizer` | VQE optimizer | `Cobyla` |

## Citation

> G. S. Barron, et al., "Quantum Optimization Benchmarking Library: The Intractable Decathlon," arXiv:2504.03832 [quant-ph], (2025).

---

👉 **Ready for larger matrices?** [Get your API key](https://dash.qoroquantum.net) and scale with QoroService.
