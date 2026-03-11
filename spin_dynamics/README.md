# Spin Dynamics of the Transverse-Field Ising Model (TFIM)

> 🚀 **Don't choke your local machine.** Qoro is giving away **$100 in free cloud compute credits.**
> Get your API key at **[dash.qoroquantum.net](https://dash.qoroquantum.net)** to run this at scale.

## Why Cloud?

Each time step in a spin dynamics simulation generates a **full Trotter circuit**. A 15-qubit chain with 15 time points means 30+ deep circuits (Exact + QDrift), each needing thousands of measurement shots. Running these sequentially on your laptop is painfully slow. QoroService runs every time-step circuit on **Maestro's GPU-accelerated simulator** in parallel.

## Step 0: Set Your API Key

```bash
pip install qoro-divi
```

Create a `.env` file in the repo root:

```
QORO_API_KEY="your_api_key_here"
```

👉 **[Get your free API key →](https://dash.qoroquantum.net)**

## Overview

Simulates a 1D chain of spins (qubits) evolving under the TFIM Hamiltonian:

$$ H = -J \sum_{i} Z_i Z_{i+1} - h \sum_{i} X_i $$

Compares two time-evolution strategies:
1. **Exact Trotterization** — deterministic Suzuki-Trotter decomposition
2. **QDrift** — stochastic sampling that drastically reduces gate count

### Phase 1 — Local Toy Problem

4 qubits, 5 time points. Runs in seconds locally.

### Phase 2 — Scale Up with QoroService

15+ qubits, 15 time points. Full ferromagnetic, paramagnetic, and Néel state simulations dispatched to Qoro Maestro.

## Quick Start

```bash
python spin_dynamics.py
```

Or explore interactively:

```bash
jupyter notebook spin_dynamics.ipynb
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_QUBITS` | 6 | Length of the 1D spin chain |
| `N_STEPS` | 5 | Number of Trotter steps |
| `T_MAX` | 3.0 | Maximum evolution time |
| `N_POINTS` | 15 | Number of time points |
| `J` | 1.0 | Coupling strength |
| `h` | 0.2/2.0 | Transverse field (weak / strong) |

## Expected Output

Three physical regimes are simulated:
1. **Ferromagnetic Phase (J ≫ h):** Slow, gentle magnetization wobble
2. **Paramagnetic Phase (h ≫ J):** Rapid oscillations from X-axis precession
3. **Néel State (|101010⟩):** Complex relaxation dynamics from anti-ferromagnetic initial state

Plots are saved as `dynamics_ferromagnetic.png`, `dynamics_paramagnetic.png`, and `dynamics_neel.png`.

---

👉 **Ready for larger spin chains?** [Get your API key](https://dash.qoroquantum.net) and scale with QoroService.
