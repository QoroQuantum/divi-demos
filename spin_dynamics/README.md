# Spin Dynamics of the Transverse-Field Ising Model (TFIM)

Quantum simulation of spin dynamics using the **TimeEvolution** module in the
[Divi](https://dash.qoroquantum.net) quantum SDK.

## Overview

This example simulates a 1D chain of spins (qubits) evolving under the
Transverse-Field Ising Model (TFIM) Hamiltonian:

$$ H = -J \sum_{i} Z_i Z_{i+1} - h \sum_{i} X_i $$

The system is initialized with all spins aligned in the +Z direction (the `|0...0>` state).
We then evolve the system over time and track the magnetization of the first spin ($\langle Z_0 \rangle$).

The script compares two different quantum time-evolution strategies available in Divi:
1. **Exact Trotterization** — A standard Suzuki-Trotter decomposition that implements all terms in the Hamiltonian deterministically.
2. **QDrift** — A stochastic Trotterization method that samples Hamiltonian terms randomly according to their coefficients. QDrift drastically reduces the gate count per step, offering a hardware-friendly approximation for large or deeply-connected systems.

## Quick Start

```bash
# 1. Provide a virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install qoro-divi

# 3. Run the script
python spin_dynamics.py
```

Or explore interactively via Jupyter Notebook:

```bash
pip install jupyter
jupyter notebook spin_dynamics.ipynb
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_QUBITS` | 6       | Length of the 1D spin chain |
| `N_STEPS`  | 5       | Number of Trotter steps |
| `T_MAX`    | 3.0     | Maximum time duration to simulate |
| `N_POINTS` | 15      | Number of time steps to measure |
| `J`        | 1.0     | Coupling strength |
| `h`        | 0.2/2.0 | Transverse magnetic field. Script tests weak (0.2) and strong (2.0) |

## Expected Output

The script simulates physical regimes with different initial states:
1. **Ferromagnetic Phase ($J \gg h$):** The spins strongly prefer to stay aligned with their neighbors. The weak field causes only a slow, gentle wobble in the magnetization.
2. **Paramagnetic Phase ($h \gg J$):** The strong transverse field overpowers the spin-spin coupling, causing rapid and dramatic oscillations (precession around the X-axis).
3. **Néel State ($\vert 101010 \rangle$):** An anti-ferromagnetic starting state. The magnetization starts at -1.0 (for the first spin) and shows complex relaxation dynamics.

Three plots (`dynamics_ferromagnetic.png`, `dynamics_paramagnetic.png`, and `dynamics_neel.png`) are automatically generated, showing the Exact Trotterization overlaid with the QDrift stochastic approximation.
