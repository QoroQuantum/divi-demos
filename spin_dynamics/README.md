# Spin Dynamics of the Transverse-Field Ising Model (TFIM)

> 🚀 **Skip the local bottleneck.** Qoro is giving away **$100 in free cloud compute credits.**
> Get your API key at **[dash.qoroquantum.net](https://dash.qoroquantum.net)** to run this at scale.

## Why Cloud?

Each time step in a spin dynamics simulation generates a **full Trotter circuit**. A 15-qubit chain at 15 time points means 30+ deep circuits (Exact + QDrift), each with thousands of measurement shots. Running them sequentially on your laptop is painful. QoroService runs every time-step circuit on Maestro's GPU-accelerated simulator in parallel.

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

Two scripts cover two different divi capabilities:

### `spin_dynamics.py` — parallel time-evolution sweep

Uses **`TimeEvolutionTrajectory`**: one program per time point, all dispatched together. Runs three physical regimes back-to-back, each comparing **Exact Trotterization** against **QDrift** (stochastic Trotter sampling):

1. **Ferromagnetic phase** (J=1.0, h=0.2) — slow magnetization wobble.
2. **Paramagnetic phase** (J=1.0, h=2.0) — rapid X-axis precession.
3. **Néel state** |101010⟩ (J=1.0, h=0.5) — relaxation from an anti-ferromagnetic initial state.

```bash
python spin_dynamics.py
```

Set `USE_CLOUD = True` at the top of the script to dispatch all time-point circuits to QoroService instead of running locally.

### `neel_dynamics.py` — hardware-targeting workflow with QASM export

Demonstrates the full export → reload → run loop you'd use for real hardware:

1. **Export** Trotter circuits as OpenQASM 2.0 files via the public `qscript_to_meta` + `dag_to_qasm_body` helpers.
2. (Optional) **Compress** the QASM externally — transpile, lay out for a target topology, etc.
3. **Reload and run** the compressed circuits via `backend.submit_circuits` and compare against:
   - Exact statevector ground truth (local).
   - Exact Trotter on hardware/cloud.
   - QDrift on hardware/cloud.

Uses `track_depth=True` to compare circuit depths across all four trajectories.

```bash
# Build and save QASM circuits.
python neel_dynamics.py export --n-qubits 15

# Run the comparison — add --cloud to dispatch to QoroService.
python neel_dynamics.py run --n-qubits 15 --compressed-dir compressed_circuits --cloud
```

## Configuration (`spin_dynamics.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_QUBITS` | 6 | Length of the 1D spin chain |
| `N_STEPS` | 5 | Number of Trotter steps |
| `T_MAX` | 3.0 | Maximum evolution time |
| `N_POINTS` | 6 | Number of time points |
| `J` | 1.0 | Coupling strength |
| `h` | 0.2 / 2.0 / 0.5 | Transverse field (one per regime) |

## Output

`spin_dynamics.py` saves three plots: `dynamics_ferromagnetic.png`, `dynamics_paramagnetic.png`, `dynamics_neel.png`. Each compares Exact vs QDrift for one regime.

`neel_dynamics.py run` saves a single `dynamics_neel.png` overlaying SV ground truth + Exact-on-hardware + Compressed-on-hardware (if `--compressed-dir` is provided) + QDrift-on-hardware.

---

👉 **Ready for larger spin chains?** [Get your API key](https://dash.qoroquantum.net) and scale with QoroService.
