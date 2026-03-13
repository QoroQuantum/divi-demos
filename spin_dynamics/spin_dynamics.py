# =============================================================================
#  Transverse-Field Ising Model (TFIM) Spin Dynamics via Divi
# =============================================================================
#
#  Demonstrates Divi's TimeEvolutionTrajectory API for simulating Hamiltonian
#  dynamics. Instead of looping over time points one-by-one, the trajectory
#  API creates all time-evolution programs up front and executes them in
#  parallel — locally or on QoroService for larger systems.
#
#  We simulate a 1D spin chain and track the magnetization ⟨Z₀⟩ under the
#  transverse-field Ising model for three physical regimes:
#    1. Ferromagnetic phase  (weak transverse field)
#    2. Paramagnetic phase   (strong transverse field)
#    3. Néel state dynamics  (anti-ferromagnetic initial state)
#
#  Each experiment compares Exact Trotterization vs QDrift.
#
# =============================================================================

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from divi.backends import ParallelSimulator, QoroService, JobConfig
from divi.hamiltonians import ExactTrotterization, QDrift
from divi.qprog import TimeEvolutionTrajectory


# ─────────────────────────────────────────────────────────────────────
#  Hamiltonian builder
# ─────────────────────────────────────────────────────────────────────

def build_tfim_hamiltonian(n_qubits: int, J: float, h: float) -> qml.Hamiltonian:
    """Build the Transverse-Field Ising Model (TFIM) Hamiltonian.

    H = -J Σ_i Z_i Z_{i+1}  -  h Σ_i X_i
    """
    coeffs, ops = [], []

    for i in range(n_qubits - 1):
        coeffs.append(-J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))

    for i in range(n_qubits):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))

    return qml.Hamiltonian(coeffs, ops)


# ─────────────────────────────────────────────────────────────────────
#  Run a trajectory (all time points in parallel)
# ─────────────────────────────────────────────────────────────────────

def run_trajectory(hamiltonian, time_points, strategy, backend,
                   n_steps=6, initial_state="Zeros"):
    """Create and run a TimeEvolutionTrajectory, return sorted (times, mags).

    All time points are dispatched as independent programs and executed
    in parallel — either on local threads or on QoroService / Maestro.
    """
    trajectory = TimeEvolutionTrajectory(
        hamiltonian=hamiltonian,
        time_points=time_points,
        observable=qml.PauliZ(0),
        backend=backend,
        trotterization_strategy=strategy,
        n_steps=n_steps,
        order=1,
        initial_state=initial_state,
    )

    trajectory.create_programs()
    trajectory.run(blocking=True)

    results = trajectory.aggregate_results()  # dict {t: <O>(t)}

    # Sort by time and return parallel arrays
    sorted_t = sorted(results.keys())
    mags = [results[t] for t in sorted_t]
    return sorted_t, mags


# ─────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_dynamics(times_exact, mag_exact, times_qdrift, mag_qdrift,
                  title, filename):
    """Plot Exact vs QDrift trajectories."""
    plt.figure(figsize=(10, 6))

    plt.plot(times_exact, mag_exact, 'o-', color='#3b82f6',
             label='Exact Trotterization', linewidth=2, markersize=8)
    plt.plot(times_qdrift, mag_qdrift, 's--', color='#f97316',
             label='QDrift (stochastic)', linewidth=2, markersize=8)

    plt.axhline(0, color='black', linestyle='-', alpha=0.2)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel(r'Magnetization $\langle Z_0 \rangle$', fontsize=12)
    plt.ylim(-1.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    print(f"\n   📈 Plot saved to {filename}")


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":

    N_QUBITS = 6
    SHOTS = 5_000
    N_STEPS = 5
    T_MAX = 3.0
    N_POINTS = 6  # Keep small locally; scale up with QoroService

    # --- Backend selection ---
    USE_CLOUD = False

    if USE_CLOUD:
        backend = QoroService(job_config=JobConfig(shots=SHOTS))
        print("☁️  Using QoroService cloud backend")
    else:
        backend = ParallelSimulator(shots=SHOTS)
        print("💻 Using local ParallelSimulator")

    # TimeEvolutionTrajectory needs t > 0 — we skip t=0
    time_points = np.linspace(0.01, T_MAX, N_POINTS).tolist()

    # =================================================================
    #  Experiment 1 — Ferromagnetic Phase (Weak transverse field)
    #
    #  J = 1.0, h = 0.2
    #  Spins want to align → slow, gentle magnetization oscillation.
    # =================================================================
    print("\n" + "=" * 70)
    print("  Experiment 1 — Ferromagnetic Phase (J=1.0, h=0.2)")
    print("=" * 70)

    H_ferro = build_tfim_hamiltonian(n_qubits=N_QUBITS, J=1.0, h=0.2)
    print(f"\n   ⚙️  Built {N_QUBITS}-qubit Hamiltonian")

    print(f"\n   🚀 Exact Trotterization — {N_POINTS} time points in parallel...")
    t0 = time.time()
    t_exact_f, m_exact_f = run_trajectory(H_ferro, time_points, ExactTrotterization(), backend, n_steps=N_STEPS)
    print(f"      Done in {time.time() - t0:.1f}s")

    print(f"\n   🚀 QDrift — {N_POINTS} time points in parallel...")
    t0 = time.time()
    t_qdrift_f, m_qdrift_f = run_trajectory(H_ferro, time_points, QDrift(sampling_budget=10), backend, n_steps=N_STEPS)
    print(f"      Done in {time.time() - t0:.1f}s")

    plot_dynamics(
        t_exact_f, m_exact_f, t_qdrift_f, m_qdrift_f,
        title="Spin Dynamics: Ferromagnetic Phase (J=1.0, h=0.2)",
        filename="dynamics_ferromagnetic.png",
    )

    # =================================================================
    #  Experiment 2 — Paramagnetic Phase (Strong transverse field)
    #
    #  J = 1.0, h = 2.0
    #  Strong field → rapid oscillations as spins precess around X.
    # =================================================================
    print("\n" + "=" * 70)
    print("  Experiment 2 — Paramagnetic Phase (J=1.0, h=2.0)")
    print("=" * 70)

    H_para = build_tfim_hamiltonian(n_qubits=N_QUBITS, J=1.0, h=2.0)
    print(f"\n   ⚙️  Built {N_QUBITS}-qubit Hamiltonian")

    print(f"\n   🚀 Exact Trotterization — {N_POINTS} time points in parallel...")
    t0 = time.time()
    t_exact_p, m_exact_p = run_trajectory(H_para, time_points, ExactTrotterization(), backend, n_steps=N_STEPS)
    print(f"      Done in {time.time() - t0:.1f}s")

    print(f"\n   🚀 QDrift — {N_POINTS} time points in parallel...")
    t0 = time.time()
    t_qdrift_p, m_qdrift_p = run_trajectory(H_para, time_points, QDrift(sampling_budget=10), backend, n_steps=N_STEPS)
    print(f"      Done in {time.time() - t0:.1f}s")

    plot_dynamics(
        t_exact_p, m_exact_p, t_qdrift_p, m_qdrift_p,
        title="Spin Dynamics: Paramagnetic Phase (J=1.0, h=2.0)",
        filename="dynamics_paramagnetic.png",
    )

    # =================================================================
    #  Experiment 3 — Néel State Dynamics
    #
    #  J = 1.0, h = 0.5
    #  Anti-ferromagnetic initial state |101010⟩ → magnetization
    #  relaxes and oscillates from its negative starting position.
    # =================================================================
    print("\n" + "=" * 70)
    print("  Experiment 3 — Néel State Dynamics (J=1.0, h=0.5, State='101010')")
    print("=" * 70)

    H_neel = build_tfim_hamiltonian(n_qubits=N_QUBITS, J=1.0, h=0.5)
    print(f"\n   ⚙️  Built {N_QUBITS}-qubit Hamiltonian")

    neel_state = "101010"

    print(f"\n   🚀 Exact Trotterization — {N_POINTS} time points in parallel...")
    t0 = time.time()
    t_exact_n, m_exact_n = run_trajectory(H_neel, time_points, ExactTrotterization(), backend, n_steps=N_STEPS, initial_state=neel_state)
    print(f"      Done in {time.time() - t0:.1f}s")

    print(f"\n   🚀 QDrift — {N_POINTS} time points in parallel...")
    t0 = time.time()
    t_qdrift_n, m_qdrift_n = run_trajectory(H_neel, time_points, QDrift(sampling_budget=10), backend, n_steps=N_STEPS, initial_state=neel_state)
    print(f"      Done in {time.time() - t0:.1f}s")

    plot_dynamics(
        t_exact_n, m_exact_n, t_qdrift_n, m_qdrift_n,
        title="Spin Dynamics: Néel State |101010⟩ (J=1.0, h=0.5)",
        filename="dynamics_neel.png",
    )

    print("\n" + "=" * 70)
    print("  ✅ All 3 experiments complete! Check the generated PNG files.")
    print("=" * 70)
