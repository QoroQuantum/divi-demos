# =============================================================================
#  Transverse-Field Ising Model (TFIM) Spin Dynamics via Divi Time Evolution
# =============================================================================
#
#  Demonstrates Divi's TimeEvolution API for simulating Hamiltonian dynamics.
#  We simulate a 1D spin chain initialized with all spins "Up" (|0...0>)
#  and track the magnetization of the first spin under time evolution.
#
#  We compare:
#    1. Exact Trotterization
#    2. QDrift (a stochastic Taylor-series approach that reduces gate depth)
#
# =============================================================================

import matplotlib
matplotlib.use("Agg")  # For headless execution
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from divi.backends import ParallelSimulator, QoroService, JobConfig
from divi.qprog import TimeEvolution
from divi.hamiltonians import ExactTrotterization, QDrift

def build_tfim_hamiltonian(n_qubits: int, J: float, h: float) -> qml.Hamiltonian:
    """Build the Transverse-Field Ising Model (TFIM) Hamiltonian.
    
    H = -J * Sum_{i}(Z_i Z_{i+1}) - h * Sum_{i}(X_i)
    
    Args:
        n_qubits: Number of spins / qubits in the chain.
        J: Coupling strength between neighboring spins.
        h: Transverse magnetic field strength.
        
    Returns:
        A PennyLane Hamiltonian representing the system.
    """
    coeffs = []
    ops = []
    
    # Nearest-neighbor ZZ couplings
    for i in range(n_qubits - 1):
        coeffs.append(-J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
        
    # Transverse field X
    for i in range(n_qubits):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))
        
    return qml.Hamiltonian(coeffs, ops)


def simulate_dynamics(hamiltonian, times, strategy, backend, n_steps=6, shots=10_000, initial_state="Zeros"):
    """Evolve the system up to each time t and measure magnetization.
    
    Args:
        hamiltonian: The system Hamiltonian.
        times: Array of time points to simulate.
        strategy: Trotterization strategy (e.g., ExactTrotterization, QDrift).
        backend: The quantum backend to use.
        n_steps: Number of Trotter steps.
        shots: Number of measurement shots.
        initial_state: String representing the initial state (e.g., "Zeros", "Ones", "101010").
        
    Returns:
        List of expected magnetizations <Z_0> at each time step.
    """
    magnetizations = []
    observable = qml.PauliZ(0)  # We measure the magnetization of the first spin

    for t in times:
        if t == 0.0:
            if initial_state == "Zeros" or (initial_state and initial_state[0] == '0'):
                magnetizations.append(1.0)
            elif initial_state == "Ones" or (initial_state and initial_state[0] == '1'):
                magnetizations.append(-1.0)
            else:
                magnetizations.append(0.0) # E.g., Superposition state has <Z> = 0
            continue
            
        te = TimeEvolution(
            hamiltonian=hamiltonian,
            time=t,
            n_steps=n_steps,
            order=1,  # First-order Suzuki-Trotter
            initial_state=initial_state,
            observable=observable,
            backend=backend,
            trotterization_strategy=strategy
        )
        
        te.run()
        magnetizations.append(te.results)
        
    return magnetizations


def plot_dynamics(times, mag_exact, mag_qdrift, title, filename):
    """Plot the results of the spin dynamics simulation."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(times, mag_exact, 'o-', color='#3b82f6', label='Exact Trotterization', linewidth=2, markersize=8)
    plt.plot(times, mag_qdrift, 's--', color='#f97316', label='QDrift (stochastic)', linewidth=2, markersize=8)
    
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
#  MAIN — Simulation Flow
# =====================================================================

if __name__ == "__main__":
    
    N_QUBITS = 6
    SHOTS = 20_000
    N_STEPS = 5     # Trotter steps (higher = more accurate but deeper circuits)
    T_MAX = 3.0     # Maximum evolution time
    N_POINTS = 15   # Number of time points to simulate
    
    # --- Backend selection ---
    USE_CLOUD = False
    
    if USE_CLOUD:
        backend = QoroService(config=JobConfig(qpu_system="qoro_maestro", shots=SHOTS))
        print("☁️  Using QoroService cloud backend")
    else:
        backend = ParallelSimulator(shots=SHOTS)
        print("💻 Using local ParallelSimulator")
        
    times = np.linspace(0, T_MAX, N_POINTS)
    
    # =================================================================
    #  Experiment 1 — Ferromagnetic Phase (Weak transverse field)
    #
    #  J = 1.0, h = 0.2
    #  The spins want to align with each other. The weak field causes
    #  a slow, gentle oscillation in the magnetization.
    # =================================================================
    print("\n" + "=" * 70)
    print("  Experiment 1 — Ferromagnetic Phase (J=1.0, h=0.2)")
    print("=" * 70)
    
    H_ferro = build_tfim_hamiltonian(n_qubits=N_QUBITS, J=1.0, h=0.2)
    print(f"\n   ⚙️  Built Hamiltonian for {N_QUBITS} qubits")
    
    print("\n   🚀 Running Exact Trotterization...")
    mag_exact_f = simulate_dynamics(H_ferro, times, ExactTrotterization(), backend, n_steps=N_STEPS)
    
    print("\n   🚀 Running QDrift (stochastic sampling)...")
    # QDrift samples terms randomly based on their coefficients.
    # We restrict it to 10 terms per step to compress the circuit depth.
    mag_qdrift_f = simulate_dynamics(H_ferro, times, QDrift(sampling_budget=10), backend, n_steps=N_STEPS)
    
    plot_dynamics(
        times, mag_exact_f, mag_qdrift_f, 
        title="Spin Dynamics: Ferromagnetic Phase (J=1.0, h=0.2)", 
        filename="dynamics_ferromagnetic.png"
    )
    
    # =================================================================
    #  Experiment 2 — Paramagnetic Phase (Strong transverse field)
    #
    #  J = 1.0, h = 2.0
    #  The strong external field quickly flips the spins, leading to
    #  rapid oscillations as they precess around the X-axis.
    # =================================================================
    print("\n" + "=" * 70)
    print("  Experiment 2 — Paramagnetic Phase (J=1.0, h=2.0)")
    print("=" * 70)
    
    H_para = build_tfim_hamiltonian(n_qubits=N_QUBITS, J=1.0, h=2.0)
    print(f"\n   ⚙️  Built Hamiltonian for {N_QUBITS} qubits")
    
    print("\n   🚀 Running Exact Trotterization...")
    mag_exact_p = simulate_dynamics(H_para, times, ExactTrotterization(), backend, n_steps=N_STEPS)
    
    print("\n   🚀 Running QDrift (stochastic sampling)...")
    mag_qdrift_p = simulate_dynamics(H_para, times, QDrift(sampling_budget=10), backend, n_steps=N_STEPS)
    
    plot_dynamics(
        times, mag_exact_p, mag_qdrift_p, 
        title="Spin Dynamics: Paramagnetic Phase (J=1.0, h=2.0)", 
        filename="dynamics_paramagnetic.png"
    )
    
    # =================================================================
    #  Experiment 3 — Néel State Dynamics
    #
    #  J = 1.0, h = 0.5
    #  We initialize the system in an anti-ferromagnetic (Néel) state:
    #  "101010" (|101010>). We observe the relaxation and oscillation 
    #  of the magnetization from its negative starting position.
    # =================================================================
    print("\n" + "=" * 70)
    print("  Experiment 3 — Néel State Dynamics (J=1.0, h=0.5, State='101010')")
    print("=" * 70)
    
    H_neel = build_tfim_hamiltonian(n_qubits=N_QUBITS, J=1.0, h=0.5)
    print(f"\n   ⚙️  Built Hamiltonian for {N_QUBITS} qubits")
    
    neel_state = "101010"
    
    print("\n   🚀 Running Exact Trotterization...")
    mag_exact_n = simulate_dynamics(H_neel, times, ExactTrotterization(), backend, n_steps=N_STEPS, initial_state=neel_state)
    
    print("\n   🚀 Running QDrift (stochastic sampling)...")
    mag_qdrift_n = simulate_dynamics(H_neel, times, QDrift(sampling_budget=10), backend, n_steps=N_STEPS, initial_state=neel_state)
    
    plot_dynamics(
        times, mag_exact_n, mag_qdrift_n, 
        title="Spin Dynamics: Néel State |101010> (J=1.0, h=0.5)", 
        filename="dynamics_neel.png"
    )
    
    print("\n" + "=" * 70)
    print("  ✅ Simulation complete! Check the generated PNG files.")
    print("=" * 70)
