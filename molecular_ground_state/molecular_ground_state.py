# =============================================================================
#  Molecular Ground State via VQE — Potential Energy Surface of H₂
# =============================================================================
#
#  Uses Divi's VQEHyperparameterSweep to scan the ground-state energy of H₂
#  across bond lengths and ansätze, then plots the potential energy surface.
#
#  Scroll to the bottom to see the main flow — it reads like plain English!
#
# =============================================================================

import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

from divi.backends import ParallelSimulator, QoroService, JobConfig

from divi.qprog import (
    VQE,
    UCCSDAnsatz,
    GenericLayerAnsatz,
    VQEHyperparameterSweep,
)
from divi.qprog.optimizers import MonteCarloOptimizer


# ─────────────────────────────────────────────────────────────────────
#  STEP 1 — Build molecular Hamiltonians at each bond length
# ─────────────────────────────────────────────────────────────────────

def build_h2_hamiltonians(
    bond_lengths_angstrom: list[float],
) -> dict[float, qml.operation.Operator]:
    """Generate the electronic Hamiltonian of H₂ at each bond length.

    PennyLane's quantum chemistry module computes the second-quantised
    molecular Hamiltonian (in the STO-3G basis) and maps it to qubits
    via the Jordan-Wigner transformation, yielding a 4-qubit operator.

    Args:
        bond_lengths_angstrom: List of H–H distances in Ångströms.

    Returns:
        Dict mapping bond_length → qubit Hamiltonian.
    """
    hamiltonians = {}

    for r_angstrom in bond_lengths_angstrom:
        # Convert Å → Bohr (PennyLane uses atomic units internally)
        r_bohr = r_angstrom / 0.529177
        half = r_bohr / 2.0

        H, n_qubits = qml.qchem.molecular_hamiltonian(
            symbols=["H", "H"],
            coordinates=np.array([0.0, 0.0, -half, 0.0, 0.0, half]),
        )
        hamiltonians[r_angstrom] = H

    return hamiltonians


# ─────────────────────────────────────────────────────────────────────
#  STEP 2 — Configure ansätze to compare
# ─────────────────────────────────────────────────────────────────────

def get_ansatze() -> list:
    """Return the ansätze to sweep over.

    - UCCSD: The gold standard for molecular chemistry — unitary coupled
      cluster singles and doubles. Chemically inspired.
    - GenericLayer (RY+RZ / CNOT): A hardware-efficient-style ansatz with
      fewer gates. Compiles well to real hardware but may struggle
      with chemical accuracy.
    """
    return [
        UCCSDAnsatz(),
        GenericLayerAnsatz(
            gate_sequence=[qml.RY, qml.RZ],
            entangler=qml.CNOT,
            entangling_layout="linear",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────
#  STEP 3 — Run the VQE sweep
# ─────────────────────────────────────────────────────────────────────

def run_sweep(
    hamiltonians: dict[float, qml.operation.Operator],
    ansatze: list,
    n_electrons: int = 2,
    max_iterations: int = 15,
    shots: int = 10_000,
    backend=None,
) -> VQEHyperparameterSweep:
    """Run a VQE hyperparameter sweep over ansätze × bond lengths.

    Divi's VQEHyperparameterSweep creates one VQE instance per
    (ansatz, Hamiltonian) combination and runs them all via
    ProgramBatch for efficient parallel execution.

    Args:
        hamiltonians: Dict mapping bond_length → qubit Hamiltonian.
        ansatze: List of ansatz circuits to compare.
        n_electrons: Number of electrons in the molecule (2 for H₂).
        max_iterations: Optimizer iterations per VQE run.
        shots: Measurement shots per circuit.
        backend: Divi backend. Defaults to ParallelSimulator.

    Returns:
        The completed VQEHyperparameterSweep instance.
    """
    if backend is None:
        backend = ParallelSimulator(shots=shots)

    sweep = VQEHyperparameterSweep(
        ansatze=ansatze,
        hamiltonians=hamiltonians,
        optimizer=MonteCarloOptimizer(population_size=10, n_best_sets=3),
        max_iterations=max_iterations,
        backend=backend,
        n_electrons=n_electrons,
    )

    print("Creating VQE programs for all (ansatz × bond_length) combinations...")
    sweep.create_programs()
    n_programs = len(sweep.programs)
    print(f"  {n_programs} VQE instances created")

    print("\n🚀 Running sweep...")
    sweep.run(blocking=True)
    print("  Sweep complete!")

    return sweep


# ─────────────────────────────────────────────────────────────────────
#  STEP 4 — Extract and display results
# ─────────────────────────────────────────────────────────────────────

def extract_pes_data(sweep: VQEHyperparameterSweep):
    """Extract potential energy surface data from the completed sweep.

    Returns:
        dict mapping ansatz_name → (bond_lengths_list, energies_list)
    """
    pes_data = {}

    for (ansatz_name, bond_length), program in sweep.programs.items():
        energy = program.best_loss

        if ansatz_name not in pes_data:
            pes_data[ansatz_name] = {"bond_lengths": [], "energies": []}

        pes_data[ansatz_name]["bond_lengths"].append(bond_length)
        pes_data[ansatz_name]["energies"].append(energy)

    # Sort by bond length within each ansatz
    for ansatz_name in pes_data:
        pairs = sorted(zip(
            pes_data[ansatz_name]["bond_lengths"],
            pes_data[ansatz_name]["energies"],
        ))
        pes_data[ansatz_name]["bond_lengths"] = [p[0] for p in pairs]
        pes_data[ansatz_name]["energies"] = [p[1] for p in pairs]

    return pes_data


def print_results(pes_data: dict, sweep: VQEHyperparameterSweep):
    """Print a summary of the PES results."""
    best_config, best_energy = sweep.aggregate_results()

    print("\n" + "=" * 70)
    print("  🔬 Potential Energy Surface — Results")
    print("=" * 70)

    for ansatz_name, data in pes_data.items():
        min_idx = int(np.argmin(data["energies"]))
        eq_bond = data["bond_lengths"][min_idx]
        eq_energy = data["energies"][min_idx]

        print(f"\n  {ansatz_name}:")
        print(f"    Equilibrium bond length : {eq_bond:.3f} Å")
        print(f"    Ground-state energy     : {eq_energy:.6f} Ha")
        print(f"    Points computed         : {len(data['bond_lengths'])}")

    print(f"\n  🏆 Best overall: {best_config[0]} at r={best_config[1]:.2f} Å"
          f"  →  E = {best_energy:.6f} Ha")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────
#  STEP 5 — Visualize the potential energy surface
# ─────────────────────────────────────────────────────────────────────

COLORS = {
    "UCCSDAnsatz": "#7c4dff",          # purple
    "GenericLayerAnsatz": "#00e5ff",    # cyan
}


def plot_pes(pes_data: dict, save_path: str | None = None):
    """Plot the potential energy surface for each ansatz.

    Generates a publication-quality dark-themed plot showing E(r)
    for each ansatz, with the equilibrium point highlighted.

    Args:
        pes_data: Output of extract_pes_data().
        save_path: If given, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for ansatz_name, data in pes_data.items():
        bond_lengths = data["bond_lengths"]
        energies = data["energies"]
        color = COLORS.get(ansatz_name, "#ffffff")

        # Plot the PES curve
        ax.plot(
            bond_lengths, energies,
            "-o", color=color, linewidth=2, markersize=5,
            label=ansatz_name, alpha=0.9,
        )

        # Mark the equilibrium point
        min_idx = int(np.argmin(energies))
        ax.plot(
            bond_lengths[min_idx], energies[min_idx],
            "*", color=color, markersize=15, zorder=5,
        )
        ax.annotate(
            f"  {energies[min_idx]:.4f} Ha",
            (bond_lengths[min_idx], energies[min_idx]),
            color=color, fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("Bond Length (Å)", color="white", fontsize=13)
    ax.set_ylabel("Ground-State Energy (Hartree)", color="white", fontsize=13)
    ax.set_title(
        "H₂ Potential Energy Surface — VQE with Divi",
        color="white", fontsize=15, fontweight="bold", pad=15,
    )
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(True, alpha=0.15, color="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"\n  📊 Plot saved to {save_path}")
    plt.show()


# =====================================================================
#  MAIN — The high-level flow (start reading here!)
# =====================================================================

if __name__ == "__main__":
    # --- Bond length scan ---
    BOND_LENGTHS = np.linspace(0.3, 2.5, 8).tolist()

    # --- Backend selection ---
    USE_CLOUD = False  # Set to True to use QoroService cloud backend

    if USE_CLOUD:
        # QoroService reads QORO_API_KEY from your environment
        # Get your API key at https://dash.qoroquantum.net
        backend = QoroService(config=JobConfig(shots=10_000))
        print("☁️  Using QoroService cloud backend")
    else:
        backend = ParallelSimulator(shots=10_000)
        print("💻 Using local ParallelSimulator")

    # 1. Build the molecular Hamiltonians at each bond length
    print(f"\nBuilding H₂ Hamiltonians for {len(BOND_LENGTHS)} bond lengths "
          f"({min(BOND_LENGTHS):.2f} – {max(BOND_LENGTHS):.2f} Å)...")
    hamiltonians = build_h2_hamiltonians(BOND_LENGTHS)
    print(f"  Done — 4 qubits per Hamiltonian")

    # 2. Choose which ansätze to compare
    ansatze = get_ansatze()
    print(f"Ansätze: {[a.name for a in ansatze]}")

    # 3. Run the VQE sweep over all (ansatz × bond_length) combinations
    sweep = run_sweep(
        hamiltonians=hamiltonians,
        ansatze=ansatze,
        n_electrons=2,  # H₂ has 2 electrons
        max_iterations=15,
        backend=backend,
    )

    # 4. Extract and display the results
    pes_data = extract_pes_data(sweep)
    print_results(pes_data, sweep)

    # 5. Plot the potential energy surface
    plot_pes(pes_data, save_path="h2_pes.png")
