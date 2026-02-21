"""
PCE vs QAOA Benchmark for the Quantum-Guided Cluster Algorithm
================================================================
Compares two correlation extraction methods for guiding the cluster
Monte Carlo:

  1. QAOA (standard) — N qubits for N variables (one qubit per node)
  2. PCE (dense)     — O(log₂N) qubits for N variables
  3. PCE (poly)      — O(√N) qubits for N variables

Both methods extract two-point correlations that guide the cluster
algorithm. The question: can PCE's dramatically compressed correlations
still produce good cluster solutions?

Usage:
    python benchmark_pce.py
"""

import os
import time

import numpy as np

from algorithm import (
    generate_random_maxcut_graph,
    ising_energy,
    extract_qaoa_correlations,
    extract_pce_correlations,
    coupling_constant_correlations,
    correlation_guided_cluster_algorithm,
    simulated_annealing,
)
from plotting import (
    setup_dark_style,
    COLORS,
    QAOA_COLORS,
)

import matplotlib.pyplot as plt
from divi.backends import ParallelSimulator


def run_pce_benchmark(
    n_nodes: int = 16,
    degree: int = 10,
    qaoa_depth: int = 2,
    pce_max_iterations: int = 15,
    qaoa_max_iterations: int = 10,
    n_iterations_factor: int = 500,
    n_repetitions: int = 30,
    lambda_scale: float = 4,
    seed: int = 42,
    shots: int = 10_000,
    skip_qaoa: bool = False,
    skip_poly: bool = False,
    use_cloud: bool = False,
    output_dir: str = "plots_pce",
):
    """Compare QAOA vs PCE as correlation sources for the cluster algorithm.

    Runs the same cluster Monte Carlo with correlations from:
      - Coupling constants (classical baseline)
      - QAOA at depth p (standard quantum approach, unless skip_qaoa=True)
      - PCE dense encoding (logarithmic qubit compression)
      - PCE poly encoding (polynomial qubit compression)

    Then generates a comparison plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    total_steps = 2 + (0 if skip_qaoa else 1) + 1 + (0 if skip_poly else 1)

    print("=" * 70)
    print("  🔬 PCE vs QAOA: Correlation Source Comparison")
    print("  📄 Using Quantum-Guided Cluster Algorithm (arXiv:2508.10656)")
    print("  ⚛️  PCE: Pauli Correlation Encoding for qubit compression")
    print("=" * 70)

    # --- Generate graph ---
    G = generate_random_maxcut_graph(n_nodes, degree, seed=seed)
    print(f"\n📊 Graph: {n_nodes} nodes, {degree}-regular, "
          f"{G.number_of_edges()} edges")
    print(f"   Cluster budget: {n_iterations_factor}×n = "
          f"{n_iterations_factor * n_nodes} iterations, {n_repetitions} restarts")

    # --- Ground state (if small enough) ---
    E_ground = None
    if n_nodes <= 22:
        print("\n🔍 Computing exact ground state...")
        t0 = time.time()
        E_ground = np.inf
        for bits in range(2**n_nodes):
            config = np.array([1 - 2 * ((bits >> i) & 1) for i in range(n_nodes)])
            E = ising_energy(G, config)
            if E < E_ground:
                E_ground = E
        print(f"   E₀ = {E_ground:.1f}  ({time.time() - t0:.1f}s)")
    else:
        print(f"\n⚠️  n={n_nodes} > 22: brute-force ground state skipped")

    # Backend setup
    if use_cloud:
        from divi.backends import QoroService, JobConfig
        backend = QoroService(config=JobConfig(qpu_system="qoro_maestro", shots=shots))
        print(f"\n☁️  Using QoroService backend (shots={shots})")
    else:
        backend = ParallelSimulator(shots=shots)
        print(f"\n💻 Using local ParallelSimulator (shots={shots})")

    # ─── Method results collector ───
    methods = {}  # name -> (result, qubit_count, elapsed, correlation_label)
    correlation_matrices = {}  # label -> Z matrix

    step = 0

    # 1. Simulated Annealing
    step += 1
    print(f"\n{'─'*70}")
    print(f"🔥 [{step}/{total_steps}] Simulated Annealing (classical baseline)...")
    t0 = time.time()
    sa_result = simulated_annealing(
        G, n_iterations_factor=n_iterations_factor,
        n_repetitions=n_repetitions, seed=seed,
    )
    sa_time = time.time() - t0
    methods["SA"] = (sa_result, 0, sa_time, "—")
    print(f"   Best E = {sa_result.best_energy:.1f}  |  {sa_time:.2f}s")

    # 2. Coupling Constants
    step += 1
    print(f"\n{'─'*70}")
    print(f"🔗 [{step}/{total_steps}] Cluster with Coupling Constants...")
    Z_cc = coupling_constant_correlations(G)
    correlation_matrices["Coupling\nConstants"] = Z_cc
    t0 = time.time()
    cc_result = correlation_guided_cluster_algorithm(
        G, Z_cc, n_iterations_factor=n_iterations_factor,
        n_repetitions=n_repetitions, lambda_scale=1, seed=seed,
    )
    cc_time = time.time() - t0
    methods["Coupling\nConst."] = (cc_result, 0, cc_time, "J_ij")
    print(f"   Best E = {cc_result.best_energy:.1f}  |  "
          f"accept = {cc_result.acceptance_rate:.1%}  |  {cc_time:.2f}s")

    # 3. QAOA (optional)
    Z_qaoa = None
    if not skip_qaoa:
        step += 1
        print(f"\n{'─'*70}")
        print(f"⚛️  [{step}/{total_steps}] QAOA (p={qaoa_depth}, {n_nodes} qubits)...")
        Z_qaoa, qaoa_inst = extract_qaoa_correlations(
            G, n_layers=qaoa_depth, max_iterations=qaoa_max_iterations,
            shots=shots, backend=backend,
        )
        correlation_matrices[f"QAOA p={qaoa_depth}\n({n_nodes} qubits)"] = Z_qaoa
        t0 = time.time()
        qaoa_result = correlation_guided_cluster_algorithm(
            G, Z_qaoa, n_iterations_factor=n_iterations_factor,
            n_repetitions=n_repetitions, lambda_scale=lambda_scale, seed=seed,
        )
        qaoa_cluster_time = time.time() - t0
        methods[f"QAOA\np={qaoa_depth}"] = (qaoa_result, n_nodes, qaoa_cluster_time, f"QAOA p={qaoa_depth}")
        print(f"   Best E = {qaoa_result.best_energy:.1f}  |  "
              f"accept = {qaoa_result.acceptance_rate:.1%}  |  "
              f"circuits = {qaoa_inst.total_circuit_count}")

    # 4. PCE dense
    step += 1
    print(f"\n{'─'*70}")
    print(f"🧬 [{step}/{total_steps}] PCE Dense Encoding...")
    Z_pce_d, pce_d_inst = extract_pce_correlations(
        G, encoding_type="dense", max_iterations=pce_max_iterations,
        shots=shots, backend=backend,
    )
    correlation_matrices[f"PCE Dense\n({pce_d_inst.n_qubits} qubits)"] = Z_pce_d
    t0 = time.time()
    pce_d_result = correlation_guided_cluster_algorithm(
        G, Z_pce_d, n_iterations_factor=n_iterations_factor,
        n_repetitions=n_repetitions, lambda_scale=lambda_scale, seed=seed,
    )
    pce_d_cluster_time = time.time() - t0
    methods["PCE\ndense"] = (pce_d_result, pce_d_inst.n_qubits, pce_d_cluster_time, "PCE dense")
    print(f"   Best E = {pce_d_result.best_energy:.1f}  |  "
          f"accept = {pce_d_result.acceptance_rate:.1%}  |  "
          f"circuits = {pce_d_inst.total_circuit_count}  |  {pce_d_cluster_time:.2f}s")

    # 5. PCE poly (optional)
    pce_p_inst = None
    if not skip_poly:
        step += 1
        print(f"\n{'─'*70}")
        print(f"🧬 [{step}/{total_steps}] PCE Poly Encoding...")
        Z_pce_p, pce_p_inst = extract_pce_correlations(
            G, encoding_type="poly", max_iterations=pce_max_iterations,
            shots=shots, backend=backend,
        )
        correlation_matrices[f"PCE Poly\n({pce_p_inst.n_qubits} qubits)"] = Z_pce_p
        t0 = time.time()
        pce_p_result = correlation_guided_cluster_algorithm(
            G, Z_pce_p, n_iterations_factor=n_iterations_factor,
            n_repetitions=n_repetitions, lambda_scale=lambda_scale, seed=seed,
        )
        pce_p_cluster_time = time.time() - t0
        methods["PCE\npoly"] = (pce_p_result, pce_p_inst.n_qubits, pce_p_cluster_time, "PCE poly")
        print(f"   Best E = {pce_p_result.best_energy:.1f}  |  "
              f"accept = {pce_p_result.acceptance_rate:.1%}  |  "
              f"circuits = {pce_p_inst.total_circuit_count}  |  {pce_p_cluster_time:.2f}s")

    # ── Summary table ──
    print(f"\n{'='*70}")
    print("  RESULTS: PCE Correlation Source Comparison")
    if E_ground is not None:
        print(f"  E₀ = {E_ground:.1f} (exact ground state)")
    else:
        print(f"  ⚠️  Ground state unknown (n={n_nodes} > 22)")
    print(f"{'='*70}")

    if E_ground is not None:
        print(f"  {'Method':<20} {'Qubits':>7} {'Best E':>8} {'Mean r':>8} {'σ(r)':>8} {'Accept':>8}")
    else:
        print(f"  {'Method':<20} {'Qubits':>7} {'Best E':>8} {'Mean E':>8} {'σ(E)':>8} {'Accept':>8}")
    print(f"  {'─'*62}")

    for name, (result, qubits, elapsed, label) in methods.items():
        name_flat = name.replace("\n", " ")
        q_str = f"{qubits}" if qubits > 0 else "—"
        line = f"  {name_flat:<20} {q_str:>7} {result.best_energy:>8.1f} "
        if E_ground is not None:
            ratios = [e / E_ground for e in result.energy_history]
            line += f"{np.mean(ratios):>8.3f} {np.std(ratios):>8.3f} "
        else:
            me = np.mean(result.energy_history)
            se = np.std(result.energy_history)
            line += f"{me:>8.1f} {se:>8.1f} "
        acc = f"{result.acceptance_rate:>7.1%}" if hasattr(result, 'acceptance_rate') and result.acceptance_rate > 0 else f"{'—':>8}"
        line += acc
        print(line)

    print(f"{'='*70}")

    # ── Qubit savings highlight ──
    print(f"\n💡 Qubit savings (vs QAOA's {n_nodes} qubits):")
    print(f"   PCE dense: {pce_d_inst.n_qubits} qubits  ({n_nodes/pce_d_inst.n_qubits:.1f}× compression)")
    if pce_p_inst is not None:
        print(f"   PCE poly:  {pce_p_inst.n_qubits} qubits  ({n_nodes/pce_p_inst.n_qubits:.1f}× compression)")

    # ── Generate comparison plot ──
    _plot_comparison(methods, E_ground, G, output_dir)

    # ── Correlation heatmap comparison ──
    _plot_correlation_comparison(correlation_matrices, G, output_dir)

    print(f"\n✅ All done! Plots saved to {output_dir}/")


def _plot_comparison(methods: dict, E_ground, G, output_dir: str):
    """Generate a bar chart comparing all methods."""
    setup_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    names = list(methods.keys())
    n_methods = len(names)

    # Dynamic color palette
    base_colors = [COLORS["sa"], COLORS["cc"], COLORS["accent"],
                   "#e056fd", "#f0932b", "#6ab04c", "#22a6b3"]
    method_colors = base_colors[:n_methods]

    # --- Left: Best energy ---
    ax = axes[0]
    energies = [methods[n][0].best_energy for n in names]
    x = np.arange(n_methods)
    bars = ax.bar(x, energies, width=0.6,
                  color=[c + "40" for c in method_colors],
                  edgecolor=method_colors, linewidth=2)
    for bar, e in zip(bars, energies):
        ax.text(bar.get_x() + bar.get_width() / 2, e - abs(e) * 0.01,
                f"{e:.1f}", ha="center", va="top", fontsize=11,
                fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Best Energy (lower = better)", fontsize=13, fontweight="bold")
    ax.set_title("Best Energy Found", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    if E_ground is not None:
        ax.axhline(y=E_ground, color=COLORS["accent"], linestyle="--",
                   alpha=0.5, label=f"E₀ = {E_ground:.1f}")
        ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.2)

    # --- Right: Qubit count ---
    ax = axes[1]
    qubits = [methods[n][1] for n in names]
    bars = ax.bar(x, qubits, width=0.6,
                  color=[c + "40" for c in method_colors],
                  edgecolor=method_colors, linewidth=2)
    for bar, q in zip(bars, qubits):
        label = str(q) if q > 0 else "0"
        ax.text(bar.get_x() + bar.get_width() / 2, max(q, 0.3) + 0.3,
                label, ha="center", va="bottom", fontsize=14,
                fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Qubits Required", fontsize=13, fontweight="bold")
    ax.set_title("Quantum Resource Usage", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle(f"PCE Cluster Algorithm Performance\n"
                 f"({G.number_of_nodes()} nodes, "
                 f"{list(dict(G.degree()).values())[0]}-regular graph)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "pce_vs_qaoa_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    print(f"📊 Saved: {save_path}")
    plt.close()


def _plot_correlation_comparison(correlation_matrices: dict, G, output_dir: str):
    """Side-by-side heatmaps of correlation matrices from each method."""
    setup_dark_style()

    n_panels = len(correlation_matrices)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    cmap = plt.cm.RdBu_r
    vmax = max(np.abs(Z).max() for Z in correlation_matrices.values())

    for ax, (label, Z) in zip(axes, correlation_matrices.items()):
        im = ax.imshow(Z, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(label, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Variable j", fontsize=10)
        ax.set_ylabel("Variable i", fontsize=10)

    fig.suptitle("Correlation Matrices Comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Correlation ⟨Z_i Z_j⟩", fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "correlation_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    print(f"📊 Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    run_pce_benchmark(
        n_nodes=180,
        degree=14,               # Dense, highly frustrated spin glass
        skip_qaoa=True,          # QAOA would need 180 qubits — impossible
        skip_poly=False,         # ~19 qubits — manageable on cloud MPS
        use_cloud=True,          # Cloud backend for PCE
        pce_max_iterations=25,   # More iterations for PCE to learn correlations
        n_iterations_factor=100, # Tight budget — forces methods to be smart
        n_repetitions=10,        # Enough restarts for statistics
        lambda_scale=4,
        seed=42,
        shots=10_000,
        output_dir="plots_pce",
    )


