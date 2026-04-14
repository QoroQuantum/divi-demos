"""
Quantum-Guided Cluster Algorithm for Max-Cut — Benchmark Runner
================================================================
Orchestrates the full benchmark: generates a graph, runs SA / coupling-constant /
QAOA-guided cluster algorithms, and produces publication-quality plots.

Based on: "Quantum-Guided Cluster Algorithms for Combinatorial Optimization"
  (arXiv:2508.10656)
See also: https://aws.amazon.com/blogs/quantum-computing/quantum-guided-cluster-algorithms-for-combinatorial-optimization/

Usage:
    python main.py
"""

import os
import time

import numpy as np

from algorithm import (
    generate_random_maxcut_graph,
    ising_energy,
    extract_qaoa_correlations,
    coupling_constant_correlations,
    correlation_guided_cluster_algorithm,
    simulated_annealing,
)
from plotting import (
    plot_approximation_ratios,
    plot_correlation_heatmaps,
    plot_circuit_efficiency,
    plot_energy_distribution,
)

from divi.backends import QiskitSimulator


def run_benchmark(
    n_nodes: int = 18,
    degree: int = 10,
    qaoa_depths: list[int] | None = None,
    n_iterations_factor: int = 100,
    n_repetitions: int = 20,
    lambda_scale: float = 6,
    seed: int = 42,
    use_cloud: bool = False,
    shots: int = 10_000,
    output_dir: str = ".",
):
    """Run the Quantum-Guided Cluster Algorithm benchmark.

    Uses dense graphs (high degree) under a tight compute budget to demonstrate
    that QAOA-guided clusters outperform classical SA and coupling-constant
    approaches. Generates publication-quality visualizations.

    Args:
        n_nodes: Number of graph nodes (= qubits for QAOA).
        degree: Graph regularity. Use 10+ for hard instances.
        qaoa_depths: List of QAOA circuit depths to compare.
        n_iterations_factor: Total iterations = factor * n_nodes. Lower = tighter budget.
        n_repetitions: Number of random restarts per method.
        lambda_scale: Cluster formation scaling parameter.
        seed: Random seed.
        use_cloud: If True, use QoroService backend for QAOA (for >18 qubits).
        shots: Number of measurement shots for QAOA.
        output_dir: Directory for saving plots.
    """
    if qaoa_depths is None:
        qaoa_depths = [1, 2, 3, 5]

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  🚀 Quantum-Guided Cluster Algorithm Benchmark")
    print("  📄 Based on arXiv:2508.10656")
    print("  ⚡ Powered by Divi's QAOA + QWC Observable Grouping")
    print("=" * 70)

    # --- Generate graph ---
    G = generate_random_maxcut_graph(n_nodes, degree, seed=seed)
    n_pos = sum(1 for u, v, w in G.edges(data="weight") if w > 0)
    n_neg = G.number_of_edges() - n_pos
    print(f"\n📊 Graph: {n_nodes} nodes, {degree}-regular, "
          f"{G.number_of_edges()} edges ({n_pos}+, {n_neg}−)")
    print(f"   Iteration budget: {n_iterations_factor}×n = "
          f"{n_iterations_factor * n_nodes} total iterations per restart")
    print(f"   Repetitions: {n_repetitions}")

    # --- Compute exact ground state ---
    E_ground = None
    if n_nodes <= 22:
        print("\n🔍 Computing exact ground state (brute force)...")
        t0 = time.time()
        E_ground = np.inf
        for bits in range(2**n_nodes):
            config = np.array([1 - 2 * ((bits >> i) & 1) for i in range(n_nodes)])
            E = ising_energy(G, config)
            if E < E_ground:
                E_ground = E
        bf_time = time.time() - t0
        print(f"   E₀ = {E_ground:.1f}  (found in {bf_time:.1f}s)")

    def approx_ratio(energy: float) -> float:
        if E_ground is None or abs(E_ground) < 1e-10:
            return float("nan")
        return energy / E_ground

    # --- Backend setup ---
    if use_cloud:
        from divi.backends import QoroService, JobConfig
        backend = QoroService(job_config=JobConfig(shots=shots))
        print(f"\n☁️  Using QoroService backend (shots={shots})")
    else:
        backend = QiskitSimulator(shots=shots)
        print(f"\n💻 Using QiskitSimulator backend (shots={shots})")

    # ─── Run all methods ───
    # 1. Simulated Annealing
    print(f"\n{'─'*70}")
    print("🔥 [1/3+] Simulated Annealing...")
    t0 = time.time()
    sa_result = simulated_annealing(
        G, n_iterations_factor=n_iterations_factor,
        n_repetitions=n_repetitions, seed=seed,
    )
    sa_time = time.time() - t0
    if E_ground is not None:
        sa_mean_r = np.mean([e / E_ground for e in sa_result.energy_history])
        print(f"   Best E = {sa_result.best_energy:.1f}  |  "
              f"mean r = {sa_mean_r:.3f}  |  best r = {approx_ratio(sa_result.best_energy):.3f}  |  {sa_time:.2f}s")
    else:
        print(f"   Best E = {sa_result.best_energy:.1f}  |  {sa_time:.2f}s")

    # 2. Coupling Constants
    print(f"\n{'─'*70}")
    print("🔗 [2/3+] Cluster with Coupling Constants...")
    Z_cc = coupling_constant_correlations(G)
    t0 = time.time()
    cc_result = correlation_guided_cluster_algorithm(
        G, Z_cc, n_iterations_factor=n_iterations_factor,
        n_repetitions=n_repetitions, lambda_scale=1, seed=seed,
    )
    cc_time = time.time() - t0
    if E_ground is not None:
        cc_mean_r = np.mean([e / E_ground for e in cc_result.energy_history])
        print(f"   Best E = {cc_result.best_energy:.1f}  |  "
              f"mean r = {cc_mean_r:.3f}  |  best r = {approx_ratio(cc_result.best_energy):.3f}  |  "
              f"accept = {cc_result.acceptance_rate:.1%}  |  {cc_time:.2f}s")
    else:
        print(f"   Best E = {cc_result.best_energy:.1f}  |  "
              f"accept = {cc_result.acceptance_rate:.1%}  |  {cc_time:.2f}s")

    # 3. QAOA-guided at each depth
    qaoa_results = {}
    qaoa_instances = {}
    qaoa_correlations = {"J (coupling\nconstants)": Z_cc}

    for idx, p in enumerate(qaoa_depths):
        print(f"\n{'─'*70}")
        print(f"⚛️  [{3+idx}/{2+len(qaoa_depths)}] QAOA-Guided (p={p})...")

        # Extract correlations
        Z_qaoa, qaoa_instance = extract_qaoa_correlations(
            G, n_layers=p, max_iterations=10, shots=shots, backend=backend, use_qdrift=False,
        )
        qaoa_instances[p] = qaoa_instance
        qaoa_correlations[f"QAOA p={p}"] = Z_qaoa

        # Run cluster algorithm
        t0 = time.time()
        qaoa_result = correlation_guided_cluster_algorithm(
            G, Z_qaoa, n_iterations_factor=n_iterations_factor,
            n_repetitions=n_repetitions, lambda_scale=lambda_scale, seed=seed,
        )
        ca_time = time.time() - t0
        qaoa_results[p] = qaoa_result

        if E_ground is not None:
            q_mean_r = np.mean([e / E_ground for e in qaoa_result.energy_history])
            print(f"   Best E = {qaoa_result.best_energy:.1f}  |  "
                  f"mean r = {q_mean_r:.3f}  |  best r = {approx_ratio(qaoa_result.best_energy):.3f}  |  "
                  f"accept = {qaoa_result.acceptance_rate:.1%}  |  "
                  f"circuits = {qaoa_instance.total_circuit_count}  |  {ca_time:.2f}s")
        else:
            print(f"   Best E = {qaoa_result.best_energy:.1f}  |  "
                  f"accept = {qaoa_result.acceptance_rate:.1%}  |  "
                  f"circuits = {qaoa_instance.total_circuit_count}  |  {ca_time:.2f}s")

    # ── Summary table ──
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    if E_ground is not None:
        print(f"  E₀ = {E_ground:.1f} (exact ground state)")
        print(f"  (lower energy = better, r closer to 1 = better)")
    else:
        print(f"  ⚠️  Ground state unknown (n={G.number_of_nodes()} > 22, brute force skipped)")
        print(f"  Using best energy found across all methods as reference.")
    print(f"{'='*70}")

    if E_ground is not None:
        print(f"  {'Method':<28} {'Best E':>8} {'Mean r':>8} {'Best r':>8} {'σ(r)':>8} {'Accept':>8}")
        print(f"  {'─'*68}")

        def print_row(name, result, show_accept=False):
            ratios = [e / E_ground for e in result.energy_history]
            mr = np.mean(ratios)
            sr = np.std(ratios)
            br = approx_ratio(result.best_energy)
            acc = f"{result.acceptance_rate:>7.1%}" if show_accept else f"{'—':>8}"
            print(f"  {name:<28} {result.best_energy:>8.1f} {mr:>8.3f} {br:>8.3f} {sr:>8.3f} {acc}")
    else:
        print(f"  {'Method':<28} {'Best E':>8} {'Mean E':>8} {'σ(E)':>8} {'Accept':>8}")
        print(f"  {'─'*60}")

        def print_row(name, result, show_accept=False):
            me = np.mean(result.energy_history)
            se = np.std(result.energy_history)
            acc = f"{result.acceptance_rate:>7.1%}" if show_accept else f"{'—':>8}"
            print(f"  {name:<28} {result.best_energy:>8.1f} {me:>8.1f} {se:>8.1f} {acc}")

    print_row("Simulated Annealing", sa_result, show_accept=False)
    print_row("Cluster (Coupling Const.)", cc_result, show_accept=True)
    for p, res in sorted(qaoa_results.items()):
        print_row(f"QAOA-Guided (p={p})", res, show_accept=True)

    print(f"{'='*70}")

    # ── Package results ──
    results = {
        "graph": G,
        "sa_result": sa_result,
        "cc_result": cc_result,
        "qaoa_results": qaoa_results,
        "E_ground": E_ground,
        "n_iterations_factor": n_iterations_factor,
    }

    # ── Generate plots ──
    print(f"\n🎨 Generating visualizations...")
    plot_approximation_ratios(results,
        save_path=os.path.join(output_dir, "1_approximation_ratios.png"))
    plot_correlation_heatmaps(G, qaoa_correlations,
        save_path=os.path.join(output_dir, "2_correlation_heatmaps.png"))
    plot_circuit_efficiency(qaoa_instances, G.number_of_edges(),
        save_path=os.path.join(output_dir, "3_circuit_efficiency.png"))
    plot_energy_distribution(results,
        save_path=os.path.join(output_dir, "4_energy_distributions.png"))

    print(f"\n✅ All done! Plots saved to {output_dir}/")
    return results


if __name__ == "__main__":
    results = run_benchmark(
        n_nodes=16,              # 16 qubits — local simulator.
        degree=12,               # Dense graph — more frustrated, harder for all methods.
        qaoa_depths=[1, 2, 3],  # p=2 is the key transition point (see AWS blog / paper Fig. 3).
        n_iterations_factor=500, # Generous budget — focus on correlation quality.
        n_repetitions=30,        # Enough restarts for statistics.
        lambda_scale=4,          # Cluster link scaling.
        seed=42,
        use_cloud=False,         # Set True for >18 qubits with QoroService.
        shots=10_000,
        output_dir="plots",
    )
