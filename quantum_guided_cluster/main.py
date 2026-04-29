"""
Quantum-Guided Cluster Algorithm for Max-Cut — benchmark runner.

Generates a graph, runs SA / coupling-constant / quantum-guided cluster
algorithms (QAOA at any depths plus optional PCE encodings), and saves the
comparison plots.

Reference: arXiv:2508.10656 (Eder et al., AWS Quantum Solutions Lab).
"""

import os
import time

import numpy as np

from algorithm import (
    ClusterAlgoResult,
    generate_random_maxcut_graph,
    ising_energy,
    extract_qaoa_correlations,
    extract_pce_correlations,
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
    pce_encodings: list[str] | None = None,
    use_qdrift: bool = False,
    n_iterations_factor: int = 100,
    n_repetitions: int = 20,
    lambda_scale: float = 6,
    seed: int = 42,
    use_cloud: bool = False,
    shots: int = 10_000,
    output_dir: str = ".",
):
    """Run the Quantum-Guided Cluster Algorithm benchmark.

    Args:
        n_nodes: Number of graph nodes.
        degree: Graph regularity. Use 10+ for hard instances.
        qaoa_depths: QAOA depths to compare. Each uses ``n_nodes`` qubits.
            ``None`` defaults to ``[1, 2, 3, 5]``; pass ``[]`` to skip QAOA.
        pce_encodings: PCE encodings (``"dense"``, ``"poly"``). Each compresses
            ``n_nodes`` variables into far fewer qubits. Defaults to ``[]``.
        use_qdrift: If True, every QAOA run uses QDrift trotterization —
            randomized Trotter sampling that produces shallower circuits at
            higher depths. Recommended for ``p ≥ 3``.
        n_iterations_factor: Total iterations = factor * n_nodes.
        n_repetitions: Number of random restarts per method.
        lambda_scale: Cluster formation scaling parameter.
        seed: Random seed.
        use_cloud: If True, use QoroService (for >18 qubits or larger PCE).
        shots: Number of measurement shots.
        output_dir: Directory for saving plots.
    """
    if qaoa_depths is None:
        qaoa_depths = [1, 2, 3, 5]
    if pce_encodings is None:
        pce_encodings = []
    os.makedirs(output_dir, exist_ok=True)

    G = generate_random_maxcut_graph(n_nodes, degree, seed=seed)
    print(f"Graph: {n_nodes} nodes, {degree}-regular, {G.number_of_edges()} edges. "
          f"Budget: {n_iterations_factor}×n iters × {n_repetitions} restarts.")

    # Brute-force ground state for n ≤ 22 — used for approximation ratios.
    E_ground: float | None = None
    if n_nodes <= 22:
        E_ground = min(
            ising_energy(G, np.array([1 - 2 * ((b >> i) & 1) for i in range(n_nodes)]))
            for b in range(2**n_nodes)
        )
        print(f"Exact ground state: E₀ = {E_ground:.1f}")

    if use_cloud:
        from divi.backends import QoroService, JobConfig
        backend = QoroService(job_config=JobConfig(shots=shots))
        print(f"Backend: QoroService (shots={shots})")
    else:
        backend = QiskitSimulator(shots=shots)
        print(f"Backend: QiskitSimulator (shots={shots})")

    def _format(label: str, result: ClusterAlgoResult, *, extra: str = "") -> str:
        line = f"  [{label:<24}] best E = {result.best_energy:7.1f}"
        if E_ground is not None:
            mean_r = float(np.mean([e / E_ground for e in result.energy_history]))
            best_r = result.best_energy / E_ground
            line += f" | mean r = {mean_r:.3f} | best r = {best_r:.3f}"
        if extra:
            line += f" | {extra}"
        return line

    # SA baseline.
    t0 = time.time()
    sa_result = simulated_annealing(
        G, n_iterations_factor=n_iterations_factor,
        n_repetitions=n_repetitions, seed=seed,
    )
    print(_format("SA", sa_result, extra=f"{time.time() - t0:.1f}s"))

    # Coupling-constant cluster.
    Z_cc = coupling_constant_correlations(G)
    t0 = time.time()
    cc_result = correlation_guided_cluster_algorithm(
        G, Z_cc, n_iterations_factor=n_iterations_factor,
        n_repetitions=n_repetitions, lambda_scale=1, seed=seed,
    )
    print(_format("Cluster (J coupling)", cc_result,
                  extra=f"accept={cc_result.acceptance_rate:.1%} | {time.time() - t0:.1f}s"))

    # Quantum-guided sources (QAOA + PCE share the same downstream pipeline).
    quantum_specs: list[tuple[str, dict]] = (
        [("qaoa", {"n_layers": p, "use_qdrift": use_qdrift}) for p in qaoa_depths]
        + [("pce", {"encoding": enc}) for enc in pce_encodings]
    )
    quantum_results: dict[str, ClusterAlgoResult] = {}
    quantum_correlations: dict[str, np.ndarray] = {"J (coupling\nconstants)": Z_cc}
    qaoa_instances: dict = {}  # int -> QAOA instance, used by the QWC efficiency plot.

    for kind, kwargs in quantum_specs:
        if kind == "qaoa":
            corr = extract_qaoa_correlations(
                G, max_iterations=10, shots=shots, backend=backend, **kwargs
            )
            qaoa_instances[kwargs["n_layers"]] = corr.instance
        else:
            corr = extract_pce_correlations(
                G, shots=shots, backend=backend, **kwargs
            )
        quantum_correlations[f"{corr.label}\n({corr.n_qubits} qubits)"] = corr.Z

        t0 = time.time()
        cluster_result = correlation_guided_cluster_algorithm(
            G, corr.Z, n_iterations_factor=n_iterations_factor,
            n_repetitions=n_repetitions, lambda_scale=lambda_scale, seed=seed,
        )
        quantum_results[corr.label] = cluster_result
        print(_format(
            f"{corr.label}-Guided", cluster_result,
            extra=(f"accept={cluster_result.acceptance_rate:.1%} | "
                   f"circuits={corr.total_circuit_count} | {time.time() - t0:.1f}s"),
        ))

    results = {
        "graph": G,
        "sa_result": sa_result,
        "cc_result": cc_result,
        "quantum_results": quantum_results,
        "E_ground": E_ground,
        "n_iterations_factor": n_iterations_factor,
    }

    plot_approximation_ratios(
        results, save_path=os.path.join(output_dir, "1_approximation_ratios.png")
    )
    plot_correlation_heatmaps(
        G, quantum_correlations,
        save_path=os.path.join(output_dir, "2_correlation_heatmaps.png"),
    )
    if qaoa_instances:
        plot_circuit_efficiency(
            qaoa_instances, G.number_of_edges(),
            save_path=os.path.join(output_dir, "3_circuit_efficiency.png"),
        )
    plot_energy_distribution(
        results, save_path=os.path.join(output_dir, "4_energy_distributions.png")
    )
    return results


if __name__ == "__main__":
    run_benchmark(
        n_nodes=16,
        degree=12,
        qaoa_depths=[1, 2, 3],
        # pce_encodings=["dense"],  # uncomment to add a PCE row to the comparison
        n_iterations_factor=500,
        n_repetitions=30,
        lambda_scale=4,
        seed=42,
        use_cloud=False,
        shots=10_000,
        output_dir="plots",
    )
