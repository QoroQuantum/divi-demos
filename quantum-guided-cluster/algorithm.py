"""
Quantum-Guided Cluster Algorithm for Max-Cut
=============================================
Core algorithm implementation from:
  "Quantum-Guided Cluster Algorithms for Combinatorial Optimization"
  (arXiv:2508.10656)

Contains:
  - Graph generation and cost functions
  - QAOA correlation extraction via Divi
  - Correlation-guided cluster Monte Carlo (Algorithm 1)
  - Simulated annealing baseline
  - Coupling-constant baseline
"""

import math
import random
import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from divi.backends import ParallelSimulator
from divi.qprog import QAOA, GraphProblem
from divi.qprog.optimizers import PymooOptimizer, PymooMethod
from divi.qprog import QDrift


# ──────────────────────────────────────────────────────────────────
# 1. Graph generation
# ──────────────────────────────────────────────────────────────────
def generate_random_maxcut_graph(
    n: int, degree: int, seed: int = 42
) -> nx.Graph:
    """Generate a random regular graph with ±1 edge weights (Ising spin glass).

    Following the paper's convention, weights are drawn uniformly from {-1, +1}.
    """
    rng = np.random.default_rng(seed)
    G = nx.random_regular_graph(degree, n, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = rng.choice([-1.0, 1.0])
    return G


# ──────────────────────────────────────────────────────────────────
# 2. Max-Cut cost evaluation
# ──────────────────────────────────────────────────────────────────
def maxcut_cost(G: nx.Graph, config: np.ndarray) -> float:
    """Compute the weighted Max-Cut cost C(x) = Σ_{(i,j)∈E} w_ij * (1 - x_i x_j) / 2."""
    cost = 0.0
    for u, v, w in G.edges(data="weight", default=1.0):
        cost += w * (1.0 - config[u] * config[v]) / 2.0
    return cost


def unweighted_cut_size(G: nx.Graph, config: np.ndarray) -> int:
    """Count the number of edges crossing the partition (unweighted)."""
    return sum(1 for u, v in G.edges() if config[u] != config[v])


def ising_energy(G: nx.Graph, config: np.ndarray) -> float:
    """Compute Ising energy H = -Σ_{(i,j)} J_ij x_i x_j."""
    energy = 0.0
    for u, v, w in G.edges(data="weight", default=1.0):
        energy -= w * config[u] * config[v]
    return energy


# ──────────────────────────────────────────────────────────────────
# 3. Extract two-point correlations from QAOA using Divi
# ──────────────────────────────────────────────────────────────────
def extract_qaoa_correlations(
    G: nx.Graph,
    n_layers: int,
    max_iterations: int = 10,
    shots: int = 10_000,
    use_qdrift: bool = False,
    backend=None,
) -> tuple[np.ndarray, "QAOA"]:
    """Run QAOA on a Max-Cut instance and extract the ZZ correlation matrix.

    The correlation Z_ij = ⟨ψ_opt| σ_i^z σ_j^z |ψ_opt⟩ is computed for every
    edge (i,j) in the graph. Divi's QWC (qubit-wise commuting) grouping batches
    all ZZ observables into a minimal number of measurement circuits.

    Args:
        G: The graph (Max-Cut instance).
        n_layers: Number of QAOA layers (circuit depth p).
        max_iterations: Optimizer iterations.
        shots: Number of measurement shots.
        use_qdrift: If True, use QDrift trotterization for shallower circuits.
        backend: Divi backend to use (e.g. ParallelSimulator or QoroService).
                 Defaults to ParallelSimulator if not provided.

    Returns:
        Z: (n x n) correlation matrix where Z[i][j] = ⟨Z_i Z_j⟩.
        qaoa_problem: The solved QAOA instance for further inspection.
    """
    n = G.number_of_nodes()

    if backend is None:
        backend = ParallelSimulator(shots=shots)

    # Build the QAOA kwargs
    qaoa_kwargs = dict(
        problem=G,
        graph_problem=GraphProblem.MAXCUT,
        n_layers=n_layers,
        optimizer=PymooOptimizer(method=PymooMethod.DE, population_size=20),
        max_iterations=max_iterations,
        backend=backend,
        # QWC grouping: groups commuting ZZ observables together
        # so multiple correlations are measured from a single circuit
        grouping_strategy="qwc",
    )

    if use_qdrift:
        qdrift = QDrift(
            keep_fraction=0.3,
            sampling_budget=8,
            n_hamiltonians_per_iteration=3,
            sampling_strategy="weighted",
            seed=42,
        )
        qaoa_kwargs["trotterization_strategy"] = qdrift

    qaoa_problem = QAOA(**qaoa_kwargs)

    print(f"  Running QAOA with p={n_layers} layers...")
    t0 = time.time()
    qaoa_problem.run(perform_final_computation=True)
    elapsed = time.time() - t0
    print(f"  QAOA done in {elapsed:.1f}s  |  "
          f"circuits={qaoa_problem.total_circuit_count}  |  "
          f"best_loss={qaoa_problem.best_loss:.4f}")

    # --- Extract ZZ correlations from the measurement distribution ---
    # Divi's best_probs is a dict keyed by CircuitTag -> {bitstring: probability}
    raw_probs = qaoa_problem.best_probs
    if raw_probs is None:
        raise RuntimeError("QAOA did not produce probability data.")

    # Extract the actual probability distribution from the first measurement group
    probs: dict[str, float] = {}
    for tag, shots_dict in raw_probs.items():
        if isinstance(shots_dict, dict):
            for bitstring, prob in shots_dict.items():
                probs[bitstring] = probs.get(bitstring, 0.0) + prob
            break  # Use the first measurement group

    if not probs:
        raise RuntimeError("Could not extract probability data from QAOA results.")

    # Compute Z_ij = ⟨Z_i Z_j⟩ = Σ_x p(x) * x_i * x_j
    # where x_i = (-1)^{bit_i}  (mapping 0->+1, 1->-1 following Ising convention)
    Z = np.zeros((n, n))

    for bitstring, prob in probs.items():
        # Convert bitstring to ±1 spin values
        spins = np.array([1 - 2 * int(b) for b in bitstring])
        # Accumulate outer product weighted by probability
        Z += prob * np.outer(spins, spins)

    # Diagonal is always 1 (⟨Z_i^2⟩ = 1), enforce it
    np.fill_diagonal(Z, 1.0)

    return Z, qaoa_problem


# ──────────────────────────────────────────────────────────────────
# 4. Correlation-Guided Cluster Algorithm (Algorithm 1 from paper)
# ──────────────────────────────────────────────────────────────────
@dataclass
class ClusterAlgoResult:
    """Result container for the cluster algorithm."""
    best_config: np.ndarray
    best_cut: int
    best_energy: float
    cut_history: list[int] = field(default_factory=list)
    energy_history: list[float] = field(default_factory=list)
    acceptance_rate: float = 0.0


def estimate_percolation_threshold(G: nx.Graph) -> float:
    """Estimate the bond percolation threshold of a graph (Eq. 4 from paper).

    p_c ≈ (⟨d⟩ - 1) / (⟨d²⟩ - ⟨d⟩)
    where ⟨d⟩ and ⟨d²⟩ are the first and second moments of the degree distribution.
    """
    degrees = np.array([d for _, d in G.degree()])
    d_mean = degrees.mean()
    d2_mean = (degrees ** 2).mean()
    if d2_mean - d_mean < 1e-10:
        return 0.5
    return (d_mean - 1.0) / (d2_mean - d_mean)


def create_cluster(
    G: nx.Graph,
    seed_node: int,
    config: np.ndarray,
    Z: np.ndarray,
    lambda_scale: float,
    p_c: float,
    mean_abs_z: float,
    rng: random.Random,
) -> set[int]:
    """Build a cluster starting from seed_node using the correlation matrix Z.

    Implements the CreateCluster procedure from the paper (Fig. 3):
    grow a cluster via graph contraction, adding neighbors with probability
    p_link proportional to the correlation strength between a cluster node
    and its neighbor.

    Args:
        G: The graph.
        seed_node: Starting node for cluster growth.
        config: Current spin configuration (±1).
        Z: Correlation matrix.
        lambda_scale: Scaling hyperparameter (λ_scale from paper).
        p_c: Estimated percolation threshold.
        mean_abs_z: Mean absolute value of nonzero off-diagonal correlations.
        rng: Random number generator.

    Returns:
        Set of node indices comprising the cluster.
    """
    cluster = {seed_node}
    # frontier stores (parent_in_cluster, neighbor_candidate) pairs
    frontier = [(seed_node, nb) for nb in G.neighbors(seed_node)]
    rng.shuffle(frontier)
    visited = {seed_node}

    while frontier:
        parent, neighbor = frontier.pop()

        if neighbor in visited:
            continue
        visited.add(neighbor)

        # Compute link probability (Eq. 3 from paper)
        # Use the correlation between the parent cluster node and the candidate
        z_ij = Z[parent, neighbor]

        # The correlation sign should align with spin agreement
        agreement = config[parent] * config[neighbor]
        effective_z = z_ij * agreement

        if effective_z <= 0:
            continue

        # p_link = lambda_scale * p_c * |Z_ij| / (2 * E[|Z|])
        if mean_abs_z < 1e-10:
            p_link = p_c
        else:
            p_link = lambda_scale * p_c * abs(z_ij) / (2.0 * mean_abs_z)

        p_link = min(p_link, 1.0)

        if rng.random() < p_link:
            cluster.add(neighbor)

            # Expand frontier from newly added node (graph contraction)
            new_edges = [(neighbor, nn) for nn in G.neighbors(neighbor) if nn not in visited]
            rng.shuffle(new_edges)
            frontier.extend(new_edges)

    return cluster


def correlation_guided_cluster_algorithm(
    G: nx.Graph,
    Z: np.ndarray,
    n_iterations_factor: int = 500,
    n_repetitions: int = 10,
    lambda_scale: float = 6,
    beta_f: float = 8.0,
    seed: int = 42,
) -> ClusterAlgoResult:
    """Run the correlation-guided cluster algorithm (Algorithm 1 from paper).

    This implements the full cluster-based simulated annealing where clusters
    are formed using the precomputed correlation matrix Z and flipped as a unit.

    Args:
        G: The Max-Cut graph.
        Z: (n x n) two-point correlation matrix from QAOA or other source.
        n_iterations_factor: Total iterations = n_iterations_factor * n.
        n_repetitions: Number of independent random restarts.
        lambda_scale: Scaling factor for link probability.
        beta_f: Final inverse temperature.
        seed: Random seed.

    Returns:
        ClusterAlgoResult with best configuration found.
    """
    n = G.number_of_nodes()
    total_iters = n_iterations_factor * n
    rng = random.Random(seed)

    # Precompute graph properties
    p_c = estimate_percolation_threshold(G)

    # Mean absolute value of nonzero off-diagonal Z entries
    mask = ~np.eye(n, dtype=bool)
    nonzero_z = np.abs(Z[mask])
    nonzero_z = nonzero_z[nonzero_z > 1e-10]
    mean_abs_z = nonzero_z.mean() if len(nonzero_z) > 0 else 1.0

    overall_best_config = None
    overall_best_cut = -np.inf
    overall_best_energy = np.inf
    all_cuts = []
    all_energies = []

    n_accepted = 0
    n_total_proposals = 0

    for rep in range(n_repetitions):
        # Initialize random spin configuration
        config = np.array([rng.choice([-1, 1]) for _ in range(n)])
        current_energy = ising_energy(G, config)

        best_config = config.copy()
        best_energy = current_energy

        iteration = 0
        while iteration < total_iters:
            # Inverse temperature schedule (linear)
            beta = 1.0 + (beta_f - 1.0) * iteration / total_iters

            # Pick a random seed node and build a cluster
            seed_node = rng.randint(0, n - 1)
            cluster = create_cluster(
                G, seed_node, config, Z, lambda_scale, p_c, mean_abs_z, rng
            )
            cluster_size = len(cluster)

            # Compute energy difference from flipping the entire cluster
            delta_e = 0.0
            for node in cluster:
                for neighbor in G.neighbors(node):
                    w = G[node][neighbor].get("weight", 1.0)
                    if neighbor in cluster:
                        pass  # Internal edge: both flip, no change
                    else:
                        # External edge: energy change = 2 * J_ij * x_i * x_j
                        delta_e += 2.0 * w * config[node] * config[neighbor]

            # Metropolis acceptance criterion
            n_total_proposals += 1
            if delta_e <= 0 or rng.random() < math.exp(-beta * delta_e):
                # Accept: flip the cluster
                for node in cluster:
                    config[node] *= -1
                current_energy += delta_e
                n_accepted += 1

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_config = config.copy()

            # Each cluster flip of size k counts as k iterations
            iteration += cluster_size

        best_cut = unweighted_cut_size(G, best_config)
        all_cuts.append(best_cut)
        all_energies.append(best_energy)

        if best_energy < overall_best_energy:
            overall_best_energy = best_energy
            overall_best_cut = best_cut
            overall_best_config = best_config.copy()

    acceptance_rate = n_accepted / max(n_total_proposals, 1)

    return ClusterAlgoResult(
        best_config=overall_best_config,
        best_cut=overall_best_cut,
        best_energy=overall_best_energy,
        cut_history=all_cuts,
        energy_history=all_energies,
        acceptance_rate=acceptance_rate,
    )


# ──────────────────────────────────────────────────────────────────
# 5. Standard Simulated Annealing (baseline)
# ──────────────────────────────────────────────────────────────────
def simulated_annealing(
    G: nx.Graph,
    n_iterations_factor: int = 500,
    n_repetitions: int = 10,
    beta_f: float = 8.0,
    seed: int = 42,
) -> ClusterAlgoResult:
    """Standard single-spin-flip simulated annealing for comparison."""
    n = G.number_of_nodes()
    total_iters = n_iterations_factor * n
    rng = random.Random(seed)

    overall_best_config = None
    overall_best_cut = -np.inf
    overall_best_energy = np.inf
    all_cuts = []
    all_energies = []

    for rep in range(n_repetitions):
        config = np.array([rng.choice([-1, 1]) for _ in range(n)])
        current_energy = ising_energy(G, config)
        best_config = config.copy()
        best_energy = current_energy

        for iteration in range(total_iters):
            beta = 1.0 + (beta_f - 1.0) * iteration / total_iters

            # Single-spin flip
            node = rng.randint(0, n - 1)
            delta_e = 0.0
            for neighbor in G.neighbors(node):
                w = G[node][neighbor].get("weight", 1.0)
                delta_e += 2.0 * w * config[node] * config[neighbor]

            if delta_e <= 0 or rng.random() < math.exp(-beta * delta_e):
                config[node] *= -1
                current_energy += delta_e

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_config = config.copy()

        best_cut = unweighted_cut_size(G, best_config)
        all_cuts.append(best_cut)
        all_energies.append(best_energy)

        if best_energy < overall_best_energy:
            overall_best_energy = best_energy
            overall_best_cut = best_cut
            overall_best_config = best_config.copy()

    return ClusterAlgoResult(
        best_config=overall_best_config,
        best_cut=overall_best_cut,
        best_energy=overall_best_energy,
        cut_history=all_cuts,
        energy_history=all_energies,
    )


# ──────────────────────────────────────────────────────────────────
# 6. Coupling-constant baseline (Z_ij = J_ij)
# ──────────────────────────────────────────────────────────────────
def coupling_constant_correlations(G: nx.Graph) -> np.ndarray:
    """Build a correlation matrix directly from the coupling constants J_ij.

    This serves as the simplest baseline — the cluster algorithm uses only
    the graph's edge weights as correlation information (Section II-C1).
    """
    n = G.number_of_nodes()
    Z = np.zeros((n, n))
    for u, v, w in G.edges(data="weight", default=1.0):
        Z[u, v] = w
        Z[v, u] = w
    np.fill_diagonal(Z, 1.0)
    return Z
