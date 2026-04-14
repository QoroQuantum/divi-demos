"""
Partitioned QAOA for MaxCut — Main Script
==========================================
Demonstrates how to solve the MaxCut problem on large, community-structured
graphs using Divi's GraphPartitioningQAOA. The graph is split into smaller
sub-graphs via spectral clustering, allowing parallel QAOA execution on
smaller quantum processors or simulators.

Usage:
    python main.py
"""

import time

import networkx as nx

from utils import generate_clustered_graph, show_graph, analyze_results

from divi.qprog.problems import MaxCutProblem
from divi.qprog.problems._graph_partitioning_utils import GraphPartitioningConfig

from divi.qprog.optimizers import (
    MonteCarloOptimizer,
)

from divi.qprog import QAOA

from divi.backends import QoroService, QiskitSimulator, JobConfig


if __name__ == "__main__":

    n_qubits = 20
    n_clusters = 4
    inter_edges = 10
    p_intra = 0.3
    seed = 42

    G, node_to_cluster, clusters = generate_clustered_graph(
        n_qubits=n_qubits,
        n_clusters=n_clusters,
        inter_edges=inter_edges,
        p_intra=p_intra,
        seed=seed,
        weight=1.0,
    )

    show_graph(G, node_to_cluster, n_qubits, n_clusters, inter_edges)

    # Classical approximation
    classical_cut_size, classical_partition = nx.approximation.one_exchange(G, seed=1)

    # Set up the optimizer
    optim = MonteCarloOptimizer(population_size=50, n_best_sets=5)

    # Set up the partitioning approach
    partition_config = GraphPartitioningConfig(
        minimum_n_clusters=4, partitioning_algorithm="spectral"
    )

    t0 = time.time()

    qaoa_problem = QAOA(
        problem=MaxCutProblem(graph=G, config=partition_config),
        n_layers=2,
        optimizer=optim,
        max_iterations=5,
        backend=QiskitSimulator(),
        grouping_strategy="qwc",
    )
    qaoa_problem.create_programs()
    qaoa_problem.run(blocking=True)
    qaoa_problem.aggregate_results()
    local_time = time.time() - t0

    analyze_results(G, qaoa_problem.solution, classical_cut_size, use_index=False)

    print("\n" + "=" * 70)
    print("  Phase 2 — Scale Up with QoroService (50 nodes)")
    print("=" * 70)

    n_qubits_cloud = 50
    n_clusters_cloud = 5

    G_cloud, node_to_cluster_cloud, clusters_cloud = generate_clustered_graph(
        n_qubits=n_qubits_cloud,
        n_clusters=n_clusters_cloud,
        inter_edges=5,
        p_intra=0.2,
        seed=seed,
        weight=1.0,
    )

    show_graph(G_cloud, node_to_cluster_cloud, n_qubits_cloud, n_clusters_cloud, 5)

    classical_cut_size_cloud, _ = nx.approximation.one_exchange(G_cloud, seed=1)

    partition_config_cloud = GraphPartitioningConfig(
        max_n_nodes_per_cluster=15, partitioning_algorithm="spectral"
    )

    qoro_service = QoroService(job_config=JobConfig(shots=50_000))

    print(f"\n☁️  Routing {n_qubits_cloud}-node graph to Qoro Maestro...")
    print(f"   Partitioning into ~15 qubit sub-circuits (parallel on Maestro)...")
    t0 = time.time()

    qaoa_cloud = QAOA(
        problem=MaxCutProblem(graph=G_cloud, config=partition_config_cloud),
        n_layers=2,
        optimizer=optim,
        max_iterations=5,
        backend=qoro_service,
        grouping_strategy="qwc",
    )
    qaoa_cloud.create_programs()
    qaoa_cloud.run(blocking=True)
    qaoa_cloud.aggregate_results()
    cloud_time = time.time() - t0

    print(f"\n   ✅ Phase 2 complete in {cloud_time:.1f}s")
    print(f"   ⚡ Cloud (Phase 2): {cloud_time:.1f}s for {n_qubits_cloud} nodes")

    analyze_results(G_cloud, qaoa_cloud.solution, classical_cut_size_cloud, use_index=False)

    print("\n" + "=" * 70)
    print("  🎉 That's the power of QoroService.")
    print(f"     Your laptop handled {n_qubits} nodes. Qoro Maestro handled {n_qubits_cloud}.")
    print("     👉 https://dash.qoroquantum.net")
    print("=" * 70)
