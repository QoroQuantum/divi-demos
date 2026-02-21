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

import networkx as nx

from utils import generate_clustered_graph, show_graph, analyze_results

from divi.qprog.algorithms import GraphProblem

from divi.qprog.optimizers import (
    MonteCarloOptimizer,
)

from divi.qprog import (
    GraphPartitioningQAOA,
    PartitioningConfig,
)

from divi.backends import QoroService, ParallelSimulator, JobConfig


if __name__ == "__main__":

    n_qubits = 50
    n_clusters = 5
    inter_edges = 5
    p_intra = 0.2
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
    partition_config = PartitioningConfig(
        minimum_n_clusters=10, partitioning_algorithm="spectral"
    )

    qaoa_problem = GraphPartitioningQAOA(
        graph=G,
        graph_problem=GraphProblem.MAXCUT,
        n_layers=2,
        optimizer=optim,
        partitioning_config=partition_config,
        max_iterations=5,
        backend=ParallelSimulator(),
        grouping_strategy="qwc",
    )
    qaoa_problem.create_programs()
    qaoa_problem.run(blocking=True)
    qaoa_problem.aggregate_results()

    analyze_results(G, qaoa_problem.solution, classical_cut_size, use_index=False)

    # With Qoro Service, assuming environment variable QORO_API_KEY is set
    # qoro_service = QoroService(config=JobConfig(shots=50_000))

    # qaoa_problem = GraphPartitioningQAOA(
    #     graph=G,
    #     graph_problem=GraphProblem.MAXCUT,
    #     n_layers=2,
    #     optimizer=optim,
    #     partitioning_config=partition_config,
    #     max_iterations=5,
    #     backend=qoro_service,
    #     grouping_strategy="qwc",
    # )
    # qaoa_problem.create_programs()
    # qaoa_problem.run(blocking=True)
    # qaoa_problem.aggregate_results()
