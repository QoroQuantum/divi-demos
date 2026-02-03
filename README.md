# Divi Examples

Example programs written in Divi

## Examples

### 1. [Cluster MaxCut](./cluster_maxcut)

Partitioned QAOA for MaxCut using graph partitioning to solve the MaxCut problem on large, community-structured graphs. This example demonstrates how to split large graphs (e.g., 50 qubits) into smaller, manageable sub-graphs using spectral clustering, allowing for parallel execution of QAOA circuits on smaller quantum processors or simulators.

**Key Features:**
- Graph partitioning with spectral clustering
- Parallel QAOA execution on sub-graphs
- Comparison of quantum vs. classical results
- Support for both local simulation and remote execution via QoroService

### 2. [Minimum Birkhoff Decomposition](./minimum_birkhoff_decomposition)

VQE-based approach to find the Birkhoff decomposition of doubly stochastic matrices. This example showcases the modular design of the `divi` library, implementing a sophisticated application by inheriting from the VQE class with minimal, targeted changes.

**Key Features:**
- Custom VQE implementation for Birkhoff decomposition
- Multi-threaded classical optimization with caching
- Command-line interface for various problem configurations
- Support for sparse and dense matrix types

### 3. [Portfolio Optimization](./portfolio_optimization)

Quantum portfolio optimization using QAOA combined with spectral partitioning to handle large-scale problems. The approach partitions the asset correlation graph into smaller sub-problems that can be solved efficiently on quantum hardware.

**Key Features:**
- QUBO formulation for portfolio optimization
- Spectral partitioning based on asset correlations
- Interactive Jupyter notebook workflow
- Solution comparison between QAOA and exact solvers
- Comprehensive visualization and financial metrics analysis
