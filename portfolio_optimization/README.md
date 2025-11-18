# Portfolio Optimization with Quantum Algorithms

This project demonstrates portfolio optimization using quantum algorithms (QAOA) combined with spectral partitioning to handle large-scale problems. The approach partitions the asset correlation graph into smaller sub-problems that can be solved efficiently on quantum hardware.

## Overview

Portfolio optimization aims to select assets that maximize return while minimizing risk. This is formulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem, which can be solved using the Quantum Approximate Optimization Algorithm (QAOA).

For large portfolios (hundreds of assets), the problem is partitioned using spectral graph partitioning based on asset correlations. Each partition is solved independently, and the solutions are aggregated into a global portfolio.

## Key Concepts

- **QUBO Formulation**: The portfolio optimization problem is expressed as `Minimize: Risk - λ·Return`, where λ balances risk and return preferences.

- **Spectral Partitioning**: Assets are grouped into clusters based on their correlation structure, ensuring that highly correlated assets are in the same partition.

- **QAOA**: Quantum Approximate Optimization Algorithm solves each partition's QUBO problem using quantum circuits.

- **Solution Aggregation**: Individual partition solutions are combined into a global portfolio bitstring, maintaining the original asset order.

## Files

- **`portfolio_optimization.ipynb`**: Main Jupyter notebook containing the complete workflow, from data loading to solution comparison. This is the primary entry point for exploring the project.

- **`utils.py`**: Utility functions for:
  - Building QUBO matrices from returns and covariance data
  - Aggregating partition solutions into global portfolios
  - Comparing QAOA and exact solver solutions
  - Computing financial metrics (return, risk, Sharpe ratio)

- **`visualization.py`**: Visualization functions for:
  - Correlation matrix heatmaps
  - Partition structure analysis
  - Partition quality metrics (modularity, separation ratio)
  - Lambda parameter selection guidance

- **`qubo_batch.py`**: Custom batch processing class for solving multiple QUBO problems in parallel using QAOA.

- **`modularity_spectral_partitioning.py`**: Spectral partitioning implementation that groups assets into clusters based on correlation structure.

## Data Files

The project uses financial data files (`.npy` format):

- `2016-01-01_returns.npy`: Expected returns for each asset
- `2016-01-01_covariance.npy`: Covariance matrix between assets
- `2016-01-01_correlation.npy`: Correlation matrix between assets

## Requirements

The core logic depends on the `divi` library from Qoro Quantum. You can install it using pip:

```bash
pip install qoro-divi==0.4.2
```

## How to Use

### Quick Start

1. **Open the Jupyter notebook**:

   ```bash
   jupyter notebook portfolio_optimization.ipynb
   ```

2. **Run the cells sequentially** to:
   - Load and scale financial data (scaling values appropriately for quantum algorithms)
   - Partition assets using spectral clustering and visualize partition quality
   - Build QUBO matrices for each partition using the risk-return formulation
   - Solve each partition using QAOA (quantum approximate) and ExactSolver (classical optimal)
   - Aggregate partition solutions into global portfolios
   - Compare solutions using financial metrics (return, risk, Sharpe ratio)

### Key Parameters

- **`MAX_PARTITION_SIZE`**: Maximum number of assets per partition (default: 20). Smaller partitions are easier for quantum hardware but may reduce solution quality.

- **`LAMBDA_PARAM`**: Risk-return trade-off parameter (default: 0.75). Higher values favor return over risk minimization.

- **QAOA Parameters**: `n_layers` (number of QAOA layers) and `max_iterations` (optimization iterations) control the quantum algorithm's performance.

## Results Interpretation

The comparison between QAOA and ExactSolver solutions provides insights into:

- **Solution Quality**: How close QAOA gets to the optimal solution
- **Risk-Return Trade-off**: Whether QAOA finds portfolios with better risk-adjusted returns
- **Asset Selection**: Which assets are selected by each method

Typically, QAOA may select more assets (higher return, higher risk) while ExactSolver finds more efficient portfolios (better Sharpe ratio).
