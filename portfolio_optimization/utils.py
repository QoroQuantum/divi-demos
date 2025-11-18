"""
Utility functions for portfolio optimization.

This module provides helper functions for building QUBO matrices
and other common operations in portfolio optimization.
"""

import numpy as np
import numpy.typing as npt
import dimod


def _get_partition_solution(
    partition_solutions: dict[int | str, npt.NDArray[np.integer]],
    partition_id: int,
) -> npt.NDArray[np.integer] | None:
    """Get partition solution, trying int key first, then str key."""
    solution = partition_solutions.get(partition_id)
    if solution is None:
        solution = partition_solutions.get(str(partition_id))
    return solution


def _extract_dimod_solution(sample_set: dimod.SampleSet) -> npt.NDArray[np.integer]:
    """Extract best solution from dimod SampleSet and convert to numpy array."""
    best_record = sample_set.lowest().record[0]
    best_sample = best_record.sample

    # Handle both dict-like SampleView and numpy array
    if hasattr(best_sample, "keys"):
        # It's a dict-like SampleView
        sorted_vars = sorted(best_sample.keys())
        return np.array([best_sample[var] for var in sorted_vars], dtype=np.int_)
    else:
        # It's already a numpy array or array-like
        return np.asarray(best_sample, dtype=np.int_)


def _build_qubo_matrix(
    returns: npt.NDArray[np.floating],
    covariance_matrix: npt.NDArray[np.floating],
    lambda_param: float = 0.75,
) -> npt.NDArray[np.floating]:
    """
    Build QUBO matrix for portfolio optimization.

    QUBO formulation: Minimize Risk - λ·Return
    Which is: x^T Σ x - λ μ^T x

    In QUBO matrix form:
    - Diagonal (linear terms): Q_ii = Σ_ii - λ·μ_i  (variance - lambda*return)
    - Off-diagonal (quadratic terms): Q_ij = Σ_ij  (covariance)

    Args:
        returns: Expected returns vector. Shape: (n,)
        covariance_matrix: Covariance matrix. Shape: (n, n)
        lambda_param: Risk-return trade-off parameter. Default 0.75.

    Returns:
        QUBO matrix Q where the objective is minimize x^T Q x
    """
    qubo_matrix = covariance_matrix.copy()

    # Diagonal = variance - lambda * return
    # (variance is already on diagonal, subtract lambda*return)
    np.fill_diagonal(qubo_matrix, np.diag(qubo_matrix) - lambda_param * returns)

    # Off-diagonal = covariance (already correct)
    # Ensure symmetry
    if not np.allclose(qubo_matrix, qubo_matrix.T):
        qubo_matrix = (qubo_matrix + qubo_matrix.T) / 2

    return qubo_matrix


def build_qubo_matrices_for_all_partitions(
    partitioned_returns_dict: dict[int, npt.NDArray[np.floating]],
    partitioned_covariance_dict: dict[int, npt.NDArray[np.floating]],
    lambda_param: float = 0.75,
) -> dict[int, npt.NDArray[np.floating]]:
    """
    Build QUBO matrices for all partitions.

    This ensures all solvers work on the exact same QUBO formulation,
    enabling fair comparison between different solving methods.

    Args:
        partitioned_returns_dict: Dictionary mapping partition IDs to return vectors
        partitioned_covariance_dict: Dictionary mapping partition IDs to covariance matrices
        lambda_param: Risk-return trade-off parameter. Default 0.75.

    Returns:
        Dictionary mapping partition IDs to QUBO matrices
    """
    qubo_matrices = {}
    for partition_id in partitioned_returns_dict.keys():
        returns = partitioned_returns_dict[partition_id]
        covariance = partitioned_covariance_dict[partition_id]
        qubo_matrices[partition_id] = _build_qubo_matrix(
            returns, covariance, lambda_param
        )
    return qubo_matrices


def aggregate_partition_solutions(
    partition_solutions: dict[int | str, npt.NDArray[np.integer]],
    partitions: npt.NDArray[np.integer],
    skip_missing: bool = False,
) -> npt.NDArray[np.integer]:
    """
    Aggregate bitstring solutions from all partitions into a global portfolio bitstring.

    Each partition's solution is in local space (indices 0 to partition_size-1).
    This function maps them back to global asset indices using efficient vectorized operations.

    Args:
        partition_solutions: Dictionary mapping partition IDs to local bitstring solutions.
                           Each bitstring is a 1D array of 0s and 1s.
                           Supports both int and str keys (tries int first, then str).
        partitions: Array mapping each global asset index to its partition ID.
                   Shape: (n_assets,)
        skip_missing: If True, skip partitions without solutions (set to 0).
                      If False, raise KeyError when partition is missing. Default: False.

    Returns:
        Global bitstring solution of shape (n_assets,) where:
        - 1 means the asset is included in the portfolio
        - 0 means the asset is excluded

    Raises:
        KeyError: If a partition is missing from solutions and skip_missing=False.
        ValueError: If a partition's solution length doesn't match its size.

    Example:
        >>> partitions = np.array([0, 0, 1, 1, 0])  # Assets 0,1,4 in partition 0; 2,3 in partition 1
        >>> partition_solutions = {
        ...     0: np.array([1, 0, 1]),  # Local: include assets 0,2 (global: 0,4)
        ...     1: np.array([0, 1])      # Local: include asset 1 (global: 3)
        ... }
        >>> aggregate_partition_solutions(partition_solutions, partitions)
        array([1, 0, 0, 1, 1])  # Global: include assets 0, 3, 4

        >>> # With string keys
        >>> partition_solutions_str = {
        ...     "0": np.array([1, 0, 1]),
        ...     "1": np.array([0, 1])
        ... }
        >>> aggregate_partition_solutions(partition_solutions_str, partitions)
        array([1, 0, 0, 1, 1])
    """
    partitions = np.asarray(partitions)
    n_assets = len(partitions)
    global_bitstring = np.zeros(n_assets, dtype=np.int_)

    unique_partitions = np.unique(partitions)

    for partition_id in unique_partitions:
        # Get partition solution (handles both int and str keys)
        local_bitstring = _get_partition_solution(partition_solutions, partition_id)

        if local_bitstring is None:
            if skip_missing:
                # Skip this partition (already set to 0)
                continue
            else:
                # Raise error with helpful message
                available_keys = list(partition_solutions.keys())[:10]
                raise KeyError(
                    f"Partition ID {partition_id} not found in solutions. "
                    f"Available keys (first 10): {available_keys}"
                )

        # Get global indices for this partition (efficient vectorized operation)
        global_indices = np.flatnonzero(partitions == partition_id)

        # Validate length with detailed error message
        if len(local_bitstring) != len(global_indices):
            # Determine which key was actually used
            used_key = (
                partition_id
                if partition_id in partition_solutions
                else str(partition_id)
            )
            # Check if there's a mismatch in key types
            other_key = (
                str(partition_id)
                if partition_id in partition_solutions
                else partition_id
            )
            other_solution = partition_solutions.get(other_key)

            error_msg = (
                f"Partition {partition_id}: Local bitstring length ({len(local_bitstring)}) "
                f"does not match partition size ({len(global_indices)}). "
                f"Used key: {used_key!r}"
            )

            # If both keys exist, warn about potential key collision
            if other_solution is not None and len(other_solution) != len(
                local_bitstring
            ):
                error_msg += (
                    f" Note: Key {other_key!r} also exists with length {len(other_solution)}. "
                    f"Available keys: {sorted(partition_solutions.keys())}"
                )

            raise ValueError(error_msg)

        # Map local solution to global positions (vectorized assignment)
        global_bitstring[global_indices] = local_bitstring

    return global_bitstring


# Compute financial metrics for both solutions
def _compute_portfolio_metrics(
    returns: np.ndarray,
    covariance_matrix: np.ndarray,
    bitstring: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute portfolio return, risk, and Sharpe ratio for a given bitstring.

    Sharpe Ratio = Return / sqrt(Risk)
    (Using variance as risk, so we take square root to get standard deviation)
    """
    portfolio_return = np.dot(returns, bitstring)
    portfolio_risk = bitstring @ covariance_matrix @ bitstring
    # Sharpe ratio: Return / Standard Deviation (sqrt of variance)
    # Handle edge case where risk is zero (shouldn't happen in practice)
    sharpe_ratio = (
        portfolio_return / np.sqrt(portfolio_risk)
        if portfolio_risk > 0
        else float("inf")
    )
    return portfolio_return, portfolio_risk, sharpe_ratio


def evaluate_solution(
    qubo_matrix: np.ndarray,
    qaoa_solution: np.ndarray,
    returns: np.ndarray,
    covariance_matrix: np.ndarray,
    partition_id: int | None = None,
):
    """
    Evaluate QAOA solution by comparing it to the optimal classical solution.

    Computes both QUBO energies and actual financial metrics (returns and risk).

    Args:
        qubo_matrix: The QUBO matrix for this partition (already built with lambda)
        qaoa_solution: The bitstring solution from QAOA
        returns: Expected returns vector for the assets in this partition
        covariance_matrix: Covariance matrix for the assets in this partition
        lambda_param: Lambda parameter used to build the QUBO (for verification)
        partition_id: Optional ID of the cluster/partition (for logging only)
    """
    # Validate inputs
    if qubo_matrix.shape[0] != qubo_matrix.shape[1]:
        raise ValueError(f"QUBO matrix must be square, got shape {qubo_matrix.shape}")

    if len(qaoa_solution) != qubo_matrix.shape[0]:
        raise ValueError(
            f"Solution length {len(qaoa_solution)} does not match QUBO size {qubo_matrix.shape[0]}"
        )

    if len(returns) != qubo_matrix.shape[0]:
        raise ValueError(
            f"Returns vector length {len(returns)} does not match QUBO size {qubo_matrix.shape[0]}"
        )

    if covariance_matrix.shape != qubo_matrix.shape:
        raise ValueError(
            f"Covariance matrix shape {covariance_matrix.shape} does not match QUBO shape {qubo_matrix.shape}"
        )

    # Solve classically to get optimal solution for comparison
    bqm = dimod.BinaryQuadraticModel(qubo_matrix, "BINARY")
    sample_set = dimod.ExactSolver().sample(bqm)
    best_classical_bitstring = _extract_dimod_solution(sample_set)
    best_classical_energy = sample_set.lowest().record[0].energy

    qaoa_energy = bqm.energy(qaoa_solution)
    is_correct = np.array_equal(best_classical_bitstring, qaoa_solution)

    # Print evaluation results
    partition_label = (
        f"Partition {partition_id}" if partition_id is not None else "Partition"
    )
    print(f"{partition_label} Evaluation:")
    print()
    print("QUBO Energy Metrics:")
    print(f"  Optimal Classical Energy: {best_classical_energy:.6f}")
    print(f"  QAOA Energy: {qaoa_energy:.6f}")
    print(f"  Energy Difference: {qaoa_energy - best_classical_energy:.6f}")
    print(f"  Matches Optimal: {is_correct}")

    # Use compare_portfolio_solutions for financial metrics (swapped order for Optimal vs QAOA)
    compare_portfolio_solutions(
        qaoa_solution,
        best_classical_bitstring,
        returns,
        covariance_matrix,
    )

    print()
    print("Solutions:")
    print(f"  Optimal Bitstring: {best_classical_bitstring}")
    print(f"  QAOA Bitstring:    {str(qaoa_solution).replace(',', '')}")


def compare_portfolio_solutions(
    qaoa_solution: npt.NDArray[np.integer],
    exact_solution: npt.NDArray[np.integer],
    returns: npt.NDArray[np.floating],
    covariance_matrix: npt.NDArray[np.floating],
) -> None:
    """
    Compare QAOA and exact solver portfolio solutions by computing and displaying financial metrics.

    Args:
        qaoa_solution: QAOA portfolio bitstring
        exact_solution: Exact solver portfolio bitstring
        returns: Expected returns vector for all assets
        covariance_matrix: Covariance matrix for all assets
    """
    # Compute financial metrics for both solutions
    qaoa_return, qaoa_risk, qaoa_sharpe = _compute_portfolio_metrics(
        returns, covariance_matrix, qaoa_solution
    )
    exact_return, exact_risk, exact_sharpe = _compute_portfolio_metrics(
        returns, covariance_matrix, exact_solution
    )

    # Count selected assets
    qaoa_n_assets = np.sum(qaoa_solution)
    exact_n_assets = np.sum(exact_solution)

    print("\nFinancial Metrics:")
    print("  QAOA Portfolio:")
    print(f"    Return: {qaoa_return:.6f}")
    print(f"    Risk (Variance): {qaoa_risk:.6f}")
    print(f"    Sharpe Ratio: {qaoa_sharpe:.6f}")
    print(f"    Number of Assets: {int(qaoa_n_assets)}")
    print("  ExactSolver Portfolio:")
    print(f"    Return: {exact_return:.6f}")
    print(f"    Risk (Variance): {exact_risk:.6f}")
    print(f"    Sharpe Ratio: {exact_sharpe:.6f}")
    print(f"    Number of Assets: {int(exact_n_assets)}")
    print("  Difference:")
    print(f"    Return Difference: {qaoa_return - exact_return:.6f}")
    print(f"    Risk Difference: {qaoa_risk - exact_risk:.6f}")
    print(f"    Sharpe Ratio Difference: {qaoa_sharpe - exact_sharpe:.6f}")

    # Check if solutions match
    solutions_match = np.array_equal(qaoa_solution, exact_solution)
    print(f"\nSolutions Match: {solutions_match}")
    if not solutions_match:
        diff_count = np.sum(qaoa_solution != exact_solution)
        print(
            f"  Number of differing positions: {diff_count} out of {len(qaoa_solution)}"
        )
        qaoa_selected = np.where(qaoa_solution == 1)[0]
        exact_selected = np.where(exact_solution == 1)[0]
        print(f"  QAOA selected assets ({len(qaoa_selected)}): {qaoa_selected}")
        print(f"  Exact selected assets ({len(exact_selected)}): {exact_selected}")


def solve_and_aggregate_partitions(
    qubo_matrices: list[npt.NDArray[np.floating]] | dict[int, npt.NDArray[np.floating]],
    partitions: npt.NDArray[np.integer],
) -> npt.NDArray[np.integer]:
    """
    Solve all partition QUBOs using ExactSolver and aggregate into a global bitstring.

    This function:
    1. Solves each partition's QUBO using dimod.ExactSolver
    2. Aggregates all partition solutions into a single global bitstring
    3. Maintains the original order of assets (mapped by partition assignments)

    Args:
        qubo_matrices: Either:
            - A list of QUBO matrices (one per partition, in order of unique partition IDs)
            - A dictionary mapping partition IDs to QUBO matrices
        partitions: Array mapping each global asset index to its partition ID.
                   Shape: (n_assets,)

    Returns:
        Global bitstring solution of shape (n_assets,) where:
        - 1 means the asset is included in the portfolio
        - 0 means the asset is excluded

    Example:
        >>> partitions = np.array([0, 0, 1, 1, 0])  # Assets 0,1,4 in partition 0; 2,3 in partition 1
        >>> qubo_list = [qubo_partition_0, qubo_partition_1]  # QUBOs for partitions 0 and 1
        >>> global_solution = solve_and_aggregate_partitions(qubo_list, partitions)
        >>> # Or with dict:
        >>> qubo_dict = {0: qubo_partition_0, 1: qubo_partition_1}
        >>> global_solution = solve_and_aggregate_partitions(qubo_dict, partitions)
    """
    unique_partitions = np.unique(partitions)
    n_partitions = len(unique_partitions)

    # Normalize input format: convert list to dict if needed
    if isinstance(qubo_matrices, list):
        if len(qubo_matrices) != n_partitions:
            raise ValueError(
                f"Number of QUBO matrices ({len(qubo_matrices)}) "
                f"does not match number of partitions ({n_partitions})"
            )
        # Assume list is ordered by sorted unique partition IDs
        qubo_dict = {
            partition_id: qubo_matrices[i]
            for i, partition_id in enumerate(sorted(unique_partitions))
        }
    else:
        qubo_dict = qubo_matrices

    # Solve each partition using ExactSolver
    partition_solutions = {}
    solver = dimod.ExactSolver()

    for partition_id in unique_partitions:
        if partition_id not in qubo_dict:
            raise ValueError(
                f"QUBO matrix missing for partition {partition_id}. "
                f"Available partitions: {sorted(qubo_dict.keys())}"
            )

        qubo_matrix = qubo_dict[partition_id]

        # Validate QUBO matrix
        if qubo_matrix.shape[0] != qubo_matrix.shape[1]:
            raise ValueError(
                f"Partition {partition_id}: QUBO matrix must be square, "
                f"got shape {qubo_matrix.shape}"
            )

        # Solve using ExactSolver
        bqm = dimod.BinaryQuadraticModel(qubo_matrix, "BINARY")
        sample_set = solver.sample(bqm)
        partition_solutions[partition_id] = _extract_dimod_solution(sample_set)

    # Aggregate all partition solutions into global bitstring
    return aggregate_partition_solutions(partition_solutions, partitions)
