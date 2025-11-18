import argparse
import json
import os

import numpy as np
import pennylane as qml
from birkhoff_vqe import BirkhoffDecomposition, combination_to_integer
from divi.backends import ParallelSimulator
from divi.qprog.algorithms._ansatze import GenericLayerAnsatz
from divi.qprog.optimizers import MonteCarloOptimizer, ScipyMethod, ScipyOptimizer


def parse_arguments():
    """Parses command-line arguments for the experiment."""
    parser = argparse.ArgumentParser(description="Run the Birkhoff Decomposition VQE.")
    parser.add_argument(
        "-n",
        "--dim",
        type=int,
        default=3,
        choices=range(3, 5),
        metavar="[3-4]",
        help="Matrix dimension (default: 3).",
    )
    parser.add_argument(
        "-k",
        "--comb",
        type=int,
        default=2,
        help="Number of permutations in the combination (default: 2).",
    )
    parser.add_argument(
        "-m",
        "--matrix_type",
        type=str,
        choices=["sparse", "dense"],
        default="sparse",
        help="Type of matrix example to use (default: sparse).",
    )
    parser.add_argument(
        "-inst",
        "--instance",
        type=int,
        default=1,
        choices=range(1, 11),
        metavar="[1-10]",
        help="Problem instance to load (default: 1).",
    )
    parser.add_argument(
        "-it",
        "--iterations",
        type=int,
        default=10,
        help="Max iterations for the VQE optimizer (default: 10).",
    )
    parser.add_argument(
        "-opt",
        "--optimizer",
        type=str,
        choices=["Cobyla", "MonteCarlo"],
        default="Cobyla",
        help="Optimizer to use for the VQE (default: Cobyla).",
    )
    return parser.parse_args()


class BColors:
    """ANSI escape sequences for coloring terminal text."""

    OKGREEN = "\033[92m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def parse_instance(
    json_data: dict, instance_idx: str
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Parses a specific instance from the problem data."""
    instance_data = json_data[instance_idx]
    n = instance_data["n"]
    D = np.array(instance_data["scaled_doubly_stochastic_matrix"]).reshape((n, n))
    perms = np.eye(n, dtype=int)[
        np.array(instance_data["permutations"]).reshape(-1, n) - 1
    ]
    return D, instance_data["scale"], perms, np.array(instance_data["weights"])


def print_matrix_with_highlights(title: str, matrix: np.ndarray, highlights: dict):
    """Prints a matrix with specified entries highlighted with color."""
    print(f"\n{BColors.BOLD}{title}{BColors.ENDC}")
    for r, row_vals in enumerate(matrix):
        line = []
        for c, val in enumerate(row_vals):
            val_str = f"{val: .4f}"
            if (r, c) in highlights:
                # Apply the color specified in the highlights dictionary
                color = highlights.get((r, c))
                val_str = f"{color}{val_str}{BColors.ENDC}"
            line.append(val_str)
        print("  [" + " ".join(line) + "]")


def present_final_results(
    original_matrix_scaled: np.ndarray,
    scale: int,
    vqe_combination: list[int],
    vqe_weights: list[float],
    all_perms_matrix: np.ndarray,
    solution_perms: np.ndarray,
    solution_weights: np.ndarray,
):
    """
    Presents a consolidated analysis of the final VQE results compared
    to the original matrix and the known solution's decomposition.
    """
    print("\n--- Final Birkhoff Decomposition Analysis ---")

    # --- 1. VQE Decomposition Details ---
    print(f"{BColors.BOLD}Decomposition (VQE Found):{BColors.ENDC}")
    for perm_idx, weight in zip(vqe_combination, vqe_weights):
        if weight > 1e-6:
            print(f"  - Permutation Index {perm_idx}: Weight = {weight:.6f}")

    # --- 2. Known Solution Decomposition ---
    print(f"\n{BColors.BOLD}Decomposition (Known Solution):{BColors.ENDC}")
    # We need to find the indices of the solution permutations
    # This is a bit complex, so we'll do a quick search.
    solution_indices = []
    for sol_perm_matrix in solution_perms:
        # Find which index in all_perms_matrix matches the solution matrix
        for idx, perm_matrix in enumerate(all_perms_matrix):
            if np.array_equal(sol_perm_matrix, perm_matrix):
                solution_indices.append(idx)
                break

    for perm_idx, weight in zip(solution_indices, solution_weights):
        if weight > 1e-6:
            print(f"  - Permutation Index {perm_idx}: Weight = {weight:.6f}")

    # --- 3. Matrix Reconstructions and Error ---
    vqe_reconstructed = sum(
        weight * all_perms_matrix[perm_idx]
        for perm_idx, weight in zip(vqe_combination, vqe_weights)
        if weight > 1e-6
    )
    original_unscaled = original_matrix_scaled / scale

    print_matrix_with_highlights("Original Matrix (Unscaled)", original_unscaled, {})
    print_matrix_with_highlights("VQE Reconstructed Matrix", vqe_reconstructed, {})

    error_matrix = original_unscaled - vqe_reconstructed
    abs_error_matrix = np.abs(error_matrix)
    max_error_pos = np.unravel_index(
        np.argmax(abs_error_matrix), abs_error_matrix.shape
    )
    zero_error_positions = np.argwhere(abs_error_matrix < 1e-9)

    error_highlights = {tuple(pos): BColors.OKGREEN for pos in zero_error_positions}
    error_highlights[max_error_pos] = BColors.FAIL

    print_matrix_with_highlights(
        "Error Matrix (Original - VQE Reconstructed)", error_matrix, error_highlights
    )

    # --- 4. Final Comparison Metric ---
    final_error_norm = np.linalg.norm(original_unscaled - vqe_reconstructed, ord=2)
    print(
        f"\n{BColors.BOLD}Final Error (L2 Norm) between Original and VQE Reconstructed: {final_error_norm:.6f}{BColors.ENDC}"
    )
    print("---------------------------------------------")


def main(args):
    """Main execution block to set up and run the VQE experiment."""
    # --- Problem Setup ---
    N_DIM = args.dim
    K_COMBINATIONS = args.comb
    INSTANCE_ID = str(args.instance)
    MAX_ITERATIONS = args.iterations
    MATRIX_TYPE = args.matrix_type
    OPTIMIZER_TYPE = args.optimizer

    print(f"--- Running Birkhoff Decomposition for n={N_DIM}, k={K_COMBINATIONS} ---")
    print(
        f"Instance: {INSTANCE_ID}, Matrix: {MATRIX_TYPE}, Optimizer: {OPTIMIZER_TYPE}, Iterations: {MAX_ITERATIONS}"
    )

    dirname = os.path.dirname(__file__)
    mat_file = os.path.join(dirname, f"qbench_0{N_DIM}_{MATRIX_TYPE}.json")
    perm_file = os.path.join(dirname, f"p{N_DIM}.dat")

    with open(mat_file) as f:
        problem_data = json.load(f)
    permutations = np.loadtxt(perm_file, dtype=int)

    all_permutation_matrices = np.eye(N_DIM, dtype=int)[permutations - 1]

    matrix, scale, sol_perms, sol_weights = parse_instance(problem_data, INSTANCE_ID)

    if OPTIMIZER_TYPE == "Cobyla":
        optimizer = ScipyOptimizer(ScipyMethod.COBYLA)
    elif OPTIMIZER_TYPE == "MonteCarlo":
        optimizer = MonteCarloOptimizer(n_param_sets=20, n_best_sets=5)
    else:
        raise ValueError(f"Unsupported optimizer type: {OPTIMIZER_TYPE}")

    qpu_backend = ParallelSimulator(shots=10000)

    ansatz = GenericLayerAnsatz([qml.RY], entangler=qml.CZ, entangling_layout="brick")

    # --- Instantiate and Run the Custom VQE Program ---
    birkhoff_vqe = BirkhoffDecomposition(
        matrix=matrix,
        scale=scale,
        n=N_DIM,
        k=K_COMBINATIONS,
        all_perms_matrix=all_permutation_matrices,
        backend=qpu_backend,
        optimizer=optimizer,
        max_iterations=MAX_ITERATIONS,
        ansatz=ansatz,
        n_layers=3,
    )

    print("Starting VQE optimization...")
    birkhoff_vqe.run()

    # --- Get Final Mathematical Decomposition ---
    combo, weights = birkhoff_vqe.find_final_decomposition()

    if combo and weights:
        # --- NEW: Calculate and print the solution's probability ---
        total_shots = sum(birkhoff_vqe.final_measurement_outcomes.values())

        # Convert the solution combination back to its integer and bitstring representation
        solution_integer = combination_to_integer(combo, birkhoff_vqe.k)
        solution_bitstring = format(solution_integer, f"0{birkhoff_vqe.n_qubits}b")

        # Find the number of shots for this specific bitstring
        solution_shots = birkhoff_vqe.final_measurement_outcomes.get(
            solution_bitstring, 0
        )

        if total_shots > 0:
            probability = (solution_shots / total_shots) * 100
            print(f"\n{BColors.BOLD}Probability of Best Solution:{BColors.ENDC}")
            print(
                f"  - Bitstring '{solution_bitstring}' was measured {solution_shots} times out of {total_shots} ({probability:.2f}%)"
            )

        present_final_results(
            original_matrix_scaled=birkhoff_vqe.matrix,
            scale=birkhoff_vqe.scale,
            vqe_combination=combo,
            vqe_weights=weights,
            all_perms_matrix=all_permutation_matrices,
            solution_perms=sol_perms,
            solution_weights=sol_weights / scale,
        )


if __name__ == "__main__":
    cli_args = parse_arguments()
    main(cli_args)
