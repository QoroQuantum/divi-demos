import math
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from itertools import groupby, repeat

import numpy as np
import pennylane as qml
from divi.qprog import VQE
from docplex.mp.model import Model


def black_box_optimizer(
    combination_ids: tuple[int],
    target_matrix: np.ndarray,
    all_permutation_matrices_flat: np.ndarray,
    scale: int,
) -> tuple[int, float, list[float] | None]:
    """
    Performs classical optimization for a given combination of permutations.
    Returns a cost, an error value, and the final decomposition weights.
    """
    n = target_matrix.shape[0]
    k = len(combination_ids)

    if not combination_ids or max(combination_ids) >= len(
        all_permutation_matrices_flat
    ):
        return k + 1, float("inf"), None

    # Use the pre-selected and pre-flattened permutation matrices
    selected_perms_flat = all_permutation_matrices_flat[list(combination_ids)]

    try:
        # Step 1: Integer Optimization
        integer_model = Model(name="integer_approximation")
        integer_model.parameters.threads = 2

        u = integer_model.integer_var_list(k, name="u")
        integer_model.add_constraint(integer_model.sum(u) == scale)
        for i in range(k):
            integer_model.add_constraint(u[i] >= 0)
            integer_model.add_constraint(u[i] <= scale)

        reconstructed_matrix_flat = u @ selected_perms_flat
        target_matrix_flat = target_matrix.flatten()
        error_expr = integer_model.sum_squares(
            target_matrix_flat[i] - reconstructed_matrix_flat[i] for i in range(n * n)
        )
        integer_model.minimize(error_expr)
        integer_solution = integer_model.solve()
        if not integer_solution:
            return k + 1, float("inf"), None
        min_error_found = integer_solution.get_objective_value()

        # Step 2: Sparsification
        continuous_model = Model(name="sparsification")
        continuous_model.parameters.threads = 2

        c = continuous_model.continuous_var_list(k, name="c")
        y = continuous_model.binary_var_list(k, name="y")
        continuous_model.add_constraint(continuous_model.sum(c) == 1)
        for i in range(k):
            continuous_model.add_constraint(c[i] >= 0)
            continuous_model.add_constraint(c[i] <= y[i])

        reconstructed_matrix_flat_c = c @ selected_perms_flat
        target_matrix_unscaled_flat = target_matrix.flatten() / scale
        error_expr_c = continuous_model.sum_squares(
            target_matrix_unscaled_flat[i] - reconstructed_matrix_flat_c[i]
            for i in range(n * n)
        )
        continuous_model.add_constraint(error_expr_c <= min_error_found / (scale**2) + 1e-9)
        continuous_model.minimize(continuous_model.sum(y))
        continuous_solution = continuous_model.solve()
        if not continuous_solution:
            return k + 1, float("inf"), None
    except Exception as e:
        if "CPLEX" in str(e) or "runtime" in str(e).lower():
            raise RuntimeError(
                "CPLEX runtime not found. Please install it with:  pip install cplex"
            ) from e
        raise

    continuous_weights = continuous_solution.get_value_list(c)

    final_error = np.linalg.norm(
        target_matrix_unscaled_flat
        - np.array(continuous_weights) @ selected_perms_flat,
        ord=2,
    )
    return (
        int(round(continuous_solution.get_objective_value())),
        final_error,
        continuous_weights,
    )


def integer_to_combination(target_id: int, k: int) -> list[int]:
    """Decodes a single integer back into a combination of k unique integer IDs."""
    combination = []
    for i in range(k, 0, -1):
        x = i - 1
        while math.comb(x, i) <= target_id:
            x += 1
        x -= 1
        combination.append(x)
        target_id -= math.comb(x, i)
    return sorted(combination)


def combination_to_integer(combination: list[int], k: int) -> int:
    """Encodes a combination of k unique integer IDs into a single integer."""
    combination = sorted(combination)
    target_id = 0
    for i in range(k, 0, -1):
        x = combination[k - i]
        target_id += math.comb(x, i)
    return target_id


class BirkhoffDecomposition(VQE):
    """
    A VQE-based program to solve the Birkhoff Decomposition problem.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        scale: int,
        k: int,
        all_perms_matrix: np.ndarray,
        n: int,
        **kwargs,
    ):
        self.matrix = matrix
        if self.matrix.shape != (n, n):
            self.matrix = self.matrix.reshape((n, n))

        self.all_perm_matrices_flat = all_perms_matrix.reshape(
            all_perms_matrix.shape[0], -1
        )
        self.scale = scale
        self.k = k

        self.penalty_value = np.linalg.norm(self.matrix.flatten() / self.scale, ord=2)
        self.n_qubits = np.ceil(np.log2(math.comb(math.factorial(n), k))).astype(int)

        hamiltonian = qml.Hamiltonian(
            [1.0] * self.n_qubits, [qml.PauliZ(i) for i in range(self.n_qubits)]
        )

        self.approx_errors = {}
        self.final_measurement_outcomes = {}

        super().__init__(hamiltonian=hamiltonian, **kwargs)

    @lru_cache(maxsize=None)
    def _cached_black_box(self, combination_ids: tuple[int]):
        """
        A thread-safe, cached wrapper for the expensive black_box_optimizer.
        """
        return black_box_optimizer(
            combination_ids,
            self.matrix,
            self.all_perm_matrices_flat,
            self.scale,
        )

    def _process_one_combination(
        self, comb: list[int], count: int, total_shots: int
    ) -> tuple[float, float, list[int]]:
        """Worker function for parallel execution."""
        comb_tuple = tuple(comb)
        _, black_box_error, _ = self._cached_black_box(comb_tuple)

        loss_value = self.penalty_value
        if not np.isinf(black_box_error):
            loss_value = black_box_error

        weighted_loss = (count / total_shots) * loss_value

        return weighted_loss, black_box_error, comb

    def _calculate_and_store_approx_error(self, all_results: list[tuple]):
        """
        Finds the best combination from a set of results, calculates its
        true L2 approximation error, and stores it.
        """
        current_iter = self.current_iteration
        if not all_results:
            self.approx_errors[current_iter] = self.penalty_value
            return

        # Unzip results to find the combination with the minimum raw error
        _, raw_errors, combinations = zip(*all_results)
        min_error_index = np.argmin(raw_errors)
        best_combination = combinations[min_error_index]

        # Get the weights for this single best combination from the cache
        _, _, best_weights = self._cached_black_box(tuple(best_combination))

        if best_weights:
            # Reconstruct the matrix from the best combination's weights
            reconstructed_matrix = sum(
                weight * self.all_perm_matrices_flat[perm_idx]
                for perm_idx, weight in zip(best_combination, best_weights)
            )

            # Calculate the L2 error against the unscaled target matrix
            target_unscaled = self.matrix / self.scale
            l2_error = np.linalg.norm(
                target_unscaled.flatten() - reconstructed_matrix, ord=2
            )
            self.approx_errors[current_iter] = l2_error
        else:
            # If the best combination was somehow invalid, record a penalty
            self.approx_errors[current_iter] = self.penalty_value

    def _post_process_results(
        self, results: dict[str, dict[str, int]]
    ) -> dict[int, float]:
        losses = {}
        get_param_id = lambda item: item[0].param_id if hasattr(item[0], 'param_id') else int(str(item[0]).split("_")[0])

        for p_idx, param_group in groupby(results.items(), key=get_param_id):
            try:
                shots_dict = next(param_group)[1]
            except StopIteration:
                continue

            total_shots = sum(shots_dict.values())
            if total_shots == 0:
                losses[p_idx] = float("inf")
                continue

            valid_combs, valid_shots = [], []
            measured_integers = map(partial(int, base=2), shots_dict.keys())

            for measured_int, shots in zip(measured_integers, shots_dict.values()):
                try:
                    valid_combs.append(integer_to_combination(measured_int, self.k))
                    valid_shots.append(shots)
                except (ValueError, IndexError):
                    continue

            if not valid_combs:
                losses[p_idx] = self.penalty_value + self.loss_constant
                continue

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results_iterator = executor.map(
                    self._process_one_combination,
                    valid_combs,
                    valid_shots,
                    repeat(total_shots),
                    chunksize=1,
                )

            all_results = list(results_iterator)

            # --- Call the new helper function ---
            self._calculate_and_store_approx_error(all_results)

            if not all_results:
                losses[p_idx] = self.penalty_value + self.loss_constant
                continue

            # The loss for the VQE optimizer is still calculated here
            weighted_losses, _, _ = zip(*all_results)
            expected_error = sum(weighted_losses)
            losses[p_idx] = expected_error + self.loss_constant

        return losses

    def find_final_decomposition(self) -> tuple[list[int] | None, list[float] | None]:
        print("\n--- Performing Final Computation ---")
        self.reporter.info(message="Running final circuit")

        self._curr_params = np.atleast_2d(self.final_params)
        self._run_solution_measurement()
        shots_dict = next(iter(self._best_probs.values()))

        self.final_measurement_outcomes = shots_dict

        sorted_outcomes = sorted(
            shots_dict.items(), key=lambda item: item[1], reverse=True
        )

        best_solution = {
            "combination": None,
            "weights": None,
            "error": float("inf"),
        }

        checked_count = 0
        for bitstring, _ in sorted_outcomes:
            if checked_count >= 10:
                break  # Stop after checking m valid outcomes

            try:
                measured_integer = int(bitstring, 2)
                combination = integer_to_combination(measured_integer, self.k)

                # Use the cached optimizer to get the results
                comb_tuple = tuple(combination)
                _, error, weights = self._cached_black_box(comb_tuple)

                if weights is not None and error < best_solution["error"]:
                    # If this is the best solution found so far, store it
                    best_solution["combination"] = combination
                    best_solution["weights"] = weights
                    best_solution["error"] = error

                checked_count += 1

            except (ValueError, IndexError):
                # This bitstring doesn't map to a valid combination, so we skip it.
                continue

        if best_solution["combination"]:
            print(
                f"Found best solution with error {best_solution['error']:.6f} from combination: {best_solution['combination']}"
            )
            return best_solution["combination"], best_solution["weights"]

        print(
            "Could not find a valid decomposition from any of the top measurement outcomes."
        )
        return None, None
