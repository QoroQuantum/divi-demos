# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import copy
from functools import partial
from typing import Any

from divi.backends import CircuitRunner
from divi.qprog import QAOA
from divi.qprog.ensemble import ProgramEnsemble
from divi.qprog.optimizers import MonteCarloOptimizer, Optimizer
from divi.qprog.problems import BinaryOptimizationProblem
from divi.typing import QUBOProblemTypes


class QUBOBatch(ProgramEnsemble):
    """A minimal ProgramBatch that solves multiple QUBO problems in parallel using QAOA.

    Takes a collection of QUBO problems and solves each one independently using QAOA,
    executing them in parallel for efficiency. Optionally supports partitioning to
    reconstruct a full bitstring from partition solutions.

    Attributes:
        qubos (dict[str, QUBOProblemTypes] | list[QUBOProblemTypes]): The QUBO problems to solve.
            If a dict, keys are used as program identifiers. If a list, indices are used.
        partitions (np.ndarray | None): Optional array mapping each variable in the original
            problem to a partition index. If provided, aggregate_results will reconstruct
            a full bitstring by placing each partition's solution in the correct positions.
        solutions (dict): Dictionary mapping program IDs to their solutions after aggregation.
        full_solution (np.ndarray | None): The reconstructed full bitstring if partitions is provided.
    """

    def __init__(
        self,
        qubos: dict[str, QUBOProblemTypes] | list[QUBOProblemTypes],
        backend: CircuitRunner,
        n_layers: int = 1,
        optimizer: Optimizer | None = None,
        max_iterations: int = 10,
        partitions: np.ndarray | None = None,
        **kwargs,
    ):
        """Initialize a QUBOBatch instance.

        Args:
            qubos (dict[str, QUBOProblemTypes] | list[QUBOProblemTypes]): The QUBO problems to solve.
                If a dict, keys are used as program identifiers. If a list, indices are used.
            backend (CircuitRunner): Backend responsible for running quantum circuits.
            n_layers (int): Number of QAOA layers to use for each QUBO. Defaults to 1.
            optimizer (Optimizer, optional): Optimizer to use for QAOA.
                Defaults to MonteCarloOptimizer().
            max_iterations (int): Maximum number of optimization iterations. Defaults to 10.
            partitions (np.ndarray | None): Optional array mapping each variable in the original
                problem to a partition index. For example, partitions = [0, 0, 1, 1, 0] means
                variables 0,1,4 are in partition 0, and variables 2,3 are in partition 1.
                If provided with a dict of qubos, the dict keys should match the partition IDs.
                Defaults to None.
            **kwargs: Additional keyword arguments passed to the QAOA constructor.
        """
        super().__init__(backend=backend)

        # Normalize input to dict format
        if isinstance(qubos, list):
            self._qubos = {str(i): qubo for i, qubo in enumerate(qubos)}
        else:
            self._qubos = qubos

        self.max_iterations = max_iterations
        self.solutions = {}
        self.full_solution = None
        self.partitions = partitions

        # Store the optimizer prototype
        self.optimizer_prototype = (
            optimizer if optimizer is not None else MonteCarloOptimizer()
        )

        # Create a partial function for constructing QAOA instances
        self._constructor = partial(
            QAOA,
            max_iterations=self.max_iterations,
            backend=self.backend,
            n_layers=n_layers,
            **kwargs,
        )

    def create_programs(self):
        """Create a QAOA program for each QUBO problem.

        Each QUBO gets its own QAOA instance that will be executed in parallel.
        Program identifiers are taken from the dict keys (or generated indices).
        """
        super().create_programs()

        for prog_id, qubo in self._qubos.items():
            # Create a deep copy of the optimizer for each program to avoid shared state
            program_optimizer = copy.deepcopy(self.optimizer_prototype)

            self._programs[prog_id] = self._constructor(
                program_id=prog_id,
                problem=BinaryOptimizationProblem(qubo),
                optimizer=program_optimizer,
                progress_queue=self._queue,
            )

    def aggregate_results(self) -> dict[str, Any] | np.ndarray:
        """Aggregate solutions from all QUBO problems.

        Collects the solution for each QUBO problem and stores them in self.solutions.
        If partitions were provided, also reconstructs the full bitstring by placing
        each partition's solution in the correct positions.

        Returns:
            dict[str, Any] | np.ndarray: If partitions is None, returns a dictionary
                mapping program IDs to their solutions. If partitions is provided,
                returns the reconstructed full bitstring as a numpy array.

        Raises:
            RuntimeError: If programs haven't been run or if solutions haven't been computed.
        """
        super().aggregate_results()

        # Check that all programs have been run and have solutions
        if any(
            not hasattr(program, "solution") or len(program.solution) == 0
            for program in self._programs.values()
        ):
            raise RuntimeError(
                "Not all programs have solutions yet. Please call `run()` first."
            )

        # Collect solutions
        self.solutions = {
            prog_id: program.solution for prog_id, program in self._programs.items()
        }

        if self.partitions is not None:
            partitions = np.asarray(self.partitions)
            n_vars = len(partitions)
            self.full_solution = np.zeros(n_vars, dtype=np.int32)

            unique_partitions = np.unique(partitions)

            # Iterate through partitions (more efficient than iterating through variables)
            for partition_id in unique_partitions:
                # Get all global indices for this partition (vectorized operation)
                global_indices = np.flatnonzero(partitions == partition_id)

                # Get partition solution (try int key first, then str)
                partition_solution = self.solutions.get(partition_id)
                if partition_solution is None:
                    partition_solution = self.solutions.get(str(partition_id))

                if partition_solution is None:
                    raise KeyError(
                        f"Partition ID {partition_id} not found in solutions. "
                        f"Available keys: {list(self.solutions.keys())[:10]}..."
                    )

                # Validate length
                if len(partition_solution) != len(global_indices):
                    raise ValueError(
                        f"Partition {partition_id}: Solution length ({len(partition_solution)}) "
                        f"does not match partition size ({len(global_indices)})"
                    )

                self.full_solution[global_indices] = partition_solution

            return self.full_solution

        return self.solutions
