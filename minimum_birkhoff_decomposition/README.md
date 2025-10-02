# VQE-based Minimum Birkhoff Decomposition

This project implements a Variational Quantum Eigensolver (VQE) to find the Birkhoff decomposition of doubly stochastic matrices.

A key strength of the `divi` library is its modular design, which allows complex new problems like the Birkhoff Decomposition to be implemented with minimal, targeted changes. The `QuantumProgram` and `VQE` classes (see source files `quantum_program.py` and `_vqe.py`) provide a robust framework that handles the most difficult parts of a hybrid quantum-classical algorithm.

This example demonstrates how a new, sophisticated application was built by inheriting from the `VQE` class and only making two key adaptations:

1. **Overriding the Post-Processing Logic:** The core VQE optimization loop, including parameter management, gradient computation, and circuit execution, was inherited without any changes. The only required modification was to override the `_post_process_results` method to connect the quantum measurement outcomes to the problem-specific `black_box_optimizer`.

2. **Implementing the Classical Routine:** All the complex machinery for communicating with quantum backends, managing jobs, and orchestrating the optimization was reused. The only new code needed was the classical `black_box_optimizer` itself, which contains the unique logic for the Birkhoff problem. This part of the algorithm is accelerated using multi-threading and caching.

This powerful separation of concerns allows researchers and developers to focus on the novel aspects of their problem without having to reinvent the underlying algorithmic infrastructure, dramatically accelerating development and experimentation.

The problem instances and the general methodology are based on the challenges presented in the Quantum Optimization Benchmarking Library.

## Requirements

The core logic of this project depends on the `divi` library from Qoro Quantum.

* `qoro-divi==0.3.4`

You can install all necessary dependencies using pip:

```bash
pip install qoro-divi==0.3.4
```

## Files

* **`birkhoff_vqe.py`**: A Python module containing the core business logic. It defines the `BirkhoffDecomposition` class which encapsulates the VQE algorithm and the classical `black_box_optimizer`.
* **`main.py`**: The main executable script used to run experiments. It handles data loading, command-line argument parsing, and the visualization of results.

## How to Run

The `main.py` script is configured via command-line arguments.

### **Usage**

```bash
python main.py [-h] [-n DIM] [-k COMB] [-m {sparse,dense}] [-inst [1-10]] [-it ITERATIONS] [-opt {Scipy,MonteCarlo}]
```

### **Arguments**

| Flag | Name | Description | Default |
|---|---|---|---|
| `-n` | `--dim` | Matrix dimension. | `4` |
| `-k` | `--comb` | Number of permutations in the combination. | `2` |
| `-m`| `--matrix_type` | Matrix example type (`sparse` or `dense`). | `sparse` |
| `-inst` | `--instance` | The problem instance to load (1-10). | `1` |
| `-it` | `--iterations`| Max VQE optimizer iterations. | `10` |
| `-opt` | `--optimizer` | The VQE optimizer to use. | `Cobyla`|

### **Examples**

* **Run with default parameters:**

    ```bash
    python main.py
    ```

* **Run a specific experiment:**

    ```bash
    python main.py -n 4 -k 2 -inst 5 -it 20
    ```

* **View all options:**

    ```bash
    python main.py --help
    ```

---

## Citation

This work is based on problems from the following paper:
> G. S. Barron, et al., "Quantum Optimization Benchmarking Library: The Intractable Decathlon," arXiv:2504.03832 [quant-ph], (2025). <https://arxiv.org/abs/2504.03832>
