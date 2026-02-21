# Partitioned QAOA for MaxCut using `divi`

This project demonstrates how to solve the **MaxCut** problem on large, community-structured graphs using the `divi` quantum programming framework.

It specifically showcases **Graph Partitioning**, a technique that splits a large graph (e.g., 50 qubits) into smaller, manageable sub-graphs using spectral clustering. This allows for the execution of Quantum Approximate Optimization Algorithm (QAOA) circuits on smaller quantum processors or simulators in parallel.

## Project Structure

To run this example, organize your files as follows:

```text
.
├── main.py           # The entry point (contains the execution logic)
├── utils.py          # Helper functions (graph generation, visualization, analysis)
└── README.md         # This file
```

## Prerequisites

```bash
pip install qoro-divi networkx matplotlib
```

## Usage

### 1. Setup

Ensure you have separated the provided code into main.py and utils.py:

- utils.py: Should contain generate_clustered_graph, show_graph, and analyze_results.
- main.py: Should contain the if __name__ == "__main__": block and imports.

### 2. Run

Run the main script to execute the local simulation:

```bash
python main.py
```

### 3. Expected Output

1. A Matplotlib window will open, visualizing the generated graph colors by cluster.
1. The terminal will log the progress of the partitioned QAOA job.
1. Finally, the script will compare the Quantum result against a Classical approximation:

```
Quantum Cut Size to Classical Cut Size Ratio = 0.98
```
(A ratio close to or > 1.0 indicates the quantum solution is competitive with or better than the classical One-Exchange approximation).

## Configuration

| Parameter            | Description                                                                                     |
|----------------------|-------------------------------------------------------------------------------------------------|
| n_qubits             | Total number of nodes in the graph (default: 50).                                               |
| n_clusters           | Number of dense communities to generate.                                                        |
| partitioning_config  | Controls how Divi splits the graph, using spectral clustering to minimize cuts between sub-circuits. |
| optimizer            | Uses MonteCarloOptimizer to find optimal QAOA parameters.                                       |

## Remote Execution (QoroService)

The code includes support for Qoro's remote backend. To use it:

1. Obtain an API key from [dash.qoroquantum.net](https://dash.qoroquantum.net).
1. Set the environment variable:
    ```bash
    export QORO_API_KEY="your_api_key_here"
    ```
1. Uncomment the QoroService lines at the bottom of main.py.

