# =============================================================================
#  Economic Load Dispatch with Prohibited Operating Zones via PCE-VQE
# =============================================================================
#
#  Solves the Economic Load Dispatch problem for a 3-generator microgrid
#  using the Divi quantum SDK.
#
#  Scroll to the bottom to see the main flow — it reads like plain English!
#
# =============================================================================

import dimod
import numpy as np
import pennylane as qml

from divi.backends import ParallelSimulator
from divi.qprog import PCE, GenericLayerAnsatz
from divi.qprog.optimizers import PymooMethod, PymooOptimizer
from divi.typing import qubo_to_matrix


# ─────────────────────────────────────────────────────────────────────
#  STEP 1 — Define the generators
# ─────────────────────────────────────────────────────────────────────

def define_generators():
    """Define the 3-generator microgrid.

    Each generator has:
      - a, b, c:          fuel cost curve  Cost = a + b·P + c·P²
      - P_min, P_max:     operating range in MW
      - poz_low, poz_high: prohibited operating zone (vibration band)

    Returns a list of generator dicts.
    """
    return [
        {
            "name": "Gen 1", "a": 20, "b": 2.0, "c": 0.010,
            "P_min": 40, "P_max": 115,
            "poz_low": 60, "poz_high": 75,
        },
        {
            "name": "Gen 2", "a": 15, "b": 1.5, "c": 0.020,
            "P_min": 20, "P_max": 95,
            "poz_low": 40, "poz_high": 55,
        },
        {
            "name": "Gen 3", "a": 25, "b": 1.8, "c": 0.015,
            "P_min": 30, "P_max": 105,
            "poz_low": 50, "poz_high": 65,
        },
    ]


# ─────────────────────────────────────────────────────────────────────
#  STEP 2 — Build the optimisation problem (QUBO)
# ─────────────────────────────────────────────────────────────────────

STEP_MW = 5           # power resolution per qubit level
N_QUBITS_PER_GEN = 4  # 2^4 = 16 discrete levels per generator
BIT_WEIGHTS = [2**b for b in range(N_QUBITS_PER_GEN)]  # [1, 2, 4, 8]


def fuel_cost(gen, power):
    """Compute fuel cost ($) for a generator at a given power (MW)."""
    return gen["a"] + gen["b"] * power + gen["c"] * power ** 2


def _qubit_name(gen_idx, bit_idx):
    """Variable name for qubit `bit_idx` of generator `gen_idx`."""
    return f"q_{gen_idx}_{bit_idx}"


def decode_power(gen_idx, generators, bit_values):
    """Convert binary qubit values back to MW for one generator."""
    integer_val = sum(
        BIT_WEIGHTS[b] * bit_values[_qubit_name(gen_idx, b)]
        for b in range(N_QUBITS_PER_GEN)
    )
    return generators[gen_idx]["P_min"] + STEP_MW * integer_val


def build_qubo(generators, demand, penalty_lambda=2000, poz_mu=5000):
    """Build the Binary Quadratic Model (BQM) for the ELD problem.

    The BQM encodes three things as a single energy function:
      1. Fuel cost    — the objective we want to minimise
      2. Demand penalty — forces total generation to equal demand
      3. POZ penalty  — forbids generators from operating in vibration bands

    Args:
        generators:     list of generator dicts from define_generators()
        demand:         target load in MW
        penalty_lambda: weight for the demand constraint
        poz_mu:         weight for prohibited operating zones

    Returns:
        bqm:         the dimod BinaryQuadraticModel
        var_names:   ordered list of qubit variable names
    """
    bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
    var_names = []

    for g in range(len(generators)):
        for b in range(N_QUBITS_PER_GEN):
            var_names.append(_qubit_name(g, b))

    # ── 1. Fuel cost objective ──
    cost_offset = 0.0
    for g, gen in enumerate(generators):
        a, b_coeff, c = gen["a"], gen["b"], gen["c"]
        p_min = gen["P_min"]
        cost_offset += a + b_coeff * p_min + c * p_min ** 2

        for k in range(N_QUBITS_PER_GEN):
            w_k = STEP_MW * BIT_WEIGHTS[k]
            var_k = _qubit_name(g, k)
            bqm.add_linear(var_k, b_coeff * w_k + c * (2 * p_min * w_k + w_k ** 2))
            for l in range(k + 1, N_QUBITS_PER_GEN):
                w_l = STEP_MW * BIT_WEIGHTS[l]
                bqm.add_quadratic(var_k, _qubit_name(g, l), c * 2 * w_k * w_l)
    bqm.offset += cost_offset

    # ── 2. Demand constraint:  λ · (P1 + P2 + P3 − demand)² ──
    d_const = sum(gen["P_min"] for gen in generators) - demand
    bqm.offset += penalty_lambda * d_const ** 2

    d_terms = []
    for g in range(len(generators)):
        for k in range(N_QUBITS_PER_GEN):
            w = STEP_MW * BIT_WEIGHTS[k]
            d_terms.append((_qubit_name(g, k), w))

    for var_i, w_i in d_terms:
        bqm.add_linear(var_i, penalty_lambda * (2 * d_const * w_i + w_i ** 2))
    for i in range(len(d_terms)):
        vi, wi = d_terms[i]
        for j in range(i + 1, len(d_terms)):
            vj, wj = d_terms[j]
            bqm.add_quadratic(vi, vj, penalty_lambda * 2 * wi * wj)

    # ── 3. POZ penalty:  μ · (1 − q_msb) · q_2nd  per generator ──
    for g in range(len(generators)):
        q_msb = _qubit_name(g, 3)
        q_2nd = _qubit_name(g, 2)
        bqm.add_linear(q_2nd, poz_mu)
        bqm.add_quadratic(q_msb, q_2nd, -poz_mu)

    return bqm, var_names


# ─────────────────────────────────────────────────────────────────────
#  STEP 3 — Find the best classical solution (brute force)
# ─────────────────────────────────────────────────────────────────────

def classical_brute_force(generators, demand, bqm):
    """Enumerate all 4,096 configurations and return the cheapest valid one.

    Returns:
        (P1, P2, P3, cost) for the best valid dispatch.
    """
    best = None

    for i1 in range(16):
        p1 = generators[0]["P_min"] + STEP_MW * i1
        poz1 = generators[0]["poz_low"] <= p1 <= generators[0]["poz_high"]
        for i2 in range(16):
            p2 = generators[1]["P_min"] + STEP_MW * i2
            poz2 = generators[1]["poz_low"] <= p2 <= generators[1]["poz_high"]
            for i3 in range(16):
                p3 = generators[2]["P_min"] + STEP_MW * i3
                poz3 = generators[2]["poz_low"] <= p3 <= generators[2]["poz_high"]

                if p1 + p2 + p3 != demand:
                    continue
                if poz1 or poz2 or poz3:
                    continue

                cost = sum(fuel_cost(generators[g], p)
                           for g, p in enumerate([p1, p2, p3]))
                if best is None or cost < best[3]:
                    best = (p1, p2, p3, cost)

    return best


# ─────────────────────────────────────────────────────────────────────
#  STEP 4 — Solve with quantum computing (PCE-VQE)
# ─────────────────────────────────────────────────────────────────────

def solve_with_pce(bqm, n_layers=3, max_iterations=20, alpha=3.0,
                   population_size=30, shots=10000):
    """Run the PCE-VQE quantum solver on the BQM.

    PCE (Pauli Correlation Encoding) compresses 12 QUBO variables into
    just 5 qubits using polynomial encoding — far fewer than the 12
    qubits QAOA would need.

    Args:
        bqm:              the Binary Quadratic Model to solve
        n_layers:         depth of the variational quantum circuit
        max_iterations:   number of Differential Evolution generations
        alpha:            binary activation hardness (higher = sharper)
        population_size:  DE population per generation
        shots:            measurement samples per circuit evaluation

    Returns:
        pce_solver:  the solved PCE object (access .solution, .get_top_solutions)
    """
    qubo_mat = qubo_to_matrix(bqm)

    ansatz = GenericLayerAnsatz(
        gate_sequence=[qml.RY, qml.RZ],
        entangler=qml.CNOT,
        entangling_layout="all-to-all",
    )

    pce_solver = PCE(
        qubo_matrix=qubo_mat,
        ansatz=ansatz,
        n_layers=n_layers,
        encoding_type="poly",
        optimizer=PymooOptimizer(method=PymooMethod.DE,
                                 population_size=population_size),
        max_iterations=max_iterations,
        alpha=alpha,
        backend=ParallelSimulator(shots=shots),
    )

    print(f"\n   PCE qubits: {pce_solver.n_qubits}  "
          f"(poly encoding of {len(bqm.variables)} variables)")
    pce_solver.run()
    return pce_solver


# ─────────────────────────────────────────────────────────────────────
#  STEP 5 — Repair quantum solutions to make them feasible
# ─────────────────────────────────────────────────────────────────────

def repair_solution(powers, generators, demand):
    """Fix a near-feasible quantum solution so it meets all constraints.

    Stage 1 — Snap any generator inside a Prohibited Operating Zone
              to the nearest allowed power level.
    Stage 2 — Greedily adjust generators (cheapest first) until total
              generation exactly matches demand.

    Args:
        powers:     list of MW values [P1, P2, P3]
        generators: list of generator dicts
        demand:     target load in MW

    Returns:
        Repaired [P1, P2, P3] list, or None if repair is impossible.
    """
    ps = list(powers)

    # Stage 1: fix POZ violations
    for g, gen in enumerate(generators):
        if gen["poz_low"] <= ps[g] <= gen["poz_high"]:
            # Find all allowed power levels for this generator
            allowed = [
                gen["P_min"] + STEP_MW * idx
                for idx in range(2 ** N_QUBITS_PER_GEN)
                if not (gen["poz_low"]
                        <= gen["P_min"] + STEP_MW * idx
                        <= gen["poz_high"])
            ]
            ps[g] = min(allowed, key=lambda lv: abs(lv - ps[g]))

    # Stage 2: fix demand mismatch
    for _ in range(50):
        gap = demand - sum(ps)
        if gap == 0:
            break

        step = STEP_MW if gap > 0 else -STEP_MW
        best_g, best_cost = None, float("inf")

        for g, gen in enumerate(generators):
            new_p = ps[g] + step
            if new_p < gen["P_min"] or new_p > gen["P_max"]:
                continue
            if gen["poz_low"] <= new_p <= gen["poz_high"]:
                continue
            marginal = abs(fuel_cost(gen, new_p) - fuel_cost(gen, ps[g]))
            if marginal < best_cost:
                best_cost = marginal
                best_g = g

        if best_g is None:
            return None
        ps[best_g] += step

    return ps if sum(ps) == demand else None


def find_best_repaired_solution(pce_solver, bqm, generators, demand, top_n=20):
    """Scan the top quantum candidates and return the best repaired dispatch.

    For each candidate in the quantum probability distribution:
      1. Decode the bitstring to MW values
      2. Repair any constraint violations
      3. Keep the cheapest valid result

    Returns:
        (powers, cost, probability) or None
    """
    top_solutions = pce_solver.get_top_solutions(n=top_n, include_decoded=True)
    best = None

    print("\n   Top quantum candidates:")
    for i, sol in enumerate(top_solutions, 1):
        if sol.decoded is None:
            continue

        sample = {var: int(val) for var, val in zip(bqm.variables, sol.decoded)}
        ps = [decode_power(g, generators, sample) for g in range(len(generators))]
        cost = sum(fuel_cost(generators[g], ps[g]) for g in range(len(generators)))
        tot = sum(ps)
        valid = (
            tot == demand
            and all(
                not (generators[g]["poz_low"] <= ps[g] <= generators[g]["poz_high"])
                for g in range(len(generators))
            )
        )
        print(f"     {i:2d}. P=[{ps[0]:.0f},{ps[1]:.0f},{ps[2]:.0f}]  "
              f"Tot={tot:.0f}  Cost={cost:.0f}$  "
              f"Prob={sol.prob:.2%}  {'✅' if valid else '❌'}")

        repaired = repair_solution(ps, generators, demand)
        if repaired is not None:
            rep_cost = sum(fuel_cost(generators[g], repaired[g])
                          for g in range(len(generators)))
            if best is None or rep_cost < best[1]:
                best = (repaired, rep_cost, sol.prob)

    return best


# ─────────────────────────────────────────────────────────────────────
#  STEP 6 — Compare quantum vs classical results
# ─────────────────────────────────────────────────────────────────────

def print_comparison(quantum_result, classical_result):
    """Print a side-by-side comparison of quantum and classical solutions."""
    q_powers, q_cost, q_prob = quantum_result
    c_p1, c_p2, c_p3, c_cost = classical_result

    print("\n" + "=" * 70)
    print("  🔬 Quantum vs Classical Comparison")
    print("=" * 70)
    print(f"\n   Classical optimum:  P1={c_p1}, P2={c_p2}, P3={c_p3} MW"
          f"  → Cost = {c_cost:.1f} $")
    print(f"   PCE-VQE result:     P1={q_powers[0]:.0f}, P2={q_powers[1]:.0f},"
          f" P3={q_powers[2]:.0f} MW  → Cost = {q_cost:.1f} $")

    if abs(q_cost - c_cost) < 0.1:
        print("\n   🎉 PCE-VQE found the global optimum!")
    else:
        print(f"\n   ⚡ Gap from optimum: {q_cost - c_cost:.1f} $")
        print("      Try increasing n_layers or max_iterations.")

    print("\n" + "=" * 70)


# =====================================================================
#  MAIN — The high-level flow (start reading here!)
# =====================================================================

if __name__ == "__main__":
    DEMAND = 195  # MW — how much power the grid needs

    # 1. Define the generators and their constraints
    generators = define_generators()

    # 2. Encode the problem as a QUBO (quantum-ready format)
    bqm, var_names = build_qubo(generators, demand=DEMAND)
    print(f"Built QUBO: {len(var_names)} variables, {len(bqm.quadratic)} interactions")

    # 3. Find the classical optimum (for comparison)
    classical_best = classical_brute_force(generators, DEMAND, bqm)
    print(f"Classical optimum: P1={classical_best[0]}, P2={classical_best[1]}, "
          f"P3={classical_best[2]} MW  → Cost = {classical_best[3]:.1f} $")

    # 4. Solve with quantum computing
    print("\n🚀 Running quantum solver (PCE-VQE)...")
    pce_solver = solve_with_pce(bqm)

    # 5. Repair the quantum solution to make it fully valid
    result = find_best_repaired_solution(pce_solver, bqm, generators, DEMAND)

    if result is not None:
        powers, cost, prob = result
        print(f"\n   → Best repaired solution: P1={powers[0]:.0f}, "
              f"P2={powers[1]:.0f}, P3={powers[2]:.0f} MW, "
              f"Cost={cost:.1f} $  (quantum seed prob={prob:.2%})")

        # 6. Compare quantum vs classical
        print_comparison(result, classical_best)
    else:
        print("\n   ⚠️  No valid solution found. Try increasing max_iterations.")
