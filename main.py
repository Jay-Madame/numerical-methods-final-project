"""
main.py — Gaussian Elimination Benchmark Runner
Group Members: Jewell Callahan, Carson Milano, Tyler Evans, Damian McCone

Generates augmented matrices of varying sizes and benchmarks each teammate's
Gaussian elimination implementation. Add each solver below as they are submitted.
"""

import numpy as np
import time
import sys

# ── Algorithm Imports ─────────────────────────────────────────────────────────
# As each teammate finishes their implementation, import it here.
# Each solver must expose: solve(matrix: np.ndarray) -> np.ndarray | None
# where matrix is an (n x n+1) augmented matrix [A | b].

from HashTable import GaussianSumOptimized   # Jewell

# TODO Carson:  from CarsonAlg import solve as carson_solve
# TODO Damien:  from DamienAlg import solve as damien_solve
# TODO Tyler:   from TylerAlg import solve as tyler_solve


# ── Jewell's Solver (via HashTable.py) ───────────────────────────────────────

def jewell_solve(aug: np.ndarray) -> np.ndarray | None:
    """
    Jewell's implementation: Gaussian elimination with partial pivoting,
    using a linked-list hash table to memoize row scale factors.
    """
    cache = GaussianSumOptimized().cache
    A = aug.astype(float).copy()
    n = len(A)

    for col in range(n):
        # Partial pivot: swap largest absolute value row into position
        max_row = col + np.argmax(np.abs(A[col:, col]))
        A[[col, max_row]] = A[[max_row, col]]

        if abs(A[col, col]) < 1e-12:
            return None  # Singular matrix

        pivot_val = round(A[col, col], 6)

        for row in range(col + 1, n):
            cache_key = hash((col, pivot_val, round(A[row, col], 6))) % (10**9)
            cached_factor = cache.get(cache_key)

            if cached_factor is not None:
                factor = cached_factor
            else:
                factor = A[row, col] / A[col, col]
                cache.put(cache_key, factor)

            A[row, col:] -= factor * A[col, col:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(A[i, i]) < 1e-12:
            return None
        x[i] = (A[i, n] - np.dot(A[i, i + 1:n], x[i + 1:n])) / A[i, i]
    return x


# ── Active Algorithms ─────────────────────────────────────────────────────────
# Add each teammate's solver here as they finish.

ALGORITHMS = {
    "Jewell (Hash + Pivot)": jewell_solve,
    # "Carson  (???)": carson_solve,
    # "Damien  (???)": damien_solve,
    # "Tyler   (???)": tyler_solve,
}

REPEAT = 5  # Number of timed runs per matrix (averaged)


# ── Matrix Generator ──────────────────────────────────────────────────────────

def generate_matrix(n: int, min_val: float = -100.0, max_val: float = 100.0,
                    seed: int = 42) -> np.ndarray:
    """Generate a random (n x n+1) augmented matrix [A | b]."""
    np.random.seed(seed + n)
    return np.random.uniform(min_val, max_val, size=(n, n + 1))


def save_matrices_to_file(filename: str, sizes: list):
    """Save all generated matrices to a text file."""
    with open(filename, 'w') as f:
        for n in sizes:
            f.write(f"SIZE: {n}\n")
            matrix = generate_matrix(n)
            np.savetxt(f, matrix, fmt='%.4f')
            f.write("\n" + "=" * 20 + "\n\n")
    print(f"  Saved matrices {sizes} → '{filename}'")


# ── Residual Check ────────────────────────────────────────────────────────────

def residual_error(aug: np.ndarray, x: np.ndarray) -> float:
    """Compute ||Ax - b|| to verify solution accuracy."""
    A = aug[:, :-1]
    b = aug[:, -1]
    return float(np.linalg.norm(A @ x - b))


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(sizes: list):
    col_w = 26
    num_w = 12

    header  = (f"  {'Algorithm':<{col_w}} | {'Size':>{num_w}} | "
               f"{'Avg Time (ms)':>{num_w}} | {'Residual ||Ax-b||':>18} | {'Status':>8}")
    divider = "  " + "-" * (len(header) - 2)

    print("\n" + "=" * len(header))
    print("  GAUSSIAN ELIMINATION — BENCHMARK RESULTS")
    print("=" * len(header))

    for n in sizes:
        matrix = generate_matrix(n)
        print(f"\n  Matrix size: {n}×{n+1}  (augmented [{n}×{n} | b])\n")
        print(header)
        print(divider)

        for name, solver in ALGORITHMS.items():
            times  = []
            result = None

            for _ in range(REPEAT):
                mat_copy = matrix.copy()
                t0 = time.perf_counter()
                result = solver(mat_copy)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            avg_ms = sum(times) / len(times)

            if result is not None:
                err    = residual_error(matrix, result)
                status = "OK"
                err_str = f"{err:.2e}"
            else:
                status  = "SINGULAR"
                err_str = "N/A"

            print(f"  {name:<{col_w}} | {n:>{num_w}} | "
                  f"{avg_ms:>{num_w}.4f} | {err_str:>18} | {status:>8}")

        print(divider)

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512]

    # Optionally pass custom sizes as CLI args: python main.py 2 4 16 64
    if len(sys.argv) > 1:
        try:
            SIZES = [int(s) for s in sys.argv[1:]]
        except ValueError:
            print("Usage: python main.py [size1 size2 ...]\nExample: python main.py 2 4 16 64")
            sys.exit(1)

    print(f"\n  Test sizes : {SIZES}")
    print(f"  Avg over   : {REPEAT} runs per matrix")

    save_matrices_to_file("matrices.txt", SIZES)
    benchmark(SIZES)
