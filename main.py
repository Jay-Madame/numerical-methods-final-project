"""
main.py — Gaussian Elimination Benchmark Runner
Group: Jewell Callahan, Carson Milano, Tyler Evans, Damian McCone

Generates augmented matrices [A | b] of varying sizes, runs each teammate's
Gaussian elimination implementation, and prints a side-by-side comparison of
timing and solution accuracy.

Usage:
    python main.py                        # default power-of-2 sizes
    python main.py 2 4 16 64 256          # custom sizes
"""

import numpy as np
import time
import sys

# ── Algorithm Imports ─────────────────────────────────────────────────────────
from HashTable  import solve as jewell_solve   # Jewell  — hash table memoization
from CarsonAlg  import solve as carson_solve   # Carson  — LU decomposition
from DamianAlg  import solve as damian_solve   # Damian  — RREF (plain Python lists)
from TylerAlg   import solve as tyler_solve    # Tyler   — graph-based elimination

ALGORITHMS = {
    "Jewell  (Hash + Pivot) ": jewell_solve,
    "Carson  (LU Decomp)    ": carson_solve,
    "Damian  (RREF)         ": damian_solve,
    "Tyler   (Graph-based)  ": tyler_solve,
}

REPEAT = 5   # runs per (algorithm, matrix) pair — result is averaged


# ── Matrix Generator ──────────────────────────────────────────────────────────

def generate_matrix(n: int, seed: int = 42) -> np.ndarray:
    """Return a random (n × n+1) augmented matrix [A | b]."""
    np.random.seed(seed + n)
    return np.random.uniform(-100, 100, size=(n, n + 1))


def save_matrices(filename: str, sizes: list):
    """Write all test matrices to a text file for reference."""
    with open(filename, "w") as f:
        for n in sizes:
            f.write(f"SIZE: {n}\n")
            np.savetxt(f, generate_matrix(n), fmt="%.4f")
            f.write("\n" + "=" * 20 + "\n\n")
    print(f"  Matrices saved → '{filename}'")


# ── Accuracy Check ────────────────────────────────────────────────────────────

def residual(aug: np.ndarray, x: np.ndarray) -> float:
    """Compute ||Ax - b|| — lower is more accurate."""
    return float(np.linalg.norm(aug[:, :-1] @ x - aug[:, -1]))


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(sizes: list):
    name_w = 26
    num_w  = 10

    sep = "  " + "-" * 78

    print("\n" + "=" * 80)
    print("   GAUSSIAN ELIMINATION — BENCHMARK RESULTS")
    print("=" * 80)

    for n in sizes:
        matrix = generate_matrix(n)
        print(f"\n  ┌─ Matrix size: {n}×{n}  (augmented [{n}×{n} | b])")
        print(f"  │  Averaged over {REPEAT} runs\n")

        header = (f"  {'Algorithm':<{name_w}}  {'Avg (ms)':>{num_w}}  "
                  f"{'Best (ms)':>{num_w}}  {'Worst (ms)':>{num_w}}  "
                  f"{'||Ax-b||':>{num_w}}  {'Status':>8}")
        print(header)
        print(sep)

        results = []

        for name, solver in ALGORITHMS.items():
            times  = []
            result = None
            failed = False

            for _ in range(REPEAT):
                mat_copy = matrix.copy()
                try:
                    t0 = time.perf_counter()
                    result = solver(mat_copy)
                    t1 = time.perf_counter()
                    times.append((t1 - t0) * 1000)
                except Exception:
                    failed = True
                    break

            if failed or not times:
                print(f"  {name:<{name_w}}  {'—':>{num_w}}  {'—':>{num_w}}  "
                      f"{'—':>{num_w}}  {'—':>{num_w}}  {'ERROR':>8}")
                continue

            avg_ms  = sum(times) / len(times)
            best_ms = min(times)
            worst_ms = max(times)

            if result is not None:
                err    = residual(matrix, result)
                status = "OK"
                err_str = f"{err:.2e}"
            else:
                status  = "SINGULAR"
                err_str = "N/A"

            results.append((name, avg_ms))
            print(f"  {name:<{name_w}}  {avg_ms:>{num_w}.3f}  "
                  f"{best_ms:>{num_w}.3f}  {worst_ms:>{num_w}.3f}  "
                  f"{err_str:>{num_w}}  {status:>8}")

        # Fastest tag
        if results:
            fastest = min(results, key=lambda r: r[1])
            print(sep)
            print(f"  ★ Fastest: {fastest[0].strip()} ({fastest[1]:.3f} ms)\n")

    print("=" * 80 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512]

    if len(sys.argv) > 1:
        try:
            SIZES = [int(s) for s in sys.argv[1:]]
        except ValueError:
            print("Usage: python main.py [size1 size2 ...]\n"
                  "Example: python main.py 2 4 16 64")
            sys.exit(1)

    print(f"\n  Test sizes : {SIZES}")
    print(f"  Avg over   : {REPEAT} runs per algorithm per matrix")
    save_matrices("matrices.txt", SIZES)
    benchmark(SIZES)
