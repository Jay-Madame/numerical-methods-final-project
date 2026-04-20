"""
main.py — Gaussian Elimination Benchmark Runner
Group: Jewell Callahan, Carson Milano, Tyler Evans, Damian McCone

Generates augmented matrices [A | b] of varying sizes, runs each teammate's
Gaussian elimination implementation 30 times per matrix for statistical
significance, prints a side-by-side comparison, and writes all timing
results to results.csv for downstream statistical analysis.

Usage:
    python main.py                        # default power-of-2 sizes
    python main.py 2 4 16 64 256          # custom sizes
"""

import numpy as np
import time
import sys
import csv

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

REPEAT = 30  # 30 runs per (algorithm, matrix) for statistical significance
CSV_FILE = "results.csv"


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


# ── CSV Writer ────────────────────────────────────────────────────────────────

def init_csv(filename: str):
    """Write the CSV header row."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "matrix_size", "run", "time_ms", "residual", "status"])
    print(f"  CSV initialized → '{filename}'")


def append_csv(filename: str, rows: list):
    """Append a batch of result rows to the CSV."""
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(sizes: list):
    name_w = 26
    num_w  = 10
    sep    = "  " + "-" * 84

    init_csv(CSV_FILE)

    print("\n" + "=" * 86)
    print("   GAUSSIAN ELIMINATION — BENCHMARK RESULTS  (n=30 runs per algorithm)")
    print("=" * 86)

    for n in sizes:
        matrix = generate_matrix(n)
        print(f"\n  ┌─ Matrix size: {n}×{n}  (augmented [{n}×{n} | b])")
        print(f"  │  {REPEAT} runs per algorithm\n")

        header = (f"  {'Algorithm':<{name_w}}  {'Avg (ms)':>{num_w}}  "
                  f"{'Std Dev':>{num_w}}  {'Best (ms)':>{num_w}}  "
                  f"{'Worst (ms)':>{num_w}}  {'||Ax-b||':>{num_w}}  {'Status':>8}")
        print(header)
        print(sep)

        results = []

        for name, solver in ALGORITHMS.items():
            times   = []
            csv_rows = []
            result  = None
            failed  = False

            for run in range(1, REPEAT + 1):
                mat_copy = matrix.copy()
                try:
                    t0 = time.perf_counter()
                    result = solver(mat_copy)
                    t1 = time.perf_counter()
                    elapsed_ms = (t1 - t0) * 1000
                    times.append(elapsed_ms)

                    err    = residual(matrix, result) if result is not None else None
                    status = "OK" if result is not None else "SINGULAR"
                    err_val = f"{err:.6e}" if err is not None else "N/A"

                    csv_rows.append([
                        name.strip(), n, run,
                        f"{elapsed_ms:.6f}", err_val, status
                    ])
                except Exception as e:
                    failed = True
                    csv_rows.append([name.strip(), n, run, "ERROR", "N/A", "ERROR"])
                    break

            append_csv(CSV_FILE, csv_rows)

            if failed or not times:
                print(f"  {name:<{name_w}}  {'ERROR':>{num_w}}")
                continue

            avg_ms   = np.mean(times)
            std_ms   = np.std(times, ddof=1)
            best_ms  = np.min(times)
            worst_ms = np.max(times)
            err_str  = f"{residual(matrix, result):.2e}" if result is not None else "N/A"
            status   = "OK" if result is not None else "SINGULAR"

            results.append((name, avg_ms))
            print(f"  {name:<{name_w}}  {avg_ms:>{num_w}.3f}  "
                  f"{std_ms:>{num_w}.3f}  {best_ms:>{num_w}.3f}  "
                  f"{worst_ms:>{num_w}.3f}  {err_str:>{num_w}}  {status:>8}")

        if results:
            fastest = min(results, key=lambda r: r[1])
            print(sep)
            print(f"  ★ Fastest: {fastest[0].strip()} ({fastest[1]:.3f} ms avg)\n")

    print("=" * 86)
    print(f"\n  All timing data written → '{CSV_FILE}'")
    print(f"  Run stats.py next to see ANOVA and post-hoc results.\n")


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
    print(f"  Runs each  : {REPEAT} (for statistical significance)")
    save_matrices("matrices.txt", SIZES)
    benchmark(SIZES)
