import numpy as np
import scipy.linalg as la


def solve_with_lu(A, b):
    """LU decomposition with partial pivoting via scipy."""
    P, L, U = la.lu(A)
    y = la.solve_triangular(L, P @ b, lower=True)
    x = la.solve_triangular(U, y, lower=False)
    return x


def solve_with_custom_elimination(A, b):
    """Custom Gaussian elimination with partial pivoting (vectorized)."""
    A_work = A.astype(float).copy()
    b_work = b.astype(float).copy()
    n = len(A_work)

    for j in range(n - 1):
        i = np.argmax(np.abs(A_work[j:, j])) + j
        if abs(A_work[i, j]) < 1e-12:
            raise np.linalg.LinAlgError("Matrix is singular")
        if i != j:
            A_work[[j, i], j:] = A_work[[i, j], j:]
            b_work[[j, i]] = b_work[[i, j]]
        for i in range(j + 1, n):
            multiplier = A_work[i, j] / A_work[j, j]
            A_work[i, j + 1:] -= multiplier * A_work[j, j + 1:]
            b_work[i] -= multiplier * b_work[j]

    x = np.zeros(n)
    x[-1] = b_work[-1] / A_work[-1, -1]
    for i in range(n - 2, -1, -1):
        x[i] = (b_work[i] - A_work[i, i + 1:] @ x[i + 1:]) / A_work[i, i]
    return x


def solve(aug: np.ndarray) -> 'np.ndarray | None':
    """
    main.py entry point — uses LU decomposition as the primary method,
    falls back to custom elimination if LU fails.
    """
    A = aug[:, :-1].astype(float).copy()
    b = aug[:, -1].astype(float).copy()
    try:
        return solve_with_lu(A, b)
    except Exception:
        try:
            return solve_with_custom_elimination(A, b)
        except Exception:
            return None


if __name__ == "__main__":
    aug = np.array([[2, 1, -1, 8],
                    [-3, -1, 2, -11],
                    [-2, 1, 2, -3]], dtype=float)
    print(solve(aug))  # Expected: [2, 3, -1]
