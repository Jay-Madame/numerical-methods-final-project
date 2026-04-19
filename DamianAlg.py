import numpy as np


def row_reduce(mat):
    """
    Damian's original RREF implementation using plain Python lists.
    Operates on a 2D list (augmented matrix [A | b]).
    Returns the matrix in reduced row echelon form.
    """
    if not mat or not mat[0]:
        return mat
    mat = [row[:] for row in mat]
    rows, cols = len(mat), len(mat[0])
    pivot_row = 0
    for col in range(cols):
        found = -1
        for r in range(pivot_row, rows):
            if mat[r][col] != 0:
                found = r
                break
        if found == -1:
            continue
        mat[pivot_row], mat[found] = mat[found], mat[pivot_row]
        scale = mat[pivot_row][col]
        mat[pivot_row] = [x / scale for x in mat[pivot_row]]
        for r in range(rows):
            if r != pivot_row and mat[r][col] != 0:
                factor = mat[r][col]
                mat[r] = [mat[r][c] - factor * mat[pivot_row][c] for c in range(cols)]
        pivot_row += 1
    return mat


def solve(aug: np.ndarray) -> 'np.ndarray | None':
    """
    main.py entry point — wraps row_reduce() to accept a numpy augmented
    matrix and return the solution vector x.
    """
    mat = aug.tolist()
    rref = row_reduce(mat)
    n = len(rref)

    # Check for singular / inconsistent system
    for r in rref:
        if all(abs(v) < 1e-12 for v in r[:-1]) and abs(r[-1]) > 1e-12:
            return None

    # Extract solution from last column
    x = np.array([rref[i][-1] for i in range(n)])
    return x


if __name__ == "__main__":
    aug = np.array([[2, 1, -1, 8],
                    [-3, -1, 2, -11],
                    [-2, 1, 2, -3]], dtype=float)
    print(solve(aug))  # Expected: [2, 3, -1]
