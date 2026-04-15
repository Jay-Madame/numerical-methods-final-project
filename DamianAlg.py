def row_reduce(mat):
    if not mat or not mat[0]:
        return mat
    mat = [row[:] for row in mat]
    rows, cols = len(mat), len(mat[0])
    pivot_row = 0
    for col in range(cols):
        # find pivot
        found = -1
        for r in range(pivot_row, rows):
            if mat[r][col] != 0:
                found = r
                break
        if found == -1:
            continue
        mat[pivot_row], mat[found] = mat[found], mat[pivot_row]
        # scale pivot row so leading entry = 1
        scale = mat[pivot_row][col]
        mat[pivot_row] = [x / scale for x in mat[pivot_row]]
        # eliminate column entries above and below pivot
        for r in range(rows):
            if r != pivot_row and mat[r][col] != 0:
                factor = mat[r][col]
                mat[r] = [mat[r][c] - factor * mat[pivot_row][c] for c in range(cols)]
        pivot_row += 1
    return mat
