import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import cg, gmres
import time

def solve_with_backslash(A, b):
    # Method 1: Using numpy's linear algebra solver
    return np.linalg.solve(A, b)

def solve_with_lu(A, b):
    # Method 2: LU decomposition with partial pivoting
    # scipy.linalg.lu returns P, L, U such that P*A = L*U
    P, L, U = la.lu(A)
    # Solve: A*x = b  =>  P*A*x = P*b  =>  L*U*x = P*b
    # Step 1: Solve L*y = P*b
    y = la.solve_triangular(L, P @ b, lower=True)
    # Step 2: Solve U*x = y
    x = la.solve_triangular(U, y, lower=False)
    return x

def solve_with_cholesky(A, b):
    #Method 3: Cholesky decomposition
    try:
        # Check if matrix is symmetric
        if not np.allclose(A, A.T):
            return None, "Matrix is not symmetric"
        
        # Attempt Cholesky decomposition
        L = la.cholesky(A, lower=True)  # A = L * L.T
        # Solve: L*y = b
        y = la.solve_triangular(L, b, lower=True)
        # Solve: L.T*x = y
        x = la.solve_triangular(L.T, y, lower=False)
        return x, "Success"
    except la.LinAlgError:
        return None, "Matrix is not positive definite"

def solve_with_custom_elimination(A, b):
    #Method 4: Custom Gaussian elimination with partial pivoting (optimized)
    # Make working copies
    A_work = A.astype(float).copy()
    b_work = b.astype(float).copy()
    n = len(A_work)
    
    # Forward elimination with partial pivoting
    for j in range(n-1):
        # Find pivot row (max in column j from row j to n-1)
        pivot_col = np.abs(A_work[j:, j])
        i = np.argmax(pivot_col) + j
        
        # Check for singular matrix
        if abs(A_work[i, j]) < 1e-12:
            raise np.linalg.LinAlgError("Matrix is singular")
        
        # Swap rows if needed
        if i != j:
            # Swap rows in A and b
            A_work[[j, i], j:] = A_work[[i, j], j:]
            b_work[[j, i]] = b_work[[i, j]]
        
        # Elimination step (vectorized)
        for i in range(j+1, n):
            multiplier = A_work[i, j] / A_work[j, j]
            # Vectorized update of remaining row elements
            A_work[i, j+1:] -= multiplier * A_work[j, j+1:]
            b_work[i] -= multiplier * b_work[j]
    
    # Backward substitution
    x = np.zeros(n)
    x[-1] = b_work[-1] / A_work[-1, -1]
    for i in range(n-2, -1, -1):
        # Vectorized dot product for the sum
        x[i] = (b_work[i] - A_work[i, i+1:] @ x[i+1:]) / A_work[i, i]
    
    return x

def solve_with_iterative_methods(A, b, method='cg'):
    # Method 5: Iterative methods for large sparse systems
    if method == 'cg':
        # Conjugate Gradient (for symmetric positive definite)
        x, info = cg(A, b, tol=1e-6, maxiter=1000)
        if info > 0:
            print(f"  Warning: CG did not converge after {info} iterations")
        return x
    elif method == 'gmres':
        # GMRES
        x, info = gmres(A, b, tol=1e-6, maxiter=1000)
        if info > 0:
            print(f"  Warning: GMRES did not converge after {info} iterations")
        return x
    else:
        raise ValueError("Method must be 'cg' or 'gmres'")

def main():
    # Input matrix and vector
    A = np.array([[0.003, 59.14],
                  [5.291, -6.13]], dtype=float)
    b = np.array([59.17, 46.78], dtype=float)
    
    n = A.shape[0]
    
    print("=" * 60)
    print("SOLVING LINEAR SYSTEM Ax = b")
    print("=" * 60)
    print(f"Matrix A:\n{A}")
    print(f"\nVector b: {b}")
    print(f"Matrix size: {n}x{n}\n")
    
    # Method 1: numpy.linalg.solve
    print("\n=== METHOD 1: numpy.linalg.solve ===")
    x1 = solve_with_backslash(A, b)
    print(f"Solution: {x1}")
    print(f"Residual norm: {np.linalg.norm(A @ x1 - b):.2e}")
    
    # Method 2: LU decomposition
    print("\n=== METHOD 2: LU decomposition ===")
    x2 = solve_with_lu(A, b)
    print(f"Solution: {x2}")
    print(f"Residual norm: {np.linalg.norm(A @ x2 - b):.2e}")
    
    # Method 3: Cholesky decomposition
    print("\n=== METHOD 3: Cholesky decomposition ===")
    x3, status = solve_with_cholesky(A, b)
    if x3 is not None:
        print(f"Solution: {x3}")
        print(f"Residual norm: {np.linalg.norm(A @ x3 - b):.2e}")
    else:
        print(f"Cholesky not applicable: {status}")
    
    # Method 4: Custom elimination
    print("\n=== METHOD 4: Optimized custom elimination ===")
    x4 = solve_with_custom_elimination(A, b)
    print(f"Solution: {x4}")
    print(f"Residual norm: {np.linalg.norm(A @ x4 - b):.2e}")
    
    # Method 5: Iterative methods
    if n <= 10:  # Small problem, just show concept
        print("\n=== METHOD 5: Iterative methods (CG/GMRES) ===")
        print("  (Skipped for this small problem - iterative methods are")
        print("   beneficial only for large sparse matrices, n > 1000)")
    
    # Verification: All methods should give same solution
    print("\n=== VERIFICATION ===")
    print(f"All methods consistent? {np.allclose(x1, x2) and np.allclose(x1, x4)}")
    
    # Performance comparison for larger matrices (optional demo)
    print("\n=== PERFORMANCE NOTES ===")
    print(f"For this small matrix (n={n}), all methods are fast.")
    print("For larger matrices (n > 1000):")
    print("  - numpy.linalg.solve uses LAPACK (optimized, ~5-10x faster than custom)")
    print("  - For symmetric positive definite: use Cholesky (2x faster)")
    print("  - For sparse matrices: use scipy.sparse.linalg.spsolve")
    print("  - For very large sparse: use iterative methods (CG/GMRES)")
    
    # Demonstrate with a larger matrix (optional, uncomment to test)
    if False:  # Set to True to test performance on larger matrix
        print("\n" + "=" * 60)
        print("PERFORMANCE TEST ON LARGER MATRIX")
        print("=" * 60)
        
        n_large = 500
        A_large = np.random.rand(n_large, n_large)
        # Make it diagonally dominant for stability
        A_large = A_large + n_large * np.eye(n_large)
        b_large = np.random.rand(n_large)
        
        # Test numpy.linalg.solve
        start = time.time()
        x_solve = np.linalg.solve(A_large, b_large)
        time_solve = time.time() - start
        print(f"First solution element (solve): {x_solve[0]:.6f}")
        
        # Test custom elimination
        start = time.time()
        x_custom = solve_with_custom_elimination(A_large, b_large)
        time_custom = time.time() - start
        print(f"First solution element (custom): {x_custom[0]:.6f}")
        
        print(f"\nnumpy.linalg.solve time: {time_solve:.3f} seconds")
        print(f"Custom elimination time: {time_custom:.3f} seconds")
        print(f"Speedup: {time_custom/time_solve:.1f}x")
        
        # Test iterative methods (if matrix is SPD)
        if np.allclose(A_large, A_large.T) and np.all(np.linalg.eigvals(A_large) > 0):
            start = time.time()
            x_cg = solve_with_iterative_methods(A_large, b_large, method='cg')
            time_cg = time.time() - start
            print(f"First solution element (CG): {x_cg[0]:.6f}")
            print(f"Conjugate Gradient time: {time_cg:.3f} seconds")
            print(f"CG vs Solve speedup: {time_solve/time_cg:.1f}x")

if __name__ == "__main__":
    main()
  
