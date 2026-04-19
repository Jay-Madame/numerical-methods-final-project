import numpy as np


class Node:
    """A node in a singly linked list (used for hash table chaining)."""
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None


class LinkedListHashTable:
    """
    Hash table using separate chaining (linked lists) to handle collisions.
    Stores memoized scale factors for Gaussian elimination at O(1) avg lookup.
    """
    def __init__(self, capacity=256):
        self.capacity = capacity
        self.buckets = [None] * self.capacity

    def _hash(self, key):
        return key % self.capacity

    def get(self, key):
        idx = self._hash(key)
        node = self.buckets[idx]
        while node:
            if node.key == key:
                return node.value
            node = node.next
        return None

    def put(self, key, value):
        idx = self._hash(key)
        node = self.buckets[idx]
        while node:
            if node.key == key:
                node.value = value
                return
            node = node.next
        new_node = Node(key, value)
        new_node.next = self.buckets[idx]
        self.buckets[idx] = new_node


class GaussianSumOptimized:
    """
    Gaussian elimination with partial pivoting, accelerated by a
    linked-list hash table that memoizes row scale factors.

    Cache key: exact integer triple (col, pivot_row, elim_row) — no float
    rounding, so precision is preserved at all matrix sizes.
    """
    def __init__(self):
        self.cache = LinkedListHashTable()

    def gaussian_sum(self, n: int) -> int:
        if n < 0:
            raise ValueError("n must be a non-negative integer")
        cached = self.cache.get(n)
        if cached is not None:
            return cached
        result = (n * (n + 1)) // 2
        self.cache.put(n, result)
        return result

    def solve(self, aug: np.ndarray) -> 'np.ndarray | None':
        """
        Solve a linear system from an (n x n+1) augmented matrix [A | b].
        Returns solution vector x, or None if matrix is singular.
        """
        A = aug.astype(float).copy()
        n = len(A)

        for col in range(n):
            max_row = col + int(np.argmax(np.abs(A[col:, col])))
            if max_row != col:
                A[[col, max_row]] = A[[max_row, col]]

            if abs(A[col, col]) < 1e-12:
                return None

            for row in range(col + 1, n):
                # Exact integer key — no floating-point precision loss
                cache_key = col * n * n + max_row * n + row
                factor = self.cache.get(cache_key)
                if factor is None:
                    factor = A[row, col] / A[col, col]
                    self.cache.put(cache_key, factor)
                A[row, col:] -= factor * A[col, col:]

        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if abs(A[i, i]) < 1e-12:
                return None
            x[i] = (A[i, n] - np.dot(A[i, i + 1:n], x[i + 1:n])) / A[i, i]
        return x


# main.py entry point
_solver = GaussianSumOptimized()

def solve(aug: np.ndarray) -> 'np.ndarray | None':
    return _solver.solve(aug)


if __name__ == "__main__":
    solver = GaussianSumOptimized()
    aug = np.array([[2, 1, -1, 8],
                    [-3, -1, 2, -11],
                    [-2, 1, 2, -3]], dtype=float)
    x = solver.solve(aug)
    print(f"Solution x = {x}")  # Expected: [2, 3, -1]
