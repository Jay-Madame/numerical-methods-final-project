class Node:
    """A node in a singly linked list (used for hash table chaining)."""
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None


class LinkedListHashTable:
    """
    Hash table using separate chaining (linked lists) to handle collisions.
    Stores previously computed Gaussian sums for O(1) average-case lookup.
    """
    def __init__(self, capacity=64):
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
        # Update existing key if found
        while node:
            if node.key == key:
                node.value = value
                return
            node = node.next
        # Prepend new node (O(1) insertion at head)
        new_node = Node(key, value)
        new_node.next = self.buckets[idx]
        self.buckets[idx] = new_node


class GaussianSumOptimized:
    """
    Computes the sum 1 + 2 + ... + n using Gauss's closed-form formula,
    with a linked-list hash table as a memoization cache.

    - Naive loop:         O(n) time
    - This approach:      O(1) time (formula) + O(1) avg cache ops
    """
    def __init__(self):
        self.cache = LinkedListHashTable()

    def gaussian_sum(self, n: int) -> int:
        if n < 0:
            raise ValueError("n must be a non-negative integer")

        # 1. Cache hit → O(1) average
        cached = self.cache.get(n)
        if cached is not None:
            return cached

        # 2. Gauss's formula → O(1), no loop needed
        result = (n * (n + 1)) // 2

        # 3. Store in hash table for future lookups
        self.cache.put(n, result)
        return result


# ── Demo ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    solver = GaussianSumOptimized()

    test_cases = [0, 1, 10, 100, 1_000, 1_000_000, 10_000_000]

    print(f"{'n':>12} | {'Gaussian Sum':>20} | {'Cache Hit?':>10}")
    print("-" * 50)

    for n in test_cases:
        result = solver.gaussian_sum(n)          # first call — computes & caches
        cached  = solver.gaussian_sum(n)          # second call — cache hit
        hit = solver.cache.get(n) is not None
        print(f"{n:>12,} | {result:>20,} | {'Yes' if hit else 'No':>10}")
