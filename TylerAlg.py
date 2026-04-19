import numpy as np
import networkx as nx


class GraphGaussianElimination:
    """
    Tyler's graph-based Gaussian elimination.
    Models the matrix as a graph (nodes = rows, edges = shared nonzero columns)
    to track structural fill-in during elimination — useful for sparse analysis.
    Visualization is disabled during benchmarking (visualize=False).
    """
    def __init__(self, A, visualize=False):
        self.A = A.astype(float)
        self.n = A.shape[0]
        self.do_visualize = visualize
        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        self.graph.clear()
        for i in range(self.n):
            self.graph.add_node(i)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if np.any((self.A[i] != 0) & (self.A[j] != 0)):
                    self.graph.add_edge(i, j)

    def visualize(self, title="Graph"):
        """Only runs when visualize=True (i.e. standalone, not benchmarking)."""
        if not self.do_visualize:
            return
        try:
            import matplotlib.pyplot as plt
            pos = nx.spring_layout(self.graph, seed=42)
            nx.draw(self.graph, pos, with_labels=True, node_color="lightblue")
            plt.title(title)
            plt.show()
        except Exception:
            pass

    def eliminate(self):
        """
        Gaussian elimination with graph fill-in tracking.
        Returns the upper triangular matrix after forward elimination.
        """
        for k in range(self.n):
            pivot = self.A[k, k]
            if abs(pivot) < 1e-10:
                continue

            for i in range(k + 1, self.n):
                if self.A[i, k] != 0:
                    factor = self.A[i, k] / pivot
                    self.A[i, k:] -= factor * self.A[k, k:]

            old_edges = set(self.graph.edges())
            self._build_graph()
            new_edges = set(self.graph.edges())
            fill_in = new_edges - old_edges

            self.visualize(title=f"After step {k}")

        return self.A

    def solve(self) -> 'np.ndarray | None':
        """
        Run elimination then back-substitute to return solution vector x.
        The last column of self.A is treated as b (augmented matrix form).
        """
        n = self.n
        self.eliminate()

        # Back substitution on upper triangular [A | b]
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if abs(self.A[i, i]) < 1e-12:
                return None
            x[i] = (self.A[i, n] - np.dot(self.A[i, i + 1:n], x[i + 1:n])) / self.A[i, i]
        return x


def solve(aug: np.ndarray) -> 'np.ndarray | None':
    """main.py entry point."""
    gge = GraphGaussianElimination(aug.copy(), visualize=False)
    return gge.solve()


if __name__ == "__main__":
    aug = np.array([[2, 1, -1, 8],
                    [-3, -1, 2, -11],
                    [-2, 1, 2, -3]], dtype=float)
    print(solve(aug))  # Expected: [2, 3, -1]
