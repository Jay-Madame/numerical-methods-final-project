import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class GraphGaussianElimination:
    def __init__(self, A):
        self.A = A.astype(float)
        self.n = A.shape[0]
        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        #Build graph: nodes = rows, edges = shared nonzero columns
        self.graph.clear()
        for i in range(self.n):
            self.graph.add_node(i)

        for i in range(self.n):
            for j in range(i + 1, self.n):
                # If rows share a nonzero column → connect them
                if np.any((self.A[i] != 0) & (self.A[j] != 0)):
                    self.graph.add_edge(i, j)

    def visualize(self, title="Graph"):
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, with_labels=True, node_color="lightblue")
        plt.title(title)
        plt.show()

    def eliminate(self):
        #Perform Gaussian elimination while tracking graph changes
        for k in range(self.n):
            print(f"\n=== Pivot step {k} ===")

            pivot = self.A[k, k]
            if abs(pivot) < 1e-10:
                print("Skipping near-zero pivot")
                continue

            for i in range(k + 1, self.n):
                if self.A[i, k] != 0:
                    factor = self.A[i, k] / pivot
                    print(f"Eliminating row {i} using row {k}, factor = {factor:.3f}")

                    # Perform row operation
                    self.A[i, k:] -= factor * self.A[k, k:]

            # Rebuild graph after elimination (captures fill-in)
            old_edges = set(self.graph.edges())
            self._build_graph()
            new_edges = set(self.graph.edges())

            fill_in_edges = new_edges - old_edges
            print(f"New edges (fill-in): {fill_in_edges}")

            self.visualize(title=f"After step {k}")

        return self.A


if __name__ == "__main__":
    # Example matrix (banded → will show fill-in over time)
    A = np.array([
        [2, 1, 0, 0],
        [1, 2, 1, 0],
        [0, 1, 2, 1],
        [0, 0, 1, 2]
    ], dtype=float)

    gge = GraphGaussianElimination(A)

    print("Initial Graph:")
    gge.visualize("Initial Graph")

    U = gge.eliminate()

    print("\nFinal Upper Triangular Matrix:")
    print(U)
    
