
import numpy as np


class Graph:

    def __init__(self, node_count=1):
        self.node_count = node_count
        self.nodes = set(range(node_count))
        self.edges = set()

    def add_node(self):
        self.nodes.add(self.node_count)
        self.node_count += 1

    def add_edge(self, u: int, v: int):
        assert u in self.nodes and v in self.nodes, f"{u} and {v} must be in the node set"

        self.edges.add((u, v))
        self.edges.add((v, u))

    def to_matrix(self):
        matrix = np.zeros((self.node_count, self.node_count))
        for u, v in self.edges:
            matrix[u, v] = 1
        return matrix

    def apply_permutation(self, sigma: np.ndarray):
        new_edges = set()
        for u, v in self.edges:
            new_edges.add((sigma[u], sigma[v]))
            new_edges.add((sigma[v], sigma[u]))
        self.edges = new_edges


