import numpy as np
from src.QR.qr import qr_algorithm
import time
from src.utils import Graph
from src.Power.matrix_analysis import *
from src.Jacobi.jacobi import jacobi_rotation_method


def isomorphism_check(g1: Graph, g2: Graph, method="power"):
    m1 = g1.to_matrix()
    m2 = g2.to_matrix()
    n = m1.shape[0]

    if method == "power":
        for i in range(n):
            lambda_1, vector_1 = find_biggest_eigen_value_vector(m1)
            lambda_2, vector_2 = find_biggest_eigen_value_vector(m2)
            np.set_printoptions(suppress=True)
            if not np.allclose(lambda_1, lambda_2):
                return False
            m1 = exclude_eigen_direction(m1, vector_1, lambda_1)
            m2 = exclude_eigen_direction(m2, vector_2, lambda_2)
        return True
    elif method == "QR":
        lambdas_1 = np.sort(qr_algorithm(m1))
        lambdas_2 = np.sort(qr_algorithm(m2))
        return np.allclose(lambdas_1, lambdas_2)
    elif method == "Jacoby":
        lambdas_1, _ = jacobi_rotation_method(m1)
        lambdas_2, _ = jacobi_rotation_method(m2)
        lambdas_1 = np.sort(lambdas_1)
        lambdas_2 = np.sort(lambdas_2)
        np.set_printoptions(suppress=True)
        return np.allclose(lambdas_1, lambdas_2)


def main():

    graph1 = Graph(7)

    graph1.add_edge(0, 1)
    graph1.add_edge(1, 2)
    graph1.add_edge(3, 1)
    graph1.add_edge(0, 4)
    graph1.add_edge(0, 5)
    graph1.add_edge(0, 6)

    graph2 = Graph(7)

    graph2.add_edge(1, 2)
    graph2.add_edge(1, 0)
    graph2.add_edge(3, 1)
    graph2.add_edge(3, 4)
    graph2.add_edge(0, 5)
    graph2.add_edge(5, 6)

    print(isomorphism_check(graph1, graph2, "power"))


if __name__ == "__main__":
    main()
