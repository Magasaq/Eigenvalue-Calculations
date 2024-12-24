
import numpy as np


def sign(x):
    return 1.0 if x >= 0 else -1.0


def find_biggest_eigen_value_vector(A: np.ndarray, eps=1e-7):
    assert A.shape[0] == A.shape[1], "matrix must be square"
    n = A.shape[0]

    x_prev = np.random.rand(n)

    while True:
        x_next = A @ x_prev

        if np.allclose(x_next, np.zeros_like(x_next)):
            return 0.0, x_prev

        x_next = x_next / np.linalg.norm(x_next)
        alpha = np.random.rand()
        x_next = alpha * x_next + (1 - alpha) * x_prev
        x_next = x_next / np.linalg.norm(x_next)

        idx = np.argmax(x_next != 0.0)
        x_next *= sign(x_next[idx])
        if np.linalg.norm(x_next - x_prev) < eps:
            break

        x_prev = x_next

    idx = np.argmax(np.abs(x_next) >= eps / 2)

    lambda_ = (A @ x_next)[idx] / x_next[idx]

    return lambda_, x_next


def exclude_eigen_direction(A: np.ndarray, vector: np.ndarray, lambda_: float):
    norm_vector = np.linalg.norm(vector)
    assert norm_vector >= 1e-7, f"eigen vector must be non zero, got vector = {vector}"
    return A - lambda_ * (vector[:, np.newaxis] @ vector[np.newaxis, :]) / norm_vector**2


def find_k_dominant_eigen_values_vectors_of_symmetric_matrix(A: np.ndarray, k: int):

    assert np.allclose(A, A.T), "matrix not symmetric"
    assert k <= A.shape[0], f"k > {A.shape[0]}. {A.shape[0]} x {A.shape[0]} matrix has {A.shape[0]} eigen values"

    lambdas = []
    vectors = []

    A_copy = A.copy()

    for i in range(k):
        lambda_, vector = find_biggest_eigen_value_vector(A_copy)
        lambdas.append(lambda_)
        vectors.append(vector)
        A_copy = exclude_eigen_direction(A_copy, vector, lambda_)
        # A_copy -= lambda_ * (vector[:, np.newaxis] @ vector[np.newaxis, :]) / np.linalg.norm(vector)**2

    return np.array(lambdas), np.array(vectors)


