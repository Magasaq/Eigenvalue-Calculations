
import numpy as np
from qr_lib import qr_decomposition, householder_transformation, qr_step_factorization,qr
from sklearn.metrics import mean_squared_error as mse

def qr_algorithm(A, max_iter=1000, tol=1e-9):
    """
    Computes the eigenvalues of a square matrix A using the QR algorithm.

    Parameters:
    - A: numpy.ndarray, the input square matrix.
    - max_iter: int, the maximum number of iterations.
    - tol: float, the tolerance for convergence.

    Returns:
    - eigenvalues: numpy.ndarray, the approximate eigenvalues of A.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    for i in range(max_iter):
        Q, R = np.linalg.qr(A)
        A_new = R @ Q

        if np.allclose(A, A_new, atol=tol, rtol=0):
            break
        A = A_new
    
    eigenvalues = np.diag(A)
    return eigenvalues
