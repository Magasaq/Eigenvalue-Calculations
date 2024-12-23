import numpy as np
from sklearn.metrics import mean_squared_error as mse

def householder_transformation(v):
    size_of_v = v.shape[0] 
    e1 = np.zeros_like(v)
    e1[0] = 1  
    vector = np.linalg.norm(v) * e1 
    if v[0] < 0:
        vector = -vector
    u = (v + vector).astype(np.float32)

    u = u.reshape(-1, 1) 

    epsilon = 1e-10
    norm_u_squared = np.matmul(u.T, u) + epsilon  

    H = np.identity(size_of_v) - (2 * np.matmul(u, u.T)) / norm_u_squared
    return H

def column_convertor(x):
    return x.reshape(-1, 1) 

def qr_decomposition(A):
    """
    Computes the QR decomposition of matrix A using Gram-Schmidt process.

    Parameters:
    - A: numpy.ndarray, the input matrix.

    Returns:
    - Q: numpy.ndarray, orthogonal matrix.
    - R: numpy.ndarray, upper triangular matrix.
    """
    m, n = A.shape
    
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for i in range(n):
        Q[:, i] = A[:, i]
        
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], A[:, i])
            Q[:, i] = Q[:, i] - R[j, i] * Q[:, j]
        
        R[i, i] = np.linalg.norm(Q[:, i])
        Q[:, i] = Q[:, i] / R[i, i]

    return Q, R

def qr_step_factorization(q, r, iter, n):
    v = column_convertor(r[iter:, iter])
    Hbar = householder_transformation(v)
    H = np.identity(n)
    H[iter:, iter:] = Hbar
    r = np.matmul(H, r)
    q = np.matmul(q, H)
    return q, r

def qr(A):
    n = A.shape[0]
    m = A.shape[1]
    Q = np.identity(n)
    R = A.astype(np.float32)
    for i in range(min(n, m)):
        Q, R = qr_step_factorization(Q, R, i, n)
    min_dim = min(m, n)
    R = np.around(R, decimals=6)
    R = R[:min_dim, :min_dim]
    Q = np.around(Q, decimals=6)

    return Q, R
