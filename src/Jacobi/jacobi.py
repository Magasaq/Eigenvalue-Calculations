import numpy as np

def jacobi_rotation_method(A, tolerance=1e-10, max_iterations=100):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    assert A.shape[1] == n, "Input matrix must be square."
    assert np.allclose(A, A.T, atol=tolerance), "Matrix must be symmetric."

    eigenvectors = np.eye(n)
    
    for iteration in range(max_iterations):
        # Find the indices of the largest off-diagonal element in A
        off_diag_abs = np.abs(np.triu(A, 1))  # Get the upper triangle without the diagonal
        p, q = np.unravel_index(np.argmax(off_diag_abs), off_diag_abs.shape)
        off_diag_max = A[p, q]

        # Break if the largest off-diagonal element is below the tolerance
        if abs(off_diag_max) < tolerance:
            break
        
        # Compute the rotation angle
        theta = 0.5 * np.arctan2(2 * A[p, q], A[p, p] - A[q, q])
        c, s = np.cos(theta), np.sin(theta)

        # Construct the rotation matrix J
        J = np.eye(n)
        J[p, p], J[q, q] = c, c
        J[p, q], J[q, p] = s, -s

        # Apply the rotation
        A = J.T @ A @ J
        eigenvectors = eigenvectors @ J

    # The eigenvalues are the diagonal elements of A
    eigenvalues = np.diag(A)
    return eigenvalues, eigenvectors
