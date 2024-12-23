import numpy as np
from sklearn.metrics import mean_squared_error as mse
from QR.qr import qr_algorithm
import time

def get_matrix(n):
    sqrtA = np.random.rand(n, n) - 0.5
    A = np.dot(sqrtA, sqrtA.T)
    return A

n = 100

A = get_matrix(n)

start_time = time.time()
eigenvalues = qr_algorithm(A)
end_time = time.time()

eigenvalues_np, _ = np.linalg.eig(A)

eigenvalues_qr = np.sort(eigenvalues)
eigenvalues_np = np.sort(eigenvalues_np)

mse_calc = mse(eigenvalues_np, eigenvalues)

output = {
    "n": n,
    "mse": mse_calc,
    "time": end_time - start_time
}

print(output)
