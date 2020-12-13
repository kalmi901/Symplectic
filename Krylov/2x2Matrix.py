import numpy as np
from GMRES import gmres

A = np.array([[2, 3], [4, -1]])
x = np.array([10.0, -2000.0])
b = np.array([1, 23])

print(gmres(A, x, b, 2, tol=1e-6))