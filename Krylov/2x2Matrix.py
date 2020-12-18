import numpy as np
from GMRES import gmres
from BICG import bicg

A = np.array([[2, 3], [4, -1]])
# x = np.array([10000.0, -2000.0])
x = np.array([5.0, -2.0])
b = np.array([1, 23])

# print(gmres(A, x, b, 20, tol=1e-6))
print(bicg(A, x, b, 20, tol=1e-6))