import numpy as np
from GMRES import gmres_solve
from BICG import bicg_solve

A = np.array([[2.0, 3.0], [4.0, -1.0]])
# x = np.array([10000.0, -2000.0])
x = np.array([5.0, 3000000.100000])
b = np.array([1.0, 23.0])

#print(gmres_solve(A, b, x, 1000, 3, tol=1e-6))
print(bicg_solve(A, b, x, 20, tol=1e-6))