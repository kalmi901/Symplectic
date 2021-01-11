import numpy as np
import IterativeSolvers as itersolve

A0 = np.array([[-5.0, 1.0, 2.0, -1.0], [1.0, 3.0, 2.0, 1.0], [2.0, 2.0, -4.0, 1.0], [1.0, 2.0, 1.0, 3.0]])
#b0 = np.array([-4.0, 12.0, 5.0, 6.5])
x0 = np.array([1.0, 1.0, 1.0, 1.0])
x = np.array([2.0, -1.0, 3.0, 2.0])
b0 = np.matmul(A0, x)



print("Gauss-Seidel iteration:")
a = np.array(A0)
b = np.array(b0)
x = np.array(x0)
x, flag, ic, err= itersolve.gauss_seidel_solve(a, b, x, tol=1e-12)
print(x)
print(flag)
print(ic)
print(err)


print("Jacobi iteration:")
a = np.array(A0)
b = np.array(b0)
x = np.array(x0)
x, flag, ic, err= itersolve.jacobi_solve(a, b, x, tol=1e-12)
print(x)
print(flag)
print(ic)
print(err)
