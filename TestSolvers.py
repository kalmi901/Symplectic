import numpy as np
import scipy.linalg
from Direct.LUP import lup_solve, lup_solve2
from Direct.Gauss import ge_solve, dge_solve
from Direct.Analytic import anal_solve
from Stationary.IterativeSolvers import gauss_seidel_solve, jacobi_solve
from Krylov.GMRES import gmres_solve
import os
import time


os.system('cls')

def magic(n):
  n = int(n)
  if n < 3:
    raise ValueError("Size must be at least 3")
  if n % 2 == 1:
    p = np.arange(1, n+1)
    return n*np.mod(p[:, None] + p - (n+3)//2, n) + np.mod(p[:, None] + 2*p-2, n) + 1
  elif n % 4 == 0:
    J = np.mod(np.arange(1, n+1), 4) // 2
    K = J[:, None] == J
    M = np.arange(1, n*n+1, n)[:, None] + np.arange(n)
    M[K] = n*n + 1 - M[K]
  else:
    p = n//2
    M = magic(p)
    M = np.block([[M, M+2*p*p], [M+3*p*p, M+p*p]])
    i = np.arange(p)
    k = (n-2)//4
    j = np.concatenate((np.arange(k), np.arange(n-k+1, n)))
    M[np.ix_(np.concatenate((i, i+p)), j)] = M[np.ix_(np.concatenate((i+p, i)), j)]
    M[np.ix_([k, k+p], [0, k])] = M[np.ix_([k+p, k], [0, k])]
  return M 


A0 = np.array([[8, 1, 6], [3, 5, 7], [4, 9, 2]])
b0 = np.array([1, 23, 3])

#A = np.array([[2, 3], [4, -1]])
#b = np.array([1, 23])
#print(lup_solve(A, b))			# [5, -3]

size = 2
A0 = np.random.rand(size,size)
#A0 = np.array(magic(size), dtype=float)
#print(A)
x = np.random.rand(size)
b0 = np.matmul(A0, x)
#b0 = np.array([1.0, 1.0, 1.0])
#x0 = np.ones(size)

#Pp, Lp, Up = scipy.linalg.lu(A)
#L, U, P = lup_factorization(A) 


	
#print("A:")
#print(A)
#print("b:")
#print(b)

#print ("P: Python")
#print(Pp)
#print ("P: Own")
#print(P)
#print ("L: Python")
#print(Lp)
#print ("L: Own")
#print(L)
#print ("U: Python")
#print(Up)
#print ("U: Own")
#print(U)

print("Initial x")
print(x)
print("\nLUP solver:")
a = np.array(A0)
b = np.array(b0)
#print("Input matrix:")
#print(a)
print("Result:")
print(lup_solve(a,b))

print("\nLUP solver2:")
a = np.array(A0)
b = np.array(b0)
#print("Input matrix:")
#print(a)
print("Result:")
print(lup_solve2(a,b))

print("\nGauss elimination:")
a = np.array(A0)
b = np.array(b0)
#print("Input matrix:")
#print(a)
print("Result:")
print(ge_solve(a, b))

print("\nDivison-FreeGauss elimination:")
a = np.array(A0)
b = np.array(b0)
#print("Input matrix:")
#print(a)
print("Result:")
print(dge_solve(a, b))

print("\nAnalitic solver:")
a = np.array(A0)
b = np.array(b0)
#print("Input matirx:")
#print(a)
print("Result:")
print(anal_solve(a, b))

print("\nGauss-Seidel iteration:")
a = np.array(A0)
b = np.array(b0)
x = np.ones([size])
print(gauss_seidel_solve(a, b, x, size*100))

print("\nJacobi iteration:")
a = np.array(A0)
b = np.array(b0)
x = np.ones([size])
print(jacobi_solve(a, b, x, size*100))

print("\nGMRES(5):")
a = np.array(A0)
b = np.array(b0)
x = np.ones([size])
print(gmres_solve(a, b, x, size*100, 5))

print("\nPython Solver:")
a = np.array(A0)
b = np.array(b0)
#print("Input matrix:")
#print(a)
print("Result:")
print(np.linalg.solve(a, b))



