import numpy as np

# ----------------------- LUP solver - basic ----------------------------------------
def lup_solve(A: np.array, b: np.array, pivot=True):
	# Solves the A*x = b equation by means of LU factorization with pivoting
	# A is the coefficient matrix (NxN)
	# x is the unkown vector		(Nx1)
	# b is the rhs vector 			(Nx1)
	
	# A*x = b --> P*A*x = P*b
	# P is the permutation patrix
	
	# 1. ) Create the P pertmuation matrix, and make the PA=LU factorization
	# 2. ) Solves L*y = P*b by means of forward substitution for y = U*x
	# 3. ) Solves U*x = y by means of backward substitution for x
	
	L, U, P = lup_factorization(A, pivot)
	Pb = np.matmul(P, b)

	y = __forward_substitution(L, Pb)
	x = __backward_substitution(U, y)
	
	return x
	

def __forward_substitution(L: np.array, b: np.array):
	n = len(L)					# Size of the matrix
	x = np.zeros([n])			# Empty array to store the results
	for i in range(n):
		s = sum(L[i,j] * x[j] for j in range(i))
		x[i] = (b[i] - s) / L[i,i]
	
	return x


def __backward_substitution(U: np.array, b: np.array):
	n = len(U)					# Size of the matrix
	x = np.zeros([n])			# Empty array to store the results
	for i in reversed(range(n)):
		s = sum(U[i, j] * x[j] for j in range(i, n))
		x[i] = (b[i] - s) / U[i,i]
		
	return	x


def lup_factorization(A: np.array, pivot=True):
	# Create the pivot matrix for A, used in Doolittle's method
	n = len(A)
	# Create an identity matrix, with floating point values. The output is stored in row-major (C-style) order
	P = np.eye(n, dtype=float, order='C')
	PA = np.array(A)
	# Rearrange the matrix such that the largest element of                                                                                                                                                                                   
    # each column of A is placed on the diagonal of of A
	
	if (pivot==True):
		for j in range(n):
			maxA = 0.0
			jmax = j
			for k in range(j, n):
				absA = abs(A[k, j])
				if (absA > maxA):
					maxA = absA
					jmax = k
			#print(maxA)
			#print(jmax)
			#print(j)
			#input()
			if j != jmax:
				# Swap the rows
				P[[j, jmax]] = P[[jmax, j]]
				PA[[j, jmax]] = PA[[jmax, j]]
				#print(P)
				#input()
	
	
	# PA = LU factorization
	# A Matrix multiplied by the Pivor Matrix
	# PA = np.matmul(P,A)

	# Initialize tero matrices for L (lower triangular) and U (upper triangular)
	L = np.zeros([n,n])
	U = np.zeros([n,n])
	
	# Decomposition of the Matrix
	for j in range(n):
		# All diagonal entrief of L are set to unity
		L[j,j] = 1.0
			
		# LaTeX: u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik} 
		for i in range(j+1):
			s1 = sum(U[k,j] * L[i,k] for k in range(i))
			U[i,j] = PA[i,j] - s1
	
		# LaTeX: l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik} )
		for i in range(j, n):
			s2 = sum(U[k,j] * L[i,k] for k in range(j))
			L[i,j] = (PA[i,j] - s2) / U[j,j]
		
	return L, U, P	

	
# ---------------------- LUP solver - storage friendly -------------------------------
def lup_solve2(A: np.array, b: np.array, pivot=True):
	# Solves the A*x = b equation by means of LU factorization with pivoting
	# A is the coefficient matrix   (NxN)
	# x is the unkown vector		(Nx1)
	# b is the rhs vector 			(Nx1)
	# A*x = b --> P*A*x = P*b
	# P is the permutation "vector"
	
	A, P = lup_factorization2(A)
	x = __substitution(A, P, b)
	
	return x

	
def lup_factorization2(A: np.array, pivot=True):
	# The matrix A is modified to contanin both matrices L-E and U as A=(L-E)+U such that P*A = L*U
	# The permutation matrix is not stored as a matrix, but in an integer vector P of size N+1
	# containing colum indexes where the permutation matrix has "1". 
	
	n = len(A)
	P = np.zeros([n], dtype=int)
	for i in range(n):
		P[i] = i			# Unit permutation matrix 
		
	# Rearrange matrix A such that the largest element of each column of A is placed on the diagonal
	if (pivot == True):
		for j in range(n):
			maxA = 0.0
			jmax = j
			for k in range(j, n):
				absA = abs(A[k, j])
				if (absA > maxA):
					maxA = absA
					jmax = k
			if j != jmax:
				# Swap the rows
				P[[j, jmax]] = P[[jmax, j]]
				A[[j, jmax]] = A[[jmax, j]]

	# Decomposition of the Matrix
	for i in range(n):
		for j in range(i):
			alpha = A[i, j]
			for k in range(j):
				alpha -= A[i,k] * A[k,j]
			A[i,j] = alpha/A[j,j]
		for j in range(i,n):
			alpha = A[i,j]
			for k in range(i):
				alpha -= A[i,k] * A[k,j]
			A[i,j] = alpha
	return A, P
	
	
def __substitution(A, P, b):
	n = len(A)
	x = np.zeros([n])
	for i in range(n): 
		x[i] = b[P[i]]
		for k in range(i):
			x[i] -= A[i,k] * x[k]
	for i in reversed(range(n)):
		for k in range(i+1, n):
			x[i] -= A[i,k] * x[k]
		x[i] /= A[i,i]
	return x