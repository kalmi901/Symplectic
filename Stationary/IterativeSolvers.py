import numpy as np

def gauss_seidel_solve(A: np.array, b: np.array, x: np.array, max_iter: int = 100, tol: float = 1e-6, pivot = True):
	# x 		- solution vector as an np.array
	# flag		- 0 converged to the desired tolerance within max_iter iterations
	#			- 1 iterated to max_iter but did not converge
	#			- 2 diverge
	
	# ic		- max number of iterations
	# error		- absol
	flag = 0
	ic = 0
	n = len(A)
	r = b - np.matmul(A,x)
	r_norm = np.linalg.norm(r)
	b_norm = np.linalg.norm(b)
	#rb_norm = 1 / b_norm if b_norm != 0 else 1
	rb_norm = 1
	error = r_norm * rb_norm	# tolerance
	
	if (error <= tol):
		# The initial guess is accurate enough
		flag = 0
		return x, 0, 0, error
	
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
			if j != jmax:
				# Swap the rows
				A[[j, jmax]] = A[[jmax, j]]
				b[[j, jmax]] = b[[jmax, j]]	
	
	for k in range(max_iter):
		for i in range(n):
			#x[i] = b[i]
			sum = 0
			for j in range(n):
				if j != i:
					#x[i] -= A[i, j] * x[j]
					sum += A[i, j] * x[j]
			#x[i] /= A[i, i]
			x[i] = (b[i] - sum) / A[i, i]
		r_norm = np.linalg.norm(b - np.matmul(A,x))
		error = r_norm * rb_norm
		
		if (error <= tol):
			return x, 0, k+1, error
			
		if (error >= 1e6):
			return x, 2, k+1, error
			
	return x, 1, max_iter, error

	
def jacobi_solve(A: np.array, b: np.array, x: np.array, max_iter: int = 100, tol: float = 1e-6, pivot = True):
	# x 		- solution vector as an np.array
	# flag		- 0 converged to the desired tolerance within max_iter iterations
	#			- 1 iterated to max_iter but did not converge
	#			- 2 diverge
	
	# ic		- max number of iterations
	# error		- absol
	flag = 0
	ic = 0
	n = len(A)
	r = b - np.matmul(A,x)
	r_norm = np.linalg.norm(r)
	b_norm = np.linalg.norm(b)
	#rb_norm = 1 / b_norm if b_norm != 0 else 1
	rb_norm = 1
	error = r_norm * rb_norm	# tolerance
	
	if (error <= tol):
		# The initial guess is accurate enough
		flag = 0
		return x, 0, 0, error
	
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
			if j != jmax:
				# Swap the rows
				A[[j, jmax]] = A[[jmax, j]]
				b[[j, jmax]] = b[[jmax, j]]	
	
	xold = np.array(x)
	for k in range(max_iter):
		for i in range(n):
			xold[i] = x[i]
			sum = 0
			for j in range(n):
				if j != i:
					sum += A[i, j] * xold[j]
			x[i] = (b[i] - sum) / A[i, i]
		r_norm = np.linalg.norm(b - np.matmul(A,x))
		error = r_norm * rb_norm
		
		if (error <= tol):
			return x, 0, k+1, error
			
		if (error >= 1e6):
			return x, 2, k+1, error
			
	return x, 1, max_iter, error