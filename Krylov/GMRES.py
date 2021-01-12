import numpy as np


def gmres_solve(A: np.array, b: np.array, x: np.array, max_iter: int = 10, restart: int = -1, tol: float = 1e-6):
    # x         - solution vector as np.array
    # flag      - 0 converged to the desired tolerance within max_iter iterations
    #           - 1 iterated to the max_iter but did not converged
    # m         - integer number of iterations between restarts
    
    # ic        - max number of iterations
    # error     - absolute difference0
    flag = 0
    ic = 0
    n = len(A)
    m = max_iter if restart < 0 else restart
    r = b - np.matmul(A, x)
    r_norm = np.linalg.norm(r)
    b_norm = np.linalg.norm(b)
    #rb_norm = 1 / b_norm if b_norm !=0 else 1
    rb_norm = 1
    error = r_norm * rb_norm
    
    if (error <= tol):
        # The initial guess is accurate enough
        flag = 0
        return x, 0, 0, error
    
    # Initialize workspace
    V = np.zeros([n, m + 1])
    H = np.zeros([m + 1, m])
    cs = np.zeros([m])
    sn = np.zeros([m])
    e = np.zeros([m + 1])
    e[0] = 1
    converged = 0
    
    # Iteration
    while ic < max_iter:
        V[:, 0] = r / r_norm    # v
        s = e * r_norm 
        for i in range(m):
            # Arnoldi steps (Gramm-Schmidth process)
            w = np.matmul(A, V[:, i])
            for k in range(i+1):
                H[k, i] = np.dot(w.transpose(), V[:, k])
                w -= H[k, i] * V[:, k]
            H[i + 1, i] = np.linalg.norm(w)
            V[:, i + 1]  = w / H[i + 1, i]
            
            for k in range(i):
                H[k : k+2, i] = __apply_plane_rotations(H[k, i], H[k + 1, i], cs[k], sn[k])
            
            cs[i], sn[i] = __generate_plane_rotations(H[i, i], H[i + 1, i], cs[i], sn[i])
            H[i: i+2, i]= __apply_plane_rotations(H[i, i], H[i + 1, i], cs[i], sn[i])
            s[i], s[i + 1] = __apply_plane_rotations(s[i], s[i + 1], cs[i], sn[i])
            
            error = abs(s[i + 1]) * rb_norm

            if (error <= tol):
                converged = 1
                break
        
        y = __backward_substitution(H[0 : i + 1, 0 : i + 1], s[0 : i + 1])  # i exist outside the loop python feature
        x += np.dot(V[:, 0 : i + 1], y)
        r = b - np.matmul(A, x)
        r_norm = np.linalg.norm(r)
        ic += 1
        
        if (converged == 1):
            break
    
    if (ic + 1 == max_iter):
        flag = 1
    
    return x, flag, ic+1, error
    
    
    
def __generate_plane_rotations(dx, dy, cs, sn):
    if (dy == 0.0):
        cs = 1.0
        sn = 0.0
    elif (abs(dy) > abs(dx)):
        temp = dx / dy
        sn = 1.0 / np.sqrt(1.0 + temp * temp)
        cs = temp * sn
    else:
        temp = dy / dx
        cs = 1.0 / np.sqrt(1.0 + temp * temp)
        sn = temp * cs
    return cs, sn


def __apply_plane_rotations(dx, dy, cs, sn):
    temp = cs * dx + sn * dy
    dy = -sn * dx + cs * dy
    dx = temp
    
    return dx, dy


def __backward_substitution(U: np.array, b: np.array):
	n = len(U)					# Size of the matrix
	x = np.zeros([n])			# Empty array to store the results
	for i in reversed(range(n)):
		s = sum(U[i, j] * x[j] for j in range(i, n))
		x[i] = (b[i] - s) / U[i,i]
		
	return	x
