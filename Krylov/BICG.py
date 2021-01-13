import numpy as np


def bicg_solve(A: np.array, b: np.array, x: np.array, max_iter: int = 100, tol: float = 1e-6):
    # x         - solution vector as np.array
    # flag      - 0 converged to the desired tolerance within max_iter iterations
    #           - 1 iterated to the max_iter but did not converged
    #           --1 breakdown (method failure) 
    # m         - integer number of iterations between restarts
    
    # ic        - max number of iterations
    # error     - absolute difference   
    flag = 0

    # Initialization
    n = len(A)
    r = b - np.matmul(A, x)
    r_norm = np.linalg.norm(r)
    b_norm = np.linalg.norm(r)
    #rb_norm = 1/ b_norm if b_norm !=0 else 1
    rb_norm = 1
    error = r_norm * rb_norm
    
    if (error <= tol):
        # The initial guess is accurate enough
        return x, 0, 0, error
    rtilde = r
    p = r
    ptilde = rtilde

    for ic in range(max_iter):
        rho1 = np.dot(r, rtilde)
        if rho1 == 0:
            # Method failure
            return x, -1, ic+1, error

        if ic != 0:
            beta = rho1/rho2
            p = r + beta * p
            ptilde = rtilde + beta * ptilde

        q = A.dot(p)
        qtilde = A.transpose().dot(ptilde)

        alpha = rho1 / np.dot(ptilde, q)
        x += alpha * p
        r = r - alpha * q
        rtilde = rtilde - alpha * qtilde

        rho2 = rho1
        error = np.linalg.norm(r) * rb_norm
        if error <= tol:
            return x, 0, ic+1, error
            
    # No convergence, return the actual state
    return x, 1, ic+1, error 
