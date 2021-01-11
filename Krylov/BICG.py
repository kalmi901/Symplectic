import numpy as np


def bicg(A: np.array, x: np.array, b: np.array, max_iter: int = 100, tol: float = 1e-6):
    flag = 0

    # Initialization
    n = len(b)

    r = b - A.dot(x)
    rtilde = r
    p = r
    ptilde = rtilde

    for i in range(0, max_iter):
        rho1 = np.dot(r, rtilde)
        if rho1 == 0:
            break

        if i != 0:
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

        if np.linalg.norm(r) <= tol:
            break
    return x
