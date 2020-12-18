import numpy as np


def gmres(A: np.array, x: np.array, b: np.array, max_iter: int = 10, restart: int = -1, tol: float = 1e-6):
    flag = 0

    # Initialization --------------------------
    n = len(b)
    if restart == -1:
        m = max_iter
    else:
        m = restart

    r = b - A.dot(x)
    normr0 = np.linalg.norm(r)
    v = r / normr0
    V = np.zeros([m + 1, n])
    V[0] = v
    H = np.zeros([m + 1, m])
    sn = np.zeros(m)
    cs = np.zeros(m)
    beta = np.zeros(m + 1)
    beta[0] = normr0  # beta*e1
    bnorm = np.linalg.norm(b)
    if (bnorm == 0):
        bnorm = 1

    e = []
    e.append(np.linalg.norm(r) / bnorm)

    if e[0] <= tol:
        return 0

    k = 0
    while k < max_iter:

        # Arnoldi steps (using Gramm-Schmidth process) ----------------
        w = A.dot(V[k])
        for j in range(0, k + 1):
            H[j][k] = V[j].dot(w)
            w -= H[j][k] * V[j]
        H[k + 1][k] = np.linalg.norm(w)
        V[k + 1] = w / H[k + 1][k]

        # End of Arnoldi steps
        # print(H)
        # print(V)
        # Eliminate the last element in H ith row and update the rotation matrix
        for i in range(0, k):
            temp = cs[i] * H[i][k] + sn[i] * H[i + 1][k]
            H[i + 1][k] = -sn[i] * H[i][k] + cs[i] * H[i + 1][k]
            H[i][k] = temp

        # update the next sin cos values for rotation
        t = np.sqrt(H[k][k] ** 2 + H[k + 1][k] ** 2)
        cs[k] = H[k][k] / t
        sn[k] = H[k + 1][k] / t

        if max_iter == 1:
            break

        # eliminate H(i + 1, i)
        H[k][k] = cs[k] * H[k][k] + sn[k] * H[k + 1][k]
        H[k + 1][k] = 0.0
        # print(H)
        # print(V)
        # input()
        # update the residual vector
        beta[k + 1] = -sn[k] * beta[k]
        beta[k] = cs[k] * beta[k]
        error = abs(beta[k + 1]) / bnorm
        # print(beta)
        # print(error)
        e.append(error)
        if error <= tol:
            break
        k += 1
    # End of while

    # calculate the results
    # print(H)
    y = np.zeros(n)
    for i in range(0, n):
        row = n - 1 - i
        sum = 0
        # print(row)
        for j in range(0, i):
            col = n - 1 - j
            sum += H[row][col] * y[col]
        y[row] = (beta[row] - sum) / H[row][row]

    x = np.array([float(x[i]) for i in range(0, x.size)])
    x += (V[0:n].transpose().dot(y))
    # print(x)
    return x

