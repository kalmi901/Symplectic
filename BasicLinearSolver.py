import numpy as np


def jacobi(a: np.array, b: np.array, max_iteration: int, tolerance: float):
    n = len(b)
    x = np.ones(n)
    y = np.zeros(n)
    for k in range(0, max_iteration):
        for i in range(0, n):
            y[i] = b[i]
            for j in range(0, n):
                if j != i:
                    y[i] -= a[i][j]*x[j]
            x[i] = y[i]/a[i][i]
    return x


def gauss(a: np.array, b: np.array, max_iteration: int, tolerance: float):
    n = len(b)
    x = np.ones(n)
    for k in range(0, max_iteration):
        for i in range(0, n):
            x[i] = b[i]
            for j in range(0, n):
                if j != i:
                    x[i] -= a[i][j]*x[j]
            x[i] /= a[i][i]
    return x

# TODO:  -> tolerance