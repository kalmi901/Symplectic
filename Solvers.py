import numpy as np
import DualNumber as dn
import BasicLinearSolver as linsolve
import Krylov.GMRES as Krylov


def solve_sympletic(func, t_span, dt, y0, args, method,
                    max_iter=10, liner_solver='Gauss', max_linear_iter=100,
                    atol=1e-6, rtol=1e-6):
    # TODO: max_iter, linear_solver, max_linear_iter, atol, rtol are unused...
    if method == 'MidPoint-Fixed-Point':
        return __midpoint_fixed_iteration(func, y0, t_span, dt, args, max_iter, atol, rtol)
    elif method == 'MidPoint_Newton':
        return __midpoint_newton_iteration(func, y0, t_span, dt, args, max_iter, atol, rtol)
    else:
        print("solve_sympletic: the required method: " + method + " does not exist.")
        return None


def __midpoint_newton_iteration(func, y0, t_span, dt, args, max_iter, atol, rtol):
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    k = len(t)
    n = len(y0)
    y = np.zeros([n, k])
    y[:, 0] = y0[:]

    # Main loop
    for i in range(1, k):
        y0 = y[0, i - 1] + dt / 2 * np.array(func(t[i - 1], y[:, i - 1], *args))        # forward euler step with half step size(predictor)
        y_mid = __newton_step(func, y0, y[:, i - 1], t[i - i] + dt / 2, dt / 2, args,
                              max_iter, atol, rtol)   # backward euler step with half step
        y[:, i] = 2 * y_mid[:] - y[:, i - 1]                                            # midpoint rule
    return t, y


def __midpoint_fixed_iteration(func, y0, t_span, dt, args, max_iter, atol, rtol):
    t = np.arange(t_span[0], t_span[1]+dt, dt)
    k = len(t)
    n = len(y0)
    y = np.zeros([n, k])
    y[:, 0] = y0[:]

    # Main loop
    for i in range(1, k):
        y0 = y[0, i - 1] + dt / 2 * np.array(func(t[i - 1], y[:, i - 1], *args))    # forward euler step with halg step size (predictor)
        y_mid = __fixed_point_step(func, y0, y[:, i-1], t[i-1]+dt/2, dt/2, args,
                                   max_iter, atol, rtol)                            # backward euler step with half step
        y[:, i] = 2 * y_mid[:] - y[:, i-1]                                          # midpoint rule
    return t, y


# Magic Happens here
def __newton_step(f, ynew, yold, tm, dt2, args, max_iter, atol, rtol):
    n = len(ynew)
    b = np.zeros(n)
    J = np.zeros([n, n])
    for newtonIter in range(0, max_iter):
        b = (ynew-yold)
        for z in range(0, n):
            jac = dn.ad_multi_params(f, tm, ynew, z, args)
            for x in range(0, n):
                J[x, z] = dt2 * jac[x].dual         # dual part stands for the derivative with respect to zth varible
                if z == 0:
                    b[x] -= dt2 * jac[x].real       # avoid the multiple addition
        J -= np.eye(n)                              # Jacobi of the lin. system.

        """
        # At this point a system of linear eq. has to be solved j*dy = b
        # Krylov (Unpreconditioned restared GMRES(m) is implemented, m is consant 4  )
        # Initialization
        dy = ynew-yold
        r = b - J.dot(dy)
        m = 4
        norm_r0 = np.linalg.norm(r)
        v = r / norm_r0
        V = np.zeros([m+1, n])
        V[0] = v
        H = np.zeros([m+1, m])
        sn = np.zeros(m)
        cs = np.zeros(m)
        beta = np.zeros(m+1)
        beta[0] = norm_r0
        b_norm = np.linalg.norm(b)
        b_norm = 1 if b_norm == 0 else b_norm
        error = np.linalg.norm(r) / b_norm
        if error < rtol:
            ynew += dy
            continue

        converged = 0
        for iterationCounter in range(0, max_iter):
            for k in range(0, m):
                # Arnoldi steps (Gramm-Schmidth)
                w = J.dot(V[k])
                for j in range(0, k+1):
                    H[j][k] = V[j].dot(w)
                    w -= H[j][k] * V[j]
                H[k+1][k] = np.linalg.norm(w)
                V[k+1] = w / H[k+1][k]

                for i in range(0, k):
                    temp = cs[i] * H[i][k] + sn[i] * H[i + 1][k]
                    H[i + 1][k] = -sn[i] * H[i][k] + cs[i] * H[i + 1][k]
                    H[i][k] = temp

                t = np.sqrt(H[k][k] ** 2 + H[k + 1][k] ** 2)
                cs[k] = H[k][k] / t
                sn[k] = H[k + 1][k] / t

                beta[k + 1] = -sn[k] * beta[k]
                beta[k] = cs[k] * beta[k]
                error = abs(beta[k + 1]) / b_norm

                if error < rtol:
                    converged = 1
                    break
            if converged == 1:
                break
        # Convergece achieved or max_iteration reached
        # Extract results
        out = np.zeros(n)
        for i in range(0, n):
            row = n - 1 - i
            sum = 0
            for j in range(0, i):
                col = n - 1 - j
                sum += H[row][col] * out[col]
            out[row] = (beta[row] - sum) / H[row][row]

        dy += (V[0:n].transpose().dot(out))
        """
        dy0 = (ynew-yold)
        dy = Krylov.gmres(J, dy0, b, max_iter=10, tol=rtol)
        # dy = linsolve.gauss(J, b, 50, rtol)
        ynew += dy
        # TODO: Implement Jacobian-Free Newton Iteration
    return ynew


# Magic happens here
def __fixed_point_step(f, ynew, yold, tm, dt2, args, max_iter, atol, rtol):
    ynext = ynew
    for i in range(0, max_iter):
        ynew = yold + dt2 * np.array(f(tm, ynext, *args))
        ynext = ynew

    # TODO: Check convergence
    return ynew

