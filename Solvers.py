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
        b = (ynew-yold)     # Mit csinál??
        for z in range(0, n):
            jac = dn.ad_multi_params(f, tm, ynew, z, args)
            for x in range(0, n):
                J[x, z] = dt2 * jac[x].dual         # dual part stands for the derivative with respect to zth varible
                if z == 0:
                    b[x] -= dt2 * jac[x].real       # avoid the multiple addition
        J -= np.eye(n)                              # Jacobi of the lin. system.


        dy0 = (ynew-yold)
        # dy = Krylov.gmres(J, dy0, b, max_iter=2, tol=rtol)
        dy = linsolve.gauss(J, b, 50, rtol)
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

