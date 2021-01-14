import numpy as np
import DualNumber as dn
from Direct.Analytic import anal_solve
from Direct.Gauss import ge_solve, dge_solve
from Direct.LUP import lup_solve2
from Stationary.IterativeSolvers import jacobi_solve, gauss_seidel_solve
from Krylov.GMRES import gmres_solve
from Krylov.BICG import bicg_solve


def solve_sympletic(func, t_span, dt, y0, args, method,
                    max_iter=10, linear_solver=None, max_linear_iter=100, pivot=True, restart=5,
                    atol=1e-6, rtol=1e-6):
    if method == 'MidPoint-Fixed-Point':
        return __midpoint_fixed_iteration(func, y0, t_span, dt, args, max_iter, atol, rtol)
    elif method == 'MidPoint-Newton':
        return __midpoint_newton_iteration(func, y0, t_span, dt, args, max_iter, atol, rtol,
                                           linear_solver, max_linear_iter, pivot, restart)
    else:
        print("solve_sympletic: the required method: " + method + " does not exist.")
        return None


# ----------------- Solver functions ----------------------------------
def __midpoint_newton_iteration(odefun, y0, t_span, dt, args, max_iter, atol, rtol,
                                linear_solver, max_linear_iter, pivot, restart):
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    k = len(t)
    n = len(y0)
    y = np.zeros([n, k])
    y[:, 0] = y0[:]

    # Main loop
    for i in range(1, k):
        y0 = y[0, i - 1] + dt / 2 * np.array(odefun(t[i - 1], y[:, i - 1], *args))      # forward euler step with half step size(predictor)
        y_mid = __newton_step(odefun, y0, y[:, i - 1], t[i - i] + dt / 2, dt / 2, args,
                              max_iter, atol, rtol, linear_solver, max_linear_iter,
                              pivot, restart)                                           # backward euler step with half step
        y[:, i] = 2 * y_mid[:] - y[:, i - 1]                                            # midpoint rule
    return t, y


def __midpoint_fixed_iteration(odefun, y0, t_span, dt, args, max_iter, atol, rtol):
    t = np.arange(t_span[0], t_span[1]+dt, dt)
    k = len(t)
    n = len(y0)
    y = np.zeros([n, k])
    y[:, 0] = y0[:]

    # Main loop
    for i in range(1, k):
        y0 = y[0, i - 1] + dt / 2 * np.array(odefun(t[i - 1], y[:, i - 1], *args))  # forward euler step with halg step size (predictor)
        y_mid = __fixed_point_step(odefun, y0, y[:, i - 1], t[i - 1] + dt / 2, dt / 2, args,
                                   max_iter, atol, rtol)                            # backward euler step with half step
        y[:, i] = 2 * y_mid[:] - y[:, i-1]                                          # midpoint rule
    return t, y


# --------------------- Stepper functions ----------------------------
def __newton_step(odefun, ynew, yold, tm, dt2, args, max_iter, atol, rtol,
                  linear_solver, max_linear_iter, pivot, restart):
    n = len(ynew)
    b = np.zeros(n)
    J = np.zeros([n, n])
    yk = ynew

    yk = yold + dt2 * np.array(odefun(tm, ynew, *args))
    r = ynew - yk
    error = np.linalg.norm(r)
    if error < max(rtol * np.linalg.norm(yk), atol):
        return yk

    for k in range(0, max_iter):
        b = (ynew-yold)
        for z in range(0, n):
            jac = dn.ad_multi_params(odefun, tm, ynew, z, args)
            for x in range(0, n):
                J[x, z] = dt2 * jac[x].dual         # dual part stands for the derivative with respect to zth varible
                if z == 0:
                    b[x] -= dt2 * jac[x].real       # avoid the multiple addition
        J -= np.eye(n)                              # Jacobi of the lin. system.

        # Select linear solver
        if linear_solver == 'analytic':
            dy = anal_solve(J, b)
        elif linear_solver == 'lu-factor':
            dy = lup_solve2(J, b, pivot)
        elif linear_solver == 'gauss':
            dy = ge_solve(J, b, pivot)
        elif linear_solver == 'div-free gauss':
            dy = dge_solve(J, b, pivot)
        elif linear_solver == 'jacobi-iter':
            dy0 = (ynew - yk)
            out = jacobi_solve(J, b, dy0, max_linear_iter, atol, pivot)
            dy = out[0]
        elif linear_solver == 'gauss-seidel-iter':
            dy0 = (ynew - yk)
            out = gauss_seidel_solve(J, b, dy0, max_linear_iter, atol, pivot)
            dy = out[0]
        elif linear_solver == 'gmres':
            dy0 = (ynew - yk)
            out = gmres_solve(J, b, dy0, max_linear_iter, restart, atol)
            dy = out[0]
        elif linear_solver == 'bicg':
            dy0 = (ynew - yk)
            out = bicg_solve(J, b, dy0, max_linear_iter, atol)
            dy = out[0]
        yk += dy

        r = yold + dt2 * np.array(odefun(tm, yk, *args))-yk
        error = np.linalg.norm(r)

        if error < max(rtol * np.linalg.norm(yk), atol):
            # print("Newton iteration converged at k = " + str(k) + " error: " + str(error))
            return yk
        ynew = yk

    return ynew


def __fixed_point_step(odefun, ynew, yold, tm, dt2, args, max_iter, atol, rtol):
    for k in range(0, max_iter):
        yk = yold + dt2 * np.array(odefun(tm, ynew, *args))
        r = ynew - yk
        error = np.linalg.norm(r)
        if error < max(rtol * np.linalg.norm(yk), atol):
            # print("Fixed-point iteration converged at k = " + str(k) + " error: " + str(error))
            return yk
        ynew = yk

    return ynew

