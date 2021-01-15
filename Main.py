import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Solvers import solve_sympletic


# Simple harmonic oscillator example
# Analytical solution
def analytic_solution(amp, freq, tt):
    return amp * np.cos(freq * tt), -amp * freq * np.sin(freq * tt)


# ODE
def hamiltonian_ode(tt, y, mm, kk):
    # y[0] = q, y[1] = p
    dy = [0 for _ in range(2)]
    dy[0] = y[1] / mm
    dy[1] = - kk * y[0]
    return dy


# Parameters
m = 10      # mass
k = 2       # spring stiffness
A = 1.0     # amplitude - x(0)
H = 0.5*k*A**2
# print(H)
t_max = 100.0
dt = 0.1
f = (k / m) ** 0.5  # angular frequency

# Get Analytic solution
t = np.linspace(0, t_max, 100000)
x, v = analytic_solution(A, f, t)
p = v * m

# Numerical solution (python solvers)
method1 = 'RK45'
sol = solve_ivp(hamiltonian_ode, [0, t_max], [A, 0.0], args=(m, k),
                method=method1, atol=1e-6, rtol=1e-6, dense_output=True)

# Numerical solution (sympletic solvers)
method2 = 'MidPoint-Fixed-Point'
t2, y2 = solve_sympletic(hamiltonian_ode, [0, t_max], 1e-2, [A, 0.0, ], args=(m, k),
                         method=method2, atol=1e-6, rtol=1e-6)

method3 = 'MidPoint-Newton'
# linsolve = 'analytic'
#linsolve = 'lu-factor'
# linsolve = 'gauss'
#linsolve = 'div-free gauss'
#linsolve = 'jacobi-iter'
#linsolve = 'gauss-seidel-iter'
#linsolve = 'gmres'
linsolve = 'bicg'
t3, y3 = solve_sympletic(hamiltonian_ode, [0, t_max], 1e-2, [A, 0.0, ], args=(m, k),
                         method=method3, linear_solver=linsolve, atol=1e-6, rtol=1e-6)


plt.figure(1)
plt.plot(t, x, 'k-', linewidth=2, label='analytic')
plt.plot(sol.t, sol.y[0], 'g-', linewidth=1, markersize=2, label='solve_ivp ' + method1)
plt.plot(t2, y2[0], 'b.', linewidth=2, markersize=2, label='solve_sympletic ' + method2)
plt.plot(t3, y3[0], 'r.', linewidth=2, markersize=2, label='solve_sympletic ' + method3 + ' @' + linsolve)
plt.xlabel(r'$t$')
plt.ylabel(r'$x$')
plt.grid('both')
plt.legend()


plt.figure(2)
plt.plot(x, p, 'k-', linewidth=2, label='analytic')
plt.plot(sol.y[0], sol.y[1], 'g-', linewidth=1, markersize=2, label='solve_ivp ' + method1)
plt.plot(y2[0], y2[1], 'b.', linewidth=2, markersize=2, label='solve_sympletic ' + method2)
plt.plot(y3[0], y3[1], 'r.', linewidth=2, markersize=2, label='solve_sympletic ' + method3 + ' @' + linsolve)
plt.xlabel(r'$x$')
plt.ylabel(r'$p$')
plt.grid('both')
plt.legend()

plt.show()
