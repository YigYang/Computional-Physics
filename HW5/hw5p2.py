import numpy as np
import matplotlib.pyplot as plt;'L'
from numba import njit

M = 100          
L = 100.0       
h = L / M        

particles = np.loadtxt("particles.dat")
x_part = particles[:, 0]
y_part = particles[:, 1]

rho = np.zeros((M, M), dtype=float)

q = -1.0   

for x, y in zip(x_part, y_part):
    gx = x / h - 0.5
    gy = y / h - 0.5

    i0 = int(np.floor(gx))
    j0 = int(np.floor(gy))

    if i0 < 0 or j0 < 0 or i0 >= M or j0 >= M:
        continue

    dx = gx - i0
    dy = gy - j0

    w00 = (1.0 - dx) * (1.0 - dy)
    w10 = dx * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w11 = dx * dy

    rho[i0, j0] += q * w00
    if i0 + 1 < M:
        rho[i0 + 1, j0] += q * w10
    if j0 + 1 < M:
        rho[i0, j0 + 1] += q * w01
    if (i0 + 1 < M) and (j0 + 1 < M):
        rho[i0 + 1, j0 + 1] += q * w11

rho /= h**2  

plt.figure(figsize=(6, 5))
extent = [0, L, 0, L]  
im = plt.imshow(rho.T, origin="lower", extent=extent)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Charge density field")
plt.colorbar(im, label=r"$\rho$")
plt.tight_layout()
plt.show()

print("Total charge in grid ~", rho.sum() * h**2)

eps0 = 1.0       
phi = np.zeros((M, M), dtype=float)  
tol = 1.0e-10    
max_iter = 200000

for it in range(max_iter):
    phi_new = phi.copy()
    phi_new[1:-1, 1:-1] = 0.25 * (
        phi[2:, 1:-1]   +   # i+1, j
        phi[:-2, 1:-1]  +   # i-1, j
        phi[1:-1, 2:]   +   # i, j+1
        phi[1:-1, :-2]  +   # i, j-1
        (h**2) * rho[1:-1, 1:-1] / eps0)
    diffmax = np.max(np.abs(phi_new - phi))
    phi = phi_new

    if diffmax < tol:
        print(f"Converged after {it+1} iterations, max diff = {diffmax:.3e}")
        break
else:
    print(f"Did not reach tolerance after {max_iter} iterations, "
          f"last max diff = {diffmax:.3e}")
plt.figure(figsize=(6, 5))
extent = [0, L, 0, L]
im = plt.imshow(phi.T, origin="lower", extent=extent)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Potential distribution$\\phi(x,y)$ (Standard relaxation)")
plt.colorbar(im, label=r"$\phi$")
plt.tight_layout()
plt.show()

@njit
def solve_poisson_sor(rho, h, eps0, omega, tol, max_iter):
    nx, ny = rho.shape
    phi = np.zeros((nx, ny), dtype=np.float64)

    for it in range(max_iter):
        diffmax = 0.0

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                phi_old = phi[i, j]

                phi_gs = 0.25 * (
                    phi[i+1, j] + phi[i-1, j] +
                    phi[i, j+1] + phi[i, j-1] +
                    (h*h) * rho[i, j] / eps0)

                phi_new = (1.0 - omega) * phi_old + omega * phi_gs
                phi[i, j] = phi_new

                d = abs(phi_new - phi_old)
                if d > diffmax:
                    diffmax = d

        if diffmax < tol:
            return it + 1, phi

    return max_iter, phi


def objective(omega, rho, h, eps0, tol, max_iter):
    n_iter, _ = solve_poisson_sor(rho, h, eps0, omega, tol, max_iter)
    print(f"omega = {omega:.4f}, iterations = {n_iter}")
    return n_iter

omega_left = 1.0
omega_right = 2.0
omega_tol = 1e-3
gr = (np.sqrt(5.0) - 1.0) / 2.0

max_iter_sor = 100000

c = omega_right - gr * (omega_right - omega_left)
d = omega_left + gr * (omega_right - omega_left)

fc = objective(c, rho, h, eps0, tol, max_iter_sor)
fd = objective(d, rho, h, eps0, tol, max_iter_sor)

best_omegas = []
step = 0

while (omega_right - omega_left) > omega_tol and step < 50:
    step += 1

    if fc < fd:
        omega_right, d, fd = d, c, fc
        c = omega_right - gr * (omega_right - omega_left)
        fc = objective(c, rho, h, eps0, tol, max_iter_sor)
    else:
        omega_left, c, fc = c, d, fd
        d = omega_left + gr * (omega_right - omega_left)
        fd = objective(d, rho, h, eps0, tol, max_iter_sor)

    if fc < fd:
        omega_best = c
    else:
        omega_best = d
    best_omegas.append(omega_best)
    # print(f"Step {step}: best omega ~ {omega_best:.5f}, "
    #       f"interval = [{omega_left:.5f}, {omega_right:.5f}]")
    
if fc < fd:
    omega_opt = c
    f_opt = fc
else:
    omega_opt = d
    f_opt = fd

print(f"Optimal omega = {omega_opt:.5f}, "
      f"iterations = {f_opt}")


n_iter_opt, phi_sor = solve_poisson_sor(rho, h, eps0, omega_opt, tol, max_iter_sor)
print(f"Using omega = {omega_opt:.5f}, converged in {n_iter_opt} iterations")

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(best_omegas) + 1), best_omegas, marker="o")
plt.xlabel("Golden-search step")
plt.ylabel(r"Best $\omega$")
plt.title(r"Evolution of optimal $\omega$ in golden-ratio search")
plt.grid(True)
plt.tight_layout()
plt.show()
