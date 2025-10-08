import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

datatable = Path("smf_cosmos.dat")
data = np.loadtxt(datatable)
log10M = data[:, 0]
n_obs = data[:, 1]
err = data[:, 2]

def numerical_gradient(func, x, eps=1e-6):
    
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):   
        dx = np.zeros_like(x, dtype=float)
        dx[i] = eps
        grad[i] = (func(x + dx) - func(x - dx)) / (2.0 * eps)
    return grad

def gradient_descent_path(func, x0, max_iter, tol, gamma):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(max_iter):
        g = numerical_gradient(func, x)
        if np.linalg.norm(g) < tol:
            break
        x = x - gamma * g
        path.append(x.copy())
    return np.array(path)

def f_simple(v):
    x, y = v
    return (x - 2.0)**2 + (y - 2.0)**2

x0 = np.array([5.0, -3.0])

path = gradient_descent_path(f_simple, x0, max_iter=1000000,tol=1e-12,gamma=0.001)

xx = np.linspace(-4, 8, 160)
yy = np.linspace(-4, 8, 160)
XX, YY = np.meshgrid(xx, yy)
ZZ = (XX - 2.0)**2 + (YY - 2.0)**2

plt.figure(figsize=(6.0, 6.0))

CS = plt.contour(XX, YY, ZZ, levels=10)
plt.clabel(CS, inline=1, fontsize=8)

plt.plot(path[:, 0], path[:, 1], lw=1.2, ms=3.5, label="descent path")


plt.scatter([x0[0]], [x0[1]], c='r', s=40, label='start point')
plt.scatter([2.0], [2.0], c='g', s=60, marker='*', label='min point (2,2)')

plt.title("Gradient Descent Path on $f(x,y) = (x-2)^2 + (y-2)^2$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()

x_final = path[-1]                 
f_final = f_simple(x_final)        
steps = len(path)  

print(f"minimum point: x = {x_final[0]}, y = {x_final[1]}")
print(f"minimum of f(x) = {f_final}")
print(f"iterations: {steps}")

def gradient(func, x0, max_iter, tol, gamma):
    x = np.array(x0, dtype=float)
    f_seq = []
    for _ in range(max_iter):
        f = func(x)
        f_seq.append(f)
        g = numerical_gradient(func, x)
        if np.linalg.norm(g) < tol:
            break
        x = x - gamma * g
    return x, f_seq

def schechter(log10M, phi_star, M_star, alpha):
    M = np.power(10.0, log10M)
    ln_n = (np.log(np.log(10.0))
            + np.log(phi_star)
            + (alpha + 1.0) * (log10M * np.log(10.0) - np.log(M_star))
            - (M / M_star))
    ln_n = np.clip(ln_n, a_min=-745.0, a_max=700.0)
    return np.exp(ln_n)

def chi2_u(u):
    phi_star = np.exp(u[0])
    M_star   = np.exp(u[1])
    alpha    = u[2]
    model = schechter(log10M, phi_star, M_star, alpha)
    return np.sum(((n_obs - model) / err)**2)

inits = [np.array([np.log(4e-3), np.log(10**10.6), -1.3]),
         np.array([np.log(3e-3), np.log(10**10.7), -1.25]),
         np.array([np.log(5e-3), np.log(10**10.8), -1.2])]

solutions = []
chi2_logs = []   

print("\n===== Calculate chi^2 from different initial parameters=====")

for u0 in inits:
    print(f"start from initial parameters = {u0}")
    u_star, f_seq = gradient(chi2_u, u0, max_iter=100000, tol=1e-8, gamma=0.0001)
    solutions.append(u_star)
    chi2_logs.append(f_seq)
    print("  final chi^2 =", f_seq[-1])

final_vals = [seq[-1] for seq in chi2_logs]
best_idx   = int(np.argmin(final_vals))
u_best     = solutions[best_idx]
phi_best, Mstar_best, alpha_best = np.exp(u_best[0]), np.exp(u_best[1]), u_best[2]

print("\n===== Fitting paremeters for  Schechter function=====")
print(f"phi*   = {phi_best} ")
print(f"M*     = {Mstar_best}  ")
print(f"alpha  = {alpha_best:}")
print(f"chi^2  = {final_vals[best_idx]}")

chi2_seq = chi2_logs[best_idx]

plt.figure()
plt.plot(np.arange(len(chi2_seq)), chi2_seq)
plt.xlabel("iteration (step i)")
plt.ylabel(r"$\chi^2$")
plt.title("Chi^2 ~ iteration")
plt.tight_layout()
plt.show()


log10M_grid = np.linspace(log10M.min(), log10M.max(), 400)
model_best  = schechter(log10M_grid, phi_best, Mstar_best, alpha_best)

plt.figure()
plt.errorbar(log10M, n_obs, yerr=err, fmt='o', capsize=3, lw=1, label="data")
plt.plot(log10M_grid, model_best, lw=2, label="best-fit")
plt.yscale("log")
plt.xlabel(r"$\log_{10} M_{\mathrm{gal}}$")
plt.ylabel(r"$n(M_{\mathrm{gal}})$")
plt.title(" Best-fit Schechter function to the data (log-log)")
plt.legend()
plt.tight_layout()
plt.show()
