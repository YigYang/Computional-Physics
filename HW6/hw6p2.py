import numpy as np
import math
import matplotlib.pyplot as plt
from numba import njit


@njit
def anneal_numba(L, n_steps, T0, tau, seed=0, periodic=False, early_step=10_000):
    np.random.seed(seed)
    occ = np.zeros((L, L), dtype=np.int8)     
    pi  = -np.ones((L, L), dtype=np.int16)
    pj  = -np.ones((L, L), dtype=np.int16)
    early_pi = -np.ones((L, L), dtype=np.int16)
    early_pj = -np.ones((L, L), dtype=np.int16)
    early_n  = 0
    has_early = 0
    n_dimers = 0
    for step in range(n_steps):
        T = T0 * np.exp(-step / tau)

        i = np.random.randint(L)
        j = np.random.randint(L)
        d = np.random.randint(4)
        i2, j2 = i, j
        if d == 0:     
            j2 = j + 1
        elif d == 1:   
            j2 = j - 1
        elif d == 2:    
            i2 = i + 1
        else:           
            i2 = i - 1

        if periodic:
            i2 %= L
            j2 %= L
        else:
            if i2 < 0 or i2 >= L or j2 < 0 or j2 >= L:
                continue
            
        if occ[i, j] == 0 and occ[i2, j2] == 0:
            occ[i, j] = 1
            occ[i2, j2] = 1
            pi[i, j] = i2
            pj[i, j] = j2
            pi[i2, j2] = i
            pj[i2, j2] = j
            n_dimers += 1
            
        else:
            if occ[i, j] == 1 and pi[i, j] == i2 and pj[i, j] == j2:
                p_acc = 0.0
                if T > 0.0:
                    p_acc = np.exp(-1.0 / T)
                if np.random.random() < p_acc:
                    occ[i, j] = 0
                    occ[i2, j2] = 0
                    pi[i, j] = -1
                    pj[i, j] = -1
                    pi[i2, j2] = -1
                    pj[i2, j2] = -1
                    n_dimers -= 1

        if (step == early_step) and (has_early == 0):
            early_pi[:, :] = pi[:, :]
            early_pj[:, :] = pj[:, :]
            early_n = n_dimers
            has_early = 1

    if has_early == 0:
        early_pi[:, :] = pi[:, :]
        early_pj[:, :] = pj[:, :]
        early_n = n_dimers

    return pi, pj, n_dimers, early_pi, early_pj, early_n


def plot_dimers(ax, pi, pj, title=""):
    L = pi.shape[0]
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-0.5, L - 0.5)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    for i in range(L):
        for j in range(L):
            i2 = int(pi[i, j])
            if i2 < 0:
                continue
            j2 = int(pj[i, j])
            if (i2 > i) or (i2 == i and j2 > j):
                ax.plot([j, j2], [i, i2], color="k", lw=0.8)


def coverage_stats(L, n_dimers):
    max_dimers = (L * L) // 2
    frac = n_dimers / max_dimers
    return max_dimers, frac

if __name__ == "__main__":
    L = 50
    n_steps = 300000
    T0 = 2.0
    tau0 = 10_000
    early_step = 500
    periodic = False
    _ = anneal_numba(10, 10, 1.0, 10.0, seed=0, periodic=False, early_step=5)
    pi_f, pj_f, n_f, pi_e, pj_e, n_e = anneal_numba(
        L=L, n_steps=n_steps, T0=T0, tau=float(tau0), seed=0,
        periodic=periodic, early_step=early_step
    )
    max_dimers, frac_f = coverage_stats(L, n_f)
    _, frac_e = coverage_stats(L, n_e)

    T_early = T0 * math.exp(-early_step / float(tau0))
    T_final = T0 * math.exp(-(n_steps) / float(tau0))

    bc_str = "periodic BC" if periodic else "open BC"
    seed = 0
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    title_left = (
        f"Early snapshot at t={early_step}\n"
        f"N_dimer={n_e}/{max_dimers}  (coverage={frac_e:.3f})"
    )
    plot_dimers(axs[0], pi_e, pj_e, title=title_left)
    title_right = (
        f"Final snapshot at t={n_steps}\n"
        f"N_dimer={n_f}/{max_dimers}  (coverage={frac_f:.3f})"
    )
    plot_dimers(axs[1], pi_f, pj_f, title=title_right)
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.show()
    tau_list = [2_000, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000]
    results = []

    for k, tau in enumerate(tau_list):
        pi, pj, n_dimers, _, _, _ = anneal_numba(
            L=L, n_steps=n_steps, T0=T0, tau=float(tau), seed=100 + k,
            periodic=periodic, early_step=-1
        )
        max_dimers, frac = coverage_stats(L, n_dimers)
        T_end = T0 * math.exp(-n_steps / tau)
        results.append((tau, n_steps, T0, T_end, n_dimers, max_dimers, frac))

    print("\n=== Cooling schedule comparison (T=T0*exp(-t/tau)) ===")
    print("tau     t_steps N_dimer\tN_max\tcoverage")
    for row in results:
        tau, steps, T0_, T_end, Nd, Nmax, frac = row
        print(f"{tau}\t{steps}\t{Nd}\t{Nmax}\t{frac:.4f}")

    taus = [r[0] for r in results]
    covs = [r[-1] for r in results]
    plt.figure(figsize=(6, 4))
    plt.plot(taus, covs, marker="o")
    plt.xlabel("tau")
    plt.ylabel("coverage = Ndimer / (L^2/2)")
    plt.title("Coverage vs cooling time constant tau")
    plt.tight_layout()
    plt.show()
