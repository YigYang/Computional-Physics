import numpy as np
import matplotlib.pyplot as plt
from numba import njit



@njit
def ising_energy(spins: np.ndarray, J: float = 1.0) -> float:
    L0, L1 = spins.shape
    acc = 0
    for i in range(L0):
        ip = (i + 1) % L0
        for j in range(L1):
            jp = (j + 1) % L1
            s = spins[i, j]
            acc += s * spins[i, jp]   
            acc += s * spins[ip, j]   
    return -J * float(acc)


@njit
def delta_E_flip(spins: np.ndarray, i: int, j: int, J: float = 1.0) -> float:
    L = spins.shape[0]  
    s_old = spins[i, j]
    up    = spins[(i - 1) % L, j]
    down  = spins[(i + 1) % L, j]
    left  = spins[i, (j - 1) % L]
    right = spins[i, (j + 1) % L]
    nn_sum = up + down + left + right
    return 2.0 * J * s_old * nn_sum


def make_log_snapshot_steps(L, n_steps, n_snaps=12, min_sweeps=1):
    steps_per_sweep = L * L
    max_sweeps = max(1, n_steps // steps_per_sweep)
    sweeps = np.unique(
        np.logspace(np.log10(min_sweeps), np.log10(max_sweeps), n_snaps).astype(np.int64)
    )
    snap_steps = sweeps * steps_per_sweep
    snap_steps = snap_steps[snap_steps < n_steps]
    sweeps = sweeps[:snap_steps.size]
    return sweeps, snap_steps



@njit
def metropolis_M_E_and_snapshots(L=20, n_steps=1_000_000, J=1.0, T=1.0,
                                 seed=0, record_stride=1, snap_steps=np.empty(0, np.int64),
                                 record_energy=False):
    np.random.seed(seed)
    N = L * L
    tmp = np.ones(N, np.int8)
    for k in range(N // 2):
        tmp[k] = -1
    for k in range(N - 1, 0, -1):
        r = np.random.randint(0, k + 1)
        tmp[k], tmp[r] = tmp[r], tmp[k]
    spins = np.empty((L, L), np.int8)
    idx = 0
    for i in range(L):
        for j in range(L):
            spins[i, j] = tmp[idx]
            idx += 1
    M = 0
    for i in range(L):
        for j in range(L):
            M += spins[i, j]
    E = ising_energy(spins, J)
    E0 = E
    M0 = M
    n_rec = (n_steps + record_stride - 1) // record_stride
    M_list = np.empty(n_rec, np.int32)
    E_list = np.empty(n_rec, np.float64) if record_energy else np.empty(0, np.float64)
    n_snap = snap_steps.size
    snaps = np.empty((n_snap, L, L), np.int8)
    M_snaps = np.empty(n_snap, np.int32)

    rec_i = 0
    snap_i = 0
    accepted = 0

    for step in range(n_steps):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)

        dE = delta_E_flip(spins, i, j, J)

        if dE <= 0.0 or np.random.random() < np.exp(-dE / T):
            s_old = spins[i, j]
            spins[i, j] = -s_old
            M -= 2 * s_old
            E += dE
            accepted += 1

        if step % record_stride == 0:
            M_list[rec_i] = M
            if record_energy:
                E_list[rec_i] = E
            rec_i += 1

        if snap_i < n_snap and step == snap_steps[snap_i]:
            snaps[snap_i, :, :] = spins
            M_snaps[snap_i] = M
            snap_i += 1

    accept_rate = accepted / n_steps
    return spins, M_list, E_list, snaps, M_snaps, E0, M0, E, M, accept_rate


def plot_snapshots_row(snaps, snap_steps, T, L):
    n = snaps.shape[0]
    plt.figure(figsize=(2.0 * n, 2.2))
    for k in range(n):
        ax = plt.subplot(1, n, k + 1)
        ax.imshow(snaps[k], vmin=-1, vmax=1, cmap="gray", interpolation="nearest")
        ax.set_title(f"step={int(snap_steps[k])}")
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f"Spin snapshots (L={L}, T={T})", y=1.05)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    L = 20
    n_steps = 1000000
    J = 1.0
    sweeps, snap_steps = make_log_snapshot_steps(L, n_steps, n_snaps=12, min_sweeps=1)
    T = 1.0
    seed = np.random.randint(0, 2**31 - 1)
    record_stride = 1 
    spins1, M_t, E_t, snaps1, M_snaps1, E0, M0, E_final, M_final, acc = metropolis_M_E_and_snapshots(
        L=L, n_steps=n_steps, J=J, T=T, seed=seed,
        record_stride=record_stride, snap_steps=snap_steps,
        record_energy=False)
    E_check = ising_energy(spins1, J)
    print("\n========== (a & b) Energ & Magnetization ==========")
    print(f"seed = {seed}, L={L}, J={J}, T={T}")
    print(f"Initial:  E0 = {E0:.1f},  M0 = {int(M0)}")
    print(f"Final:    E_final = {E_final:.1f},  M_final = {int(M_final)}")
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)
    dE_local = delta_E_flip(spins1, i, j, J)

    E_before = ising_energy(spins1, J)
    spins_tmp = spins1.copy()
    spins_tmp[i, j] *= -1
    E_after = ising_energy(spins_tmp, J)
    dE_global = E_after - E_before

    x = np.arange(M_t.size)
    plt.figure()
    plt.plot(x, M_t, linewidth=1.0)
    plt.xlabel("Monte Carlo step")
    plt.ylabel("Magnetization M")
    plt.title(f"M(t): L={L}, J={J}, T={T}")
    plt.tight_layout()
    plt.show()
    plot_snapshots_row(snaps1, snap_steps, T=T, L=L)
    for T in [2.0, 3.0]:
        seedT = np.random.randint(0, 2**31 - 1)
        spinsT, M_tT, E_tT, snapsT, M_snapsT, E0T, M0T, E_finalT, M_finalT, accT = metropolis_M_E_and_snapshots(
            L=L, n_steps=n_steps, J=J, T=T, seed=seedT,
            record_stride=50,         
            snap_steps=snap_steps,
            record_energy=False
        )
        plot_snapshots_row(snapsT,snap_steps, T=T, L=L)
