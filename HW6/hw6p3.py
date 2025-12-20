import numpy as np
import matplotlib.pyplot as plt
from numba import njit

A = 1664525
C = 1013904223
M = 2**32

class LCG:
    def __init__(self, seed=123456789, a=A, c=C, m=M):
        self.a = int(a)
        self.c = int(c)
        self.m = int(m)
        self.state = int(seed) % self.m

    def next_uint(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def random(self):
        return self.next_uint() / self.m 

    def random_array(self, n):
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self.random()
        return out

@njit
def gaussian_from_lcg_box_muller(n, seed, a=A, c=C, m=M):
    out = np.empty(n, dtype=np.float64)

    state = np.uint64(seed)
    aa = np.uint64(a)
    cc = np.uint64(c)
    mm = np.uint64(m)

    denom = float(m) + 1.0
    two_pi = 2.0 * np.pi

    i = 0
    while i < n:
        state = (aa * state + cc) % mm
        u1 = (float(state) + 1.0) / denom
        state = (aa * state + cc) % mm
        u2 = (float(state) + 1.0) / denom
        r = np.sqrt(-2.0 * np.log(u1))
        theta = two_pi * u2
        out[i] = r * np.cos(theta)
        i += 1
        if i < n:
            out[i] = r * np.sin(theta)
            i += 1

    return out, int(state)


def standard_normal_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def power_spectrum_1d(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean() 
    F = np.fft.rfft(x)
    P = np.abs(F) ** 2

    k = np.arange(1, P.size)
    P = P[1:]
    return k, P


def log_bin_spectrum(k, P, n_bins=30):
    k = np.asarray(k)
    P = np.asarray(P)

    edges = np.logspace(np.log10(k.min()), np.log10(k.max()), n_bins + 1)

    kb = []
    Pb = []
    for i in range(n_bins):
        mask = (k >= edges[i]) & (k < edges[i + 1])
        if np.any(mask):
            kb.append(np.exp(np.mean(np.log(k[mask]))))
            Pb.append(np.exp(np.mean(np.log(P[mask]))))

    return np.array(kb), np.array(Pb)


def fit_loglog_slope(kb, Pb, kmin_fit, kmax_fit):
    mask = (kb >= kmin_fit) & (kb <= kmax_fit)
    alpha, b = np.polyfit(np.log(kb[mask]), np.log(Pb[mask]), 1)
    return alpha, b


def main():

    N = 10_000
    g, final_seed = gaussian_from_lcg_box_muller(N, seed=42)

    bins = 80
    xlim = (-5, 5)
    xs = np.linspace(xlim[0], xlim[1], 600)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(g, bins=bins, range=xlim, density=True, alpha=0.7,
            label="LCG (N=10000)")
    ax.plot(xs, standard_normal_pdf(xs), linewidth=2.0, label="Standard Gaussian Probability")
    ax.set_yscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel("Probability density (log scale)")
    ax.set_title("Standard Gaussian Check")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    plt.show()

    k_g, P_g = power_spectrum_1d(g)  
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.loglog(k_g, P_g, linestyle='-', linewidth=0.8, alpha=0.8)
    ax2.set_xlabel("k")
    ax2.set_ylabel("Power P(k)")
    ax2.set_title("Power spectrum of Gaussian variables")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

    x = np.empty(N + 1, dtype=np.float64)
    x[0] = 0.0
    x[1:] = np.cumsum(g)

    i = np.arange(N + 1)

    fig3, ax3 = plt.subplots(figsize=(9, 4.5))
    ax3.plot(i, x, linewidth=1.0)
    ax3.set_xlabel("iteration i")
    ax3.set_ylabel("x(i)")
    ax3.set_title("Random walk constructed from Gaussian variables")
    ax3.grid(True, linestyle="--", linewidth=0.5)
    plt.show()

    x_series = x[1:]                
    k_x, P_x = power_spectrum_1d(x_series)
    
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.loglog(k_x, P_x, linestyle='-', linewidth=0.8, alpha=0.8)
    ax4.set_xlabel("k")
    ax4.set_ylabel("Power P(k)")
    ax4.set_title("Raw power spectrum of random walk")
    ax4.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()


if __name__ == "__main__":
    main()
