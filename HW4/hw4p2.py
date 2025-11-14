import numpy as np
import math
import matplotlib.pyplot as plt


G_DIMLESS = 1.0
M_BH_DIMLESS = 1.0
MU = G_DIMLESS * M_BH_DIMLESS / 4.0 

R_S = 1.0e-7          
D_TOL = 1.0e-8    
TIME_UNIT_MYR = 1.459429602



def acceleration_total(r_vec, v_vec, A_df=0.0, B_df=0.0):
    x, y = r_vec
    vx, vy = v_vec
    r = math.hypot(x, y)

    factor = -MU / r ** 3
    ax_grav = factor * x
    ay_grav = factor * y

    if A_df != 0.0:
        v = math.hypot(vx, vy)
        denom = v ** 3 + B_df
        ax_df = -A_df * vx / denom
        ay_df = -A_df * vy / denom
    else:
        ax_df = 0.0
        ay_df = 0.0
    return np.array([ax_grav + ax_df, ay_grav + ay_df])


def derivatives(t, y, A_df, B_df):
    x, y_pos, vx, vy = y
    ax, ay = acceleration_total((x, y_pos), (vx, vy), A_df, B_df)
    return np.array([vx, vy, ax, ay])


def rk4_step(t, y, h, A_df, B_df):
    k1 = derivatives(t, y, A_df, B_df)
    k2 = derivatives(t + 0.5 * h, y + 0.5 * h * k1, A_df, B_df)
    k3 = derivatives(t + 0.5 * h, y + 0.5 * h * k2, A_df, B_df)
    k4 = derivatives(t + h,       y + h * k3,       A_df, B_df)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def adaptive_rk4_step(t, y, h, d_tol, A_df, B_df):
    safety = 0.9
    max_factor = 4.0
    min_factor = 0.1

    while True:
        y_big = rk4_step(t, y, h, A_df, B_df)

        y_half = rk4_step(t, y, 0.5 * h, A_df, B_df)
        y_small = rk4_step(t + 0.5 * h, y_half, 0.5 * h, A_df, B_df)

        err_vec = y_small[:2] - y_big[:2]
        err = np.linalg.norm(err_vec) / 15.0

        if err == 0.0:
            factor = max_factor
        else:
            tol = d_tol * h
            factor = safety * (tol / err) ** 0.2
            factor = min(max_factor, max(min_factor, factor))

        if err <= d_tol * h:
            t_new = t + h
            y_new = y_small
            h_new = h * factor
            return t_new, y_new, h_new
        else:
            h *= factor


def specific_energy(y):
    x, y_pos, vx, vy = y
    r = math.hypot(x, y_pos)
    v2 = vx * vx + vy * vy
    return 0.5 * v2 - MU / r

def initial_conditions_elliptical():
    r_peri = R_S
    r_apo = 1.0

    a = 0.5 * (r_peri + r_apo)
    e = (r_apo - r_peri) / (r_apo + r_peri)
    h2 = MU * a * (1.0 - e ** 2)
    v_tan = math.sqrt(h2) / r_apo

    x0, y0 = r_apo, 0.0
    vx0, vy0 = 0.0, v_tan
    y_vec = np.array([x0, y0, vx0, vy0], dtype=float)
    return y_vec, a


def initial_conditions_df(v_factor=0.8):
    r0 = 1.0
    v_circ = math.sqrt(MU / r0)
    v0 = v_factor * v_circ

    x0, y0 = r0, 0.0
    vx0, vy0 = 0.0, v0
    y_vec = np.array([x0, y0, vx0, vy0], dtype=float)
    return y_vec

def integrate_until_time(t_end, A_df, B_df, y0, d_tol=D_TOL):
    """
    Integrate until a fixed time t_end (for part a).
    """
    y = y0.copy()
    t = 0.0

    T_guess = 2.0 * math.pi * math.sqrt(1.0 ** 3 / MU)
    h = T_guess / 1000.0

    times = [t]
    states = [y.copy()]

    while t < t_end:
        if t + h > t_end:
            h = t_end - t
        t, y, h = adaptive_rk4_step(t, y, h, d_tol, A_df, B_df)
        times.append(t)
        states.append(y.copy())

    return np.array(times), np.vstack(states)


def integrate_until_rs(A_df, B_df, y0, d_tol=D_TOL, max_time_factor=200.0):
    y = y0.copy()
    t = 0.0

    T_guess = 2.0 * math.pi * math.sqrt(1.0 ** 3 / MU)
    t_max = max_time_factor * T_guess
    h = T_guess / 1000.0

    times = [t]
    states = [y.copy()]

    while t < t_max:
        r = math.hypot(y[0], y[1])
        if r <= R_S:
            break

        t, y, h = adaptive_rk4_step(t, y, h, d_tol, A_df, B_df)
        times.append(t)
        states.append(y.copy())

    return np.array(times), np.vstack(states)


def time_to_rs(A_df, B_df, v_factor=0.8, d_tol=D_TOL):
    y0 = initial_conditions_df(v_factor=v_factor)
    times, states = integrate_until_rs(A_df, B_df, y0, d_tol=d_tol)
    radii = np.hypot(states[:, 0], states[:, 1])
    idx = np.where(radii <= R_S)[0]
    if len(idx) == 0:
        return None
    return float(times[idx[0]])

def run_part_a():
    print("=== Part (a): test accuracy without dynamical friction ===")

    y0, a = initial_conditions_elliptical()
    T_orbit = 2.0 * math.pi * math.sqrt(a ** 3 / MU)
    n_orbits = 10
    t_end = n_orbits * T_orbit

    times, states = integrate_until_time(t_end, A_df=0.0, B_df=0.0, y0=y0)
    radii = np.hypot(states[:, 0], states[:, 1])
    print("Per-orbit r_min and r_max:")

    r_min_list = []
    r_max_list = []

    for i in range(n_orbits):
        t_start = i * T_orbit
        t_stop = (i + 1) * T_orbit
        mask = (times >= t_start) & (times < t_stop)
        r_seg = radii[mask]
        r_min_i = np.min(r_seg)
        r_max_i = np.max(r_seg)
        r_min_list.append(r_min_i)
        r_max_list.append(r_max_i)
        print(f"Orbit {i+1:2d}: r_min = {r_min_i}, r_max = {r_max_i}")

def run_part_b():
    print("\n=== Part (b): orbit with dynamical friction A = B = 1 ===")

    A_df = 1.0
    B_df = 1.0

    y0 = initial_conditions_df(v_factor=0.8)
    times, states = integrate_until_rs(A_df, B_df, y0)

    radii = np.hypot(states[:, 0], states[:, 1])
    t_end = times[-1]
    t_end_myr = t_end * TIME_UNIT_MYR

    print(f"Time to reach r_s =  {t_end_myr} Myr")

    plt.figure()
    plt.plot(states[:, 0], states[:, 1], label="orbit")
    plt.plot(states[-1, 0], states[-1, 1], "o", markersize=5, color="red", label="final position")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.title("BH orbit with dynamical friction (A=B=1)")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(times, np.log10(radii))
    plt.xlabel("t (code units)")
    plt.ylabel("log10(r)")
    plt.title("log10(r) - time (A=B=1)")
    plt.tight_layout()

def run_part_c():
    print("\n=== Part (c): time to r_s as a function of B/A ===")

    ratio_values = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 10.0]
    A_df = 1.0

    times_code = []
    times_myr = []
    ratios_used = []

    for ratio in ratio_values:
        B_df = ratio * A_df
        if not (0.5 <= A_df <= 10.0 and 0.5 <= B_df <= 10.0):
            continue

        t_rs = time_to_rs(A_df, B_df, v_factor=0.8)
        if t_rs is None:
            print(f"B/A = {ratio}: did not reach r_s within max time")
            continue

        t_myr = t_rs * TIME_UNIT_MYR
        ratios_used.append(ratio)
        times_code.append(t_rs)
        times_myr.append(t_myr)

        print(f"B/A = {ratio}: total time is {t_myr} Myr")

    ratios_used = np.array(ratios_used)
    times_myr = np.array(times_myr)

    plt.figure()
    plt.plot(np.log(ratios_used), np.log(times_myr), marker="o")
    plt.xlabel("B / A")
    plt.ylabel("time to reach r_s (Myr)")
    plt.title("   time to reach r_s - B/A")
    plt.tight_layout()

def run_part_d():
    ratio = 1.0
    A_values = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 10.0]
    times_A_code = []
    times_A_myr = []
    A_used = []

    print("\nSame B/A = 1, different A and B = A:")
    for A_df in A_values:
        B_df = ratio * A_df
        t_rs = time_to_rs(A_df, B_df, v_factor=0.8)
        if t_rs is None:
            print(f"A = B = {A_df}: did not reach r_s")
            continue

        t_myr = t_rs * TIME_UNIT_MYR
        A_used.append(A_df)
        times_A_code.append(t_rs)
        times_A_myr.append(t_myr)

        print(f"A = B = {A_df}: total time is {t_myr} Myr")

    if len(A_used) > 0:
        A_used = np.array(A_used)
        times_A_myr = np.array(times_A_myr)

        plt.figure()
        plt.plot(A_used, times_A_myr, marker="o")
        plt.xlabel("A (with B = A)")
        plt.ylabel("time to reach r_s (Myr)")
        plt.xscale("log")
        plt.yscale("log")
        plt.title("time to r_s - A (B/A = 1)")
        plt.tight_layout()


    v_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    v_used = []
    times_v_code = []
    times_v_myr = []

    print("\nSame A = B = 1, different initial velocity")
    for vf in v_factors:
        t_rs = time_to_rs(1.0, 1.0, v_factor=vf)
        if t_rs is None:
            print(f"v_factor = {vf}: did not reach r_s")
            continue

        t_myr = t_rs * TIME_UNIT_MYR
        v_used.append(vf)
        times_v_code.append(t_rs)
        times_v_myr.append(t_myr)

        print(f"v_factor = {vf}: total time is {t_myr} Myr")

    if len(v_used) > 0:
        v_used = np.array(v_used)
        times_v_myr = np.array(times_v_myr)

        plt.figure()
        plt.plot(v_used, times_v_myr, marker="o")
        plt.xlabel("initial velocity factor")
        plt.ylabel("time to reach r_s (Myr)")
        plt.xscale("log")
        plt.yscale("log")
        plt.title("time to r_s - initial velocity")
        plt.tight_layout()

def main():
    run_part_a()
    run_part_b()
    run_part_c()
    run_part_d()
    plt.show()


if __name__ == "__main__":
    main()

