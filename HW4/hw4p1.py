import numpy as np
import matplotlib.pyplot as plt

a = 1.0
b = 3.0
delta = 1e-10

def derivatives(x, y):
    dxdt = 1 - (b + 1) * x + a * x**2 * y
    dydt = b * x - a * x**2 * y
    return dxdt, dydt

def modified_midpoint(x0, y0, t0, H, n):
    h = H / n
    x_prev = x0
    y_prev = y0
    dx, dy = derivatives(x0, y0)
    x_temp = x0 + h * dx
    y_temp = y0 + h * dy
    for i in range(1, n):
        dx, dy = derivatives(x_temp, y_temp)
        x_next = x_prev + 2 * h * dx
        y_next = y_prev + 2 * h * dy
        
        x_prev = x_temp
        y_prev = y_temp
        x_temp = x_next
        y_temp = y_next
    dx, dy = derivatives(x_temp, y_temp)
    x_final = 0.5 * (x_temp + x_prev + h * dx)
    y_final = 0.5 * (y_temp + y_prev + h * dy)
    
    return x_final, y_final

def richardson_extrapolation(h_vals, r_vals):
    n = len(r_vals)
    R = np.zeros((n, n))
    R[:, 0] = r_vals
    for j in range(1, n):
        for i in range(j, n):
            ratio = (h_vals[i] / h_vals[i-j])**2
            R[i, j] = (ratio * R[i, j-1] - R[i-1, j-1]) / (ratio - 1)
    return R[-1, -1]

def bulirsch_stoer_step(x0, y0, t0, H, delta, n_max=8):
    n_values = [2, 4, 6, 8, 10, 12, 14, 16][:n_max]
    results_x = []
    results_y = []
    h_values = []
    for n in n_values:
        x_est, y_est = modified_midpoint(x0, y0, t0, H, n)
        results_x.append(x_est)
        results_y.append(y_est)
        h_values.append(H / n)
    if len(results_x) >= 2:
        x_extrap = richardson_extrapolation(h_values, results_x)
        y_extrap = richardson_extrapolation(h_values, results_y)
        error_x = abs(results_x[-1] - results_x[-2])
        error_y = abs(results_y[-1] - results_y[-2])
        error = max(error_x, error_y)
        
        return x_extrap, y_extrap, error
    else:
        return results_x[-1], results_y[-1], float('inf')
    
def solve_brusselator(t_start, t_end, x0, y0, H_initial, delta, n_max):
    t = t_start
    x = x0
    y = y0
    H = H_initial
    t_values = [t]
    x_values = [x]
    y_values = [y]
    interval_boundaries_t = [t]
    interval_boundaries_x = [x]
    interval_boundaries_y = [y]
    while t < t_end:
        if t + H > t_end:
            H = t_end - t
        success = False
        attempts = 0
        max_attempts = 20 
        while not success and attempts < max_attempts:
            x_new, y_new, error = bulirsch_stoer_step(x, y, t, H, delta, n_max)
            if error < delta or H < 1e-12:
                t = t + H
                x = x_new
                y = y_new
                t_values.append(t)
                x_values.append(x)
                y_values.append(y)
                interval_boundaries_t.append(t)
                interval_boundaries_x.append(x)
                interval_boundaries_y.append(y)
                success = True
                if error < delta / 10 and H < H_initial:
                    H = min(H * 1.5, H_initial)
            else:
                H = H / 2
                attempts += 1        
        if not success:
            print(f"Warning: Could not achieve desired accuracy at t={t}")
            break
    
    return (np.array(t_values), np.array(x_values), np.array(y_values),
            np.array(interval_boundaries_t), np.array(interval_boundaries_x), 
            np.array(interval_boundaries_y))

print("Brusselator Chemical Oscillator Solver")
print("Parameters:")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  Target accuracy: delta = {delta}")

t_vals, x_vals, y_vals, bound_t, bound_x, bound_y = solve_brusselator(
    t_start=0.0,
    t_end=20.0,
    x0=0.0,
    y0=0.0,
    H_initial=0.1,
    delta=delta,
    n_max=8)

plt.figure(1,figsize=(12, 5))
plt.plot(t_vals, x_vals, 'b-', linewidth=1.5, label='x(t)', alpha=0.8)
plt.plot(t_vals, y_vals, 'r-', linewidth=1.5, label='y(t)', alpha=0.8)
plt.xlabel('Time t', fontsize=11)
plt.ylabel('Concentration', fontsize=11)
plt.title('Brusselator Oscillations: Concentrations - Time', fontsize=12, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(2,figsize=(12, 5))
plt.scatter(bound_t, bound_x, c='blue', marker='o', s=0.1, alpha=0.5, label='x boundaries')
plt.scatter(bound_t, bound_y, c='red', marker='o', s=0.1, alpha=0.5, label='y boundaries')
plt.xlabel('Time t', fontsize=11)
plt.ylabel('Concentration', fontsize=11)
plt.title('Boundary distribution - Time', fontsize=12, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

