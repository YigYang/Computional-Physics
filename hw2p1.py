import numpy as np

def f(x):
    return 1 - np.exp(-2*x)

x, tol = 0.5, 1e-6
for n in range(1, 1000001):              
    x_new = f(x)       
    if abs(x_new - x) < tol:             
        x = x_new
        break
    x = x_new
print(f"By relaxation, the solution is x = {x}")
print(f"The iteration needs for relaxation method is {n}")

x, tol, omega = 0.5, 1e-6, 0.5
for n in range(1, 1000001):
        x_new = (1 + omega) * f(x) - omega * x
        if abs(x_new - x) < tol:
            break
        x = x_new
print(f"By over-relaxation, the solution is x = {x}")
print(f"The iteration needs for over-relaxation method with ω = 0.5 is {n}")

x, tol, omega = 0.5, 1e-6, 0.7
for n in range(1, 1000001):
        x_new = (1 + omega) * f(x) - omega * x
        if abs(x_new - x) < tol:
            break
        x = x_new
print(f"By over-relaxation, the solution is x = {x}")
print(f"The iteration needs for over-relaxation method with ω = {omega} is {n}")