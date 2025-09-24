import numpy as np
import matplotlib.pyplot as plt

#set all numbers in single precision
SP = np.float32
def f(t):
    return np.exp(-t, dtype=SP)

#define the integral functions
def midpoint(a, b, N):
    a = SP(a); b = SP(b); N = int(N)
    h = (b - a) / SP(N)
    i = np.arange(N, dtype=SP)
    x_mid = a + (i + SP(0.5)) * h
    return SP(h) * f(x_mid).sum(dtype=SP)

def trapezoid(a, b, N):
    a = SP(a); b = SP(b); N = int(N)
    h = (b - a) / SP(N)
    x = a + h * np.arange(N + 1, dtype=SP)
    y = f(x)
    s = SP(0.5*(y[0] + y[-1])) + y[1:-1].sum(dtype=SP)
    return SP(h*s)

def simpson(a, b, N):
    a = SP(a); b = SP(b); N = int(N)
    #To make sure the odd and even terms match the coefficients
    if N % 2 == 1:
        N += 1
    h = (b - a) / SP(N)
    x = a + h * np.arange(N + 1, dtype=SP)
    y = f(x)
    Atotal = y[0] + y[-1] + SP(4) * y[1:-1:2].sum(dtype=SP) + SP(2) * y[2:-2:2].sum(dtype=SP)
    return SP(h*Atotal / 3) 

#Calculate the exact value of integral
real_value = 1.0 - np.exp(-1.0)

#Set the total points N
N_val=np.logspace(0.6, 6, 54, dtype=SP)

#set the integral limit
a=0
b=1

#Calculate relative error
midpoint_error = []
trapezoid_error = []
simpson_error = []
for N in N_val:
    cal_value1 = midpoint(a, b, N)
    error1 = abs((cal_value1 - real_value) / real_value)
    midpoint_error.append(error1)
    
    cal_value2 = trapezoid(a, b, N)
    error2 = abs((cal_value2 - real_value) / real_value)
    trapezoid_error.append(error2)
    
    cal_value3 = simpson(a, b, N)
    error3 = abs((cal_value3 - real_value) / real_value)
    simpson_error.append(error3)

#plot the figures
N = np.array(N_val)
e_midpoint = np.array(midpoint_error)
e_trapezoid = np.array(trapezoid_error)
e_simpson = np.array(simpson_error)

plt.loglog(N, e_midpoint, label="midpoint")
plt.loglog(N, e_trapezoid, label="trapezoid")
plt.loglog(N, e_simpson, label="simpson")
plt.legend()
plt.xlabel("N")
plt.ylabel("Relative error")
plt.title("Error and Bin number in log-log scale")
plt.show()