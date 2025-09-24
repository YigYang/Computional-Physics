import numpy as np
import matplotlib.pyplot as plt

x1=0.1
x2=10

#define the functions in single precision
def f_cos(x):
    return np.cos(x, dtype=np.float32)
def f_exp(x):
    return np.exp(x, dtype=np.float32)

#define the differential methods in single precision
def forward_diff(f, x, h):
    x = np.float32(x)
    h = np.float32(h)
    return np.float32(f(x + h) - f(x)) / h
def central_diff(f, x, h):
    x = np.float32(x)
    h = np.float32(h)
    return (np.float32(f(x + h) - f(x - h))) / (np.float32(2) * h)
def extrapolated_diff(f, x, h):
    d1 = central_diff(f, x, h)
    d2 = central_diff(f, x, h/2)
    return (np.float32(4) * d2 - d1) / np.float32(3)

#define the exact differentiations
def df_cos_exact(x):
    return -np.sin(np.float32(x))
def df_exp_exact(x):
    return np.exp(np.float32(x))

#define the interval h:
h_val = np.logspace(-7, 0, 211, dtype=np.float32)

#Calculate relative error for cos x @ 0.1
forward_error_cos0p1 = []
central_error_cos0p1 = []
extrapolated_error_cos0p1 = []

target = np.float32(1e-3)

for h in h_val:
    x=x1

    cal_value1 = forward_diff(f_cos, x, h)
    error1 = abs((cal_value1 - df_cos_exact(x)) / df_cos_exact(x))
    forward_error_cos0p1.append(error1)
    
    cal_value2 = central_diff(f_cos, x, h)
    error2 = abs((cal_value2 -  df_cos_exact(x)) / df_cos_exact(x))
    central_error_cos0p1.append(error2)
    
    cal_value3 = extrapolated_diff(f_cos, x, h)
    error3 = abs((cal_value3 - df_cos_exact(x)) / df_cos_exact(x))
    extrapolated_error_cos0p1.append(error3)
    if 0.0009<h<0.0011:
        print("When h=0.001, the calculated derivative of cos(x) at 0.1 is",cal_value1,cal_value2,cal_value3)

#plot the figure
h = np.array(h_val, dtype=float)
e1_cos_0p1 = np.array(forward_error_cos0p1)
e2_cos_0p1 = np.array(central_error_cos0p1)
e3_cos_0p1 = np.array(extrapolated_error_cos0p1)

plt.figure(1)
plt.loglog(h, e1_cos_0p1, label="forward")
plt.loglog(h, e2_cos_0p1, label="central")
plt.loglog(h, e3_cos_0p1, label="extrapolated")
plt.legend()
plt.xlabel("h")
plt.ylabel("Relative error")
plt.title("error of cos0.1 in log-log scale")
plt.show()

forward_error_cos10 = []
central_error_cos10 = []
extrapolated_error_cos10 = []


for h in h_val:
    x=x2

    cal_value1 = forward_diff(f_cos, x, h)
    error1 = abs((cal_value1 - df_cos_exact(x)) / df_cos_exact(x))
    forward_error_cos10.append(error1)
    
    cal_value2 = central_diff(f_cos, x, h)
    error2 = abs((cal_value2 -  df_cos_exact(x)) / df_cos_exact(x))
    central_error_cos10.append(error2)
    
    cal_value3 = extrapolated_diff(f_cos, x, h)
    error3 = abs((cal_value3 - df_cos_exact(x)) / df_cos_exact(x))
    extrapolated_error_cos10.append(error3)
    if 0.0009<h<0.0011:
        print("When h=0.001, the calculated derivative of cos(x) at 10 is",cal_value1,cal_value2,cal_value3)

h = np.array(h_val, dtype=float)
e1_cos_10 = np.array(forward_error_cos10)
e2_cos_10 = np.array(central_error_cos10)
e3_cos_10 = np.array(extrapolated_error_cos10)

plt.figure(2)
plt.loglog(h, e1_cos_10, label="forward")
plt.loglog(h, e2_cos_10, label="central")
plt.loglog(h, e3_cos_10, label="extrapolated")
plt.legend()
plt.xlabel("h")
plt.ylabel("Relative error")
plt.title("error of cos10 in log-log scale")
plt.show()

forward_error_exp0p1 = []
central_error_exp0p1 = []
extrapolated_error_exp0p1 = []

for h in h_val:
    x=x1

    cal_value1 = forward_diff(f_exp, x, h)
    error1 = abs((cal_value1 - df_exp_exact(x)) / df_exp_exact(x))
    forward_error_exp0p1.append(error1)
    
    cal_value2 = central_diff(f_exp, x, h)
    error2 = abs((cal_value2 -  df_exp_exact(x)) / df_exp_exact(x))
    central_error_exp0p1.append(error2)
    
    cal_value3 = extrapolated_diff(f_exp, x, h)
    error3 = abs((cal_value3 - df_exp_exact(x)) / df_exp_exact(x))
    extrapolated_error_exp0p1.append(error3)
    if 0.0009<h<0.0011:
        print("When h=0.001, the calculated derivative of exp(x) at 0.1 is",cal_value1,cal_value2,cal_value3)

h = np.array(h_val, dtype=float)
e1_exp_0p1 = np.array(forward_error_exp0p1)
e2_exp_0p1 = np.array(central_error_exp0p1)
e3_exp_0p1 = np.array(extrapolated_error_exp0p1)

plt.figure(3)
plt.loglog(h, e1_exp_0p1, label="forward")
plt.loglog(h, e2_exp_0p1, label="central")
plt.loglog(h, e3_exp_0p1, label="extrapolated")
plt.legend()
plt.xlabel("h")
plt.ylabel("Relative error")
plt.title("error of exp0.1 in log-log scale")
plt.show()

forward_error_exp10 = []
central_error_exp10 = []
extrapolated_error_exp10 = []

for h in h_val:
    x=x2

    cal_value1 = forward_diff(f_exp, x, h)
    error1 = abs((cal_value1 - df_exp_exact(x)) / df_exp_exact(x))
    forward_error_exp10.append(error1)
    
    cal_value2 = central_diff(f_exp, x, h)
    error2 = abs((cal_value2 -  df_exp_exact(x)) / df_exp_exact(x))
    central_error_exp10.append(error2)
    
    cal_value3 = extrapolated_diff(f_exp, x, h)
    error3 = abs((cal_value3 - df_exp_exact(x)) / df_exp_exact(x))
    extrapolated_error_exp10.append(error3)
    if 0.0009<h<0.0011:
        print("When h=0.001, the calculated derivative of exp(x) at 10 is",cal_value1,cal_value2,cal_value3)

h = np.array(h_val, dtype=float)
e1_exp_10 = np.array(forward_error_exp10)
e2_exp_10 = np.array(central_error_exp10)
e3_exp_10 = np.array(extrapolated_error_exp10)

plt.figure(4)
plt.loglog(h, e1_exp_10, label="forward")
plt.loglog(h, e2_exp_10, label="central")
plt.loglog(h, e3_exp_10, label="extrapolated")
plt.legend()
plt.xlabel("h")
plt.ylabel("Relative error")
plt.title("error of exp10 in log-log scale")
plt.show()

df1 = df_cos_exact(x1)
df2 = df_cos_exact(x2)
df3 = df_exp_exact(x1)
df4 = df_exp_exact(x2)
print(df1,"is exact differentiation of cos(x) at 0.1")
print(df2,"is exact differentiation of cos(x) at 10")
print(df3,"is exact differentiation of exp(x) at 0.1")
print(df4,"is exact differentiation of exp(x) at 10")
