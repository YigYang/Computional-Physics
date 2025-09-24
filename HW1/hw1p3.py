#All the variables Xi was named as Epsilon
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
datatable = Path("lcdm_z0.matter_pk")
spectrum = np.loadtxt(datatable)
k = spectrum[:, 0]
P = spectrum[:, 1]
logk = np.log(k)
logP = np.log(P)
plt.figure(1)
plt.plot(logk, logP)
plt.xlabel("log(k)")
plt.ylabel("log(P)")
plt.show()

N=k.size

findkeq1 = (k>=1)
logk_hi = np.log(k[findkeq1])
logP_hi = np.log(P[findkeq1])
m, b = np.polyfit(logk_hi, logP_hi, deg=1)
N_ext = 7*N
logk_max = 8*logk[-1]
logk_ext = np.linspace(logk[-1], logk_max, N_ext+1)
k_ext = np.exp(logk_ext)
logP_ext = m * logk_ext + b
P_ext = np.exp(logP_ext)

logk_all = np.concatenate([logk, logk_ext[1:]])
k_all = np.concatenate([k, k_ext[1:]])
logP_all = np.concatenate([logP, logP_ext[1:]])
P_all = np.concatenate([P, P_ext[1:]])

plt.figure(2)
plt.plot(logk_all,logP_all)
plt.xlabel("log(k_extended)")
plt.ylabel("log(P_extended)")
plt.show()

N_new = int(0.5*logk_all.size)
Ndouble = logk_all.size
print(N_new, "is the new total bin number")
print("upper limit log(k) is", logk_all[N_new-1])
Ndouble = logk_all.size

def h(r, k, P):
    return 0.5 * k * k * P * np.sin(k*r) / (r * np.pi**2)
def integral(r):
    S = 0
    for i in range(N_new-1):
        S += 0.5 * (logk_all[i+1]-logk_all[i]) * (h(r, k_all[i], P_all[i]) + 
                                                  h(r, k_all[i+1], P_all[i+1]))
    return S

r_range=np.linspace(50,120,211)
Epsilon_list = []
Epsilonr2_list = []


for r in r_range:
    Epsilon = integral(r)
    Epsilonr2 = r * r * Epsilon
    Epsilon_list.append(Epsilon)
    Epsilonr2_list.append(Epsilonr2)
   
        
    
#Convergence test
Earray = np.array(Epsilon_list)
Er2array = np.array(Epsilonr2_list)


max_idx = np.nanargmax(Er2array)
r_max = r_range[max_idx]
r2Epsilon_max = Er2array[max_idx]

print("BAO peak happen around r =",r_max)

plt.figure(3)
plt.plot(r_range,Er2array,label=r"$r^2\,\xi(r)$")
ax = plt.gca()
ax.vlines(r_max, ymax=r2Epsilon_max, ymin=-110,linestyle='--', linewidth=1, color='r',label="Peak position")
plt.xlabel("r [Mpc/h]")
plt.ylabel(r"$r^2 \, \xi(r)$ [$(Mpc/h)^2$]")
plt.ylim(-110,150)
plt.legend()
plt.show()


#Convergence test


def bigger_intergral(r):
    S = 0
    for i in range(Ndouble-1):
        S += 0.5 * (logk_all[i+1]-logk_all[i]) * (h(r, k_all[i], P_all[i]) + 
                                                  h(r, k_all[i+1], P_all[i+1]))
    return S
Epsilon_bigger_list = []
Epsilonr2_bigger_list = []
for r in r_range:
    Epsilon_bigger = bigger_intergral(r)
    Epsilonr2_bigger = r * r *bigger_intergral(r)
    Epsilon_bigger_list.append(Epsilon_bigger)
    Epsilonr2_bigger_list.append(Epsilonr2_bigger)
Ebiggerarray = np.array(Epsilon_bigger_list)
Er2biggerarray = np.array(Epsilonr2_bigger_list)
err = abs((Ebiggerarray - Earray)/Earray)

plt.figure(4)
plt.plot(r_range,err)
plt.ylim(0,0.00001)
plt.xlabel("r [Mpc/h]")
plt.ylabel("relative error")
plt.show()



    