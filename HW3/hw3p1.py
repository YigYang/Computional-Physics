import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

datatable = Path("sunspots.txt")
sunspot = np.loadtxt(datatable)
t = sunspot[:,0]
x = sunspot[:,1]
N = len(x)

plt.figure(1)
plt.plot(t, x)
plt.xlabel("Month since Jan 1749")
plt.ylabel("Sunspot number")
plt.title("Monthly sunspots ~ time")
ax = plt.gca()  
ax.text(0.2, 0.9,                       
        "estimated period : 130.43 months",
        transform=ax.transAxes,         
        bbox=dict(fc="w", alpha=0.8))
plt.tight_layout()
plt.show()


y = x - x.mean()                  
Y = np.fft.rfft(y)               
freq = np.fft.rfftfreq(N,d=1)  
power = (np.abs(Y)**2) / N**2     

k_peak = np.argmax(power)
f_peak = freq[k_peak]
T_months = 1.0 / f_peak

plt.figure(2)
plt.plot(freq, power)
plt.scatter([f_peak], [power[k_peak]], s=40)
plt.annotate(f"peak f = {f_peak:.4f} cyc/mo\nT = {T_months:.2f} mo",
             xy=(f_peak, power[k_peak]),
             xytext=(f_peak*10, power[k_peak]*0.7),
             arrowprops=dict(arrowstyle="->"))
plt.xlabel("Frequency (cycles per month)")
plt.ylabel(r"Power $|c_k|^2$")
plt.title("Power spectrum of monthly sunspots")
plt.tight_layout()
plt.show()