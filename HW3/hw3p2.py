import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

x = np.loadtxt("dow.txt")           
N = x.size
t = np.arange(N)              

datafile = Path("dow.txt")            
x = np.loadtxt(datafile)              
N = len(x)
t = np.arange(N)
plt.figure(1)
plt.plot(t, x)
plt.xlabel("day")
plt.ylabel("daily closing")
plt.title("Daily closing ~ Time")
plt.tight_layout()
plt.show()

Y = np.fft.rfft(x)                    
M = Y.size
K10 = int(np.ceil(0.10 * M))
Y10 = Y.copy()
Y10[K10:] = 0                        
x10 = np.fft.irfft(Y10, n=N) 
plt.figure(2)
plt.plot(t, x, label="original")
plt.plot(t, x10, label="first 10% coeffs (low-pass)")
plt.xlabel("Business day index")
plt.ylabel("Close")
plt.title("low-pass filter: keep 10% of max frequency")
plt.legend()
plt.tight_layout()
plt.show()

K2 = int(np.ceil(0.02 * M))
Y2 = Y.copy()
Y2[K2:] = 0                        
x2 = np.fft.irfft(Y2, n=N)
plt.figure(3)
plt.plot(t, x, label="original")
plt.plot(t, x2, label="first 2% coeffs (strong low-pass)")
plt.xlabel("Business day index")
plt.ylabel("Close")
plt.title("low-pass filter: keep 10% of max frequency")
plt.legend()
plt.tight_layout()

plt.show()

