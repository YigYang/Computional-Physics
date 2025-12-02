import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit

hbar = 1.054571817e-34     
m_e  = 9.109e-31              
L  = 1.0e-8                    
x0 = L / 2.0                  
sigma = 1.0e-10             
kappa = 5.0e10                 

N = 1000                       
a = L / N                     
dt = 1.0e-18                   

x = np.linspace(a, L - a, N - 1)
psi0 = np.exp(- (x - x0)**2 / (2.0 * sigma**2)) * np.exp(1j * kappa * x)
psi0 = np.asarray(psi0, dtype=np.complex128)

n = N - 1                     
alpha = 1j * hbar * dt / (4.0 * m_e * a**2)

a1 = 1.0 + 2.0 * alpha
a2 = -alpha
b1 = 1.0 - 2.0 * alpha
b2 = alpha

mainA = np.full(n, a1, dtype=np.complex128)      
offA  = np.full(n-1, a2, dtype=np.complex128)    
mainB = np.full(n, b1, dtype=np.complex128)      
offB  = np.full(n-1, b2, dtype=np.complex128)    

@njit
def cn_step(psi, mainA, offA, mainB, offB):
    n = psi.shape[0]
    v = np.empty(n, dtype=np.complex128)
    for i in range(n):
        s = mainB[i] * psi[i]
        if i > 0:
            s += offB[i-1] * psi[i-1]
        if i < n-1:
            s += offB[i] * psi[i+1]
        v[i] = s
    cp = np.empty(n-1, dtype=np.complex128)   
    dp = np.empty(n,   dtype=np.complex128)   

    cp[0] = offA[0] / mainA[0]
    dp[0] = v[0] / mainA[0]

    for i in range(1, n-1):
        denom = mainA[i] - offA[i-1] * cp[i-1]
        cp[i] = offA[i] / denom
        dp[i] = (v[i] - offA[i-1] * dp[i-1]) / denom

    denom_last = mainA[n-1] - offA[n-2] * cp[n-2]
    dp[n-1] = (v[n-1] - offA[n-2] * dp[n-2]) / denom_last

    x = np.empty(n, dtype=np.complex128)
    x[n-1] = dp[n-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x

psi = psi0.copy()
n_steps = 5000                

psi_t_list = [psi0.copy()]
for step in range(n_steps):
    psi = cn_step(psi, mainA, offA, mainB, offB)
    psi_t_list.append(psi.copy())

n_frames = 600   
frame_indices = np.linspace(0, n_steps, n_frames, dtype=int)

fig, ax = plt.subplots(figsize=(8, 5))

line, = ax.plot(x, np.real(psi_t_list[0]), lw=2)
ax.set_xlabel("x (m)")
ax.set_ylabel(r"Re$\{\psi(x,t)\}$")
ax.set_title("Dynamics of real part of wavefunction")

ymax = 1.2 * np.max(np.abs(np.real(psi0)))
ax.set_ylim(-ymax, ymax)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def update(frame):
    idx = frame_indices[frame]
    psi_frame = psi_t_list[idx]
    line.set_ydata(np.real(psi_frame))
    time_text.set_text(f"step = {idx}, t = {idx*dt:.1e} s")
    return line, time_text

ani = FuncAnimation(fig, update, frames=n_frames,
                    interval=30, blit=True)

ani.save("schrodinger_realpart.gif", writer="pillow", fps=1000//30)
print("GIF saved as schrodinger_realpart_long.gif")