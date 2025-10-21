import numpy as np
import matplotlib.pyplot as plt
path = "blur.txt"
blur = np.loadtxt(path)               
plt.figure(1)
plt.imshow(blur, origin='upper')     
plt.axis()
plt.colorbar()
plt.title("density plot")
plt.tight_layout()
plt.show()

H,W = blur.shape
sigma = 25
y = np.arange(H)
x = np.arange(W)
dy = np.minimum(y, H - y)[:, None]       
dx = np.minimum(x, W - x)[None, :]       
psf = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
plt.figure(2)
plt.imshow(psf, origin='upper')     
plt.axis()
plt.colorbar()
plt.title(f"PSF (σ={sigma})")
plt.show()

psf /= psf.sum()
B = np.fft.rfft2(blur)
F = np.fft.rfft2(psf)
EPS = 1e-3                           
magF = np.abs(F)
A = np.empty_like(B)
mask = magF >= EPS
A[mask] = B[mask] / F[mask]
A[~mask] = B[~mask]            
clear = np.fft.irfft2(A, s=(H, W))
plt.figure(3)
plt.imshow(clear, origin='upper')
plt.title(f"Reconstructed image (ε={EPS})")
plt.axis()
plt.colorbar()
plt.tight_layout()
plt.show()