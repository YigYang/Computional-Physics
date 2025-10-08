import numpy as np
def f(x):
    return 5*np.exp(-x) + x - 5

def binary(lo, hi, eps=1e-6, max_iter=10_000):
    flo, fhi = f(lo), f(hi)
    assert flo * fhi < 0, "Root is not bracketed"
    for _ in range(max_iter):
        mid = 0.5*(lo + hi)
        fmid = f(mid)
        if abs(hi - lo) < eps or abs(fmid) < eps:
            return mid
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5*(lo + hi)

x = binary(4, 6, eps=1e-6)
print(f"The solution of the equation is x = {x}")

h  = 6.62607015e-34      
c  = 299792458       
kB = 1.380649e-23   

b = h*c/(kB*x)
print(f"Wien displacement constant b = {b} m·K")     
