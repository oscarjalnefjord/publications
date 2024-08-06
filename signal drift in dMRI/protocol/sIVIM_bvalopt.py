import numpy as np
import ivim

# sIVIM protocol
n = 5
f = np.repeat(np.linspace(0.02,0.1,n),n)
D = np.tile(np.linspace(0.6e-3,1.1e-3,n),n)
b,a = ivim.optimize.crlb.sIVIM(D,f,None,200,800)

print('Optimized b-values:')
print(np.round(b))
print()
print('Optimized proportions:')
print(a)
print()
print('Optimized distribution of 6 measurements:')
print(np.round(6*a))