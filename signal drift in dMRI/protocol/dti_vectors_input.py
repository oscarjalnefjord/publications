import os
import numpy as np
import ivim

# sIVIM (3 b-values)
bvals = [0,200,800,200,0,200] # based on optimization
fname = os.path.join('protocol','dti_vectors_input_sIVIM3b.txt')
ivim.io.philips.generate_dti_vectors_input(bvals,fname,dualpol=True)
# Philips does not accept non-unique rows, fixed by slightly altering the directions when needed
X = ivim.io.philips.read_dti_vectors_input(fname)
for b in [0,200,800]:
    for i in range(3):
        for sign in [-1,1]:
            cond = (X[:,3]==b) & (X[:,i]==sign)
            Xsub = X[cond,:]
            for r in range(np.sum(cond)):
                Xsub[r,(i+1)%3] += r*0.002
            X[cond,:] = Xsub 
ivim.io.philips.write_dti_vectors_input(X,fname,header='sIVIM 3b')

# IVIM (10 b-values)
bvals = [0,800,5,500,10,200,20,100,30,50]
fname = os.path.join('protocol','dti_vectors_input_IVIM10b.txt')
ivim.io.philips.generate_dti_vectors_input(bvals,fname,dualpol=True,header='IVIM 10b')

# IVIM FC/NC (7 b-values)
bvals = [0,200,5,100,10,30,20]
fname = os.path.join('protocol','dti_vectors_input_IVIM-FNC-7b.txt')
ivim.io.philips.generate_dti_vectors_input(bvals,fname,dualpol=True,header='IVIM FC/NC 7b')
