import os
import nibabel as nib
import numpy as np
import ivim

root = '/tmp'
sub = 'phantom'

x = 100
n = 50

b = np.array(2*[0, 5, 10, 20, 30, 100, 200])
Delta = 10e-3
delta = 8.6e-3
seq = ivim.seq.sde.BIPOLAR
c = ivim.seq.sde.calc_c(ivim.seq.sde.G_from_b(b,Delta,delta,seq=seq),Delta,delta,seq=seq)
c[:c.size//2] = 0
bval_file = os.path.join(root,sub,'{}_validation.bval'.format(sub))
cval_file = bval_file.replace('bval','cval')
ivim.io.base.write_bval(bval_file,b)
ivim.io.base.write_cval(cval_file,c)

Db = 1.75e-3    
for i, (D,f,vd) in enumerate(zip([[0.5e-3, 1.5e-3],[0.5e-3,3e-3]],[[1e-2,5e-2],[5e-2, 55e-2]],[[1,3],[np.sqrt((10e-3-Db)*(Delta-delta/3)/Delta**2),np.sqrt((100e-3-Db)*(Delta-delta/3)/Delta**2)]])):
    pars = {'D': D, 'f': f, 'vd': vd}

    parmap_files = {}
    for parname,parvals in pars.items():
        parmap = (parvals[1]-parvals[0])*np.random.rand(x,x,n) + parvals[0]
        nii = nib.Nifti1Image(parmap,affine=np.eye(4))
        parmap_files[parname] = os.path.join(root,sub,'{}_validation-{}_{}.nii.gz'.format(sub,i+1,parname))
        nib.save(nii,parmap_files[parname])
    nii_roi = nib.Nifti1Image(np.ones(parmap.shape,dtype=float),affine=np.eye(4))
    roi_file = os.path.join(root,sub,'{}_validation-{}_ROI.nii.gz'.format(sub,i+1))
    nib.save(nii_roi, roi_file)

    noise_sigma = [1/150] # S0 = 1
    outbase = os.path.join(root,sub,'{}_validation-{}'.format(sub,i+1))
    ivim.sim.noise(parmap_files['D'], parmap_files['f'], 'ballistic', bval_file, noise_sigma, outbase, vd_file = parmap_files['vd'], cval_file = cval_file)
