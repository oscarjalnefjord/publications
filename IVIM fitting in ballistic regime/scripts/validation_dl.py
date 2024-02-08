import os
import nibabel as nib
import numpy as np
import IVIMNET.deep as deep
from hyperparams import hyperparams as hp

root = '/tmp'
sub = 'phantom'
for i in range(2):
    bval_file = os.path.join(root,sub,'{}_validation-{}.bval'.format(sub,i+1))
    cval_file = bval_file.replace('bval','cval')

    outbase_dl = os.path.join(root,sub,'{}_validation_IVIM-dl-{}'.format(sub,i+1))
    im_file = os.path.join(root,sub,'{}_validation-{}.nii.gz'.format(sub,i+1))
    roi_file = os.path.join(root,sub,'{}_validation-{}_ROI.nii.gz'.format(sub,i+1))
    nii = nib.load(im_file)
    roi = nib.load(roi_file).get_fdata().astype(bool)
    im = nii.get_fdata()
    roi &= ~np.any(np.isnan(im),axis=3)        
    Y = im[roi, :]
    b = np.loadtxt(bval_file)
    c = np.loadtxt(cval_file)

    arg = hp()
    arg.net_pars.cons_min = [0, 0, 0.05, 0]  # Dt, Fp, vd2, S0
    arg.net_pars.cons_max = [0.005, 0.7, 5.0**2, 2*np.max(Y)]  # Dt, Fp, vd2, S0
    arg.net_pars.ballistic = True

    arg = deep.checkarg(arg)
    net = deep.learn_IVIM(Y, b, arg, cvalues=c)
    paramsNN = deep.predict_IVIM(Y, b, net, arg)
    for par, parname in zip(paramsNN, ['D','f','vd','S0']):
        im = np.full(roi.shape, np.nan,dtype=float)
        if parname == 'vd':
            im[roi] = np.sqrt(par)
        else:
            im[roi] = par
        nib.save(nib.Nifti1Image(im, nii.affine, nii.header), outbase_dl+'_{}.nii.gz'.format(parname))