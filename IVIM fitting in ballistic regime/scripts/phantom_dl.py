import os
import nibabel as nib
import numpy as np
import IVIMNET.deep as deep
from hyperparams import hyperparams as hp

root = '/path/to/results'

sub = 'phantom'
bval_file = os.path.join(root,'sub-01','sub-01_run-1_IVIM-avg.bval')
cval_file = bval_file.replace('bval','cval')
noise_sigmas = [1/snr for snr in np.arange(25,301,25)] # S0 = 1
for noise_sigma in noise_sigmas:
    snr = 1/noise_sigma
    im_file = os.path.join(root,sub,'{}_SNR-{:.0f}_IVIM-avg.nii.gz'.format(sub,snr))
    mask_file = os.path.join(root,sub,'phantom_ROI.nii.gz')

    outbase_dl = os.path.join(root,sub,'{}_SNR-{:.0f}_IVIM-dl'.format(sub,snr))

    nii = nib.load(im_file)
    roi = nib.load(mask_file).get_fdata().astype(bool)
    im = nii.get_fdata()
    roi &= ~np.any(np.isnan(im),axis=3)        
    Y = im[roi, :]
    b = np.loadtxt(bval_file)
    c = np.loadtxt(cval_file)

    if not os.path.exists(outbase_dl+'_D.nii.gz'):
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