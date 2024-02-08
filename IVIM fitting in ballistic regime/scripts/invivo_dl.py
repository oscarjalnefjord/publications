import os
import numpy as np
import nibabel as nib
import IVIMNET.deep as deep
from hyperparams import hyperparams as hp

root = '/mnt/d/fc-ivim-estimation'
input_root = os.path.join(root, 'data')
output_root = os.path.join(root, 'results')

subs = os.listdir(input_root)
times = []

for sub in subs:
    input_folder = os.path.join(input_root,sub)
    output_folder = os.path.join(output_root,sub)

    # IVIM parameter estimation
    for run in ['run-{}'.format(r+1) for r in range(2)]:
        im_file = os.path.join(output_folder,'{}_{}_IVIM-avg.nii.gz'.format(sub,run))
        bval_file = im_file.replace('.nii.gz','.bval')
        cval_file = im_file.replace('.nii.gz','.cval')
        mask_file = os.path.join(output_folder,'{}_IVIM-b0-mask.nii.gz'.format(sub))

        outbase_dl = os.path.join(output_folder,'{}_{}_IVIM-dl'.format(sub,run))

        nii = nib.load(im_file)
        roi = nib.load(mask_file).get_fdata().astype(bool)
        Y = nii.get_fdata()[roi, :]
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