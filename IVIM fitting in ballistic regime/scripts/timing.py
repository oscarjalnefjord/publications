import os
import numpy as np
import nibabel as nib
import time
import pandas as pd

output_root = '/path/to/results'
sub = 'sub-01'
im_file = os.path.join(output_root,sub,'{}_run-1_IVIM-avg.nii.gz'.format(sub))
bval_file = im_file.replace('nii.gz','bval')
cval_file = im_file.replace('nii.gz','cval')
mask_file = os.path.join(output_root,sub,'{}_IVIM-b0-mask.nii.gz'.format(sub))
outbase = '/tmp/tmp_time'

dl = False
if dl:
    import IVIMNET.deep as deep
    from hyperparams import hyperparams as hp
    t_start = time.time()

    outbase_dl = os.path.join(output_root,sub,'{}_run-1_IVIM-dl'.format(sub))

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
        nib.save(nib.Nifti1Image(im, nii.affine, nii.header), outbase+'_{}.nii.gz'.format(parname))
    t = {'dl': [time.time() - t_start]}
else:
    import ivim
    t = {}

    t_start = time.time()
    ivim.fit.seg(im_file, bval_file, 'ballistic', bthr = 100, cval_file = cval_file, cthr = 0, roi_file = mask_file, outbase=outbase, verbose=True)
    t['seg'] = [time.time() - t_start]

    t_start = time.time()
    ivim.fit.nlls(im_file, bval_file, 'ballistic', cval_file = cval_file, roi_file = mask_file, outbase=outbase, verbose=True)
    t['nlls'] = [time.time() - t_start]

    n = 5000
    burns = 1000

    t_start = time.time()
    ivim.fit.bayes(im_file, bval_file, 'ballistic', cval_file = cval_file, roi_file = mask_file, outbase=outbase, verbose=True, n=n, burns=burns, ctm='mean')
    t['bayesu'] = [time.time() - t_start]

    t_start = time.time()
    ivim.fit.bayes(im_file, bval_file, 'ballistic', cval_file = cval_file, roi_file = mask_file, outbase=outbase, verbose=True, n=n, burns=burns, ctm='mean', spatial_prior = True)
    t['bayess'] = [time.time() - t_start]

csv_file = 'tables/time.csv'
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    df.join(pd.DataFrame(t))
else:
    df = pd.DataFrame(t)



df.to_csv(csv_file)