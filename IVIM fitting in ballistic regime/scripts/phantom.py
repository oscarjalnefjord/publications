import os
import numpy as np
import nibabel as nib
import ivim

root = '/path/to/results'

# Generate phantom
# 'WM','CGM','DGM','CSF'
pars = {'D': [0.85e-3, 0.95e-3, 0.90e-3, 1.70e-3], 'f': [1.5e-2, 2.5e-2, 2.0e-2, 3.5e-2], 'vd': [1.75, 1.75, 1.5, 1.5]}
dx = 15
x = 100
n = 50
seg = np.zeros((x,x,n),dtype=int)
for i in range(4):
    seg[dx*i:x-dx*i,dx*i:x-dx*i,:] = i+1
nii_seg = nib.Nifti1Image(seg.astype(float),affine=np.eye(4))
seg_file = os.path.join(root,'phantom','phantom_seg.nii.gz')
nib.save(nii_seg, seg_file)

parmap_files = {}
for parname,parvals in pars.items():
    parmap = np.zeros_like(seg,dtype=float)
    for i in range(4):
        parmap[seg==i+1] = parvals[i]
    nii = nib.Nifti1Image(parmap,affine=np.eye(4))
    parmap_files[parname] = os.path.join(root,'phantom','phantom_{}.nii.gz'.format(parname))
    nib.save(nii,parmap_files[parname])
nii_roi = nib.Nifti1Image(np.ones(seg.shape,dtype=float),affine=np.eye(4))
roi_file = os.path.join(root,'phantom','phantom_ROI.nii.gz')
nib.save(nii_roi, roi_file)

bval_file = os.path.join(root,'sub-01','sub-01_run-1_IVIM-avg.bval')
cval_file = bval_file.replace('bval','cval')
noise_sigmas = [1/snr for snr in np.arange(25,301,25)] # S0 = 1
for noise_sigma in noise_sigmas:
    outbase = os.path.join(root,'phantom','phantom_SNR-{:.0f}_IVIM-avg'.format(1/noise_sigma))
    if os.path.exists(outbase+'.nii.gz'):
        continue
    ivim.sim.noise(parmap_files['D'], parmap_files['f'], 'ballistic', bval_file, noise_sigma, outbase, vd_file = parmap_files['vd'], cval_file = cval_file)

# Fit to phantom data
for noise_sigma in noise_sigmas:
    snr = 1/noise_sigma
    im_file = os.path.join(root,'phantom','phantom_SNR-{:.0f}_IVIM-avg.nii.gz'.format(snr))

    outbase_seg = os.path.join(root,'phantom','phantom_SNR-{:.0f}_IVIM-seg'.format(snr))
    if os.path.exists(outbase_seg+'.nii.gz'):
        continue
    ivim.fit.seg(im_file, bval_file, 'ballistic', bthr = 100, cval_file = cval_file, cthr = 0, outbase=outbase_seg, verbose=True, roi_file=roi_file)

    outbase_nlls = os.path.join(root,'phantom','phantom_SNR-{:.0f}_IVIM-nlls'.format(snr))
    ivim.fit.nlls(im_file, bval_file, 'ballistic', cval_file = cval_file, outbase=outbase_nlls, verbose=True, roi_file=roi_file)

    n = 5000
    burns = 1000

    outbase_bayesu = os.path.join(root,'phantom','phantom_SNR-{:.0f}_IVIM-bayesu'.format(snr))
    ivim.fit.bayes(im_file, bval_file, 'ballistic', cval_file = cval_file, outbase=outbase_bayesu, verbose=True, n=n, burns=burns, ctm='mean', roi_file=roi_file)
    
    outbase_bayess = os.path.join(root,'phantom','phantom_SNR-{:.0f}_IVIM-bayess'.format(snr))
    ivim.fit.bayes(im_file, bval_file, 'ballistic', cval_file = cval_file, outbase=outbase_bayess, verbose=True, n=n, burns=burns, ctm='mean', spatial_prior = True, roi_file=roi_file)
