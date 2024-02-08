import os
import nibabel as nib
import numpy as np
import ivim
import pandas as pd
import time

output_root = '/path/to/results'

subs = [sub for sub in os.listdir(output_root) if sub.startswith('sub')]
pars = ['D','f','vd']
times = {}
for method in ['seg','nlls','bayesu','bayess']:
    times[method] = []

for sub in subs:
    # Gather input files
    parmap_files = {par:os.path.join(output_root,sub,'{}_run-1_IVIM-bayess_{}.nii.gz'.format(sub,par)) for par in pars}
    bval_file = os.path.join(output_root,sub,'{}_run-1_IVIM-avg.bval'.format(sub))
    cval_file = bval_file.replace('bval','cval')

    # Generate simulated data
    noise_sigmas = [1/snr for snr in np.arange(25,301,25)] # S0 = 1
    for noise_sigma in noise_sigmas:
        outbase = os.path.join(output_root,sub,'{}_SNR-{:.0f}_IVIM-avg'.format(sub,1/noise_sigma))
        ivim.sim.noise(parmap_files['D'], parmap_files['f'], 'ballistic', bval_file, noise_sigma, outbase, vd_file = parmap_files['vd'], cval_file = cval_file)

    # Parameter estimation
    for noise_sigma in noise_sigmas:
        snr = 1/noise_sigma
        im_file = os.path.join(output_root,sub,'{}_SNR-{:.0f}_IVIM-avg.nii.gz'.format(sub,snr))
        mask_file = os.path.join(output_root,sub,'{}_IVIM-b0-mask.nii.gz'.format(sub))

        if noise_sigma == noise_sigmas[0]:
            t_start = time.time()
        outbase_seg = os.path.join(output_root,sub,'{}_SNR-{:.0f}_IVIM-seg'.format(sub,snr))
        if os.path.exists(outbase_seg+'_D.nii.gz'):
            continue
        ivim.fit.seg(im_file, bval_file, 'ballistic', bthr = 100, cval_file = cval_file, cthr = 0, roi_file = mask_file, outbase=outbase_seg, verbose=True)
        if noise_sigma == noise_sigmas[0]:
            times['seg'].append(time.time() - t_start)

        if noise_sigma == noise_sigmas[0]:
            t_start = time.time()
        outbase_nlls = os.path.join(output_root,sub,'{}_SNR-{:.0f}_IVIM-nlls'.format(sub,snr))
        ivim.fit.nlls(im_file, bval_file, 'ballistic', cval_file = cval_file, roi_file = mask_file, outbase=outbase_nlls, verbose=True)
        if noise_sigma == noise_sigmas[0]:
            times['nlls'].append(time.time() - t_start)

        n = 5000
        burns = 1000

        if noise_sigma == noise_sigmas[0]:
            t_start = time.time()
        outbase_bayesu = os.path.join(output_root,sub,'{}_SNR-{:.0f}_IVIM-bayesu'.format(sub,snr))
        ivim.fit.bayes(im_file, bval_file, 'ballistic', cval_file = cval_file, roi_file = mask_file, outbase=outbase_bayesu, verbose=True, n=n, burns=burns, ctm='mean')
        if noise_sigma == noise_sigmas[0]:
            times['bayesu'].append(time.time() - t_start)

        if noise_sigma == noise_sigmas[0]:
            t_start = time.time()
        outbase_bayess = os.path.join(output_root,sub,'{}_SNR-{:.0f}_IVIM-bayess'.format(sub,snr))
        ivim.fit.bayes(im_file, bval_file, 'ballistic', cval_file = cval_file, roi_file = mask_file, outbase=outbase_bayess, verbose=True, n=n, burns=burns, ctm='mean', spatial_prior = True)
        if noise_sigma == noise_sigmas[0]:
            times['bayess'].append(time.time() - t_start)

times_avg = {}
for method in times.keys():
    times_avg[method] = np.mean(times[method])

df = pd.DataFrame(times_avg)
df.to_csv('tables/time.csv')