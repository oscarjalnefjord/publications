import os
import numpy as np
import pandas as pd
import nibabel as nib
import ivim

tissues = {'WM':3,'CGM':2,'DGM':4}
titles = {'WM':'White matter','CGM':'Cortical gray matter','DGM':'Deep gray matter'}

output_root = '/path/to/results'
subs = [sub for sub in os.listdir(output_root) if sub.startswith('sub')]
pars = ['D','f','vd','S0']

s0 = {}
snrrep = {}
snrfit = {}
for tissue in tissues.keys():
    s0[tissue] = []
    snrrep[tissue] = []
    snrfit[tissue] = []

for sub in subs:

    output_folder = os.path.join(output_root,sub)

    seg = nib.load(os.path.join(output_folder,'{}_seg-reg.nii.gz'.format(sub))).get_fdata()
    mask = nib.load(os.path.join(output_folder,'{}_IVIM-b0-mask.nii.gz'.format(sub))).get_fdata().astype(bool)
    seg[~mask] = 0 # remove voxels outside mask
    
    s0_map = nib.load(os.path.join(output_folder,'{}_IVIM-b0-mean.nii.gz'.format(sub))).get_fdata()
    
    par_maps = {}
    for par in pars:
        par_maps[par] = nib.load(os.path.join(output_folder,'{}_run-1_IVIM-nlls_{}.nii.gz'.format(sub,par))).get_fdata()
    im_file = os.path.join(output_folder,'{}_run-1_IVIM-avg.nii.gz'.format(sub))
    im = nib.load(im_file).get_fdata()
    b = ivim.io.base.read_bval(im_file.replace('nii.gz','bval'))
    c = ivim.io.base.read_cval(im_file.replace('nii.gz','cval'))
    im_hat = ivim.models.ballistic(b, c, par_maps['D'], par_maps['f'], par_maps['vd'], par_maps['S0'])
    snrfit_map = np.zeros_like(s0_map)
    snrfit_map[mask] = par_maps['S0'][mask]/np.std(im-im_hat, axis=3)[mask]

    snrrep_map = nib.load(os.path.join(output_folder,'{}_IVIM-b0-snr.nii.gz'.format(sub))).get_fdata()

    for tissue,seg_idx in tissues.items():
        seg_mask = mask&(seg==seg_idx)
        seg_mask[...,:seg_mask.shape[2]//2] = False
        seg_mask[...,seg_mask.shape[2]//2+1:] = False
        s0_median = np.nanmedian(s0_map[seg_mask])
        if tissue == 'WM':
            s0_wm = s0_median
        s0[tissue].append(s0_median/s0_wm)

        snrfit[tissue].append(np.nanmedian(snrfit_map[seg_mask]))
        snrrep[tissue].append(np.nanmedian(snrrep_map[seg_mask])*np.sqrt(6))

d_snrfit = {}
d_snrrep = {}
for tissue in tissues.keys():
    d_snrfit[titles[tissue]] = '{:.0f} ({:.0f}, {:.0f})'.format(np.median(snrfit[tissue]),np.quantile(snrfit[tissue],0.25),np.quantile(snrfit[tissue],0.75))
    d_snrrep[titles[tissue]] = '{:.0f} ({:.0f}, {:.0f})'.format(np.median(snrrep[tissue]),np.quantile(snrrep[tissue],0.25),np.quantile(snrrep[tissue],0.75))
df = pd.DataFrame([d_snrfit,d_snrrep],index=['SNRfit','SNRrep'])
df.to_csv(os.path.join('tables','table1.csv'))