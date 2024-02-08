import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
from permutations_stats.permutations import friedman
from scipy.stats import wilcoxon

tissues = {'WM':3,'CGM':2,'DGM':4,'CSF':1}
titles = {'WM':'White matter','CGM':'Cortical gray matter','DGM':'Deep gray matter','CSF':'Cerebrospinal fluid'}
pars = ['D','f','vd']
units = {'D':'µm$^2$/ms','f':'%','vd':'mm/s'}
scale = {'D':1000,'f':100,'vd':1}
methods = ['nlls','seg','bayesu','bayess','dl']
display_methods = {'nlls':'NLLS','seg':'Segmented','bayesu':'B: Uniform','bayess':'B: Spatial','dl':'DL'}

output_root = '/mnt/d/fc-ivim-estimation/results'
subs = [sub for sub in os.listdir(output_root) if sub.startswith('sub')]
example_ranks = np.zeros(len(subs))

for tab in range(3): # 0: mean, 1: rep, 2: std
    
    table = []
    index = []
    index_n = (len(methods)-1)//2
    for par in pars:
        X = {tissue:np.full((len(subs),len(methods)), np.nan) for tissue in tissues.keys()}
        for m,method in enumerate(methods):
            for s,sub in enumerate(subs):
                output_folder = os.path.join(output_root,sub)

                seg = nib.load(os.path.join(output_folder,'{}_seg-reg.nii.gz'.format(sub))).get_fdata()
                mask = nib.load(os.path.join(output_folder,'{}_IVIM-b0-mask.nii.gz'.format(sub))).get_fdata().astype(bool)
                seg[~mask] = 0 # remove voxels outside mask

                parmap1 = nib.load(os.path.join(output_folder,'{}_run-1_IVIM-{}_{}.nii.gz'.format(sub,method,par))).get_fdata() * scale[par]
                if tab == 1:
                    parmap2 = nib.load(os.path.join(output_folder,'{}_run-2_IVIM-{}_{}.nii.gz'.format(sub,method,par))).get_fdata() * scale[par]

                for tissue,seg_idx in tissues.items():
                    seg_mask = mask&(seg==seg_idx)
                    seg_mask[...,:seg_mask.shape[2]//2] = False
                    seg_mask[...,seg_mask.shape[2]//2+1:] = False
                    if tab == 0:
                        x = np.nanmedian(parmap1[seg_mask])
                    elif tab == 1:
                        x = np.nanmedian(np.abs(parmap1[seg_mask]-parmap2[seg_mask])/2)
                    else: # tab == 2:
                        x = np.nanquantile(parmap1[seg_mask], 0.75) - np.nanquantile(parmap1[seg_mask], 0.25)
                    X[tissue][s, m] = x

        p_friedman = np.full(len(tissues), np.nan)
        p_signrank = np.full((len(methods)**2, len(tissues)), np.nan)
        for t, tissue in enumerate(tissues.keys()):
            p_friedman[t] = friedman(X[tissue])
            if p_friedman[t] < 0.05:
                for i, _ in enumerate(methods):
                    for j, _ in enumerate(methods):
                        if i == j:
                            continue
                        res = wilcoxon(X[tissue][:, i], X[tissue][:, j])
                        p_signrank[i*len(methods)+j, t] = res.pvalue
    
        for m, method in enumerate(methods):
            d = {' ':display_methods[method]}
            for t, tissue in enumerate(tissues.keys()):
                median = np.median(X[tissue][:, m])
                if (tab < 2) and (tissue != 'csf'):
                    example_ranks += np.argsort(np.abs(X[tissue][:, m] - median))
                stats = '{:.2f} ({:.2f}, {:.2f}) '.format(median,np.quantile(X[tissue][:, m],0.25),np.quantile(X[tissue][:, m],0.75))
                for i, _ in enumerate(methods):
                    if (i != m) and (p_signrank[len(methods)*m + i, t] < 0.05):
                        stats += str(i+1)
                d[titles[tissue]] = stats
        
            index.append('{} [{}]'.format(par,units[par]))

            table.append(d)



    df = pd.DataFrame(table,index=index)
    df.to_csv(os.path.join('tables','table{}.csv'.format(tab+2)))

example_sub = 'sub-{:02d}'.format(np.argmin(example_ranks)+1)
print('Example subject: {}'.format(example_sub))
with open(os.path.join('tables','example_sub.json'),'w') as f:
    json.dump({'sub':example_sub},f)