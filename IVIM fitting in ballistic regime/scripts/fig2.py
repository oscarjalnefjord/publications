import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import json
import ivim

tissues = {'WM':3,'CGM':2,'DGM':4}
ylabels = {'WM':'White matter','CGM':'Cortical gray matter','DGM':'Deep gray matter'}
pars = ['D','f','vd']
plot_par = {'D':'$D$','f':' $f$ ','vd':'$v_d$'}
units = {'D':'µm$^2$/ms','f':'%','vd':'mm/s'}
scale = {'D':1000,'f':100,'vd':1}
method = 'nlls'

output_root = '/path/to/results'
with open(os.path.join('tables','example_sub.json')) as f:
    sub = json.load(f)['sub']
run = 'run-1'

output_folder = os.path.join(output_root,sub)

s0 = nib.load(os.path.join(output_folder,'{}_IVIM-b0-mean.nii.gz'.format(sub))).get_fdata()
seg = nib.load(os.path.join(output_folder,'{}_seg-reg.nii.gz'.format(sub))).get_fdata()
mask = nib.load(os.path.join(output_folder,'{}_IVIM-b0-mask.nii.gz'.format(sub))).get_fdata().astype(bool)
seg[~mask] = 0 # remove voxels outside mask

slice_ex = seg.shape[2]//2

im_file = os.path.join(output_folder,'{}_{}_IVIM-avg.nii.gz'.format(sub,run))
bval_file = im_file.replace('nii.gz','bval')
cval_file = im_file.replace('nii.gz','cval')
_,bval,cval = ivim.io.base.data_from_file(im_file,bval_file,cval_file=cval_file)
im = nib.load(im_file).get_fdata()

fig,axess = plt.subplots(len(tissues),3,figsize=(5*3,5*len(tissues)))
for c,axes in enumerate(axess.T):
    for r,(ax,(tissue,seg_idx)) in enumerate(zip(axes,tissues.items())):
        if c == 0: # display ROIs
            ax.imshow(np.rot90(s0[...,slice_ex]),cmap='gray')
            plot_mask = np.full(mask.shape[:2],np.nan)
            plot_mask[seg[...,slice_ex]==seg_idx] = 1
            ax.imshow(np.rot90(plot_mask),cmap='rainbow',interpolation='none',vmin=0,vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])

        elif c == 1: # signal vs. b of ROI average
            y = np.squeeze(np.nanmedian(im[seg==seg_idx,:],axis=0))

            filename = os.path.join(output_folder,'{}_{}_IVIM-ROI-{}.nii.gz'.format(sub,run,tissue))
            ivim.io.base.file_from_data(filename,y[np.newaxis,:],np.full((1,1,1),True))
            ivim.fit.nlls(filename,bval_file,'ballistic',cval_file=cval_file,
                                    outbase=filename.replace('.nii.gz',''))
            par_ests = {}
            for par in pars+['S0']: 
                par_ests[par] = np.squeeze(nib.load('{}_{}.nii.gz'.format(filename.replace('.nii.gz',''),par)).get_fdata())

        elif c == 2: # signal vs. b of typical voxel
            example_margin = 0.15
            df = pd.read_csv(os.path.join('tables','table2.csv'),index_col=0)
            voxel_mask = seg == seg_idx
            for par in pars:
                par_val = float(df[ylabels[tissue]][df[' ']=='Segmented']['{} [{}]'.format(par,units[par])].split()[0])/scale[par]
                par_map = nib.load(os.path.join(output_folder,'{}_{}_IVIM-{}_{}.nii.gz'.format(sub,run,method,par))).get_fdata()
                voxel_mask &= (par_map > par_val*(1-example_margin)) & (par_map < par_val*(1+example_margin))
            voxel_mask_file = os.path.join(output_folder,'{}_{}_IVIM-voxel-{}.nii.gz'.format(sub,run,tissue))
            ivim.io.base.write_im(voxel_mask_file,voxel_mask,im_file)

            Y = im[voxel_mask[...,slice_ex],slice_ex,:]
            par_maps = {}
            for par in pars+['S0']:
                par_maps[par] = nib.load(os.path.join(output_folder,'{}_{}_IVIM-{}_{}.nii.gz'.format(sub,run,method,par))).get_fdata()

            r2 = np.zeros(Y.shape[0])
            par_tmp = {}
            for i,y_tmp in enumerate(Y):
                for par in pars+['S0']:
                    par_tmp[par] = par_maps[par][voxel_mask[...,slice_ex],slice_ex][i]
                yhat = ivim.models.ballistic(bval,cval,par_tmp['D'],par_tmp['f'],par_tmp['vd'],par_tmp['S0'])
                r2[i] = np.sum((y_tmp-yhat)**2)/np.sum(y_tmp**2)

            idx = np.argsort(r2)[r2.size//2]
            y = Y[idx,:]
            for par in pars+['S0']:
                par_ests[par] = par_maps[par][voxel_mask[...,slice_ex],slice_ex][idx]

        if c > 0:
            for enc in ['FC','NC']:
                if enc == 'FC':
                    cond = cval == 0
                    fill = 'none'
                else:
                    cond = (cval>0) | (bval==0)
                    fill = 'k'
                ax.scatter(bval[cond],y[cond]/y[bval==0],facecolors=fill,edgecolors='k',label=enc)

            for enc in ['FC','NC']:
                if enc == 'FC':
                    cond = cval == 0
                else:
                    cond = (cval>0) | (bval==0)
                signal = ivim.models.ballistic(bval[cond],cval[cond],par_ests['D'],par_ests['f'],par_ests['vd'],par_ests['S0']/y[bval==0])
                if enc == 'FC':
                    ax.plot(bval[cond],np.squeeze(signal),'k-',label='fit')
                else:
                    ax.plot(bval[cond],np.squeeze(signal),'k-')

            s = ''
            for par in pars:
                s += '{} = {:0.2f} {}\n'.format(plot_par[par],par_ests[par]*scale[par],units[par])
            ax.text(0,np.min(y/y[bval==0]),s,fontsize=12)
            ax.legend()

        if r == 0:
            if c == 0:
                title = 'ROIs'
            elif c == 1:
                title = 'Signal vs. b (ROI)'
            elif c == 2:
                title = 'Signal vs. b (voxel)'
            ax.set_title(title, fontsize=15)
        if c == 0:
            ax.set_ylabel(ylabels[tissue], fontsize=15)
        if (c > 0) and (r == len(tissues)-1):
            ax.set_xlabel('b-value [s/mm$^2$]', fontsize=15)

fig.savefig(os.path.join('figures','fig2.tiff'),bbox_inches='tight')