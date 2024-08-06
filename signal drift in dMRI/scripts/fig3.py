import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import nibabel as nib
import ivim

root = '/path/to/root'
input_root = os.path.join(root,'data')
results_root = os.path.join(root,'results')

corrs = ['voxelwisetemporal','spatiotemporal']
plot_corrs = {'voxelwisetemporal':'Voxelwise temporal \npolynomial correction','spatiotemporal':'Spatiotemporal \npolynomial correction'}
subs = [sub for sub in os.listdir(results_root) if sub.startswith('sub')]
scans = ['sIVIM','IVIM-10b','IVIM-FC','IVIM-NC']
titles = {'sIVIM':'sIVIM','IVIM-10b':'Diffusive','IVIM-FC':'Ballistic'}
runs = ['run-{}'.format(r+1) for r in range(2)]

for sub in subs:
    mask = ivim.io.base.read_im(os.path.join(results_root,sub,'{}_mask.nii.gz'.format(sub))).astype(bool)
    for run in runs:
        nii = nib.load(os.path.join(results_root,sub,f'{sub}_b0ref.nii.gz'))
        aspect = np.abs(nii.affine[2,2]/nii.affine[1,1])
        fig = plt.figure(figsize=(len(scans)*1.5*3,len(corrs)*3))
        gs = mpl.gridspec.GridSpec(2*len(corrs), len(scans), wspace=0.2, hspace=0.1)   
        ax = fig.add_subplot(gs[1:3,0])
        ax.imshow(np.rot90(nii.get_fdata()[nii.shape[0]//2,:,:].T,2),cmap='gray',aspect=aspect)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('b = 0 image',fontsize=15)
        for c,corr in enumerate(corrs):
            for s,scan in enumerate(scans[:-1]):
                ax = fig.add_subplot(gs[2*c:2*(c+1),s+1])
                corrfield = ivim.io.base.read_im(os.path.join(results_root,sub,'{}_{}_{}-{}_corrfield.nii.gz'.format(sub,run,scan,corr)))
                corrfield[~mask] = np.nan

                json_file = os.path.join(input_root,sub,'{}_{}_{}.json'.format(sub,run,scan))
                with open(json_file) as f:
                    d = json.load(f)
                tr = d['RepetitionTime']

                im = corrfield[corrfield.shape[0]//2,:,:,-1]/(corrfield.shape[-1]*2*tr)*60*5*100 # % drift / 5min

                ax.imshow(np.zeros_like(im.T),vmin=0,vmax=1,cmap='gray',aspect=aspect) # black background
                ax.imshow(np.rot90(im.T,2),vmin=-10,vmax=10,cmap='bwr',aspect=aspect,interpolation='none') # blue to red colormap
                ax.set_xticks([])
                ax.set_yticks([])
                if corr == corrs[0]:
                    ax.set_title(titles[scan] + ' protocol',fontsize=15)
                if s == 0:
                    ax.set_ylabel(plot_corrs[corr],fontsize=13)
                
        fig.savefig('figures/fig3_{}_{}.png'.format(sub,run),bbox_inches='tight',dpi=400)
        plt.close(fig)