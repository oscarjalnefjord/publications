import os
import nibabel as nib
import matplotlib.pyplot as plt

root = '/path/to/results/phantom'
pars = ['D','f','vd']
plot_range = {'D':[0.5,1.2],'f':[0,5],'vd':[1.5,2.0]}
scale = {'D':1e3,'f':100,'vd':1}
plot_methods = {'gt':'Ground truth','nlls':'NLLS','seg':'Segmented fitting','bayesu':'Bayesian: Uniform','bayess':'Bayesian: Spatial','dl':'Deep learning'}
fig,axess = plt.subplots(len(plot_methods),len(pars),figsize=(5*len(pars),4*len(plot_methods)))
for (method,plot_method),axes in zip(plot_methods.items(),axess):
    for par,ax in zip(pars,axes):
        if method == 'gt':
            parmap_file = os.path.join(root,'phantom_{}.nii.gz'.format(par))
        else:
            parmap_file = os.path.join(root,'phantom_SNR-150_IVIM-{}_{}.nii.gz'.format(method,par))
        parmap = nib.load(parmap_file).get_fdata()[:,:,0]*scale[par]
        m = ax.imshow(parmap,cmap='gray',vmin=plot_range[par][0],vmax=plot_range[par][1])
        if ax in axess[0]:
            plt.colorbar(m, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        if ax in axess[0]:
            ax.set_title(par,fontsize=15)
        if ax == axes[0]:
            ax.set_ylabel(plot_method,fontsize=15)
fig.savefig(os.path.join('figures','figS1.tiff'),bbox_inches='tight',dpi=400)
