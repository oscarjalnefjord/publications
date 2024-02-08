import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

root = '/tmp'
sub = 'phantom'

fig,axess = plt.subplots(2,3,figsize=(3*5,2*5))
for a, axes in enumerate(axess):
    for ax,parname, unit, scale, plotmax in zip(axes,['D','f','vd'],['µm2/ms','%','mm/s'],[1e3,1e2,1],[3e-3,0.6,5]):
        parmap_file = os.path.join(root,sub,'{}_validation-{}_{}.nii.gz'.format(sub,2-a,parname))
        pars_true = nib.load(parmap_file).get_fdata().flatten()*scale
        parest_file = os.path.join(root,sub,'{}_validation_IVIM-dl-{}_{}.nii.gz'.format(sub,2-a,parname))
        pars_est = nib.load(parest_file).get_fdata().flatten()*scale
        if (a==1) and (parname=='f'):
            plotmax /= 10
        ranges = [[0, plotmax*scale],[0, plotmax*scale]]
        ax.hist2d(pars_true,pars_est,bins=30,range=ranges)
        ax.plot(ranges[0],ranges[1],'w',lw=2)
        if a == 0:
            ax.set_title('{} [{}]'.format(parname,unit),fontsize=15)
            ylabel = 'Parameter ranges: Kaandorp 2021'
        else:
            ylabel = 'Parameter ranges: Brain'            
        if ax == axes[0]:
            ax.set_ylabel(ylabel,fontsize=15)
        print([np.min(pars_true), np.max(pars_true)])
fig.savefig(os.path.join('figures','validation.png'))
