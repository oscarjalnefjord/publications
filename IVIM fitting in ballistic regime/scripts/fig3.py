import os
import json
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

pars = ['D','f','vd']
plot_par = {'D':'$D$','f':' $f$ ','vd':'$v_d$'}
units = {'D':'µm$^2$/ms','f':'%','vd':'mm/s'}
scale = {'D':1000,'f':100,'vd':1}
plot_range = {'D':[0,2],'f':[0,10],'vd':[0,5]}
yticks = {'D':4,'f':4,'vd':6}
methods = ['nlls','seg','bayesu','bayess','dl']
plot_methods = {'gt':'Ground truth','nlls':'NLLS','seg':'Segmented','bayesu':'Bayesian:\nUniform prior','bayess':'Bayesian:\nSpatial prior','dl':'Deep learning'}

output_root = '/path/to/results'
with open(os.path.join('tables','example_sub.json')) as f:
    sub = json.load(f)['sub']
output_folder = os.path.join(output_root,sub)

mask = np.rot90(nib.load(os.path.join(output_folder,'{}_IVIM-b0-mask.nii.gz'.format(sub))).get_fdata().astype(bool))
seg = np.rot90(nib.load(os.path.join(output_folder,'{}_seg-reg.nii.gz'.format(sub))).get_fdata())
slice_ex = mask.shape[2]//2 #13
trim = 10
sz = [mask.shape[0]-2*trim,mask.shape[1]-2*trim]
barwidth = 20
barhight = sz[0]-barwidth

fig,axes = plt.subplots(2,1,figsize=(len(methods)+1,2*3))

for ax in axes:
    im = np.full((len(pars)*sz[0],(len(methods)+1)*sz[1]),np.nan)
    if ax == axes[0]: # in vivo
        run = 'run-1'
    else: # simulation
        run = 'SNR-150'

    for p,par in enumerate(pars):
        if ax == axes[0]: # in vivo: colorbar
            im[(sz[0]-barhight)//2+p*sz[0]:(p+1)*sz[0]-(sz[0]-barhight)//2,sz[1]//2-barwidth//2:sz[1]//2+barwidth//2] = np.linspace(1,0,barhight)[:,np.newaxis]
        else: # sim
            truemap = np.rot90(nib.load(os.path.join(output_root,sub,'{}_run-1_IVIM-bayess_{}.nii.gz'.format(sub,par))).get_fdata() * scale[par])
            truemap[~mask] = np.nan
            im[p*sz[0]:(p+1)*sz[0],:sz[1]] = (truemap[trim:-trim,trim:-trim,slice_ex]-plot_range[par][0])/(plot_range[par][1]-plot_range[par][0])

        for m,method in enumerate(methods):
            parmap = np.rot90(nib.load(os.path.join(output_root,sub,'{}_{}_IVIM-{}_{}.nii.gz'.format(sub,run,method,par))).get_fdata() * scale[par])
            parmap[~mask] = np.nan
            im[p*sz[0]:(p+1)*sz[0],(m+1)*sz[1]:(m+2)*sz[1]] = (parmap[trim:-trim,trim:-trim,slice_ex]-plot_range[par][0])/(plot_range[par][1]-plot_range[par][0])


    ax.imshow(np.zeros_like(im),cmap='gray')
    ax.imshow(im,cmap='jet',vmin=0,vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])

    for m,method in enumerate(['gt']+methods):
        if (m>0) or (ax==axes[1]):
            ax.text((m+0.5)*sz[1],-10,plot_methods[method],fontsize=6,
                    horizontalalignment='center',verticalalignment='bottom')
    for p,par in enumerate(pars):
        s = '{}'.format(plot_par[par])
        if ax == axes[0]:
            s += ' [{}]'.format(units[par])
        ax.text(-10,(p+0.5)*sz[0],s,fontsize=6,
                horizontalalignment='right',verticalalignment='center',rotation=90)

    if ax == axes[0]:
        supertitle = 'in vivo'
    else:
        supertitle = 'Simulation'
    ax.text(im.shape[1]/2,-40,supertitle,fontsize=8,
            horizontalalignment='center',verticalalignment='bottom')

for p,par in enumerate(pars):
    for n in range(yticks[par]):
        axes[0].text(18,(p+1)*sz[0]-(sz[0]-barhight)/2-n/(yticks[par]-1)*barhight,
                '{:.0f}'.format(plot_range[par][0]+n/(yticks[par]-1)*(plot_range[par][1]-plot_range[par][0])),
                horizontalalignment='right',verticalalignment='center',color='w',fontsize=6)

fig.savefig(os.path.join('figures','fig3.tiff'),bbox_inches='tight',dpi=400)