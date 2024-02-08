import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

tissues = {'WM':3,'CGM':2,'DGM':4}
titles = {'WM':'White matter','CGM':'Cortical gray matter','DGM':'Deep gray matter'}
pars = ['D','f','vd']
plot_par = {'D':'$D$','f':'$f$','vd':'$v_d$'}
units = {'D':'µm$^2$/ms','f':'%','vd':'mm/s'}
scale = {'D':1000,'f':100,'vd':1}
plot_range = {'D':[-0.1,3.1],'f':[-1,16],'vd':[-0.2,5.2]}
methods = ['nlls','seg','bayesu','bayess','dl']
plot_methods = {'nlls':'NLLS','seg':'Segmented','bayesu':'Bayesian: Uniform','bayess':'Bayesian: Spatial','dl':'Deep learning'}

output_root = '/path/to/results'
subs = [sub for sub in os.listdir(output_root) if sub.startswith('sub')]

fig,axess = plt.subplots(len(pars),2,figsize=(5*2,5*len(pars)))
pos = [j+1+i*(len(methods)+1) for i,_ in enumerate(tissues) for j,_ in enumerate(methods)]

for a,axes in enumerate(axess.T):
    if a == 0:
        run = 'run-1'
        title = 'in vivo'
    else: 
        run = 'SNR-150'
        title = 'Simulation'


    for ax,par in zip(axes,pars):

        data = [np.zeros(0) for _ in tissues for _ in methods]
        for sub in subs:
            output_folder = os.path.join(output_root,sub)
            mask_sub = sub

            seg = nib.load(os.path.join(output_root,mask_sub,'{}_seg-reg.nii.gz'.format(mask_sub))).get_fdata()
            mask = nib.load(os.path.join(output_root,mask_sub,'{}_IVIM-b0-mask.nii.gz'.format(mask_sub))).get_fdata().astype(bool)
            seg[~mask] = 0 # remove voxels outside mask
            
            idx = 0
            for tissue,seg_idx in tissues.items():
                for method in methods:
                    parmap = nib.load(os.path.join(output_root,sub,'{}_{}_IVIM-{}_{}.nii.gz'.format(sub,run,method,par))).get_fdata() * scale[par]
                    ex_slice = mask.shape[2]//2
                    data[idx] = np.append(data[idx],parmap[(mask&(seg==seg_idx)&~np.isnan(parmap))[...,ex_slice],ex_slice])
                    idx += 1

        parts = ax.violinplot(data,showextrema=False,positions=pos,points=1000,bw_method=0.1)
        for pc in parts['bodies']:
            pc.set_facecolor('gray')

        for i,d in enumerate(data):
            q1,q2,q3 = np.quantile(d,[0.25,0.5,0.75])
            ax.vlines(pos[i],q1,q3,color='k',lw=1)
            ax.plot(pos[i],q2,'o',color='k',markersize=3)

            ax.set_xticks([])

            if a == 0:
                ax.set_ylabel('{} [{}]'.format(plot_par[par],units[par]), fontsize=12)
            ax.set_ylim(plot_range[par])
        
        for t,tissue in enumerate(tissues.keys()):
            ax.text((t+0.5)*(len(methods)+1),
                    plot_range[par][0]-0.02*(plot_range[par][1]-plot_range[par][0]),
                    tissue,
                    verticalalignment='top',horizontalalignment='center', fontsize=12)
            if t > 0:
                ax.axvline(t*(len(methods)+1),color='k',lw=0.5)

    for m,method in enumerate(methods):
        axes[0].text(m+1,1.6,plot_methods[method],rotation=90,
                    verticalalignment='bottom',horizontalalignment='center', fontsize=12)

    axes[0].set_title(title, fontsize=15)

fig.savefig(os.path.join('figures','fig4.eps'),bbox_inches='tight')
