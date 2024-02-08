import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

tissues = {'WM':3,'CGM':2,'DGM':4}
titles = {'WM':'White matter','CGM':'Cortical gray matter','DGM':'Deep gray matter'}
pars = ['D','f','vd']
plot_par = {'D':'$D$','f':' $f$ ','vd':'$v_d$'}
units = {'D':'µm$^2$/ms','f':'%','vd':'mm/s'}
scale = {'D':1000,'f':100,'vd':1}
plot_range = [{'D':[[-0.20,0.12],[-0.25,0.20]],'f':[[-1,12.5],[-5,30]],'vd':[[-0.75,1.7],[-0.4,2.5]]},{'D':[[0,0.75],[0,0.85]],'f':[[0,20],[0,35]],'vd':[[0,2.6],[-0.1,2.6]]}]
methods = ['nlls','seg','bayesu','bayess','dl']
plot_methods = {'nlls':'NLLS','seg':'Segm.','bayesu':'B: Uni.','bayess':'B: Spat','dl':'DL'}
markers = ['v','^','<','>','D']
SNRs = np.arange(25,301,25)
xlim = [-50,325]

output_root = '/path/to/results'

for j in range(2):
    if j == 0:
        subs = [sub for sub in os.listdir(output_root) if sub.startswith('sub')]
    else:
        subs = ['phantom']

    for i in range(2):
        fig,axess = plt.subplots(len(pars),len(tissues.keys()),figsize=(len(tissues.keys())*5,len(pars)*5))
        for r,(par,axes) in enumerate(zip(pars,axess)):
            
            for c,(ax,(tissue,seg_idx)) in enumerate(zip(axes,tissues.items())):
                if i == 0:
                    ax.plot(xlim,[0,0],color='k',lw=0.5)

                ypos = np.zeros(len(methods))
                x0 = []
                for m, (method,marker) in enumerate(zip(methods,markers)):
                    if j == 0:
                        X = np.zeros((len(subs),len(SNRs)))
                    else:
                        X = np.zeros((50,len(SNRs)))
                    for snr_idx, snr in enumerate(SNRs):
                        for sub_idx, sub in enumerate(subs):
                            if j == 0:
                                seg = nib.load(os.path.join(output_root,sub,'{}_seg-reg.nii.gz'.format(sub))).get_fdata()
                                mask = nib.load(os.path.join(output_root,sub,'{}_IVIM-b0-mask.nii.gz'.format(sub))).get_fdata().astype(bool)
                                seg[~mask] = 0 # remove voxels outside mask
                            else:
                                seg = nib.load(os.path.join(output_root,sub,'{}_seg.nii.gz'.format(sub))).get_fdata()
                            estimates = nib.load(os.path.join(output_root,sub,'{}_SNR-{}_IVIM-{}_{}.nii.gz'.format(sub,snr,method,par))).get_fdata()*scale[par]
                            if j == 0:
                                ground_truth = nib.load(os.path.join(output_root,sub,'{}_run-1_IVIM-bayess_{}.nii.gz'.format(sub,par))).get_fdata()*scale[par]
                            else:
                                ground_truth = nib.load(os.path.join(output_root,sub,'{}_{}.nii.gz'.format(sub,par))).get_fdata()*scale[par]
                            if j == 0:
                                if i == 0: # accuracy
                                    X[sub_idx,snr_idx] = np.nanmean(estimates[seg==seg_idx]-ground_truth[seg==seg_idx])
                                else: # precision
                                    X[sub_idx,snr_idx] = np.nanstd(estimates[seg==seg_idx]-ground_truth[seg==seg_idx])
                            else:
                                
                                for slice_idx in range(seg.shape[-1]):
                                    if i == 0: # accuracy
                                        X[slice_idx,snr_idx] = np.nanmean(estimates[seg[...,slice_idx]==seg_idx,slice_idx]-ground_truth[seg[...,slice_idx]==seg_idx,slice_idx])
                                    else: # precision
                                        X[slice_idx,snr_idx] = np.nanstd(estimates[seg[...,slice_idx]==seg_idx,slice_idx]-ground_truth[seg[...,slice_idx]==seg_idx,slice_idx])
                    ax.errorbar(SNRs,np.mean(X,axis=0),np.std(X,axis=0),fmt=marker+'-',color='k')
                    if c>=0:#c == 0:
                        ypos[m] = np.mean(X[:,0])
                if c>=0:#c == 0:
                    idxs = np.argsort(ypos)
                    mindiff = 0.05*(np.diff(plot_range[i][par][j]))
                    ytext = ypos[idxs]
                    for m, idx in enumerate(idxs):
                        ytext[m] = ytext[0] + np.sum(np.maximum(mindiff,np.diff(ytext))[:m]) 
                        ax.text((xlim[0]+SNRs[0])/2,ytext[m],plot_methods[methods[idx]],verticalalignment='center',horizontalalignment='center')
                
                if r == 0:
                    ax.set_title(titles[tissue], fontsize=15)
                if c == 0:
                    ax.set_ylabel('{} [{}]'.format(plot_par[par],units[par]), fontsize=15)

                ax.set_xlim(xlim)

                ax.set_xticks([SNRs[idx] for idx in range(len(SNRs)) if idx == 0 or idx%2 == 1])
                if r == len(pars)-1:
                    ax.set_xlabel('SNR', fontsize=15)
                ax.set_ylim(plot_range[i][par][j])
        if j == 0:
            fign = 5+i
            etx = 'eps'
        else:
            fign = 'S{}'.format(i+2)
            etx = 'tiff'
        fig.savefig(os.path.join('figures','fig{}.{}'.format(fign,etx)),bbox_inches='tight')