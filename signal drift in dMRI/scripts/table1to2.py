import os
import numpy as np
from permutations_stats.permutations import friedman
from scipy.stats import wilcoxon
import pandas as pd
import ivim

root = '/path/to/root'
input_root = os.path.join(root,'data')
results_root = os.path.join(root,'results')

subs = [sub for sub in os.listdir(results_root) if sub.startswith('sub')]

scans = ['sIVIM','IVIM-10b','IVIM-NFC']
rois = ['pfc','cso','cb']

pars = {'sIVIM':['D','f'],'IVIM-10b':['D','f','Dstar'],'IVIM-NFC':['D','f','vd']}
scale = {'D':1e3,'f':1e2,'Dstar':1e3,'vd':1}
units = {'D':'µm$^2$/ms','f':'%','Dstar':'µm$^2$/ms','vd':'mm/s'}
plot_pars = {'D':'D','f':'f','Dstar':'D$^*$','vd':'v$_d$'}

legends = {'':['Mixed protocol','Ordered protocol'],'S':['No correction','Global temporal','Voxelwise temporal','Spatiotemporal']}

for table in range(1,3):
    if table == 1:
        vars = ['reg','bordered'] 
    else:
        vars = ['reg','globaltemporal','spatiotemporal','voxelwisetemporal']

    d = {}

    for scan_idx,scan in enumerate(scans):


        data = {}

        for par in pars[scan]:
            Xmean = np.zeros((len(subs),len(rois)*len(vars)))
            Xdiff = np.zeros((len(subs),len(rois)*len(vars)))
            Xres = np.zeros((len(subs)*2,len(rois)*len(vars)))
            idx = 0

            for roi in rois:
                for var in vars:    
                    if par != '':
                        for s,sub in enumerate(subs):
                            roi_file = os.path.join(results_root,sub,'{}_ROI-{}.nii.gz'.format(sub,roi))
                            mask = ivim.io.base.read_im(roi_file).astype(bool)
                            parmap_file = os.path.join(results_root,sub,'{}_run-1_{}-{}_{}.nii.gz'.format(sub,scan,var,par))
                            p1 = ivim.io.base.read_im(parmap_file)[mask]
                            p2 = ivim.io.base.read_im(parmap_file.replace('run-1','run-2'))[mask]
                            Xmean[s,idx] = np.nanmedian((p1+p2)/2)
                            Xdiff[s,idx] = np.nanmedian(np.abs(p1-p2))

                            for r,run in enumerate(['run-{}'.format(r+1) for r in range(2)]):
                                im_file = os.path.join(results_root,sub,'{}_{}_{}-{}-avg.nii.gz'.format(sub,run,scan,var))
                                bval_file = im_file.replace('nii.gz','bval')
                                S0_file = os.path.join(results_root,sub,'{}_{}_{}-{}_S0.nii.gz'.format(sub,run,scan,var))
                                D_file = os.path.join(results_root,sub,'{}_{}_{}-{}_D.nii.gz'.format(sub,run,scan,var))
                                f_file = os.path.join(results_root,sub,'{}_{}_{}-{}_f.nii.gz'.format(sub,run,scan,var))
                                
                                if scan == 'IVIM-NFC':
                                    Y,b,c = ivim.io.base.data_from_file(im_file,bval_file,roi_file=roi_file,cval_file=bval_file.replace('bval','cval'))
                                else:
                                    Y,b = ivim.io.base.data_from_file(im_file,bval_file,roi_file=roi_file)
                                S0 = ivim.io.base.read_im(S0_file)[mask]
                                D = ivim.io.base.read_im(D_file)[mask]
                                f = ivim.io.base.read_im(f_file)[mask]

                                if scan == 'IVIM-10b':
                                    Dstar_file = os.path.join(results_root,sub,'{}_{}_{}-{}_Dstar.nii.gz'.format(sub,run,scan,var))
                                    Dstar = ivim.io.base.read_im(Dstar_file)[mask]
                                    Yhat = ivim.models.diffusive(b,D,f,Dstar,S0)
                                elif scan == 'IVIM-NFC':
                                    vd_file = os.path.join(results_root,sub,'{}_{}_{}-{}_vd.nii.gz'.format(sub,run,scan,var))
                                    vd = ivim.io.base.read_im(vd_file)[mask]
                                    Yhat = ivim.models.ballistic(b,c,D,f,vd,S0)
                                else:
                                    Yhat = S0[:,np.newaxis]*((1-f[:,np.newaxis])*np.exp(-np.outer(D,b))+f[:,np.newaxis]*(b==0)[np.newaxis,:])
                                Xres[2*s+r,idx] = np.nanmedian(np.mean(np.abs(Yhat-Y),axis=1)/S0)
                    idx += 1
            data[par] = [Xmean,Xdiff,Xres]

        cols = [[] for _ in range(2*len(vars))]
        for s, stat in enumerate(['Avg.','Diff.']):
            for par in ['D','f','Dstar','vd']:
                for r,roi in enumerate(rois):
                    if par in pars[scan]:
                        X = data[par][s][:len(subs),r*len(vars):(r+1)*len(vars)]*scale[par]
                        p_suff = len(vars)*['']
                        if friedman(X) < 0.05:
                            for v1 in range(len(vars)):
                                for v2 in range(len(vars)):
                                    if v1 != v2:
                                        if wilcoxon(X[:, v1], X[:, v2]).pvalue < 0.05:
                                            if p_suff[v1] == '':
                                                p_suff[v1] = '{}'.format(v2+1)
                                            else:
                                                p_suff[v1] = '{},{}'.format(p_suff[v1],v2+1)

                    for v,_ in enumerate(vars):
                        if par in pars[scan]:
                            txt = '{:.2f} ({:.2f}-{:.2f}){}'.format(np.median(X[:,v]),np.quantile(X[:,v],0.25),np.quantile(X[:,v],0.75),p_suff[v])
                        else:
                            txt = 'n.a.'
                        cols[s*len(vars)+v].append(txt)                   
                    
                        
        for d_idx in range(2*len(vars)):
            d[scan+'-{}'.format(d_idx)] = cols[d_idx]

      
    df = pd.DataFrame(d)
    df.to_csv('tables/table{}.csv'.format(table))        