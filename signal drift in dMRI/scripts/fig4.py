import matplotlib.pyplot as plt
import numpy as np

import os
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import json
import ivim

root = '/path/to/root'
input_root = os.path.join(root,'data')
results_root = os.path.join(root,'results')

subs = [sub for sub in os.listdir(results_root) if sub.startswith('sub')]
scan = 'sIVIM'
runs = ['run-{}'.format(r+1) for r in range(2)]

rois = ['pfc','cso','cb','brain']
roi_plotnames = {'pfc':'Prefrontal white matter','cso':'Centrum Semiovale','cb':'Cerebellum','brain':'Whole brain'}
markers = ['v','<','>','^']
corrs = ['reg']
legend = ['No correction','Global temporal','Voxelwise temporal','Spatiotemporal']

# Extract data
signal = []
drift = []
x0 = []
brefs = [0,200]
for bi,_ in enumerate(brefs):
    signal.append({})
    drift.append({})
    for roi in rois:
        signal[bi][roi] = []
        drift[bi][roi] = []
        for r in range(2):
            signal[bi][roi].append({corr:[] for corr in corrs})
            drift[bi][roi].append({corr:[] for corr in corrs})
    x0.append([[],[]])

dti_vectors = ivim.io.philips.read_dti_vectors_input(os.path.join('protocol','dti_vectors_input_sIVIM3b.txt'))
unique_200vectors = np.unique(np.round(dti_vectors[:,:3]),axis=0)

for sub in subs:
    for roi in rois:
        if roi == 'brain':
            roi_file = os.path.join(results_root,sub,'{}_mask.nii.gz'.format(sub))
        else:
            roi_file = os.path.join(results_root,sub,'{}_ROI-{}.nii.gz'.format(sub,roi))
        t0 = 0
        for r,run in enumerate(runs):
            for bi,bref in enumerate(brefs):
                json_file = os.path.join(input_root,sub,'{}_{}_{}.json'.format(sub,run,scan))
                bval_file = json_file.replace('json','bval')
                with open(json_file) as f:
                    d = json.load(f)
                tr = d['RepetitionTime']            
                acq_time = d['AcquisitionTime'].split(':')
                t_start = float(acq_time[0])*3600 + float(acq_time[1])*60 + float(acq_time[2].replace(',','.')) 
                
                if bref > 0:
                    t = np.zeros((unique_200vectors.shape[0],np.sum(dti_vectors[:,3]==bref)//unique_200vectors.shape[0]))
                    for idx,unique_200vector in enumerate(unique_200vectors):
                        t[idx,:] = np.squeeze(np.argwhere(np.all((np.round(dti_vectors[:,:3])==unique_200vector),axis=1)&(dti_vectors[:,3]==bref)))
                else: # b == 0
                    t = np.argwhere(dti_vectors[:,3]==bref)
                x0[bi][r].append(t*tr*2+t_start) # x2 to account for Nyquist correction

                for corr in corrs:
                    if corr == 'reg':
                        im_file = os.path.join(results_root,sub,'{}_{}_{}-reg.nii.gz'.format(sub,run,scan))
                    else:
                        im_file = os.path.join(results_root,sub,'{}_{}_{}-{}.nii.gz'.format(sub,run,scan,corr)) 
                    Y,b = ivim.io.base.data_from_file(im_file,bval_file,roi_file=roi_file)

                    Y0 = Y[:,b==0]
                    if bref > 0:
                        Yref = np.zeros((unique_200vectors.shape[0],np.sum(dti_vectors[:,3]==bref)//unique_200vectors.shape[0]))
                        k = np.zeros(unique_200vectors.shape[0])
                        for idx,unique_200vector in enumerate(unique_200vectors):
                            y = Y[:,np.all(np.round(dti_vectors[:,:3])==unique_200vector,axis=1)&(dti_vectors[:,3]==bref)]
                            Yref[idx,:] = np.median(y,axis=0)/np.median(Y0[:,0])*100
                            A = np.vstack((np.ones_like(x0[bi][r][-1][idx]).T,x0[bi][r][-1][idx].T)).T
                            p = np.linalg.lstsq(A,Yref[idx,:],rcond=None)[0]
                            k[idx] = p[1]
                            x = x0[bi][r][-1]
                        md = sm.MixedLM(Yref.flatten(),np.vstack((np.ones_like(x0[bi][r][-1]).flatten(),x0[bi][r][-1].flatten())).T,groups=np.repeat(np.arange(Yref.shape[0]),Yref.shape[1]))
                        mdf = md.fit()
                        signal[bi][roi][r][corr].append(Yref)
                        drift[bi][roi][r][corr].append(mdf.params[1])
                    else:
                        signal[bi][roi][r][corr].append(np.median(Y0,axis=0)/np.median(Y0[:,0])*100)
                        A = np.vstack((np.ones_like(x0[bi][r][-1]).T,x0[bi][r][-1].T)).T
                        drift[bi][roi][r][corr].append(np.linalg.lstsq(A,signal[bi][roi][r][corr][-1],rcond=None)[0][1])
                
# Plot results
fig,ax = plt.subplots(1,1,figsize=(5,5))
for ridx,(roi_name,marker) in enumerate(zip(rois,markers)):
    if roi_name == 'brain':
        roi_file = os.path.join(results_root,sub,'{}_mask.nii.gz'.format(sub))
    else:
        roi_file = os.path.join(results_root,sub,'{}_ROI-{}.nii.gz'.format(sub,roi_name))
    roi = ivim.io.base.read_im(roi_file)
    if roi_name == 'brain':
        slice_ex = (roi.shape[2]-1)//2
    else:
        slice_ex = int(np.argwhere(np.sum(np.sum(roi,axis=0),axis=0)))

    lim = [-7,7]
    ax.plot(lim,lim,'k--')
    for r in range(2):
        for ss,_ in enumerate(subs):
            ax.plot(drift[0][roi_name][r][corr][ss]*np.ones_like(drift[1][roi_name][r][corr][ss])*60*5,(drift[1][roi_name][r][corr][ss])*60*5,'k'+marker)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_ylabel('b = 200 s/mm$^2$ drift / 5 min [%]')
    ax.set_xticks(np.arange(-6,7,3))
    ax.set_yticks(np.arange(-6,7,3))

    if roi_name == rois[-1]:
        ax.set_xlabel('b = 0 drift / 5 min [%]')

for marker,roi_plotname in zip(markers,roi_plotnames.values()):
    ax.plot(-10,-10,'k'+marker,label=roi_plotname)
ax.legend()
if roi_name == rois[0]:
    titles = ['ROIs','Example\nb=0 & b=200 s/mm$^2$ \nsignal drift','Group summary\nb=0 and b=200 s/mm$^2$ \nsignal drift']
fig.subplots_adjust(wspace=0.35)
fig.savefig('figures/fig4.eps'.format(sub),bbox_inches='tight')
plt.close(fig)