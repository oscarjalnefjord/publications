import os
import numpy as np
from scipy.stats import rankdata, wilcoxon
import pandas as pd
import matplotlib.pyplot as plt
import json
from permutations_stats.permutations import friedman
import ivim

root = '/path/to/root'
input_root = os.path.join(root,'data')
results_root = os.path.join(root,'results')

subs = [sub for sub in os.listdir(results_root) if sub.startswith('sub')]
scans = ['sIVIM','IVIM-10b','IVIM-FC','IVIM-NC']
plot_scans = {'sIVIM':'sIVIM','IVIM-10b':'Diffusive','IVIM-FC':'Ballistic'}
runs = ['run-{}'.format(r+1) for r in range(2)]

rois = ['pfc','cso','cb','brain']
roi_plotnames = {'pfc':'Prefrontal white matter','cso':'Centrum Semiovale','cb':'Cerebellum','brain':'Whole brain'}
markers = ['o','s','d','+']
corrs = ['reg','globaltemporal','voxelwisetemporal','spatiotemporal']
legend = ['No correction','Global temporal','Voxelwise temporal','Spatiotemporal']

x0_file = os.path.join(results_root,'x0.npy')
drift_file = os.path.join(results_root,'drift.npy')
signal_file = os.path.join(results_root,'signal.npy')

# Extract data
if not os.path.exists(x0_file):
    signal = {}
    drift = {}
    x0 = {}
    for scan in scans:
        signal[scan] = {}
        drift[scan] = {}
        for roi in rois:
            signal[scan][roi] = []
            drift[scan][roi] = []
            for r in range(2):
                signal[scan][roi].append({corr:[] for corr in corrs})
                drift[scan][roi].append({corr:[] for corr in corrs})
        x0[scan] = [[],[]]

        b0s = 0
        for sub in subs:
            for roi in rois:
                if roi == 'brain':
                    roi_file = os.path.join(results_root,sub,'{}_mask.nii.gz'.format(sub))
                else:
                    roi_file = os.path.join(results_root,sub,'{}_ROI-{}.nii.gz'.format(sub,roi))
                t0 = 0
                for r,run in enumerate(runs):
                    json_file = os.path.join(input_root,sub,'{}_{}_{}.json'.format(sub,run,scan))
                    with open(json_file) as f:
                        d = json.load(f)
                    tr = d['RepetitionTime']            
                    acq_time = d['AcquisitionTime'].split(':')
                    t_start = float(acq_time[0])*3600 + float(acq_time[1])*60 + float(acq_time[2].replace(',','.')) 
                    b = ivim.io.base.read_bval(json_file.replace('json','bval'))
                    x0[scan][r].append(np.argwhere(b==0)*tr*2+t_start) # x2 to account for Nyquist correction
                    
                    bval_file = os.path.join(input_root,sub,'{}_{}_{}.bval'.format(sub,run,scan))
                    for corr in corrs:
                        if corr == 'reg':
                            im_file = os.path.join(results_root,sub,'{}_{}_{}-reg.nii.gz'.format(sub,run,scan))
                        else:
                            im_file = os.path.join(results_root,sub,'{}_{}_{}-{}.nii.gz'.format(sub,run,scan,corr)) 
                        Y,b = ivim.io.base.data_from_file(im_file,bval_file,roi_file=roi_file)

                        Y0 = Y[:,b==0]
                        signal[scan][roi][r][corr].append(np.median(Y0,axis=0)/np.median(Y0[:,0])*100)
                        drift[scan][roi][r][corr].append(np.abs(np.median(Y0[:,-1])-np.median(Y[:,0]))/np.median(Y0[:,0])*100/float(x0[scan][r][-1][-1]-x0[scan][r][-1][0])*5*60) # per 5 min
                        if b0s == 0:
                            b0s = np.sum(b==0)

                print('{} {}'.format(sub,roi))

    # Build table
    d = {}
    col_names = {'sIVIM':'mono3b','IVIM-10b':'mono10b','IVIM-FC':'bi7b-fc','IVIM-NC':'bi7b-nc'}
    index = roi_plotnames.values()
    for scan in scans:
        X = np.full((len(subs),len(rois)),np.nan)
        for r,roi in enumerate(rois):
            X[:,r] = drift[scan][roi][0]['reg']

        col_name = col_names[scan]
        p_suff = len(rois)*['']
        if friedman(X[:,:len(rois)-1]) < 0.05:
            col_name += '*'
            for r1 in range(len(rois)-1):
                for r2 in range(len(rois)-1):
                    if r1 != r2:
                        if wilcoxon(X[:, r1], X[:, r2]).pvalue < 0.05:
                            if p_suff[r1] == '':
                                p_suff[r1] = '{}'.format(r2+1)
                            else:
                                p_suff[r1] = '{},{}'.format(p_suff[r1],r2+1)
        d[col_name] = []
        for r,_ in enumerate(rois):
            d[col_name].append('{:.2f} ({:.2f}-{:.2f}){}'.format(np.median(X[:,r]),np.quantile(X[:,r],0.25),np.quantile(X[:,r],0.75),p_suff[r]))

    df = pd.DataFrame(d,index=index)
    df.to_csv('tables/table1.csv')

    np.save(x0_file,x0)
    np.save(drift_file,drift)
    np.save(signal_file,signal)

else:
    with open(os.path.join('tables','example.json'),'r') as f:
        d = json.load(f)
    example_sub = d['sub']
    example_run = d['run']

    x0 = np.load(x0_file,allow_pickle=True).item()
    drift = np.load(drift_file,allow_pickle=True).item()
    signal = np.load(signal_file,allow_pickle=True).item()

    # Identify example subject
    example_ranks = np.zeros(len(subs)*2)
    for scan in scans[:-1]:
        for roi_name in rois:
            for r,run in enumerate(runs):
                corr = corrs[0]
                example_ranks[r::2] += rankdata(np.abs(drift[scan][roi_name][r][corr] - np.median(drift[scan][roi_name][r][corr])))
                print(drift[scan][roi_name][r][corr][int(example_sub.replace('sub-',''))-1])
    example_sub = 'sub-{:02d}'.format(np.argmin(example_ranks)//2+1)
    example_run = 'run-{:d}'.format(np.argmin(example_ranks)%2+1)
    with open(os.path.join('tables','example.json'),'w') as f:
        json.dump({'sub':example_sub,'run':example_run},f)

    # Plot results
    b0_file = os.path.join(results_root,example_sub,f'{example_sub}_b0ref.nii.gz')
    S0 = ivim.io.base.read_im(b0_file)

    widths = [1,1.2,1.2]
    fig,axess = plt.subplots(len(rois),3,figsize=(np.sum(widths)*3,len(rois)*3),gridspec_kw={'width_ratios':widths})
    for axes,roi_name in zip(axess,rois):
        if roi_name == 'brain':
            roi_file = os.path.join(results_root,example_sub,f'{example_sub}_mask.nii.gz')
        else:
            roi_file = os.path.join(results_root,example_sub,f'{example_sub}_ROI-{roi_name}.nii.gz')
        roi = ivim.io.base.read_im(roi_file)
        if roi_name == 'brain':
            slice_ex = (roi.shape[2]-1)//2
        else:
            slice_ex = int(np.argwhere(np.sum(np.sum(roi,axis=0),axis=0)))
        for a,ax in enumerate(axes):
            if a == 0:
                ax.imshow(np.rot90(S0[...,slice_ex]),cmap='gray')
                plot_mask = np.full(roi.shape[:2],np.nan)
                plot_mask[roi[...,slice_ex]==1] = 1
                ax.imshow(np.rot90(plot_mask),cmap='rainbow',interpolation='none',vmin=0,vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel(roi_plotnames[roi_name])
            elif a == 1:
                ylim = [95,105]
                t0 = 0
                t0_old = 0
                tmax = []
                xticks = []
                xticklabels = []
                for scan in scans[:-1]:
                    r = int(example_run[-1])-1
                    for o,corr in enumerate(corrs):
                        s = int(example_sub.replace('sub-',''))-1
                        t = (x0[scan][r][s]-x0[scan][0][s][0])/60
                        ax.plot(t+t0,signal[scan][roi_name][r][corr][s],color='C{}'.format(o))
                    tmax.append(t[-1])
                    ticks = np.arange(0,t[-1],2)
                    xticklabels += [f'{int(i)}' for i in ticks]
                    xticks += list(ticks+t0)
                    ax.vlines(t0,ylim[0],ylim[1],'k',lw=0.5)
                    t0 += 1 + t[-1]
                    ax.text((t0+t0_old)/2,ylim[1]-0.1*(ylim[1]-ylim[0]),plot_scans[scan],ha='center')
                    t0_old = t0+0
                    

                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
                if roi_name == rois[-1]:
                    ax.set_xlabel('Time from start of scan [min]')
                
                ax.set_ylim(ylim)
                ax.set_ylabel('Signal [a.u.]')
            else:
                bps = []
                for idx,scan in enumerate(scans[:-1]):
                    r = int(example_run[-1])-1
                    for o,corr in enumerate(corrs):
                        data = []
                        for r in range(2):
                            data += drift[scan][roi_name][r][corr]
                            if scan == 'IVIM-FC':
                                data += drift['IVIM-NC'][roi_name][r][corr]
                        bp = ax.boxplot(data,positions=[o+idx*(len(corrs)+2)],whis=[0,100],widths=1,patch_artist=True)
                        bp['boxes'][0].set_facecolor('C{}'.format(o))
                        bp['medians'][0].set_color('black')
                        if (idx == 0) and (scan == scans[0]):
                            bps.append(bp['boxes'][0])
                    if idx > 0:
                        ax.vlines(idx*(len(corrs)+2)-1.5,0,12,'k',lw=0.5)
                x = [(len(corrs)-1)/2+idx*(len(corrs)+2) for idx in range(len(scans)-1)]
                ax.set_xticks([])#)
                if roi_name == rois[-1]:
                    xticklabels = plot_scans.values()
                else:
                    ax.set_xticks([])
                if roi_name == rois[1]:
                    ax.legend(bps,legend,loc='upper left')
                else:
                    for xx,plot_scan in zip(x,plot_scans.values()):
                        ax.text(xx,11,plot_scan,ha='center')
                ax.set_xlim([-1,(len(scans)-1)*(len(corrs)+2)-1])
                ax.set_ylim([0,12])
                ax.set_ylabel('Signal drift per 5 min [%]')
            if roi_name == rois[0]:
                titles = ['ROIs','Example\nb=0 signal drift','Group summary\nb=0 signal drift']
                ax.set_title(titles[a])
    fig.subplots_adjust(wspace=0.35)
    fig.savefig(f'figures/fig2.png',bbox_inches='tight',dpi=400)
    plt.close(fig)

