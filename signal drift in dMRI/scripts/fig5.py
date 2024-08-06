import os
import numpy as np
import matplotlib.pyplot as plt
import ivim

root = '/path/to/root'
input_root = os.path.join(root,'data')
results_root = os.path.join(root,'results')

subs = [sub for sub in os.listdir(results_root) if sub.startswith('sub')]
runs = ['run-{}'.format(r+1) for r in range(2)]

scans = ['sIVIM','IVIM-10b','IVIM-NFC']
plot_scan = {'sIVIM':'sIVIM','IVIM-10b':'Diffusive','IVIM-NFC':'Ballistic'}
rois = ['pfc','cso','cb']

pars = {'sIVIM':['D','f'],'IVIM-10b':['D','f','Dstar'],'IVIM-NFC':['D','f','vd']}
npars = 8
scale = {'D':1e3,'f':1e2,'Dstar':1e3,'vd':1}
units = {'D':'µm$^2$/ms','f':'%','Dstar':'µm$^2$/ms','vd':'mm/s'}
plot_pars = {'D':'D','f':'f','Dstar':'D$^*$','vd':'v$_d$'}

title =['Average of repeated scans','Absolute difference between repeated scans']
legend = ['Mixed protocol\nNo correction','Ordered protocol\nNo correction','Mixed protocol\nGlobal temporal','Mixed protocol\nVoxelwise temporal','Mixed protocol\nSpatiotemporal']

ylim_avg = {'D':[0.4,1.2],'f':[-0.1,12],'Dstar':[-5,60],'vd':[-0.1,4]}
ylim_diff = {'D':[-0.01,0.15],'f':[-0.1,8],'Dstar':[-5,60],'vd':[-0.1,6]}

vars = ['reg','bordered','globaltemporal','spatiotemporal','voxelwisetemporal']
colors = ['C{}'.format(c) for c in [0,1,2,3,7]]
xticks = []
x = 0

for _ in range(len(rois)):
    for _ in range(len(vars)):
        xticks.append(x)
        x += 1
    x += 1

data = {}
maps = {}


fig,axess = plt.subplots(npars,2,figsize=(1.8*2*len(vars),1.8*npars))

for a,axes in enumerate(axess):
    if a < 2:
        tech = 'sIVIM'
        idx = a
    elif a < 5:
        tech = 'IVIM-10b'
        idx = a - 2
    else:
        tech = 'IVIM-NFC'
        idx = a - 5
    tech_par = pars[tech][idx]

    X = np.zeros((len(subs),len(vars)*len(rois),2))
    idx = 0
    for s,sub in enumerate(subs):
        for v,var in enumerate(vars):
            parmap1 = ivim.io.base.read_im(os.path.join(results_root,sub,'{}_run-1_{}-{}_{}.nii.gz'.format(sub,tech,var,tech_par)))
            parmap2 = ivim.io.base.read_im(os.path.join(results_root,sub,'{}_run-2_{}-{}_{}.nii.gz'.format(sub,tech,var,tech_par)))
            for r,roi_name in enumerate(rois):
                roi_file = os.path.join(results_root,sub,'{}_ROI-{}.nii.gz'.format(sub,roi_name))
                roi = ivim.io.base.read_im(roi_file).astype(bool)
                X[s,v+r*len(vars),0] = np.median((parmap1[roi]+parmap2[roi])/2)
                X[s,v+r*len(vars),1] = np.median(np.abs(parmap1[roi]-parmap2[roi]))

    for aa,ax in enumerate(axes):
        bp = ax.boxplot(X[...,aa]*scale[tech_par],positions=xticks,whis=[0,100],widths=1,patch_artist=True)
        bps = []
        for box_idx,(box,median,color) in enumerate(zip(bp['boxes'],bp['medians'],len(rois)*[colors[c_idx] for c_idx in [0,4,1,2,3]])):
            box.set_facecolor(color)
            median.set_color('black')
            if box_idx < len(vars):
                bps.append(box)
            if (aa == 1) and (a == 1):
               ax.legend(bps,legend,loc='upper right',ncol=3)
        ax.set_ylabel(f'{plot_pars[tech_par]} [{units[tech_par]}]')
        if a == (npars-1):
            ax.set_xticks([(len(vars)-1)/2+(len(vars)+1)*idx for idx in range(len(rois))])
            ax.set_xticklabels(['Prefrontal\nwhite matter','Centrum\nsemiovale','Cerebellum'])
        else:
            ax.set_xticks([])
        if a == 0:
            ax.set_title(title[aa],fontsize=15)
        ax.set_xlim([xticks[0]-1,xticks[-1]+1])
        if aa == 0:
            ax.set_ylim(ylim_avg[tech_par])
        else:
            ax.set_ylim(ylim_diff[tech_par])
    if tech_par == 'f':
        axes[0].text(-4,5+9*(tech=='sIVIM'),plot_scan[tech],rotation=90,va='center',fontsize=18)
        axes[0].plot([-3,-3],[-10+18*(tech=='sIVIM'),20+0*(tech=='sIVIM')], color='k', clip_on=False)

fig.savefig('figures/fig5.png',bbox_inches='tight',dpi=400)
plt.close(fig)
