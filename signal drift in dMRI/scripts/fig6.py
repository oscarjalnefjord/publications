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

legend = ['Mixed protocol\nNo correction','Ordered protocol\nNo correction','Mixed protocol\nGlobal temporal','Mixed protocol\nVoxelwise temporal','Mixed protocol\nSpatiotemporal']


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

for sub in subs:

    fig,axess = plt.subplots(len(pars),2,figsize=(4.1*2*len(vars),4.1*npars),gridspec_kw={'height_ratios':[2,3,3]})

    for axes,(tech,tech_pars) in zip(axess,pars.items()):
        parmaps = {}
        for tech_par in tech_pars:
            parmaps[tech_par] = []
            for var in vars:
                for run in ['run-{}'.format(r+1) for r in range(2)]:
                    parmap_file = os.path.join(results_root,sub,'{}_{}_{}-{}_{}.nii.gz'.format(sub,run,tech,var,tech_par))
                    rm = 20
                    parmaps[tech_par].append(ivim.io.base.read_im(parmap_file)[rm:-rm,rm:-rm,:])
        sz = parmaps[tech_pars[0]][0].shape
        vmin = 0
        vmax = {'D':3e-3,'f':0.09,'Dstar':30e-3,'vd':3}
        step = {'D':1e-3,'f':0.03,'Dstar':10e-3,'vd':1}
        cb_sz = [int(sz[1]*0.8),sz[0]//5]
        im_avg = np.full((len(tech_pars)*sz[0],len(vars)*sz[1]+cb_sz[1]),np.nan)
        im_diff = np.full_like(im_avg,np.nan)
        for t,tech_par in enumerate(tech_pars):
            for v,_ in enumerate(vars):
                im_avg[t*sz[0]:(t+1)*sz[0],cb_sz[1]+v*sz[1]:cb_sz[1]+(v+1)*sz[1]] = np.rot90((parmaps[tech_par][2*v][...,sz[2]//2]+parmaps[tech_par][2*v+1][...,sz[2]//2])/2-vmin)/(vmax[tech_par]-vmin)
                im_diff[t*sz[0]:(t+1)*sz[0],cb_sz[1]+v*sz[1]:cb_sz[1]+(v+1)*sz[1]] = np.rot90(np.abs(parmaps[tech_par][2*v][...,sz[2]//2]-parmaps[tech_par][2*v+1][...,sz[2]//2]))/(vmax[tech_par]-vmin)
            im_avg[(sz[0]-cb_sz[0])//2+t*sz[0]:(sz[0]+cb_sz[0])//2+t*sz[0],:cb_sz[1]] = np.linspace(1,0,cb_sz[0])[:,np.newaxis]
            im_diff[(sz[0]-cb_sz[0])//2+t*sz[0]:(sz[0]+cb_sz[0])//2+t*sz[0],:cb_sz[1]] = np.linspace(1,0,cb_sz[0])[:,np.newaxis]
        for ax,im,title in zip(axes,[im_avg,im_diff],['Average of repeated scans','Absolute difference between repeated scans']):
            ax.imshow(~np.isnan(im),cmap='gray',vmin=0,vmax=1)
            ax.imshow(im,cmap='jet',vmin=0,vmax=1,aspect='equal')

            ax.set_xticks([])
            ax.set_yticks([])
            if tech == 'sIVIM':
                for l,text in enumerate(legend):
                    ax.text(cb_sz[1]+(l+1/2)*(sz[1]),-0.03*sz[0],text,va='bottom',ha='center',fontsize=20)
                ax.text(im_avg.shape[1]/2,-0.25*sz[0],title,va='bottom',ha='center',fontsize=25)
            for t,tech_par in enumerate(tech_pars):
                ax.text(-cb_sz[1],(t+1/2)*sz[0],f'{plot_pars[tech_par]} [{units[tech_par]}]',va='center',ha='right',fontsize=22,rotation=90) 
                steps = int((vmax[tech_par]-vmin)/step[tech_par])+1
                for i in range(steps):
                    ax.text(-0.2*cb_sz[1] ,t*sz[0]+(sz[0]+cb_sz[0])//2 - i*step[tech_par]/(vmax[tech_par]-vmin)*cb_sz[0],int(scale[tech_par]*(vmin+i*step[tech_par])),ha='right',va='center',fontsize=20)
            ax.axis('off')
        axes[0].text(-3*cb_sz[1],im_avg.shape[0]/2,plot_scan[tech],rotation=90,va='center',fontsize=25)

    fig.savefig('figures/fig6_{}.png'.format(sub),bbox_inches='tight',dpi=400)
    plt.close(fig)
    