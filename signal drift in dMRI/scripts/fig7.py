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

fig,axess = plt.subplots(len(scans),2,figsize=(6*(len(vars)//2+1),4*len(scans)),gridspec_kw={'width_ratios':[len(vars)//2,1]})

for scan_idx,scan in enumerate(scans):
    Xres = np.zeros((2*len(subs),len(xticks)))

    maps[scan] = []

    for s,sub in enumerate(subs):
        for r,run in enumerate(runs):
            maps[scan].append([])
            idx = 0
            for roi in rois:
                roi_file = os.path.join(results_root,sub,'{}_ROI-{}.nii.gz'.format(sub,roi))
                mask = ivim.io.base.read_im(roi_file).astype(bool)
                for var in vars:    
                    if scan == 'sIVIM': # requires custom averaging to keep the repetitions of b-values
                        im_file = os.path.join(results_root,sub,'{}_{}_{}-{}.nii.gz'.format(sub,run,scan,var))
                        if var == 'bordered':
                            bval_file = im_file.replace('nii.gz','bval')
                        else:
                            bval_file = os.path.join(input_root,sub,'{}_{}_{}.bval'.format(sub,run,scan))
                        bvec_file = bval_file.replace('bval','bvec')
                        Yfull,bfull,bvec = ivim.io.base.data_from_file(im_file,bval_file,bvec_file=bvec_file)
                        ubs = np.unique(bfull)
                        n = np.zeros_like(ubs) # repetitions at each b-value
                        for ui,ub in enumerate(ubs):
                            n[ui] = np.sum(bfull==ub)/6

                        b = np.zeros(int(np.sum(n)))
                        Y = np.zeros((Yfull.shape[0],b.size))
                        
                        taken = np.full(bfull.size,False)
                        n_taken = np.zeros_like(n)
                        b_idx = 0
                        for ii,bb in enumerate(bfull):
                            if taken[ii]:
                                continue
                            if bb == 0:
                                candidates = (bfull==bb)&(~taken)
                                Y[:,b_idx] = np.mean(Yfull[:,candidates][:,:6],axis=1) # first six not taken
                                taken_candidates = np.full(np.sum(candidates),False)
                                taken_candidates[:6] = True
                                taken[candidates] = taken_candidates
                            else:
                                vidx = np.zeros(6,dtype=int)
                                vii = 0
                                for d in range(3):
                                    for sign in [-1,1]:
                                        v = np.zeros(3)
                                        v[d] = sign
                                        vidx[vii] = np.argmax(np.all(np.round(bvec)==v[:,np.newaxis],axis=0) & (bfull==bb) & ~taken)
                                        vii += 1
                                Y[:,b_idx] = np.mean(Yfull[:,vidx],axis=1)
                                taken[vidx] = True
                            b[b_idx] = bb
                            b_idx += 1
                    else:
                        im_file = os.path.join(results_root,sub,'{}_{}_{}-{}-avg.nii.gz'.format(sub,run,scan,var))
                        bval_file = im_file.replace('nii.gz','bval')
                        if scan == 'IVIM-NFC':
                            Y,b,c = ivim.io.base.data_from_file(im_file,bval_file,cval_file=bval_file.replace('bval','cval'))
                        else:
                            Y,b = ivim.io.base.data_from_file(im_file,bval_file)

                    S0_file = os.path.join(results_root,sub,'{}_{}_{}-{}_S0.nii.gz'.format(sub,run,scan,var))
                    D_file = os.path.join(results_root,sub,'{}_{}_{}-{}_D.nii.gz'.format(sub,run,scan,var))
                    f_file = os.path.join(results_root,sub,'{}_{}_{}-{}_f.nii.gz'.format(sub,run,scan,var))
                    S0 = ivim.io.base.read_im(S0_file).flatten()
                    D = ivim.io.base.read_im(D_file).flatten()
                    f = ivim.io.base.read_im(f_file).flatten()

                    if scan == 'IVIM-10b':
                        Dstar_file = os.path.join(results_root,sub,'{}_{}_{}-{}_Dstar.nii.gz'.format(sub,run,scan,var))
                        Dstar = ivim.io.base.read_im(Dstar_file).flatten()
                        Yhat = ivim.models.diffusive(b,D,f,Dstar,S0)
                    elif scan == 'IVIM-NFC':
                        vd_file = os.path.join(results_root,sub,'{}_{}_{}-{}_vd.nii.gz'.format(sub,run,scan,var))
                        vd = ivim.io.base.read_im(vd_file).flatten()
                        Yhat = ivim.models.ballistic(b,c,D,f,vd,S0)
                    else:
                        Yhat = S0[:,np.newaxis]*((1-f[:,np.newaxis])*np.exp(-np.outer(D,b))+f[:,np.newaxis]*(b==0)[np.newaxis,:])
                    res = np.mean(np.abs(Yhat-Y),axis=1)/S0
                    resmap = np.reshape(res,mask.shape)

                    maps[scan][2*s+r].append(resmap)
                    Xres[2*s+r,idx] = np.nanmedian(res[mask.flatten()])
                    idx += 1
    data[scan] = Xres

idx = 0
for sub in subs:
    for run in runs:
        fig,axess = plt.subplots(len(scans),2,figsize=(3*(len(vars)+1),4*len(scans)),gridspec_kw={'width_ratios':[len(vars),1.5]})

        for scan,axes in zip(scans,axess):
            for aa,ax in enumerate(axes):
                if aa == 0:
                    sz = maps[scan][idx][0].shape
                    rm = 20
                    
                    im = np.zeros((sz[0]-2*rm,len(vars)*(sz[1]-2*rm)))
                    
                    
                    for v in range(len(vars)):
                        im[:,v*(sz[1]-2*rm):(v+1)*(sz[1]-2*rm)] = np.rot90(maps[scan][idx][v][rm:-rm,rm:-rm,sz[2]//2])
                    ax.imshow(im==0,cmap='gray',vmin=0,vmax=1,aspect='equal')
                    ax.imshow(im,cmap='jet',vmin=0,vmax=np.nanquantile(im,0.95),aspect='equal') # 

                    ax.set_xticks([])
                    ax.set_yticks([])

                    for l,text in enumerate(legend):
                        ax.text((l+1/2)*(sz[1]-2*rm),-0.03*sz[0],text,va='bottom',ha='center',fontsize=12)
                    ax.set_ylabel(plot_scan[scan],fontsize=14)
                else:
                    bp = ax.boxplot(data[scan]*100,positions=xticks,whis=[0,100],widths=1,patch_artist=True)
                    bps = []
                    for box_idx,(box,median,color) in enumerate(zip(bp['boxes'],bp['medians'],len(rois)*[colors[c_idx] for c_idx in [0,4,1,2,3]])):
                        box.set_facecolor(color)
                        median.set_color('black')
                        if box_idx < len(vars):
                            bps.append(box)
                    if scan == scans[-1]:
                        ax.legend(bps,legend,loc='upper right')
                    ax.set_ylabel('Normalized residual [%]')
                    if scan == scans[-1]:
                        ax.set_xticks([(len(vars)-1)/2+(len(vars)+1)*idx for idx in range(len(rois))])
                        ax.set_xticklabels(['Prefrontal\nwhite matter','Centrum\nsemiovale','Cerebellum'])
                    else:
                        ax.set_xticks([])
                    ax.set_xlim([xticks[0]-1,xticks[-1]+1])

        fig.savefig('figures/fig7_{}_{}.png'.format(sub,run),bbox_inches='tight',dpi=400)
        plt.close(fig)
        idx += 1
        