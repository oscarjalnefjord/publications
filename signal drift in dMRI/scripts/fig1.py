import os
import json
import numpy as np
import matplotlib.pyplot as plt
import ivim

fig = plt.figure(constrained_layout=True,figsize=(10, 5))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

examcard_filess = [{'NC':os.path.join('protocol','IVIM_10b_1.txt')},
                   {enc:os.path.join('protocol','IVIM_EPI_{}_1.txt'.format(enc)) for enc in ['FC','NC']}]
gfile1 = os.path.join('protocol','FWF_CUSTOM005.txt')
gfile2 = os.path.join('protocol','FWF_CUSTOM006.txt')
gfile3 = os.path.join('protocol','FWF_CUSTOM007.txt')
gfiless = [{'NC':[gfile3,gfile3]},{'FC': [gfile1,gfile1], 'NC':[gfile1,gfile2]}]

titles = ['Monopolar','Bipolar']

# Pulse sequence diagrams
for ax,examcard_files,gfiles,title in zip((ax1,ax2),examcard_filess,gfiless,titles):

    # RF timing
    with open(examcard_files['NC']) as f:
        for line in f:
            if 'Act. TE (ms)' in line:
                TE = float(line.split('"')[1])*1e-3 # ms -> s
    dur_rf = 5e-3 # ad hoc
    n_rf = 100
    t_rf = np.linspace(0,dur_rf,n_rf) 

    # plot pulse sequence

    ax.plot([-dur_rf,TE+10e-3],[0,0],'k') # time axis

    ax.plot(t_rf-dur_rf/2,np.sinc(np.linspace(-2,2,n_rf))/2,'k')       #  90 degree pulse
    ax.plot(t_rf-dur_rf/2+TE/2,np.sinc(np.linspace(-3,3,n_rf)),'k')    # 180 degree pulse

    style = {'FC':'-','NC':'--'}
    color = {'FC':'gray','NC':'black'}
    label = {'FC':'FC','NC':'NC'}
    for enc,examcard_file in examcard_files.items():
        gnorm,dt = ivim.io.philips.gnorm_from_txt(examcard_file,gfiles[enc][0],gfiles[enc][1])
        dur_gr = dt*(gnorm.shape[0]-1)
        tplot = np.linspace(0,dur_gr,gnorm.shape[0])+TE/2-dur_gr/2
        gplot = gnorm[:,0]
        gplot[tplot>TE/2] *= -1 # show actual gradients rather than effective ones
        ax.plot(tplot,gplot*0.8,style[enc],color=color[enc],label=label[enc]) # gradient waveforms
    if title == 'Bipolar':
        ax.legend(loc='upper right')

    if title == 'Bipolar':
        Delta = (np.argmax(gnorm[1:,0]==0)+1)*dt
        delta = Delta - np.argmax(gnorm[:,0]==1)*dt
    else:
        Delta = (np.argmax(gnorm[tplot>TE/2,0]>0)+np.sum(tplot<=TE/2)-1)*dt
        delta = np.argmax(gnorm[1:,0]==0)*dt
    
    ax.annotate(text='',xy=(TE/2-dur_gr/2,-0.55),xytext=(TE/2-dur_gr/2+Delta,-0.55),arrowprops={'arrowstyle':'<->','shrinkA':0,'shrinkB':0})
    ax.text(TE/2-dur_gr/2+Delta/2,-0.6,r'$\Delta=${:.1f} ms'.format(Delta*1e3),verticalalignment='top',horizontalalignment='center',fontsize=7) # Delta
    ax.annotate(text='',xy=(TE/2-dur_gr/2,-0.25),xytext=(TE/2-dur_gr/2+delta,-0.25),arrowprops={'arrowstyle':'<->','shrinkA':0,'shrinkB':0})
    ax.text(TE/2-dur_gr/2+delta/2,-0.3,r'$\delta=${:.1f} ms'.format(delta*1e3),verticalalignment='top',horizontalalignment='center',fontsize=7) # delta
    ax.annotate(text='',xy=(TE/2-dur_gr/2,-0.9),xytext=(TE/2+dur_gr/2,-0.9),arrowprops={'arrowstyle':'<->','shrinkA':0,'shrinkB':0})
    ax.text(TE/2,-0.95,r'$T=${} ms'.format(dur_gr*1e3),verticalalignment='top',horizontalalignment='center',fontsize=7) # encoding time

    ax.text(TE/2+dur_gr/2+5e-3,0,'EPI readout',horizontalalignment='left',verticalalignment='center',bbox={'facecolor':'white','alpha':1}) # readout

    ax.set_ylim([-1.1,1.1])
    ax.axis('off')
    ax.text(TE/2,1.1,title,ha='center',fontsize=12)

# Protocol
input_root = '/path/to/data/sub-01'

seqs = ['sIVIM','IVIM-10b','IVIM-FC','IVIM-NC']
plot_seq = {'sIVIM':'sIVIM','IVIM-10b':'Diffusive','IVIM-FC':'Ballistic FC','IVIM-NC':'Ballistic NC'}
dti_seq = {'sIVIM':'sIVIM3b','IVIM-10b':'IVIM10b','IVIM-FC':'IVIM-FNC-7b','IVIM-NC':'IVIM-FNC-7b'}
runs = [f'run-{run+1}' for run in range(2)]

start = np.inf
stop = 0
for seq in seqs:
    for r,run in enumerate(runs):
        json_file = os.path.join(input_root,'sub-01_{}_{}.json'.format(run,seq))
        with open(json_file) as f:
            d = json.load(f)
        tr = d['RepetitionTime']            
        acq_time = d['AcquisitionTime'].split(':')
        t_start = float(acq_time[0])*3600 + float(acq_time[1])*60 + float(acq_time[2].replace(',','.')) 
        acq_dur = d['AcquisitionDuration']

        bval_file = os.path.join(input_root,'sub-01_{}_{}.bval'.format(run,seq))
        bvec_file = bval_file.replace('bval','bvec')
        bval = ivim.io.base.read_bval(bval_file)
        bvec = ivim.io.base.read_bvec(bvec_file)
        
        dti_vectors = np.loadtxt('protocol/dti_vectors_input_{}.txt'.format(dti_seq[seq]))

        if start > t_start:
            start = t_start
        if stop < (t_start + acq_dur):
            stop = t_start + acq_dur

        text = '{}\n{}\n\nb [s/mm$^2$]:'.format(plot_seq[seq],'Rep. {}'.format(run[-1]))
        ax3.text(t_start+acq_dur/2,0.9,text,va='top',ha='center',fontsize=8)

        btext = ''
        for i in [0,1,2,3,-1,dti_vectors.shape[0]-2,dti_vectors.shape[0]-1]:
            if i == -1:
                btext += '\n...'
            else:
                idx = np.argmax(np.abs(dti_vectors[i,:3]))
                letter = '-' if dti_vectors[i,idx] < 0 else '+'
                letter += ['x','y','z'][idx]
                btext += '\n({:.1f}, {})'.format(dti_vectors[i,3],letter)

        ax3.text(t_start+acq_dur/2,-0.45,btext,ha='center',fontsize=7)

        p = plt.Rectangle((t_start, -0.5), acq_dur, 1.5, fill=False)
        ax3.add_patch(p)

        ax3.set_xlim([start-0.01*(stop-start),stop+0.01*(stop-start)])
        ax3.set_ylim([-1,1])

for t in np.arange(start,stop,600):
    ax3.text(t,-0.8,'{:.0f} min'.format((t-start)/60))
ax3.text((start+stop)/2,-1,'Time from first acquisition',ha='center')
ax3.axis('off')
ax3.text((start+stop)/2,1.1,'Protocol',ha='center',fontsize=12)

fig.savefig(os.path.join('figures','fig1.eps'),bbox_inches='tight')