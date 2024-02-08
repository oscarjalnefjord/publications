import os
import numpy as np
import matplotlib.pyplot as plt
import ivim

# protocol files
examcard_files = {enc:os.path.join('protocol','IVIM_EPI_{}_1.txt'.format(enc)) for enc in ['FC','NC']}
gfile1 = os.path.join('protocol','FWF_CUSTOM005.txt')
gfile2 = os.path.join('protocol','FWF_CUSTOM006.txt')
gfiles = {'FC': [gfile1,gfile1], 'NC':[gfile1,gfile2]}

# RF timing
with open(examcard_files['FC']) as f:
    for line in f:
        if 'Act. TE (ms)' in line:
            TE = float(line.split('"')[1])*1e-3 # ms -> s
dur_rf = 5e-3 # ad hoc
n_rf = 100
t_rf = np.linspace(0,dur_rf,n_rf) 

# plot pulse sequence
fig,ax = plt.subplots(1,1,figsize=(6,3))

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
ax.legend(loc='upper right')

Delta = (np.argmax(gnorm[1:,0]==0)+1)*dt
delta = Delta - np.argmax(gnorm[:,0]==1)*dt
ax.annotate(text='',xy=(TE/2-dur_gr/2,-0.25),xytext=(TE/2-dur_gr/2+delta,-0.25),arrowprops={'arrowstyle':'<->','shrinkA':0,'shrinkB':0})
ax.text(TE/2-dur_gr/2+delta/2,-0.3,r'$\delta=${:.1f} ms'.format(delta*1e3),verticalalignment='top',horizontalalignment='center',fontsize=7) # delta
ax.annotate(text='',xy=(TE/2-dur_gr/2,-0.55),xytext=(TE/2-dur_gr/2+Delta,-0.55),arrowprops={'arrowstyle':'<->','shrinkA':0,'shrinkB':0})
ax.text(TE/2-dur_gr/2+delta/2,-0.6,r'$\Delta=${:.1f} ms'.format(Delta*1e3),verticalalignment='top',horizontalalignment='center',fontsize=7) # Delta
ax.annotate(text='',xy=(TE/2-dur_gr/2,-0.9),xytext=(TE/2+dur_gr/2,-0.9),arrowprops={'arrowstyle':'<->','shrinkA':0,'shrinkB':0})
ax.text(TE/2,-0.95,r'$T=${} ms'.format(dur_gr*1e3),verticalalignment='top',horizontalalignment='center',fontsize=7) # encoding time

ax.text(TE/2+dur_gr/2+5e-3,0,'EPI readout',horizontalalignment='left',verticalalignment='center',bbox={'facecolor':'white','alpha':1}) # readout

ax.set_ylim([-1.1,1.1])
ax.axis('off')

fig.savefig(os.path.join('figures','fig1.eps'),bbox_inches='tight')