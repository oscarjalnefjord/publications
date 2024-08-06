import os
import numpy as np
import ivim

root = '/path/to/root'
input_root = os.path.join(root,'data')
results_root = os.path.join(root,'results')

subs = [sub for sub in os.listdir(input_root) if sub.startswith('sub-')][-4:]
scans = ['sIVIM','IVIM-10b','IVIM-FC','IVIM-NC']
runs = ['run-{}'.format(r+1) for r in range(2)]

# Generate cval files
for sub in subs:
    for run in runs:
        for scan in ['IVIM-FC','IVIM-NC']:
            bval_file = os.path.join(input_root,sub,'{}_{}_{}.bval'.format(sub,run,scan))
            enc = scan.split('-')[1]
            examcard_file = os.path.join('protocol','IVIM_EPI_{}_{}.txt'.format(enc,run.replace('run-','')))
            gfile1 = os.path.join('protocol','FWF_CUSTOM005.txt')
            if enc == 'FC':
                gfile2 = gfile1
            else:
                gfile2 = gfile1.replace('5','6')
            ivim.io.philips.cval_from_files(bval_file,examcard_file,gfile1,gfile2)

# Temporal registration
for sub in subs:
    # Extract first b = 0 image in the first scan and use as reference
    im_file_first = os.path.join(input_root,sub,'{}_run-1_sIVIM.nii.gz'.format(sub))
    if not os.path.exists(os.path.join(results_root,sub)):
        os.mkdir(os.path.join(results_root,sub))
    b0ref = os.path.join(results_root,sub,'{}_b0ref.nii.gz'.format(sub))
    idx = np.argwhere(ivim.io.base.read_bval(im_file_first.replace('nii.gz','bval'))==0)[0]
    os.system('fslroi {} {} {} 1'.format(im_file_first,b0ref,idx))

    # Generate mask
    b0ref_masked = os.path.join(results_root,sub,'{}_b0ref-masked.nii.gz'.format(sub))
    os.system('bet {} {} -m'.format(b0ref,b0ref_masked))
    mask = os.path.join(results_root,sub,'{}_mask.nii.gz'.format(sub))
    os.rename(b0ref_masked.replace('masked','masked_mask'),mask)

    # Go through scans and run registration to reference scan
    for run in runs:
        for scan in scans:

            im_file = os.path.join(input_root,sub,'{}_{}_{}.nii.gz'.format(sub,run,scan))
            im_file4reg = os.path.join(results_root,sub,'{}_{}_{}-4reg.nii.gz'.format(sub,run,scan))
            os.system('fslmerge -t {} {} {}'.format(im_file4reg,b0ref,im_file))

            im_file_reg = os.path.join(results_root,sub,'{}_{}_{}-reg.nii.gz'.format(sub,run,scan))
            os.system('eddy_correct {} {} 0'.format(im_file4reg,im_file_reg))
            os.system('fslroi {} {} 1 -1'.format(im_file_reg,im_file_reg))

# Drift corrections
for sub in subs:
    for run in runs:
        for scan in scans:
            im_file = os.path.join(results_root,sub,'{}_{}_{}-reg.nii.gz'.format(sub,run,scan))
            bval_file = os.path.join(input_root,sub,'{}_{}_{}.bval'.format(sub,run,scan))
            mask_file = os.path.join(results_root,sub,'{}_mask.nii.gz'.format(sub))
            for voxelwise in [False,True]:
                if voxelwise:
                    vx = 'voxelwise'
                else:
                    vx = 'global'
                outbase = os.path.join(results_root,sub,'{}_{}_{}-{}temporal'.format(sub,run,scan,vx))
                ivim.preproc.signal_drift.temporal(im_file,bval_file,outbase,mask_file=mask_file,order=2,ctm='median',voxelwise=voxelwise)
            outbase = os.path.join(results_root,sub,'{}_{}_{}-spatiotemporal'.format(sub,run,scan))
            ivim.preproc.signal_drift.spatiotemporal(im_file,bval_file,mask_file,outbase,order=2)

# Rearrange acquisition order
for sub in subs:
    for run in runs:
        for scan in scans:
            # Load data
            cfield_file = os.path.join(results_root,sub,'{}_{}_{}-spatiotemporal_corrfield.nii.gz'.format(sub,run,scan))
            borig_file = os.path.join(input_root,sub,'{}_{}_{}.bval'.format(sub,run,scan))
            corr_field,_ = ivim.io.base.data_from_file(cfield_file,borig_file)
            imorig_file = cfield_file.replace('_corrfield','')
            if scan in ['IVIM-FC','IVIM-NC']:
                Y,b_orig,v_orig,c_orig = ivim.io.base.data_from_file(imorig_file,borig_file,bvec_file=borig_file.replace('bval','bvec'),cval_file=borig_file.replace('bval','cval'))
            else:
                Y,b_orig,v_orig = ivim.io.base.data_from_file(imorig_file,borig_file,bvec_file=borig_file.replace('bval','bvec'))

            # Order by b-value
            idx = np.zeros_like(b_orig)
            start = 0
            stop = 0
            for j,ub in enumerate(np.sort(np.unique(b_orig))):
                bsz = np.sum(b_orig==ub)
                stop += bsz
                idx[start:stop] = np.squeeze(np.argwhere(b_orig==ub))
                start += bsz
            idx = idx.astype(int)
            Yuncorr = Y[:,idx]*(1+corr_field) # applies inverse correction
            buncorr = b_orig[idx]
            vuncorr = v_orig[:,idx]
            if scan in ['IVIM-FC','IVIM-NC']:
                cuncorr = c_orig[idx]

            # Save data
            outbase = os.path.join(results_root,sub,'{}_{}_{}-bordered'.format(sub,run,scan))
            ivim.io.base.file_from_data(outbase+'.nii.gz',Yuncorr,imref_file=cfield_file)
            ivim.io.base.write_bval(outbase+'.bval',buncorr)
            ivim.io.base.write_bvec(outbase+'.bvec',vuncorr)
            if scan in ['IVIM-FC','IVIM-NC']:
                ivim.io.base.write_cval(outbase+'.cval',cuncorr)

# Parameter estimation
for sub in subs:
    mask_file = os.path.join(results_root,sub,'{}_mask.nii.gz'.format(sub))
    for run in runs:
        for scenario in ['reg','bordered','globaltemporal','voxelwisetemporal','spatiotemporal']:
            for scan in ['sIVIM','IVIM-10b','IVIM-FC']:

                # Merge FC/NC
                if scan == 'IVIM-FC':
                    dwi_files = [os.path.join(results_root,sub,'{}_{}_IVIM-{}-{}.nii.gz'.format(sub,run,enc,scenario)) for enc in ['FC','NC']]
                    if scenario == 'bordered':
                        bval_files = [dwi_file.replace('nii.gz','bval') for dwi_file in dwi_files]
                    else:
                        bval_files = [os.path.join(input_root,sub,'{}_{}_IVIM-{}.bval'.format(sub,run,enc)) for enc in ['FC','NC']]
                    bvec_files = [bval_file.replace('bval','bvec') for bval_file in bval_files]
                    cval_files = [bval_file.replace('bval','cval') for bval_file in bval_files]
                    inbase = os.path.join(results_root,sub,'{}_{}_IVIM-NFC-{}'.format(sub,run,scenario))
                    ivim.preproc.base.combine(dwi_files,bval_files,inbase,bvec_files=bvec_files,cval_files=cval_files,normb=0,mask_file=mask_file)
                    bval_file = inbase+'.bval'
                    cval_file = inbase+'.cval'
                else:
                    inbase = os.path.join(results_root,sub,'{}_{}_{}-{}'.format(sub,run,scan,scenario))
                    if scenario == 'bordered':
                        bval_file = inbase+'.bval'
                    else:
                        bval_file = os.path.join(input_root,sub,'{}_{}_{}.bval'.format(sub,run,scan))

                # Directional average
                im_file = inbase+'.nii.gz'
                avg = inbase+'-avg'
                if scan == 'IVIM-FC':
                    ivim.preproc.base.average(im_file,bval_file,avg,cval_file=cval_file)
                else:
                    ivim.preproc.base.average(im_file,bval_file,avg)

                im_avg_file = avg+'.nii.gz'
                bval_avg_file = avg+'.bval'
                
                # Estimate parameters
                outbase = avg.replace('-avg','')
                if scan == 'sIVIM':
                    ivim.fit.seg.sIVIM(im_file,bval_file,roi=mask_file,outbase=outbase,verbose=True)
                elif scan == 'IVIM-10b':
                    ivim.fit.nlls.diffusive(im_avg_file,bval_avg_file,roi=mask_file,outbase=outbase,verbose=True)
                else: # scan == 'IVIM-NFC'
                    cval_avg_file = avg+'.cval'
                    ivim.fit.nlls.ballistic(im_avg_file,bval_avg_file,cval_avg_file,roi=mask_file,outbase=outbase,verbose=True)