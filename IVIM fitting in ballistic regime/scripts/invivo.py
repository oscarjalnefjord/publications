import os
import numpy as np
import nibabel as nib
import ivim

root = '/path/to/root'
input_root = os.path.join(root, 'data')
output_root = os.path.join(root, 'results')

steps = [1, 2, 3, 4]

subs = os.listdir(input_root)

for sub in subs:
    input_folder = os.path.join(input_root,sub)
    output_folder = os.path.join(output_root,sub)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # define dMRI space, run topup and generate SNR map
    if 1 in steps:
        # extract b0 from IVIM-SNR (one b = 10 was acquired for technical reasons)
        inbase_extract = os.path.join(input_folder,'{}_IVIM-SNR'.format(sub))
        outbase_extract = os.path.join(output_folder,'{}_IVIM-SNR-b0'.format(sub))
        ivim.preproc.base.extract(inbase_extract+'.nii.gz',
                                inbase_extract+'.bval',
                                outbase_extract,
                                inbase_extract+'.bvec')

        # merge into one b0 volume
        bval_file = os.path.join(output_folder,'{}_run-1_IVIM-b0.bval'.format(sub))
        ivim.io.base.write_bval(bval_file,np.atleast_1d(0))
        outbase_merge = os.path.join(output_folder,'{}_IVIM-b0s-29'.format(sub))
        ivim.preproc.base.combine(dwi_files=[outbase_extract+'.nii.gz',os.path.join(input_folder,'{}_run-1_IVIM-b0.nii.gz'.format(sub))],
                                bval_files=[outbase_extract+'.bval',bval_file],
                                outbase=outbase_merge)

        # remove extra slice (29 slices is incompatible with subsampling in topup)
        my_b0_images = os.path.join(output_folder,'{}_IVIM-b0s.nii.gz'.format(sub))
        os.system('fslroi {} {} 0 -1 0 -1 0 28 0 -1'.format(
                outbase_merge+'.nii.gz',
                my_b0_images)) 

        # generate topup input
        b = ivim.io.base.read_bval(os.path.join(output_folder,'{}_IVIM-SNR-b0.bval'.format(sub)))
        acqparams = os.path.join(output_folder,'{}_acqparams.txt'.format(sub))
        with open(acqparams,'w') as f:
            for _ in range(b.size):
                f.write('0 1 0 0.050\n')
            f.write('0 -1 0 0.050\n')

        # run topup
        my_topup_results = os.path.join(output_folder,'{}_IVIM-topup'.format(sub))
        my_field = os.path.join(output_folder,'{}_IVIM-field.nii.gz'.format(sub))
        my_unwarped_images = os.path.join(output_folder,'{}_IVIM-b0-unwarped.nii.gz'.format(sub))
        os.system('topup --imain={} --datain={} --config=b02b0.cnf --out={} --fout={} --iout={} --verbose'.format(my_b0_images,
                                                                                                                acqparams,
                                                                                                                my_topup_results,
                                                                                                                my_field,
                                                                                                                my_unwarped_images))

        # generate SNR map
        nii = nib.load(my_unwarped_images)
        im = nii.get_fdata()[...,:-1] # skip the one with reversed PE
        m = np.mean(im,axis=-1)
        s = np.std(im,axis=-1,ddof=1)
        snr = np.full(m.shape,np.nan)
        snr[s>0] = m[s>0]/s[s>0]
        for par_name,par in zip(['mean','std','snr'],[m,s,snr]):
            nii_map = nib.Nifti1Image(par,nii.affine,nii.header)
            nib.save(nii_map,os.path.join(output_folder,'{}_IVIM-b0-{}.nii.gz'.format(sub,par_name)))


    # Segmentation into white matter, cortical gray matter, deep gray matter and cerebrospinal fluid
    if 2 in steps:
        # brain extraction
        t1w = os.path.join(input_folder,'{}_T1W.nii.gz'.format(sub))
        t1w_1mm = os.path.join(output_folder,'{}_T1W-1mm.nii.gz'.format(sub))
        os.system('flirt -in {} -ref {} -out {} -applyisoxfm 1'.format(t1w,t1w,t1w_1mm))
        t1w_masked = os.path.join(output_folder,'{}_T1W-masked.nii.gz'.format(sub))
        os.system('bet {} {} -v'.format(t1w_1mm,t1w_masked))
        b0_masked = os.path.join(output_folder,'{}_IVIM-b0-masked.nii.gz'.format(sub))
        os.system('bet {} {} -m -v'.format(os.path.join(output_folder,'{}_IVIM-b0-mean.nii.gz'.format(sub)),b0_masked))
        os.rename(b0_masked.replace('masked','masked_mask'),b0_masked.replace('masked','mask'))

        # FAST: segmentation into wm, cgm, csf
        fast_seg = os.path.join(output_folder,'{}_fast'.format(sub))
        os.system('fast -v -o {} {}'.format(fast_seg,t1w_masked))

        # FIRST: segmentation of dgm (after run_first_all is a workaround to get the output that should be available)
        first_seg = os.path.join(output_folder,'{}_first'.format(sub))
        os.system('run_first_all -v -d -i {} -o {}'.format(t1w_1mm,first_seg))
        ims_corr = [os.path.join(output_folder, file) for file in os.listdir(output_folder) if file.endswith('_corr.nii.gz')]
        first_result = first_seg + '_all_fast_firstseg.nii.gz'
        cmd1 = 'fslmerge -t {}'.format(first_result)
        for im_corr in ims_corr:
            cmd1 += ' {}'.format(im_corr)
        os.system(cmd1)
        cmd2 = 'first_mult_bcorr -i {} -u {} -c {} -o {}'.format(t1w_1mm,
                                                        first_seg + '_all_fast_origsegs.nii.gz',
                                                        first_result,
                                                        first_result)
        os.system(cmd2)

        # combine segmentations as 0: background, 1: csf, 2: cgm, 3: wm, 4: dgm
        nii_fast = nib.load(fast_seg+'_seg.nii.gz')
        fast = nii_fast.get_fdata()
        first = nib.load(first_result).get_fdata()
        fast[first>0] = 4 
        seg = os.path.join(output_folder,'{}_seg.nii.gz'.format(sub))
        nii_seg = nib.Nifti1Image(fast,nii_fast.affine,header=nii_fast.header)
        nib.save(nii_seg,seg)

        # move segmentation to dMRI space
        regmat = os.path.join(output_folder,'{}_t1w-2-dwi.mat'.format(sub))
        seg_reg = os.path.join(output_folder,'{}_seg-reg.nii.gz'.format(sub))
        b0 = os.path.join(output_folder,'{}_IVIM-b0-mean.nii.gz'.format(sub))
        t1w_reg = os.path.join(output_folder,'{}_T1W-reg.nii.gz'.format(sub))
        os.system('flirt -in {} -ref {} -out {} -omat {}'.format(t1w_1mm,b0,t1w_reg,regmat)) # get xfm from reg t1w -> dwi space
        os.system('flirt -in {} -ref {} -out {} -applyxfm -init {} -interp nearestneighbour'.format(seg,b0,seg_reg,regmat)) # move segmentation to dwi space

    # Preprocessing
    if 3 in steps:
        for run in ['run-{}'.format(r+1) for r in range(2)]:
            encs = ['FC','NC']
            # generate cval files
            for enc in encs:
                bval_file = os.path.join(input_folder,'{}_{}_IVIM-{}.bval'.format(sub,run,enc))
                examcard_file = os.path.join('protocol','IVIM_EPI_{}_{}.txt'.format(enc,run.replace('run-','')))
                gfile1 = os.path.join('protocol','FWF_CUSTOM005.txt')
                if enc == 'FC':
                    gfile2 = gfile1
                else:
                    gfile2 = gfile1.replace('5','6')
                ivim.io.philips.cval_from_files(bval_file,examcard_file,gfile1,gfile2)

            # remove extra slice (29 slices is incompatible with subsampling in topup)
            dwi_files = [os.path.join(input_folder,'{}_{}_IVIM-{}.nii.gz'.format(sub,run,enc)) for enc in encs]
            for dwi_file in dwi_files:
                os.rename(dwi_file,dwi_file.replace('.nii.gz','-29.nii.gz'))
                os.system('fslroi {} {} 0 -1 0 -1 0 28 0 -1'.format(
                        dwi_file.replace('.nii.gz','-29.nii.gz'),
                        dwi_file)) 

            # combine into single file
            outbase_combine = os.path.join(output_folder,'{}_{}_IVIM'.format(sub,run))
            ivim.preproc.base.combine(dwi_files,
                                    [f.replace('nii.gz','bval') for f in dwi_files],
                                    outbase_combine,
                                    cval_files=[f.replace('nii.gz','cval') for f in dwi_files], 
                                    normb=0,
                                    roi_file = os.path.join(output_folder,'{}_IVIM-b0-mask.nii.gz'.format(sub)))

            # apply topup
            acqparams = os.path.join(output_folder,'{}_acqparams.txt'.format(sub))
            my_topup_results = os.path.join(output_folder,'{}_IVIM-topup'.format(sub))
            dwi_unwarped = outbase_combine+'-unwarped.nii.gz'

            os.system('applytopup --imain={} --datain={} --inindex=1 --topup={} --out={} --method=jac --verbose'.format(
                outbase_combine+'.nii.gz',
                acqparams,
                my_topup_results,
                dwi_unwarped))

            # eddy_correct (first add b0-mean as ref vol and remove afterwards)
            ecin = dwi_unwarped.replace('unwarped','ecin')
            ecc = dwi_unwarped.replace('unwarped','ecc')
            ecout = dwi_unwarped.replace('unwarped','ecout')

            os.system('fslmerge -t {} {} {}'.format(ecin,
                                                    os.path.join(output_folder,'{}_IVIM-b0-mean.nii.gz'.format(sub)),
                                                    dwi_unwarped))
            os.system('eddy_correct {} {} 0'.format(ecin,ecout))
            os.system('fslroi {} {} 1 -1'.format(ecout,ecc))

            # average across encoding directions
            outbase_avg = dwi_unwarped.replace('unwarped.nii.gz','avg')
            ivim.preproc.base.average(ecc,outbase_combine+'.bval',outbase_avg,
                                    cval_file=outbase_combine+'.cval')


    # IVIM parameter estimation
    if 4 in steps:
        for run in ['run-{}'.format(r+1) for r in range(2)]:
            im_file = os.path.join(output_folder,'{}_{}_IVIM-avg.nii.gz'.format(sub,run))
            bval_file = im_file.replace('.nii.gz','.bval')
            cval_file = im_file.replace('.nii.gz','.cval')
            mask_file = os.path.join(output_folder,'{}_IVIM-b0-mask.nii.gz'.format(sub))

            outbase_seg = os.path.join(output_folder,'{}_{}_IVIM-seg'.format(sub,run))
            ivim.fit.seg(im_file, bval_file, 'ballistic', cval_file = cval_file, bthr=100, cthr=0, roi_file = mask_file, outbase=outbase_seg, verbose=True)

            outbase_nlls = os.path.join(output_folder,'{}_{}_IVIM-nlls'.format(sub,run))
            ivim.fit.nlls(im_file, bval_file, 'ballistic', cval_file = cval_file, roi_file = mask_file, outbase=outbase_nlls, verbose=True)

            n = 5000
            burns = 1000

            outbase_bayesu = os.path.join(output_folder,'{}_{}_IVIM-bayesu'.format(sub,run))
            ivim.fit.bayes(im_file, bval_file, 'ballistic', cval_file = cval_file, roi_file = mask_file, outbase=outbase_bayesu, verbose=True, n=n, burns=burns, ctm='mean')

            outbase_bayess = os.path.join(output_folder,'{}_{}_IVIM-bayess'.format(sub,run))
            ivim.fit.bayes(im_file, bval_file, 'ballistic', cval_file = cval_file, roi_file = mask_file, outbase=outbase_bayess, verbose=True, spatial_prior=True, n=n, burns=burns, ctm='mean')