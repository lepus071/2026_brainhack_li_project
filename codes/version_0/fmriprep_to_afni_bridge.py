#!/usr/bin/env python3
# ==============================================================================
# fmriprep_to_afni_bridge.py
#
# fMRIPrep -> AFNI GLM bridge script (Stage 2)
#
# Function:
#   1. Convert fMRIPrep output BOLD images to Percent Signal Change (mean=100)
#   2. Extract head motion parameters from confounds_timeseries.tsv -> .1D format
#   3. Create Censor mask based on Framewise Displacement > 0.5mm
#   4. Output to afni_fmriprep/{subj}/ directory for subsequent GLM scripts
#
# Task mapping (fMRIPrep task -> GLM run):
#   leftbaseline  -> run 1  (Rot0_Left)
#   rightbaseline -> run 2  (Rot0_Right)
#   rightlearning -> run 3  (Rot45_Right, learning block)
#   lefttransfer  -> run 4  (Rot45_Left)
# ==============================================================================

import os
import sys
import subprocess
import numpy as np
import pandas as pd

# -- Task order definition ------------------------------------------------------------
TASK_ORDER = [
    'leftbaseline',
    'rightbaseline',
    'rightlearning',
    'lefttransfer',
]
SPACE = 'MNI152NLin6Asym'
FD_THRESHOLD = 0.5

# -- Path settings -----------------------------------------------------------------
PROJ_DIR    = '/home/ser/2026_brainheck_li_project'
FMRIPREP_DIR = f'{PROJ_DIR}/data/derivatives/fmriprep'
OUT_BASE    = f'{PROJ_DIR}/data/derivatives/afni_fmriprep'
SESSION     = 'ses-02'


def run_cmd(cmd):
    """Execute Shell command and output in real-time"""
    print(f'  $ {cmd}')
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f'Command failed! Exit code: {ret.returncode}')
        sys.exit(1)


def scale_bold(in_nii, out_prefix, mask_nii):
    """Convert BOLD image to Percent Signal Change (mean = 100)"""
    mean_file = f'{out_prefix}_mean.nii.gz'
    resampled_mask = f'{out_prefix}_mask_resampled.nii.gz'
    
    # 1. Calculate the time axis average for each Voxel
    run_cmd(f'3dTstat -overwrite -mean -prefix {mean_file} {in_nii}')
    
    # 2. Resample the brain mask to the same resolution and grid as BOLD (solves 204838 vs 902629 voxel mismatch)
    run_cmd(f'3dresample -overwrite -master {in_nii} -input {mask_nii} -prefix {resampled_mask}')
    
    # 3. Conversion: value / mean * 100, calculated only within the resampled mask
    run_cmd(
        f'3dcalc -overwrite -a {in_nii} -b {mean_file} -m {resampled_mask} '
        f'-expr "m * a/b * 100" '
        f'-prefix {out_prefix}_scaled.nii.gz'
    )
    
    # Clean up temporary files
    if os.path.exists(mean_file):
        os.remove(mean_file)
    if os.path.exists(resampled_mask):
        os.remove(resampled_mask)
        
    print(f'  Percent Signal Change completed: {out_prefix}_scaled.nii.gz')


def extract_confounds(tsv_path, out_1d, out_censor):
    """Extract motion parameters from confounds TSV and create Censor file"""
    df = pd.read_csv(tsv_path, sep='\t')

    # Motion parameters (6): translation + rotation
    motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    available = [c for c in motion_cols if c in df.columns]
    if len(available) < 6:
        print(f'  Warning: Could not find all 6 motion parameters, only found: {available}')

    motion_df = df[available].fillna(0)
    np.savetxt(out_1d, motion_df.values, fmt='%.8f')
    print(f'  Motion parameters .1D completed: {out_1d} (shape: {motion_df.shape})')

    # Censor file: TR with FD > 0.5 marked as 0 (exclude)
    if 'framewise_displacement' in df.columns:
        fd = df['framewise_displacement'].fillna(0).values
        # The first TR's FD is usually NaN, set to 0 (do not exclude)
        censor = (fd <= FD_THRESHOLD).astype(int)
    else:
        print('  Warning: Could not find framewise_displacement, all TRs will be included in the analysis')
        censor = np.ones(len(df), dtype=int)

    np.savetxt(out_censor, censor, fmt='%d')
    n_censored = np.sum(censor == 0)
    n_total    = len(censor)
    print(f'  Censor file completed: {out_censor} (excluded {n_censored}/{n_total} TRs)')


def process_subject(subj):
    fmriprep_subj = os.path.join(FMRIPREP_DIR, subj, SESSION, 'func')
    out_dir       = os.path.join(OUT_BASE, subj)
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n{"="*55}')
    print(f'  Start bridging process: {subj}')
    print(f'  Output directory: {out_dir}')
    print(f'{"="*55}')

    for run_idx, task in enumerate(TASK_ORDER, start=1):
        print(f'\n> Run {run_idx}: task-{task}')

        # Auto-detect Run number (solves issues like sub-03 having run-2)
        import glob
        search_pattern = os.path.join(fmriprep_subj, f'{subj}_{SESSION}_task-{task}_run-*_space-{SPACE}_desc-smoothAROMAnonaggr_bold.nii.gz')
        found_bolds = glob.glob(search_pattern)
        
        if not found_bolds:
            print(f'  Could not find BOLD file matching pattern: {search_pattern}')
            sys.exit(1)
            
        bold_file = found_bolds[0]
        prefix_fp = os.path.basename(bold_file).split(f'_space-{SPACE}')[0]
        
        mask_file  = os.path.join(fmriprep_subj,
            f'{prefix_fp}_space-{SPACE}_desc-brain_mask.nii.gz')
        conf_file  = os.path.join(fmriprep_subj,
            f'{prefix_fp}_desc-confounds_timeseries.tsv')

        # Check if auxiliary files exist
        for f in [mask_file, conf_file]:
            if not os.path.exists(f):
                print(f'  Could not find required file: {f}')
                sys.exit(1)

        # Define output paths
        out_prefix = os.path.join(out_dir, f'r{run_idx:02d}')
        out_1d     = os.path.join(out_dir, f'mot_demean.r{run_idx:02d}.1D')
        out_censor = os.path.join(out_dir, f'censor_r{run_idx:02d}.1D')

        # 1. Scale BOLD -> Percent Signal Change
        scale_bold(bold_file, out_prefix, mask_file)

        # 2. Extract motion parameters + Create Censor file
        extract_confounds(conf_file, out_1d, out_censor)

    # Combine Censor from all runs into one combined Censor file
    print('\n> Combining Censor files from all Runs...')
    all_censor = []
    for run_idx in range(1, len(TASK_ORDER) + 1):
        c = np.loadtxt(os.path.join(out_dir, f'censor_r{run_idx:02d}.1D'), dtype=int)
        all_censor.append(c)
    combined_censor = np.concatenate(all_censor)
    combined_path   = os.path.join(out_dir, f'censor_{subj}_combined.1D')
    np.savetxt(combined_path, combined_censor, fmt='%d')
    print(f'  Combined Censor completed: {combined_path} (total {len(combined_censor)} TRs)')

    print(f'\n{subj} bridge processing completed!')
    print(f'   Preprocessed BOLD images: {out_dir}/r01_scaled.nii.gz ~ r04_scaled.nii.gz')
    print(f'   Motion parameters: {out_dir}/mot_demean.r01.1D ~ r04.1D')
    print(f'   Censor:   {combined_path}')


# -- Main Program -------------------------------------------------------------------
if __name__ == '__main__':
    subj_list_path = os.path.join(PROJ_DIR, 'codes', 'subjectlist.txt')
    if not os.path.exists(subj_list_path):
        print(f'Could not find subject list: {subj_list_path}')
        sys.exit(1)

    with open(subj_list_path) as f:
        subjects = [
            f'sub-{line.strip().lstrip("sub-")}' 
            for line in f if line.strip()
        ]

    print(f'Start bridging process for {len(subjects)} subjects: {subjects}')
    for subj in subjects:
        fp_dir = os.path.join(FMRIPREP_DIR, subj, SESSION, 'func')
        if not os.path.isdir(fp_dir):
            print(f'Warning: Could not find fMRIPrep output, skipping: {subj}')
            continue
        process_subject(subj)

    print('\n\nAll subjects bridge processing completed!')
    print('Next please run: tcsh run_glm_fmriprep_Triarchic.tcsh sub-XX')
