#!/usr/bin/env python3
# ==============================================================================
# run_afni_condition_glm.py
#
# Hybrid Dynamic Neural Decoding Pipeline - Module 1A: 
# AFNI Condition GLM & Individual M1 Peak Localization
#
# This script executes the following for each subject's Right Learning phase (Run 3):
# 1. Task Main Effect regression (AFNI 3dDeconvolve)
# 2. Filter T-map using AAL3 Left Precentral Gyrus mask
# 3. Find the Peak Voxel coordinates with the maximum T-value
# 4. Create an exclusive M1 NIfTI mask with a 6mm radius at the peak (3dUndump)
# ==============================================================================

import os
import sys
import subprocess
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import datasets, image

# -- 1. Path and parameter settings ----------------------------------------------------------
PROJ_DIR = '/home/ser/2026_brainheck_li_project'
AFNI_DIR = f'{PROJ_DIR}/data/derivatives/afni_fmriprep'
SUBJECT_LIST = f'{PROJ_DIR}/codes/subjectlist.txt'
RADIUS = 6.0  # mm

# AFNI binary location (Python subprocess does not inherit .bashrc PATH)
AFNI_BIN = '/home/ser/abin'

def run_cmd(cmd, capture=False):
    """Run a shell command, injecting AFNI_BIN into PATH so AFNI tools are found
    even when Python subprocess does not inherit the user's .bashrc PATH."""
    import os as _os
    env = _os.environ.copy()
    env['PATH'] = AFNI_BIN + ':' + env.get('PATH', '')
    print(f'  [CMD] {cmd}')
    if capture:
        return subprocess.check_output(cmd, shell=True, env=env).decode().strip()
    else:
        ret = subprocess.run(cmd, shell=True, env=env)
        if ret.returncode != 0:
            print(f'  Execution failed:\n{cmd}')
            sys.exit(1)

def main():
    if not os.path.exists(SUBJECT_LIST):
        print(f"Cannot find subject list: {SUBJECT_LIST}")
        return

    with open(SUBJECT_LIST, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    print("Downloading/loading AAL3 atlas to get Left Precentral Gyrus mask...")
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    aal = datasets.fetch_atlas_aal()
    aal_img = nib.load(aal.maps)
    
    # Precentral_L in AAL3 is index 2001, in older AAL it is 1.
    # To be safe, we find the index directly by name
    labels = aal.labels
    precentral_l_idx = int(aal.indices[labels.index('Precentral_L')])
    print(f"Targeting Precentral_L (Index: {precentral_l_idx})")

    for subj in subjects:
        print(f"\n========================================================")
        print(f"Processing subject: {subj}")
        print(f"========================================================")
        
        subj_dir = f"{AFNI_DIR}/{subj}"
        glm_dir = f"{subj_dir}/GLM_Condition"
        os.makedirs(glm_dir, exist_ok=True)
        
        # Input file check
        fmri_r03 = f"{subj_dir}/r03_scaled.nii.gz"
        censor_r03 = f"{subj_dir}/censor_r03.1D"
        mot_r03 = f"{subj_dir}/mot_demean.r03.1D"
        stim_file = f"{PROJ_DIR}/data/derivatives/behavioral_features/{subj}/Onset_Times/Rot45_Right.1D"
        
        # Auto-generate raw onset file if missing but parametric CCN onset exists
        ccn_file = f"{PROJ_DIR}/data/derivatives/behavioral_features/{subj}/Onset_Times/Rot45_Right_CCN.1D"
        if not os.path.exists(stim_file) and os.path.exists(ccn_file):
            print(f"  [AUTO] Generating raw onset file {stim_file} from {ccn_file}...")
            with open(ccn_file, 'r') as f_in:
                lines = f_in.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                new_parts = []
                for p in parts:
                    if p == '*':
                        new_parts.append('*')
                    else:
                        new_parts.append(p.split('*')[0])
                new_lines.append(' '.join(new_parts))
            os.makedirs(os.path.dirname(stim_file), exist_ok=True)
            with open(stim_file, 'w') as f_out:
                f_out.write('\n'.join(new_lines) + '\n')
        
        if not all(os.path.exists(f) for f in [fmri_r03, censor_r03, mot_r03, stim_file]):
            print(f"Warning: {subj} is missing necessary input files, skipping.")
            continue

        # -- Step 1: Execute AFNI 3dDeconvolve --------------------------------------
        print(f"\nStep 1: Execute Task Main Effect GLM (3dDeconvolve) ...")
        stats_prefix = f"{glm_dir}/stats_task_r03"
        stats_file   = f"{stats_prefix}.nii.gz"

        if not os.path.exists(stats_file):

            # ---- Extract run-3 onset row ----------------------------------------
            # Rot45_Right.1D uses LOCAL timing (one row per run, 4 rows total).
            # 3dDeconvolve receives only r03_scaled.nii.gz (1 time block), so we
            # must provide a single-row onset file containing only run 3's events.
            # Run 3 corresponds to Block 3 (45° rotation block).
            # block_to_run mapping: {1:1, 2:2, 3:3, 6:4}  →  run 3 = row index 2.
            RUN3_ROW_INDEX = 2   # 0-based index in the 4-row LOCAL timing file

            with open(stim_file, 'r') as f_stim:
                all_rows = [ln.rstrip('\n') for ln in f_stim.readlines()]

            # Guard: file might have fewer rows than expected
            if len(all_rows) <= RUN3_ROW_INDEX:
                print(f"  [ERROR] {stim_file} has only {len(all_rows)} row(s); "
                      f"expected at least {RUN3_ROW_INDEX+1}. Skipping {subj}.")
                continue

            run3_row = all_rows[RUN3_ROW_INDEX].strip()
            if not run3_row or run3_row == '*':
                print(f"  [ERROR] Run-3 row in {stim_file} is empty or '*' — "
                      f"no rotation events found for {subj}. Skipping.")
                continue

            stim_r03_file = f"{glm_dir}/Rot45_Right_r03_only.1D"
            with open(stim_r03_file, 'w') as f_out:
                f_out.write(run3_row + '\n')
            print(f"  > Run-3 onset file written ({len(run3_row.split())} events): {stim_r03_file}")

            # ---- Remove stale partial outputs from any prior failed attempt ------
            import glob as _glob
            for stale in _glob.glob(f"{stats_prefix}*"):
                if stale != stats_file:   # keep the target .nii.gz if it exists
                    os.remove(stale)

            # ---- Run 3dDeconvolve -----------------------------------------------
            cmd = f"""
            3dDeconvolve -force_TR 2.0 \\
                -input {fmri_r03} \\
                -censor {censor_r03} \\
                -ortvec {mot_r03} mot_demean \\
                -polort 2 \\
                -num_stimts 1 \\
                -stim_times 1 {stim_r03_file} 'BLOCK(4.5,1)' \\
                -stim_label 1 RightLearning \\
                -tout -bucket {stats_prefix} \\
                -GOFORIT 5 \\
                -jobs 4
            """
            # Design notes:
            # -polort 2  : Intentionally conservative (removes only scanner drift).
            #   AFNI recommends polort=7 for ~984 s runs, but fMRIPrep + AFNI
            #   confound regression already removed slow drifts. High-order polort
            #   would absorb the slowly-evolving learning-related BOLD signal.
            # -GOFORIT 5 : Fixed ITI=6s causes low-frequency collinearity between
            #   BLOCK(4.5,1) and the polort baseline → matrix inverse error ≈0.01.
            #   Condition numbers are VERY GOOD; model is sound. -GOFORIT bypasses
            #   AFNI's protective halt.
            run_cmd(cmd)

            # Convert to NIfTI: AFNI may write +orig or +tlrc depending on space.
            # Try +orig first (native space, typical for task data), fall back to +tlrc.
            for afni_suffix in ['+orig', '+tlrc']:
                afni_bucket = f"{stats_prefix}{afni_suffix}"
                head_file = f"{afni_bucket}.HEAD"
                if os.path.exists(head_file):
                    run_cmd(f"3dAFNItoNIFTI -prefix {stats_file} {afni_bucket}")
                    print(f"  > Converted {afni_bucket} to {stats_file}")
                    break
            else:
                print(f"  [WARN] Could not find AFNI stats bucket for {subj}. "
                      f"Checked: {stats_prefix}+orig, {stats_prefix}+tlrc")
                continue
        else:
            print(f"Found existing GLM stats file: {stats_file}")

        # -- Step 2: Find Individual M1 Peak -----------------------------------------
        print(f"\nStep 2: Find the Peak Voxel of the left primary motor cortex ...")

        # 3dAFNItoNIFTI may produce a 5D array (X,Y,Z,1,N_bricks) — the singleton
        # dim 3 is a leftover AFNI time dimension. np.squeeze removes it.
        # Expected sub-brick order with -tout -bucket:
        #   [0] Full_Fstat, [1] RightLearning_Coef, [2] RightLearning_Tstat
        stats_img  = nib.load(stats_file)
        stats_data = np.squeeze(stats_img.get_fdata())   # (X,Y,Z,N) or (X,Y,Z)
        print(f"  > Stats array shape after squeeze: {stats_data.shape}")

        if stats_data.ndim == 4:
            t_map_data  = stats_data[..., 2] if stats_data.shape[-1] > 2 \
                          else stats_data[..., -1]
            coef_data   = stats_data[..., 1] if stats_data.shape[-1] > 1 \
                          else stats_data[..., 0]
        elif stats_data.ndim == 3:
            t_map_data  = stats_data
            coef_data   = stats_data
        else:
            print(f"  [ERROR] Unexpected stats shape {stats_data.shape}. Skipping {subj}.")
            continue

        # Save the coefficient map as a standalone 3D NIfTI for group analysis.
        # This avoids unreliable [1] sub-brick selector syntax in tcsh/3dttest++.
        coef_file = f"{glm_dir}/Coef_RightLearning.nii.gz"
        ref_img   = nib.Nifti1Image(coef_data.astype(np.float32),
                                    stats_img.affine)
        nib.save(ref_img, coef_file)
        print(f"  > Coefficient map saved: {coef_file}")

        # Resample AAL atlas to fMRI resolution
        resampled_aal = image.resample_to_img(aal_img, stats_img, interpolation='nearest')
        aal_data = resampled_aal.get_fdata()

        # Create Precentral_L mask
        m1_mask = (aal_data == precentral_l_idx)
        
        # Mask T-map
        masked_t_map = t_map_data * m1_mask
        
        if np.max(masked_t_map) <= 0:
            print(f"Warning: No positive activation in Left Precentral Gyrus for {subj}, enabling fallback: use known left hand area (Hand Knob) coordinates [-38, -22, 56]!")
            fallback_mni = np.array([-38.0, -22.0, 56.0])
            peak_mni = fallback_mni
            # Reverse engineer Voxel coordinates for debugging consistency (though subsequent 3dUndump only needs MNI)
            inv_affine = np.linalg.inv(stats_img.affine)
            fallback_homo = np.array([-38.0, -22.0, 56.0, 1.0])
            peak_idx = np.round(inv_affine.dot(fallback_homo)[:3]).astype(int)
        else:
            peak_idx = np.unravel_index(np.argmax(masked_t_map), masked_t_map.shape)
            print(f"  > Max T-value: {masked_t_map[peak_idx]:.3f}")
            # Convert Voxel Index to MNI coordinates
            affine = stats_img.affine
            peak_idx_homo = np.array([peak_idx[0], peak_idx[1], peak_idx[2], 1])
            peak_mni = affine.dot(peak_idx_homo)[:3]
        print(f"  > Peak MNI coordinates: {peak_mni}")

        # -- Step 3: Create 6mm Exclusive Seed Sphere --------------------------------
        print(f"\nStep 3: Create 6mm radius individualized Seed mask ...")
        xyz_file  = f"{glm_dir}/peak_xyz.1D"
        seed_file = f"{glm_dir}/M1_indiv_seed.nii.gz"

        with open(xyz_file, 'w') as f:
            f.write(f"{peak_mni[0]} {peak_mni[1]} {peak_mni[2]}\n")

        # Remove any stale seed outputs (AFNI sometimes compresses to .BRIK.gz)
        seed_base = f"{glm_dir}/M1_indiv_seed"
        for stale in [
            seed_file,
            f"{seed_base}+orig.BRIK",  f"{seed_base}+orig.BRIK.gz",
            f"{seed_base}+orig.HEAD",
            f"{seed_base}+tlrc.BRIK",  f"{seed_base}+tlrc.BRIK.gz",
            f"{seed_base}+tlrc.HEAD",
        ]:
            if os.path.exists(stale):
                os.remove(stale)

        # 3dUndump writes NIfTI natively when the prefix ends in .nii.gz.
        # This avoids the AFNI BRIK -> NIfTI conversion step entirely and
        # sidesteps .BRIK.gz compression issues.
        # Argument order: ALL options before the positional xyz_file argument.
        run_cmd(
            f"3dUndump -prefix {seed_file} -master {fmri_r03} "
            f"-xyz -srad {RADIUS} {xyz_file}"
        )

        if not os.path.exists(seed_file):
            print(f"  [WARN] 3dUndump did not produce {seed_file}. Skipping {subj}.")
            continue

        print(f"  > Individualized M1 seed saved: {seed_file}")


if __name__ == '__main__':
    main()
