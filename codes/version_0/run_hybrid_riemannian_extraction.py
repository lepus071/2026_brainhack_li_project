#!/usr/bin/env python3
# ==============================================================================
# run_hybrid_riemannian_extraction.py
#
# Hybrid Dynamic Neural Decoding Pipeline - Module 2:
# Riemannian Feature Extraction Engine
#
# This script strictly follows the mathematical guidelines:
# 1. Read Combined_Atlas_{subj}.nii.gz (465 ROIs)
# 2. Extract Resting-State (r01) to calculate individual baseline SPD matrix (C_ref)
# 3. Extract RightLearning (r03) to perform 30 TR sliding window estimation of Task matrix (C_task)
# 4. Use Ledoit-Wolf Shrinkage to ensure all matrices are strictly symmetric positive definite (SPD)
# 5. Calculate S_task = logm( C_ref^{-1/2} * C_task * C_ref^{-1/2} ) for alignment projection
# 6. Seed-based Filtration: Only extract features of M1 (row 0) to the other 464 ROIs from the S_task matrix
# 7. Save the filtered (N_windows, 464) feature vectors for the Machine Learning module
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from sklearn.covariance import LedoitWolf
from pyriemann.utils.base import invsqrtm, logm

PROJ_DIR = '/home/ser/2026_brainheck_li_project'
AFNI_DIR = f'{PROJ_DIR}/data/derivatives/afni_fmriprep'
SUBJECT_LIST = f'{PROJ_DIR}/codes/subjectlist.txt'
COMBINED_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases'
OUTPUT_DIR   = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/features'
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 30
TR = 2.0

def main():
    if not os.path.exists(SUBJECT_LIST):
        print(f"Cannot find subject list: {SUBJECT_LIST}")
        return

    with open(SUBJECT_LIST, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    # Instantiate LedoitWolf estimator (used to ensure SPD)
    lw_estimator = LedoitWolf()

    for subj in subjects:
        print(f"\n========================================================")
        print(f"Riemannian Feature Projection: {subj}")
        print(f"========================================================")
        
        atlas_file = f"{COMBINED_DIR}/Combined_Atlas_{subj}.nii.gz"
        rest_file  = f"{AFNI_DIR}/{subj}/r01_scaled.nii.gz"
        task_file  = f"{AFNI_DIR}/{subj}/r03_scaled.nii.gz"
        censor_rest= f"{AFNI_DIR}/{subj}/censor_r01.1D"
        censor_task= f"{AFNI_DIR}/{subj}/censor_r03.1D"
        
        if not all(os.path.exists(f) for f in [atlas_file, rest_file, task_file]):
            print(f"Warning: {subj} is missing NIfTI files, skipping.")
            continue

        # -- Step 1: Extract 465 region time series ------------------------------------------
        print("  > Extracting 465 ROIs time series...")
        masker = NiftiLabelsMasker(
            labels_img=atlas_file, 
            standardize='zscore_sample', 
            resampling_target='labels', # Ensure dimension perfectly matches atlas (will not drop missing labels)
            memory='nilearn_cache', 
            verbose=0
        )
        
        # Since we do not have a dummy fit, we need to get the number of labels to ensure there are 465 regions
        masker.fit(rest_file)
        # M1 is index 1, which corresponds to column 0 of TS (TS[:, 0])
        print(f"  > Detected {len(masker.labels_)} valid ROIs (expected 465).")

        # Read Censor processing (excluding censored TRs)
        try:
            cen_rest = np.loadtxt(censor_rest)
            ts_rest_raw = masker.transform(rest_file)
            ts_rest = ts_rest_raw[cen_rest == 1]
            
            cen_task = np.loadtxt(censor_task)
            ts_task_raw = masker.transform(task_file)
            # Note: During sliding window, removing Censor points will break time continuity.
            # Usually in the machine learning feature extraction phase, we maintain continuity, but ignore bad points during label alignment,
            # or handle them with interpolation. Here, to maintain 30 TR continuity, we directly use the complete ts_task_raw.
            ts_task = ts_task_raw 
        except Exception as e:
            print(f"  Extraction failed: {e}")
            continue
            
        print(f"  > Resting TS Shape: {ts_rest.shape}")
        print(f"  > Task TS Shape: {ts_task.shape}")

        # -- Step 2: Establish individual resting state baseline (Reference SPD Matrix) --------------------
        print("  > Calculating individual resting state baseline matrix (C_ref) and ensuring SPD...")
        lw_estimator.fit(ts_rest)
        C_ref = lw_estimator.covariance_
        
        # Calculate C_ref^{-1/2} for later use
        try:
            C_ref_invsqrt = invsqrtm(C_ref)
        except Exception as e:
            print(f"  C_ref inverse square root calculation failed (not SPD?): {e}")
            continue

        # -- Step 3: High-density sliding window and Riemannian projection --------------------------------------
        n_volumes = ts_task.shape[0]
        n_windows = n_volumes - WINDOW_SIZE + 1
        print(f"  > Executing sliding window (Size={WINDOW_SIZE}, Step=1) for a total of {n_windows} windows...")
        
        filtered_features = []
        
        for i in range(n_windows):
            window_ts = ts_task[i : i+WINDOW_SIZE]
            
            # 1. Estimate Task covariance (ensure SPD)
            lw_estimator.fit(window_ts)
            C_task = lw_estimator.covariance_
            
            # 2. Riemannian tangent space geometric projection: S_task = logm( C_ref^{-1/2} * C_task * C_ref^{-1/2} )
            # This forcibly maps C_task to a flat space centered at C_ref
            S_task = logm(C_ref_invsqrt @ C_task @ C_ref_invsqrt)
            
            # 3. Seed-based Filtration
            # S_task is a 465x465 symmetric matrix.
            # M1 is the 0th column/row. We only need the connection weights between M1 and the other 464 regions.
            m1_connections = S_task[0, 1:]  # Shape: (464,)
            
            # Off-diagonal elements in Riemannian space need to be multiplied by sqrt(2) when vectorized to preserve distance (Isometric mapping)
            m1_connections_scaled = m1_connections * np.sqrt(2.0)
            
            filtered_features.append(m1_connections_scaled)
            
        filtered_features = np.array(filtered_features) # Shape: (n_windows, 464)
        print(f"  > Brute-force dimensionality reduction successful! Feature matrix compressed from 100,000 dimensions to {filtered_features.shape}")
        
        # -- Step 4: Save feature matrix ------------------------------------------------
        out_path = f"{OUTPUT_DIR}/Riemannian_Features_{subj}.npy"
        np.save(out_path, filtered_features)
        print(f"  > Saved to: {out_path}")

if __name__ == '__main__':
    main()
