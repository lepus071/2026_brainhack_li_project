#!/usr/bin/env python3
# ==============================================================================
# run_phase5_inverse_vpa_mapping.py
#
# Hybrid Dynamic Neural Decoding Pipeline - Phase 5:
# Inverse Variance Mapping (Inverse R² Mapping)
#
# Functions:
# 1. Read the Group-level and Individual-level VPA R² score arrays calculated in Phase 4.
# 2. Map the 464-dimensional scores back to 3D NIfTI space using Combined_Atlas.
# 3. Output 3D VPA neural maps including all groups and individuals.
# ==============================================================================

import os
import glob
import numpy as np
import nibabel as nib

PROJ_DIR = '/home/ser/2026_brainheck_li_project'
ATLAS_DIR    = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases'
ML_DIR       = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results'
OUTPUT_DIR   = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/nifti_axes'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_nifti(weights, atlas_data, affine, header, out_path):
    axis_data = np.zeros_like(atlas_data, dtype=np.float32)
    for i, w in enumerate(weights):
        # Note: Atlas ROI id is 2 to 465 (background is 0, 1 is other)
        axis_data[atlas_data == i + 2] = w
    axis_img = nib.Nifti1Image(axis_data, affine=affine, header=header)
    nib.save(axis_img, out_path)

def main():
    print("\n========================================================")
    print("Start Phase 5: Inverse Variance 3D Mapping (Inverse VPA Mapping)")
    print("========================================================")
    
    # Find Atlas
    atlas_files = glob.glob(f"{ATLAS_DIR}/Combined_Atlas_sub-*.nii.gz")
    if not atlas_files:
        print(f"Cannot find spatial template atlas at: {ATLAS_DIR}")
        return
        
    atlas_path = atlas_files[0]
    print(f"Loading neural spatial template: {os.path.basename(atlas_path)}")
    atlas_img = nib.load(atlas_path)
    # The atlas data must be rounded and cast to int to avoid floating point precision mismatches (e.g., 1.997 != 2)
    atlas_data = np.round(atlas_img.get_fdata()).astype(int)
    
    # Find all generated R2 npy files
    npy_files = glob.glob(f"{ML_DIR}/*_R2*.npy")
    if not npy_files:
        print(f"Cannot find Phase 4 VPA R2 arrays at: {ML_DIR}")
        return
        
    print(f"Found {len(npy_files)} R² arrays, starting mapping...")
    
    for npy_file in npy_files:
        w = np.load(npy_file)
        if len(w) != 481:
            print(f"  Warning: {os.path.basename(npy_file)} dimension is not 481, skipping.")
            continue
            
        basename = os.path.basename(npy_file).replace('.npy', '')
        out_path = f"{OUTPUT_DIR}/VPA_Map_{basename}.nii.gz"
        
        generate_nifti(w, atlas_data, atlas_img.affine, atlas_img.header, out_path)
        print(f"  > Wrote NIfTI: VPA_Map_{basename}.nii.gz")
        
    print("\nPhase 5 inverse mapping complete! All VPA maps are ready.")

if __name__ == '__main__':
    main()
