#!/usr/bin/env python3
# ==============================================================================
# run_inverse_vpa_mapping.py  (version_2: Smith et al. 2006 dual-rate, 2-var)
#
# Hybrid Dynamic Neural Decoding Pipeline - Phase 5:
# Inverse Variance Mapping (Inverse R² Mapping)
#
# Functions:
# 1. Read the Group-level and Individual-level 2-var VPA R² score arrays
#    calculated in Phase 4 (ml_results_v2/).
# 2. Map the 481-ROI scores back to 3D NIfTI space using Combined_Atlas.
# 3. Output 3D VPA neural maps including all groups and individuals.
# ==============================================================================

import os
import glob
import numpy as np
import nibabel as nib

PROJ_DIR = '/home/ser/2026_brainhack_li_project'
ATLAS_DIR  = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases'
ML_DIR     = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results_v2'
OUTPUT_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/nifti_axes_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_nifti(weights, atlas_data, affine, header, out_path):
    axis_data = np.zeros_like(atlas_data, dtype=np.float32)
    for i, w in enumerate(weights):
        # Note: Atlas ROI id is 2 to 482 (background is 0, 1 is M1 seed)
        axis_data[atlas_data == i + 2] = w
    axis_img = nib.Nifti1Image(axis_data, affine=affine, header=header)
    nib.save(axis_img, out_path)


def main():
    print("\n========================================================")
    print("Start Phase 5 (v2): Inverse Variance 3D Mapping (2-var dual-rate)")
    print("========================================================")

    atlas_files = glob.glob(f"{ATLAS_DIR}/Combined_Atlas_sub-*.nii.gz")
    if not atlas_files:
        print(f"Cannot find spatial template atlas at: {ATLAS_DIR}")
        return

    atlas_path = atlas_files[0]
    print(f"Loading neural spatial template: {os.path.basename(atlas_path)}")
    atlas_img = nib.load(atlas_path)
    atlas_data = np.round(atlas_img.get_fdata()).astype(int)

    npy_files = glob.glob(f"{ML_DIR}/*_R2*.npy")
    if not npy_files:
        print(f"Cannot find Phase 4 (v2) VPA R2 arrays at: {ML_DIR}")
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

    print("\nPhase 5 (v2) inverse mapping complete! All 2-var VPA maps are ready.")


if __name__ == '__main__':
    main()
