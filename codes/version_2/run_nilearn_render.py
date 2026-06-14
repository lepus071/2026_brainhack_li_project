#!/usr/bin/env python3
# ==============================================================================
# run_nilearn_render.py  (version_2: Smith et al. 2006 dual-rate, 2-var)
#
# Hybrid Dynamic Neural Decoding Pipeline - Phase 8B:
# Static Brain Renderings via nilearn
#
# Functions:
# 1. Scan group-level VPA NIfTI maps in nifti_axes_v2/.
# 2. Render each as a glass-brain and an orthogonal stat-map PNG.
# ==============================================================================

import os
import glob
from nilearn import plotting

PROJ_DIR = '/home/ser/2026_brainheck_li_project'
INPUT_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/nifti_axes_v2'
OUTPUT_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/nilearn_renders_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLD = 0.01
VMAX = 0.5

def main():
    print("\n========================================================")
    print("Start Phase 8B: Static Brain Renderings (v2)")
    print("========================================================")

    vpa_files = sorted(glob.glob(f"{INPUT_DIR}/VPA_Map_*_R2.nii.gz"))
    if not vpa_files:
        print(f"[ERROR] Cannot find group VPA maps in: {INPUT_DIR}")
        return

    print(f"[INFO] Found {len(vpa_files)} VPA maps. Rendering...")

    for nii_file in vpa_files:
        basename = os.path.basename(nii_file).replace('.nii.gz', '')

        glass_out = os.path.join(OUTPUT_DIR, f"{basename}_GlassBrain.png")
        ortho_out = os.path.join(OUTPUT_DIR, f"{basename}_OrthoView.png")

        print(f"  > Rendering {basename}...")

        plotting.plot_glass_brain(
            nii_file, threshold=THRESHOLD, vmax=VMAX,
            colorbar=True, plot_abs=False,
            title=basename, output_file=glass_out,
        )

        plotting.plot_stat_map(
            nii_file, threshold=THRESHOLD, vmax=VMAX,
            display_mode='ortho',
            title=basename, output_file=ortho_out,
        )

    print(f"\n[SUCCESS] Renders saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
