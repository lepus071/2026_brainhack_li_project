#!/usr/bin/env python3
# ==============================================================================
# run_nilearn_render.py
#
# Hybrid Dynamic Neural Decoding Pipeline - Phase 8B (nilearn fallback):
# Static Brain Renderings via nilearn
#
# Functions:
# 1. Scan group-level VPA NIfTI maps in nifti_axes/.
# 2. Render each as a glass-brain and an orthogonal stat-map PNG.
# ==============================================================================

import os
import glob
from nilearn import plotting

PROJ_DIR = '/home/ser/2026_brainheck_li_project'
INPUT_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/nifti_axes'
OUTPUT_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/nilearn_renders'
os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLD = 0.01
VMAX = 0.15

# Rename components to Triarchic Theory naming for figure titles
NAME_MAP = {
    'Unique_e': 'Unique_Meta',
    'Unique_xf': 'Unique_Con',
    'Unique_xs': 'Unique_Rep',
    'Shared_e_xf': 'Shared_Meta_Con',
    'Shared_e_xs': 'Shared_Meta_Rep',
    'Shared_xf_xs': 'Shared_Con_Rep',
    'Shared_all': 'Shared_all',
}

def report_name(basename):
    for key, renamed in NAME_MAP.items():
        if basename == f"VPA_Map_{key}_R2":
            return renamed
    return basename

def main():
    print("\n========================================================")
    print("Start Phase 8B (nilearn): Static Brain Renderings")
    print("========================================================")

    vpa_files = sorted(glob.glob(f"{INPUT_DIR}/VPA_Map_*_R2.nii.gz"))
    if not vpa_files:
        print(f"[ERROR] Cannot find Group VPA maps in: {INPUT_DIR}")
        return

    print(f"[INFO] Found {len(vpa_files)} VPA maps. Rendering...")

    for nii_file in vpa_files:
        basename = os.path.basename(nii_file).replace('.nii.gz', '')
        name = report_name(basename)

        glass_out = os.path.join(OUTPUT_DIR, f"{name}_GlassBrain.png")
        ortho_out = os.path.join(OUTPUT_DIR, f"{name}_OrthoView.png")

        print(f"  > Rendering {name}...")

        plotting.plot_glass_brain(
            nii_file, threshold=THRESHOLD, vmax=VMAX,
            colorbar=True, plot_abs=False,
            title=name, output_file=glass_out,
        )

        plotting.plot_stat_map(
            nii_file, threshold=THRESHOLD, vmax=VMAX,
            display_mode='ortho',
            title=name, output_file=ortho_out,
        )

    print(f"\n[SUCCESS] Renders saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
