#!/usr/bin/env python3
# ==============================================================================
# run_interactive_html.py  (version_2: Smith et al. 2006 dual-rate, 2-var)
#
# Hybrid Dynamic Neural Decoding Pipeline - Phase 8A:
# Interactive 3D HTML Visualization (Nilearn)
#
# Functions:
# 1. Read group-level VPA NIfTI maps from nifti_axes_v2/.
# 2. Use Nilearn to generate interactive 3D glass-brain HTML files.
# ==============================================================================

import os
import glob
from nilearn import plotting

PROJ_DIR = '/home/ser/2026_brainheck_li_project'
INPUT_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/nifti_axes_v2'
OUTPUT_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/html_views_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CMAP = 'hot'
THRESHOLD = 0.01

def main():
    print("\n========================================================")
    print("Start Phase 8A: Interactive HTML 3D Visualization (v2)")
    print("========================================================")

    # Group-level maps end in '_R2.nii.gz'; individual maps end in
    # '_R2_sub-XX.nii.gz' and are excluded here.
    vpa_files = sorted(glob.glob(f"{INPUT_DIR}/VPA_Map_*_R2.nii.gz"))

    if not vpa_files:
        print(f"[ERROR] Cannot find group VPA maps in: {INPUT_DIR}")
        return

    print(f"[INFO] Found {len(vpa_files)} VPA maps. Generating interactive HTMLs...")

    for nii_file in vpa_files:
        basename = os.path.basename(nii_file).replace('.nii.gz', '')
        out_html = os.path.join(OUTPUT_DIR, f"{basename}_Interactive.html")

        print(f"  > Processing: {basename}")

        try:
            html_view = plotting.view_img(
                stat_map_img=nii_file,
                threshold=THRESHOLD,
                cmap=CMAP,
                title=basename,
                colorbar=True,
                symmetric_cmap=False
            )
            html_view.save_as_html(out_html)
            print(f"  [SUCCESS] Saved: {out_html}")
        except Exception as e:
            print(f"  [ERROR] Failed to generate HTML for {basename}: {e}")

    print("\n[SUCCESS] Phase 8A complete! You can open the HTML files in any web browser.")

if __name__ == '__main__':
    main()
