#!/usr/bin/env python3
# ==============================================================================
# run_phase8A_interactive_html.py
#
# Hybrid Dynamic Neural Decoding Pipeline - Phase 8A:
# Interactive 3D HTML Visualization (Nilearn)
#
# Functions:
# 1. Read Group-level VPA NIfTI maps.
# 2. Use Nilearn to generate interactive 3D Glass Brain HTML files.
# 3. Apply specific color palettes to different variance components.
# ==============================================================================

import os
import glob
from nilearn import plotting
import nibabel as nib

PROJ_DIR = '/home/ser/2026_brainheck_li_project'
INPUT_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/nifti_axes'
OUTPUT_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/html_views'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("\n========================================================")
    print("Start Phase 8A: Interactive HTML 3D Visualization")
    print("========================================================")

    # Find group-level VPA maps. Group maps end in '_R2.nii.gz',
    # while individual maps end in '_R2_sub-XX.nii.gz'.
    vpa_files = glob.glob(f"{INPUT_DIR}/VPA_Map_*_R2.nii.gz")

    
    if not vpa_files:
        print(f"[ERROR] Cannot find Group VPA maps in: {INPUT_DIR}")
        return

    print(f"[INFO] Found {len(vpa_files)} VPA maps. Generating interactive HTMLs...")

    for nii_file in vpa_files:
        basename = os.path.basename(nii_file).replace('.nii.gz', '')
        
        # Rename components to Triarchic Theory
        report_name = basename.replace('_e', '_Meta').replace('_xf', '_Con').replace('_xs', '_Rep')
        
        # Academic color scheme
        cmap = 'hot'
                
        out_html = os.path.join(OUTPUT_DIR, f"{report_name}_Interactive.html")
        
        print(f"  > Processing: {basename} -> {report_name} (Color: {cmap})")
        
        try:
            # We use view_img for a 3D glass brain overlay with interactivity
            # Assuming threshold of 0.05 R2 (5% variance explained)
            html_view = plotting.view_img(
                stat_map_img=nii_file,
                threshold=0.01, 
                cmap=cmap,
                title=report_name,
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
