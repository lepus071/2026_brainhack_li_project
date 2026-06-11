#!/usr/bin/env python3
# ==============================================================================
# run_phase8B_surfice_render.py
#
# Hybrid Dynamic Neural Decoding Pipeline - Phase 8B:
# High-Resolution SurfIce Rendering Macro Generation
#
# Functions:
# 1. Scan Group VPA NIfTI maps.
# 2. Generate a SurfIce script (.gls) to automatically render and save images.
# ==============================================================================

import os
import glob

PROJ_DIR = '/home/ser/2026_brainheck_li_project'
INPUT_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/nifti_axes'
OUTPUT_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/surfice_renders'
os.makedirs(OUTPUT_DIR, exist_ok=True)

GLS_SCRIPT_PATH = os.path.join(OUTPUT_DIR, 'render_vpa_maps.gls')

COLOR_SCHEMES = {
    'Unique_e': 'Yellow',
    'Unique_xf': 'Red',
    'Unique_xs': 'Blue',
    'Shared_e_xf': 'Orange',
    'Shared_e_xs': 'Green',
    'Shared_xf_xs': 'Purple',
    'Shared_Global': 'White'
}

def main():
    print("\n========================================================")
    print("Start Phase 8B: SurfIce Macro Generation")
    print("========================================================")

    # Find group-level VPA maps. Group maps end in '_R2.nii.gz'.
    vpa_files = glob.glob(f"{INPUT_DIR}/VPA_Map_*_R2.nii.gz")
    
    if not vpa_files:
        print(f"[ERROR] Cannot find Group VPA maps in: {INPUT_DIR}")
        return

    print(f"[INFO] Found {len(vpa_files)} VPA maps. Generating SurfIce GLS script...")

    with open(GLS_SCRIPT_PATH, 'w') as f:
        f.write("BEGIN\n")
        f.write("  RESETDEFAULTS;\n")
        f.write("  BACKCOLOR(255, 255, 255);\n") # White background
        f.write("  MESHLOAD('BrainMesh_ICBM152.rh.mz3');\n") # Load standard right hemisphere
        f.write("  OVERLAYTRANSLUCENT(1, true);\n")
        
        for nii_file in vpa_files:
            basename = os.path.basename(nii_file).replace('.nii.gz', '')
            
            # Rename components to Triarchic Theory
            report_name = basename.replace('_e', '_Meta').replace('_xf', '_Con').replace('_xs', '_Rep')
            
            # Match color
            cmap = 'Red' # default
            for key, color in COLOR_SCHEMES.items():
                if key in basename:
                    cmap = color
                    break
                    
            # Use relative paths for maximum compatibility (bypasses Windows/WSL path boundary issues)
            rel_nii_file = f"../nifti_axes/{basename}.nii.gz"
            rel_out_png_right = f"{report_name}_RightView.png"
            rel_out_png_top = f"{report_name}_TopView.png"
            
            f.write(f"  // Rendering: {report_name}\n")
            f.write("  OVERLAYCLOSEALL;\n")
            f.write(f"  OVERLAYLOAD('{rel_nii_file}');\n")
            f.write(f"  OVERLAYCOLORNAME(1, '{cmap}');\n")
            f.write("  OVERLAYMINMAX(1, 0.01, 0.15);\n") # Set R2 threshold 1% to 15%
            
            # Render Right view
            f.write("  AZIMUTHELEVATION(90, 15);\n")
            f.write(f"  SAVEBMP('{rel_out_png_right}');\n")
            
            # Render Top view
            f.write("  AZIMUTHELEVATION(180, 90);\n")
            f.write(f"  SAVEBMP('{rel_out_png_top}');\n\n")

        f.write("END.\n")

    print(f"[SUCCESS] Generated SurfIce macro script: {GLS_SCRIPT_PATH}")
    print("[INFO] Instructions:")
    print("       1. Open the SurfIce software.")
    print("       2. Drag and drop 'render_vpa_maps.gls' into the SurfIce window.")
    print("       3. SurfIce will automatically render all maps and save PNGs to the folder.")

if __name__ == '__main__':
    main()
