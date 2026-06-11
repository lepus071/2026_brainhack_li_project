import os
import shutil

# WSL Paths
INPUT_DIR = '/home/ser/2026_brainheck_li_project/data/derivatives/riemannian_decoding/nifti_axes'
# Mapped Windows path
WIN_DIR_WSL = '/mnt/c/Users/User/.gemini/antigravity/brain/31b5acc4-4250-4580-8ea4-07cc5e06f4ec/surfice_export'
# Pure Windows Path for script
WIN_DIR_NATIVE = 'C:\\Users\\User\\.gemini\\antigravity\\brain\\31b5acc4-4250-4580-8ea4-07cc5e06f4ec\\surfice_export'

os.makedirs(WIN_DIR_WSL, exist_ok=True)

# The 7 maps
vpa_maps = [
    'VPA_Map_Shared_e_xs_R2.nii.gz',
    'VPA_Map_Unique_xs_R2.nii.gz',
    'VPA_Map_Unique_e_R2.nii.gz',
    'VPA_Map_Shared_xf_xs_R2.nii.gz',
    'VPA_Map_Unique_xf_R2.nii.gz',
    'VPA_Map_Shared_all_R2.nii.gz',
    'VPA_Map_Shared_e_xf_R2.nii.gz'
]

COLOR_SCHEMES = {
    'Unique_e': 'Yellow',
    'Unique_xf': 'Red',
    'Unique_xs': 'Blue',
    'Shared_e_xf': 'Orange',
    'Shared_e_xs': 'Green',
    'Shared_xf_xs': 'Purple',
    'Shared_Global': 'White'
}

# 1. Copy files
print("Copying files to C: drive...")
for f in vpa_maps:
    src = os.path.join(INPUT_DIR, f)
    dst = os.path.join(WIN_DIR_WSL, f)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {f}")

# 2. Write Python script for SurfIce
py_path = os.path.join(WIN_DIR_WSL, 'render_vpa_maps_windows.py')
with open(py_path, 'w') as f:
    f.write("import gl\n\n")
    f.write("gl.resetdefaults()\n")
    f.write("gl.meshload('BrainMesh_ICBM152.rh.mz3')\n\n")
    
    for basename_ext in vpa_maps:
        basename = basename_ext.replace('.nii.gz', '')
        report_name = basename.replace('_e', '_Meta').replace('_xf', '_Con').replace('_xs', '_Rep')
        
        # Simply use classic academic 'Red' colormap for all renders
        cmap = 'Red'
        
        # Forward slashes for Python strings
        abs_nii = f"{WIN_DIR_NATIVE}\\{basename_ext}".replace('\\', '/')
        abs_out_r = f"{WIN_DIR_NATIVE}\\{report_name}_RightView.png".replace('\\', '/')
        abs_out_t = f"{WIN_DIR_NATIVE}\\{report_name}_TopView.png".replace('\\', '/')
        
        f.write(f"# Rendering: {report_name}\n")
        f.write("gl.overlaycloseall()\n")
        f.write(f"gl.overlayload('{abs_nii}')\n")
        f.write(f"gl.overlaycolorname(1, '{cmap}')\n")
        f.write("gl.overlayminmax(1, 0.01, 0.15)\n")
        
        f.write("gl.azimuthelevation(90, 15)\n")
        f.write(f"gl.savebmp('{abs_out_r}')\n")
        
        f.write("gl.azimuthelevation(180, 90)\n")
        f.write(f"gl.savebmp('{abs_out_t}')\n\n")

print(f"SurfIce Python Script written to {py_path}")
