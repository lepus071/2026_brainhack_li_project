import nibabel as nib
import numpy as np

img = nib.load('/home/ser/2026_brainheck_li_project/data/derivatives/riemannian_decoding/nifti_axes/VPA_Map_Unique_e_R2.nii.gz')
d = img.get_fdata()
print("Max value in VPA_Map_Unique_e_R2:", np.max(d))
print("Min value in VPA_Map_Unique_e_R2:", np.min(d))
print("Any non-zero?", np.any(d != 0))

atlas_img = nib.load('/home/ser/2026_brainheck_li_project/data/derivatives/riemannian_decoding/atlases/Combined_Atlas_sub-04.nii.gz')
ad = atlas_img.get_fdata()
print("Max value in Atlas:", np.max(ad))
print("Unique values in Atlas (first 10):", np.unique(ad)[:10])
print("Unique values in Atlas (last 10):", np.unique(ad)[-10:])
