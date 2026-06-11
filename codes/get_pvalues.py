import os
import nibabel as nib
import numpy as np
from scipy import stats

def check_stats():
    # 10 subjects -> df = 9
    df = 9
    OUTPUT_DIR = '/home/ser/2026_brainheck_li_project/data/derivatives/riemannian_decoding/nifti_axes'
    
    parts = ['Unique_e', 'Unique_xf', 'Unique_xs']
    for p in parts:
        filepath = f"{OUTPUT_DIR}/GroupStat_{p}_mean+tlrc.HEAD"
        if not os.path.exists(filepath):
            print(f"Waiting for {filepath}...")
            continue
            
        img = nib.load(filepath)
        data = img.get_fdata()
        # In 3dttest++, sub-brick 0 is the Mean, sub-brick 1 is the T-stat
        if data.ndim == 4:
            t_stat_vol = data[:,:,:,1]
        else:
            t_stat_vol = data
            
        max_t = np.max(t_stat_vol)
        # two-tailed p-value
        p_val = stats.t.sf(np.abs(max_t), df) * 2
        
        print(f"--- {p} ---")
        print(f"Max T-value: {max_t:.4f}")
        print(f"Corresponding P-value: {p_val:.2e}")
        
if __name__ == '__main__':
    check_stats()
