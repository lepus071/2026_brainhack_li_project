#!/usr/bin/env python3
# ==============================================================================
# check_vpa_top_rois_corrected.py
#
# Method C anatomical sanity check (corrected version): for each of the 7
# group-level VPA R^2 components, list the top-10 ROIs with atlas labels.
# Unique_e uses Unique_e_partial_R2 (controlling for window-mean
# global_signal, see check_unique_e_partial_r2.py); the other 6 components
# use their original VPA outputs (unaffected by the global-signal check).
# ==============================================================================

import numpy as np
import pandas as pd

PROJ_DIR = '/home/ser/2026_brainheck_li_project'
ML_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results'
ATLAS_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases'
TOP_N = 10

COMPONENTS = [
    ('Unique_e_R2 (corrected -> Unique_e_partial_R2)', 'Unique_e_partial_R2'),
    ('Unique_xf_R2', 'Unique_xf_R2'),
    ('Unique_xs_R2', 'Unique_xs_R2'),
    ('Shared_e_xf_R2', 'Shared_e_xf_R2'),
    ('Shared_xf_xs_R2', 'Shared_xf_xs_R2'),
    ('Shared_e_xs_R2', 'Shared_e_xs_R2'),
    ('Shared_all_R2', 'Shared_all_R2'),
]


def main():
    labels = pd.read_csv(f'{ATLAS_DIR}/Combined_Atlas_Labels.csv').set_index('Index')['Label']
    orig_e = np.load(f'{ML_DIR}/Unique_e_R2.npy')

    for display_name, fname in COMPONENTS:
        w = np.load(f'{ML_DIR}/{fname}.npy')
        order = np.argsort(w)[::-1][:TOP_N]

        print(f"\n=== {display_name} (top {TOP_N} ROIs) ===")
        for rank, i in enumerate(order, 1):
            if w[i] <= 0:
                break
            roi_id = i + 2
            label = labels.get(roi_id, "UNKNOWN")
            if fname == 'Unique_e_partial_R2':
                print(f"  {rank:2d}. ROI#{roi_id:3d}  R2={w[i]:.4f}  (orig_unique_e={orig_e[i]:.4f})  {label}")
            else:
                print(f"  {rank:2d}. ROI#{roi_id:3d}  R2={w[i]:.4f}  {label}")


if __name__ == '__main__':
    main()
