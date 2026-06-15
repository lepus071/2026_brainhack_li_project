#!/usr/bin/env python3
# ==============================================================================
# check_unique_e_partial_top_rois.py
#
# Method C anatomical sanity check, but for Unique_e_partial_R2 (Unique_e
# after controlling for window-mean global_signal, see
# check_unique_e_partial_r2.py). Lists top-10 ROIs with atlas labels, and
# also shows the corresponding original Unique_e R2 for comparison.
# ==============================================================================

import numpy as np
import pandas as pd

PROJ_DIR = '/home/ser/2026_brainhack_li_project'
ML_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results'
ATLAS_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases'
TOP_N = 10


def main():
    labels = pd.read_csv(f'{ATLAS_DIR}/Combined_Atlas_Labels.csv').set_index('Index')['Label']

    partial = np.load(f'{ML_DIR}/Unique_e_partial_R2.npy')
    orig = np.load(f'{ML_DIR}/Unique_e_R2.npy')

    order = np.argsort(partial)[::-1][:TOP_N]

    print(f"\n=== Unique_e_partial_R2 (top {TOP_N} ROIs, controlling for global_signal) ===")
    for rank, i in enumerate(order, 1):
        if partial[i] <= 0:
            break
        roi_id = i + 2
        label = labels.get(roi_id, "UNKNOWN")
        print(f"  {rank:2d}. ROI#{roi_id:3d}  partial_R2={partial[i]:.4f}  (orig={orig[i]:.4f})  {label}")


if __name__ == '__main__':
    main()
