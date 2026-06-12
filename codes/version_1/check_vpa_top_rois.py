#!/usr/bin/env python3
# ==============================================================================
# check_vpa_top_rois.py
#
# Method C anatomical sanity check: for each group-level VPA R^2 vector
# (Phase 4 output, 481 ROIs), list the top-N ROIs by R^2 and their atlas
# labels (Schaefer 400 / AAL3v2 / Individualized M1), per the index mapping
# used in run_inverse_vpa_mapping.py (weights[i] -> Atlas ROI id i+2).
# ==============================================================================

import numpy as np
import pandas as pd

PROJ_DIR = '/home/ser/2026_brainheck_li_project'
ML_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results'
ATLAS_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases'
TOP_N = 10

COMPONENTS = [
    'Unique_e_R2', 'Unique_xf_R2', 'Unique_xs_R2',
    'Shared_e_xf_R2', 'Shared_xf_xs_R2', 'Shared_e_xs_R2', 'Shared_all_R2',
]


def main():
    labels = pd.read_csv(f'{ATLAS_DIR}/Combined_Atlas_Labels.csv').set_index('Index')['Label']

    for name in COMPONENTS:
        w = np.load(f'{ML_DIR}/{name}.npy')
        order = np.argsort(w)[::-1][:TOP_N]

        print(f"\n=== {name} (top {TOP_N} ROIs) ===")
        for rank, i in enumerate(order, 1):
            if w[i] <= 0:
                break
            roi_id = i + 2  # matches run_inverse_vpa_mapping.py: atlas_data == i+2
            label = labels.get(roi_id, "UNKNOWN")
            print(f"  {rank:2d}. ROI#{roi_id:3d}  R2={w[i]:.4f}  {label}")


if __name__ == '__main__':
    main()
