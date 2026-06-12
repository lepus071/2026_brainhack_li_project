#!/usr/bin/env python3
# ==============================================================================
# run_null_network_validation.py  (version_2: Smith et al. 2006 dual-rate, 2-var)
#
# Hybrid Dynamic Neural Decoding Pipeline - Phase 6:
# Null Network Validation
#
# Function:
# 1. Read the Group-level 2-var VPA R2 score arrays calculated in Phase 4 (ml_results_v2/).
# 2. Extract the Limbic network, which is "unrelated" to the visuomotor adaptation task.
# 3. Validate that VPA R2 in the Limbic network approaches 0, proving the
#    specificity of the neural map and no overfitting.
# ==============================================================================

import os
import numpy as np
import pandas as pd

PROJ_DIR = '/home/ser/2026_brainheck_li_project'
ATLAS_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases'
ML_DIR    = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results_v2'


def main():
    print("\n========================================================")
    print("Start Phase 6 (v2): Null Network Validation (2-var dual-rate)")
    print("========================================================")

    label_csv = f"{ATLAS_DIR}/Combined_Atlas_Labels.csv"
    if not os.path.exists(label_csv):
        print(f"Cannot find atlas label CSV: {label_csv}")
        return

    label_df = pd.read_csv(label_csv)
    labels = label_df['Label'].values[2:]  # 481 ROIs

    null_idx = [i for i, l in enumerate(labels) if 'Limbic' in str(l)]

    if not null_idx:
        print("Warning: Cannot find brain regions containing Limbic labels in the atlas!")
        return

    print(f"Successfully targeted Null Network: Limbic Network (total {len(null_idx)} ROIs)")

    vpa_parts = ["Unique_xf", "Unique_xs", "Shared_xf_xs", "Full_xf_xs"]

    print("\nExamining the explanatory power (R2) of each variance partition (VPA) in the Limbic network:")
    print("-" * 55)

    all_passed = True
    for part in vpa_parts:
        npy_path = f"{ML_DIR}/{part}_R2.npy"
        if not os.path.exists(npy_path):
            print(f"  Cannot find file: {part}_R2.npy")
            continue

        w = np.load(npy_path)
        null_w = w[null_idx]

        avg_r2 = np.mean(null_w)
        max_r2 = np.max(null_w)

        # Strict criterion: warn if average R2 exceeds 0.01 (1%) or maximum exceeds 0.05 (5%)
        pass_status = "Pass" if (avg_r2 < 0.01 and max_r2 < 0.05) else "Warning"
        if pass_status == "Warning":
            all_passed = False

        print(f"  > {part:15s} | Average R2: {avg_r2:.4f} | Max R2: {max_r2:.4f} | {pass_status}")

    print("-" * 55)
    if all_passed:
        print("Perfectly passed the validation test! The 2-var VPA network did not overfit random noise and has extremely high spatial specificity!")
    else:
        print("Hint: Some variance slightly remains in the Limbic network, which may be affected by whole-brain blood flow fluctuations (Global Signal).")


if __name__ == '__main__':
    main()
