#!/usr/bin/env python3
# ==============================================================================
# jackknife_leave2out_unique_e.py
#
# Sensitivity analysis: is the Unique_e thalamus finding / Limbic-network
# warning driven by sample size (a couple of high |corr(e, global_signal)|
# subjects dominating the N=10 group), or is it systematic?
#
# Re-runs the Phase 4 LOSO + VPA pipeline (Full vs No-e models only, since
# Unique_e = R2_full - R2_no_e) on three subject sets:
#   1. All subjects (baseline)
#   2. Drop the 2 most extreme |corr(e, global_signal)| subjects (sub-02, sub-10)
#   3. Drop the 3 most extreme (add sub-01)
#
# For each set, reports the top-5 ROIs for Unique_e and the mean Unique_e R2
# within the Limbic network (Phase 6's null network).
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_hybrid_group_decoding import load_and_align_data, PROJ_DIR

ATLAS_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases'
SUBJECT_LIST = f'{PROJ_DIR}/codes/subjectlist.txt'

# Subjects ranked by |corr(e, global_signal)| from check_e_global_signal_corr.py:
# sub-02 (0.828), sub-10 (0.798), sub-01 (0.715)
EXCLUSION_SETS = {
    "Baseline (N=10, all)": [],
    "Drop top-2 |r(e,GS)| (sub-02, sub-10) -> N=8": ['sub-02', 'sub-10'],
    "Drop top-3 |r(e,GS)| (+ sub-01) -> N=7": ['sub-02', 'sub-10', 'sub-01'],
}


def run_unique_e(subjects):
    X_behav_raw, Y_brain_raw, cv_groups = load_and_align_data(subjects)
    if X_behav_raw is None:
        return None

    Y_pred_full = np.zeros_like(Y_brain_raw)
    Y_pred_no_e = np.zeros_like(Y_brain_raw)

    logo = LeaveOneGroupOut()
    alphas = np.logspace(-3, 3, 10)

    for train_idx, test_idx in logo.split(X_behav_raw, Y_brain_raw, cv_groups):
        X_tr_raw, X_te_raw = X_behav_raw[train_idx], X_behav_raw[test_idx]
        Y_tr_raw, Y_te_raw = Y_brain_raw[train_idx], Y_brain_raw[test_idx]

        scaler_X = StandardScaler().fit(X_tr_raw)
        scaler_Y = StandardScaler().fit(Y_tr_raw)
        X_tr = scaler_X.transform(X_tr_raw)
        X_te = scaler_X.transform(X_te_raw)
        Y_tr = scaler_Y.transform(Y_tr_raw)

        groups_tr = cv_groups[train_idx]
        n_inner = len(np.unique(groups_tr))

        def make_cv():
            return GroupKFold(n_splits=n_inner).split(X_tr, Y_tr, groups=groups_tr)

        clf = RidgeCV(alphas=alphas, cv=make_cv()).fit(X_tr, Y_tr)
        Y_pred_full[test_idx] = scaler_Y.inverse_transform(clf.predict(X_te))

        clf = RidgeCV(alphas=alphas, cv=make_cv()).fit(X_tr[:, 1:], Y_tr)
        Y_pred_no_e[test_idx] = scaler_Y.inverse_transform(clf.predict(X_te[:, 1:]))

    def calc_r2(pred):
        return np.maximum(0, r2_score(Y_brain_raw, pred, multioutput='raw_values'))

    r2_full = calc_r2(Y_pred_full)
    r2_no_e = calc_r2(Y_pred_no_e)
    return np.maximum(0, r2_full - r2_no_e)


def main():
    with open(SUBJECT_LIST, 'r') as f:
        all_subjects = [line.strip() for line in f if line.strip()]

    labels_df = pd.read_csv(f'{ATLAS_DIR}/Combined_Atlas_Labels.csv').set_index('Index')['Label']
    limbic_idx = [i for i, lbl in labels_df.items() if 'Limbic' in str(lbl)]

    for name, exclude in EXCLUSION_SETS.items():
        subjects = [s for s in all_subjects if s not in exclude]
        print(f"\n========================================================")
        print(f"{name}  (n_subjects={len(subjects)})")
        print(f"========================================================")

        unique_e = run_unique_e(subjects)
        if unique_e is None:
            print("  No data, skipping.")
            continue

        order = np.argsort(unique_e)[::-1][:5]
        print("  Top-5 ROIs (Unique_e):")
        for rank, i in enumerate(order, 1):
            roi_id = i + 2
            label = labels_df.get(roi_id, "UNKNOWN")
            print(f"    {rank}. ROI#{roi_id:3d}  R2={unique_e[i]:.4f}  {label}")

        # Limbic network mean R2 (atlas ROI id = i+2, so i = roi_id - 2)
        limbic_vals = [unique_e[roi_id - 2] for roi_id in limbic_idx if 0 <= roi_id - 2 < len(unique_e)]
        print(f"  Limbic network mean Unique_e R2: {np.mean(limbic_vals):.4f} (n_ROI={len(limbic_vals)})")


if __name__ == '__main__':
    main()
