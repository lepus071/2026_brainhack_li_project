#!/usr/bin/env python3
# ==============================================================================
# run_vpa_partial_gs.py
#
# Full 4-variable VPA / commonality analysis: (e, x_f, x_s, global_signal).
# global_signal (gs) is forced into every model as a nuisance covariate, so
# all 7 VPA components (Unique_e/xf/xs, Shared_e_xf/xf_xs/e_xs, Shared_all)
# are computed AFTER partialling out gs -- not just Unique_e (cf.
# check_unique_e_partial_r2.py, which only handled Unique_e).
#
# For S subset of {e, xf, xs}, define:
#   f(S) = R2(S U {gs}) - R2({gs})
# Then apply the same commonality formulas as run_hybrid_group_decoding.py's
# Phase 4 VPA, but using f(.) instead of raw R2(.):
#
#   Unique_e      = f(e,xf,xs) - f(xf,xs)   = R2(e,xf,xs,gs) - R2(xf,xs,gs)
#   Unique_xf     = f(e,xf,xs) - f(e,xs)
#   Unique_xs     = f(e,xf,xs) - f(e,xf)
#   Shared_e_xf   = f(e) + f(xf) - f(xs)... wait: f(e)+f(xf)-f(no_xs)
#   Shared_xf_xs  = f(xf) + f(xs) - f(no_e)
#   Shared_e_xs   = f(e) + f(xs) - f(no_xf)
#   Shared_all    = f(e,xf,xs) - (sum of the above 6)
#
# Both inner (GroupKFold) and outer (LeaveOneGroupOut) CV are subject-grouped.
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
from run_hybrid_group_decoding import PROJ_DIR
from check_unique_e_partial_r2 import load_and_align_data_with_gs, ATLAS_DIR

SUBJECT_LIST = f'{PROJ_DIR}/codes/subjectlist.txt'
OUTPUT_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results'

# X_behav columns: 0=e, 1=xf, 2=xs, 3=gs
MODELS = {
    'full':     [0, 1, 2, 3],  # e, xf, xs, gs
    'no_e':     [1, 2, 3],     # xf, xs, gs
    'no_xf':    [0, 2, 3],     # e, xs, gs
    'no_xs':    [0, 1, 3],     # e, xf, gs
    'only_e':   [0, 3],        # e, gs
    'only_xf':  [1, 3],        # xf, gs
    'only_xs':  [2, 3],        # xs, gs
    'only_gs':  [3],           # gs
}


def run_loso_r2(X_behav_raw, Y_brain_raw, cv_groups, cols):
    """LOSO + inner GroupKFold RidgeCV on X_behav_raw[:, cols] -> Y_brain_raw."""
    Y_pred = np.zeros_like(Y_brain_raw)
    logo = LeaveOneGroupOut()
    alphas = np.logspace(-3, 3, 10)

    for train_idx, test_idx in logo.split(X_behav_raw, Y_brain_raw, cv_groups):
        X_tr_raw, X_te_raw = X_behav_raw[train_idx][:, cols], X_behav_raw[test_idx][:, cols]
        Y_tr_raw, Y_te_raw = Y_brain_raw[train_idx], Y_brain_raw[test_idx]

        scaler_X = StandardScaler().fit(X_tr_raw)
        scaler_Y = StandardScaler().fit(Y_tr_raw)
        X_tr = scaler_X.transform(X_tr_raw)
        X_te = scaler_X.transform(X_te_raw)
        Y_tr = scaler_Y.transform(Y_tr_raw)

        groups_tr = cv_groups[train_idx]
        n_inner = len(np.unique(groups_tr))
        cv_inner = GroupKFold(n_splits=n_inner).split(X_tr, Y_tr, groups=groups_tr)

        clf = RidgeCV(alphas=alphas, cv=cv_inner).fit(X_tr, Y_tr)
        Y_pred[test_idx] = scaler_Y.inverse_transform(clf.predict(X_te))

    return np.maximum(0, r2_score(Y_brain_raw, Y_pred, multioutput='raw_values'))


def main():
    with open(SUBJECT_LIST, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    print("Loading data with global_signal as 4th predictor (e, x_f, x_s, gs)...")
    X_behav_raw, Y_brain_raw, cv_groups = load_and_align_data_with_gs(subjects)
    print(f"  > Total samples: {X_behav_raw.shape[0]} windows, subjects: {len(np.unique(cv_groups))}")

    r2 = {}
    for name, cols in MODELS.items():
        print(f"Fitting model '{name}' (cols={cols})...")
        r2[name] = run_loso_r2(X_behav_raw, Y_brain_raw, cv_groups, cols)

    r2_gs = r2['only_gs']

    # f(S) = R2(S U {gs}) - R2({gs})
    f_full  = r2['full']    - r2_gs
    f_no_e  = r2['no_e']    - r2_gs
    f_no_xf = r2['no_xf']   - r2_gs
    f_no_xs = r2['no_xs']   - r2_gs
    f_e     = r2['only_e']  - r2_gs
    f_xf    = r2['only_xf'] - r2_gs
    f_xs    = r2['only_xs'] - r2_gs

    unique_e  = np.maximum(0, f_full - f_no_e)
    unique_xf = np.maximum(0, f_full - f_no_xf)
    unique_xs = np.maximum(0, f_full - f_no_xs)

    shared_e_xf  = np.maximum(0, f_e + f_xf - f_no_xs)
    shared_xf_xs = np.maximum(0, f_xf + f_xs - f_no_e)
    shared_e_xs  = np.maximum(0, f_e + f_xs - f_no_xf)

    shared_all = np.maximum(
        0, f_full - (unique_e + unique_xf + unique_xs + shared_e_xf + shared_xf_xs + shared_e_xs)
    )

    components = {
        'Unique_e_partial4_R2': unique_e,
        'Unique_xf_partial4_R2': unique_xf,
        'Unique_xs_partial4_R2': unique_xs,
        'Shared_e_xf_partial4_R2': shared_e_xf,
        'Shared_xf_xs_partial4_R2': shared_xf_xs,
        'Shared_e_xs_partial4_R2': shared_e_xs,
        'Shared_all_partial4_R2': shared_all,
    }

    print("\nSaving 4-variable (gs-controlled) VPA components...")
    for name, arr in components.items():
        np.save(f"{OUTPUT_DIR}/{name}.npy", arr)
        print(f"  {name}: mean={arr.mean():.4f}  max={arr.max():.4f}  nonzero={np.sum(arr>0)}/{len(arr)}")

    # Comparison with original 3-variable VPA
    labels_df = pd.read_csv(f'{ATLAS_DIR}/Combined_Atlas_Labels.csv').set_index('Index')['Label']
    orig_map = {
        'Unique_e_partial4_R2': 'Unique_e_R2',
        'Unique_xf_partial4_R2': 'Unique_xf_R2',
        'Unique_xs_partial4_R2': 'Unique_xs_R2',
        'Shared_e_xf_partial4_R2': 'Shared_e_xf_R2',
        'Shared_xf_xs_partial4_R2': 'Shared_xf_xs_R2',
        'Shared_e_xs_partial4_R2': 'Shared_e_xs_R2',
        'Shared_all_partial4_R2': 'Shared_all_R2',
    }

    print("\n========================================================")
    print("Comparison: original (3-var) vs partial4 (gs-controlled)")
    print("========================================================")
    for name, arr in components.items():
        orig = np.load(f"{OUTPUT_DIR}/{orig_map[name]}.npy")
        print(f"\n--- {name}  vs  {orig_map[name]} ---")
        print(f"  mean R2: orig={orig.mean():.4f}  partial4={arr.mean():.4f}")
        print(f"  max  R2: orig={orig.max():.4f}  partial4={arr.max():.4f}")
        print(f"  nonzero ROIs: orig={np.sum(orig>0)}/{len(orig)}  partial4={np.sum(arr>0)}/{len(arr)}")

        order = np.argsort(arr)[::-1][:5]
        print("  Top-5 ROIs (partial4):")
        for rank, i in enumerate(order, 1):
            if arr[i] <= 0:
                break
            roi_id = i + 2
            label = labels_df.get(roi_id, "UNKNOWN")
            print(f"    {rank}. ROI#{roi_id:3d}  partial4={arr[i]:.4f}  (orig={orig[i]:.4f})  {label}")

    # Limbic network mean, for all 7 components
    limbic_idx = [i for i, lbl in labels_df.items() if 'Limbic' in str(lbl)]
    print("\n========================================================")
    print("Limbic network mean R2 (Phase 6 null network)")
    print("========================================================")
    for name, arr in components.items():
        orig = np.load(f"{OUTPUT_DIR}/{orig_map[name]}.npy")
        limbic_partial = [arr[r-2] for r in limbic_idx if 0 <= r-2 < len(arr)]
        limbic_orig = [orig[r-2] for r in limbic_idx if 0 <= r-2 < len(orig)]
        print(f"  {name}: orig={np.mean(limbic_orig):.4f}  partial4={np.mean(limbic_partial):.4f}")


if __name__ == '__main__':
    main()
