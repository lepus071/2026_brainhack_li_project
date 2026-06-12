#!/usr/bin/env python3
# ==============================================================================
# run_marginal_r2_analysis.py
#
# Follow-up to Phase 4 (run_hybrid_group_decoding.py):
#
# Instead of the VPA unique/shared decomposition (full - reduced subtraction,
# which is sensitive to collinearity among e, x_f, x_s), this script reports
# MARGINAL R^2 for each behavioral predictor alone:
#
#     R2_e  = R2( Y ~ e  )
#     R2_xf = R2( Y ~ x_f )
#     R2_xs = R2( Y ~ x_s )
#
# using the same nested LOSO (outer) + GroupKFold (inner) RidgeCV design as
# Phase 4. Each predictor is also tested in a per-subject LINEAR-DETRENDED
# version, to check whether the marginal R^2 survives removal of a shared
# session-level drift (cf. Appendix A3's e-vs-global_signal drift finding).
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import detrend
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_hybrid_group_decoding import load_and_align_data  # reuse Phase 4 data loading

PROJ_DIR     = '/home/ser/2026_brainheck_li_project'
SUBJECT_LIST = f'{PROJ_DIR}/codes/subjectlist.txt'
ATLAS_LABELS = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases/Combined_Atlas_Labels.csv'
OUTPUT_DIR   = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRED_NAMES = ['e', 'xf', 'xs']


def detrend_per_subject(X, groups):
    """Linear-detrend each behavioral column within each subject's window sequence."""
    X_dt = X.copy()
    for subj_id in np.unique(groups):
        idx = groups == subj_id
        X_dt[idx] = detrend(X[idx], axis=0, type='linear')
    return X_dt


def run_marginal_loso(X_behav, Y_brain, groups, alphas):
    """Nested LOSO (outer) + GroupKFold (inner) RidgeCV, one predictor at a time."""
    logo = LeaveOneGroupOut()
    Y_pred = {n: np.zeros_like(Y_brain) for n in PRED_NAMES}

    for train_idx, test_idx in logo.split(X_behav, Y_brain, groups):
        X_tr_raw, X_te_raw = X_behav[train_idx], X_behav[test_idx]
        Y_tr_raw, Y_te_raw = Y_brain[train_idx], Y_brain[test_idx]

        scaler_X = StandardScaler().fit(X_tr_raw)
        scaler_Y = StandardScaler().fit(Y_tr_raw)
        X_tr = scaler_X.transform(X_tr_raw)
        X_te = scaler_X.transform(X_te_raw)
        Y_tr = scaler_Y.transform(Y_tr_raw)

        groups_tr = groups[train_idx]
        n_inner_splits = len(np.unique(groups_tr))

        for i, name in enumerate(PRED_NAMES):
            cv_inner = GroupKFold(n_splits=n_inner_splits).split(
                X_tr[:, [i]], Y_tr, groups=groups_tr
            )
            clf = RidgeCV(alphas=alphas, cv=cv_inner).fit(X_tr[:, [i]], Y_tr)
            Y_pred[name][test_idx] = scaler_Y.inverse_transform(clf.predict(X_te[:, [i]]))

    return {
        name: np.maximum(0, r2_score(Y_brain, Y_pred[name], multioutput='raw_values'))
        for name in PRED_NAMES
    }


def load_roi_labels(n_rois):
    """Best-effort load of ROI labels, dropping 'Background' entries to align with feature columns."""
    if not os.path.exists(ATLAS_LABELS):
        return [f"ROI_{i}" for i in range(n_rois)]
    df = pd.read_csv(ATLAS_LABELS)
    labels = df.loc[df['Label'] != 'Background', 'Label'].tolist()
    if len(labels) != n_rois:
        print(f"  ! Warning: {len(labels)} non-background labels vs {n_rois} ROIs - "
              f"label alignment may be off, falling back to raw indices.")
        return [f"ROI_{i}" for i in range(n_rois)]
    return labels


def print_top(name, r2, labels, k=10):
    order = np.argsort(r2)[::-1][:k]
    print(f"\n  Top-{k} ROIs for {name}:")
    for rank, idx in enumerate(order, 1):
        print(f"    {rank:2d}. #{idx:3d} {labels[idx]:35s} R2={r2[idx]:.4f}")


def main():
    if not os.path.exists(SUBJECT_LIST):
        print(f"Subject list not found: {SUBJECT_LIST}")
        return
    with open(SUBJECT_LIST, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    print("Loading and aligning data (same as Phase 4)...")
    X_behav_raw, Y_brain_raw, cv_groups = load_and_align_data(subjects)
    if X_behav_raw is None:
        print("Cannot load data.")
        return

    print(f"  > Total samples: {X_behav_raw.shape[0]} windows, "
          f"{len(np.unique(cv_groups))} subjects, {Y_brain_raw.shape[1]} ROIs")

    X_behav_detrend = detrend_per_subject(X_behav_raw, cv_groups)
    alphas = np.logspace(-3, 3, 10)
    labels = load_roi_labels(Y_brain_raw.shape[1])

    for tag, X_src in [('raw', X_behav_raw), ('detrend', X_behav_detrend)]:
        print(f"\n{'=' * 60}\nMarginal R^2 ({tag} behavioral regressors)\n{'=' * 60}")
        results = run_marginal_loso(X_src, Y_brain_raw, cv_groups, alphas)

        for name in PRED_NAMES:
            r2 = results[name]
            np.save(f"{OUTPUT_DIR}/Marginal_{name}_R2_{tag}.npy", r2)
            n = len(r2)
            print(f"  Marginal_{name:3s} ({tag:7s}) "
                  f"mean={r2.mean():.4f}  max={r2.max():.4f}  "
                  f"nonzero={np.sum(r2 > 0)}/{n}")
            print_top(f"{name} ({tag})", r2, labels)

    print("\nDone. Saved Marginal_{e,xf,xs}_R2_{raw,detrend}.npy to ml_results/.")


if __name__ == '__main__':
    main()
