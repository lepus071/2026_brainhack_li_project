#!/usr/bin/env python3
# ==============================================================================
# run_time_index_control.py
#
# Control analysis for run_marginal_r2_analysis.py:
#
# Instead of asking "is there signal left after removing the trend in
# e/x_f/x_s" (which may unfairly penalize x_f/x_s, whose definitions ARE
# smooth learning trajectories), this script asks:
#
#     Does a GENERIC monotonic "time index" (each window's relative position
#     within the session, 0 -> 1) explain brain connectivity about as well
#     as e / x_f / x_s, with overlapping top ROIs?
#
# If e/x_f/x_s's marginal R^2 and top-ROI sets are essentially indistinguishable
# from the time_index control, that's evidence the dual-rate model's specific
# *shape* (x_f fast-decay vs x_s slow-decay vs e's envelope+fluctuations) adds
# nothing beyond "where in the session you are". If they differ (higher R^2
# and/or different top ROIs), that supports content-specific encoding.
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
from run_hybrid_group_decoding import load_and_align_data

PROJ_DIR     = '/home/ser/2026_brainhack_li_project'
SUBJECT_LIST = f'{PROJ_DIR}/codes/subjectlist.txt'
ATLAS_LABELS = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases/Combined_Atlas_Labels.csv'
OUTPUT_DIR   = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_time_index(groups):
    """Per-subject relative position (0 -> 1) within that subject's kept windows,
    assuming sample order within each subject's block is chronological
    (true for load_and_align_data's output)."""
    t = np.zeros(len(groups), dtype=float)
    for subj_id in np.unique(groups):
        idx = np.where(groups == subj_id)[0]
        n = len(idx)
        t[idx] = np.linspace(0.0, 1.0, n) if n > 1 else 0.0
    return t


def run_marginal_loso_single(X_col, Y_brain, groups, alphas):
    """Nested LOSO (outer) + GroupKFold (inner) RidgeCV for a single predictor column."""
    logo = LeaveOneGroupOut()
    Y_pred = np.zeros_like(Y_brain)
    X_col = X_col.reshape(-1, 1)

    for train_idx, test_idx in logo.split(X_col, Y_brain, groups):
        X_tr_raw, X_te_raw = X_col[train_idx], X_col[test_idx]
        Y_tr_raw, Y_te_raw = Y_brain[train_idx], Y_brain[test_idx]

        scaler_X = StandardScaler().fit(X_tr_raw)
        scaler_Y = StandardScaler().fit(Y_tr_raw)
        X_tr = scaler_X.transform(X_tr_raw)
        X_te = scaler_X.transform(X_te_raw)
        Y_tr = scaler_Y.transform(Y_tr_raw)

        groups_tr = groups[train_idx]
        n_inner_splits = len(np.unique(groups_tr))
        cv_inner = GroupKFold(n_splits=n_inner_splits).split(X_tr, Y_tr, groups=groups_tr)
        clf = RidgeCV(alphas=alphas, cv=cv_inner).fit(X_tr, Y_tr)
        Y_pred[test_idx] = scaler_Y.inverse_transform(clf.predict(X_te))

    return np.maximum(0, r2_score(Y_brain, Y_pred, multioutput='raw_values'))


def load_roi_labels(n_rois):
    if not os.path.exists(ATLAS_LABELS):
        return [f"ROI_{i}" for i in range(n_rois)]
    df = pd.read_csv(ATLAS_LABELS)
    labels = df.loc[df['Label'] != 'Background', 'Label'].tolist()
    if len(labels) != n_rois:
        return [f"ROI_{i}" for i in range(n_rois)]
    return labels


def print_top(name, r2, labels, k=10):
    order = np.argsort(r2)[::-1][:k]
    print(f"\n  Top-{k} ROIs for {name}:")
    for rank, idx in enumerate(order, 1):
        print(f"    {rank:2d}. #{idx:3d} {labels[idx]:35s} R2={r2[idx]:.4f}")
    return set(order)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main():
    with open(SUBJECT_LIST, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    print("Loading and aligning data (same as Phase 4)...")
    X_behav_raw, Y_brain_raw, cv_groups = load_and_align_data(subjects)
    print(f"  > Total samples: {X_behav_raw.shape[0]} windows, "
          f"{len(np.unique(cv_groups))} subjects, {Y_brain_raw.shape[1]} ROIs")

    alphas = np.logspace(-3, 3, 10)
    labels = load_roi_labels(Y_brain_raw.shape[1])

    # --- Time-index control regressor ---
    time_idx = compute_time_index(cv_groups)
    print(f"\n{'=' * 60}\nMarginal R^2 (time_index control)\n{'=' * 60}")
    r2_time = run_marginal_loso_single(time_idx, Y_brain_raw, cv_groups, alphas)
    np.save(f"{OUTPUT_DIR}/Marginal_time_index_R2.npy", r2_time)
    print(f"  Marginal_time_index  mean={r2_time.mean():.4f}  max={r2_time.max():.4f}  "
          f"nonzero={np.sum(r2_time > 0)}/{len(r2_time)}")
    top_time = print_top("time_index", r2_time, labels)

    # --- Compare with e / x_f / x_s raw marginal R^2 ---
    print(f"\n{'=' * 60}\nComparison: e / x_f / x_s (raw) vs time_index\n{'=' * 60}")
    pred_names = ['e', 'xf', 'xs']
    for i, name in enumerate(pred_names):
        path = f"{OUTPUT_DIR}/Marginal_{name}_R2_raw.npy"
        if not os.path.exists(path):
            print(f"  ! {path} not found - run run_marginal_r2_analysis.py first.")
            continue
        r2_pred = np.load(path)
        top_pred = set(np.argsort(r2_pred)[::-1][:10])
        jac = jaccard(top_time, top_pred)
        print(f"\n  {name:3s}: mean R2={r2_pred.mean():.4f} (time_index={r2_time.mean():.4f})  "
              f"max R2={r2_pred.max():.4f} (time_index={r2_time.max():.4f})")
        print(f"       Top-10 ROI overlap with time_index (Jaccard) = {jac:.2f} "
              f"({len(top_time & top_pred)}/10 shared)")

    print("\nDone. Saved Marginal_time_index_R2.npy to ml_results/.")


if __name__ == '__main__':
    main()
