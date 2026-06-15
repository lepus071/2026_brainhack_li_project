#!/usr/bin/env python3
# ==============================================================================
# run_between_within_decomposition.py
#
# Follow-up to run_marginal_r2_analysis.py / run_time_index_control.py.
#
# The raw marginal R^2 of e/x_f/x_s collapses to 0 after linear detrending,
# while a generic time_index control also has R^2=0 everywhere. This suggests
# the raw marginal R^2 is driven by BETWEEN-SUBJECT (trait-level) differences
# in overall e/x_f/x_s magnitude, which the Riemannian per-subject recentering
# (Phase 3) and per-subject behavioral fitting (Phase 0B) are specifically
# designed to make comparable across subjects.
#
# This script splits each variable (X = e/x_f/x_s, Y = brain ROI features)
# into two components per subject s, window w:
#
#     X[s,w] = Xbar[s] + Xdev[s,w]      (between-subject mean + within-subject deviation)
#     Y[s,w] = Ybar[s] + Ydev[s,w]
#
# and runs two separate analyses:
#
#  (A) BETWEEN-SUBJECT (N=10, df=8):
#      For each ROI, Pearson r(Xbar, Ybar) across the 10 subjects, plus a
#      leave-one-subject-out jackknife to check whether the correlation is
#      driven by a single subject.
#
#  (B) WITHIN-SUBJECT (trial-level):
#      Marginal R^2 (nested LOSO + GroupKFold RidgeCV, same design as Phase 4)
#      using Xdev to predict Ydev -- i.e. "after removing each subject's own
#      mean level, is there still trial-by-trial predictive power?"
#      This is a cleaner within-subject control than linear detrending, since
#      subject-mean centering does not distort the shape of x_f/x_s's learning
#      trajectory.
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
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

PRED_NAMES = ['e', 'xf', 'xs']


def subject_means(X, groups):
    """Per-subject mean of each column, expanded back to per-window shape,
    plus the raw per-subject mean matrix (n_subjects x n_features)."""
    subj_ids = np.unique(groups)
    means = np.zeros((len(subj_ids), X.shape[1]))
    X_bar = np.zeros_like(X)
    for i, subj_id in enumerate(subj_ids):
        idx = groups == subj_id
        m = X[idx].mean(axis=0)
        means[i] = m
        X_bar[idx] = m
    return X_bar, means, subj_ids


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
    return order


def between_subject_analysis(X_means, Y_means, pred_names, labels, k=10):
    """For each predictor, Pearson r(Xbar, Ybar) across subjects (N, df=N-2),
    plus leave-one-subject-out jackknife on the top ROI."""
    n_subj = X_means.shape[0]
    results = {}
    for i, name in enumerate(pred_names):
        x = X_means[:, i]
        r = np.array([pearsonr(x, Y_means[:, j])[0] for j in range(Y_means.shape[1])])
        r = np.nan_to_num(r)
        results[name] = r

        order = np.argsort(np.abs(r))[::-1][:k]
        print(f"\n  Top-{k} ROIs for {name} (between-subject |r|, N={n_subj}, df={n_subj - 2}):")
        for rank, idx in enumerate(order, 1):
            print(f"    {rank:2d}. #{idx:3d} {labels[idx]:35s} r={r[idx]:+.3f}")

        # Jackknife the single top ROI
        top_idx = order[0]
        print(f"\n  Jackknife (leave-one-subject-out) for top ROI #{top_idx} ({labels[top_idx]}, "
              f"full r={r[top_idx]:+.3f}):")
        jk_r = []
        for s in range(n_subj):
            mask = np.ones(n_subj, dtype=bool)
            mask[s] = False
            r_jk, _ = pearsonr(x[mask], Y_means[mask, top_idx])
            jk_r.append(r_jk)
        jk_r = np.array(jk_r)
        print(f"    leave-out r range: [{jk_r.min():+.3f}, {jk_r.max():+.3f}]  "
              f"(sign stable: {np.all(np.sign(jk_r) == np.sign(r[top_idx]))})")

    return results


def main():
    with open(SUBJECT_LIST, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    print("Loading and aligning data (same as Phase 4)...")
    X_behav_raw, Y_brain_raw, cv_groups = load_and_align_data(subjects)
    print(f"  > Total samples: {X_behav_raw.shape[0]} windows, "
          f"{len(np.unique(cv_groups))} subjects, {Y_brain_raw.shape[1]} ROIs")

    alphas = np.logspace(-3, 3, 10)
    labels = load_roi_labels(Y_brain_raw.shape[1])

    X_bar, X_means, subj_ids = subject_means(X_behav_raw, cv_groups)
    Y_bar, Y_means, _ = subject_means(Y_brain_raw, cv_groups)

    X_dev = X_behav_raw - X_bar
    Y_dev = Y_brain_raw - Y_bar

    # ---------------------------------------------------------------
    # (A) Between-subject: Pearson r(Xbar, Ybar), N=10, df=8 + jackknife
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}\n(A) BETWEEN-SUBJECT correlations (subject means, N={len(subj_ids)})\n{'=' * 60}")
    between_results = between_subject_analysis(X_means, Y_means, PRED_NAMES, labels)
    for name in PRED_NAMES:
        np.save(f"{OUTPUT_DIR}/Between_{name}_r.npy", between_results[name])

    # ---------------------------------------------------------------
    # (B) Within-subject: marginal R^2 of subject-mean-centered Xdev -> Ydev
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}\n(B) WITHIN-SUBJECT marginal R^2 (subject-mean-centered)\n{'=' * 60}")
    within_results = run_marginal_loso(X_dev, Y_dev, cv_groups, alphas)
    for name in PRED_NAMES:
        r2 = within_results[name]
        np.save(f"{OUTPUT_DIR}/Within_{name}_R2.npy", r2)
        n = len(r2)
        print(f"  Within_{name:3s}  mean={r2.mean():.4f}  max={r2.max():.4f}  "
              f"nonzero={np.sum(r2 > 0)}/{n}")
        print_top(f"{name} (within-subject)", r2, labels)

    print("\nDone. Saved Between_{e,xf,xs}_r.npy and Within_{e,xf,xs}_R2.npy to ml_results/.")


if __name__ == '__main__':
    main()
