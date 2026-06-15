#!/usr/bin/env python3
# ==============================================================================
# run_between_subject_permutation.py
#
# Permutation test for the between-subject correlations found by
# run_between_within_decomposition.py (Pearson r(Xbar, Ybar), N=10, df=8).
#
# For each predictor (e, x_f, x_s), we:
#   1. Recompute the per-subject-mean correlation r(Xbar, Ybar) for all 481 ROIs.
#   2. Randomly permute the assignment of Xbar values to subjects (Ybar fixed),
#      recompute r for all 481 ROIs, and record the max |r| across ROIs
#      (controls for the "pick the best of 481" multiple-comparisons problem).
#   3. p-value for the observed top-ROI |r| = fraction of permutations whose
#      max |r| (across all 481 ROIs) >= observed top |r|.
#
# With N=10 subjects, there are 10! orderings but only C(10, k) distinct
# correlation values matter for Pearson r up to sign/permutation symmetry --
# we just brute-force random permutations (default 10000).
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_hybrid_group_decoding import load_and_align_data
from run_between_within_decomposition import subject_means, load_roi_labels

PROJ_DIR     = '/home/ser/2026_brainhack_li_project'
SUBJECT_LIST = f'{PROJ_DIR}/codes/subjectlist.txt'
ATLAS_LABELS = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases/Combined_Atlas_Labels.csv'
N_PERM       = 10000
PRED_NAMES   = ['e', 'xf', 'xs']
SEED         = 0


def corr_all_rois(x, Y):
    """Pearson r(x, Y[:, j]) for every column j. x: (N,), Y: (N, n_rois)."""
    x = x - x.mean()
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = x @ Y
    den = np.sqrt((x ** 2).sum()) * np.sqrt((Y ** 2).sum(axis=0))
    den[den == 0] = np.nan
    r = num / den
    return np.nan_to_num(r)


def main():
    with open(SUBJECT_LIST, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    print("Loading and aligning data (same as Phase 4)...")
    X_behav_raw, Y_brain_raw, cv_groups = load_and_align_data(subjects)
    print(f"  > Total samples: {X_behav_raw.shape[0]} windows, "
          f"{len(np.unique(cv_groups))} subjects, {Y_brain_raw.shape[1]} ROIs")

    labels = load_roi_labels(Y_brain_raw.shape[1])
    _, X_means, subj_ids = subject_means(X_behav_raw, cv_groups)
    _, Y_means, _ = subject_means(Y_brain_raw, cv_groups)
    n_subj = len(subj_ids)

    rng = np.random.default_rng(SEED)

    print(f"\n{'=' * 60}\nPermutation test (N_perm={N_PERM}, N_subj={n_subj}, df={n_subj - 2})\n{'=' * 60}")

    for i, name in enumerate(PRED_NAMES):
        x = X_means[:, i]
        r_obs = corr_all_rois(x, Y_means)
        top_idx = np.argmax(np.abs(r_obs))
        r_top = r_obs[top_idx]

        # Null distribution: max |r| across all 481 ROIs, per permutation
        max_abs_r_perm = np.empty(N_PERM)
        for p in range(N_PERM):
            perm = rng.permutation(n_subj)
            r_perm = corr_all_rois(x[perm], Y_means)
            max_abs_r_perm[p] = np.abs(r_perm).max()

        p_value = np.mean(max_abs_r_perm >= np.abs(r_top))
        # Also report the per-ROI (uncorrected) null threshold for reference
        unc_threshold_95 = np.percentile(np.abs(corr_all_rois(x[rng.permutation(n_subj)], Y_means)), 95)

        print(f"\n  {name}: top ROI #{top_idx} ({labels[top_idx]}), r={r_top:+.3f}")
        print(f"    Max-|r|-across-481-ROIs null: mean={max_abs_r_perm.mean():.3f}, "
              f"95th pct={np.percentile(max_abs_r_perm, 95):.3f}, "
              f"max={max_abs_r_perm.max():.3f}")
        print(f"    p-value (multiple-comparisons corrected, max-stat) = {p_value:.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
