#!/usr/bin/env python3
# ==============================================================================
# check_detrend_variance.py
#
# Diagnostic for run_marginal_r2_analysis.py: how much of each behavioral
# regressor's (e, x_f, x_s) variance, per subject, is just a session-level
# linear trend?
#
#     retained_var_ratio = Var(detrended) / Var(raw)
#
# A ratio close to 0 means the regressor is almost entirely a linear ramp
# over the session (i.e., the marginal-R2 collapse to 0 after detrending is
# expected, regardless of whether any "content" signal exists).
# A ratio close to 1 means detrending barely changed the regressor, so the
# R2 collapse reflects a genuine absence of content-level signal.
# ==============================================================================

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_hybrid_group_decoding import load_and_align_data
from run_marginal_r2_analysis import detrend_per_subject

PROJ_DIR     = '/home/ser/2026_brainhack_li_project'
SUBJECT_LIST = f'{PROJ_DIR}/codes/subjectlist.txt'
PRED_NAMES   = ['e', 'xf', 'xs']


def main():
    with open(SUBJECT_LIST, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    X_behav_raw, Y_brain_raw, cv_groups = load_and_align_data(subjects)
    X_behav_detrend = detrend_per_subject(X_behav_raw, cv_groups)

    print(f"{'Subject':10s} " + "".join(f"{n:>12s}" for n in PRED_NAMES))
    ratios = {n: [] for n in PRED_NAMES}
    for subj_id in np.unique(cv_groups):
        idx = cv_groups == subj_id
        row = f"sub-{int(subj_id):02d}".ljust(10)
        for i, name in enumerate(PRED_NAMES):
            raw_var = X_behav_raw[idx, i].var()
            dt_var = X_behav_detrend[idx, i].var()
            ratio = dt_var / raw_var if raw_var > 0 else np.nan
            ratios[name].append(ratio)
            row += f"{ratio:12.3f}"
        print(row)

    print("\nMean retained variance ratio across subjects:")
    for name in PRED_NAMES:
        vals = np.array(ratios[name])
        print(f"  {name:3s}: mean={np.nanmean(vals):.3f}  "
              f"min={np.nanmin(vals):.3f}  max={np.nanmax(vals):.3f}")


if __name__ == '__main__':
    main()
