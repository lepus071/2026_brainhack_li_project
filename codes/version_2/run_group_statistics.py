#!/usr/bin/env python3
# ==============================================================================
# run_group_statistics.py
#
# Hybrid Dynamic Neural Decoding Pipeline - Phase 7:
# Group Statistics & Behavioral Alignment
#
# Function:
# 1. Extract per-subject behavioral summary scalars from the state-space model output CSV:
#    - Delta       : Absolute behavioral difference (onset mean - plateau mean).
#                    Directly computed from Actual_Performance column.
#                    PURPOSE: Group-level covariate in 3dttest++ only (NOT in Phase 4 encoding).
#                    RATIONALE: Captures total learning magnitude (behavioral outcome),
#                    complementing the dynamic parameters (A_f, B_f, A_s, B_s) which
#                    capture process dynamics. Allows the key question:
#                    "Do subjects who adapted more show stronger neural coupling?"
#    - LearnRate   : Trial index at which performance crosses 50% of its plateau value
#                    (half-saturation index). Proxy for adaptation speed.
#    - Plateau_Var : Variance of performance in the final 5 trials. Proxy for stability.
#    - Af, Bf, As, Bs : Individual state-space model parameters (from model fit metadata,
#                    if available). These capture learning/retention dynamics separately
#                    from Delta (the behavioral outcome).
#
# 2. Write Group_Behavioral_Covariates.txt for AFNI 3dttest++ -covariates flag.
#
# 3. Print ready-to-run 3dttest++ commands for whole-brain group statistics:
#    - Main model  : Tests whether group-mean VPA R2 > 0 (basic group effect)
#    - +Delta model: Tests Delta as moderator of neural-behavioral coupling strength
#    - Full model  : Delta + LearnRate + Plateau_Var as simultaneous covariates
#
# ==============================================================================

import os
import pandas as pd
import numpy as np

PROJ_DIR     = '/home/ser/2026_brainheck_li_project'
SUBJECT_LIST = f'{PROJ_DIR}/codes/subjectlist.txt'
AFNI_DIR     = f'{PROJ_DIR}/data/derivatives/afni_fmriprep'
OUTPUT_DIR   = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/nifti_axes'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of trials to average for onset-state and plateau-state estimates
N_ONSET_TRIALS   = 5   # First N trials of the rotation block => learning onset
N_PLATEAU_TRIALS = 5   # Last  N trials of the rotation block => plateau state


def compute_delta_from_performance(perf_array):
    """
    Compute Delta = mean(first N_ONSET_TRIALS) - mean(last N_PLATEAU_TRIALS).

    This is derived directly from Actual_Performance in the state-space model CSV,
    which already reflects the model-smoothed behavioral trajectory.
    The subtraction direction is positive when the subject successfully reduced error.

    Note: Delta is a STATIC scalar (one value per subject, invariant across windows).
    It must NOT be used as a predictor in the Phase 4 trial-by-trial encoding model
    because it would appear as a constant column in the regression matrix, adding
    collinearity with x_s without contributing temporal dynamics.
    Its correct role is as a BETWEEN-SUBJECTS covariate at the Phase 7 group level.
    """
    if len(perf_array) < (N_ONSET_TRIALS + N_PLATEAU_TRIALS):
        return np.nan
    onset_mean   = np.nanmean(perf_array[:N_ONSET_TRIALS])
    plateau_mean = np.nanmean(perf_array[-N_PLATEAU_TRIALS:])
    return float(onset_mean - plateau_mean)


def compute_learn_rate(perf_array):
    """
    Half-saturation trial index: first trial where performance falls below 50% of plateau mean.
    Returns the trial index (smaller = faster learner).
    """
    if len(perf_array) < 10:
        return np.nan
    plateau_mean = np.nanmean(perf_array[-N_PLATEAU_TRIALS:])
    half_target  = plateau_mean / 2.0
    crossing     = np.where(perf_array <= half_target)[0]
    return float(crossing[0]) if len(crossing) > 0 else float(len(perf_array))


def main():
    print("\n========================================================")
    print("Start Phase 7: Group Statistics & Behavioral Alignment")
    print("========================================================")

    if not os.path.exists(SUBJECT_LIST):
        print(f"Cannot find subject list: {SUBJECT_LIST}")
        return

    with open(SUBJECT_LIST, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    records = []

    print(f"\nExtracting behavioral scalars for {len(subjects)} subjects...")
    print(f"  Delta definition : mean(first {N_ONSET_TRIALS} trials) - mean(last {N_PLATEAU_TRIALS} trials)")
    print(f"  Data source      : Actual_Performance column in *_Triarchic_Fitted.csv\n")

    for subj in subjects:
        behav_csv = (f"{PROJ_DIR}/data/derivatives/behavioral_features/"
                     f"{subj}/CSV_Data/{subj}_Triarchic_Fitted.csv")

        delta_val    = np.nan
        lr_val       = np.nan
        plateau_var  = np.nan

        if os.path.exists(behav_csv):
            df_b = pd.read_csv(behav_csv)

            # Use rotation block (Block == 3) for learning metrics
            if 'Block' in df_b.columns:
                b3 = df_b[df_b['Block'] == 3]
            else:
                b3 = df_b  # fallback: use entire file

            perf = b3['Actual_Performance'].dropna().values if 'Actual_Performance' in b3.columns else np.array([])

            if len(perf) >= (N_ONSET_TRIALS + N_PLATEAU_TRIALS):
                delta_val   = compute_delta_from_performance(perf)
                lr_val      = compute_learn_rate(perf)
                plateau_var = float(np.var(perf[-N_PLATEAU_TRIALS:]))
            else:
                print(f"  [WARN] {subj}: not enough valid trials in Block 3 ({len(perf)} trials)")
        else:
            print(f"  [WARN] {subj}: behavioral CSV not found at {behav_csv}")

        records.append({
            'Subj':        subj,
            'Delta':       delta_val,
            'LearnRate':   lr_val,
            'Plateau_Var': plateau_var,
        })

    df_cov = pd.DataFrame(records)

    # --- Summary statistics (sanity check) ---
    print("Covariate summary (valid subjects only):")
    valid = df_cov.dropna()
    for col in ['Delta', 'LearnRate', 'Plateau_Var']:
        print(f"  {col:15s}: mean={valid[col].mean():.3f}  sd={valid[col].std():.3f}"
              f"  min={valid[col].min():.3f}  max={valid[col].max():.3f}  N={valid[col].notna().sum()}")

    # --- Write covariate file ---
    cov_file = f"{OUTPUT_DIR}/Group_Behavioral_Covariates.txt"
    valid.to_csv(cov_file, sep='\t', index=False, float_format='%.6f')
    print(f"\n  > Covariates file written: {cov_file}")

    # --- Print 3dttest++ commands ---
    print("\n" + "=" * 60)
    print("AFNI 3dttest++ commands (copy-paste to terminal):")
    print("=" * 60)
    print(f"\ncd {OUTPUT_DIR}\n")

    vpa_parts = [
        "Unique_xf_2v", "Unique_xs_2v", "Shared_xf_xs_2v", "Full_xf_xs"
    ]

    for part in vpa_parts:
        # --- Model A: Group-mean only (intercept test: VPA R2 > 0) ---
        print(f"# Model A - {part}: group-mean VPA R2 > 0")
        print(f"3dttest++ -prefix GroupStat_{part}_mean \\")
        print(f"          -setA VPA_Map_{part}_sub-*.nii.gz")
        print()

        # --- Model B: Delta as sole covariate ---
        # Asks: "Does total learning magnitude (Delta) modulate neural-behavioral coupling?"
        # Scientific rationale:
        #   Delta = static behavioral outcome scalar (one value per subject).
        #   MUST NOT be used in Phase 4 trial-by-trial encoding (it is temporally constant).
        #   Here at Phase 7 it acts as a between-subjects moderator, asking whether
        #   subjects who adapted more (larger Delta) show stronger neural coupling.
        print(f"# Model B - {part}: Delta as group-level moderator (behavioral outcome predictor)")
        print(f"# Key question: Do subjects who learned more show stronger {part} neural coupling?")
        print(f"3dttest++ -prefix GroupStat_{part}_delta \\")
        print(f"          -covariates {cov_file}'[Subj,Delta]' \\")
        print(f"          -setA VPA_Map_{part}_sub-*.nii.gz")
        print()

        # --- Model C: Full covariate model ---
        # Adds LearnRate (process speed) + Plateau_Var (stability) alongside Delta (outcome).
        # Allows partial regression: each covariate's unique contribution is isolated.
        print(f"# Model C - {part}: full covariate model (Delta + LearnRate + Plateau_Var)")
        print(f"3dttest++ -prefix GroupStat_{part}_full \\")
        print(f"          -covariates {cov_file} \\")
        print(f"          -setA VPA_Map_{part}_sub-*.nii.gz")
        print()

    print("=" * 60)
    print("Note on Delta interpretation at group level:")
    print("  - Positive Beta(Delta) in GroupStat_Unique_xf_delta => regions where")
    print("    greater total adaptation is associated with stronger CCN-M1 coupling.")
    print("  - Positive Beta(Delta) in GroupStat_Unique_xs_delta => regions where")
    print("    greater total adaptation is associated with stronger implicit memory coupling.")
    print("  - This separates 'who adapted more' (Delta) from 'how they adapted'")
    print("    (which is captured by the dynamic VPA maps themselves).")
    print("=" * 60)


if __name__ == '__main__':
    main()
