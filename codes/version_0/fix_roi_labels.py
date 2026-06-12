#!/usr/bin/env python3
# ==============================================================================
# fix_roi_labels.py
#
# Recovers the true 481-ROI label list used by the feature vectors in
# Riemannian_Features_*.npy (m1_connections = S_task[0, 1:]).
#
# NiftiLabelsMasker only keeps atlas labels that actually have voxels after
# resampling (masker.labels_). The feature vector drops index 0 of that list
# (M1, the seed itself), leaving len(masker.labels_) - 1 entries == 481.
#
# This script fits the masker on one subject's combined atlas (same call as
# run_hybrid_riemannian_extraction.py), maps masker.labels_[1:] back to the
# Combined_Atlas_Labels.csv names, and writes ROI_Labels_481.csv.
# ==============================================================================

import os
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker

PROJ_DIR     = '/home/ser/2026_brainheck_li_project'
ATLAS_DIR    = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases'
AFNI_DIR     = f'{PROJ_DIR}/data/derivatives/afni_fmriprep'
OUTPUT_DIR   = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results'
SUBJECT_LIST = f'{PROJ_DIR}/codes/subjectlist.txt'


def main():
    with open(SUBJECT_LIST, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    label_df = pd.read_csv(f'{ATLAS_DIR}/Combined_Atlas_Labels.csv')
    index_to_label = dict(zip(label_df['Index'], label_df['Label']))

    label_sets = {}
    for subj in subjects:
        atlas_file = f'{ATLAS_DIR}/Combined_Atlas_{subj}.nii.gz'
        rest_file  = f'{AFNI_DIR}/{subj}/r01_scaled.nii.gz'
        if not (os.path.exists(atlas_file) and os.path.exists(rest_file)):
            print(f"  [SKIP] {subj}: missing files")
            continue

        masker = NiftiLabelsMasker(
            labels_img=atlas_file,
            standardize='zscore_sample',
            resampling_target='labels',
            memory='nilearn_cache',
            verbose=0,
        )
        masker.fit(rest_file)
        labels_present = list(masker.labels_)
        print(f"  {subj}: {len(labels_present)} labels detected (M1 + {len(labels_present) - 1} features)")
        label_sets[subj] = labels_present

    # Sanity check: are all subjects' detected label sets identical?
    sets = {subj: set(l) for subj, l in label_sets.items()}
    ref_subj = next(iter(sets))
    all_same = all(s == sets[ref_subj] for s in sets.values())
    print(f"\nAll subjects share identical detected-label set: {all_same}")
    if not all_same:
        for subj, s in sets.items():
            diff = s.symmetric_difference(sets[ref_subj])
            if diff:
                print(f"  {subj} differs from {ref_subj} by indices: {sorted(diff)}")

    # masker.labels_ comes back as resampled (slightly non-integer) atlas
    # indices, e.g. 1.0023346... for index 1. Round to the nearest int so
    # they match Combined_Atlas_Labels.csv's integer Index column.
    ref_labels = [int(round(i)) for i in label_sets[ref_subj]]
    print("ref_labels[:5] =", ref_labels[:5])

    # ref_labels is sorted ascending: [Background(0), M1(1), Background(2), real ROIs...].
    # The feature vector (S_task[0, 1:]) drops only the first two entries
    # (the leading Background and the M1 seed), keeping the duplicate
    # Background(2) plus the 480 real ROIs => 481 entries.
    feature_indices = ref_labels[2:]
    print(f"\nFeature vector length: {len(feature_indices)} (expected 481)")

    feature_label_names = [index_to_label.get(i, f"UNKNOWN_{i}") for i in feature_indices]

    out_df = pd.DataFrame({
        'feature_col': range(len(feature_indices)),
        'atlas_index': feature_indices,
        'label': feature_label_names,
    })
    out_path = f'{OUTPUT_DIR}/ROI_Labels_481.csv'
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(out_df.head(15))
    print("ref_labels[:5] =", ref_labels[:5])

if __name__ == '__main__':
    main()
