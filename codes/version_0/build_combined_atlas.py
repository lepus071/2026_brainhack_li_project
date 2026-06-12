#!/usr/bin/env python3
# ==============================================================================
# build_combined_atlas.py
#
# Hybrid Dynamic Neural Decoding Pipeline - Phase 2:
# Multi-Atlas Node Construction
#
# Builds a subject-specific combined parcellation atlas with:
#   Index 1         : Individualized M1 seed (from Phase 1)
#   Index 2 ~ 400   : Schaefer 400 cortical parcels (one SomMot node replaced by M1)
#   Index 401+      : AAL3v2 subcortical + cerebellar regions (all 82 non-cortical ROIs)
#
# Total: 1 (M1) + 399 (Schaefer cortex) + 82 (AAL3 subcortex/cerebellum) = 482 ROIs
#
# Design rationale:
#   - Schaefer 400 provides fine-grained functional cortical parcellation.
#   - AAL3v2 provides the full subcortical and cerebellar coverage,
#     including all thalamic sub-nuclei, basal ganglia, hippocampus, amygdala,
#     brainstem nuclei (VTA, SN, Red Nucleus, Locus Coeruleus, Raphe),
#     and all cerebellum / vermis lobules.
#   - AAL3 cortical parcels (coarser and fully overlapping with Schaefer) are
#     EXCLUDED to avoid collinearity and resolution mismatch.
#   - The individualized M1 seed overwrites the Schaefer SomMot node that covers
#     the same anatomy, preserving exact spatial correspondence.
#
# AAL3v2 subcortex/cerebellum index reference (verified against nilearn fetch_atlas_aal):
#   41-42  : Hippocampus L/R
#   45-46  : Amygdala L/R
#   75-80  : Caudate L/R, Putamen L/R, Pallidum L/R
#   95-120 : Cerebellum Crus1/2, lobules 3-10, Vermis 1-10
#   121-150: Thalamic sub-nuclei (AV, LP, VA, VL, VPL, IL, Re, MDm, MDl, LGN, MGN, Pu)
#   151-152: ACC_sub L/R (subcortical part of anterior cingulate)
#   157-158: Nucleus Accumbens L/R
#   159-160: VTA L/R
#   161-164: Substantia Nigra (pars compacta / pars reticulata) L/R
#   165-166: Red Nucleus L/R
#   167-168: Locus Coeruleus L/R
#   169-170: Raphe (Dorsal / Median)
#
# NOTE: ParaHippocampal (43-44) and ACC_pre/ACC_sup (153-156) are cortical and excluded.
# ==============================================================================

import os
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import datasets, image

PROJ_DIR     = '/home/ser/2026_brainheck_li_project'
AFNI_DIR     = f'{PROJ_DIR}/data/derivatives/afni_fmriprep'
SUBJECT_LIST = f'{PROJ_DIR}/codes/subjectlist.txt'
COMBINED_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases'
os.makedirs(COMBINED_DIR, exist_ok=True)

# ==============================================================================
# AAL3v2 subcortex + cerebellum indices (verified against nilearn aal_3v2 dataset)
# These are the ONLY AAL3 regions included; all cortical AAL3 parcels are excluded.
# ==============================================================================
AAL3_SUBCORTEX_CEREBELLUM_INDICES = set([
    # Hippocampus
    41, 42,
    # Amygdala
    45, 46,
    # Basal Ganglia: Caudate, Putamen, Pallidum
    75, 76, 77, 78, 79, 80,
    # Cerebellum lobules: Crus1, Crus2, 3, 4-5, 6, 7b, 8, 9, 10 (L + R)
    95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
    105, 106, 107, 108, 109, 110, 111, 112,
    # Vermis: 1-2, 3, 4-5, 6, 7, 8, 9, 10
    113, 114, 115, 116, 117, 118, 119, 120,
    # Thalamic sub-nuclei: AV, LP, VA, VL, VPL, IL, Re, MDm, MDl, LGN, MGN, PuI, PuM, PuA, PuL
    121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
    131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
    141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
    # ACC_sub (subcortical part of ACC, distinct from cortical ACC_pre/sup)
    151, 152,
    # Nucleus Accumbens
    157, 158,
    # VTA (Ventral Tegmental Area)
    159, 160,
    # Substantia Nigra pars compacta + pars reticulata
    161, 162, 163, 164,
    # Red Nucleus
    165, 166,
    # Locus Coeruleus
    167, 168,
    # Raphe nuclei (Dorsal + Median)
    169, 170,
])

# Expected count: 82 regions
assert len(AAL3_SUBCORTEX_CEREBELLUM_INDICES) == 82, \
    f"Expected 82 AAL3 subcortex/cerebellum regions, got {len(AAL3_SUBCORTEX_CEREBELLUM_INDICES)}"


def main():
    if not os.path.exists(SUBJECT_LIST):
        print(f"Cannot find subject list: {SUBJECT_LIST}")
        return

    with open(SUBJECT_LIST, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    print("Loading base atlases...")
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    # ------------------------------------------------------------------
    # Load Schaefer 400 cortical atlas
    # ------------------------------------------------------------------
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
    schaefer_img  = nib.load(schaefer.maps)
    schaefer_data = schaefer_img.get_fdata()
    schaefer_labels = [label.decode('utf-8') if isinstance(label, bytes) else label
                       for label in schaefer.labels]

    # Identify the Schaefer node covering left M1 hand area (will be replaced by the
    # individualized seed at the end, so we skip it during cortical merging)
    m1_schaefer_idx = None
    for i, label in enumerate(schaefer_labels):
        if '7Networks_LH_SomMot_25' in label:
            m1_schaefer_idx = i + 1   # nilearn parcel indices are 1-based
            break
    if m1_schaefer_idx is None:
        print("  [WARN] Could not find 7Networks_LH_SomMot_25 in Schaefer labels.")
        print("         Proceeding without exclusion. M1 seed will still overwrite at the end.")
    else:
        print(f"  > Schaefer M1 node found (parcel index {m1_schaefer_idx}), will be replaced by individualized seed.")

    # ------------------------------------------------------------------
    # Load AAL3v2 atlas
    # ------------------------------------------------------------------
    aal = datasets.fetch_atlas_aal()
    aal_img  = nib.load(aal.maps)
    aal_data = aal_img.get_fdata()
    aal3_indices = [int(i) for i in aal.indices]
    aal3_labels  = aal.labels

    # Verify we have AAL3v2 (167 non-background regions, max index 170)
    print(f"  > AAL atlas loaded: {len(aal3_labels)} regions, index range {min(aal3_indices)}-{max(aal3_indices)}")
    if max(aal3_indices) < 150:
        print("  [WARN] Loaded atlas appears to be classic AAL (not AAL3). "
              "Check nilearn dataset version. Subcortical coverage may be incomplete.")

    # ------------------------------------------------------------------
    # Build combined label list and index mappings
    # ------------------------------------------------------------------
    combined_labels  = ["Background", "Individual_M1_Seed"]   # index 0 and 1
    schaefer_mapping = {}   # original Schaefer parcel index -> new combined index
    aal3_mapping     = {}   # original AAL3 index -> new combined index
    current_idx = 2         # new combined indices start at 2

    # Schaefer cortical nodes (skip the one replaced by M1)
    for i, label in enumerate(schaefer_labels, start=1):
        if i == m1_schaefer_idx:
            continue
        combined_labels.append(label)
        schaefer_mapping[i] = current_idx
        current_idx += 1

    # AAL3 subcortex + cerebellum nodes
    for orig_idx, label in zip(aal3_indices, aal3_labels):
        if orig_idx in AAL3_SUBCORTEX_CEREBELLUM_INDICES:
            combined_labels.append(f"AAL3_{label}")
            aal3_mapping[orig_idx] = current_idx
            current_idx += 1

    n_rois = len(combined_labels) - 1   # subtract background
    print(f"  > Combined atlas: {n_rois} ROIs total "
          f"(1 individualized M1 + {len(schaefer_mapping)} Schaefer cortex + "
          f"{len(aal3_mapping)} AAL3 subcortex/cerebellum)")
    assert len(aal3_mapping) == 82, \
        f"Expected 82 AAL3 regions in mapping, got {len(aal3_mapping)}"

    # Save label reference table
    label_df = pd.DataFrame({"Index": range(len(combined_labels)), "Label": combined_labels})
    label_csv = f"{COMBINED_DIR}/Combined_Atlas_Labels.csv"
    label_df.to_csv(label_csv, index=False)
    print(f"  > Label reference table saved: {label_csv}")

    # Save human-readable breakdown
    breakdown_path = f"{COMBINED_DIR}/Combined_Atlas_Breakdown.txt"
    with open(breakdown_path, 'w') as f:
        f.write("Hybrid Dynamic Neural Decoding Pipeline - Combined Atlas Breakdown\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Total ROIs          : {n_rois}\n")
        f.write(f"  [1]    Indiv. M1 Seed\n")
        f.write(f"  [2-{1+len(schaefer_mapping)}]  Schaefer 400 cortex ({len(schaefer_mapping)} parcels, one replaced by M1)\n")
        f.write(f"  [{2+len(schaefer_mapping)}-{n_rois}]  AAL3v2 subcortex + cerebellum ({len(aal3_mapping)} regions)\n\n")
        f.write("AAL3 subcortical regions included:\n")
        for orig_idx, new_idx in sorted(aal3_mapping.items()):
            label = aal3_labels[aal3_indices.index(orig_idx)]
            f.write(f"  [{new_idx:4d}] AAL3_{label} (orig AAL3 idx {orig_idx})\n")
    print(f"  > Atlas breakdown saved: {breakdown_path}")

    # ------------------------------------------------------------------
    # Per-subject: merge individualized M1 seed into combined volume
    # ------------------------------------------------------------------
    for subj in subjects:
        print(f"\n{'='*60}")
        print(f"Building combined atlas: {subj}")
        print(f"{'='*60}")

        seed_file = f"{AFNI_DIR}/{subj}/GLM_Condition/M1_indiv_seed.nii.gz"
        if not os.path.exists(seed_file):
            print(f"  [SKIP] Individualized M1 seed not found: {seed_file}")
            print(f"         Run run_afni_condition_glm.py first.")
            continue

        seed_img  = nib.load(seed_file)
        seed_data = seed_img.get_fdata()

        # Resample Schaefer and AAL3 to M1 seed voxel space (nearest-neighbour)
        res_schaefer = image.resample_to_img(schaefer_img, seed_img, interpolation='nearest')
        res_aal3     = image.resample_to_img(aal_img,     seed_img, interpolation='nearest')

        s_data = res_schaefer.get_fdata()
        a_data = res_aal3.get_fdata()

        combined_data = np.zeros_like(seed_data)

        # Fill order (later steps overwrite earlier, resolving overlaps):
        # 1. AAL3 subcortex/cerebellum  (lowest priority — sits under cortex)
        for orig_idx, new_idx in aal3_mapping.items():
            combined_data[a_data == orig_idx] = new_idx

        # 2. Schaefer cortex  (overwrites any subcortical bleed at cortical boundaries)
        for orig_idx, new_idx in schaefer_mapping.items():
            combined_data[s_data == orig_idx] = new_idx

        # 3. Individualized M1 seed  (absolute priority — always index 1)
        combined_data[seed_data > 0] = 1

        combined_img = nib.Nifti1Image(combined_data,
                                        affine=seed_img.affine,
                                        header=seed_img.header)
        out_file = f"{COMBINED_DIR}/Combined_Atlas_{subj}.nii.gz"
        nib.save(combined_img, out_file)
        print(f"  > Saved: {out_file}")

        # Quick voxel count check
        unique_vals = np.unique(combined_data[combined_data > 0])
        print(f"  > Assigned ROIs in volume: {len(unique_vals)} "
              f"(expected {n_rois}, missing = parcels with no voxels at this resolution)")


if __name__ == '__main__':
    main()
