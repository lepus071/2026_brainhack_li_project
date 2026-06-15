# Dual-Rate 2-Variable VPA Pipeline (Smith et al. 2006)

Main analysis direction: using the two state variables of Smith et al. (2006)'s
dual-rate state-space model (x_f, x_s) as the sole behavioral predictors, run a
3-component VPA (Unique_xf, Unique_xs, Shared_xf_xs) + Full_xf_xs. Does not
include the error signal e / Chein & Schneider's three-system framework (that
attempt is treated as a not-adopted direction, fully preserved in
`../version_1/` for reference only).

This folder is now a **fully self-contained pipeline**: the Phase 0A-3, 7
scripts and utilities have been copied here from the `codes/` parent directory
(same content, following the shared-pipeline naming convention). It does not
depend on `../version_1/` or any file in the parent directory, and can be run
end-to-end through Phase 8B.

## Environment

Python dependencies are listed in [`requirements.txt`](requirements.txt)
(numpy, scipy, pandas, scikit-learn, nibabel, nilearn, pyriemann,
matplotlib) — install with `pip install -r requirements.txt`.

In addition to the Python environment, the following external (non-pip)
dependencies are required for specific phases:
- **Docker** or **Singularity** (Phase 0A) — used to run the fMRIPrep
  container.
- **fMRIPrep** (Phase 0A) — run via the Docker/Singularity container above.
  fMRIPrep's internal FreeSurfer steps require a personal **FreeSurfer
  license file** (`license.txt`), free from
  https://surfer.nmr.mgh.harvard.edu/registration.html. Place your own copy
  where `run_fmriprep_batch.sh` expects it — **do not commit this file**, as
  it contains your registered email and license key.
- **AFNI** (Phases 1 and 7) — provides `3dDeconvolve`, `3dttest++`, and the
  `tcsh` runtime used by `run_group_M1_GLM.tcsh`.

## How to run (example commands)

Each phase's script reads the previous phase's output from
`data/derivatives/riemannian_decoding/` (paths are hardcoded at the top of
each script via a `PROJ_DIR` variable — edit this to your local project
root before running). Example invocation order, after Phase 0A/0B/1/2 have
produced their inputs:

```bash
cd codes/version_2

# Phase 3: Riemannian projection + dynamic FC -> ml_results_v2/*.npy
python run_hybrid_riemannian_extraction.py
python fix_roi_labels.py            # produces ml_results/ROI_Labels_481.csv

# Phase 4: nested LOSO + GroupKFold RidgeCV VPA
python run_hybrid_group_decoding.py

# Phase 5: project VPA vectors back to NIfTI volumes
python run_inverse_vpa_mapping.py

# Phase 6: Limbic-network negative control
python run_null_network_validation.py

# Follow-up analyses
python run_marginal_r2_analysis.py
python run_between_within_decomposition.py
python run_between_subject_permutation.py

# Phase 7: group-level statistics (requires AFNI's 3dttest++)
python run_group_statistics.py

# Phase 8A/8B: visualization
python run_interactive_html.py       # -> html_views_v2/*.html
python run_nilearn_render.py         # -> nilearn_renders_v2/*.png
```

## Files (in Phase order)

- **Phase 0A**: `run_fmriprep_batch.sh`, `fmriprep_to_afni_bridge.py`
- **Phase 0B**: `run_hybrid_behavioral_fitting.py`
- **Phase 1**: `run_afni_condition_glm.py`, `run_group_M1_GLM.tcsh`
- **Phase 2**: `build_combined_atlas.py`
- **Phase 3**: `run_hybrid_riemannian_extraction.py`, `fix_roi_labels.py`
  (dynamic FC is estimated with a high-density sliding window over the task
  run: window size = 30 TR (= 60s at TR=2.0s), step = 1 TR, giving
  `n_volumes - 29` overlapping windows per subject)
  (`fix_roi_labels.py` depends on the same `r01_scaled.nii.gz` + atlas as
  Phase 3, and produces `ml_results/ROI_Labels_481.csv`, used by the Phase 4
  follow-up analysis scripts to look up named ROIs)
- **`run_hybrid_group_decoding.py`** (Phase 4): LOSO + GroupKFold RidgeCV,
  outputs group-level and individual-level `.npy` to
  `data/derivatives/riemannian_decoding/ml_results_v2/`:
  - `Unique_xf_R2.npy`, `Unique_xs_R2.npy`, `Shared_xf_xs_R2.npy`, `Full_xf_xs_R2.npy`
  - per-subject: `<component>_R2_sub-XX.npy`
- **`run_inverse_vpa_mapping.py`** (Phase 5): maps `ml_results_v2/*_R2*.npy`
  back to 3D NIfTI, output to `data/derivatives/riemannian_decoding/nifti_axes_v2/`.
- **`run_null_network_validation.py`** (Phase 6): Limbic null-network validation.
- **`run_marginal_r2_analysis.py`**, **`run_between_within_decomposition.py`**,
  **`run_between_subject_permutation.py`**: follow-up analyses to Phase 4,
  focused only on x_f/x_s (marginal R², between/within decomposition,
  permutation test), output to `ml_results_v2/`. Named ROI labels are read
  from `ml_results/ROI_Labels_481.csv` (produced by `fix_roi_labels.py`, the
  correct 481-dimensional feature_col -> label mapping).
- **Phase 7**: `run_group_statistics.py`
- **`run_interactive_html.py`** (Phase 8A): interactive 3D glass-brain HTML
  views of the group-level `nifti_axes_v2/VPA_Map_*_R2.nii.gz` maps, output to
  `html_views_v2/`.
- **`run_nilearn_render.py`** (Phase 8B): static glass-brain + orthogonal
  stat-map PNGs of the same maps, output to `nilearn_renders_v2/`.
- **`subjectlist.txt`**, **`license.txt`**: shared config/license files.

## Execution status: officially run ✅

Phase 4-8B have been rerun in the official pipeline:
- Phase 4 -> `data/derivatives/riemannian_decoding/ml_results_v2/`: 4
  group-level `.npy` + 40 individual-level `.npy` (10 subjects x 4 components).
- Phase 5 -> `data/derivatives/riemannian_decoding/nifti_axes_v2/`: 44
  `VPA_Map_*.nii.gz` files.
- Phase 6: **4/4 PASS** — Limbic network (26 ROIs) mean/max R² is 0.0000 for
  `Unique_xf`, `Unique_xs`, `Shared_xf_xs`, and `Full_xf_xs`.
- Phase 7 (`run_group_statistics.py`): has been run (`3dttest++` group models
  on the per-subject VPA maps). At N=10, no voxel/ROI survives correction —
  consistent with the Section 6 permutation-test result in
  `Pilot_Results_Public_v2.md`. Treat as exploratory/candidate output only,
  not as a statistically validated group result.
- Phase 8A/8B -> `html_views_v2/` (interactive HTML) and
  `nilearn_renders_v2/` (glass-brain + ortho PNG), generated from the
  group-level `nifti_axes_v2/VPA_Map_*_R2.nii.gz` maps.

## `results/` (exploration-stage reference values, for cross-checking against the official group-level results)

- `Unique_xf_2v_R2.npy` (mean=0.0064, max=0.4664)
- `Unique_xs_2v_R2.npy` (mean=0.0020, max=0.1066)
- `Shared_xf_xs_2v_R2.npy` (mean=0.0054, max=0.2004)
- `Full_xf_xs_R2.npy` (mean=0.0073, max=0.4697)

The official output is in `ml_results_v2/`, with names that don't have the
`_2v` suffix; same data, same model, values should match.

## Anatomical localization summary (top hits, from the exploration stage)

- `Unique_xf`: SomMot_33 (0.4664), SomMot_26, Cont_PFCl_4, SomMot_7, DorsAttn_Post_15
- `Unique_xs`: Vis_9 (0.1066), SomMot_35, SomMot_7, AAL3_SN_pr_L, Vis_21
- `Shared_xf_xs`: SomMot_24 (0.2004), Vis_30, Default_PFC_20, Cerebellum_4_5_L

## Follow-up analysis results (marginal R² / between-within / permutation)

- **Marginal R² (raw)**: x_f mean=0.0082, max=0.4361 (28/481 ROIs); x_s
  mean=0.0034, max=0.2223 (22/481 ROIs). Both drop to 0 after detrending.
- **Between-subject (N=10, df=8)**: x_f top |r|=0.877 (`7Networks_LH_SomMot_33`,
  jackknife range [-0.894,-0.858], sign stable); x_s top |r|=0.819
  (`7Networks_RH_Vis_9`, jackknife range [+0.732,+0.882], sign stable).
- **Within-subject**: x_f mean=0.0017, max=0.0516 (98/481 ROIs); x_s
  mean=0.0022, max=0.0421 (99/481 ROIs).
- **Permutation test (N_perm=10000)**: x_f p=0.2914, x_s p=0.7582 — neither
  survives multiple-comparisons correction, consistent with the version_1
  conclusion.

## Why e is not included

`../version_1/` fully preserves the previously attempted 3-variable
(e, x_f, x_s) + Chein & Schneider three-system route. The reasons for
dropping it are described in `Pilot_Results_Public_v2.md`'s appendix:
(1) e is severely collinear with x_f mathematically, and (2) the Meta system
anatomically overlaps with the Con system under the Schaefer atlas and cannot
be independently verified.
