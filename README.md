# 2026_brainheck_li_project
<a href="https://github.com/lepus071"> 
  <img src="https://avatars.githubusercontent.com/u/232158763?v=4&s=100" width="100px;" alt=""/>
  <br />
  <sub><b>Ching-Yi Li</b></sub> 
</a>

---

## Hybrid Dynamic Neural Decoding Pipeline

An fMRI analysis pipeline for visuomotor rotation (VMR) adaptation tasks, based on
**Smith et al. (2006)'s Dual-Rate State-Space Model**: motor learning is driven by
a fast-adapting/quickly-forgetting process ($x_f$) and a slow-learning/persistent
process ($x_s$). The pipeline tracks the dynamic evolution of brain network
connectivity over the course of learning and tests how much of that connectivity
variance is uniquely explained by $x_f$ and $x_s$.

**Key techniques:** fMRIPrep preprocessing · individualized M1 seed localization · Riemannian manifold tangent-space projection · Ridge regression variance partitioning analysis (VPA) · SPM HRF convolution for cross-modal temporal alignment · AFNI group statistics with Delta as a between-subjects covariate.

**Dataset:** OpenNeuro [ds005598](https://openneuro.org/datasets/ds005598) — right-handed VMR task with a 45° cursor rotation block (no washout), preceded by a resting-state scan.

**Atlas:** 482 ROIs total — 1 individualized M1 + 399 Schaefer 400 cortical parcels + 82 AAL3v2 subcortical/cerebellar regions.

**Results:** see [Pilot_Results_Public_v2.md](Pilot_Results_Public_v2.md) (N=10 pilot report).

## Main pipeline: `codes/version_2/`

[`codes/version_2/`](codes/version_2/) is the current, self-contained pipeline (Phase 0A-7
plus follow-up analyses), built around the 2-variable ($x_f$, $x_s$) dual-rate model.

### Pipeline Stages & Corresponding Code

| Stage | Description | Script(s) |
|:------|:------------|:----------|
| **0A — fMRI Preprocessing** | fMRIPrep (motion correction, co-registration, normalization to MNI); convert to Percent Signal Change, generate censoring masks and motion regressors | [`run_fmriprep_batch.sh`](codes/version_2/run_fmriprep_batch.sh), [`fmriprep_to_afni_bridge.py`](codes/version_2/fmriprep_to_afni_bridge.py) |
| **0B — Behavioral Modeling** | Fit the Smith et al. dual-rate state-space model per subject (L-BFGS-B) to extract $x_f$, $x_s$ time series | [`run_hybrid_behavioral_fitting.py`](codes/version_2/run_hybrid_behavioral_fitting.py) |
| **1 — Individualized M1 Localization** | Subject-specific M1 peak via condition GLM (`3dDeconvolve`) within a precentral gyrus mask, with anatomical fallback and group sanity check | [`run_afni_condition_glm.py`](codes/version_2/run_afni_condition_glm.py), [`run_group_M1_GLM.tcsh`](codes/version_2/run_group_M1_GLM.tcsh) |
| **2 — Multi-Atlas Node Construction** | Build the 482-ROI parcellation: Schaefer 400 (cortex) + AAL3v2 (subcortex/cerebellum) + individualized M1 | [`build_combined_atlas.py`](codes/version_2/build_combined_atlas.py) |
| **3 — Riemannian Manifold Projection & Dynamic FC** | Resting-state covariance as subject-specific origin; sliding-window task-state FC (Ledoit-Wolf shrinkage); tangent-space projection; extract M1 seed connectivity vector; recover the 481-ROI label alignment | [`run_hybrid_riemannian_extraction.py`](codes/version_2/run_hybrid_riemannian_extraction.py), [`fix_roi_labels.py`](codes/version_2/fix_roi_labels.py) |
| **4 — HRF Alignment + Variance Partitioning (VPA)** | Convolve $x_f$/$x_s$ with the SPM canonical HRF; nested LOSO + GroupKFold RidgeCV to compute the 3-component VPA (Unique $x_f$, Unique $x_s$, Shared $x_f$-$x_s$) + Full model | [`run_hybrid_group_decoding.py`](codes/version_2/run_hybrid_group_decoding.py) |
| **5 — Inverse VPA Mapping** | Back-project the 481-ROI VPA vectors into whole-brain NIfTI statistical maps | [`run_inverse_vpa_mapping.py`](codes/version_2/run_inverse_vpa_mapping.py) |
| **6 — Null Network Validation** | VPA on the Limbic network (26 ROIs, no expected motor-learning signal) as a negative control | [`run_null_network_validation.py`](codes/version_2/run_null_network_validation.py) |
| **7 — Group Statistics** | Hierarchical `3dttest++` group models | [`run_group_statistics.py`](codes/version_2/run_group_statistics.py) |
| **Follow-up — Marginal R² / Between-Within / Permutation** | Marginal R² for $x_f$/$x_s$ alone, between- vs. within-subject decomposition, and a permutation test on the between-subject correlations | [`run_marginal_r2_analysis.py`](codes/version_2/run_marginal_r2_analysis.py), [`run_between_within_decomposition.py`](codes/version_2/run_between_within_decomposition.py), [`run_between_subject_permutation.py`](codes/version_2/run_between_subject_permutation.py) |
| **8A/8B — Visualization** | Interactive 3D glass-brain HTML views and static glass-brain/orthogonal PNG renderings of the group-level VPA maps | [`run_interactive_html.py`](codes/version_2/run_interactive_html.py), [`run_nilearn_render.py`](codes/version_2/run_nilearn_render.py) |

See [`codes/version_2/README.md`](codes/version_2/README.md) for execution status and results.

## Earlier versions (background only)

- [`codes/version_0/`](codes/version_0/) — the original shared pipeline (Phase 0A-3, 7), as first written.
- [`codes/version_1/`](codes/version_1/) — an exploratory 3-variable model adding a prediction-error
  signal $e$ and a Chein & Schneider (2012) Triarchic Theory interpretation
  (Meta/Con/Rep systems). **Not adopted** for the main analysis — see the appendix
  of [Pilot_Results_Public_v2.md](Pilot_Results_Public_v2.md) for why. Kept for reference.
