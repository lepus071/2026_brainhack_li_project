# version_0 — Shared Pipeline (Phase 0A-3, 7)

This is the original pipeline description, covering the shared preprocessing
and analysis stages (Phase 0A-3, 7) plus the original 7-component VPA
(Phase 4-8, including the error signal $e$ and Chein & Schneider mapping).
The Phase 4-8 scripts referenced below now live in
[`../version_1/`](../version_1/) (archived as a not-adopted exploratory
direction — see the root [README](../../README.md)). The current main
pipeline is [`../version_2/`](../version_2/).

## Hybrid Dynamic Neural Decoding Pipeline

An fMRI analysis pipeline for visuomotor rotation (VMR) adaptation tasks, bridging two complementary theories:

- **Chein & Schneider (2012) — Triarchic Theory of Learning**: skill acquisition engages three parallel systems — a metacognitive monitoring system, a cognitive control network (CCN), and an implicit representation system.
- **Smith et al. (2006) — Dual-Rate State-Space Model**: motor learning is driven by a fast-adapting/quickly-forgetting process ($x_f$) and a slow-learning/persistent process ($x_s$), both driven by sensory prediction error ($e$).

The pipeline tracks the dynamic evolution of brain network connectivity over the course of learning and tests how much of that connectivity variance is uniquely explained by each of the three behavioral components ($e$, $x_f$, $x_s$).

**Key techniques:** fMRIPrep preprocessing · individualized M1 seed localization · Riemannian manifold tangent-space projection · Ridge regression variance partitioning analysis (VPA) · SPM HRF convolution for cross-modal temporal alignment · AFNI group statistics with Delta as a between-subjects covariate.

**Dataset:** OpenNeuro [ds005598](https://openneuro.org/datasets/ds005598) — right-handed VMR task with a 45° cursor rotation block (no washout), preceded by a resting-state scan.

**Atlas:** 482 ROIs total — 1 individualized M1 + 399 Schaefer 400 cortical parcels + 82 AAL3v2 subcortical/cerebellar regions.

### Pipeline Stages & Corresponding Code

| Stage | Description | Script(s) |
|:------|:------------|:----------|
| **0A — fMRI Preprocessing** | fMRIPrep (motion correction, co-registration, normalization to MNI); convert to Percent Signal Change, generate censoring masks and motion regressors | [`run_fmriprep_batch.sh`](run_fmriprep_batch.sh), [`fmriprep_to_afni_bridge.py`](fmriprep_to_afni_bridge.py) |
| **0B — Behavioral Modeling** | Fit the Smith et al. dual-rate state-space model per subject (L-BFGS-B) to extract $e$, $x_f$, $x_s$ time series | [`run_hybrid_behavioral_fitting.py`](run_hybrid_behavioral_fitting.py) |
| **1 — Individualized M1 Localization** | Subject-specific M1 peak via condition GLM (`3dDeconvolve`) within a precentral gyrus mask, with anatomical fallback and group sanity check | [`run_afni_condition_glm.py`](run_afni_condition_glm.py), [`run_group_M1_GLM.tcsh`](run_group_M1_GLM.tcsh) |
| **2 — Multi-Atlas Node Construction** | Build the 482-ROI parcellation: Schaefer 400 (cortex) + AAL3v2 (subcortex/cerebellum) + individualized M1 | [`build_combined_atlas.py`](build_combined_atlas.py) |
| **3 — Riemannian Manifold Projection & Dynamic FC** | Resting-state covariance as subject-specific origin; sliding-window task-state FC (Ledoit-Wolf shrinkage); tangent-space projection; extract M1 seed connectivity vector | [`run_hybrid_riemannian_extraction.py`](run_hybrid_riemannian_extraction.py), [`fix_roi_labels.py`](fix_roi_labels.py) |
| **4 — HRF Alignment + Variance Partitioning (VPA)** | Convolve behavioral time series with the SPM canonical HRF; run 7 RidgeCV models (LOSO-CV) to compute the 7 VPA components (Unique/Shared variance for Meta, Con, Rep) | [`../version_1/run_hybrid_group_decoding.py`](../version_1/run_hybrid_group_decoding.py) |
| **5 — Inverse VPA Mapping** | Back-project the 464-dimensional VPA vectors into whole-brain NIfTI statistical maps | [`../version_1/run_inverse_vpa_mapping.py`](../version_1/run_inverse_vpa_mapping.py) |
| **6 — Null Network Validation** | Repeat VPA on auditory cortex ROIs (no auditory stimulation in task) as a negative control | [`../version_1/run_null_network_validation.py`](../version_1/run_null_network_validation.py) |
| **7 — Anatomical Validation & Group Statistics** | Compare Unique VPA peaks against Smith et al. / Chein & Schneider anatomical predictions; extract Delta (total adaptation magnitude) and run hierarchical `3dttest++` group models | [`run_group_statistics.py`](run_group_statistics.py) |
| **8A — Interactive 3D Visualization** | Interactive glass-brain HTML visualizations of group-level VPA maps | [`../version_1/run_interactive_html.py`](../version_1/run_interactive_html.py) |
| **8B — Static Brain Renderings** | Glass-brain and orthogonal stat-map PNGs of group-level VPA maps via nilearn | [`../version_1/run_nilearn_render.py`](../version_1/run_nilearn_render.py) |
