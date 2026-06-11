# 2026_brainheck_li_project
<a href="https://github.com/lepus071"> 
  <img src="https://avatars.githubusercontent.com/u/232158763?v=4&s=100" width="100px;" alt=""/>
  <br />
  <sub><b>Ching-Yi Li</b></sub> 
</a>

---

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
| **0A — fMRI Preprocessing** | fMRIPrep (motion correction, co-registration, normalization to MNI); convert to Percent Signal Change, generate censoring masks and motion regressors | [`codes/run_fmriprep_batch.sh`](codes/run_fmriprep_batch.sh), [`codes/fmriprep_to_afni_bridge.py`](codes/fmriprep_to_afni_bridge.py) |
| **0B — Behavioral Modeling** | Fit the Smith et al. dual-rate state-space model per subject (L-BFGS-B) to extract $e$, $x_f$, $x_s$ time series | [`codes/run_hybrid_behavioral_fitting.py`](codes/run_hybrid_behavioral_fitting.py) |
| **1 — Individualized M1 Localization** | Subject-specific M1 peak via condition GLM (`3dDeconvolve`) within a precentral gyrus mask, with anatomical fallback and group sanity check | [`codes/run_afni_condition_glm.py`](codes/run_afni_condition_glm.py), [`codes/run_group_M1_GLM.tcsh`](codes/run_group_M1_GLM.tcsh) |
| **2 — Multi-Atlas Node Construction** | Build the 482-ROI parcellation: Schaefer 400 (cortex) + AAL3v2 (subcortex/cerebellum) + individualized M1 | [`codes/build_combined_atlas.py`](codes/build_combined_atlas.py) |
| **3 — Riemannian Manifold Projection & Dynamic FC** | Resting-state covariance as subject-specific origin; sliding-window task-state FC (Ledoit-Wolf shrinkage); tangent-space projection; extract M1 seed connectivity vector | [`codes/run_hybrid_riemannian_extraction.py`](codes/run_hybrid_riemannian_extraction.py) |
| **4 — HRF Alignment + Variance Partitioning (VPA)** | Convolve behavioral time series with the SPM canonical HRF; run 7 RidgeCV models (LOSO-CV) to compute the 7 VPA components (Unique/Shared variance for Meta, Con, Rep) | [`codes/run_hybrid_group_decoding.py`](codes/run_hybrid_group_decoding.py) |
| **5 — Inverse VPA Mapping** | Back-project the 464-dimensional VPA vectors into whole-brain NIfTI statistical maps | [`codes/run_inverse_vpa_mapping.py`](codes/run_inverse_vpa_mapping.py) |
| **6 — Null Network Validation** | Repeat VPA on auditory cortex ROIs (no auditory stimulation in task) as a negative control | [`codes/run_null_network_validation.py`](codes/run_null_network_validation.py) |
| **7 — Anatomical Validation & Group Statistics** | Compare Unique VPA peaks against Smith et al. / Chein & Schneider anatomical predictions; extract Delta (total adaptation magnitude) and run hierarchical `3dttest++` group models | [`codes/run_group_statistics.py`](codes/run_group_statistics.py) |
| **8A — Interactive 3D Visualization** | Interactive glass-brain HTML visualizations of group-level VPA maps | [`codes/run_interactive_html.py`](codes/run_interactive_html.py) |
| **8B — High-Resolution Surface Rendering** | Auto-generate SurfIce scripts for publication-quality surface renderings | [`codes/run_surfice_render.py`](codes/run_surfice_render.py), [`codes/export_surfice.py`](codes/export_surfice.py) |

### Other Scripts

| Script | Purpose |
|:-------|:--------|
| [`codes/check_shape.py`](codes/check_shape.py) | Sanity-check array/data shapes during development |
| [`codes/debug_vpa.py`](codes/debug_vpa.py) | Debugging helper for the VPA computation |
| [`codes/get_pvalues.py`](codes/get_pvalues.py) | Extract p-values from group statistics outputs |
| [`codes/investigate_bg_cb.py`](codes/investigate_bg_cb.py) | Investigate basal ganglia / cerebellum results |
| [`codes/verify_vpa_regions.py`](codes/verify_vpa_regions.py) | Verify VPA results against expected anatomical regions |
| [`codes/subjectlist.txt`](codes/subjectlist.txt) | List of subject IDs used across the pipeline |

For the full methodological write-up (rationale, equations, and design notes for each stage), see [`codes/Hybrid_Dynamic_Decoding_Pipeline.md`](codes/Hybrid_Dynamic_Decoding_Pipeline.md).
