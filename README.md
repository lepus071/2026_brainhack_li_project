# 2026_brainhack_li_project
<a href="https://github.com/lepus071"> 
  <img src="https://avatars.githubusercontent.com/u/232158763?v=4&s=100" width="100px;" alt=""/>
  <br />
  <sub><b>Ching-Yi Li</b></sub> 
</a>

---

## Summary

When you adapt to a visual rotation while reaching, your motor system relies on two
processes that learn and forget at different speeds: a **fast** one ($x_f$) and a
**slow** one ($x_s$) (Smith et al., 2006). This pipeline takes fMRI data from 10
subjects performing a visuomotor rotation task, fits each subject's $x_f$/$x_s$
trajectories from their behavior, and asks: **which brain regions' connectivity
dynamics uniquely track the fast process, which track the slow process, and which
track both?** It does so with a leakage-free nested cross-validation scheme
(Leave-One-Subject-Out + GroupKFold) and a negative-control network to check the
results aren't artifacts.

- **Status**: Phases 0A-8B complete for N=10. Results are directional pilot
  findings, not yet statistically significant at the whole-brain level — see
  [Pilot_Results_Public_v2.md](docs/Pilot_Results_Public_v2.md) for the honest
  numbers.
- **Where to start reading**: this README → [Pilot_Results_Public_v2.md](docs/Pilot_Results_Public_v2.md)
  → [`codes/version_2/README.md`](codes/version_2/README.md) for execution details.

## Documentation

| Document | Purpose |
|---|---|
| [README.md](README.md) (this file) | Project overview, pipeline stages, dataset/task description, references |
| [Pilot_Results_Public_v2.md](docs/Pilot_Results_Public_v2.md) | **Main results report** (N=10): group-level VPA maps, anatomical localization, null-network validation, group statistics, and honest discussion of significance |
| [Physiological_Mechanisms_Smith2006.md](docs/Physiological_Mechanisms_Smith2006.md) | Background reading on the physiological interpretation of the Smith et al. (2006) dual-rate model ($x_f$/$x_s$) |
| [Riemannian_Manifold_Projection_Review.md](docs/Riemannian_Manifold_Projection_Review.md) | Methodological literature review: Riemannian tangent-space projection vs. Pearson/partial-correlation/ICA/GNN methods, with implementation guidelines and pitfalls |
| [`codes/version_2/README.md`](codes/version_2/README.md) | Execution status and per-script notes for the current pipeline |
| [`codes/version_2/requirements.txt`](codes/version_2/requirements.txt) | Python dependencies + non-pip external tools (Docker/Singularity, FreeSurfer license, AFNI) |

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

**Results:** see [Pilot_Results_Public_v2.md](docs/Pilot_Results_Public_v2.md) (N=10 pilot report).

## Environment

Python dependencies are listed in [`codes/version_2/requirements.txt`](codes/version_2/requirements.txt)
(`numpy`, `scipy`, `pandas`, `scikit-learn`, `nibabel`, `nilearn`, `pyriemann`, `matplotlib`).

Phase 0A additionally requires:
- Docker or Singularity, to run the fMRIPrep container
- A FreeSurfer license file (`license.txt`) — get a free copy from [surfer.nmr.mgh.harvard.edu](https://surfer.nmr.mgh.harvard.edu/registration.html); do **not** commit this file
- AFNI, for `3dDeconvolve` (Phase 1) and `3dttest++` (Phase 7)

## Data

Raw and derivative data are **not** committed to this repository (`data/` and `work/`
are in `.gitignore`). Raw fMRI/behavioral data come from OpenNeuro dataset
[ds005598](https://openneuro.org/datasets/ds005598); all derivatives (preprocessed
images, features, VPA maps, figures used in reports) are regenerated locally by
running the pipeline stages below.

## Main pipeline: `codes/version_2/`

[`codes/version_2/`](codes/version_2/) is the current, self-contained pipeline (Phase 0A-8
plus follow-up analyses), built around the 2-variable ($x_f$, $x_s$) dual-rate model.

### Pipeline Stages & Corresponding Code

| Stage | Description | Script(s) |
|:------|:------------|:----------|
| **0A — fMRI Preprocessing** | fMRIPrep (motion correction, co-registration, normalization to MNI); convert to Percent Signal Change, generate censoring masks and motion regressors | [`run_fmriprep_batch.sh`](codes/version_2/run_fmriprep_batch.sh), [`fmriprep_to_afni_bridge.py`](codes/version_2/fmriprep_to_afni_bridge.py) |
| **0B — Behavioral Modeling** | Fit the Smith et al. dual-rate state-space model per subject (L-BFGS-B) to extract $x_f$, $x_s$ time series | [`run_hybrid_behavioral_fitting.py`](codes/version_2/run_hybrid_behavioral_fitting.py) |
| **1 — Individualized M1 Localization** | Subject-specific M1 peak via condition GLM (`3dDeconvolve`) within a precentral gyrus mask, with anatomical fallback and group sanity check | [`run_afni_condition_glm.py`](codes/version_2/run_afni_condition_glm.py), [`run_group_M1_GLM.tcsh`](codes/version_2/run_group_M1_GLM.tcsh) |
| **2 — Multi-Atlas Node Construction** | Build the 482-ROI parcellation: Schaefer 400 (cortex) + AAL3v2 (subcortex/cerebellum) + individualized M1 | [`build_combined_atlas.py`](codes/version_2/build_combined_atlas.py) |
| **3 — Riemannian Manifold Projection & Dynamic FC** | Resting-state covariance as subject-specific origin; sliding-window (30 TR / 60s window, step = 1 TR) task-state FC (Ledoit-Wolf shrinkage); tangent-space projection; extract M1 seed connectivity vector; recover the 481-ROI label alignment | [`run_hybrid_riemannian_extraction.py`](codes/version_2/run_hybrid_riemannian_extraction.py), [`fix_roi_labels.py`](codes/version_2/fix_roi_labels.py) |
| **4 — HRF Alignment + Variance Partitioning (VPA)** | Convolve $x_f$/$x_s$ with the SPM canonical HRF; nested LOSO + GroupKFold RidgeCV to compute the 3-component VPA (Unique $x_f$, Unique $x_s$, Shared $x_f$-$x_s$) + Full model | [`run_hybrid_group_decoding.py`](codes/version_2/run_hybrid_group_decoding.py) |
| **5 — Inverse VPA Mapping** | Back-project the 481-ROI VPA vectors into whole-brain NIfTI statistical maps | [`run_inverse_vpa_mapping.py`](codes/version_2/run_inverse_vpa_mapping.py) |
| **6 — Null Network Validation** | VPA on the Limbic network (26 ROIs, no expected motor-learning signal) as a negative control | [`run_null_network_validation.py`](codes/version_2/run_null_network_validation.py) |
| **7 — Group Statistics** | Hierarchical `3dttest++` group models | [`run_group_statistics.py`](codes/version_2/run_group_statistics.py) |
| **Follow-up — Marginal R² / Between-Within / Permutation** | Marginal R² for $x_f$/$x_s$ alone, between- vs. within-subject decomposition, and a permutation test on the between-subject correlations | [`run_marginal_r2_analysis.py`](codes/version_2/run_marginal_r2_analysis.py), [`run_between_within_decomposition.py`](codes/version_2/run_between_within_decomposition.py), [`run_between_subject_permutation.py`](codes/version_2/run_between_subject_permutation.py) |
| **8A/8B — Visualization** | Interactive 3D glass-brain HTML views and static glass-brain/orthogonal PNG renderings of the group-level VPA maps | [`run_interactive_html.py`](codes/version_2/run_interactive_html.py), [`run_nilearn_render.py`](codes/version_2/run_nilearn_render.py) |

See [`codes/version_2/README.md`](codes/version_2/README.md) for execution status and results.

### Dataset and task description

This pipeline uses OpenNeuro dataset [ds005598](https://openneuro.org/datasets/ds005598), a right-handed **visuomotor rotation (VMR) adaptation** task (N=10 subjects used in this pilot: sub-01–04, 06–11).

**What subjects did in the scanner:**
- Subjects held an MRI-compatible joystick/trackball and made center-out reaching movements toward an on-screen visual target, viewing only a cursor representing their hand position (direct hand vision occluded).
- The session began with a **resting-state scan** (eyes open, no task) — used in Phase 3 as each subject's individual Riemannian reference point.
- During the task run, after a **baseline (unrotated) block**, the visual feedback cursor was rotated **45° relative to actual hand-movement direction**, with no washout block — subjects had to adapt to and sustain the rotation for the remainder of the run, with no final de-adaptation phase.
- Each trial's **angular error** (difference between target direction and initial movement direction) is the raw behavioral measure, fed into the Smith et al. (2006) dual-rate model fit (Phase 0B) to extract the $x_f$ (fast) and $x_s$ (slow) state trajectories used throughout the pipeline.

### What each stage is for

- **Phase 0A — fMRI Preprocessing**: Raw fMRI data is dominated by head-motion artifacts and scanner drift. fMRIPrep standardizes motion correction, slice-timing correction, and normalization to MNI space across all subjects, and the bridge script converts the output to percent-signal-change with motion/censoring regressors — without this, any downstream connectivity measure would be confounded by subject-specific motion patterns.
- **Phase 0B — Behavioral Modeling**: Raw trial-by-trial angular error is noisy (hand tremor, attention lapses). Fitting Smith et al.'s dual-rate state-space model via L-BFGS-B decomposes this noisy error signal into two smooth latent trajectories, $x_f$ (fast, quickly-adapting/quickly-forgetting) and $x_s$ (slow, persistent) — these become the behavioral regressors for everything downstream.
- **Phase 1 — Individualized M1 Localization**: The hand-area of primary motor cortex (M1) varies in exact location across individuals. Using each subject's own task-evoked GLM activation (rather than a group-average atlas coordinate) ensures the seed region used for connectivity analysis is anatomically accurate for that subject.
- **Phase 2 — Multi-Atlas Node Construction**: Combines the Schaefer 400 cortical parcellation with the AAL3v2 subcortical/cerebellar atlas plus the individualized M1 seed, producing one consistent 482-ROI parcellation that gives whole-brain coverage (cortex + subcortex + cerebellum) while preserving subject-specific motor localization.
- **Phase 3 — Riemannian Manifold Projection & Dynamic FC**: Functional connectivity matrices are Symmetric Positive Definite (SPD) and do not live in Euclidean space — naively averaging or comparing them ignores their geometric structure and conflates each subject's idiosyncratic "resting" connectivity baseline with task-related change. This stage projects each subject's sliding-window task-state covariance matrices into the tangent space at that subject's own resting-state reference point, removing individual baseline differences while respecting SPD geometry (see Riemannian geometry references below). `fix_roi_labels.py` recovers the correct 481-ROI label alignment for the resulting feature vectors.
- **Phase 4 — HRF Alignment + Variance Partitioning (VPA)**: The behavioral signal ($x_f$/$x_s$) and the BOLD signal are on different timescales — neural events are instantaneous but BOLD responses lag by ~4-6 seconds. Convolving $x_f$/$x_s$ with the SPM canonical hemodynamic response function (HRF) aligns them with the BOLD timeseries. Variance partitioning then statistically separates the connectivity variance uniquely attributable to $x_f$, uniquely to $x_s$, and shared between them — all under leakage-free nested LOSO + GroupKFold cross-validation.
- **Phase 5 — Inverse VPA Mapping**: The VPA results exist as 481-length vectors (one value per ROI). This stage projects those vectors back into whole-brain NIfTI volumes so the results can be visualized and interpreted anatomically. Correct alignment is critical here: each vector position must map to the same ROI it represented in Phase 3/4. This is guaranteed by `ml_results/ROI_Labels_481.csv` (produced by `fix_roi_labels.py` in Phase 3), which records the feature-column-to-ROI-label correspondence; Phase 5 uses this table to look up each ROI's mask in the combined atlas (Phase 2) and writes its R² value into the corresponding voxels.
- **Phase 6 — Null Network Validation**: A core question for any decoding pipeline is "could this just be a global artifact?" Re-running the VPA on the Limbic network — which has no expected motor-learning signal in this task — and finding R²≈0 is the negative control that establishes spatial specificity of the main findings.
- **Phase 7 — Group Statistics**: Aggregates subject-level results into hierarchical group-level statistical models (`3dttest++`), enabling population-level inference rather than single-subject anecdote.
- **Follow-up analyses (Marginal R² / Between-Within / Permutation)**: Because $x_f$ and $x_s$ are themselves correlated, these scripts further decompose each predictor's marginal contribution into between-subject vs. within-subject variance, and run a permutation test to assess whether the observed between-subject correlations exceed what's expected by chance at N=10.
- **Phase 8A/8B — Visualization**: Produces interactive (HTML) and static (PNG) brain renderings of the group-level VPA maps, used for both exploration and reporting (see [Pilot_Results_Public_v2.md](docs/Pilot_Results_Public_v2.md)).

### Key references

- Smith, M. A., Ghazizadeh, A., & Shadmehr, R. (2006). Interacting Adaptive Processes with Different Timescales Underlie Short-Term Motor Learning. *PLoS Biology*, 4(6), e179. https://doi.org/10.1371/journal.pbio.0040179
- Varoquaux, G., Baronnet, F., Kleinschmidt, A., Fillard, P., & Thirion, B. (2010). Detection of brain functional-connectivity difference in post-stroke patients using group-level covariance modeling. *MICCAI 2010*, LNCS vol. 6361, pp. 200–208. https://doi.org/10.1007/978-3-642-15705-9_25 — the foundational paper introducing SPD-matrix-manifold and tangent-space modeling for comparing fMRI functional connectivity across subjects/groups; proposes a probabilistic model for between-subject comparison on the SPD manifold and demonstrates higher statistical sensitivity in a stroke-patient vs. healthy-control comparison.
- Pennec, X., Fillard, P., & Ayache, N. (2006). A Riemannian Framework for Tensor Computing. *International Journal of Computer Vision*, 66(1), 41–66. https://doi.org/10.1007/s11263-005-3222-z — establishes the affine-invariant Riemannian metric and the Log/Exp maps on the SPD manifold used for tangent-space projection in Phase 3.
- Schaefer, A., Kong, R., Gordon, E. M., et al. (2018). Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI. *Cerebral Cortex*, 28(9), 3095–3114. https://doi.org/10.1093/cercor/bhx179
- Rolls, E. T., Huang, C.-C., Lin, C.-P., Feng, J., & Joliot, M. (2020). Automated anatomical labelling atlas 3 (AAL3). *NeuroImage*, 206, 116189. https://doi.org/10.1016/j.neuroimage.2019.116189

## Earlier versions (background only)

- [`codes/version_0/`](codes/version_0/) — the original shared pipeline (Phase 0A-3, 7), as first written.
- [`codes/version_1/`](codes/version_1/) — an exploratory 3-variable model adding a prediction-error
  signal $e$ and a Chein & Schneider (2012) Triarchic Theory interpretation
  (Meta/Con/Rep systems). **Not adopted** for the main analysis — see the appendix
  of [Pilot_Results_Public_v2.md](docs/Pilot_Results_Public_v2.md) for why. Kept for reference.

## Citation

This is a pilot-stage, not-yet-peer-reviewed analysis. If you use or build on this
code, please cite this repository:

```
Li, C.-Y. (2026). 2026_brainhack_li_project: Hybrid Dynamic Neural Decoding Pipeline
[Computer software]. https://github.com/lepus071/2026_brainhack_li_project
```

Please also cite the underlying dataset (OpenNeuro [ds005598](https://openneuro.org/datasets/ds005598))
and the key methodological references listed above (Smith et al. 2006; Pennec et al.
2006; Varoquaux et al. 2010; Schaefer et al. 2018; Rolls et al. 2020) as appropriate.

## License

This project is licensed under the [MIT License](LICENSE) — you are free to use,
modify, and distribute this code (including for commercial purposes), provided the
copyright notice and license text are retained. The data and figures referenced
in [Pilot_Results_Public_v2.md](docs/Pilot_Results_Public_v2.md) are derived from
OpenNeuro dataset [ds005598](https://openneuro.org/datasets/ds005598), which has its
own license terms on OpenNeuro.
