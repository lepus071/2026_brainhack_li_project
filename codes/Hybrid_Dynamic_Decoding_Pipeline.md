# Hybrid Dynamic Neural Decoding Pipeline

## Project Overview

The **Hybrid Dynamic Neural Decoding Pipeline** is an fMRI analysis framework designed for visuomotor rotation (VMR) adaptation tasks. It bridges two complementary theories:

- **Chein & Schneider (2012) — Triarchic Theory of Learning**: the brain engages three parallel systems during skill acquisition: a metacognitive monitoring system, a cognitive control network (CCN), and an implicit representation system.
- **Smith et al. (2006) — Dual-Rate State-Space Model**: motor learning is driven by a fast-adapting but quickly-forgetting process ($x_f$) and a slow-learning but persistent process ($x_s$), each driven by sensory prediction error ($e$).

The pipeline operationalises these theories by tracking the *dynamic evolution* of brain network connectivity over the course of learning, and testing how much of that connectivity variance is uniquely explained by each of the three behavioral components.

**Key techniques:** fMRIPrep preprocessing · Individualized M1 seed localization · Riemannian manifold tangent-space projection · Ridge regression variance partitioning analysis (VPA) · SPM HRF convolution for cross-modal temporal alignment · AFNI group statistics with Delta as a between-subjects covariate.

**Dataset:** OpenNeuro [ds005598](https://openneuro.org/datasets/ds005598) — right-handed VMR task with a 45° cursor rotation block (no washout), preceded by a resting-state scan.

**Atlas:** 482 ROIs total — 1 individualized M1 + 399 Schaefer 400 cortical parcels + 82 AAL3v2 subcortical/cerebellar regions.

---

## Data Source & Preprocessing

### Public Dataset

- **Source:** OpenNeuro [ds005598](https://openneuro.org/datasets/ds005598)
- **Task paradigm:** Right-handed Visuomotor Rotation (VMR). Participants control a cursor/joystick with their right hand. The paradigm includes a resting-state baseline run followed by multiple blocks of 45° unidirectional rotation adaptation.
- **No washout phase** is present in this dataset; identifiability of the dual-rate model is enforced through strict parameter bounds.

---

## Phase 0A — fMRI Preprocessing

**Goal:** Produce clean, denoised BOLD time-series with a high signal-to-noise ratio, ready for functional connectivity analysis.

**Why this step?**
Raw BOLD signals are contaminated by head motion, scanner drift, cardiac/respiratory noise, and geometric distortions. All of these must be removed before any neural signal can be reliably extracted.

**Steps:**
1. **fMRIPrep** (Docker) performs: slice-timing correction, head-motion correction (6-DOF rigid-body realignment), functional–anatomical co-registration, and nonlinear normalization to MNI152 space.
2. **`fmriprep_to_afni_bridge.py`** converts fMRIPrep outputs to AFNI format:
   - Converts BOLD intensity to **Percent Signal Change (PSC)** — making signal amplitudes comparable across subjects and sessions.
   - Generates **censoring masks** (`censor_*.1D`): time points with framewise displacement > 0.5 mm are flagged and excluded from all subsequent analyses.
   - Generates **motion-demeaned regressors** (`mot_demean.r*.1D`) for confound regression in GLMs.

**Scripts:** `run_fmriprep_batch.sh`, `fmriprep_to_afni_bridge.py`

---

## Phase 0B — Behavioral Modeling: Dual-Rate State-Space Fitting

**Goal:** Decompose each subject's trial-by-trial error trajectory into three theoretically grounded time-series that will later serve as neural predictors.

**Why this step?**
Raw angular error values are noisy and do not isolate the contributions of distinct learning systems. The dual-rate state-space model provides a principled decomposition into:

| Variable | Definition | Neuroscientific mapping |
|:---------|:-----------|:------------------------|
| $e(t)$ | Sensory prediction error at trial $t$ | **Metacognitive system** — monitors ongoing performance discrepancy |
| $x_f(t)$ | Fast-process internal state | **Cognitive Control Network (CCN)** — rapid, effortful strategy adjustment |
| $x_s(t)$ | Slow-process internal state | **Implicit representation** — gradual consolidation in cerebellum / basal ganglia |

**Model equations:**
$$x_f(t+1) = A_f \cdot x_f(t) + B_f \cdot e(t)$$
$$x_s(t+1) = A_s \cdot x_s(t) + B_s \cdot e(t)$$
$$e(t) = r(t) - \bigl(x_f(t) + x_s(t)\bigr)$$

where $r(t) = 45°$ (rotation magnitude) and parameters satisfy $A_f < A_s$, $B_f > B_s$, ensuring the fast process learns quickly but forgets quickly, while the slow process accumulates stably.

**Steps:**
1. Load raw behavioral CSV; clean invalid trials (timeouts, NaNs).
2. Fit four parameters $(A_f, B_f, A_s, B_s)$ **per subject** using L-BFGS-B optimization with physiologically motivated bounds.
3. Export three dynamic feature time-series ($e$, $x_f$, $x_s$) per subject as CSV files.

> **Note on identifiability:** Because this dataset has no washout phase, the four parameters are somewhat collinear. Strict bounds + Ridge regression regularization in Phase 4 absorb residual collinearity.

**Script:** `run_hybrid_behavioral_fitting.py`

---

## Phase 1 — Individualized M1 Seed Localization

**Goal:** Identify each subject's true primary motor cortex (M1) peak voxel rather than relying on a population-average atlas coordinate.

**Why this step?**
Standard atlases (e.g., Schaefer, AAL3) are built from group-average anatomy. Individual variability in sulcal depth and gyral folding can shift functional M1 by 8–20 mm relative to atlas coordinates — far beyond the 6 mm seed radius we use. A mislocalized seed corrupts all downstream connectivity measures.

**Steps:**
1. Run a **condition GLM** (`3dDeconvolve`) on the preprocessed BOLD using `BLOCK(4.5,1)` convolution (matching the 4.5 s motor execution window) and conservative `-polort 2` (preserves slow learning-related BOLD trends while removing scanner drift).
2. Search within a **left precentral gyrus anatomical mask** (contralateral to right-hand use) for the peak T-value voxel.
3. **Fallback:** if no significant activation is found, use the neuroanatomical prior MNI $[-38, -22, 56]$ (Hand Knob region).
4. Build a **6 mm radius sphere** (~200 voxels) around the peak — balancing spatial specificity against localization error tolerance.
5. **Group-level sanity check:** run `3dttest++` on all subjects' GLM maps to confirm consistent M1 activation across the group.

**Scripts:** `run_afni_condition_glm.py`, `run_group_M1_GLM.tcsh`

---

## Phase 2 — Multi-Atlas Node Construction (482 ROIs)

**Goal:** Define a whole-brain parcellation that covers cortex, subcortex, and cerebellum — all regions relevant to the motor learning circuit.

**Why this step?**
No single atlas covers all the structures implicated in visuomotor learning. Cortical atlases (e.g., Schaefer) offer fine-grained functional parcellation but omit subcortex and cerebellum; anatomical atlases (e.g., AAL3) cover subcortex comprehensively but have coarse cortical parcellations that would overlap redundantly with Schaefer. A hybrid approach is necessary — and critically, **AAL3 cortical parcels are excluded** to avoid collinearity with Schaefer nodes covering the same tissue.

**Node composition:**

| Atlas | Regions included | Count |
|:------|:----------------|:------|
| **Schaefer 400** | Whole cortex (functional parcellation based on resting-state networks); one SomMot node replaced by individualized M1 | 399 |
| **AAL3v2 — all subcortex & cerebellum** | Hippocampus, Amygdala, Caudate, Putamen, Pallidum, all Cerebellum lobules (Crus1/2, 3–10) + Vermis (1–10), all 15 Thalamic sub-nuclei (AV/LP/VA/VL/VPL/IL/Re/MD/LGN/MGN/Pu), ACC_sub, Nucleus Accumbens, VTA, Substantia Nigra (pc+pr), Red Nucleus, Locus Coeruleus, Raphe (Dorsal+Median) | 82 |
| **Individualized M1** (Phase 1 seed) | Subject-specific primary motor cortex (Index 1, overwrites Schaefer SomMot) | 1 |
| **Total** | | **482** |

**Why include the full AAL3 subcortex (including hippocampus, amygdala, brainstem nuclei)?**
Although hippocampus and amygdala are not classical motor learning structures, including them allows the data to determine whether they show any systematic dynamic coupling with M1 during VMR adaptation. If they do not (as theory predicts), their near-zero VPA $R^2$ values strengthen the spatial specificity argument. If they do show signal (e.g., hippocampus co-activating with $x_s$ during implicit consolidation), that is a scientifically interesting finding. Excluding them a priori would prevent this discovery.

The M1 seed is inserted as **Index 1** — a separate extra node — so its seed-specific connectivity vector can be cleanly extracted in Phase 3. Spatial overlap with Schaefer motor nodes is tolerated because Ridge regularization distributes weights stably across correlated predictors.

**Script:** `build_combined_atlas.py`

**Script:** `build_combined_atlas.py`

---

## Phase 3 — Riemannian Manifold Projection & Dynamic FC Extraction

**Goal:** Compute dynamic functional connectivity (FC) matrices across sliding windows of the task, removing each subject's idiosyncratic baseline "neural fingerprint" so that group-level comparisons reflect only task-driven network changes.

**Why this step?**
Even at rest, every individual has a unique FC pattern (their "neural fingerprint"). If we compute task-state FC directly and compare across subjects, machine learning models may learn *who* the subject is rather than *how* they are learning. The Riemannian projection removes this baseline effect mathematically.

**Steps:**

1. **Resting-state reference matrix:**
   Compute each subject's resting-state covariance matrix $\Sigma_{rest}^{(i)}$ from the pre-task resting-state BOLD (long $T$, well-conditioned SPD matrix). This serves as the subject-specific *geometric origin* on the Riemannian manifold of symmetric positive definite (SPD) matrices.

2. **Sliding-window task-state FC:**
   For each 30-TR window (60 s, step = 1 TR) during the task, estimate $\Sigma_{task,w}^{(i)}$ using **Ledoit-Wolf shrinkage** to enforce positive definiteness (since window length $T = 30 \ll N = 465$).

3. **Tangent-space projection:**
   Project each task-state window onto the tangent plane anchored at $\Sigma_{rest}^{(i)}$:
   $$S_w^{(i)} = \log_m\!\left(\Sigma_{rest}^{(i)^{-1/2}} \cdot \Sigma_{task,w}^{(i)} \cdot \Sigma_{rest}^{(i)^{-1/2}}\right)$$
   The resulting tangent vector $S_w^{(i)}$ lives in Euclidean space and represents the *deviation from rest*, not the absolute connectivity state. Vectors from different subjects are now geometrically comparable.

4. **Seed-based feature extraction:**
   Retain only the row/column of $S_w^{(i)}$ corresponding to the M1 seed (Index 1), yielding a **464-dimensional connectivity vector** per window per subject.

**Window size trade-off:** 30 TRs provides enough observations to stabilize the 465×465 covariance estimate while preserving sufficient temporal resolution to track behavioral dynamics (each window spans ~5–6 trials, or ~10 trials including HRF tail effects).

**Script:** `run_hybrid_riemannian_extraction.py`

---

## Phase 4 — HRF Convolution Alignment + Joint Encoding + Variance Partitioning

**Goal:** Determine how much of the dynamic M1-to-whole-brain connectivity variance is *uniquely* explained by each behavioral component ($e$, $x_f$, $x_s$), after rigorously separating their shared contributions.

### 4A — HRF Temporal Alignment (Cross-modal Synchronization)

**Why this step?**
Behavioral parameters are recorded at trial resolution (~every 10.5 s), while BOLD is sampled every TR (2 s). Crucially, the BOLD signal lags neural activity by 4–6 s due to neurovascular coupling (the hemodynamic response function, HRF). Naively shifting the behavioral series by a fixed delay ignores the *temporal spread* of the HRF (~30 s total duration).

**Solution — SPM canonical HRF convolution (Scheme A):**
1. Build an impulse train at TR resolution: inject each trial's behavioral value ($e_t$, $x_{f,t}$, or $x_{s,t}$) at the corresponding onset time.
2. Convolve the impulse train with the SPM double-gamma HRF:
$$h(t) = \frac{t^{a_1-1}e^{-t/b_1}}{b_1^{a_1}\Gamma(a_1)} - c\cdot\frac{t^{a_2-1}e^{-t/b_2}}{b_2^{a_2}\Gamma(a_2)}$$
   $(a_1=6,\, b_1=1,\, a_2=16,\, b_2=1,\, c=1/6)$
3. The convolved series $\tilde{B}(t)$ now lives in BOLD space with the same TR resolution and temporal spread as the measured BOLD signal.
4. Average $\tilde{B}(t)$ within each 30-TR sliding window to match the temporal resolution of the Phase 3 FC features.

### 4B — Joint Encoding Model & Variance Partitioning Analysis (VPA)

**Why this step?**
$e$, $x_f$, and $x_s$ are mathematically coupled (generated by the same dynamical system), so simple univariate models cannot separate their neural contributions. VPA uses set-theoretic subtraction to isolate the *unique* variance explained by each predictor, net of all others.

**Seven Ridge regression models** (RidgeCV, Leave-One-Subject-Out cross-validation):

| Model | Predictors | $R^2$ estimated |
|:------|:----------|:---------------|
| Full | $e,\, x_f,\, x_s$ | $R^2_{full}$ |
| Leave-$e$-out | $x_f,\, x_s$ | $R^2_{-e}$ |
| Leave-$x_f$-out | $e,\, x_s$ | $R^2_{-f}$ |
| Leave-$x_s$-out | $e,\, x_f$ | $R^2_{-s}$ |
| $e$ only | $e$ | $R^2_e$ |
| $x_f$ only | $x_f$ | $R^2_f$ |
| $x_s$ only | $x_s$ | $R^2_s$ |

**Seven VPA components** (via inclusion-exclusion principle):

| Component | Formula | Interpretation |
|:----------|:--------|:--------------|
| Unique $e$ | $R^2_{full} - R^2_{-e}$ | Variance uniquely explained by metacognitive error monitoring |
| Unique $x_f$ | $R^2_{full} - R^2_{-f}$ | Variance uniquely explained by cognitive control |
| Unique $x_s$ | $R^2_{full} - R^2_{-s}$ | Variance uniquely explained by implicit consolidation |
| Shared $e \cap x_f$ | Derived via inclusion-exclusion | Shared between metacognition & CCN |
| Shared $x_f \cap x_s$ | Derived via inclusion-exclusion | Shared between CCN & implicit memory |
| Shared $e \cap x_s$ | Derived via inclusion-exclusion | Shared between metacognition & implicit memory |
| Shared $e \cap x_f \cap x_s$ | Derived via inclusion-exclusion | Globally shared across all three systems |

Each of the 464 ROI–M1 connectivity edges receives a separate set of 7 $R^2$ values. Negative values are mathematically expected under high collinearity and are reported as-is (or floored to 0 for visualization).

**Script:** `run_hybrid_group_decoding.py`

---

## Phase 5 — Inverse VPA Mapping (Back-projection to Brain Space)

**Goal:** Convert the 464-dimensional VPA vectors into whole-brain 3D NIfTI statistical maps, enabling spatial visualization and group-level voxelwise statistics.

**Steps:**
1. For each subject and each of the 7 VPA components, fill a blank MNI-space NIfTI volume: each ROI's voxels receive the ROI's $R^2$ value; non-ROI voxels receive 0.
2. Output: **7 NIfTI maps × N subjects** — each map encodes how much of a given ROI's dynamic connectivity with M1 is explained by a specific behavioral component.

**Script:** `run_inverse_vpa_mapping.py`

---

## Phase 6 — Null Network Validation (Negative Control)

**Goal:** Verify that the VPA $R^2$ values found in motor/cognitive regions are driven by genuine neural signal, not algorithmic overfitting to noise.

**Why this step?**
Ridge regression operates in 464 dimensions. With enough features, even random noise can yield non-zero out-of-sample $R^2$. A principled negative control is mandatory.

**Logic:**
The VMR task has **no auditory stimulation whatsoever** (purely visual feedback + hand movement). Therefore, the auditory cortex (Heschl's Gyrus / primary auditory cortex) should have *zero* systematic dynamic connectivity with M1 that is predictable from $e$, $x_f$, or $x_s$.

**Steps:**
1. Repeat the full VPA pipeline (Phase 4) but replace the 464 ROI features with connectivity to **auditory cortex ROIs only**.
2. Compute all 7 VPA $R^2$ components for this null network.
3. **Expected result:** all 7 components approach zero and are not significantly different from zero at group level.
4. This confirms that the pipeline does not manufacture $R^2$ from noise, and that the significant results in motor/prefrontal/cerebellar regions reflect genuine task-specific neural-behavioral coupling.

**Script:** `run_null_network_validation.py`

---

## Phase 7 — Anatomical Validation & Group Statistics with Delta Covariate

**Goal:** (1) Confirm that the three Unique VPA maps fall in anatomically predicted regions. (2) Test whether total learning magnitude (Delta) moderates the strength of neural-behavioral coupling at the group level.

### Step 1 — Anatomical Prior Validation

The three Unique components carry specific spatial predictions derived from the neuroscience literature:

| Component | Predicted regions | Literature basis |
|:----------|:-----------------|:----------------|
| **Unique $e$ (metacognitive monitoring)** | Anterior PFC (vmPFC, anterior dlPFC) | Ridderinkhof et al. (2004): aPFC monitors error and adjusts strategy |
| **Unique $x_f$ (cognitive control)** | Posterior dlPFC, ACC | Anguera et al. (2010): ACC/dlPFC active early in motor adaptation, then diminishes |
| **Unique $x_s$ (implicit consolidation)** | Cerebellum, basal ganglia (putamen/caudate) | Taylor & Ivry (2011); Shmuelof et al. (2012): implicit adaptation is cerebellar; procedural consolidation involves BG |

If all three Unique maps show statistically significant clusters in their predicted anatomical locations, this constitutes a **triple anatomical dissociation**, providing strong spatial validation of the theoretical framework.

### Step 2 — Delta as a Group-Level Covariate

**What is Delta?**

$$\Delta^{(i)} = \bar{e}_{onset}^{(i)} - \bar{e}_{plateau}^{(i)} = \text{mean}\!\left(e_{1:5}\right) - \text{mean}\!\left(e_{T-4:T}\right)$$

Delta is a **static scalar** (one value per subject) representing the total behavioral adaptation magnitude — how much error the subject reduced from the start to the plateau of the rotation block. It is computed directly from the `Actual_Performance` column of the state-space model output CSV.

**Why Delta belongs here (Phase 7) — not in Phase 4:**

| Property | Phase 4 predictors ($e_t$, $x_{f,t}$, $x_{s,t}$) | Delta |
|:---------|:----------------------------------------------|:------|
| Temporal nature | Time-series — different value every window | Static scalar — constant across all windows |
| Information content | *How* the subject learned (process dynamics) | *How much* the subject learned (behavioral outcome) |
| If added to Phase 4 regression | Contributes temporal variance | Acts as a constant column — adds only an intercept shift |
| Relationship with $x_s$ | Independent time-series | $\Delta \approx x_s(t_{plateau})$ — highly collinear |

Adding Delta to Phase 4 would contribute no temporal dynamics and would inflate collinearity with $x_s$.

**Three-model hierarchical strategy (`run_group_statistics.py`):**

| Model | AFNI command | Scientific question |
|:------|:------------|:-------------------|
| **Model A** — Group mean | `3dttest++ -setA VPA_*.nii.gz` | Is group-average VPA $R^2 > 0$? |
| **Model B** — Delta moderator | `3dttest++ -covariates [Subj,Delta] -setA VPA_*.nii.gz` | Do subjects who adapted more show stronger neural coupling? |
| **Model C** — Full covariates | `3dttest++ -covariates [Subj,Delta,LearnRate,Plateau_Var] -setA VPA_*.nii.gz` | What is the *independent* contribution of total adaptation magnitude, controlling for learning speed and stability? |

**Most scientifically informative Delta × VPA combinations:**

- **Delta × Unique $x_f$**: Does greater CCN engagement predict how much a subject ultimately learns? (CCN drives early rapid adaptation; its strength may determine whether the slow system has enough error signal to consolidate fully.)
- **Delta × Unique $x_s$**: Does stronger implicit consolidation coupling predict greater total adaptation? (More efficient cerebellar/BG encoding → larger eventual behavioral change.)

### Step 3 — Group-Level Statistical Maps

Run AFNI `3dttest++` for all three model variants across all 7 VPA components. Apply multiple-comparison correction (GRF theory or FDR). Output whole-brain statistical maps identifying regions where VPA $R^2$ is significantly non-zero and/or modulated by Delta across subjects.

**Scripts:** `run_group_statistics.py`

---

## Phase 8A — Interactive 3D Visualization (HTML)

**Goal:** Generate shareable, interactive glass-brain HTML files for exploring group-level VPA maps without requiring specialist software.

**Steps:**
1. Read group-level VPA NIfTI maps from Phase 5–7.
2. Use Nilearn's `plot_glass_brain` in interactive HTML mode to render each of the 7 VPA components.
3. Apply color palettes matched to variance component identity (Unique $e$ = yellow; Unique $x_f$ = red; Unique $x_s$ = blue; pairwise shared = mixed colors; global shared = white).

**Script:** `run_interactive_html.py`

---

## Phase 8B — High-Resolution Surface Rendering (SurfIce)

**Goal:** Produce publication-quality brain surface renderings of the VPA statistical maps.

**Steps:**
1. Scan group VPA NIfTI maps.
2. Auto-generate a **SurfIce script** (`.gls`) that loads each map, applies thresholds and colormaps, and saves PNG renders to disk.
3. Outputs are suitable for direct inclusion in manuscripts and posters.

**Script:** `run_surfice_render.py`

---

## Script Reference Table

| Order | Script | Phase | Function & Tools |
|:------|:-------|:------|:----------------|
| **0** | `run_fmriprep_batch.sh`<br>`fmriprep_to_afni_bridge.py` | 0A | fMRIPrep (Docker), PSC normalization, censor mask generation |
| **1** | `run_hybrid_behavioral_fitting.py` | 0B | L-BFGS-B optimization (SciPy), Smith dual-rate state-space model fitting |
| **2** | `run_afni_condition_glm.py`<br>`run_group_M1_GLM.tcsh` | 1 | AFNI `3dDeconvolve` for individual M1 localization; `3dttest++` group sanity check |
| **3** | `build_combined_atlas.py` | 2 | Nilearn atlas handling, Schaefer 400 + AAL3 ROI fusion → 465 nodes |
| **4** | `run_hybrid_riemannian_extraction.py` | 3 | PyRiemann / sklearn `LedoitWolf`, tangent-space projection, sliding-window FC |
| **5** | `run_hybrid_group_decoding.py` | 4 | SPM HRF convolution, scikit-learn `RidgeCV`, LOSO-CV, VPA computation |
| **6** | `run_inverse_vpa_mapping.py` | 5 | Nibabel, inverse $R^2$ back-projection to NIfTI space |
| **7** | `run_null_network_validation.py` | 6 | Auditory cortex null-network VPA, specificity verification |
| **8** | `run_group_statistics.py` | 7 | Delta extraction from `Actual_Performance`, behavioral covariates file, three-model `3dttest++` command generation |
| **9A** | `run_interactive_html.py` | 8A | Nilearn interactive glass-brain 3D HTML, VPA color mapping |
| **9B** | `run_surfice_render.py` | 8B | SurfIce `.gls` macro generation, batch high-resolution PNG output |
