# Dual-Rate 2-Variable VPA Pipeline (Smith et al. 2006)

主分析方向：以 Smith et al. (2006) dual-rate state-space model 的兩個狀態變量
(x_f, x_s) 作為唯一行為預測變量，做 3-component VPA（Unique_xf, Unique_xs,
Shared_xf_xs）+ Full_xf_xs。不含誤差訊號 e / Chein & Schneider 三系統
（該嘗試視為失敗的方向，完整保留在 `../version_1/`，僅供參考）。

本資料夾現為**完整獨立 pipeline**：Phase 0A-3, 7 + 工具腳本已從 `codes/` 上層
複製一份至此（與上層共享相同內容，依 shared pipeline 命名慣例），不依賴
`../version_1/` 或上層任何檔案即可從頭跑到 Phase 7。

## 檔案 (依 Phase 順序)

- **Phase 0A**：`run_fmriprep_batch.sh`, `fmriprep_to_afni_bridge.py`
- **Phase 0B**：`run_hybrid_behavioral_fitting.py`
- **Phase 1**：`run_afni_condition_glm.py`, `run_group_M1_GLM.tcsh`
- **Phase 2**：`build_combined_atlas.py`
- **Phase 3**：`run_hybrid_riemannian_extraction.py`, `fix_roi_labels.py`
  （`fix_roi_labels.py` 依賴與 Phase 3 相同的 `r01_scaled.nii.gz` + atlas，
  產生 `ml_results/ROI_Labels_481.csv`，供 Phase 4 後續分析腳本對照具名 ROI）
- **`run_hybrid_group_decoding.py`** (Phase 4)：LOSO + GroupKFold RidgeCV，
  輸出 group-level 與個體層級 `.npy` 至 `data/derivatives/riemannian_decoding/ml_results_v2/`：
  - `Unique_xf_R2.npy`, `Unique_xs_R2.npy`, `Shared_xf_xs_R2.npy`, `Full_xf_xs_R2.npy`
  - 個體版：`<component>_R2_sub-XX.npy`
- **`run_inverse_vpa_mapping.py`** (Phase 5)：將 `ml_results_v2/*_R2*.npy` 映射回
  3D NIfTI，輸出至 `data/derivatives/riemannian_decoding/nifti_axes_v2/`。
- **`run_null_network_validation.py`** (Phase 6)：Limbic null-network 驗證。
- **`run_marginal_r2_analysis.py`**、**`run_between_within_decomposition.py`**、
  **`run_between_subject_permutation.py`**：Phase 4 的後續分析，僅針對
  x_f/x_s（marginal R²、between/within 分解、permutation test），輸出至
  `ml_results_v2/`。ROI 具名標籤讀取
  `ml_results/ROI_Labels_481.csv`（由 `fix_roi_labels.py` 產生，
  feature_col → label 的正確 481 維對照表）。
- **Phase 7**：`run_group_statistics.py`
- **`subjectlist.txt`**, **`license.txt`**：共用設定/授權檔。

## 待補 (Phase 8A/8B)

互動式 HTML / SurfIce 渲染尚未做 v2 版本，可參考 `../version_1/run_interactive_html.py`
與 `run_nilearn_render.py`，把輸入路徑改成 `nifti_axes_v2/` 即可。

## 執行狀態：已正式跑完 ✅

Phase 4-6 已於正式 pipeline 重跑完成：
- Phase 4 → `data/derivatives/riemannian_decoding/ml_results_v2/`：4 個 group-level
  `.npy` + 40 個個體層級 `.npy`（10 subjects × 4 components）。
- Phase 5 → `data/derivatives/riemannian_decoding/nifti_axes_v2/`：44 個
  `VPA_Map_*.nii.gz`。
- Phase 6：**4/4 PASS**，Limbic network (26 ROIs) 在 `Unique_xf`, `Unique_xs`,
  `Shared_xf_xs`, `Full_xf_xs` 的 mean/max R² 全部為 0.0000。

## `results/`（探索階段參考值，供與正式 group-level 結果核對）

- `Unique_xf_2v_R2.npy` (mean=0.0064, max=0.4664)
- `Unique_xs_2v_R2.npy` (mean=0.0020, max=0.1066)
- `Shared_xf_xs_2v_R2.npy` (mean=0.0054, max=0.2004)
- `Full_xf_xs_R2.npy` (mean=0.0073, max=0.4697)

正式版輸出於 `ml_results_v2/`，命名不帶 `_2v`，同一份資料、同一個模型，數值應一致。

## 解剖定位摘要 (top hits, 來自探索階段)

- `Unique_xf`: SomMot_33 (0.4664), SomMot_26, Cont_PFCl_4, SomMot_7, DorsAttn_Post_15
- `Unique_xs`: Vis_9 (0.1066), SomMot_35, SomMot_7, AAL3_SN_pr_L, Vis_21
- `Shared_xf_xs`: SomMot_24 (0.2004), Vis_30, Default_PFC_20, Cerebellum_4_5_L

## 後續分析結果 (marginal R² / between-within / permutation)

- **Marginal R² (raw)**：x_f mean=0.0082, max=0.4361 (28/481 ROIs)；
  x_s mean=0.0034, max=0.2223 (22/481 ROIs)。detrend 後皆為 0。
- **Between-subject (N=10, df=8)**：x_f top |r|=0.877（ROI#63，jackknife 範圍
  [-0.894,-0.858]，sign stable）；x_s top |r|=0.819（ROI#208，jackknife 範圍
  [+0.732,+0.882]，sign stable）。
- **Within-subject**：x_f mean=0.0017, max=0.0516 (98/481 ROIs)；
  x_s mean=0.0022, max=0.0421 (99/481 ROIs)。
- **Permutation test (N_perm=10000)**：x_f p=0.2914，x_s p=0.7582 —— 兩者皆
  **未通過多重比較校正後的顯著性**，與 version_1 結論一致。
- ROI#63/#208 等具名結果待重跑（已改用 `ROI_Labels_481.csv` 對照表）後補上。

## 為何不含 e

`../version_1/` 完整保留了曾經嘗試的 3-var (e, x_f, x_s) + Chein & Schneider
三系統路線。捨棄原因見 `Pilot_Report_N11.md` 附錄 A：(1) e 與 x_f 數學上嚴重共線，
(2) Meta 系統在 Schaefer atlas 下與 Con 系統解剖重疊、無法獨立驗證。
