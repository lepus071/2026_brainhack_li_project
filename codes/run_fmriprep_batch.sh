#!/bin/bash

# ==============================================================================
# run_fmriprep_batch.sh
#
# fMRIPrep Docker 批次前處理腳本 (符合 BIDS 格式與標準論文重現性)
# 
# 參數說明：
# - 版本: nipreps/fmriprep:20.2.7 (LTS 穩定版)
# - 空間: MNI152NLin6Asym 與 MNI152NLin2009cAsym
# - 去噪: --use-aroma (自動執行 6mm 平滑與 ICA-AROMA 運動雜訊移除)
# - 表面: 預設執行 FreeSurfer recon-all
# ==============================================================================

# ── 1. 路徑與授權設定 (請根據你的電腦實際狀況修改) ──────────────────────

# BIDS 資料夾的根目錄
BIDS_DIR="/home/ser/2026_brainheck_li_project/data"

# fMRIPrep 輸出的目標資料夾
DERIVS_DIR="${BIDS_DIR}/derivatives"

# FreeSurfer License 檔案的絕對路徑 (⚠️ 請確認這行路徑正確！)
# 如果你還沒申請，請至 https://surfer.nmr.mgh.harvard.edu/registration.html 申請
FS_LICENSE="/home/ser/2026_brainheck_li_project/codes/license.txt"

# 受試者名單檔案
SUBJ_LIST="/home/ser/2026_brainheck_li_project/codes/subjectlist.txt"

# 電腦資源分配
NTHREADS=8       # 分配給 fMRIPrep 的 CPU 核心數
MEM_MB=16384     # 分配給 Docker 的記憶體 (以 MB 計算，這裡設為 16GB)

# ─────────────────────────────────────────────────────────────────────────────

# 檢查檔案與資料夾是否存在
if [ ! -d "${BIDS_DIR}" ]; then
    echo "❌ 找不到 BIDS 資料夾: ${BIDS_DIR}"
    exit 1
fi

if [ ! -f "${FS_LICENSE}" ]; then
    echo "❌ 找不到 FreeSurfer License 檔案: ${FS_LICENSE}"
    echo "請確認檔案存在，或去官網申請後放在該路徑下。"
    exit 1
fi

if [ ! -f "${SUBJ_LIST}" ]; then
    echo "❌ 找不到受試者名單: ${SUBJ_LIST}"
    exit 1
fi

mkdir -p "${DERIVS_DIR}"

# ── 2. 執行迴圈 ──────────────────────────────────────────────────────────────

echo "================================================="
echo "  開始啟動 fMRIPrep 批次處理"
echo "  輸出目錄: ${DERIVS_DIR}/fmriprep"
echo "================================================="

# 讀取 subjectlist.txt，逐行執行
# 支援寫 sub-01 或是 01 都可以
while read -r line || [[ -n "$line" ]]; do
    # 忽略空行
    if [[ -z "$line" ]]; then continue; fi
    
    # 確保受試者 ID 去除 'sub-' 前綴 (fMRIPrep 要求的格式)
    SUBJ=${line#sub-}
    
    echo "▶ 正在處理受試者: sub-${SUBJ}"
    
    # 執行 Docker
    # 參數解釋：
    # -v: 將 WSL 的路徑掛載進 Docker 容器內
    # --use-aroma: 啟動 ICA-AROMA
    # --output-spaces: 輸出指定的標準腦空間
    # --stop-on-first-crash: 若報錯則立刻停止，方便除錯
    
    docker run -t --rm -v ${BIDS_DIR}:/data:ro -v ${DERIVS_DIR}:/out -v ${FS_LICENSE}:/opt/freesurfer/license.txt:ro nipreps/fmriprep:20.2.7 /data /out/fmriprep participant --participant-label ${SUBJ} --output-spaces MNI152NLin6Asym MNI152NLin2009cAsym --use-aroma --nthreads ${NTHREADS} --omp-nthreads ${NTHREADS} --mem_mb ${MEM_MB} --skip_bids_validation --stop-on-first-crash
        
    if [ $? -eq 0 ]; then
        echo "✅ sub-${SUBJ} 處理完成！"
    else
        echo "❌ sub-${SUBJ} 處理失敗，請檢查 Docker 報錯訊息。"
        # 可以選擇不退出，繼續跑下一個人，或者在這裡 exit 1
    fi
    echo "-------------------------------------------------"

done < "${SUBJ_LIST}"

echo "🎉 所有受試者的 fMRIPrep 批次處理皆已執行完畢！"
