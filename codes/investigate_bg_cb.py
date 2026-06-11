import os
import numpy as np
import pandas as pd

PROJ_DIR = '/home/ser/2026_brainheck_li_project'
ATLAS_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases'
ML_DIR    = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results'

def main():
    label_csv = f"{ATLAS_DIR}/Combined_Atlas_Labels.csv"
    label_df = pd.read_csv(label_csv)
    labels = label_df['Label'].values[2:]
    
    # Identify Basal Ganglia and Cerebellum indices
    bg_cb_indices = []
    for i, l in enumerate(labels):
        if any(keyword in l for keyword in ['Cerebellum', 'Caudate', 'Putamen', 'Pallidum', 'Vermis']):
            bg_cb_indices.append(i)
            
    vpa_parts = ["Unique_e", "Unique_xf", "Unique_xs", "Shared_e_xf", "Shared_xf_xs", "Shared_e_xs", "Shared_all"]
    
    # Store results
    results = {idx: {} for idx in bg_cb_indices}
    
    for part in vpa_parts:
        npy_file = f"{ML_DIR}/{part}_R2.npy"
        if os.path.exists(npy_file):
            w = np.load(npy_file)
            for idx in bg_cb_indices:
                results[idx][part] = w[idx]
                
    # Output top 15 regions of BG/CB by max R2 across all components
    max_r2_per_roi = {idx: max(results[idx].values()) for idx in bg_cb_indices}
    top_rois = sorted(bg_cb_indices, key=lambda idx: max_r2_per_roi[idx], reverse=True)[:15]
    
    print(f"Top 15 Basal Ganglia & Cerebellum regions and their Variance Partitioning (R2):")
    print("-" * 85)
    print(f"{'Region':<25} | {'Max_Part':<15} | {'Max R2':<7} | {'Unique_xs':<9} | {'Shared_xf_xs':<12}")
    print("-" * 85)
    
    for idx in top_rois:
        r2_vals = results[idx]
        max_part = max(r2_vals, key=r2_vals.get)
        max_r2 = r2_vals[max_part]
        uxs = r2_vals.get('Unique_xs', 0.0)
        sxs = r2_vals.get('Shared_xf_xs', 0.0)
        
        print(f"{labels[idx]:<25} | {max_part:<15} | {max_r2:.4f}  | {uxs:.4f}    | {sxs:.4f}")

if __name__ == '__main__':
    main()
