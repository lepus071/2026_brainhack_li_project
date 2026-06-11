import os
import numpy as np
import pandas as pd

PROJ_DIR = '/home/ser/2026_brainheck_li_project'
ATLAS_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases'
ML_DIR    = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results'

def main():
    label_csv = f"{ATLAS_DIR}/Combined_Atlas_Labels.csv"
    label_df = pd.read_csv(label_csv)
    # The first 2 rows are Background (0) and M1_Seed (1)
    # The features in .npy files (length 481) correspond to labels from index 2 onwards
    labels = label_df['Label'].values[2:]
    
    targets = ['Unique_e', 'Unique_xf', 'Unique_xs']
    
    for t in targets:
        npy_file = f"{ML_DIR}/{t}_R2.npy"
        if not os.path.exists(npy_file):
            print(f"File not found: {npy_file}")
            continue
            
        w = np.load(npy_file)
        # Get top 10 indices
        top_indices = np.argsort(w)[-10:][::-1]
        
        print(f"\n======================================")
        print(f"Top 10 regions for {t}:")
        print(f"======================================")
        for rank, idx in enumerate(top_indices):
            print(f"  {rank+1}. {labels[idx]} (R2 = {w[idx]:.4f})")

if __name__ == '__main__':
    main()
