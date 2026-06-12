#!/usr/bin/env python3
# ==============================================================================
# check_e_components_shape_corr.py
#
# A6 follow-up: quantify how similar the per-network distribution "shape" is
# across the three e-related partial4 components (Unique_e, Shared_e_xf,
# Shared_e_xs), using Spearman correlation over the 8 network-bucket means
# (Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default, AAL3(other)).
# ==============================================================================

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJ_DIR = '/home/ser/2026_brainheck_li_project'
ML_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results'
ATLAS_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases'

NETWORKS = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']

COMPONENTS = {
    'Unique_e': 'Unique_e_partial4_R2',
    'Shared_e_xf': 'Shared_e_xf_partial4_R2',
    'Shared_e_xs': 'Shared_e_xs_partial4_R2',
}


def assign_network(label):
    if not isinstance(label, str):
        return 'AAL3 (other)'
    for net in NETWORKS:
        if f'_{net}_' in label or label.endswith(f'_{net}'):
            return net
    return 'AAL3 (other)'


def main():
    labels_df = pd.read_csv(f'{ATLAS_DIR}/Combined_Atlas_Labels.csv').set_index('Index')['Label']
    n_roi = 481
    roi_ids = np.arange(2, 2 + n_roi)
    nets = pd.Series([assign_network(labels_df.get(r, None)) for r in roi_ids], index=np.arange(n_roi))

    buckets = NETWORKS + ['AAL3 (other)']

    means = {}
    for name, fname in COMPONENTS.items():
        arr = np.load(f'{ML_DIR}/{fname}.npy')
        means[name] = [arr[nets[nets == net].index.values].mean() for net in buckets]

    df = pd.DataFrame(means, index=buckets)
    print("Per-network mean R2 (partial4):")
    print(df.to_string(float_format='%.4f'))

    print("\nSpearman correlation of network-distribution shapes:")
    names = list(COMPONENTS.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho, p = spearmanr(df[names[i]], df[names[j]])
            print(f"  {names[i]:<14} vs {names[j]:<14}: rho={rho:.3f}  p={p:.3f}")


if __name__ == '__main__':
    main()
