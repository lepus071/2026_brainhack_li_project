#!/usr/bin/env python3
# ==============================================================================
# check_network_breakdown.py
#
# Phase 6 follow-up: instead of treating "Limbic" as the only null network,
# break down mean R^2 (for each of the 7 VPA components, both original and
# gs-controlled partial4 versions) by ALL Schaefer 7-network labels (Vis,
# SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default) plus an "AAL3 (other)"
# bucket for non-Schaefer ROIs.
#
# This contextualizes whether the Unique_e / Shared_e_xf "Limbic warning" is
# something special about Limbic specifically, or just part of a broader
# pattern where e-related components have elevated R^2 across MANY networks
# (in which case "e signal also appears in Limbic" is less surprising / less
# diagnostic on its own).
# ==============================================================================

import re
import numpy as np
import pandas as pd

PROJ_DIR = '/home/ser/2026_brainheck_li_project'
ML_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/ml_results'
ATLAS_DIR = f'{PROJ_DIR}/data/derivatives/riemannian_decoding/atlases'

NETWORKS = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']

COMPONENT_PAIRS = [
    ('Unique_e_R2', 'Unique_e_partial4_R2'),
    ('Unique_xf_R2', 'Unique_xf_partial4_R2'),
    ('Unique_xs_R2', 'Unique_xs_partial4_R2'),
    ('Shared_e_xf_R2', 'Shared_e_xf_partial4_R2'),
    ('Shared_xf_xs_R2', 'Shared_xf_xs_partial4_R2'),
    ('Shared_e_xs_R2', 'Shared_e_xs_partial4_R2'),
    ('Shared_all_R2', 'Shared_all_partial4_R2'),
]


def assign_network(label):
    if not isinstance(label, str):
        return 'AAL3 (other)'
    for net in NETWORKS:
        if f'_{net}_' in label or label.endswith(f'_{net}'):
            return net
    if label.startswith('7Networks_'):
        return 'AAL3 (other)'
    return 'AAL3 (other)'


def main():
    labels_df = pd.read_csv(f'{ATLAS_DIR}/Combined_Atlas_Labels.csv').set_index('Index')['Label']
    n_roi = 481
    roi_ids = np.arange(2, 2 + n_roi)
    nets = pd.Series([assign_network(labels_df.get(r, None)) for r in roi_ids], index=np.arange(n_roi))

    print("ROI counts per network bucket:")
    print(nets.value_counts().to_string())

    for orig_name, partial_name in COMPONENT_PAIRS:
        orig = np.load(f'{ML_DIR}/{orig_name}.npy')
        try:
            partial = np.load(f'{ML_DIR}/{partial_name}.npy')
        except FileNotFoundError:
            partial = None

        print(f"\n=== {orig_name} ===")
        print(f"{'Network':<14} {'n_ROI':>6} {'orig_mean':>10} {'orig_max':>10}"
              + ("" if partial is None else f" {'partial4_mean':>14} {'partial4_max':>13}"))
        whole_orig = orig.mean()
        whole_partial = partial.mean() if partial is not None else None
        for net in NETWORKS + ['AAL3 (other)']:
            idx = nets[nets == net].index.values
            if len(idx) == 0:
                continue
            o_mean, o_max = orig[idx].mean(), orig[idx].max()
            line = f"{net:<14} {len(idx):>6} {o_mean:>10.4f} {o_max:>10.4f}"
            if partial is not None:
                p_mean, p_max = partial[idx].mean(), partial[idx].max()
                line += f" {p_mean:>14.4f} {p_max:>13.4f}"
            print(line)
        line = f"{'WHOLE BRAIN':<14} {n_roi:>6} {whole_orig:>10.4f} {orig.max():>10.4f}"
        if partial is not None:
            line += f" {whole_partial:>14.4f} {partial.max():>13.4f}"
        print(line)


if __name__ == '__main__':
    main()
