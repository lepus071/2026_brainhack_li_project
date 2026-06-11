import os
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json
from scipy.ndimage import gaussian_filter1d

# ==========================================
# 1. Parameter and path settings (Please check your CSV placement path)
# ==========================================
raw_data_path = r'/home/ser/2026_brainheck_li_project/data/derivatives/behavioral/day2_behavioral_data.csv' 
# Note that sub-* here is just a string template, we will dynamically replace it via .replace('sub-*', sub_id) in the loop to generate directories
output_base_dir_template = r'/home/ser/2026_brainheck_li_project/data/derivatives/behavioral_features/sub-*'

# [Smart Patch] Automatically fix cross-system issues
if os.name == 'nt':  # If it is a Windows system (PsychoPy)
    if raw_data_path.startswith('/'):
        raw_data_path = r'\\wsl.localhost\Ubuntu' + raw_data_path.replace('/', '\\')
    if output_base_dir_template.startswith('/'):
        output_base_dir_template = r'\\wsl.localhost\Ubuntu' + output_base_dir_template.replace('/', '\\')

# ==========================================
# 2. Core function: Independent fitting and normalization
# ==========================================
def fit_single_block(target_data, init_xf, init_xs):
    def cost_function(p):
        af, bf, as_v, bs = p
        if af >= as_v or bf <= bs:
            return 1e10

        xf, xs = init_xf, init_xs
        xf_arr, xs_arr = np.zeros(len(target_data)), np.zeros(len(target_data))
        xf_arr[0], xs_arr[0] = xf, xs

        for t in range(len(target_data)-1):
            err = target_data[t] - (xf + xs)
            xf = af * xf + bf * err
            xs = as_v * xs + bs * err
            xf_arr[t+1], xs_arr[t+1] = xf, xs

        return np.nanmean(((xf_arr + xs_arr) - target_data)**2)

    init_guess = [0.59, 0.21, 0.992, 0.02]
    res = minimize(cost_function, init_guess, bounds=[(0, 1)]*4, method='L-BFGS-B')
    return res.x

def global_normalize(signal):
    span = np.max(signal) - np.min(signal)
    return (signal - np.min(signal)) / (span if span > 0 else 1e-8)

# ==========================================
# 3. Process single subject (Fit -> Plot -> Save)
# ==========================================
def process_single_subject(sub_id, df_sub):
    # Dynamically generate a dedicated output folder for this subject to avoid WinError 123
    output_base_dir = output_base_dir_template.replace('sub-*', sub_id)
    out_data_dir = os.path.join(output_base_dir, 'CSV_Data')
    out_plot_dir = os.path.join(output_base_dir, 'Plots')
    out_fmri_dir = os.path.join(output_base_dir, 'fMRI_Regressors')
    
    os.makedirs(out_data_dir, exist_ok=True)
    os.makedirs(out_plot_dir, exist_ok=True)
    os.makedirs(out_fmri_dir, exist_ok=True)

    # Remove TimeOut (NaN) trials, and reset the index
    df_sub = df_sub.dropna(subset=['hitAngle_hand_good']).copy()
    if len(df_sub) == 0:
        return False, "No valid data"
        
    df_sub.reset_index(drop=True, inplace=True) 
    
    total_trials = len(df_sub)
    perf_actual = df_sub['hitAngle_hand_good'].values
    rot_target = df_sub['cursorRotation'].fillna(0).values
    
    # Get original trialNo for fMRI alignment
    original_trial_nos = df_sub['trialNo'].values if 'trialNo' in df_sub.columns else np.arange(1, total_trials + 1)
    block_nos = df_sub['blockNo'].values

    # Dynamically generate Block information
    blocks_info = []
    unique_blocks = df_sub['blockNo'].unique()
    for i, b_no in enumerate(unique_blocks):
        b_df = df_sub[df_sub['blockNo'] == b_no]
        blocks_info.append({
            'name': f'Block {b_no}',
            'start': b_df.index[0],
            'end': b_df.index[-1] + 1,
            'rest_before': i > 0
        })

    # --- A. Independent fitting block by block ---
    xf_raw, xs_raw, error_raw = np.zeros(total_trials), np.zeros(total_trials), np.zeros(total_trials)
    s_xf, s_xs = 0.0, 0.0

    for b in blocks_info:
        idx_s, idx_e = b['start'], b['end']
        if b['rest_before']: s_xf = 0.0

        target_b = perf_actual[idx_s:idx_e]
        af_opt, bf_opt, as_opt, bs_opt = fit_single_block(target_b, s_xf, s_xs)

        length = idx_e - idx_s
        xf_b, xs_b = np.zeros(length), np.zeros(length)
        xf_b[0], xs_b[0] = s_xf, s_xs

        for t in range(length - 1):
            err = rot_target[idx_s + t] - (xf_b[t] + xs_b[t])
            error_raw[idx_s + t] = err
            xf_b[t+1] = af_opt * xf_b[t] + bf_opt * err
            xs_b[t+1] = as_opt * xs_b[t] + bs_opt * err

        last_err = rot_target[idx_e - 1] - (xf_b[-1] + xs_b[-1])
        error_raw[idx_e - 1] = last_err

        xf_raw[idx_s:idx_e] = xf_b
        xs_raw[idx_s:idx_e] = xs_b

        s_xf = af_opt * xf_b[-1] + bf_opt * last_err
        s_xs = as_opt * xs_b[-1] + bs_opt * last_err
        
        if '3' in str(b['name']): # Block 3 is the main adaptation block
            block3_params = {'A_f': float(af_opt), 'B_f': float(bf_opt), 'A_s': float(as_opt), 'B_s': float(bs_opt)}

    # Save the parameters to JSON
    with open(os.path.join(out_data_dir, f'{sub_id}_Triarchic_Params.json'), 'w') as f:
        json.dump(block3_params, f, indent=4)

    # --- B. Smoothing and Normalization ---
    metacog_raw = np.abs(error_raw)
    ccn_raw = np.abs(xf_raw)
    rep_raw = np.abs(xs_raw)

    metacog_s, ccn_s, rep_s = np.zeros(total_trials), np.zeros(total_trials), np.zeros(total_trials)

    for b in blocks_info:
        idx = slice(b['start'], b['end'])
        metacog_s[idx] = gaussian_filter1d(metacog_raw[idx], sigma=3)
        ccn_s[idx] = gaussian_filter1d(ccn_raw[idx], sigma=3)
        rep_s[idx] = gaussian_filter1d(rep_raw[idx], sigma=3)

    metacog_final = global_normalize(metacog_s)
    ccn_final = global_normalize(ccn_s)
    rep_final = global_normalize(rep_s)

    metacog_raw_norm = global_normalize(metacog_raw)
    ccn_raw_norm = global_normalize(ccn_raw)
    rep_raw_norm = global_normalize(rep_raw)

    # --- C. Export complete experimental data CSV ---
    df_output = pd.DataFrame({
        'Original_TrialNo': original_trial_nos,
        'Block': block_nos,
        'Target_Rotation': rot_target,
        'Actual_Performance': perf_actual,
        'Raw_Error_e': error_raw,
        'Raw_Fast_xf': xf_raw,
        'Raw_Slow_xs': xs_raw,
        'Metacognitive_Engagement': metacog_final,
        'CCN_Engagement': ccn_final,
        'Representation_Engagement': rep_final
    })
    csv_path = os.path.join(out_data_dir, f'{sub_id}_Triarchic_Fitted.csv')
    df_output.to_csv(csv_path, index=False)

    # --- D. Export fMRI GLM specific regressor CSV ---
    df_fmri = pd.DataFrame({
        'Original_TrialNo': original_trial_nos,
        'Block': block_nos,
        'PM_Metacog_Raw_Unsmoothed': metacog_raw,
        'PM_CCN_Raw_Unsmoothed': ccn_raw,
        'PM_Rep_Raw_Unsmoothed': rep_raw,
        'PM_Metacog_Unsmoothed_Centered': metacog_raw - np.mean(metacog_raw),
        'PM_CCN_Unsmoothed_Centered': ccn_raw - np.mean(ccn_raw),
        'PM_Rep_Unsmoothed_Centered': rep_raw - np.mean(rep_raw),
        'PM_Metacog_Smoothed_Centered': metacog_final - np.mean(metacog_final),
        'PM_CCN_Smoothed_Centered': ccn_final - np.mean(ccn_final),
        'PM_Rep_Smoothed_Centered': rep_final - np.mean(rep_final)
    })
    fmri_csv_path = os.path.join(out_fmri_dir, f'{sub_id}_fMRI_Regressors.csv')
    df_fmri.to_csv(fmri_csv_path, index=False)

    # --- E. Plot and save: First plot (Smoothed version) ---
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, b in enumerate(blocks_info):
        idx = slice(b['start'], b['end'])
        t = np.arange(b['start'], b['end'])

        ax.plot(t, metacog_final[idx], color='#F39C12', linewidth=4, label='Metacognitive System (Detection $e$)' if i == 0 else None)
        ax.plot(t, ccn_final[idx], color='#E74C3C', linewidth=4, label='Cognitive Control Network (Strategy $x_f$)' if i == 0 else None)
        ax.plot(t, rep_final[idx], color='#3498DB', linewidth=4, label='Representation System (Slow $x_s$)' if i == 0 else None)
        if b['start'] > 0: ax.axvline(x=b['start'], color='gray', linestyle='--', alpha=0.5)

    ax.set_title(f'Triarchic Neural Engagement (Smoothed) - {sub_id} (NaN Removed)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Relative Neural Engagement', fontsize=12)
    ax.set_xlabel('Trial Sequence (Valid Trials Only)', fontsize=12)
    ax.set_yticks([])
    ax.legend(loc='upper left', framealpha=1, fontsize=6)

    plt.tight_layout()
    plot_path_smooth = os.path.join(out_plot_dir, f'{sub_id}_Triarchic_Plot.png')
    plt.savefig(plot_path_smooth, dpi=200, bbox_inches='tight')
    plt.close(fig) # Close canvas to release memory

    # --- F. Plot and save: Second plot (Unsmoothed Raw version) ---
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    for i, b in enumerate(blocks_info):
        idx = slice(b['start'], b['end'])
        t = np.arange(b['start'], b['end'])

        ax2.plot(t, metacog_raw_norm[idx], color='#F39C12', linewidth=1.5, alpha=0.8, label='Metacognitive System (Raw $|e|$)' if i == 0 else None)
        ax2.plot(t, ccn_raw_norm[idx], color='#E74C3C', linewidth=1.5, alpha=0.8, label='Cognitive Control Network (Raw $|x_f|$)' if i == 0 else None)
        ax2.plot(t, rep_raw_norm[idx], color='#3498DB', linewidth=1.5, alpha=0.8, label='Representation System (Raw $|x_s|$)' if i == 0 else None)
        if b['start'] > 0: ax2.axvline(x=b['start'], color='gray', linestyle='--', alpha=0.5)

    ax2.set_title(f'Triarchic Neural Engagement (Unsmoothed) - {sub_id} (NaN Removed)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Relative Neural Engagement (Raw)', fontsize=12)
    ax2.set_xlabel('Trial Sequence (Valid Trials Only)', fontsize=12)
    ax2.set_yticks([])
    ax2.legend(loc='upper left', framealpha=1, fontsize=6)

    plt.tight_layout()
    plot_path_unsmooth = os.path.join(out_plot_dir, f'{sub_id}_Triarchic_Plot_Unsmoothed.png')
    plt.savefig(plot_path_unsmooth, dpi=200, bbox_inches='tight')
    plt.close(fig2) # Close canvas to release memory
    # --- G. Plot and save: Third plot (Raw behavioral data) ---
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    for i, b in enumerate(blocks_info):
        idx = slice(b['start'], b['end'])
        t = np.arange(b['start'], b['end'])
        
        # Draw target angle and actual performance
        ax3.plot(t, rot_target[idx], color='black', linestyle='--', linewidth=2, label='Target Rotation' if i == 0 else None)
        ax3.plot(t, perf_actual[idx], color='purple', marker='o', markersize=4, linestyle='-', linewidth=1.5, alpha=0.7, label='Actual Performance' if i == 0 else None)
        
        if b['start'] > 0: ax3.axvline(x=b['start'], color='gray', linestyle='--', alpha=0.5)

    ax3.set_title(f'Behavioral Performance - {sub_id} (NaN Removed)', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Angle (degrees)', fontsize=12)
    ax3.set_xlabel('Trial Sequence (Valid Trials Only)', fontsize=12)
    ax3.legend(loc='upper right', framealpha=1, fontsize=10)

    plt.tight_layout()
    plot_path_behav = os.path.join(out_plot_dir, f'{sub_id}_Behavioral_Plot.png')
    plt.savefig(plot_path_behav, dpi=200, bbox_inches='tight')
    plt.close(fig3)

    # --- H. Export AM2 fMRI Onset file ---
    # Prepare for run_glm_3dDeconvolve_Triarchic.tcsh
    out_onset_dir = os.path.join(output_base_dir, 'Onset_Times')
    os.makedirs(out_onset_dir, exist_ok=True)
    
    BUFFER_SEC = 16.0
    PREP_SEC = 0.5
    TRIAL_PERIOD = 4.5 + 1.5
    def get_motor_onset(k): return BUFFER_SEC + PREP_SEC + k * TRIAL_PERIOD

    block_to_run = {1: 1, 2: 2, 3: 3, 6: 4}
    n_runs = 4
    
    am2_events = {
        'Rot0_Right_AM2': {r: [] for r in range(1, n_runs + 1)},
        'Rot45_Right_Metacog': {r: [] for r in range(1, n_runs + 1)},
        'Rot45_Right_CCN': {r: [] for r in range(1, n_runs + 1)},
        'Rot45_Right_Rep': {r: [] for r in range(1, n_runs + 1)}
    }
    
    for b_no in [2, 3]:
        if b_no not in block_to_run: continue
        run_idx = block_to_run[b_no]
        b_df = df_sub[df_sub['blockNo'] == b_no].reset_index(drop=True)
        b_fmri = df_fmri[df_fmri['Block'] == b_no].reset_index(drop=True)
        
        for k in range(len(b_df)):
            if str(b_df.loc[k, 'hand']).strip().upper() != 'RIGHT':
                continue
            t = get_motor_onset(k)
            # Use smoothed values for Parametric Modulation
            m_c = b_fmri.loc[k, 'PM_Metacog_Smoothed_Centered']
            c_c = b_fmri.loc[k, 'PM_CCN_Smoothed_Centered']
            r_c = b_fmri.loc[k, 'PM_Rep_Smoothed_Centered']
            
            if b_no == 2:
                am2_events['Rot0_Right_AM2'][run_idx].append(f'{t:.3f}*{m_c:.6g},{c_c:.6g},{r_c:.6g}')
            else:
                am2_events['Rot45_Right_Metacog'][run_idx].append(f'{t:.3f}*{m_c:.6g}')
                am2_events['Rot45_Right_CCN'][run_idx].append(f'{t:.3f}*{c_c:.6g}')
                am2_events['Rot45_Right_Rep'][run_idx].append(f'{t:.3f}*{r_c:.6g}')
            
    for cond_name, run_dict in am2_events.items():
        rows = []
        for r in range(1, n_runs + 1):
            if run_dict[r]:
                run_dict[r].sort(key=lambda x: float(x.split('*')[0]))
                rows.append(' '.join(run_dict[r]))
            else:
                rows.append('*')
        with open(os.path.join(out_onset_dir, f'{cond_name}.1D'), 'w') as f:
            f.write('\n'.join(rows) + '\n')

    return True, f"Successfully processed {total_trials} trials"

# ==========================================
# 4. Main program: Batch execute all subjects
# ==========================================
def batch_process_all_subjects():
    print("Batch processing started...")
    try:
        df_all = pd.read_csv(raw_data_path)
    except Exception as e:
        print(f"Failed to read data: {e}")
        return

    # Extract all subject IDs from the file
    if 'sub' not in df_all.columns:
        print("Cannot find 'sub' column in CSV, please check data format.")
        return
        
    all_subjects_in_csv = set(df_all['sub'].dropna().unique())
    
    # Read subjectlist.txt
    subjectlist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'subjectlist.txt')
    if os.path.exists(subjectlist_path):
        with open(subjectlist_path, 'r') as f:
            valid_subjects = {line.strip() for line in f if line.strip()}
    else:
        print(f"Cannot find {subjectlist_path}, please confirm the file exists.")
        return
        
    # Intersection: Only process subjects that exist in CSV and are listed in subjectlist.txt
    subjects = [s for s in all_subjects_in_csv if s in valid_subjects]
    subjects.sort()
    
    print(f"Start batch processing, found {len(subjects)} subjects in subjectlist.txt...\n")

    success_count = 0

    for sub_id in subjects:
        print(f"Processing: {sub_id} ... ", end="", flush=True)
        df_sub = df_all[df_all['sub'] == sub_id].copy()

        try:
            is_success, msg = process_single_subject(sub_id, df_sub)
            if is_success:
                print(f"Completed ({msg})")
                success_count += 1
            else:
                print(f"Skipped ({msg})")
        except Exception as e:
            print(f"Error occurred: {e}")

    print(f"\nBatch processing finished successfully! Fully processed {success_count}/{len(subjects)} subjects.")
    print(f"Generated charts (PNG) and data (CSV) have been saved to the corresponding AFNI derivatives directories.")

# Execute batch script
if __name__ == '__main__':
    batch_process_all_subjects()