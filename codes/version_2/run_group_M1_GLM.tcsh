#!/usr/bin/tcsh
# ==============================================================================
# run_group_M1_GLM.tcsh
#
# Phase 1 Sanity Check: Group-level GLM for M1 Localization
#
# This script reads the basic statistical maps (stats.sub-*+tlrc) calculated for all subjects 
# in Phase 1 (run_afni_condition_glm.py), extracts the Beta values for the 45-degree rotation task (Rot45) 
# or the main baseline task (BLOCK(4.5,1)), and runs a `3dttest++` group test.
# The resulting group map serves as validation to confirm that the task stably activates the primary motor cortex across the group.
# ==============================================================================

# Set paths
set proj_dir = "/home/ser/2026_brainhack_li_project"
set afni_dir = "${proj_dir}/data/derivatives/afni_fmriprep"
set output_dir = "${proj_dir}/data/derivatives/group_stats_phase1"

if ( ! -d $output_dir ) then
    mkdir -p $output_dir
endif

set list_file = "${proj_dir}/codes/subjectlist.txt"
if ( ! -f $list_file ) then
    echo "Error: Cannot find $list_file"
    exit 1
endif

set subjects = `cat $list_file | grep -v '^\s*$'`

echo "================================================================="
echo " Start preparing Phase 1 Group-level GLM (M1 Sanity Check)"
echo "================================================================="

# Prepare the input list for 3dttest++
# In the Phase 1 GLM, stimulus coefficients are usually located in sub-briks (e.g., "Task#0_Coef")
# Since the exact output label of `run_afni_condition_glm.py` might be 'Task#0_Coef', we use generic syntax to extract it
# Here we use AFNI's sub-brick selection syntax: 'stats.sub-01+tlrc[Task#0_Coef]'

set input_files = ()
set valid_count = 0

foreach subj ( $subjects )
    # run_afni_condition_glm.py exports the RightLearning beta coefficient
    # as a standalone 3D NIfTI (sub-brick [1] from stats_task_r03.nii.gz).
    # Using this file directly avoids tcsh/AFNI sub-brick selector issues.
    set stat_file = "${afni_dir}/${subj}/GLM_Condition/Coef_RightLearning.nii.gz"

    if ( -f $stat_file ) then
        set input_files = ($input_files "${stat_file}")
        @ valid_count++
    else
        echo "Cannot find Phase 1 statistical results for $subj ($stat_file)"
    endif
end

if ( $valid_count < 3 ) then
    echo "The number of valid subjects is less than 3, cannot perform group test."
    exit 1
endif

echo "Found $valid_count valid statistical files, starting 3dttest++ ..."

# Enter output directory to execute
cd $output_dir

# Execute 3dttest++
# Each element of $input_files is a plain 3D NIfTI (no sub-brick selector needed).
3dttest++ -prefix Group_Phase1_M1_Activation \
          -setA $input_files

echo ""
echo "Group M1 GLM (Sanity Check) execution complete!"
echo "Results saved in: $output_dir/Group_Phase1_M1_Activation+tlrc"
echo "Open in AFNI to confirm that the largest cluster falls on Left Precentral Gyrus (Hand Knob area)."
