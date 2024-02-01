# NOTE: Unit for all time-related parameters should be second (s).

MRI = {
    # number of slices in a fMRI volume. 
    'slices': 35,

    # repetition time (TR) in seconds (s).
    'tr': 2
}

# parameters for the real-time task.
RT = {
    # TODO: reset this.

    # number of base volumes in registration.
    'base_vols': 5,

    # instruction duration in seconds (s).
    'ins_duration': 10, # 1, # 10,

    # fixation duration in seconds (s).
    'fix_duration': 60, # 1, # 60,

    # ready duration in seconds (s).
    'ready_duration': 6, # 1, # 6,

    # change cue duration for each block in seconds (s).
    'change_cue_duration': 2,

    # number of blocks per modality.
    'n_blocks_per_modality': 2,

    # modality cue (I/S/L) duration in seconds (s) per block.
    'cue_duration_per_block': 2,

    # fixation duration in seconds (s) per block.
    'fixation_duration_per_block': 6, # 1, # 6, 

    # number of trials (stimuli) per block.
    'n_trials_per_block': 7,

    # trial(stimulus) duration in seconds (s).
    'trial_duration': 2,

    # trial interstimulus interval (ISI) duration in seconds (s).
    'trial_isi_duration': 2,

    # feedback duration in seconds (s) per block
    'feedback_duration_per_block': 2,

    # probe duration in seconds (s) per block
    'probe_duration_per_block': 2,

    # max question duration in seconds (s) per block
    'question_duration_per_block': 5,

    # inter-block (and before feedback) duration in seconds (s)
    'inter_block_duration': 5,
}

# parameters for the non real-time task.
NRT = {
    # instruction duration in seconds (s).
    'ins_duration': 5,

    # fixation duration in seconds (s).
    'fix_duration': 5, # 1, # 60,

    # fixation duration after video in seconds (s) per block.
    'fixation_duration_per_block': 2,

    # max question duration in seconds (s) per block.
    'question_duration_per_block': 5,

    # inter-block duration in seconds (s)
    'inter_block_duration': 5,

    # number of blocks (= trials) per run
    'n_blocks_per_run': 6
}

PATH = {
    # fMRI raw data shared directory (Samba) path.
    'fmri_raw_path': '',

    # HCP mask path.
    'mask_path': '',

    # selected coco sample npz file path.
    'selected_coco_sample_npz_path': '',

    # coco image directory (train2014) path
    'coco_img_dir_path': '',

    # coco caption sound directory path
    'coco_sound_dir_path': '',

    # pytorch DNN model weight path
    'torch_dnn_model_path': '', 

    # pytorch CNN model weight path
    'torch_cnn_model_path': '',
}