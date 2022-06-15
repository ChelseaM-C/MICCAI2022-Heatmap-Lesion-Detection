# Experiment setup
# TODO: Fill out before starting the experiments
project_name = None  # only if comet_ml is enabled
workspace = None
# CSV files containing 1 - lesion centres, 2 - input key mapping, 3 - lesion counts per input (if desired),
# 4 - lesion sizes (matches keys in lesion centres and input key mapping)
point_file = 'gad_lesion_points.npy'
key_file = 'gad_lesion_points_keys.npy'
count_file = 'gad_lesion_counts.csv'
size_file = 'gad_lesion_points_sizes.npy'
# common hyperparameters for a batch of experiments
# size: desired shape to pad/crop inputs
size = (64, 192, 192)