import os

import random
import torch
import numpy as np

def seed_everything(seed):
    """
    Set the seed on everything I can think of.
    Hopefully this should ensure reproducibility.

    Credits: Florin
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def search_files_containing_string(rootdir, search_string, substring_location = "ending"):
    filepaths = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if substring_location == "ending":
                if file.endswith(search_string):
                    filepath = os.path.join(subdir, file)
                    filepaths.append(filepath)
            elif substring_location == "starting":
                if file.startswith(search_string):
                    filepath = os.path.join(subdir, file)
                    filepaths.append(filepath)
            elif substring_location == "containing":
                if search_string in file:
                    filepath = os.path.join(subdir, file)
                    filepaths.append(filepath)
    return filepaths



def split_path_at_substring(path, substring):
    index = path.find(substring)
    if index != -1:
        left = path[:index]
        right = path[index+len(substring)+1:]
    else:
        left = ""
        right = path
    return left, right

def collect_training_output_files(experiments_folder, append_none=False):

    # find the configurations, each training experiment has one
    config_file_path_list = search_files_containing_string(
        experiments_folder, "config"
    )

    training_experiment_folders = [
        os.path.dirname(file) for file in config_file_path_list
    ]

    experiment_paths = []
    for experiment_path in training_experiment_folders:
        exp_paths = {}

        config_file_path_list = search_files_containing_string(
            experiment_path, "config"
        )

        models_folder_path = os.path.join(experiment_path, "model_checkpoints")
        model_file_path_list = search_files_containing_string(models_folder_path, "mck", substring_location="containing")

        if len(model_file_path_list) == 0:
            if append_none:
                experiment_paths.append(None)
            continue

        model_file_path_list.sort(key=lambda x: int(x.split("_")[-1]))
        exp_paths["model_path"] = model_file_path_list[-1]

        stats_file_path_list = search_files_containing_string(experiment_path, "train_stats")

        exp_paths["training_folder_path"] = experiment_path
        exp_paths["config_path"] = config_file_path_list[0]
        exp_paths["stats_path"] = stats_file_path_list[0]
        experiment_paths.append(exp_paths)

    return experiment_paths

def collect_pruning_output_files(experiments_folder):

    # find the configurations, each training experiment has one
    config_file_path_list = search_files_containing_string(
        experiments_folder, "config"
    )

    pruning_experiment_folders = [
        os.path.dirname(file) for file in config_file_path_list
    ]

    experiment_paths = []
    for experiment_path in pruning_experiment_folders:
        exp_paths = {}

        config_file_path_list = search_files_containing_string(
            experiment_path, "config"
        )

        pruning_stats_file_path_list = search_files_containing_string(experiment_path, "pruning_results",  substring_location="starting")
        baseline_stats_file_path_list = search_files_containing_string(experiment_path, "baseline",  substring_location="starting")

        exp_paths["pruning_folder_path"] = experiment_path
        exp_paths["config_path"] = config_file_path_list[0]
        exp_paths["pruning_stats_paths"] = pruning_stats_file_path_list # more results expected
        exp_paths["baseline_stats_path"] = baseline_stats_file_path_list[0]
        experiment_paths.append(exp_paths)

    return experiment_paths
    
