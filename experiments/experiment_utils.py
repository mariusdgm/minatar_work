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

def search_files_ending_with_string(rootdir, search_string):
    filepaths = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(search_string):
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
