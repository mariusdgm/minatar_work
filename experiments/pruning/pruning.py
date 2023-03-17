import os
import sys

import yaml

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(proj_root)

import datetime
import random
import multiprocessing


import torch
import torch.nn.utils.prune as prune

import numpy as np

from pathlib import Path
import argparse

from minatar_dqn.my_dqn import Conv_QNET, Conv_QNET_one, AgentDQN
from minatar_dqn.utils import my_logging
from minatar_dqn.my_dqn import build_environment

from experiments.experiment_utils import (
    seed_everything,
    search_files_ending_with_string,
    split_path_at_substring,
)


os.environ["OMP_NUM_THREADS"] = "2"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def get_state(s):
    """
    Converts the state given by the environment to a tensor of size (in_channel, 10, 10), and then
    unsqueeze to expand along the 0th dimension so the function returns a tensor of size (1, in_channel, 10, 10).

    Args:
        s: current state as numpy array

    Returns:
        current state as tensor, permuted to match expected dimensions
    """
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


class PruningExperiment(AgentDQN):
    def __init__(
        self,
        logger,
        config,
        model_path,
        exp_out_file,
        pruning_function,
        experiment_info,
    ):
        self.logger = logger
        self.config = config
        self.model_path = model_path
        self.exp_out_file = exp_out_file
        self.pruning_function = pruning_function
        self.exp_info = experiment_info

        agent_params = config["agent_params"]["args_"]
        self.validation_step_cnt = agent_params["validation_step_cnt"]
        self.validation_epsilon = agent_params["validation_epsilon"]
        
        # other inits needed in AgentDQN function calls
        self.t = 0
        
    def initialize_experiment(self):

        seed_everything(self.config["seed"])

        self.validation_env = build_environment(
            self.config["environment"], self.config["seed"]
        )

        # returns state as [w, h, channels]
        state_shape = self.validation_env.observation_space.shape

        # permute to get batch, channel, w, h shape
        # specific to minatar
        self.in_features = (state_shape[2], state_shape[0], state_shape[1])
        self.in_channels = self.in_features[0]
        self.num_actions = self.validation_env.action_space.n

        self.train_s, info = self.validation_env.reset()

        estimator_settings = self.config.get(
            "estimator", {"model": "Conv_QNET", "args_": {}}
        )

        if estimator_settings["model"] == "Conv_QNET":
            self.policy_model = Conv_QNET(
                self.in_features,
                self.in_channels,
                self.num_actions,
                **estimator_settings["args_"],
            )

        elif estimator_settings["model"] == "Conv_QNET_one":
            self.policy_model = Conv_QNET_one(
                self.in_features,
                self.in_channels,
                self.num_actions,
                **estimator_settings["args_"],
            )

        else:
            estiamtor_name = estimator_settings["model"]
            raise ValueError(f"Could not setup estimator. Tried with: {estiamtor_name}")

        self.load_model_params()

    def load_model_params(self):
        checkpoint = torch.load(self.model_path)
        self.policy_model.load_state_dict(checkpoint["policy_model_state_dict"])

    def prune_model_globally(self, pruning_amount):
        for name, module in self.policy_model.named_modules():
            # structured pruning in all 2D-conv layers
            if isinstance(module, torch.nn.Conv2d):
                prune.ln_structured(
                    module, name="weight", amount=pruning_amount, n=2, dim=1
                )

            # unstructured pruning in linear layers
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=pruning_amount)

    def save_experiment_results(self, experiment_results):
        torch.save(
            {
                "pruning_validation_results": experiment_results,
                "experiment_info": self.exp_info,
            },
            self.exp_out_file,
        )

    def perform_multiple_experiments(self, pruning_values):
        experiment_results = {}
        for pv in pruning_values:
            self.logger.info(
                f"Starting pruning experiment with pruning method {self.pruning_function.__name__} using pruning factor {pv}"
            )
            stats = self.single_experiment(pv)
            experiment_results[pv] = stats

        self.save_experiment_results(experiment_results)
        self.logger.info(
            f"Ended experiment with pruning method {self.pruning_function.__name__}."
        )

    def perform_single_experiment(self, pruning_value):
        experiment_results = {}

        if self.pruning_function:
            self.logger.info(
                f"Starting pruning experiment with pruning method {self.pruning_function.__name__} using pruning factor {pruning_value}"
            )
        else:
            self.logger.info(
                f"No pruning function defined, running baseline experiment"
            )
        stats = self.single_experiment(pruning_value)
        experiment_results[pruning_value] = stats

        self.save_experiment_results(experiment_results)

        if self.pruning_function:
            self.logger.info(
                f"Ended experiment with pruning method {self.pruning_function.__name__}."
            )
        else:
            self.logger.info(f"Baseline experiment ended")

    def single_experiment(self, pruning_value):

        self.initialize_experiment()
        if pruning_value > 0 and self.pruning_function:
            self.pruning_function(self.policy_model, pruning_value)

        validation_stats = self.validate_epoch()
        self.display_validation_epoch_info(validation_stats)

        return validation_stats


########## Define pruning functions ##########
def pruning_method_1(model, pruning_factor):
    """
    Prune the second convolutional layer and the first
    linear layer in the output layer using unstructured pruning
    with L1 norm.
    """
    conv_layer = model.features[2]
    prune.l1_unstructured(conv_layer, name="weight", amount=pruning_factor)

    lin_layer = model.fc[0]
    prune.l1_unstructured(lin_layer, name="weight", amount=pruning_factor)


def pruning_method_2(model, pruning_factor):
    """
    Prune all feature extractor and the first linear layer
    unstructured pruning with L1 norm.
    """
    conv_layer1 = model.features[0]
    prune.l1_unstructured(conv_layer1, name="weight", amount=pruning_factor)

    conv_layer2 = model.features[2]
    prune.l1_unstructured(conv_layer2, name="weight", amount=pruning_factor)

    lin_layer = model.fc[0]
    prune.l1_unstructured(lin_layer, name="weight", amount=pruning_factor)


def pruning_method_3(model, pruning_factor):
    """
    Prune all layers using
    unstructured pruning with L1 norm.
    """
    conv_layer1 = model.features[0]
    prune.l1_unstructured(conv_layer1, name="weight", amount=pruning_factor)

    conv_layer2 = model.features[2]
    prune.l1_unstructured(conv_layer2, name="weight", amount=pruning_factor)

    lin_layer1 = model.fc[0]
    prune.l1_unstructured(lin_layer1, name="weight", amount=pruning_factor)

    lin_layer2 = model.fc[2]
    prune.l1_unstructured(lin_layer2, name="weight", amount=pruning_factor)


def pruning_method_4(model, pruning_factor):
    """
    Prune all feature extractor using structured pruning
    with the L2 norm along dim 0.
    """
    conv_layer1 = model.features[0]
    prune.ln_structured(conv_layer1, name="weight", amount=pruning_factor, n=2, dim=0)

    conv_layer2 = model.features[2]
    prune.ln_structured(conv_layer2, name="weight", amount=pruning_factor, n=2, dim=0)


### Experiment jobs
def create_baseline_experiment_result(logger, config, model_path, exp_out_folder):

    exp_out_file = os.path.join(exp_out_folder, "baseline")

    seed_everything(0)

    pruning_experiment = PruningExperiment(
        logger=logger,
        config=config,
        model_path=model_path,
        exp_out_file=exp_out_file,
        pruning_function=None,
        experiment_info="Baseline performance, no pruning was done.",
    )

    pruning_experiment.perform_single_experiment(0.00)


def run_experiment_with_params(
    logger, config, model_path, path_to_pruning_experiment_folder, pruning_params
):
    # Extract individual parameters from the dictionary

    exp_out_file_name = pruning_params["exp_out_file_name"]
    pruning_function = pruning_params["pruning_function"]

    if "experiment_info" in pruning_params:
        experiment_info = pruning_params["experiment_info"]
    else:
        experiment_info = pruning_function.__doc__

    logger.info(f"Initializing experiment: {exp_out_file_name}")

    ### Setup and run pruning experiment
    exp_out_file = os.path.join(path_to_pruning_experiment_folder, exp_out_file_name)

    seed_everything(0)

    pruning_experiment = PruningExperiment(
        logger=logger,
        config=config,
        model_path=model_path,
        exp_out_file=exp_out_file,
        pruning_function=pruning_function,
        experiment_info=experiment_info,
    )

    pruning_experiment.perform_multiple_experiments(
        [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    )

    return True


def run_parallel_pruning_experiment(
    logger,
    config,
    model_path,
    path_to_pruning_experiment_folder,
):
    exp_1_params = {
        "exp_out_file_name": "pruning_results_1",
        "pruning_function": pruning_method_1,
    }

    exp_2_params = {
        "exp_out_file_name": "pruning_results_2",
        "pruning_function": pruning_method_2,
    }

    exp_3_params = {
        "exp_out_file_name": "pruning_results_3",
        "pruning_function": pruning_method_3,
    }

    exp_4_params = {
        "exp_out_file_name": "pruning_results_4",
        "pruning_function": pruning_method_4,
    }

    experiment_params = [exp_1_params, exp_2_params, exp_3_params, exp_4_params]

    for pruning_params in experiment_params:
        run_experiment_with_params(
            logger,
            config,
            model_path,
            path_to_pruning_experiment_folder,
            pruning_params,
        )


def collect_config_and_model_file(experiments_folder):

    # find the configurations, each training experiment has one
    config_file_path_list = search_files_ending_with_string(
        experiments_folder, "config"
    )

    training_experiment_folders = [
        os.path.dirname(file) for file in config_file_path_list
    ]

    experiment_paths = []
    for experiment_path in training_experiment_folders:
        exp_paths = {}

        config_file_path_list = search_files_ending_with_string(
            experiment_path, "config"
        )

        model_file_path_list = search_files_ending_with_string(experiment_path, "model")

        exp_paths["training_folder_path"] = experiment_path
        exp_paths["config_path"] = config_file_path_list[0]
        exp_paths["model_path"] = model_file_path_list[0]
        experiment_paths.append(exp_paths)

    return experiment_paths


def run_pruning_experiment(experiment_paths):

    model_path = experiment_paths["model_path"]
    config_path = experiment_paths["config_path"]
    training_timestamp_folder = experiment_paths["training_timestamp_folder"]
    pruning_outputs_folder_path = experiment_paths["pruning_output_base_path"]

    left_path, abs_path_experiment_config = split_path_at_substring(
        config_path, training_timestamp_folder
    )
    folder_structure, config_filename = os.path.split(abs_path_experiment_config)
    path_to_pruning_experiment_folder = os.path.join(
        pruning_outputs_folder_path, training_timestamp_folder, folder_structure
    )
    Path(path_to_pruning_experiment_folder).mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger = my_logging.setup_logger(
        env_name=config["environment"],
        folder_path=None,
        identifier_string=folder_structure,
    )

    create_baseline_experiment_result(
        logger,
        config,
        model_path,
        path_to_pruning_experiment_folder,
    )
    run_parallel_pruning_experiment(
        logger,
        config,
        model_path,
        path_to_pruning_experiment_folder,
    )

    return True


def start_parallel_pruning_session(
    experiment_paths, training_timestamp_folder, pruning_output_path, processes=8
):
    for exp_paths in experiment_paths:
        exp_paths["pruning_output_base_path"] = pruning_output_path
        exp_paths["training_timestamp_folder"] = training_timestamp_folder

    with multiprocessing.Pool(processes=processes) as pool:
        statuses = list(pool.map(run_pruning_experiment, experiment_paths))

    print(f"Parallel job run statuses: {statuses}")


def main():
    """
    Look in the folder with trained model for trained models
    The output of the pruning experiment mirrors the folder structure of the training experiment
    There is one additional nesting level for the pruning method
    """

    seed_everything(0)
    logger = my_logging.setup_logger()

    # Collect all paths to models in a specified folder
    training_outputs_folder_path = (
        r"D:\Work\PhD\minatar_work\experiments\training\outputs"
    )
    training_timestamp_folder = "2023_03_17-02_50_37"

    experiment_paths = collect_config_and_model_file(
        os.path.join(training_outputs_folder_path, training_timestamp_folder)
    )

    # filter out the experiments with a single layer for now...
    experiment_paths = [
        exp_paths
        for exp_paths in experiment_paths
        if "conv_model_one_16_layer" not in exp_paths["training_folder_path"]
    ]

    pruning_outputs_folder_path = (
        r"D:\Work\PhD\minatar_work\experiments\pruning\outputs"
    )

    start_parallel_pruning_session(
        experiment_paths,
        training_timestamp_folder=training_timestamp_folder,
        pruning_output_path=pruning_outputs_folder_path,
    )

    my_logging.cleanup_file_handlers()


if __name__ == "__main__":
    main()
